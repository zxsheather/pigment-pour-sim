"""Reduced water model: tilt-driven discharge and gravity-driven parcels."""

from dataclasses import dataclass
from math import acos

import mujoco
import numpy as np


@dataclass
class Parcel:
    xyz: np.ndarray
    velocity: np.ndarray
    ml: float


class PourWater:
    """Track water volume and flight using the simulated bottle pose."""

    def __init__(self, model, data, config):
        self.model = model
        self.body = model.body("held_water_bottle").id
        self.mouth = model.site("held_bottle_mouth").id
        self.water_geom = model.geom("held_water").id
        self.cylinder = data.site("cylinder_mouth").xpos.copy()
        self.radius = config["cylinder"]["radius"] - .001
        self.capacity_ml = config["cylinder"]["capacity_ml"]
        self.initial_ml = config["bottle"]["initial_ml"]
        self.remaining_ml = self.initial_ml
        self.collected_ml = 0.
        self.spilled_ml = 0.
        self.parcels = []
        self.impacts = []
        self.pending_ml = 0.
        self.last_mouth = data.site(self.mouth).xpos.copy()
        self.table_height = config["table"]["height"]
        self.table_center = np.asarray(config["table"]["center_xy"])
        self.table_half_size = np.asarray(config["table"]["size_xy"]) / 2
        self.cylinder_height = config["cylinder"]["height"]
        self.lip_radius = config["bottle"]["radius"] - .008
        self.gravity = model.opt.gravity.copy()

    @property
    def in_flight_ml(self):
        return sum(parcel.ml for parcel in self.parcels)

    def apply_load(self, data):
        # Cancel the weight of water that has left the rigid mass proxy.
        force = -self.gravity * ((self.initial_ml - self.remaining_ml) * .001)
        offset = data.geom_xpos[self.water_geom] - data.xipos[self.body]
        data.xfrc_applied[self.body] = np.r_[force, np.cross(offset, force)]

    def advance_parcels(self, dt):
        """Advance water already discharged into the static workspace."""
        survivors = []
        for parcel in self.parcels:
            previous = parcel.xyz
            xyz = previous + dt * (parcel.velocity + .5 * dt * self.gravity)
            if previous[2] >= self.cylinder[2] > xyz[2]:
                fraction = (previous[2] - self.cylinder[2]) / (previous[2] - xyz[2])
                crossing = previous + fraction * (xyz - previous)
                if np.linalg.norm(crossing[:2] - self.cylinder[:2]) <= self.radius:
                    caught = min(parcel.ml, self.capacity_ml - self.collected_ml)
                    self.collected_ml += caught
                    self.spilled_ml += parcel.ml - caught
                    continue
            # Outside the tabletop footprint, missed water falls to the floor.
            if previous[2] >= self.table_height > xyz[2]:
                blend = (previous[2] - self.table_height) / (previous[2] - xyz[2])
                edge_crossing = previous + blend * (xyz - previous)
                on_table = np.all(np.abs(edge_crossing[:2] - self.table_center)
                                  <= self.table_half_size)
            else:
                on_table = False
            if on_table or xyz[2] <= 0.:
                self.spilled_ml += parcel.ml
                impact = edge_crossing if on_table else np.r_[xyz[:2], 0.]
                self.impacts.append(impact + np.array([0., 0., .001]))
            else:
                parcel.xyz = xyz
                parcel.velocity += dt * self.gravity
                survivors.append(parcel)
        self.parcels = survivors

    def step(self, data, dt):
        self.advance_parcels(dt)
        mouth = data.site(self.mouth).xpos.copy()
        axis = data.body(self.body).xmat.reshape(3, 3)[:, 2]
        tilt = acos(np.clip(axis[2], -1., 1.))
        # Provisional discharge curve. Gravity and rim crossing are evaluated
        # in world space independently of this empirical flow-rate assumption.
        ramp = np.clip((tilt - np.deg2rad(80.)) / np.deg2rad(25.), 0., 1.)
        self.pending_ml += 12. * ramp * dt
        released = min(self.pending_ml, self.remaining_ml)
        if released >= .10 or (released > 0. and (ramp == 0. or released == self.remaining_ml)):
            down = np.array([0., 0., -1.])
            lip = down - np.dot(down, axis) * axis
            outlet = mouth + self.lip_radius * lip / max(np.linalg.norm(lip), 1e-9)
            velocity = (mouth - self.last_mouth) / dt + .06 * axis
            self.parcels.append(Parcel(outlet, velocity, released))
            self.remaining_ml -= released
            self.pending_ml = 0.
        self.last_mouth = mouth

    def draw(self, scene, data):
        fraction = self.remaining_ml / self.initial_ml
        for geom in scene.geoms[:scene.ngeom]:
            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid == self.water_geom:
                half_height = geom.size[2]
                geom.size[2] = max(.0001, half_height * fraction)
                axis = data.body(self.body).xmat.reshape(3, 3)[:, 2]
                geom.pos[:] -= axis * (half_height - geom.size[2])
                break
        rgba = np.array([.12, .50, .95, .86], dtype=np.float32)
        identity = np.eye(3).ravel()

        def add(kind, size, position):
            if scene.ngeom < scene.maxgeom:
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], kind, np.array(size),
                                    position, identity, rgba)
                scene.ngeom += 1

        for parcel in self.parcels:
            add(mujoco.mjtGeom.mjGEOM_SPHERE, [.003, 0., 0.], parcel.xyz)
        for impact in self.impacts:
            add(mujoco.mjtGeom.mjGEOM_ELLIPSOID, [.005, .004, .001], impact)
        if self.collected_ml > 0.:
            height = (self.cylinder_height - .006) * self.collected_ml / self.capacity_ml
            base = self.cylinder[2] - self.cylinder_height
            center = np.r_[self.cylinder[:2], base + height / 2]
            add(mujoco.mjtGeom.mjGEOM_CYLINDER, [self.radius, height / 2, 0.], center)
