"""Workspace contact queries for the pouring scenes."""


def workspace_contacts(model, data):
    """Report floor-independent contacts that involve the robot or held bottle."""
    root = model.body("robot_planar_root").id

    def on_robot(geom_id):
        body = model.geom_bodyid[geom_id]
        while body > 0:
            if body == root:
                return True
            body = model.body_parentid[body]
        return False

    ground = model.geom("ground").id
    return [
        (int(a), int(b)) for contact in data.contact
        for a, b in [contact.geom]
        if ground not in (a, b) and on_robot(a) != on_robot(b)
    ]
