# Deployment validation

Validated 2026-11-24. No real robot was connected or commanded during these checks.

| Check | Result |
| --- | --- |
| Release archive | 646,494,627 bytes; SHA256 `d6ce2696c86ce23e3f23c428ecc05836820cecd6f0f65bfe0e57f6840cd59da4` |
| Portability/privacy | Ten regular, top-level checkpoint files; no owner names or source filesystem paths in archive metadata; original absolute processor path sanitized; small metadata/normalizers inspected |
| Frozen base | Full 8,331,486,672-byte hash matched `37f8f327d3f9c27f056ed078db43c0e39b8caf3df0dc0163cdb8cf92c167968a`; independently matched pinned Hugging Face LFS object ID |
| Relocated checkpoint | Archive extracted into a fresh directory; all file sizes and SHA256 values matched manifest. Frozen base linked to the pre-existing, hash-verified upstream file |
| Independent source | Fresh checkout of Wall-X `ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f`; optional FlashAttention import patch applied; no experimental `PYTHONPATH` dependency overrides |
| GPU smoke inference | RTX 4090 24 GB, Python 3.11.16, torch 2.7.0+cu128, torchvision 0.22.0+cu128, transformers 5.2.0, accelerate 1.13.0, NumPy 1.26.4; native SDPA |
| Load audit | 257 action-expert overlay tensors, missing/unexpected/unloaded-meta lists all empty; vision intermediate size 3420, alignment 1 |
| Observation/output | Included initial simulated head/right-wrist RGB + proprioception; output finite `follow2_pos` shape 32×7, one inference 7.52 seconds including first-call overhead |
| Unit/integration tests | **15 passed**, including two contract tests using the official `x2robot==1.1.2` wheel with `betterproto2==0.10.0` and `grpcio==1.76.0`. Geometry/TCP round trip, unit conversion, bad quaternion/rotation rejection, default-unfilled config refusal, step/gripper limits, stale/moving feedback refusal, preview zero writes, SDK read-only adapter, right-only dispatch/result checking, archive path/link/hash checks |
| SDK contract | Official Pose/PoseStamped/GripperPosition serialization and dispatch; timeout keyword verified on the real generated arm/gripper/camera/head/lift/chassis methods. No network connection in these tests |
| Default robot entry | `preview --config deploy/robot.example.json` refused the uncalibrated template before model load or SDK connection, as intended |

The offline result is included as [samples/smoke_result.json](samples/smoke_result.json). Numeric actions may differ across GPU/runtime builds; treat finite output shape and checkpoint audit as the installation check, not an exact grasp-success test.

The test reused an existing Python inference environment. A completely fresh Ubuntu OS installation and the full bootstrap dependency installation were not recreated. The source and checkpoint were independently restored and relocated. The smoke test used the already available, full-hash-verified base file instead of downloading 8.33 GB again; the Hugging Face revision and LFS hash were checked independently.

The legacy checkpoint YAML may emit `Typed config load failed ... task.dof_config` before the upstream loader uses its legacy compatibility path. The successful audited inference above used this same configuration. A failure to produce the final finite 32×7 output should not be dismissed as this warning.

Unverified on hardware: SDK server/firmware compatibility, tool/reference-frame calibration, gripper/angle/lift units, camera synchronization and visual domain, end-pose mode setup, real arm reachability/collisions, execution timing, grasp force and retention. Mock/SDK-message checks establish interface behavior, not mechanical safety or task success.
