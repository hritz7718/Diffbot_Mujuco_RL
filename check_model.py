from pathlib import Path
import json
import math
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name('diffbot.xml')))
results = {}
for name, command in [('rest', [0, 0]), ('forward', [3, 3]), ('turn', [-3, 3])]:
    data = mujoco.MjData(model)
    for _ in range(1500):
        mujoco.mj_step(model, data)
    origin = data.qpos[:3].copy()
    data.ctrl[:] = command
    for _ in range(1500):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all(), name
    assert sum(w.number for w in data.warning) == 0, name
    rotation = data.body('base_link').xmat.reshape(3, 3)
    assert rotation[2, 2] > 0.95, f'{name}: tipped'
    displacement = data.qpos[:3] - origin
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    if name == 'rest':
        assert np.linalg.norm(displacement[:2]) < 0.02, 'rest drift'
    elif name == 'forward':
        assert displacement[0] > 0.15, 'not moving forward'
        assert abs(displacement[1]) < 0.03, 'not driving straight'
    else:
        assert yaw > 0.2, 'not turning left'
    results[name] = {'displacement_m': displacement.tolist(), 'yaw_rad': yaw,
                     'height_m': float(data.qpos[2]), 'contacts': data.ncon}
print(json.dumps({'mujoco_version': mujoco.__version__, 'checks': results}, indent=2))
