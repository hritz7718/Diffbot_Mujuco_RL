"""First-stage target reaching with ideal simulated localization; no ROS."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import gymnasium as gym
import mujoco
import numpy as np


class DiffBotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        root = ET.parse(Path(__file__).with_name("diffbot.xml")).getroot()
        ET.SubElement(root.find("worldbody"), "site", name="target",
                      type="cylinder", size="0.12 0.003", pos="1 0 0.004",
                      rgba="0.1 1 0.2 0.7")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.action_space = gym.spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        # body-frame target xy, forward velocity, yaw rate, two wheel velocities
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self.goal = np.zeros(2)
        self.viewer = None
        self.steps = 0
        self.dt = 0.05

#Let's define the observation function that returns the current state of the robot in the environment. The observation includes the position of the goal relative to the robot, the robot's forward velocity, yaw rate, and wheel velocities.
    def _obs(self): 
        rot = self.data.body("base_link").xmat.reshape(3, 3)
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        delta = self.goal - self.data.qpos[:2]
        return np.array([c*delta[0]+s*delta[1], -s*delta[0]+c*delta[1],
                         c*self.data.qvel[0]+s*self.data.qvel[1],
                         self.data.sensor("imu_gyro").data[2],
                         self.data.sensor("left_speed").data[0],
                         self.data.sensor("right_speed").data[0]], dtype=np.float32)

#Lets define the reset function that initializes the environment at the start of each episode. It randomizes the robot's initial orientation and position, sets a new goal location, and resets the simulation state.
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        yaw = self.np_random.uniform(-math.pi, math.pi)
        self.data.qpos[3:7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
        # Settle the caster and wheels before starting an episode.
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=500)
        mujoco.mj_forward(self.model, self.data)
        angle = self.np_random.uniform(-math.pi, math.pi)
        radius = self.np_random.uniform(0.4, 1.2)
        self.goal = self.data.qpos[:2].copy() + radius*np.array([math.cos(angle), math.sin(angle)])
        self.model.site("target").pos[:2] = self.goal
        self.steps = 0
        self.previous_distance = float(np.linalg.norm(self.goal-self.data.qpos[:2]))
        mujoco.mj_forward(self.model, self.data)
        if self.render_mode == "human":
            self.render()
        return self._obs(), {"is_success": False}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Expected two finite action values")
        v, w = 0.15*action[0], 1.0*action[1]
        wheels = np.array([(v-0.13*w)/0.035, (v+0.13*w)/0.035])
        wheels *= min(1., 6./max(6., float(np.max(np.abs(wheels)))))
        self.data.ctrl[:] = wheels
        mujoco.mj_step(self.model, self.data, nstep=25)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or any(x.number for x in self.data.warning):
            raise RuntimeError("Unstable simulation; stop and inspect the model")
        self.steps += 1
        distance = float(np.linalg.norm(self.goal-self.data.qpos[:2]))
        failed = bool(self.data.body("base_link").xmat[8] < 0.8 or
                      np.linalg.norm(self.data.qpos[:2]) > 3)
        success = bool(distance < 0.12 and not failed)
        reward = 10*(self.previous_distance-distance) - 0.005 - 0.001*float(action@action)
        reward += 10 if success else (-10 if failed else 0)
        self.previous_distance = distance
        terminated = success or failed
        truncated = self.steps >= 600 and not terminated
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, {
            "is_success": success, "distance": distance}

    def render(self):
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 2.8
            self.viewer.cam.elevation = -55
        if self.viewer.is_running():
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
