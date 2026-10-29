DIFFBOT - FIRST MUJOCO MODEL

Double-click Open DiffBot.cmd, or drag diffbot.xml into MuJoCo.
Press Space to pause/resume. Expand Control to change wheel speeds.
Both controls +3 rad/s: forward. Left -3, right +3: turn left.
Both zero: stop. Reset returns to the initial state.
Check DiffBot.cmd runs mechanical checks and prints results or an error.

Preserved dimensions, masses, inertia, wheel joint damping/friction and fixed
caster from the supplied Xacro files. Added a floating base, floor, lighting,
wheel velocity actuators, and decorative wheel stripes. No meshes or ROS needed.
The caster extends 3 mm below the drive wheels; slight chassis pitch is expected.
It slides rather than rolling. Sensor housings remain massless and visual-only.

Provisional values: wheel/floor friction 0.8, velocity gain 1, command limit
+/-6 rad/s, and torque limit 1 Nm. These are simulation tuning values, NOT
verified real motor data; the original Gazebo torque limit was 20 Nm.
Caster friction 0.05 follows Gazebo, using contact priority to override the floor.
Friction laws differ between engines. Gazebo acceleration limiting is not ported.

Wheel encoders, ideal gyro/accelerometer and a forward camera are defined.
Camera vertical FOV corresponds to the source horizontal FOV at 640x480.
LiDAR housing is included, but a functional scan is NOT implemented yet.
ROS topics, odometry publishing, sensor noise, camera clipping, capture resolution
and sensor update rates need a later wrapper. Gazebo plugins are not executable
in this scene. This is a mechanical starter, not a trained or deployable policy.
