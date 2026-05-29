# vitulus_safety

Safety monitoring node for the Vitulus robot platform.

## Overview

This package provides real-time safety monitoring that protects the robot and its environment by detecting hazardous conditions and taking immediate corrective action. All safety events are logged and reported to the webui and Nextion display.

## Features

- **Drive motor monitoring** – detects motor communication loss, faults, and overtemperature
- **Mower motor temperature monitoring** – shuts down mower if motor overheats
- **IMU tilt monitoring** – stops mower blade when robot exceeds safe tilt angle
- **Centralized safety logging** – all events published to `/nextion/log_info` (webui + Nextion display)
- **Audible alarm** – triggers power module buzzer on critical failures

## Dependencies

- `rospy`
- `std_msgs`
- `sensor_msgs`
- `vitulus_msgs`

## Installation

```bash
cd ~/catkin_ws
catkin build vitulus_safety
source devel/setup.bash
```

## Usage

```bash
roslaunch vitulus_safety safety.launch
```

## Node: safety_node

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/base/front_left_wheel_state` | `vitulus_msgs/Moteus_controller_state` | Front-left motor state |
| `/base/front_right_wheel_state` | `vitulus_msgs/Moteus_controller_state` | Front-right motor state |
| `/base/rear_left_wheel_state` | `vitulus_msgs/Moteus_controller_state` | Rear-left motor state |
| `/base/rear_right_wheel_state` | `vitulus_msgs/Moteus_controller_state` | Rear-right motor state |
| `/bno085/imu` | `sensor_msgs/Imu` | IMU orientation data |
| `/mower/status` | `vitulus_msgs/Mower` | Mower status including temperature |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/base/motor_power` | `std_msgs/Bool` | Drive motor power control |
| `/mower/set_motor_on` | `std_msgs/Bool` | Mower motor on/off |
| `/mower/set_power` | `std_msgs/Bool` | Mower power on/off |
| `/pm/play_melody` | `std_msgs/Int16` | Power module alarm buzzer |
| `/nextion/log_info` | `std_msgs/String` | Safety event log (webui + Nextion) |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `~max_tilt_x` | float | 30.0 | Max roll angle (degrees) before mower shutdown |
| `~max_tilt_y` | float | 30.0 | Max pitch angle (degrees) before mower shutdown |
| `~motor_timeout` | float | 2.0 | Seconds without motor state to trigger failure |
| `~motor_check_rate` | float | 5.0 | Motor check frequency (Hz) |
| `~startup_grace` | float | 10.0 | Grace period after start (seconds) |
| `~motor_max_temp` | float | 70.0 | Max drive motor temperature (°C) |
| `~mower_max_temp` | float | 70.0 | Max mower motor temperature (°C) |
| `~alarm_melody` | int | 4 | Melody ID for power module buzzer |

## Safety Actions

| Condition | Action |
|-----------|--------|
| Drive motor timeout / fault / no data | All motors OFF + alarm |
| Drive motor temperature > `motor_max_temp` | All motors OFF + alarm |
| Mower motor temperature > `mower_max_temp` | Mower OFF |
| IMU tilt exceeds `max_tilt_x` or `max_tilt_y` | Mower OFF (auto-recovers when tilt returns to normal) |

## Configuration

Parameters are loaded from `config/safety.yaml` via the launch file. Edit values there or override via rosparam.

## License

MIT
