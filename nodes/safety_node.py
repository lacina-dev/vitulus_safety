#!/usr/bin/env python3

import rospy
import math
from std_msgs.msg import Bool, Int16, String
from sensor_msgs.msg import Imu
from vitulus_msgs.msg import Moteus_controller_state, Mower


def quaternion_to_euler(x, y, z, w):
    """Convert quaternion to euler angles (roll, pitch, yaw) in degrees."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class SafetyMonitor:
    def __init__(self):
        # Parameters
        self.max_tilt_x = rospy.get_param('~max_tilt_x', 30.0)
        self.max_tilt_y = rospy.get_param('~max_tilt_y', 30.0)
        self.motor_timeout = rospy.get_param('~motor_timeout', 2.0)
        self.motor_check_rate = rospy.get_param('~motor_check_rate', 5.0)
        self.alarm_melody = rospy.get_param('~alarm_melody', 4)
        self.startup_grace = rospy.get_param('~startup_grace', 10.0)
        self.motor_max_temp = rospy.get_param('~motor_max_temp', 70.0)
        self.mower_max_temp = rospy.get_param('~mower_max_temp', 70.0)

        # State
        self.start_time = rospy.Time.now()
        self.motor_states = {}  # name -> {time, mode, fault, temperature}
        self.motor_names = [
            'FRONT_LEFT',
            'FRONT_RIGHT',
            'REAR_LEFT',
            'REAR_RIGHT'
        ]
        self.motors_killed = False
        self.motors_killed_by_temp = False
        self.mower_stopped_by_tilt = False
        self.mower_stopped_by_temp = False

        # Publishers
        self.pub_motor_power = rospy.Publisher('/base/motor_power', Bool, queue_size=1)
        self.pub_alarm = rospy.Publisher('/pm/play_melody', Int16, queue_size=1)
        self.pub_mower_motor_on = rospy.Publisher('/mower/set_motor_on', Bool, queue_size=1)
        self.pub_mower_power = rospy.Publisher('/mower/set_power', Bool, queue_size=1)
        self.pub_log_info = rospy.Publisher('/nextion/log_info', String, queue_size=10)

        # Subscribers - motor states
        rospy.Subscriber('/base/front_left_wheel_state', Moteus_controller_state, self.motor_state_cb)
        rospy.Subscriber('/base/front_right_wheel_state', Moteus_controller_state, self.motor_state_cb)
        rospy.Subscriber('/base/rear_left_wheel_state', Moteus_controller_state, self.motor_state_cb)
        rospy.Subscriber('/base/rear_right_wheel_state', Moteus_controller_state, self.motor_state_cb)

        # Subscriber - IMU
        rospy.Subscriber('bno085/imu', Imu, self.imu_cb)

        # Subscriber - Mower status
        rospy.Subscriber('/mower/status', Mower, self.mower_status_cb)

        # Timer for periodic motor check
        rospy.Timer(rospy.Duration(1.0 / self.motor_check_rate), self.check_motors)

        rospy.loginfo("[safety] Safety monitor started. max_tilt_x=%.1f, max_tilt_y=%.1f, "
                      "motor_timeout=%.1f, startup_grace=%.1f, motor_max_temp=%.1f, mower_max_temp=%.1f",
                      self.max_tilt_x, self.max_tilt_y, self.motor_timeout,
                      self.startup_grace, self.motor_max_temp, self.mower_max_temp)

    def safety_log(self, message):
        """Log safety event to ROS, webui and nextion display."""
        rospy.logwarn("[safety] %s", message)
        self.pub_log_info.publish(String(data="[SAFETY] " + message))

    def motor_state_cb(self, msg):
        """Update motor state tracking."""
        self.motor_states[msg.name] = {
            'time': rospy.Time.now(),
            'mode': msg.mode,
            'fault': msg.fault,
            'temperature': msg.temperature
        }

    def check_motors(self, event):
        """Check that all 4 motors are alive, healthy, and within temperature."""
        if self.motors_killed:
            return

        # Skip checks during startup grace period
        if (rospy.Time.now() - self.start_time).to_sec() < self.startup_grace:
            return

        now = rospy.Time.now()
        all_ok = True
        problem_motors = []
        overtemp_motors = []

        for name in self.motor_names:
            state = self.motor_states.get(name)
            if state is None:
                all_ok = False
                problem_motors.append("{} (no data)".format(name))
            elif (now - state['time']).to_sec() > self.motor_timeout:
                all_ok = False
                problem_motors.append("{} (timeout)".format(name))
            elif state['fault'] != 'OK':
                all_ok = False
                problem_motors.append("{} (fault: {})".format(name, state['fault']))
            elif state['temperature'] > self.motor_max_temp:
                overtemp_motors.append("{} ({:.1f}C)".format(name, state['temperature']))

        if not all_ok:
            msg = "Motor failure: {}. Motors OFF.".format(", ".join(problem_motors))
            self.safety_log(msg)
            self.shutdown_motors()
        elif overtemp_motors and not self.motors_killed_by_temp:
            msg = "Motor overtemp: {}. Motors OFF.".format(", ".join(overtemp_motors))
            self.safety_log(msg)
            self.shutdown_motors()
            self.motors_killed_by_temp = True

    def shutdown_motors(self):
        """Turn off all motors and sound alarm."""
        self.motors_killed = True
        self.pub_motor_power.publish(Bool(data=False))
        self.pub_alarm.publish(Int16(data=self.alarm_melody))

    def mower_status_cb(self, msg):
        """Check mower motor temperature."""
        if self.mower_stopped_by_temp:
            return
        if msg.temp > self.mower_max_temp:
            log_msg = "Mower overtemp: {:.1f}C. Mower OFF.".format(msg.temp)
            self.safety_log(log_msg)
            self.stop_mower_temp()
            self.mower_stopped_by_temp = True

    def imu_cb(self, msg):
        """Check IMU tilt and stop mower if excessive."""
        roll, pitch, _ = quaternion_to_euler(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        )

        tilt_exceeded = abs(roll) > self.max_tilt_x or abs(pitch) > self.max_tilt_y

        if tilt_exceeded and not self.mower_stopped_by_tilt:
            log_msg = "Tilt exceeded! roll={:.1f} pitch={:.1f}. Mower OFF.".format(roll, pitch)
            self.safety_log(log_msg)
            self.stop_mower_tilt()
            self.mower_stopped_by_tilt = True
        elif not tilt_exceeded and self.mower_stopped_by_tilt:
            rospy.loginfo("[safety] Tilt back to normal. roll=%.1f pitch=%.1f", roll, pitch)
            self.mower_stopped_by_tilt = False

    def stop_mower_tilt(self):
        """Stop mower due to tilt."""
        self.pub_mower_motor_on.publish(Bool(data=False))
        self.pub_mower_power.publish(Bool(data=False))

    def stop_mower_temp(self):
        """Stop mower due to overtemperature."""
        self.pub_mower_motor_on.publish(Bool(data=False))
        self.pub_mower_power.publish(Bool(data=False))


if __name__ == '__main__':
    rospy.init_node('safety_node')
    monitor = SafetyMonitor()
    rospy.spin()
