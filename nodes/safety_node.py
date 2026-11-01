#!/usr/bin/env python3

import collections
import json
import math
import os
import time

import rospy
from std_msgs.msg import Bool, Int16, String
from sensor_msgs.msg import Imu
from vitulus_msgs.msg import Moteus_controller_state, Mower, Power_status


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


class MowerTempGuard(object):
    """Is the mower motor really hot?  Pure: the caller gives the clock.

    Incident 1. 11. 2026: the firmware reported 20.0, 42.5, 90.7, 20.9 C in four
    one second windows, around a 6 s stall of the mower MCU.  The motor's real
    rate that day was 0.2 C/min.  One sample above the limit stopped the program.

    Trip     above limit_c for sustain_s of consecutive samples, or above hard_c
             for hard_s.  A jump never excuses a reading that stays: it only
             names the excursion.  A sensor stuck high trips the same way.
    Spike    a change faster than max_rate_c_s: warning and record, no stop.
    Fault    out of [valid_min_c, valid_max_c] or NaN for fault_s.  An open NTC
             reads -273 C, which the firmware limit never sees, so this is the
             only protection left: it stops the blade like a trip.
    Rearm    at or below cool_c for rearm_s of consecutive samples.

    The limits leave the firmware's own 1 s trip untouched: it still stops the
    blade in the first window.  This class decides about power and the mission.
    Readings come at 2 Hz and change at 1 Hz, so one window is 2 samples.
    """

    DEFAULTS = {
        'limit_c': 70.0, 'cool_c': 55.0, 'hard_c': 100.0, 'sustain_s': 3.0,
        'hard_s': 1.5, 'max_rate_c_s': 5.0, 'jump_min_c': 5.0, 'valid_min_c': -30.0,
        'valid_max_c': 150.0, 'fault_s': 3.0, 'rearm_s': 10.0, 'settle_s': 10.0,
    }

    def __init__(self, **params):
        self.p = dict(self.DEFAULTS)
        self.p.update(params)
        self.state = 'ok'           # ok | overtemp | sensor_fault
        self.ref = None             # (t, temp) of the last believed reading
        self.spike = None           # the running excursion
        self.hot_since = None
        self.hot_n = 0
        self.hard_since = None
        self.hard_n = 0
        self.bad_since = None
        self.bad_n = 0
        self.cool_since = None
        self.spikes = 0

    def _valid(self, temp):
        return (not math.isnan(temp) and not math.isinf(temp)
                and self.p['valid_min_c'] <= temp <= self.p['valid_max_c'])

    def _run(self, now, active, since, count):
        if not active:
            return None, 0
        return (now if since is None else since), count + 1

    def _track_spike(self, now, temp, valid, out):
        """Name a jump, follow it, say how it ended.  Never decides a stop."""
        if self.spike is not None:
            s = self.spike
            s['peak'] = temp if not valid or abs(temp - s['from']) > abs(s['peak'] - s['from']) \
                else s['peak']
            back = valid and abs(temp - s['from']) <= (
                self.p['jump_min_c'] + self.p['max_rate_c_s'] * (now - s['t']))
            if back:
                self.spike = None
                self.ref = (now, temp)
                out.append(('spike_end', {'from': s['from'], 'peak': s['peak'], 'to': temp,
                                          'duration_s': round(now - s['t'], 2)}))
            elif now - s['t'] >= self.p['settle_s']:
                # It stayed.  Hot readings have tripped by now; the rest is the new level.
                self.spike = None
                self.ref = (now, temp) if valid else self.ref
                out.append(('level_shift', {'from': s['from'], 'to': temp}))
            return
        if not valid:
            if self.ref is not None:
                self._open_spike(now, temp, out)
            return
        if self.ref is None:
            self.ref = (now, temp)
            return
        # One window at the firmware's 1 Hz is the shortest real dt.  A long gap in
        # the stream earns no credit: nothing that fast was measured across it.
        dt = min(max(now - self.ref[0], 1.0), 2.0)
        change = abs(temp - self.ref[1])
        if change >= self.p['jump_min_c'] and change / dt > self.p['max_rate_c_s']:
            self._open_spike(now, temp, out)
        else:
            self.ref = (now, temp)

    def _open_spike(self, now, temp, out):
        self.spikes += 1
        self.spike = {'t': now, 'from': self.ref[1], 'peak': temp}
        out.append(('spike', {'from': self.ref[1], 'to': temp,
                              'dt_s': round(now - self.ref[0], 2)}))

    def feed(self, now, temp):
        """One /mower/status reading.  Returns [(event, detail)]."""
        temp = float(temp)
        out = []
        # 0.0 is the firmware's value before its first window: no jump, but it is
        # no reading either, and a stuck one is a fault like any other.
        zero = temp == 0.0
        valid = self._valid(temp) and not zero
        if not zero:
            self._track_spike(now, temp, valid, out)

        hot = not valid or temp > self.p['limit_c']
        self.hot_since, self.hot_n = self._run(now, hot, self.hot_since, self.hot_n)
        self.hard_since, self.hard_n = self._run(
            now, valid and temp > self.p['hard_c'], self.hard_since, self.hard_n)
        self.bad_since, self.bad_n = self._run(now, not valid, self.bad_since, self.bad_n)

        if self.state == 'ok':
            if self._held(now, self.bad_since, self.bad_n, self.p['fault_s']):
                self.state = 'sensor_fault'
                out.append(('fault', {'temp': temp}))
            elif (self._held(now, self.hot_since, self.hot_n, self.p['sustain_s'])
                  or self._held(now, self.hard_since, self.hard_n, self.p['hard_s'])):
                self.state = 'overtemp'
                out.append(('trip', {'temp': temp, 'spike': self.spike is not None}))
        elif self.state == 'overtemp' and self._held(
                now, self.bad_since, self.bad_n, self.p['fault_s']):
            self.state = 'sensor_fault'
            out.append(('fault', {'temp': temp}))
        else:
            cool = valid and temp <= self.p['cool_c']
            self.cool_since = (self.cool_since if self.cool_since is not None else now) \
                if cool else None
            if cool and now - self.cool_since >= self.p['rearm_s']:
                was, self.state, self.cool_since = self.state, 'ok', None
                out.append(('rearm', {'was': was, 'temp': temp}))
        return out

    def _held(self, now, since, count, seconds):
        """Lasted `seconds` of consecutive 2 Hz samples: a gap in the stream
        (the MCU stall of 1. 11.) must not make two readings look like three seconds."""
        return (since is not None and now - since >= seconds
                and count >= int(round(seconds * 2.0)) + 1)


class TempRing(object):
    """Temperature around events, in a file, for the next incident.

    Needs nothing but /mower/status: no map, no rules, no mission, and it records
    while docked too.  Each event writes the last ring_s at once, so a crash
    keeps it, and after_s later one more line with what followed.
    """

    def __init__(self, path, ring_s=120.0, after_s=30.0, max_bytes=2000000, warn=None):
        self.path = path
        self.ring_s = ring_s
        self.after_s = after_s
        self.max_bytes = max_bytes
        self.warn = warn
        self.samples = collections.deque()
        self.pending = []
        self.failed = False

    def add(self, now, sample):
        self.samples.append(dict(sample, t=round(now, 3)))
        while self.samples and now - self.samples[0]['t'] > self.ring_s:
            self.samples.popleft()

    def event(self, now, kind, detail=None):
        self._write({'event': kind, 't': round(now, 3),
                     'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now)),
                     'detail': detail or {}, 'before': list(self.samples)})
        self.pending.append((now, kind))

    def flush(self, now):
        due = [p for p in self.pending if now - p[0] >= self.after_s]
        self.pending = [p for p in self.pending if now - p[0] < self.after_s]
        for t0, kind in due:
            self._write({'event': kind + ':after', 't': round(now, 3), 'event_t': round(t0, 3),
                         'after': [s for s in self.samples if s['t'] >= t0]})

    def _write(self, record):
        # The log must never take the safety node down.
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            if os.path.exists(self.path) and os.path.getsize(self.path) > self.max_bytes:
                os.replace(self.path, self.path + '.1')
            with open(self.path, 'a') as out:
                out.write(json.dumps(record, allow_nan=True) + '\n')
                out.flush()
                os.fsync(out.fileno())
        except (OSError, TypeError, ValueError) as exc:
            if not self.failed and self.warn is not None:
                self.warn('temperature log not written: %r' % (exc,))
            self.failed = True


class SafetyMonitor:
    def __init__(self):
        # Parameters
        self.max_tilt_x = rospy.get_param('~max_tilt_x', 30.0)
        self.max_tilt_y = rospy.get_param('~max_tilt_y', 30.0)
        self.motor_timeout = rospy.get_param('~motor_timeout', 2.0)
        self.motor_check_rate = rospy.get_param('~motor_check_rate', 5.0)
        self.alarm_melody = rospy.get_param('~alarm_melody', 4)
        self.startup_grace = rospy.get_param('~startup_grace', 10.0)
        self.motor_power_on_grace = rospy.get_param('~motor_power_on_grace', 8.0)
        self.motor_max_temp = rospy.get_param('~motor_max_temp', 70.0)
        self.mower_max_temp = rospy.get_param('~mower_max_temp', 70.0)
        self.temp_guard = MowerTempGuard(
            limit_c=self.mower_max_temp,
            cool_c=rospy.get_param('~mower_cool_temp', 55.0),
            hard_c=rospy.get_param('~mower_hard_temp', 100.0),
            sustain_s=rospy.get_param('~mower_temp_sustain_s', 3.0),
            hard_s=rospy.get_param('~mower_temp_hard_s', 1.5),
            max_rate_c_s=rospy.get_param('~mower_temp_max_rate_c_s', 5.0),
            valid_min_c=rospy.get_param('~mower_temp_valid_min', -30.0),
            valid_max_c=rospy.get_param('~mower_temp_valid_max', 150.0),
            fault_s=rospy.get_param('~mower_temp_fault_s', 3.0))
        self.temp_ring = TempRing(
            os.path.expanduser(rospy.get_param(
                '~mower_temp_log', '~/.vitulus/safety/mower_temp_events.jsonl')),
            warn=lambda text: rospy.logwarn('[safety] %s', text))
        self.status_gap_s = rospy.get_param('~mower_status_gap_s', 2.0)
        self.mower_rpm_max = rospy.get_param('~mower_rpm_max', 7000)

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
        # Drive motor supply (power module motor output). None = never reported
        # (no power module) -> legacy behaviour: motors are always checked.
        # Motors are OFF after boot and while docked; unpowered controllers send
        # no state, which is NOT a failure.
        self.motor_supply_on = None
        self.motor_supply_on_time = None
        self.mower_stopped_by_tilt = False
        self.mower_stopped_by_temp = False
        self.mower_status_t = None
        self.mower_status_last = None
        self.mower_rpm_bad = False
        self.mower_stop_again_t = 0.0
        self.batt_v = None
        self.batt_a = None

        # Publishers
        self.pub_motor_power = rospy.Publisher('/base/motor_power', Bool, queue_size=1)
        self.pub_alarm = rospy.Publisher('/pm/play_melody', Int16, queue_size=1)
        self.pub_mower_motor_on = rospy.Publisher('/mower/set_motor_on', Bool, queue_size=1)
        self.pub_mower_power = rospy.Publisher('/mower/set_power', Bool, queue_size=1)
        self.pub_log_info = rospy.Publisher('/nextion/log_info', String, queue_size=10)
        # Latched: the mission asks this after a blade stop it did not order.
        self.pub_mower_temp = rospy.Publisher('/safety/mower_temp', String, queue_size=1,
                                              latch=True)

        # Subscribers - motor states
        rospy.Subscriber('/base/front_left_wheel_state', Moteus_controller_state, self.motor_state_cb)
        rospy.Subscriber('/base/front_right_wheel_state', Moteus_controller_state, self.motor_state_cb)
        rospy.Subscriber('/base/rear_left_wheel_state', Moteus_controller_state, self.motor_state_cb)
        rospy.Subscriber('/base/rear_right_wheel_state', Moteus_controller_state, self.motor_state_cb)

        # Subscriber - power module (is the drive motor supply switched on?)
        rospy.Subscriber('/pm/power_status', Power_status, self.power_status_cb)

        # Subscriber - IMU
        rospy.Subscriber('bno085/imu', Imu, self.imu_cb)

        # Subscriber - Mower status
        rospy.Subscriber('/mower/status', Mower, self.mower_status_cb)

        # Timer for periodic motor check
        rospy.Timer(rospy.Duration(1.0 / self.motor_check_rate), self.check_motors)
        rospy.Timer(rospy.Duration(1.0), lambda _evt: self.temp_ring.flush(time.time()))

        rospy.loginfo("[safety] Safety monitor started. max_tilt_x=%.1f, max_tilt_y=%.1f, "
                      "motor_timeout=%.1f, startup_grace=%.1f, motor_max_temp=%.1f, mower_max_temp=%.1f",
                      self.max_tilt_x, self.max_tilt_y, self.motor_timeout,
                      self.startup_grace, self.motor_max_temp, self.mower_max_temp)
        # After a respawn the latched state is this one, not the dead node's overtemp.
        self.publish_temp_state(time.time(), 'start', None)

    def safety_log(self, message):
        """Log safety event to ROS, webui and nextion display."""
        rospy.logwarn("[safety] %s", message)
        self.pub_log_info.publish(String(data="[SAFETY] " + message))

    def power_status_cb(self, msg):
        """Track the drive motor supply; re-arm the motor monitor on every power-on."""
        self.batt_v = msg.battery_voltage
        self.batt_a = msg.battery_discharge_current
        supply_on = bool(msg.motor_out_switch)
        if supply_on and not self.motor_supply_on:
            # Deliberate power-on: forget states from before the power-off and
            # monitor again (a persisting failure kills the motors again).
            self.motor_states = {}
            self.motor_supply_on_time = rospy.Time.now()
            self.motors_killed = False
            self.motors_killed_by_temp = False
        self.motor_supply_on = supply_on

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

        # Motors deliberately unpowered (boot default, docked): nothing to check.
        if self.motor_supply_on is False:
            return
        # Just powered on: controllers need a moment before they report.
        if (self.motor_supply_on_time is not None and
                (now - self.motor_supply_on_time).to_sec() < self.motor_power_on_grace):
            return

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
        """Mower motor temperature: see MowerTempGuard."""
        now = time.time()
        gap = None if self.mower_status_t is None else now - self.mower_status_t
        self.mower_status_t = now
        self.temp_ring.add(now, {
            'seq': msg.header.seq, 'status': msg.status, 'temp': round(msg.temp, 2),
            'rpm': msg.moto_rpm, 'setpoint': msg.setpoint_rpm,
            'batt_v': self.batt_v, 'batt_a': self.batt_a})
        if gap is not None and gap >= self.status_gap_s:
            self.temp_ring.event(now, 'status_gap', {'gap_s': round(gap, 2), 'status': msg.status})
        if msg.status == 'TEMP' and self.mower_status_last != 'TEMP':
            self.temp_ring.event(now, 'firmware_temp', {'temp': msg.temp})
        self.mower_status_last = msg.status

        # Implausible rpm is the firmware's own number after a stall (1. 11. 2026:
        # 33570 rpm in an int16).  A bad sample is named, it decides nothing: the
        # temperature below still counts, so a stuck rpm cannot blind the trip.
        rpm_bad = not 0 <= msg.moto_rpm <= self.mower_rpm_max
        if rpm_bad and not self.mower_rpm_bad:
            self.temp_ring.event(now, 'rpm_implausible', {'rpm': msg.moto_rpm})
            self.safety_log("Mower rpm implausible: {}. Sample ignored, "
                            "not a stop.".format(msg.moto_rpm))
        self.mower_rpm_bad = rpm_bad

        for event, detail in self.temp_guard.feed(now, msg.temp):
            self.on_temp_event(now, event, detail, msg)

        # A latched stop holds: nobody restarts the blade while it stands.
        if (self.mower_stopped_by_temp and msg.status == 'RUN'
                and now - self.mower_stop_again_t >= 2.0):
            self.mower_stop_again_t = now
            self.pub_mower_motor_on.publish(Bool(data=False))

    def on_temp_event(self, now, event, detail, msg):
        self.temp_ring.event(now, event, detail)
        if event == 'spike':
            self.safety_log("Mower temp sensor spike: {:.1f} -> {:.1f}C in {:.1f}s. "
                            "Too fast for the motor, not a stop.".format(
                                detail['from'], detail['to'], detail['dt_s']))
        elif event == 'spike_end':
            rospy.loginfo("[safety] Mower temp spike over: peak %.1fC, %.1fs, back at %.1fC",
                          detail['peak'], detail['duration_s'], detail['to'])
        elif event == 'trip':
            self.safety_log("Mower overtemp: {:.1f}C. Mower OFF.".format(msg.temp))
            self.stop_mower_temp()
            self.mower_stopped_by_temp = True
        elif event == 'fault':
            rospy.logerr("[safety] Mower temp sensor fault: reading %s. Mower OFF, "
                         "check the NTC cable.", msg.temp)
            self.pub_log_info.publish(String(
                data="[SAFETY] Mower temp sensor fault: reading {}. Mower OFF, "
                     "check the NTC cable.".format(msg.temp)))
            self.pub_alarm.publish(Int16(data=self.alarm_melody))
            self.stop_mower_temp()
            self.mower_stopped_by_temp = True
        elif event == 'rearm':
            rospy.loginfo("[safety] Mower temperature back to %.1fC, monitor armed.", msg.temp)
            self.mower_stopped_by_temp = False
        self.publish_temp_state(now, event, msg.temp)

    def publish_temp_state(self, now, event, temp):
        self.pub_mower_temp.publish(String(data=json.dumps({
            'state': self.temp_guard.state, 'event': event, 'temp': temp, 't': now,
            'spikes': self.temp_guard.spikes})))

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
