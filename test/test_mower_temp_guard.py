#!/usr/bin/env python3
"""Mower temperature: spike or heat.  Offline, fake clock, no master.

    python3 test/test_mower_temp_guard.py      (sourced workspace)
"""
import json
import os
import shutil
import tempfile
import types
import unittest

os.environ['ROS_MASTER_URI'] = 'http://127.0.0.1:1'     # never the robot's master

import importlib.util  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    'safety_node', os.path.join(HERE, '..', 'nodes', 'safety_node.py'))
sn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sn)

# /mower/status as received on 1. 11. 2026 (ROS receipt times, firmware values).
INCIDENT = [(17.0, 20.01), (17.5, 20.01), (18.0, 20.02), (24.05, 42.51), (24.55, 42.51),
            (25.05, 90.72), (25.55, 90.72), (26.05, 20.93), (26.55, 20.93), (27.05, 20.51)]


def run(guard, readings):
    events = []
    for t, temp in readings:
        events.extend(e for e, _ in guard.feed(t, temp))
    return events


def steady(temp, start, seconds, hz=2.0):
    return [(start + i / hz, temp) for i in range(int(seconds * hz) + 1)]


class Silent(unittest.TestCase):
    """The ring goes to disk when the status ends, not only when it comes back."""

    def _node(self, tmp):
        node = sn.SafetyMonitor.__new__(sn.SafetyMonitor)
        node.temp_ring = sn.TempRing(os.path.join(tmp, 'e.jsonl'))
        node.status_silent_s = 5.0
        node.status_silent_said = False
        node.mower_status_last = 'RUN'
        node.mower_status_t = None
        return node

    def test_written_once_with_what_came_before(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        node = self._node(tmp)
        node.temp_ring_tick(None)
        self.assertFalse(os.path.exists(node.temp_ring.path))     # never spoke: says nothing
        node.temp_ring.add(sn.time.time() - 7.0, {'status': 'RUN', 'rpm': 3389})
        node.mower_status_t = sn.time.time() - 6.0
        node.temp_ring_tick(None)
        node.temp_ring_tick(None)
        rows = [json.loads(line) for line in open(node.temp_ring.path)]
        self.assertEqual([r['event'] for r in rows], ['status_silent'])
        self.assertEqual(rows[0]['before'][-1]['rpm'], 3389)
        self.assertEqual(rows[0]['detail']['status'], 'RUN')


class Guard(unittest.TestCase):

    def test_incident_is_a_spike_not_a_stop(self):
        g = sn.MowerTempGuard()
        events = run(g, INCIDENT)
        self.assertEqual(g.state, 'ok')
        self.assertIn('spike', events)
        self.assertIn('spike_end', events)
        self.assertNotIn('trip', events)

    def test_one_window_above_limit_is_ignored(self):
        g = sn.MowerTempGuard()
        events = run(g, steady(20.0, 0, 5) + [(5.5, 95.0), (6.0, 95.0)] + steady(20.0, 6.5, 3))
        self.assertNotIn('trip', events)
        self.assertEqual(g.state, 'ok')

    def test_slow_heating_trips_after_sustain(self):
        g = sn.MowerTempGuard()
        ramp = [(i * 0.5, 60.0 + i * 0.1) for i in range(100)] + steady(71.0, 50, 2.5)
        self.assertNotIn('trip', run(g, ramp))
        self.assertEqual(run(g, [(53.0, 71.0)]), ['trip'])
        self.assertEqual(g.state, 'overtemp')

    def test_jump_that_stays_still_trips(self):
        g = sn.MowerTempGuard()
        events = run(g, steady(20.0, 0, 5) + steady(90.0, 5.5, 3))
        self.assertIn('spike', events)
        self.assertIn('trip', events)

    def test_hard_limit_is_faster(self):
        g = sn.MowerTempGuard()
        run(g, steady(95.0, -100, 0))
        events = []
        for t, temp in steady(120.0, 0, 1.5):
            events.extend(e for e, _ in g.feed(t, temp))
            if t < 1.5:
                self.assertNotIn('trip', events)
        self.assertIn('trip', events)

    def test_stuck_high_trips(self):
        g = sn.MowerTempGuard()
        self.assertIn('trip', run(g, steady(85.0, 0, 3)))     # first reading, nothing to jump from

    def test_gap_does_not_fake_three_seconds(self):
        g = sn.MowerTempGuard()
        self.assertNotIn('trip', run(g, [(0.0, 80.0), (6.0, 80.0), (6.5, 80.0)]))
        self.assertIn('trip', run(g, steady(80.0, 7.0, 2.5)))

    def test_out_of_range_is_a_fault(self):
        for bad in (-273.15, 431.0, float('nan'), float('inf')):
            g = sn.MowerTempGuard()
            events = run(g, steady(20.0, 0, 2) + steady(bad, 2.5, 3))
            self.assertIn('fault', events, bad)
            self.assertNotIn('trip', events, bad)
            self.assertEqual(g.state, 'sensor_fault')

    def test_short_out_of_range_is_only_a_spike(self):
        g = sn.MowerTempGuard()
        events = run(g, steady(20.0, 0, 2) + [(2.5, -273.15), (3.0, -273.15)] + steady(20.0, 3.5, 3))
        self.assertIn('spike', events)
        self.assertEqual(g.state, 'ok')

    def test_rearm_after_cooling(self):
        g = sn.MowerTempGuard()
        run(g, steady(80.0, 0, 3))
        self.assertEqual(g.state, 'overtemp')
        self.assertNotIn('rearm', run(g, steady(60.0, 4, 30)))
        self.assertIn('rearm', run(g, steady(50.0, 40, 10)))
        self.assertIn('trip', run(g, steady(80.0, 60, 3)))     # the latch came back

    def test_stuck_zero_is_a_fault(self):
        g = sn.MowerTempGuard()
        events = run(g, steady(20.0, 0, 2) + steady(0.0, 2.5, 3))
        self.assertIn('fault', events)
        self.assertNotIn('spike', events)
        self.assertEqual(g.state, 'sensor_fault')

    def test_boot_zero_is_no_spike(self):
        g = sn.MowerTempGuard()
        self.assertEqual(run(g, [(0.0, 0.0), (1.0, 20.0), (2.0, 20.0)]), [])


class Node(unittest.TestCase):
    """The rospy shell, on real callbacks with recording publishers."""

    class Pub(object):
        def __init__(self):
            self.sent = []

        def publish(self, msg):
            self.sent.append(msg.data)

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.path = os.path.join(self.dir, 'sub', 'events.jsonl')
        self.logs = []
        self.real = (sn.rospy.logwarn, sn.rospy.logerr, sn.rospy.loginfo, sn.time.time)
        sn.rospy.logwarn = lambda f, *a: self.logs.append(f % a if a else f)
        sn.rospy.logerr = sn.rospy.logwarn
        sn.rospy.loginfo = lambda *a: None
        self.now = [1000.0]
        sn.time.time = lambda: self.now[0]
        m = sn.SafetyMonitor.__new__(sn.SafetyMonitor)
        m.temp_guard = sn.MowerTempGuard()
        m.temp_ring = sn.TempRing(self.path, after_s=5.0)
        m.status_gap_s = 2.0
        m.mower_status_t = None
        m.mower_status_last = None
        m.mower_rpm_bad = False
        m.mower_rpm_max = 7000
        m.mower_stop_again_t = 0.0
        m.mower_stopped_by_temp = False
        m.batt_v = 22.4
        m.batt_a = 4.2
        m.alarm_melody = 4
        for name in ('pub_mower_motor_on', 'pub_mower_power', 'pub_log_info', 'pub_mower_temp',
                     'pub_alarm'):
            setattr(m, name, self.Pub())
        self.m = m

    def tearDown(self):
        sn.rospy.logwarn, sn.rospy.logerr, sn.rospy.loginfo, sn.time.time = self.real

    def feed(self, t, temp, status='RUN', rpm=3400):
        self.now[0] = 1000.0 + t
        self.m.mower_status_cb(types.SimpleNamespace(
            header=types.SimpleNamespace(seq=int(t * 2)), status=status, temp=temp,
            moto_rpm=rpm, setpoint_rpm=3400))

    def records(self):
        with open(self.path) as src:
            return [json.loads(line) for line in src]

    def test_incident_keeps_power_and_leaves_a_record(self):
        for t, temp in INCIDENT:
            self.feed(t, temp, 'TEMP' if 25.0 < t < 26.0 else 'RUN')
        self.assertEqual(self.m.pub_mower_power.sent, [])
        self.assertEqual(self.m.pub_mower_motor_on.sent, [])
        self.assertFalse(self.m.mower_stopped_by_temp)
        self.assertTrue(any('spike' in line for line in self.logs))
        kinds = [r['event'] for r in self.records()]
        for kind in ('status_gap', 'firmware_temp', 'spike'):
            self.assertIn(kind, kinds)
        gap = [r for r in self.records() if r['event'] == 'status_gap'][0]
        self.assertAlmostEqual(gap['detail']['gap_s'], 6.05, places=2)
        self.assertEqual(gap['before'][-1]['batt_a'], 4.2)
        self.m.temp_ring.flush(1040.0)
        self.assertTrue(any(r['event'].endswith(':after') for r in self.records()))

    def test_firmware_temp_is_logged_on_entry_only(self):
        for i in range(10):
            self.feed(i * 0.5, 20.0, 'TEMP')
        self.feed(5.0, 20.0, 'READY')
        self.feed(5.5, 20.0, 'TEMP')
        kinds = [r['event'] for r in self.records()]
        self.assertEqual(kinds.count('firmware_temp'), 2)

    def test_init_publishes_the_state(self):
        real = (sn.rospy.Publisher, sn.rospy.Subscriber, sn.rospy.Timer, sn.rospy.get_param,
                sn.rospy.Time.now)
        pubs = {}

        def publisher(name, *_a, **_k):
            return pubs.setdefault(name, self.Pub())
        sn.rospy.Publisher = publisher
        sn.rospy.Subscriber = sn.rospy.Timer = lambda *a, **k: None
        sn.rospy.get_param = lambda _name, default=None: default
        sn.rospy.Time.now = staticmethod(lambda: 0)
        try:
            sn.SafetyMonitor()
        finally:
            (sn.rospy.Publisher, sn.rospy.Subscriber, sn.rospy.Timer, sn.rospy.get_param,
             sn.rospy.Time.now) = real
        state = json.loads(pubs['/safety/mower_temp'].sent[-1])
        self.assertEqual((state['state'], state['event']), ('ok', 'start'))

    def test_implausible_rpm_is_named_once_and_stops_nothing(self):
        self.feed(0.0, 20.0)
        self.feed(0.5, 20.0, rpm=-31966)
        self.feed(1.0, 20.0, rpm=-31966)
        self.feed(1.5, 20.0, rpm=2596)
        self.assertEqual(sum('rpm implausible' in line for line in self.logs), 1)
        self.assertEqual([r['event'] for r in self.records()], ['rpm_implausible'])
        self.assertEqual(self.m.pub_mower_motor_on.sent, [])
        self.assertEqual(self.m.pub_mower_power.sent, [])

    def test_stuck_rpm_does_not_blind_the_trip(self):
        for t, temp in steady(80.0, 0, 3):
            self.feed(t, temp, rpm=-31966)
        self.assertEqual(self.m.pub_mower_power.sent, [False])

    def test_trip_stops_and_holds(self):
        for t, temp in steady(80.0, 0, 3):
            self.feed(t, temp)
        self.assertEqual(self.m.pub_mower_power.sent, [False])
        self.assertEqual(self.m.pub_mower_motor_on.sent[0], False)
        self.assertTrue(any('Mower overtemp: 80.0C. Mower OFF.' in line for line in self.logs))
        self.assertEqual(json.loads(self.m.pub_mower_temp.sent[-1])['state'], 'overtemp')
        before = len(self.m.pub_mower_motor_on.sent)
        self.feed(10.0, 80.0, 'RUN')                  # someone started it again
        self.assertEqual(len(self.m.pub_mower_motor_on.sent), before + 1)

    def test_fault_is_reported_apart(self):
        for t, temp in steady(-273.15, 0, 3):
            self.feed(t, temp)
        self.assertTrue(any('temp sensor fault' in line for line in self.logs))
        self.assertTrue(any('temp sensor fault' in line for line in self.m.pub_log_info.sent))
        self.assertEqual(self.m.pub_alarm.sent, [4])
        self.assertEqual(json.loads(self.m.pub_mower_temp.sent[-1])['state'], 'sensor_fault')

    def test_log_failure_does_not_stop_the_node(self):
        blocker = os.path.join(self.dir, 'file')
        open(blocker, 'w').close()
        self.m.temp_ring = sn.TempRing(os.path.join(blocker, 'x', 'y.jsonl'),
                                       warn=self.logs.append)
        for t, temp in INCIDENT:
            self.feed(t, temp)
        self.assertEqual(sum('not written' in line for line in self.logs), 1)

    def test_ring_is_short_and_file_is_capped(self):
        ring = sn.TempRing(self.path, ring_s=10.0, max_bytes=200)
        for i in range(100):
            ring.add(float(i), {'temp': 20.0})
        self.assertLessEqual(len(ring.samples), 11)
        ring.event(100.0, 'a')
        ring.event(101.0, 'b')
        self.assertTrue(os.path.exists(self.path + '.1'))


if __name__ == '__main__':
    unittest.main()
