#!/usr/bin/env python3
"""map_rules_guard zone fence, offline: the blade cuts only in the zone the
program mows.  No ROS master, a fake pose and a fake clock.

    python3 -m pytest test/test_map_rules_zone.py      (or plain python3)
"""
import importlib.machinery
import importlib.util
import json
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(os.path.dirname(HERE), 'nodes', 'map_rules_guard')
_loader = importlib.machinery.SourceFileLoader('map_rules_guard', _PATH)
_spec = importlib.util.spec_from_loader('map_rules_guard', _loader)
g = importlib.util.module_from_spec(_spec)
_loader.exec_module(g)

TICK = 1.0 / g.DEFAULTS['rate_hz']


def zone(name='A', x0=0.0, y0=0.0, x1=10.0, y1=10.0, program='P', frame='map'):
    return json.dumps({'program': program, 'zone': name, 'frame': frame,
                       'points': [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]})


def fence(raw=None, **params):
    f = g.ZoneFence(params)
    f.set_zone(zone() if raw is None else raw)
    return f


def drive(f, xs, y=5.0, t0=0.0):
    """The robot centre at x in xs, one guard tick each: [outside] (None, still in
    the hold, keeps the last)."""
    out = []
    for i, x in enumerate(xs):
        r = f.update(t0 + i * TICK, (x, y))
        out.append(f.out if r is None else r)
    return out


class Edge(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(g.DEFAULTS['zone_edge_m'], 0.15)
        self.assertEqual(g.DEFAULTS['zone_hysteresis_m'], 0.03)
        self.assertEqual(g.DEFAULTS['zone_hold_s'], 0.2)
        self.assertEqual(g.DEFAULTS['zone_on_gap_s'], 1.0)
        self.assertTrue(g.DEFAULTS['zone_fence'])

    def test_inside_and_on_the_edge_cut(self):
        # The outline path runs >= 6 cm inside, the trail saw 6 cm outside at worst.
        f = fence()
        self.assertEqual(drive(f, [5.0, 9.9, 10.0, 10.06, 10.10, 10.15] * 4), [False] * 24)
        self.assertEqual(f.item()['outside'], False)

    def test_beyond_the_edge_stops_after_the_hold(self):
        f = fence()
        out = drive(f, [10.0, 10.17, 10.2, 10.25, 10.5])
        self.assertEqual(out, [False, False, True, True, True])     # 2 ticks = 0.2 s

    def test_one_tick_spike_is_no_stop(self):
        f = fence()
        self.assertEqual(drive(f, [10.0, 10.4, 10.0, 10.4, 10.05, 10.0]), [False] * 6)

    def test_hold_zero_stops_at_once(self):
        f = fence(zone_hold_s=0.0)
        self.assertEqual(drive(f, [10.0, 10.16]), [False, True])

    def test_hysteresis_no_flapping_on_the_band(self):
        f = fence()
        out = drive(f, [10.5, 10.5, 10.14, 10.13, 10.16, 10.14, 10.125])
        self.assertEqual(out, [False, True, True, True, True, True, True])
        self.assertEqual(drive(f, [10.12], t0=2.0), [False])
        self.assertEqual(drive(f, [10.14, 10.15], t0=3.0), [False, False])

    def test_any_side_and_corner(self):
        f = fence(zone_hold_s=0.0)
        self.assertTrue(f.update(0.0, (-0.2, 5.0)))
        f = fence(zone_hold_s=0.0)
        self.assertTrue(f.update(0.0, (10.12, 10.12)))   # 17 cm from the corner
        f = fence(zone_hold_s=0.0)
        self.assertFalse(f.update(0.0, (10.1, 10.1)))    # 14 cm

    def test_unknown_pose_keeps_the_last(self):
        f = fence(zone_hold_s=0.0)
        self.assertTrue(f.update(0.0, (11.0, 5.0)))
        self.assertIsNone(f.update(0.2, None))
        self.assertTrue(f.out)

    def test_parameters(self):
        f = fence(zone_edge_m=0.3, zone_hysteresis_m=0.1, zone_hold_s=0.0)
        self.assertEqual(drive(f, [10.25, 10.31, 10.21, 10.19]), [False, True, True, False])


class Zones(unittest.TestCase):
    def test_no_zone_no_fence(self):
        f = fence('{}')
        self.assertEqual(drive(f, [50.0, 50.0, 50.0]), [False] * 3)
        self.assertIsNone(f.item())

    def test_between_zones_no_fence(self):
        f = fence(zone_hold_s=0.0)
        self.assertTrue(f.update(0.0, (12.0, 5.0)))
        self.assertTrue(f.set_zone('{}'))       # SET_MOTOR_OFF_AND_HOME: changed
        self.assertFalse(f.set_zone('{}'))      # once
        self.assertFalse(f.update(0.2, (12.0, 5.0)))
        self.assertIsNone(f.item())

    def test_next_zone_starts_fresh(self):
        f = fence(zone_hold_s=0.0)
        self.assertTrue(f.update(0.0, (12.0, 5.0)))
        self.assertFalse(f.set_zone(zone('A')))  # the same zone again: no change
        self.assertTrue(f.set_zone(zone('B', 11.0, 0.0, 20.0, 10.0)))
        self.assertFalse(f.update(0.2, (12.0, 5.0)))
        self.assertEqual(f.item()['name'], 'B')
        self.assertTrue(f.update(0.4, (5.0, 5.0)))   # back in A is outside B

    def test_unusable_zone_is_no_fence(self):
        for raw in ('nonsense', '[1, 2]', zone(frame='odom'),
                    json.dumps({'zone': 'A', 'points': [[0, 0], [1, 1]]})):
            said = []
            f = g.ZoneFence(warn=said.append)
            f.set_zone(raw)
            self.assertFalse(f.update(0.0, (50.0, 50.0)), raw)
            self.assertTrue(said, raw)

    def test_any_true(self):
        self.assertTrue(g.any_true(True, None))
        self.assertTrue(g.any_true(False, True))
        self.assertIsNone(g.any_true(False, None))
        self.assertFalse(g.any_true(False, False))


def mowing():
    b = g.Blade()
    b.on_program(busy=True, program='P', stop_reason='')
    b.on_mower('RUN', 3400)
    b.on_external(True, 0.0)               # the mission's SET_MOTOR_ON
    return b


def run(b, f, xs, t0=1.0, noblade=False):
    """Ticks of the guard; the firmware answers each command."""
    sent = []
    for i, x in enumerate(xs):
        now = t0 + i * TICK
        out = f.update(now, (x, 5.0))
        for value in b.update(now, g.any_true(noblade, out), out is True and noblade is not True):
            sent.append((round(now, 1), value))
            b.on_mower('RUN' if value else 'READY', 3400 if value else 0)
    return sent


class BladeAcrossTheEdge(unittest.TestCase):
    def test_off_outside_on_again_inside(self):
        b, f = mowing(), fence()
        out = [10.0 + 0.1 * i for i in range(1, 21)]      # 2 m out
        back = [12.0 - 0.1 * i for i in range(1, 31)]     # and back in
        stay = [5.0] * 30
        sent = run(b, f, out + back + stay)
        values = [v for _t, v in sent]
        self.assertEqual(values[0], False)
        self.assertEqual(values[-1], True)
        self.assertEqual(values.count(True), 1)           # one start, no cycling
        t_off = sent[0][0]
        t_on = [t for t, v in sent if v][0]
        self.assertLessEqual(t_off, 1.0 + 3 * TICK + 1e-6)   # 30 cm past the edge at most
        self.assertGreaterEqual(t_on - t_off, g.DEFAULTS['zone_on_gap_s'] - 1e-6)
        self.assertEqual(b.state(), 'ok')

    def test_back_after_a_long_way_out_starts_at_once(self):
        b, f = mowing(), fence()
        sent = run(b, f, [11.0] * 50 + [9.0] * 5)          # 10 s out, 1 Hz re-asserts
        self.assertEqual(sent[-1], (round(1.0 + 50 * TICK, 1), True))

    def test_crossed_twice_quickly_waits_the_gap(self):
        b, f = mowing(), fence()
        sent = run(b, f, [11.0] * 3 + [9.0] * 30)
        t_off = sent[0][0]
        t_on = [t for t, v in sent if v][0]
        self.assertAlmostEqual(t_on - t_off, g.DEFAULTS['zone_on_gap_s'], places=6)

    def test_noblade_keeps_its_gap(self):
        b, f = mowing(), fence('{}')
        sent = run(b, f, [11.0] * 3, noblade=True) + run(b, f, [11.0] * 30, t0=1.6)
        t_off = sent[0][0]
        t_on = [t for t, v in sent if v][0]
        self.assertGreaterEqual(t_on - t_off, g.DEFAULTS['blade_on_gap_s'] - 1e-6)
        self.assertEqual(g.blade_text(True, False, ''), 'on, left the noblade area')

    def test_state_while_outside(self):
        b, f = mowing(), fence()
        run(b, f, [11.0] * 5)
        self.assertEqual(b.state(), 'stopped_by_rule')
        self.assertEqual(b.held_by(), 'zone')
        self.assertTrue(f.item()['outside'])
        self.assertEqual(f.item()['out_m'], 1.0)

    def test_a_true_outside_is_answered(self):
        b, f = mowing(), fence()
        run(b, f, [11.0] * 5)
        b.on_mower('RUN', 3400)                 # recovery restarts the blade out there
        self.assertTrue(b.on_external(True, 3.0))
        self.assertEqual(b.state(), 'stopped_by_rule')

    def test_the_missions_own_off_stays_off(self):
        b, f = mowing(), fence()
        run(b, f, [11.0] * 5)
        b.on_external(False, 2.0)              # a detour / zone end of the mission
        sent = run(b, f, [5.0] * 40, t0=2.2)
        self.assertEqual([v for _t, v in sent if v], [])

    def test_edge_mowing_never_touches_the_blade(self):
        b, f = mowing(), fence()
        edge = [10.0, 10.04, 10.06, 10.09, 10.1, 10.05, 9.95] * 20
        self.assertEqual(run(b, f, edge), [])

    def test_transit_between_zones(self):
        b, f = mowing(), fence()
        b.on_external(False, 0.5)               # SET_MOTOR_OFF_AND_HOME
        b.on_mower('READY', 0)
        f.set_zone('{}')
        self.assertEqual(run(b, f, [11.0 + 0.1 * i for i in range(40)]), [])
        f.set_zone(zone('B', 14.0, 0.0, 30.0, 10.0))
        self.assertEqual(run(b, f, [13.0, 13.5, 14.5, 15.0], t0=10.0), [])
        b.on_external(True, 11.0)               # SET_MOTOR_ON in B
        b.on_mower('RUN', 3400)
        self.assertEqual(run(b, f, [15.0] * 10, t0=11.2), [])

    def test_zone_end_before_the_missions_false(self):
        # safe_blade_shutdown: {} may overtake the mission's False on the wire.
        b, f = mowing(), fence()
        run(b, f, [11.0] * 5)
        self.assertEqual(b.held_by(), 'zone')
        if f.set_zone('{}'):
            b.zone_changed()
        sent = run(b, f, [11.0] * 10 + [5.0] * 20, t0=2.0)
        b.on_external(False, 8.0)
        sent += run(b, f, [5.0] * 20, t0=8.2)
        self.assertEqual([v for _t, v in sent if v], [])
        self.assertEqual(b.state(), 'ok')

    def test_zone_end_after_the_missions_false(self):
        b, f = mowing(), fence()
        run(b, f, [11.0] * 5)
        b.on_external(False, 2.0)
        if f.set_zone('{}'):
            b.zone_changed()
        self.assertEqual([v for _t, v in run(b, f, [5.0] * 20, t0=2.2) if v], [])

    def test_next_zone_drops_the_debt(self):
        b, f = mowing(), fence()
        run(b, f, [11.0] * 5)
        if f.set_zone(zone('B', 11.0, 0.0, 20.0, 10.0)):
            b.zone_changed()
        self.assertEqual([v for _t, v in run(b, f, [12.0] * 20, t0=2.0) if v], [])

    def test_noblade_on_the_way_back_keeps_the_fence_stop(self):
        b, f = mowing(), fence()
        sent = run(b, f, [11.0] * 5)                       # out of the zone: off
        sent += run(b, f, [9.0] * 5, t0=2.0, noblade=True)  # back in, a noblade area
        self.assertEqual(b.held_by(), 'zone')
        self.assertEqual(b.state(), 'stopped_by_rule')
        self.assertNotIn(True, [v for _t, v in sent])
        sent = run(b, f, [5.0] * 30, t0=3.0)                # out of the area
        t_on = [t for t, v in sent if v][0]
        self.assertLessEqual(t_on - 3.0, g.DEFAULTS['zone_on_gap_s'] + 1e-6)

    def test_noblade_debt_survives_a_zone_change(self):
        b, f = mowing(), fence('{}')
        run(b, f, [5.0] * 5, noblade=True)
        f.set_zone(zone())
        b.zone_changed()
        self.assertEqual(b.held_by(), 'noblade')

    def test_log_lines(self):
        self.assertEqual(g.blade_text(False, True, 'Bigger'), 'off, outside the zone Bigger')
        self.assertEqual(g.blade_text(True, True, 'Bigger'), 'on, back in the zone Bigger')
        self.assertEqual(g.blade_text(False, False, 'Bigger'), 'off, noblade area')


class Respawn(unittest.TestCase):
    def restarted(self, owed_zone='A', zone_now='A'):
        b = g.Blade()
        b.restore({'program': 'P', 'zone': owed_zone, 'stamp': 0.0})
        b.on_program(busy=True, program='P', stop_reason='')
        b.on_mower('READY', 0)
        if zone_now is not None:
            b.on_zone(zone_now)
        return b

    def test_waits_for_the_zone_message(self):
        # The latched zone comes after the other inputs: no restore_failed meanwhile.
        b, f = self.restarted(zone_now=None), fence()
        run(b, f, [11.0] * 10)
        self.assertFalse(b.failed)
        self.assertNotEqual(b.state(), 'restore_failed')
        b.on_zone('A')
        run(b, f, [11.0] * 3, t0=3.0)
        self.assertEqual(b.state(), 'stopped_by_rule')

    def test_no_zone_message_gives_up(self):
        b, f = self.restarted(zone_now=None), fence()
        run(b, f, [11.0] * 40)                   # > owed_wait_s
        self.assertIsNone(b.resume)
        self.assertNotEqual(b.state(), 'stopped_by_rule')

    def test_other_zone_stays_failed(self):
        b, f = self.restarted(owed_zone='B'), fence()
        run(b, f, [11.0] * 5)
        self.assertEqual(b.state(), 'restore_failed')

    def test_old_format_without_zone_stays_failed(self):
        b = g.Blade()
        b.restore({'program': 'P', 'stamp': 0.0})
        b.on_program(busy=True, program='P', stop_reason='')
        b.on_mower('READY', 0)
        run(b, fence(), [11.0] * 5)
        self.assertEqual(b.state(), 'restore_failed')

    def test_owed_zone_is_saved(self):
        b, f = mowing(), fence()
        b.on_zone('A')
        run(b, f, [11.0] * 5)
        self.assertEqual((b.owed(), b.owed_zone()), ('P', 'A'))
        b2, f2 = mowing(), fence('{}')
        b2.on_zone('')
        run(b2, f2, [5.0] * 5, noblade=True)
        self.assertEqual((b2.owed(), b2.owed_zone()), ('P', ''))

    def test_outside_the_zone_is_a_stop_by_rule(self):
        b, f = self.restarted(), fence()
        run(b, f, [11.0] * 5)
        self.assertEqual(b.state(), 'stopped_by_rule')
        self.assertFalse(b.failed)
        sent = run(b, f, [11.0] * 5 + [5.0] * 10, t0=2.0)
        self.assertIn(True, [v for _t, v in sent])           # back in the zone: on

    def test_unknown_pose_waits(self):
        b, f = self.restarted(), fence()
        for i in range(30):
            self.assertEqual(b.update(1.0 + i * TICK, None), [])
        self.assertFalse(b.failed)
        run(b, f, [11.0] * 3, t0=8.0)
        self.assertEqual(b.state(), 'stopped_by_rule')

    def test_inside_the_zone_stays_failed(self):
        b, f = self.restarted(), fence()
        run(b, f, [5.0] * 5)
        self.assertEqual(b.state(), 'restore_failed')

    def test_noblade_area_stays_failed(self):
        b, f = self.restarted(), fence('{}')
        run(b, f, [5.0] * 5, noblade=True)
        self.assertNotEqual(b.state(), 'stopped_by_rule')


class BladeNoProgram(unittest.TestCase):
    def test_no_program_no_command(self):
        b, f = g.Blade(), fence()
        b.on_program(busy=False, program='', stop_reason='')
        b.on_mower('RUN', 3400)
        self.assertEqual(run(b, f, [12.0] * 10), [])


if __name__ == '__main__':
    unittest.main()
