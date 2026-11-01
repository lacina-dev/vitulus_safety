#!/usr/bin/env python3
"""map_rules_guard: one reproduction per defect of the independent review.

    python3 test/test_map_rules_review.py
"""
import importlib.machinery
import importlib.util
import json
import math
import os
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(os.path.dirname(HERE), 'nodes', 'map_rules_guard')
_loader = importlib.machinery.SourceFileLoader('map_rules_guard_r', _PATH)
_spec = importlib.util.spec_from_loader('map_rules_guard_r', _loader)
g = importlib.util.module_from_spec(_spec)
_loader.exec_module(g)


def box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def rule(kind, points, name='R', **kw):
    return dict({'id': 'o_' + name, 'type': kind, 'name': name, 'points': points}, **kw)


def rules_at(rules, x=0.0, y=0.0, yaw=0.0, now=10.0):
    r = g.Rules()
    r.set_rules(rules)
    r.set_pose(now, x, y, yaw)
    return r


def run(b, t0, t1, inside, step=0.2):
    out = []
    t = t0
    while t < t1:
        out += [c for c in b.update(t, inside)]
        t += step
    return out


def mowing(status='RUN', rpm=3000):
    b = g.Blade()
    b.on_program(busy=True, program='Zahrada', stop_reason='')
    b.on_mower(status, rpm)
    return b


class H1NeverAgainstTheMission(unittest.TestCase):
    def test_a_mission_false_while_coasting_inside(self):
        b = mowing()
        self.assertEqual(b.update(0.0, True), [False])
        b.on_mower('READY', 1500)             # coasting, firmware says not running
        b.on_external(False, 0.5)             # the mission's own stop
        self.assertEqual(b.update(1.2, True), [False], 'still asserted while it spins')
        b.on_mower('READY', 0)
        self.assertEqual(run(b, 1.4, 20.0, False), [])
        self.assertEqual(b.state(), 'ok')

    def test_b_mission_false_outside_then_coasts_in(self):
        b = mowing()
        b.update(0.0, False)
        b.on_external(False, 0.1)             # detour, NavBackup
        b.on_mower('READY', 1500)
        self.assertEqual(b.update(0.5, True), [False])
        b.on_mower('READY', 0)
        self.assertEqual(run(b, 0.7, 20.0, False), [])

    def test_c_coast_seen_with_no_command_known_is_no_debt(self):
        # A respawn during a coast-down: rpm alone proves nothing.
        b = mowing('READY', 1500)
        self.assertEqual(b.update(0.0, True), [False])
        b.on_mower('READY', 0)
        self.assertEqual(run(b, 0.2, 20.0, False), [])

    def test_d_firmware_running_is_proof(self):
        b = mowing()
        self.assertEqual(b.update(0.0, True), [False])
        b.on_mower('READY', 0)
        self.assertEqual(run(b, 0.2, 6.0, False), [True])

    def test_e_observed_true_is_proof(self):
        b = mowing('READY', 0)
        b.update(0.0, True)
        self.assertTrue(b.on_external(True, 0.1))
        self.assertEqual(run(b, 0.2, 6.0, False), [True])


class H2Feed(unittest.TestCase):
    def feed(self):
        return g.RuleFeed()

    @staticmethod
    def raw(rules, site='A'):
        return json.dumps({'site': site, 'seq': 1, 'rules': rules})

    def test_null_site_while_running_keeps_the_rules(self):
        f = self.feed()
        f.on_rules(self.raw([{'x': 1}]))
        self.assertEqual(f.current(running=True), [{'x': 1}])
        f.on_rules(self.raw([], site=None))
        self.assertIsNone(f.current(running=True), 'no decision: keep what is in force')

    def test_null_site_idle_is_empty(self):
        f = self.feed()
        f.on_rules(self.raw([], site=None))
        self.assertEqual(f.current(running=False), [])

    def test_a_message_without_rules_does_not_clear(self):
        f = self.feed()
        f.on_rules(self.raw([{'x': 1}]))
        f.on_rules(json.dumps({'site': 'A', 'seq': 2}))
        self.assertEqual(f.current(running=False), [{'x': 1}])

    def test_not_the_served_site_is_dropped(self):
        f = self.feed()
        f.on_status(json.dumps({'serving': {'site': 'B'}}))
        f.on_rules(self.raw([{'x': 1}], site='A'))
        self.assertEqual(f.current(running=True), [])

    def test_served_site_passes(self):
        f = self.feed()
        f.on_status(json.dumps({'serving': {'site': 'A'}}))
        f.on_rules(self.raw([{'x': 1}], site='A'))
        self.assertEqual(f.current(running=True), [{'x': 1}])

    def test_unserved_for_a_moment_keeps_the_rules(self):
        f = self.feed()
        f.on_status(json.dumps({'serving': {'site': 'A'}}))
        f.on_rules(self.raw([{'x': 1}], site='A'))
        f.on_status(json.dumps({'serving': None}))
        self.assertEqual(f.current(running=True), [{'x': 1}])


class H3NeverDies(unittest.TestCase):
    def test_all_nan_points(self):
        nan = float('nan')
        r = rules_at([rule('nogo', [[nan, nan]] * 4, 'N')])
        lim = r.limit(10.0)
        self.assertEqual((lim['v_fwd'], lim['v_back'], lim['w']), (0.0, 0.0, 0.0))
        self.assertEqual(lim['reason'], 'nogo_unparsable:N')

    def test_a_rule_that_is_not_an_object(self):
        r = g.Rules()
        r.set_rules(['x', 7, None])
        r.set_pose(10.0, 0.0, 0.0, 0.0)
        self.assertEqual(r.limit(10.0)['v_fwd'], 0.0)
        self.assertEqual(r.limit(10.0)['rules'], 3)

    def test_unparsable_nogo_is_a_hard_limit(self):
        lim = rules_at([rule('nogo', [[0, 0]], 'N')]).limit(10.0)
        self.assertEqual(lim['v_fwd'], 0.0)

    def test_bad_rules_are_listed(self):
        r = rules_at([rule('slow', [[0, 0]], 'S')])
        self.assertEqual(r.bad, [{'type': 'slow', 'name': 'S'}])
        self.assertEqual(r.limit(10.0)['v_fwd'], g.DEFAULTS['v_fwd_max'])

    def test_a_tick_exception_ends_the_process(self):
        seen = []

        def boom(_evt):
            raise RuntimeError('geos')
        g.guarded(boom, seen.append)(None)
        self.assertEqual(len(seen), 1)


class H4FirstTick(unittest.TestCase):
    def test_nothing_before_the_first_rules(self):
        f = g.RuleFeed()
        self.assertFalse(f.ready(0.0))
        self.assertFalse(f.ready(g.DEFAULTS['no_publisher_s'] - 0.1))

    def test_after_the_first_rules(self):
        f = g.RuleFeed()
        f.on_rules(json.dumps({'site': 'A', 'rules': []}))
        f.current(running=False)
        self.assertTrue(f.ready(0.0))

    def test_no_publisher_for_a_while(self):
        f = g.RuleFeed()
        self.assertFalse(f.ready(0.0))
        self.assertFalse(f.ready(g.DEFAULTS['no_publisher_s'] - 0.1))
        self.assertTrue(f.ready(g.DEFAULTS['no_publisher_s'] + 0.1))


class M5Leaving(unittest.TestCase):
    def test_side_overlap_does_not_block_along(self):
        # Fence on the left overlapping the footprint by 5.6 cm.
        lim = rules_at([rule('nogo', box(-5, 0.20, 5, 2))], x=-4.0).limit(10.0)
        self.assertGreater(lim['v_fwd'], 0.0)
        self.assertGreater(lim['v_back'], 0.0)

    def test_pose_noise_at_a_fence_keeps_the_line(self):
        lim = rules_at([rule('nogo', box(-5, 0.246, 5, 2))], x=-4.0).limit(10.0)
        self.assertGreater(lim['v_fwd'], 0.0)

    def test_going_deeper_is_still_blocked(self):
        lim = rules_at([rule('nogo', box(0.1, -3, 8, 3))]).limit(10.0)
        self.assertEqual(lim['v_fwd'], 0.0)

    def test_parallel_edge_can_turn_a_little(self):
        lim = rules_at([rule('nogo', box(-5, 0.286, 5, 2))]).limit(10.0)   # 3 cm off
        self.assertGreater(max(lim['w_left'], lim['w_right']), 0.0)

    def test_no_sweep_in_at_contact(self):
        # 3 mm off, nose turned in: left sweeps in, whatever the noise rule says.
        wall = [rule('nogo', box(-5, 0.60, 5, 3))]
        yaw, lo, hi = 0.3, 0.0, 0.59
        probe = g.Rules()
        probe.set_rules(wall)
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if probe.footprint(0, mid, yaw).distance(probe.nogo) > 0.003 \
                else (lo, mid)
        self.assertEqual(rules_at(wall, y=lo, yaw=yaw).limit(10.0)['w_left'], 0.0)

    def test_pocket_keeps_a_way_to_turn(self):
        pocket = [rule('nogo', box(0.30, -2, 2, 2), 'F'), rule('nogo', box(-2, -2, -0.35, 2), 'B'),
                  rule('nogo', box(-2, 0.26, 2, 2), 'L')]
        lim = rules_at(pocket).limit(10.0)
        self.assertEqual((lim['v_fwd'], lim['v_back']), (0.0, 0.0))
        self.assertGreater(max(lim['w_left'], lim['w_right']), 0.0)


class M7PublishUnderTheLock(unittest.TestCase):
    def test_true_goes_out_with_the_lock_held(self):
        lock = threading.Lock()
        b = mowing('READY', 0)
        b.update(0.0, True)
        b.on_external(True, 0.0)
        held = []
        g.blade_step(lock, b, 10.0, False, lambda v: held.append((v, lock.locked())))
        self.assertEqual(held, [(True, True)])


class Review2(unittest.TestCase):
    def test_l1_mission_false_right_after_our_true_gets_one_more(self):
        b = mowing('READY', 0)
        b.update(0.0, True)
        b.on_external(True, 0.0)
        self.assertEqual(b.update(4.0, False), [True])
        self.assertTrue(g.blade_answer(b, '/mower_unit_smach', '/g', False, 4.5))
        self.assertFalse(g.blade_answer(b, '/mower_unit_smach', '/g', False, 4.6), 'once')

    def test_l1_a_later_false_is_plain(self):
        b = mowing('READY', 0)
        b.update(0.0, True)
        b.on_external(True, 0.0)
        b.update(4.0, False)
        self.assertFalse(g.blade_answer(b, '/mower_unit_smach', '/g', False, 5.5))

    def test_l5_resume_waits_for_the_mower_status(self):
        b = g.Blade()
        b.restore({'program': 'Zahrada'})
        b.on_program(busy=True, program='Zahrada', stop_reason='')
        self.assertEqual(b.update(0.0, False), [])
        self.assertEqual(b.owed(), 'Zahrada', 'running or not is not known yet')
        b.on_mower('RUN', 3000)
        b.update(1.0, False)
        self.assertEqual(b.state(), 'ok', 'it runs: nothing to warn about')

    def test_l5_with_the_timeout(self):
        b = g.Blade()
        b.restore({'program': 'Zahrada'})
        b.on_program(busy=True, program='Zahrada', stop_reason='')
        b.update(0.0, False)
        b.update(g.DEFAULTS['owed_wait_s'] + 0.1, False)
        self.assertIsNone(b.owed())

    def test_l6_said_starts_from_the_saved_value(self):
        self.assertEqual(g.first_said({'program': 'Z', 'stamp': 1.0}), ['Z', 0.0])
        self.assertEqual(g.first_said(None), [None, 0.0])
        self.assertEqual(g.first_said('garbage'), [None, 0.0])

    def test_l7_a_true_from_anybody_clears_failed(self):
        b = mowing('READY', 0)
        b.failed = True
        b.on_external(True, 1.0)
        self.assertEqual(b.state(), 'ok')

    def test_l8_rules_from_the_file_when_the_server_is_down(self):
        f = g.RuleFeed()
        f.on_status(json.dumps({'serving': {'site': 'A'}}))
        loaded = [rule('nogo', box(1, 1, 2, 2))]
        t = g.DEFAULTS['no_publisher_s'] + 0.1
        f.ready(0.0)
        self.assertFalse(f.ready(t, load=lambda site: loaded, engaged=True),
                         'not on this tick: the rules are not applied yet')
        self.assertEqual(f.current(running=True), loaded)
        self.assertTrue(f.ready(t + 0.2, load=lambda site: loaded, engaged=True))

    def test_l8_fail_closed_when_unreadable_and_engaged(self):
        f = g.RuleFeed()
        f.on_status(json.dumps({'serving': {'site': 'A'}}))
        t = g.DEFAULTS['no_publisher_s'] + 0.1
        f.ready(0.0)
        self.assertFalse(f.ready(t, load=lambda site: None, engaged=True))

    def test_l8_no_site_known_and_engaged_stays_closed(self):
        f = g.RuleFeed()
        t = g.DEFAULTS['no_publisher_s'] + 0.1
        f.ready(0.0)
        self.assertFalse(f.ready(t, load=lambda site: [], engaged=True))

    def test_l8_never_engaged_and_unreadable_is_no_rules(self):
        f = g.RuleFeed()
        t = g.DEFAULTS['no_publisher_s'] + 0.1
        f.ready(0.0)
        self.assertTrue(f.ready(t, load=lambda site: None, engaged=False))

    def test_l8_unreadable_noblade_near_the_footprint_counts(self):
        nan = float('nan')
        r = rules_at([rule('noblade', [[0, 0], [1, 0], [nan, nan]], 'N')])
        self.assertTrue(r.on_noblade(10.0))
        r.set_pose(10.0, 50.0, 50.0, 0.0)
        self.assertFalse(r.on_noblade(10.0))

    def test_l8_noblade_without_any_bounds_counts_everywhere(self):
        r = rules_at([rule('noblade', 'x', 'N')], x=50.0)
        self.assertTrue(r.on_noblade(10.0))


class Delta(unittest.TestCase):
    T = g.DEFAULTS['no_publisher_s'] + 0.1

    def served(self, site='A'):
        f = g.RuleFeed()
        f.on_status(json.dumps({'serving': {'site': site}}))
        f.ready(0.0)
        return f

    def test_d2_rules_from_disk_at_boot_with_a_silent_publisher(self):
        # ready() no longer asks who is connected: a publisher that never speaks
        # cannot keep rules on disk unenforced.
        f = self.served()
        rules = [rule('nogo', box(1, 1, 2, 2))]
        f.ready(self.T, load=lambda s: rules)
        self.assertEqual(f.current(running=False), rules)
        self.assertTrue(f.ready(self.T + 0.2, load=lambda s: rules))

    def test_d2_a_site_known_late_is_still_loaded(self):
        f = g.RuleFeed()
        f.ready(0.0)
        self.assertTrue(f.ready(self.T, load=lambda s: []), 'nothing known: no rules')
        f.on_status(json.dumps({'serving': {'site': 'A'}}))
        rules = [rule('nogo', box(1, 1, 2, 2))]
        f.ready(self.T + 2.0, load=lambda s: rules)
        self.assertEqual(f.current(running=False), rules)

    def decided(self, rules):
        f = self.served()
        f.ready(self.T, load=lambda s: rules)
        f.current(running=False)
        return f

    def test_d4_reloads_every_5_s_until_the_topic_speaks(self):
        f = self.decided([])
        later = [rule('nogo', box(1, 1, 2, 2))]
        f.ready(self.T + 4.0, load=lambda s: later)
        self.assertEqual(f.current(running=False), [])
        f.ready(self.T + 5.1, load=lambda s: later)
        self.assertEqual(f.current(running=False), later)

    def test_d4_the_topic_ends_the_reloading(self):
        f = self.decided([])
        topic = [rule('slow', box(1, 1, 2, 2))]
        f.on_rules(json.dumps({'site': 'A', 'rules': topic}))
        f.ready(self.T + 10.0, load=lambda s: [rule('nogo', box(1, 1, 2, 2))])
        self.assertEqual(f.current(running=False), topic)

    def test_d4_unreadable_reload_keeps_what_stands(self):
        rules = [rule('nogo', box(1, 1, 2, 2))]
        f = self.decided(rules)
        self.assertTrue(f.ready(self.T + 6.0, load=lambda s: None))
        self.assertEqual(f.current(running=False), rules)

    def test_d4_a_served_site_change_works(self):
        f = self.decided([rule('nogo', box(1, 1, 2, 2))])
        f.on_status(json.dumps({'serving': {'site': 'B'}}))
        other = [rule('slow', box(1, 1, 2, 2))]
        f.ready(self.T + 1.5, load=lambda s: other if s == 'B' else None)
        self.assertEqual(f.current(running=False), other)

    def test_d4_a_new_site_unreadable_fails_closed(self):
        f = self.decided([rule('nogo', box(1, 1, 2, 2))])
        f.on_status(json.dumps({'serving': {'site': 'B'}}))
        self.assertFalse(f.ready(self.T + 1.5, load=lambda s: None, engaged=True))


class BadMarker(unittest.TestCase):
    def test_a_marked_nogo_stops_even_with_good_points(self):
        r = rules_at([rule('nogo', box(50, 50, 51, 51), 'X', bad='bad_type')])
        lim = r.limit(10.0)
        self.assertEqual((lim['v_fwd'], lim['w']), (0.0, 0.0))
        self.assertEqual(lim['reason'], 'nogo_unparsable:X')

    def test_a_marked_noblade_without_points_stops_the_blade_not_the_robot(self):
        r = rules_at([rule('noblade', [], 'N', bad='bad_geometry')], x=50.0)
        self.assertTrue(r.on_noblade(10.0))
        self.assertEqual(r.limit(10.0)['v_fwd'], g.DEFAULTS['v_fwd_max'])
        self.assertEqual(r.bad, [{'type': 'noblade', 'name': 'N'}])


class M10Geometry(unittest.TestCase):
    def test_a_bowtie_keeps_both_lobes(self):
        bow = [[0, 0], [2, 2], [2, 0], [0, 2]]
        r = rules_at([rule('nogo', bow)], x=50.0)
        self.assertAlmostEqual(r.nogo.area, 2.0)
        self.assertEqual(r.bad, [])


class Low(unittest.TestCase):
    def test_dock_lane_is_as_wide_as_the_tail(self):
        objs = [{'family': 'waypoint', 'name': n, 'points': [[x, 0.0]]}
                for n, x in (('UNDOCK', 3.0), ('DOCK', 1.5), ('DOCKED', 0.0))]
        r = rules_at([rule('nogo', box(1.0, 0.42, 2.0, 1.0), 'Near')])
        r.set_dock_points(objs)
        self.assertEqual(r.dock_conflicts(), ['Near'])

    def test_a_big_nogo_is_quick(self):
        r = rules_at([rule('nogo', box(-500, -500, 500, 500))])
        t0 = time.monotonic()
        r.limit(10.0)
        self.assertLess(time.monotonic() - t0, 0.1)

    def test_a_clock_step_back_is_stale(self):
        self.assertIsNone(g.dated(10.0, -5.0))
        self.assertEqual(g.dated(10.0, 0.5), 9.5)
        self.assertEqual(g.dated(10.0, -0.05), 10.0)

    def test_boundary_cycling_has_a_gap(self):
        b = mowing()
        b.update(0.0, True)
        b.on_mower('READY', 0)
        self.assertEqual(b.update(0.2, False), [], 'just out: not yet')
        self.assertEqual(run(b, 0.4, g.DEFAULTS['blade_on_gap_s'] - 0.2, False), [])
        self.assertEqual(run(b, g.DEFAULTS['blade_on_gap_s'], 5.0, False), [True])


if __name__ == '__main__':
    unittest.main()
