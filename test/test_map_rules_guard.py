#!/usr/bin/env python3
"""map_rules_guard logic, offline: no ROS master, a fake pose and a fake clock.

    python3 -m pytest test/test_map_rules_guard.py      (or plain python3)
"""
import importlib.machinery
import importlib.util
import math
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(os.path.dirname(HERE), 'nodes', 'map_rules_guard')
_loader = importlib.machinery.SourceFileLoader('map_rules_guard', _PATH)
_spec = importlib.util.spec_from_loader('map_rules_guard', _loader)
g = importlib.util.module_from_spec(_spec)
_loader.exec_module(g)


def box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def rule(kind, points, name='R', rid=None, **kw):
    return dict({'id': rid or 'o_' + name, 'type': kind, 'name': name, 'points': points}, **kw)


def rules_at(rules, x=0.0, y=0.0, yaw=0.0, now=10.0):
    r = g.Rules()
    r.set_rules(rules)
    r.set_pose(now, x, y, yaw)
    return r


class NoRules(unittest.TestCase):
    def test_no_rules_changes_nothing(self):
        r = g.Rules()
        r.set_rules([])
        lim = r.limit(10.0)                       # no pose either
        self.assertEqual(lim['rules'], 0)
        self.assertEqual(lim['reason'], '')
        self.assertEqual(lim['v_fwd'], g.DEFAULTS['v_fwd_max'])
        self.assertEqual(r.on_noblade(10.0), False)
        self.assertEqual(r.human_warnings(10.0), [])
        self.assertEqual(r.dock_conflicts(), [])

    def test_blade_never_commanded_without_rules(self):
        b = g.Blade()
        b.on_program(busy=True, program='P', stop_reason='')
        b.on_mower('RUN', 3000)
        b.on_external(True)
        out = [b.update(float(t), False) for t in range(10)]
        self.assertEqual(sum(out, []), [])
        self.assertEqual(b.state(), 'ok')


class Nogo(unittest.TestCase):
    AHEAD = [rule('nogo', box(1.0, -2.0, 2.0, 2.0), 'Bazen')]

    def test_far_is_not_limited(self):
        lim = rules_at([rule('nogo', box(20, -1, 21, 1))]).limit(10.0)
        self.assertEqual(lim['v_fwd'], g.DEFAULTS['v_fwd_max'])
        self.assertEqual(lim['rules'], 1)

    def test_tapers_to_zero_at_the_margin(self):
        front = 0.238
        speeds = []
        for gap in (0.8, 0.5, 0.3, 0.2, 0.16, 0.15, 0.10):
            lim = rules_at(self.AHEAD, x=1.0 - front - gap).limit(10.0)
            speeds.append(lim['v_fwd'])
            self.assertIn('nogo:Bazen', lim['reason'])
            self.assertEqual(lim['v_back'], g.DEFAULTS['v_back_max'], 'leaving stays free')
        self.assertEqual(speeds, sorted(speeds, reverse=True))
        self.assertGreater(speeds[0], 0.3)
        self.assertEqual(speeds[-2], 0.0)
        self.assertEqual(speeds[-1], 0.0)

    def test_stops_short_of_the_margin_when_simulated(self):
        # Worst case: reads limit at 5 Hz, then 0.2 s of old command, braking at 1 m/s2.
        x, v = -1.0, 0.75
        for _ in range(200):
            cap = rules_at(self.AHEAD, x=x).limit(10.0)['v_fwd']
            for _ in range(20):
                want = min(0.75, cap)
                v = max(want, v - 1.0 * 0.01) if v > want else want
                x += v * 0.01
        self.assertLess(x + 0.238, 1.0 - 0.14)
        self.assertLess(v, 0.02)

    def test_direction_behind(self):
        behind = [rule('nogo', box(-2.0, -2.0, -1.0, 2.0))]
        lim = rules_at(behind, x=-1.0 + 0.288 + 0.10).limit(10.0)
        self.assertEqual(lim['v_back'], 0.0)
        self.assertEqual(lim['v_fwd'], g.DEFAULTS['v_fwd_max'])

    def test_beside_does_not_slow_along(self):
        beside = [rule('nogo', box(-5.0, 0.30, 5.0, 2.0))]
        lim = rules_at(beside).limit(10.0)
        self.assertEqual(lim['v_fwd'], g.DEFAULTS['v_fwd_max'])
        self.assertEqual(lim['v_back'], g.DEFAULTS['v_back_max'])

    def test_rotation_cannot_sweep_in_and_can_turn_away(self):
        # A wall of nogo on the left, nose turned 0.3 rad towards it, 3 mm off.
        wall = [rule('nogo', box(-5.0, 0.60, 5.0, 3.0))]
        yaw = 0.3
        y = None
        lo, hi = 0.0, 0.59
        for _ in range(30):
            mid = (lo + hi) / 2
            d = rules_at(wall, y=mid, yaw=yaw).footprint(0, mid, yaw).distance(
                rules_at(wall).nogo)
            lo, hi = (mid, hi) if d > 0.003 else (lo, mid)
            y = lo
        r = rules_at(wall, y=y, yaw=yaw)
        lim = r.limit(10.0)
        self.assertEqual(lim['w_left'], 0.0, 'turning in')
        self.assertGreater(lim['w_right'], 0.1, 'turning away')
        self.assertEqual(lim['w'], lim['w_right'])
        self.assertEqual(lim['v_fwd'], 0.0)
        self.assertGreater(lim['v_back'], 0.0)

    def test_the_turn_tapers_before_the_touch(self):
        wall = [rule('nogo', box(-5.0, 0.33, 5.0, 3.0))]
        lim = rules_at(wall).limit(10.0)      # corner radius 0.35 m, 7 cm gap
        self.assertLess(lim['w_left'], g.DEFAULTS['w_max'])
        self.assertGreater(lim['w_left'], 0.0)

    def test_inside_leaves_the_short_way(self):
        pool = [rule('nogo', box(0.1, -3.0, 8.0, 3.0))]   # nose in by 0.138 m
        lim = rules_at(pool).limit(10.0)
        self.assertEqual(lim['v_fwd'], 0.0)
        self.assertEqual(lim['v_back'], g.DEFAULTS['escape_v'])
        self.assertGreater(lim['w'], 0.0)

    def test_deep_inside_moves_at_all(self):
        lim = rules_at([rule('nogo', box(-3, -3, 3, 3))]).limit(10.0)
        self.assertEqual(lim['v_fwd'], g.DEFAULTS['escape_v'])
        self.assertEqual(lim['v_back'], g.DEFAULTS['escape_v'])
        self.assertEqual(lim['w_left'], g.DEFAULTS['escape_w'])

    def test_tf_stale_stops(self):
        r = rules_at(self.AHEAD, now=10.0)
        lim = r.limit(11.5)
        self.assertEqual((lim['v_fwd'], lim['v_back'], lim['w']), (0.0, 0.0, 0.0))
        self.assertEqual(lim['reason'], 'tf_stale')

    def test_never_any_pose_stops_with_rules(self):
        r = g.Rules()
        r.set_rules(self.AHEAD)
        self.assertEqual(r.limit(10.0)['v_fwd'], 0.0)

    def test_unusable_rule_still_counts(self):
        r = rules_at([rule('nogo', [[0, 0]])])
        self.assertEqual(r.limit(10.0)['rules'], 1)

    def test_human_warning_once_per_entry(self):
        r = rules_at(self.AHEAD, x=1.2)
        self.assertEqual(r.human_warnings(10.0), [])    # software: no warning
        r.set_human('teleop')
        w = r.human_warnings(10.0)
        self.assertEqual([x['name'] for x in w], ['Bazen'])
        self.assertEqual(r.human_warnings(10.0), [])
        r.set_pose(10.0, -3.0, 0.0, 0.0)
        r.human_warnings(10.0)
        r.set_pose(10.0, 1.2, 0.0, 0.0)
        self.assertEqual(len(r.human_warnings(10.0)), 1)

    def test_dock_conflict_is_reported(self):
        objs = [{'family': 'waypoint', 'name': n, 'points': [[x, 0.0]]}
                for n, x in (('UNDOCK', 3.0), ('DOCK', 1.5), ('docked', 0.0))]
        r = rules_at([rule('nogo', box(1.0, 0.3, 2.0, 1.0), 'Near'),
                      rule('nogo', box(1.0, 2.0, 2.0, 3.0), 'Far')])
        r.set_dock_points(objs)
        self.assertEqual(r.dock_conflicts(), ['Near'])
        objs[2]['name'] = 'DOCKED'             # both spellings of the protected point
        r.set_dock_points(objs)
        self.assertEqual(r.dock_conflicts(), ['Near'])


class Slow(unittest.TestCase):
    ZONE = [rule('slow', box(0.5, -2, 5, 2), 'Pes', speed_mps=0.1)]

    def test_cap_on_overlap_only(self):
        self.assertEqual(rules_at(self.ZONE, x=-1.0).limit(10.0)['v_fwd'], 1.4)
        lim = rules_at(self.ZONE, x=0.4).limit(10.0)
        self.assertEqual((lim['v_fwd'], lim['v_back']), (0.1, 0.1))
        self.assertEqual(lim['reason'], 'slow:Pes')
        self.assertEqual(lim['w'], g.DEFAULTS['w_max'])

    def test_boundary_noise_does_not_flap(self):
        r = rules_at(self.ZONE, x=0.3)
        caps = []
        for x in (0.3, 0.25, 0.3, 0.2, 0.28, 0.21, 0.27, 0.2):   # front edge 0.44-0.54
            r.set_pose(10.0, x, 0.0, 0.0)
            caps.append(r.limit(10.0)['v_fwd'])
        self.assertEqual(set(caps), {0.1})
        r.set_pose(10.0, 0.0, 0.0, 0.0)
        self.assertEqual(r.limit(10.0)['v_fwd'], 1.4)

    def test_speed_is_bounded_and_defaulted(self):
        self.assertEqual(rules_at([rule('slow', box(-1, -1, 1, 1))]).limit(10.0)['v_fwd'], 0.2)
        fast = [rule('slow', box(-1, -1, 1, 1), speed_mps=9)]
        self.assertEqual(rules_at(fast).limit(10.0)['v_fwd'], 0.4)


class Noblade(unittest.TestCase):
    def test_overlap_and_hysteresis(self):
        r = rules_at([rule('noblade', box(0.2, -2, 5, 2))])
        self.assertTrue(r.on_noblade(10.0))
        r.set_pose(10.0, -0.1, 0.0, 0.0)       # 6 cm out
        self.assertTrue(r.on_noblade(10.0))
        r.set_pose(10.0, -0.3, 0.0, 0.0)
        self.assertFalse(r.on_noblade(10.0))

    def test_unknown_pose_is_unknown(self):
        r = rules_at([rule('noblade', box(0.2, -2, 5, 2))])
        self.assertIsNone(r.on_noblade(20.0))


def run_blade(b, t0, t1, inside, step=0.2):
    out = []
    t = t0
    while t < t1:
        out += [(round(t, 1), c) for c in b.update(t, inside)]
        t += step
    return out


def mowing():
    b = g.Blade()
    b.on_program(busy=True, program='Zahrada', stop_reason='')
    b.on_mower('RUN', 3000)
    return b


class Blade(unittest.TestCase):
    def test_off_then_on_after_exit(self):
        b = mowing()
        self.assertEqual(b.update(0.0, True), [False])
        self.assertEqual(b.state(), 'stopped_by_rule')
        b.on_mower('READY', 0)
        self.assertEqual(b.update(0.2, False), [])
        self.assertEqual(b.state(), 'restoring')
        self.assertEqual(b.update(3.2, False), [True])
        b.on_mower('RUN', 2500)
        b.update(3.4, False)
        self.assertEqual(b.state(), 'ok')
        self.assertEqual(run_blade(b, 3.6, 8.0, False), [])

    def test_reassert_at_most_1hz(self):
        b = mowing()
        cmds = run_blade(b, 0.0, 5.0, True)
        self.assertEqual([c for _, c in cmds], [False] * 5)

    def test_smach_turns_it_on_inside(self):
        b = mowing()
        b.update(0.0, True)
        b.on_mower('READY', 0)
        cmds = run_blade(b, 1.2, 3.0, True)
        self.assertEqual([c for _, c in cmds], [False, False])
        self.assertGreaterEqual(cmds[1][0] - cmds[0][0], 1.0)
        self.assertTrue(b.on_external(True, 3.2), 'recovery / DETOUR on: answered at once')
        self.assertEqual(b.state(), 'stopped_by_rule')

    def test_err_autoretry_is_reasserted(self):
        b = mowing()
        b.update(0.0, True)
        b.on_mower('ERR', 0)
        self.assertEqual(b.update(1.0, True), [False])
        b.on_mower('RUN', 800)                # firmware retried on its own
        self.assertEqual(b.update(2.0, True), [False])

    def test_never_on_if_we_did_not_stop_it(self):
        b = g.Blade()
        b.on_program(busy=True, program='Zahrada', stop_reason='')
        b.on_mower('READY', 0)                # transit, blade off
        self.assertEqual(run_blade(b, 0.0, 3.0, True), [])
        self.assertEqual(run_blade(b, 3.0, 12.0, False), [])

    def test_no_on_after_a_stop_reason(self):
        b = mowing()
        b.update(0.0, True)
        b.on_mower('READY', 0)
        b.on_program(stop_reason='interrupted:battery')
        self.assertEqual(run_blade(b, 1.0, 9.0, False), [])

    def test_no_on_after_the_program_ended(self):
        b = mowing()
        b.update(0.0, True)
        b.on_mower('READY', 0)
        b.on_program(busy=False, program=' ')
        self.assertEqual(run_blade(b, 1.0, 9.0, False), [])

    def test_no_on_when_the_mission_switched_it_off(self):
        # Transit between zones, or a detour: the mission's own False stands.
        b = mowing()
        b.update(0.0, True)
        b.on_mower('READY', 0)
        b.on_external(False)
        self.assertEqual(run_blade(b, 1.0, 9.0, False), [])

    def test_chain_next_program_inside(self):
        b = mowing()
        b.update(0.0, True)
        b.on_mower('READY', 0)
        b.on_external(False)                  # safe_blade_shutdown of program A
        b.on_program(program='B')
        self.assertEqual(run_blade(b, 1.0, 2.0, True), [])
        self.assertTrue(b.on_external(True, 2.0))      # B: SET_MOTOR_ON
        self.assertEqual(b.update(5.2, False), [True])

    def test_chain_boundary_keeps_running(self):
        # active_program goes A -> B with busy true all along: no gap.
        b = mowing()
        b.update(0.0, True)
        b.on_program(program='B')
        self.assertEqual(b.update(1.0, True), [False])

    def test_temp_refusal(self):
        b = mowing()
        b.update(0.0, True)
        b.on_mower('TEMP', 0)
        self.assertEqual(run_blade(b, 1.0, 20.0, False), [], 'firmware drops it, do not count')
        self.assertEqual(b.state(), 'restoring')
        b.on_mower('READY', 0)                # cooled, no stop reason
        self.assertEqual(b.update(20.0, False), [True])

    def test_temp_then_stop_reason_gives_up(self):
        b = mowing()
        b.update(0.0, True)
        b.on_mower('TEMP', 0)
        run_blade(b, 1.0, 2.0, False)
        b.on_program(stop_reason='interrupted:temp')
        b.on_mower('READY', 0)
        self.assertEqual(run_blade(b, 2.0, 9.0, False), [])
        self.assertEqual(b.state(), 'ok')

    def test_soft_start_is_not_a_fault(self):
        b = mowing()
        b.update(0.0, True)
        b.on_mower('READY', 0)
        self.assertEqual(run_blade(b, 3.0, 8.9, False), [(3.0, True)])
        self.assertEqual(b.state(), 'restoring')

    def test_gives_up_loud_after_attempts(self):
        b = mowing()
        b.update(0.0, True)
        b.on_mower('READY', 0)
        cmds = run_blade(b, 3.0, 30.0, False)
        self.assertEqual([c for _, c in cmds], [True] * 3)
        self.assertEqual(b.state(), 'restore_failed')

    def test_lost_pose_stays_as_it_is(self):
        b = mowing()
        b.update(0.0, True)
        b.on_mower('READY', 0)
        self.assertEqual(b.update(1.0, None), [False])

    def test_unknown_program_stops_but_never_starts(self):
        b = g.Blade()
        b.on_mower('RUN', 3000)
        self.assertEqual(b.update(0.0, True), [False])
        b.on_mower('READY', 0)
        self.assertEqual(run_blade(b, 1.0, 9.0, False), [])

    def test_disabled_state(self):
        self.assertEqual(g.Blade().state(enabled=False), 'disabled')


def respawned(saved, busy=True, program='Zahrada', reason=''):
    b = g.Blade()
    b.restore(saved)
    b.on_program(busy=busy, program=program, stop_reason=reason)
    b.on_mower('READY', 0)
    return b


class Respawn(unittest.TestCase):
    OWED = {'program': 'Zahrada', 'stamp': 0.0}

    def test_owed_is_what_gets_saved(self):
        b = mowing()
        self.assertIsNone(b.owed())
        b.update(0.0, True)
        self.assertEqual(b.owed(), 'Zahrada')
        b.on_mower('READY', 0)
        b.update(3.0, False)
        b.on_mower('RUN', 2500)
        b.update(3.2, False)
        self.assertIsNone(b.owed())

    def test_same_program_never_gets_a_true(self):
        b = respawned(self.OWED)
        self.assertEqual(run_blade(b, 0.0, 30.0, False), [])
        self.assertEqual(b.state(), 'restore_failed')
        self.assertIsNone(b.owed())

    def test_still_inside_is_visible_and_never_a_true(self):
        b = respawned(self.OWED)
        self.assertEqual(run_blade(b, 0.0, 3.0, True), [])
        self.assertEqual(b.state(), 'restore_failed')
        self.assertEqual(run_blade(b, 3.0, 30.0, False), [])
        self.assertEqual(b.state(), 'restore_failed')

    def test_off_enforcement_survives_the_respawn(self):
        b = respawned(self.OWED)
        b.update(0.0, True)
        b.on_mower('RUN', 3000)               # recovery or the firmware started it
        self.assertEqual(b.update(1.0, True), [False])
        self.assertEqual(b.state(), 'stopped_by_rule')
        b.on_mower('READY', 0)                # seen after the restart: a live owe
        self.assertEqual(b.update(4.0, False), [True])

    def test_answer_after_a_respawn_still_works(self):
        b = respawned(self.OWED)
        b.update(0.0, True)
        self.assertTrue(g.blade_answer(b, '/mower_unit_smach', '/g', True, 0.3))

    def test_blade_running_owes_no_warning(self):
        b = respawned(self.OWED)
        b.on_mower('RUN', 3000)
        self.assertEqual(run_blade(b, 0.0, 3.0, False), [])
        self.assertEqual(b.state(), 'ok')

    def test_warning_state_ends_with_the_program(self):
        b = respawned(self.OWED)
        b.update(0.0, False)
        b.on_program(busy=False, program=' ')
        b.update(0.2, False)
        self.assertEqual(b.state(), 'ok')

    def test_warning_state_ends_when_the_mission_restarts_it(self):
        b = respawned(self.OWED)
        b.update(0.0, False)
        b.on_mower('RUN', 3000)
        b.update(0.2, False)
        self.assertEqual(b.state(), 'ok')

    def test_other_program_owes_nothing(self):
        b = respawned(self.OWED, program='Jiny')
        self.assertEqual(run_blade(b, 0.0, 9.0, False), [])
        self.assertIsNone(b.owed())

    def test_ended_program_owes_nothing(self):
        b = respawned(self.OWED, busy=False, program=' ')
        self.assertEqual(run_blade(b, 0.0, 9.0, False), [])
        self.assertEqual(b.state(), 'ok')

    def test_stop_reason_owes_nothing(self):
        b = respawned(self.OWED, reason='interrupted:battery')
        self.assertEqual(run_blade(b, 0.0, 9.0, False), [])

    def test_waits_for_the_program_inputs(self):
        b = g.Blade()
        b.restore(self.OWED)
        b.on_mower('READY', 0)
        self.assertEqual(b.update(0.0, False), [])
        self.assertEqual(b.owed(), 'Zahrada', 'still kept for the next respawn')
        b.on_program(busy=True, program='Zahrada', stop_reason='')
        self.assertEqual(b.update(1.0, False), [])
        self.assertEqual(b.state(), 'restore_failed')

    def test_no_mower_smach_gives_up(self):
        b = g.Blade()
        b.restore(self.OWED)
        self.assertEqual(run_blade(b, 0.0, 5.0, False), [])
        self.assertIsNone(b.owed())

    def test_bad_saved_value_is_nothing(self):
        b = g.Blade()
        b.restore({'stamp': 1.0})
        self.assertIsNone(b.owed())

    def test_program_change_clears_live(self):
        b = mowing()
        b.update(0.0, True)
        b.on_mower('READY', 0)
        b.on_program(program='B')
        self.assertIsNone(b.owed())
        self.assertEqual(run_blade(b, 1.0, 9.0, False), [])

    def test_program_end_clears_live(self):
        b = mowing()
        b.update(0.0, True)
        b.on_program(program=' ')
        self.assertIsNone(b.owed())

    def test_name_arriving_late_is_no_change(self):
        b = g.Blade()
        b.on_mower('RUN', 3000)
        b.update(0.0, True)                   # latched name not here yet
        b.on_program(busy=True, program='Zahrada', stop_reason='')
        self.assertEqual(b.owed(), 'Zahrada')


class Answer(unittest.TestCase):
    OWN = '/map_rules_guard'

    def inside(self):
        b = mowing()
        b.update(0.0, True)
        b.on_mower('READY', 0)
        return b

    def test_others_true_is_answered_at_once(self):
        b = self.inside()
        self.assertTrue(g.blade_answer(b, '/mower_unit_smach', self.OWN, True, 0.3))

    def test_twice_in_a_second_is_answered_twice(self):
        b = self.inside()
        self.assertTrue(g.blade_answer(b, '/mower_unit_smach', self.OWN, True, 0.3))
        self.assertTrue(g.blade_answer(b, '/mower_unit_smach', self.OWN, True, 0.4))

    def test_periodic_cap_still_holds(self):
        b = self.inside()
        g.blade_answer(b, '/mower_unit_smach', self.OWN, True, 0.3)
        self.assertEqual(b.update(0.5, True), [])
        self.assertEqual(b.update(1.3, True), [False])

    def test_own_messages_change_nothing(self):
        b = self.inside()
        for value in (True, False):
            self.assertFalse(g.blade_answer(b, self.OWN, self.OWN, value, 0.3))
        self.assertEqual(b.owed(), 'Zahrada')

    def test_a_false_is_never_answered(self):
        # Even with the callerid lost nothing can loop: only a True is answered.
        b = self.inside()
        self.assertFalse(g.blade_answer(b, None, self.OWN, False, 0.3))

    def test_outside_is_not_answered(self):
        b = mowing()
        b.update(0.0, False)
        self.assertFalse(g.blade_answer(b, '/mower_unit_smach', self.OWN, True, 0.3))

    def test_no_program_is_not_answered(self):
        b = g.Blade()
        b.on_program(busy=False, program=' ', stop_reason='')
        b.inside = True
        self.assertFalse(g.blade_answer(b, '/ui', self.OWN, True, 0.3))


class Geometry(unittest.TestCase):
    def test_footprint_matches_the_config(self):
        path = os.path.join(HERE, '..', '..', '..', '..', '..', 'vitulus_navi', 'config',
                            'costmap_common_params.yaml')
        if os.path.exists(path):
            self.assertEqual(g.load_footprint(path, print), g.FOOTPRINT)

    def test_stop_speed(self):
        self.assertEqual(g.stop_speed(0.0, 0.25, 0.4), 0.0)
        v = g.stop_speed(0.5, 0.25, 0.4)
        self.assertTrue(math.isclose(v * 0.4 + v * v / 0.5, 0.5))


if __name__ == '__main__':
    unittest.main()
