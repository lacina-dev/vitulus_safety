#!/usr/bin/env python3
"""Jednotkový test detektoru lidar_objects na SYNTETICKÝCH scanech.

Nepotřebuje běžící ROS ani robota:
    python3 src/vitulus/vitulus_safety/test/test_lidar_objects.py

Scénáře:
  A) prázdná garáž + šum ±3 cm ("vlnící se bodíky")      → 0 objektů
  B) stejná garáž + objekt 0,4 m ve 3 m, který se posouvá → 1 objekt medium/large
  C) objekt, který 90 s stojí (auto/zeď)                  → zestárne do pozadí
  D) jeden zákmit jediného paprsku                        → 0 objektů (bez trvání)
"""

import importlib.util
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NODE = os.path.join(HERE, '..', 'nodes', 'lidar_objects.py')

_spec = importlib.util.spec_from_file_location('lidar_objects', NODE)
_mod = importlib.util.module_from_spec(_spec)
sys.modules['lidar_objects'] = _mod
_spec.loader.exec_module(_mod)
ObjectDetector = _mod.ObjectDetector

N = 860
ANGLE_MIN = -math.pi
ANGLE_INC = 2.0 * math.pi / N
ANGLES = ANGLE_MIN + ANGLE_INC * np.arange(N)
WALL = 4.0          # kruhová "garáž" o poloměru 4 m
DT = 0.1            # 10 Hz


def garage(noise_m=0.0, rng=None):
    r = np.full(N, WALL)
    if noise_m:
        r = r + np.array([(rng.random() * 2.0 - 1.0) * noise_m for _ in range(N)])
    return r


def put_object(ranges, x, y, width):
    """Vloží do scanu válec o průměru `width` na (x, y) — zastíní paprsky."""
    out = ranges.copy()
    d = math.hypot(x, y)
    if d < 1e-3:
        return out
    bearing = math.atan2(y, x)
    half = math.atan2(width / 2.0, d)
    da = np.abs(np.arctan2(np.sin(ANGLES - bearing), np.cos(ANGLES - bearing)))
    hit = da <= half
    # povrch válce (zjednodušeně koule) — vzdálenost mírně klesá ke středu
    chord = np.sqrt(np.maximum(0.0, (width / 2.0) ** 2 - (d * np.sin(da)) ** 2))
    out[hit] = np.minimum(out[hit], d - chord[hit])
    return out


def run(frames, cfg=None, seed=1):
    """frames: seznam funkcí (t) -> ranges. Vrací seznam výsledků per scan."""
    det = ObjectDetector(cfg or {'bg_min_samples': 8})
    res = []
    for i, f in enumerate(frames):
        t = i * DT
        res.append(det.process(f(t), ANGLE_MIN, ANGLE_INC, t))
    return det, res


def warmup(rng, seconds=20.0, noise=0.01):
    return [(lambda t, rng=rng: garage(noise, rng)) for _ in range(int(seconds / DT))]


def test_a_empty_garage_noise():
    rng = random.Random(7)
    frames = warmup(rng, 40.0, 0.03)        # ±3 cm vlnění po celou dobu
    det, res = run(frames)
    counts = [len(r) for r in res]
    assert max(counts) == 0, 'šum ±3 cm dal %d objektů' % max(counts)
    print('A) prázdná garáž + šum ±3 cm: max %d objektů  OK' % max(counts))


def test_b_moving_object():
    rng = random.Random(11)
    frames = warmup(rng, 30.0, 0.01)
    moves = []
    for k in range(40):                      # 4 s pohybu napříč zorným polem
        def f(t, k=k, rng=rng):
            y = -1.0 + 0.05 * k              # 0,5 m/s bokem
            return put_object(garage(0.01, rng), 3.0, y, 0.40)
        moves.append(f)
    det, res = run(frames + moves)
    tail = res[len(frames) + 5:]
    counts = [len(r) for r in tail]
    assert min(counts) >= 1, 'pohyblivý objekt zmizel (min %d)' % min(counts)
    assert max(counts) == 1, 'objekt se rozpadl na %d kusů' % max(counts)
    last = tail[-1][0]
    assert last['cls'] in ('medium', 'large'), 'třída %s' % last['cls']
    assert 2.5 < last['range_m'] < 3.5, 'vzdálenost %.2f' % last['range_m']
    assert last['age_s'] > 0.3, 'trvání %.2f s' % last['age_s']
    print('B) objekt 0,4 m ve 3 m v pohybu: 1 objekt, cls=%s, r=%.2f m, '
          'v=%.2f m/s, age=%.1f s  OK'
          % (last['cls'], last['range_m'], last['speed'], last['age_s']))


def test_c_static_object_ages_into_background():
    rng = random.Random(13)
    frames = warmup(rng, 20.0, 0.01)
    static = [(lambda t, rng=rng: put_object(garage(0.01, rng), 3.0, 0.0, 0.40))
              for _ in range(int(120.0 / DT))]
    det, res = run(frames + static)
    early = len(res[len(frames) + 10])
    late = len(res[-1])
    assert early >= 1, 'stojící objekt nebyl vidět ani na začátku'
    assert late == 0, 'po 120 s stále hlásí %d objektů' % late
    print('C) stojící objekt: hned po příchodu %d, po 120 s %d  OK' % (early, late))


def test_d_single_beam_flicker():
    rng = random.Random(17)
    frames = warmup(rng, 30.0, 0.01)
    flick = []
    for k in range(30):
        def f(t, k=k, rng=rng):
            r = garage(0.01, rng)
            if k % 3 == 0:                   # jediný paprsek, ob scan
                r[430] = 2.0
            return r
        flick.append(f)
    det, res = run(frames + flick)
    counts = [len(r) for r in res[len(frames):]]
    assert max(counts) == 0, 'zákmit jednoho paprsku dal %d objektů' % max(counts)
    print('D) zákmit jednoho paprsku: %d objektů  OK' % max(counts))


def test_e_motion_tracking_and_events():
    """Pohyb: objekt jde k robotu → 'approaching' + záporná radiální rychlost,
    stopa polohy; po zmizení událost 'left', na začátku 'appeared'."""
    rng = random.Random(5)
    frames = warmup(rng, 30.0, 0.01)
    moves = []
    for k in range(40):                      # ze 3,5 m na 1,5 m, 0,5 m/s
        def f(t, k=k, rng=rng):
            x = 3.5 - 0.05 * k
            return put_object(garage(0.01, rng), x, 0.0, 0.40)
        moves.append(f)
    gone = warmup(rng, 2.0, 0.01)
    det = ObjectDetector({'bg_min_samples': 8})
    events = []
    res = []
    for i, f in enumerate(frames + moves + gone):
        t = i * DT
        res.append(det.process(f(t), ANGLE_MIN, ANGLE_INC, t))
        events.extend(det.take_events())
    mid = res[len(frames) + 20][0]
    assert mid['motion'] == 'approaching', 'stav %s' % mid['motion']
    assert mid['radial_mps'] < -0.2, 'radiální %.2f' % mid['radial_mps']
    assert mid['heading_deg'] is not None and abs(abs(mid['heading_deg']) - 180) < 30, \
        'směr pohybu k robotu (~180°), je %s' % mid['heading_deg']
    assert len(mid['path']) >= 5, 'stopa %d' % len(mid['path'])
    kinds = [e['type'] for e in events]
    assert 'appeared' in kinds and 'left' in kinds, 'události %s' % kinds
    left = [e for e in events if e['type'] == 'left'][-1]
    assert left['seen_s'] > 2.0, 'odešel po %.1f s' % left['seen_s']
    assert len(res[-1]) == 0, 'po odchodu nic'
    print('E) přiblížení 3,5→1,5 m: motion=%s, radial=%.2f m/s, heading=%s°, '
          'stopa %d bodů; události %s  OK'
          % (mid['motion'], mid['radial_mps'], mid['heading_deg'],
             len(mid['path']), sorted(set(kinds))))


def test_f_still_object_is_still_not_moving():
    """Stojící (ale ještě nezestárlý) objekt hlásí 'still', ne šum rychlosti."""
    rng = random.Random(9)
    frames = warmup(rng, 30.0, 0.01)
    stand = [(lambda t, rng=rng: put_object(garage(0.01, rng), 2.5, 0.5, 0.5))
             for _ in range(30)]
    det, res = run(frames + stand)
    o = res[-1][0]
    assert o['motion'] == 'still', 'stav %s (v=%.2f)' % (o['motion'], o['speed'])
    print('F) stojící objekt: motion=still, v=%.2f m/s  OK' % o['speed'])


def test_g_sectors_and_noise_adaptive_threshold():
    """Orientace (x = předek): sektory; a klidový rozptyl zvedne práh —
    vlnění ±8 cm na jednom místě NEdá popředí, člověk dá."""
    from lidar_objects import sector_of
    assert sector_of(10) == 'front' and sector_of(-30) == 'front'
    assert sector_of(90) == 'left' and sector_of(-90) == 'right'
    assert sector_of(170) == 'back' and sector_of(-170) == 'back'
    rng = random.Random(3)
    # garáž, kde jeden úsek stěny (30 paprsků) v klidu „dýchá" ±8 cm
    def calm(t, rng=rng):
        r = garage(0.01, rng)
        r[400:430] = WALL + (rng.random() * 2 - 1) * 0.08
        return r
    frames = [calm for _ in range(300)]
    det, res = run(frames)
    assert all(len(x) == 0 for x in res[-50:]), 'dýchající stěna nesmí být objekt'
    nz = det.noise()
    assert nz.get('mad_median_m', 1) < 0.03, 'klid: medián MAD malý (%s)' % nz
    person = [(lambda t, rng=rng: put_object(calm(t), 2.0, 0.3, 0.45)) for _ in range(20)]
    det2, res2 = run(frames + person)
    assert len(res2[-1]) == 1 and res2[-1][0]['sector'] == 'front', 'člověk vpředu se pozná: %s' % res2[-1]
    print('G) sektory OK; dýchající stěna ±8 cm → 0 objektů, klid MAD %.3f m; člověk vpředu → 1 objekt front  OK' % nz['mad_median_m'])


if __name__ == '__main__':
    fails = 0
    for fn in (test_a_empty_garage_noise, test_b_moving_object,
               test_c_static_object_ages_into_background,
               test_d_single_beam_flicker, test_e_motion_tracking_and_events,
               test_f_still_object_is_still_not_moving,
               test_g_sectors_and_noise_adaptive_threshold):
        try:
            fn()
        except AssertionError as e:
            fails += 1
            print('FAIL %s: %s' % (fn.__name__, e))
    print('---')
    print('FAILED %d' % fails if fails else 'ALL OK')
    sys.exit(1 if fails else 0)
