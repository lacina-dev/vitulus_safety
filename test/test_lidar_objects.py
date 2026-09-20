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
    # ohlášení až po appear_min_s (1 s) — objekt v chatu má mít i snímek
    tail = res[len(frames) + 12:]
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


def test_h_far_flicker_gives_no_event():
    """Mihotání dálky: objekt ve 7 m, který se ukáže na 0,5 s, nesmí dát
    událost; blízký člověk sledovaný > 1 s ano."""
    rng = random.Random(21)
    frames = warmup(rng, 30.0, 0.01)
    blik = [(lambda t, rng=rng: put_object(garage(0.01, rng), 7.0, 2.0, 0.4)) for _ in range(5)]
    det = ObjectDetector({'bg_min_samples': 8})
    events = []
    for i, f in enumerate(frames + blik + warmup(rng, 2.0, 0.01)):
        det.process(f(i * DT), ANGLE_MIN, ANGLE_INC, i * DT); events.extend(det.take_events())
    assert not events, 'mihotání v 7 m nesmí dát událost: %s' % events
    det2 = ObjectDetector({'bg_min_samples': 8}); ev2 = []
    near = [(lambda t, rng=rng: put_object(garage(0.01, rng), 2.0, 0.2, 0.45)) for _ in range(15)]
    for i, f in enumerate(frames + near):
        det2.process(f(i * DT), ANGLE_MIN, ANGLE_INC, i * DT); ev2.extend(det2.take_events())
    assert [e['type'] for e in ev2] == ['appeared'], 'člověk ve 2 m po 1,5 s: %s' % ev2
    print('H) mihotání 0,5 s v 7 m → 0 událostí; člověk ve 2 m po 1,5 s → appeared  OK')


def test_i_two_legs_are_one_person_and_track_survives_gaps():
    """Dvě nohy (2 shluky 0,2 m, 0,3 m od sebe) = jeden objekt; stopa přežije
    krátký výpadek (max_missed ~1 s) — 2. 9. 22:03 se člověk rozpadl na dva
    tracky a „odcházel" každou sekundu."""
    rng = random.Random(7)
    frames = warmup(rng, 30.0, 0.01)
    walk = []
    for k in range(40):
        def f(t, k=k, rng=rng):
            x = 3.0 - 0.04 * k
            r = put_object(garage(0.01, rng), x, 0.15, 0.20)
            r = put_object(r, x, -0.15, 0.20)
            if 18 <= k <= 22:            # 0,5 s výpadek (zákryt)
                return garage(0.01, rng)
            return r
        walk.append(f)
    det = ObjectDetector({'bg_min_samples': 8}); ev = []; res = []
    for i, f in enumerate(frames + walk):
        res.append(det.process(f(i * DT), ANGLE_MIN, ANGLE_INC, i * DT)); ev.extend(det.take_events())
    tail = res[len(frames) + 6:]
    assert max(len(r) for r in tail) == 1, 'nohy nesmí být dva objekty (max %d)' % max(len(r) for r in tail)
    ids = {o['id'] for r in tail for o in r}
    assert len(ids) == 1, 'stopa má přežít výpadek 0,5 s, ale ID: %s' % ids
    assert [e['type'] for e in ev] == ['appeared'], 'jen jedno objevení: %s' % [e['type'] for e in ev]
    print('I) dvě nohy = 1 objekt (%s), jedno ID přes 0,5s výpadek, 1× appeared  OK' % tail[-1][0]['cls'])


def test_j_depth_height_classifies_person_dog_cat():
    """Z hloubkového obrazu (640×480, K jako D435) se změří výška objektu
    pod směrem z lidaru: člověk 1,7 m → large, pes 0,5 m → medium, kočka
    0,25 m → small; objekt mimo záběr → None."""
    from lidar_objects import depth_object_extent, class_from_height
    K = [616.8, 0, 326.2, 0, 616.7, 235.1, 0, 0, 1]
    fx, cx, fy, cy = K[0], K[2], K[4], K[5]

    def scene(height_m, z=2.0, bearing=10.0):
        d = np.full((480, 640), 4.0, dtype=np.float32)          # zeď ve 4 m
        u = int(cx - fx * math.tan(math.radians(bearing)))
        hw = int(fx * 0.25 / z)                                   # 0,5 m široký
        rows = int(fy * height_m / z)
        v1 = int(cy + fy * 0.6 / z)                               # spodek 0,6 m pod osou (v záběru)
        v0 = max(0, v1 - rows)
        d[v0:v1, u - hw:u + hw] = z
        return d
    out = {}
    # člověk ve 2 m se do svislého záběru (~42°) nevejde → měří se ve 3 m;
    # blízký oříznutý objekt hlásí clipped_top a třída se z výšky NEurčuje.
    ext = depth_object_extent(scene(1.7, z=2.0), K, 10.0, 2.0 / math.cos(math.radians(10.0)))
    assert ext and ext['clipped_top'], 'blízký člověk je oříznutý: %s' % ext
    for name, h, z in (('člověk', 1.7, 3.0), ('pes', 0.5, 2.0), ('kočka', 0.25, 2.0)):
        ext = depth_object_extent(scene(h, z=z), K, 10.0, z / math.cos(math.radians(10.0)))
        assert ext and abs(ext['height_m'] - h) < 0.08, '%s: %s' % (name, ext)
        out[name] = (ext['height_m'], class_from_height(ext['height_m']))
    assert out['člověk'][1] == 'large' and out['pes'][1] == 'medium' and out['kočka'][1] == 'small'
    assert depth_object_extent(scene(1.7), K, 80.0, 2.0) is None, 'mimo záběr'
    print('J) hloubka: člověk %.2f m→%s, pes %.2f m→%s, kočka %.2f m→%s; mimo záběr → None  OK'
          % (out['člověk'][0], out['člověk'][1], out['pes'][0], out['pes'][1], out['kočka'][0], out['kočka'][1]))


def _run_events(frames, cfg=None):
    det = ObjectDetector(cfg or {'bg_min_samples': 8})
    ev, res = [], []
    for i, f in enumerate(frames):
        res.append(det.process(f(i * DT), ANGLE_MIN, ANGLE_INC, i * DT))
        ev.extend(det.take_events())
    return det, res, ev


def edge_flicker(rng, beams, near, far=WALL, episode_p=0.06, length=(2, 8)):
    """HRANA, na které 2–4 paprsky náhodně přeskakují mezi vzdálenější
    plochou (`far`) a bližší (`near`) — přesně to, co dělalo (−1,3; 3,75) a
    (1,7; 4,75) v garáži: epizody 0,2–0,8 s, pár za minutu, pořád na témž
    místě.  Vrací továrnu na rámce se stavem."""
    state = {'left': 0}

    def f(t):
        r = garage(0.01, rng)
        if state['left'] > 0:
            state['left'] -= 1
            for b in beams:
                # smíšené pixely: každý paprsek jinde mezi bližší a
                # vzdálenější plochou → shluk má radiální „rozměr" 0,2–0,3 m
                # jako skutečné záznamy (size 0,16–0,35 m), ne 5 cm
                r[b] = near - rng.random() * 0.25 + 0.05
        elif rng.random() < episode_p:
            state['left'] = rng.randint(*length)
        return r
    return f


def test_k_recurring_edge_flicker_gives_nothing_but_person_is_seen():
    """Opakované mihotání hrany: 3 paprsky přeskakují 4,0 m ↔ 3,75 m po
    dobu 3 minut → 0 událostí (staré jádro jich dávalo desítky).  Potom
    člověk 0,45 m projde PŘES tytéž paprsky → ohlášen."""
    rng = random.Random(23)
    beams = [519, 520, 521, 522, 523]        # ~45° vlevo vpředu, 5 paprsků
    flick = edge_flicker(rng, beams, near=3.75)
    frames = [flick for _ in range(int(180.0 / DT))]
    det, res, ev = _run_events(frames)
    assert not ev, 'mihotání hrany dalo události: %s' % [(e['type'], e.get('evidence')) for e in ev]
    assert all(len(r) == 0 for r in res), 'mihotání hrany dalo objekt'
    nz = det.noise()
    assert nz.get('flaky_beams', 0) >= 2, 'mihotavé paprsky se měly poznat: %s' % nz
    # člověk jde napříč přes hranu (y roste), 0,8 m/s ve 3,3 m
    walk = []
    for k in range(40):
        def g(t, k=k):
            r = flick(t)
            return put_object(r, 2.4, 1.6 + 0.08 * k, 0.45)
        walk.append(g)
    det2, res2, ev2 = _run_events(frames + walk)
    kinds = [e['type'] for e in ev2]
    assert kinds[:1] == ['appeared'], 'člověk přes mihotavou hranu musí být ohlášen: %s' % kinds
    seen_frames = [i for i, r in enumerate(res2) if r]
    assert seen_frames and seen_frames[0] - len(frames) <= 20, \
        'ohlášení do 2 s od příchodu (bylo po %d rámcích)' % (seen_frames[0] - len(frames))
    print('K) mihotání hrany 3 min → 0 událostí, %d mihotavých paprsků, %d duchů zahozeno; '
          'člověk přes tutéž hranu → appeared po %.1f s  OK'
          % (nz.get('flaky_beams', 0), nz.get('ghosts_dropped', 0),
             (seen_frames[0] - len(frames)) * DT))


def test_l_random_jitter_cluster_is_not_an_object():
    """Roztřesený shluk: 3 paprsky, které se každý rámec objeví JINDE
    (náhodný směr 3–6 m, ale pořád před zdí) — pohyb bez směru není zvíře."""
    rng = random.Random(29)
    frames = warmup(rng, 30.0, 0.01)
    jit = []
    for k in range(60):
        def f(t, k=k):
            r = garage(0.01, rng)
            b = 300 + rng.randint(-6, 6)         # ±2,5° kolem jednoho místa
            for j in range(3):
                r[b + j] = 3.0 + rng.random() * 0.9
            return r
        jit.append(f)
    det, res, ev = _run_events(frames + jit)
    assert not ev, 'roztřesený shluk dal události: %s' % [(e['type'], e.get('evidence')) for e in ev]
    print('L) roztřesený shluk 6 s → 0 událostí, 0 objektů  OK')


def test_m_far_person_crossing_is_reported_with_evidence():
    """Člověk 0,5 m jde napříč v 6,5 m rychlostí 1,2 m/s (staré jádro
    v > 5 m nehlásilo nic) → appeared s důkazy (série, pevné paprsky, přímost)."""
    rng = random.Random(31)
    wall = 8.0
    def far_garage(t):
        r = garage(0.01, rng)
        return r * (wall / WALL)
    frames = [far_garage for _ in range(int(30.0 / DT))]
    walk = []
    for k in range(40):
        def f(t, k=k):
            return put_object(far_garage(t), 6.5, -2.0 + 0.12 * k, 0.5)
        walk.append(f)
    det, res, ev = _run_events(frames + walk)
    ap = [e for e in ev if e['type'] == 'appeared']
    assert ap, 'člověk v 6,5 m nebyl ohlášen'
    evd = ap[0]['evidence']
    assert evd['streak'] >= 6 and evd['solid'] >= 0.9, 'důkazy: %s' % evd
    assert evd['straight'] is None or evd['straight'] > 0.8, 'jde rovně: %s' % evd
    assert 6.0 < ap[0]['range_m'] < 7.2, 'vzdálenost %.2f' % ap[0]['range_m']
    print('M) člověk napříč v 6,5 m → appeared r=%.1f m, důkazy %s  OK' % (ap[0]['range_m'], evd))


def test_n_still_cat_near_wall_is_reported_edge_only_for_wall_range():
    """Kočka 0,25 m, která si sedne 0,8 m PŘED zeď a nehýbe se → ohlášena
    (není to odštěpek zdi, je před ní); tři paprsky NA vzdálenosti zdi
    vedle jejího lomu (odštěpek) → nic."""
    rng = random.Random(37)
    frames = warmup(rng, 30.0, 0.01)
    cat = [(lambda t: put_object(garage(0.01, rng), 3.2, 0.4, 0.25)) for _ in range(30)]
    det, res, ev = _run_events(frames + cat)
    assert [e['type'] for e in ev] == ['appeared'], 'kočka před zdí: %s' % ev
    assert ev[0]['cls'] == 'small', 'třída %s' % ev[0]['cls']
    # odštěpek: 3 paprsky, které se ukážou 0,25 m před zdí přesně tam, kde
    # sousední paprsky zeď vidí (hrana) — 3 s v kuse
    def chip(t):
        r = garage(0.01, rng)
        r[600:603] = WALL - 0.25
        return r
    det2, res2, ev2 = _run_events(frames + [chip for _ in range(30)])
    assert not ev2, 'odštěpek stěny nesmí být objekt: %s' % ev2
    print('N) kočka 0,25 m před zdí → appeared small; odštěpek stěny 0,25 m → nic  OK')


if __name__ == '__main__':
    fails = 0
    for fn in (test_a_empty_garage_noise, test_b_moving_object,
               test_c_static_object_ages_into_background,
               test_d_single_beam_flicker, test_e_motion_tracking_and_events,
               test_f_still_object_is_still_not_moving,
               test_g_sectors_and_noise_adaptive_threshold,
               test_h_far_flicker_gives_no_event,
               test_i_two_legs_are_one_person_and_track_survives_gaps,
               test_j_depth_height_classifies_person_dog_cat,
               test_k_recurring_edge_flicker_gives_nothing_but_person_is_seen,
               test_l_random_jitter_cluster_is_not_an_object,
               test_m_far_person_crossing_is_reported_with_evidence,
               test_n_still_cat_near_wall_is_reported_edge_only_for_wall_range):
        try:
            fn()
        except AssertionError as e:
            fails += 1
            print('FAIL %s: %s' % (fn.__name__, e))
    print('---')
    print('FAILED %d' % fails if fails else 'ALL OK')
    sys.exit(1 if fails else 0)
