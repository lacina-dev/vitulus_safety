#!/usr/bin/env python3
"""Přehrání skutečných scanů z rosbagu detektorem lidar_objects — offline.

    python3 test/replay_bag.py garage.bag                  # aktuální detektor
    python3 test/replay_bag.py garage.bag --node /cesta/k/stare/verzi.py
    python3 test/replay_bag.py garage.bag --flaky          # kde jsou mihotavé paprsky

Bez běžícího ROS (jen knihovna `rosbag` z /opt/ros).  Vypíše, kolik stop
vzniklo, kolik bylo ohlášeno, události (s důkazy) a kolik „duchů" detektor
zahodil.  Slouží k porovnání dvou verzí detektoru na TÝCHŽ datech — prázdná
garáž = každé ohlášení je falešný poplach.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time

sys.path.insert(0, '/opt/ros/noetic/lib/python3/dist-packages')
import warnings                                      # noqa: E402
import numpy as np                                   # noqa: E402

# paprsky, které nikdy nic netrefí (inf), dávají v nanmedian „All-NaN slice" —
# je to očekávané a v uzlu neškodné, tady by to jen zaplavilo výpis
warnings.filterwarnings('ignore', category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NODE = os.path.join(HERE, '..', 'nodes', 'lidar_objects.py')


def load_detector(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--node', default=DEFAULT_NODE, help='cesta k lidar_objects.py')
    ap.add_argument('--scan', default='/scan')
    ap.add_argument('--every', type=int, default=1, help='brát každý N-tý scan (uzel v klidu bere 10.)')
    ap.add_argument('--flaky', action='store_true', help='vypsat mihotavé paprsky na konci')
    ap.add_argument('--cfg', default='{}', help='JSON s přepisem parametrů')
    ap.add_argument('--events-json', default=None, help='kam uložit události (JSONL)')
    args = ap.parse_args()

    import rosbag
    mod = load_detector(args.node, 'lo_' + str(abs(hash(args.node)) % 10000))
    det = mod.ObjectDetector(json.loads(args.cfg))

    n = 0
    t0 = time.time()
    tracks_seen = set()
    announced = set()
    events = []
    per_frame_objs = 0
    first_stamp = last_stamp = None
    with rosbag.Bag(args.bag) as bag:
        for i, (_topic, msg, _t) in enumerate(bag.read_messages(topics=[args.scan])):
            if i % max(1, args.every):
                continue
            stamp = msg.header.stamp.to_sec()
            first_stamp = first_stamp if first_stamp is not None else stamp
            last_stamp = stamp
            try:
                objs = det.process(msg.ranges, msg.angle_min, msg.angle_increment, stamp,
                                   range_max=msg.range_max,
                                   intensities=msg.intensities or None)
            except TypeError:           # starší verze detektoru bez intenzit
                objs = det.process(msg.ranges, msg.angle_min, msg.angle_increment, stamp,
                                   range_max=msg.range_max)
            n += 1
            for t in det.tracks:
                tracks_seen.add(t.id)
                if t.announced:
                    announced.add(t.id)
            per_frame_objs += len(objs)
            for e in det.take_events():
                e['_when'] = time.strftime('%H:%M:%S', time.localtime(stamp))
                events.append(e)
    dur = (last_stamp - first_stamp) if first_stamp is not None else 0.0
    print('scanů %d (%.0f s záznamu), zpracováno za %.1f s' % (n, dur, time.time() - t0))
    print('stop založeno %d, ohlášeno %d, objektů-scanů %d' % (len(tracks_seen), len(announced), per_frame_objs))
    print('šum:', json.dumps(det.noise(), ensure_ascii=False))
    for e in events:
        ev = e.get('evidence') or {}
        print('  %s %-8s #%-3s %-7s %-6s r=%.2f  %s' % (
            e['_when'], e['type'], e['id'], e.get('cls'), e.get('sector'),
            float(e.get('range_m') or math.hypot(e.get('x', 0), e.get('y', 0))),
            ('seen %.1fs' % e['seen_s']) if e['type'] == 'left' else json.dumps(ev)))
    if args.events_json:
        with open(args.events_json, 'w') as fh:
            for e in events:
                fh.write(json.dumps(e, ensure_ascii=False) + '\n')
    if args.flaky and getattr(det, '_flaky', None) is not None:
        fl = np.nonzero(det._flaky)[0]
        print('mihotavé paprsky: %d' % fl.size)
        for i in fl[:40]:
            ang = math.degrees(det._angles[i])
            print('  paprsek %3d  %6.1f°  medián %.2f m  bližší mód %.2f m  přeskoků %d' % (
                i, ang, det._bg_median[i], det._alt_bg[i], det._flips[i]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
