#!/usr/bin/env python3
"""lidar_objects — detekce větších pohyblivých objektů (kočka / pes / člověk) z 2D lidaru.

Proč vzniklo: agentní recept porovnával dva po sobě jdoucí scany a hlásil
"motion" pokaždé, když se pár bodíků zavlnilo (117 hlášení za den v prázdné
garáži). Tenhle uzel řeší to samé, ale jako vlastnost ROBOTA:

  1) statické pozadí = klouzavý MEDIÁN vzdálenosti per paprsek přes ~60 s
     → zeď, dok, zaparkované auto se do pozadí "usadí" a přestanou být vidět;
  2) popředí = paprsky výrazně BLÍŽ než pozadí (absolutní + relativní mez);
  3) shlukování popředí v kartézských souřadnicích (jednoduchý průchod po
     úhlu, práh mezery roste s dosahem — body jsou dál od sebe);
  4) filtr velikosti 0,15–1,2 m a minimálního počtu bodů podle vzdálenosti;
  5) sledování shluků mezi scany (nejbližší soused, ID, rychlost) a potvrzení
     až po N po sobě jdoucích scanech → vlnící se jednotlivý bod nepřežije;
  6) hrubá klasifikace podle šířky a rychlosti: small / medium / large / unknown.

Výstup:
  /safety/lidar_objects          std_msgs/String (JSON) @ 5 Hz
  /safety/lidar_objects_markers  visualization_msgs/MarkerArray

Uzel je READ-ONLY vůči robotovi (jen odebírá /scan a odometrii, nic neřídí)
a nemá žádnou vazbu na runtime agenta.
"""

import json
import math
import os
import threading

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


# --------------------------------------------------------------------------
# Čistá detekční logika (bez rospy) — testovatelná syntetickými scany.
# --------------------------------------------------------------------------

DEFAULTS = {
    'still_mps': 0.15,          # pod tim je objekt v klidu (sum zakmitu shluku ~0,1 m/s)
    # pozadí
    'bg_window_s': 60.0,        # délka okna klouzavého mediánu
    'bg_sample_dt': 0.5,        # jak často se scan ukládá do pozadí
    'bg_min_samples': 10,       # dokud jich není tolik, nedetekuje se nic
    # popředí
    'fg_margin_m': 0.20,        # musí být aspoň o tolik blíž než pozadí
    'fg_margin_frac': 0.06,     # ... nebo o tolik procent dosahu (bere se max)
    'fg_max_range_m': 10.0,     # dál než tohle neřešíme (šum, řídké body)
    'fg_min_range_m': 0.15,
    # shlukování
    'cluster_gap_m': 0.18,      # základní mezera mezi sousedními body shluku
    'cluster_gap_k': 3.0,       # + k × (r × Δθ) — řídnutí bodů s dosahem
    # filtr kandidátů
    'min_size_m': 0.15,         # kočka
    'max_size_m': 1.20,         # člověk / velký pes
    'min_points_floor': 2,      # absolutní minimum (daleké objekty)
    'min_points_near': 4,       # blíž než near_range_m chceme aspoň tolik bodů
    'near_range_m': 3.0,
    # sledování
    'assoc_dist_m': 0.60,       # okno pro přiřazení shluku k trace
    'confirm_frames': 3,        # trvání: kolikrát po sobě musí být viděn
    'max_missed': 3,            # kolik scanů smí chybět, než trace zanikne
    'speed_alpha': 0.4,         # vyhlazení rychlosti (EMA)
    'max_speed_mps': 4.0,       # rychlejší "objekt" je skok asociace, ne zvíře
    # klasifikace
    'cls_small_max_m': 0.30,
    'cls_medium_max_m': 0.60,
    # jízda
    'motion_speed_mps': 0.05,   # nad tuhle rychlost robota je pozadí neplatné
}


class Track(object):
    """Jeden sledovaný shluk mezi scany."""

    _next_id = 1

    def __init__(self, cx, cy, size, stamp):
        self.id = Track._next_id
        Track._next_id += 1
        self.x = cx
        self.y = cy
        self.size = size
        self.size_min = size
        self.size_max = size
        self.speed = 0.0
        self.first_seen = stamp
        self.last_seen = stamp
        self.hits = 1
        self.missed = 0
        # SLEDOVÁNÍ POHYBU (majitel 2. 9.: „navrhni sledování objektů a jejich
        # pohybu"): stopa posledních poloh, směr pohybu, radiální rychlost
        # (záporná = přibližuje se k robotu) a z toho stav.  Vyhlazuje se
        # EMA stejně jako rychlost — jeden roztřesený scan nesmí otočit směr.
        self.path = [(round(cx, 2), round(cy, 2), round(stamp, 2))]
        self.vx = 0.0
        self.vy = 0.0
        self.radial = 0.0            # m/s vůči robotu: <0 blíž, >0 dál
        self.announced = False       # už ohlášen jako „objevil se"?

    def update(self, cx, cy, size, stamp, alpha):
        dt = max(1e-3, stamp - self.last_seen)
        d = math.hypot(cx - self.x, cy - self.y)
        self.speed = (1.0 - alpha) * self.speed + alpha * (d / dt)
        vx, vy = (cx - self.x) / dt, (cy - self.y) / dt
        self.vx = (1.0 - alpha) * self.vx + alpha * vx
        self.vy = (1.0 - alpha) * self.vy + alpha * vy
        r_old, r_new = math.hypot(self.x, self.y), math.hypot(cx, cy)
        self.radial = (1.0 - alpha) * self.radial + alpha * ((r_new - r_old) / dt)
        self.path.append((round(cx, 2), round(cy, 2), round(stamp, 2)))
        if len(self.path) > 20:
            del self.path[0]
        self.x = cx
        self.y = cy
        self.size = 0.5 * (self.size + size)
        self.size_min = min(self.size_min, size)
        self.size_max = max(self.size_max, size)
        self.last_seen = stamp
        self.hits += 1
        self.missed = 0

    @property
    def age_s(self):
        return self.last_seen - self.first_seen

    def motion(self, still_mps=0.15):
        """'still' | 'approaching' | 'leaving' | 'passing' — podle radiální
        složky rychlosti; pod `still_mps` je objekt v klidu (šum rychlosti
        ze zákmitů shluku je typicky do 0,1 m/s)."""
        if self.speed < still_mps:
            return 'still'
        if self.radial < -0.5 * self.speed:
            return 'approaching'
        if self.radial > 0.5 * self.speed:
            return 'leaving'
        return 'passing'

    @property
    def heading_deg(self):
        """Směr pohybu v rámu robota (0° = vpřed), None v klidu."""
        if math.hypot(self.vx, self.vy) < 0.05:
            return None
        return round(math.degrees(math.atan2(self.vy, self.vx)), 1)


class ObjectDetector(object):
    """Pozadí + popředí + shluky + tracky. Nezná ROS, jen čísla."""

    def __init__(self, cfg=None):
        self.cfg = dict(DEFAULTS)
        if cfg:
            self.cfg.update({k: v for k, v in cfg.items() if k in DEFAULTS})
        self.tracks = []
        self.events = []             # appeared/left od posledního vyzvednutí
        self._bg_buf = None          # (N, beams) ring buffer
        self._bg_n = 0               # kolik vzorků je platných
        self._bg_i = 0               # kam se zapíše další
        self._bg_median = None
        self._bg_dirty = True
        self._last_bg_sample = None
        self._angles = None
        self._n_beams = 0
        self.background_ready = False

    # -- pozadí -----------------------------------------------------------
    def take_events(self):
        """Vyzvednout a vyprázdnit události (appeared/left) od minula."""
        ev, self.events = self.events, []
        return ev

    def reset(self):
        """Zapomene pozadí i tracky (jízda robota, restart lidaru)."""
        self._bg_buf = None
        self._bg_n = 0
        self._bg_i = 0
        self._bg_median = None
        self._last_bg_sample = None
        self.background_ready = False
        self.tracks = []
        self.events = []

    def _ensure_geometry(self, angle_min, angle_inc, n):
        if self._angles is None or self._n_beams != n:
            self._angles = angle_min + angle_inc * np.arange(n, dtype=np.float64)
            self._n_beams = n
            self._bg_buf = None
            self._bg_n = 0
            self._bg_i = 0
            self._bg_median = None
            self.background_ready = False

    def _push_background(self, ranges, stamp):
        cfg = self.cfg
        if self._last_bg_sample is not None and \
                (stamp - self._last_bg_sample) < cfg['bg_sample_dt']:
            return
        self._last_bg_sample = stamp
        depth = max(3, int(round(cfg['bg_window_s'] / max(1e-3, cfg['bg_sample_dt']))))
        if self._bg_buf is None or self._bg_buf.shape != (depth, self._n_beams):
            self._bg_buf = np.full((depth, self._n_beams), np.nan, dtype=np.float32)
            self._bg_n = 0
            self._bg_i = 0
        self._bg_buf[self._bg_i, :] = ranges
        self._bg_i = (self._bg_i + 1) % depth
        self._bg_n = min(depth, self._bg_n + 1)
        self._bg_dirty = True
        self.background_ready = self._bg_n >= cfg['bg_min_samples']

    def _background(self):
        if self._bg_dirty and self._bg_n > 0:
            with np.errstate(invalid='ignore'):
                # nanmedian: paprsek, který je občas inf (nic nezasáhl), se
                # počítá jen z platných měření; když nikdy nic nevrátil, zůstane NaN
                self._bg_median = np.nanmedian(self._bg_buf[:self._bg_n, :], axis=0)
            self._bg_dirty = False
        return self._bg_median

    # -- jeden scan -------------------------------------------------------
    def process(self, ranges, angle_min, angle_inc, stamp, range_max=None):
        """Zpracuje scan, vrátí seznam POTVRZENÝCH objektů (dict).

        ranges: list/np.array vzdáleností (inf/nan = nic)
        """
        cfg = self.cfg
        r = np.asarray(ranges, dtype=np.float64)
        # inf i nan → NaN, ať s tím počítá jeden kód
        r = np.where(np.isfinite(r), r, np.nan)
        self._ensure_geometry(angle_min, angle_inc, r.shape[0])
        self._push_background(r, stamp)

        clusters = []
        if self.background_ready:
            bg = self._background()
            clusters = self._clusters(r, bg, angle_inc)
        self._track(clusters, stamp)
        return self.confirmed()

    def _foreground_mask(self, r, bg):
        cfg = self.cfg
        margin = np.maximum(cfg['fg_margin_m'], cfg['fg_margin_frac'] * bg)
        valid = np.isfinite(r) & (r > cfg['fg_min_range_m']) & (r < cfg['fg_max_range_m'])
        # paprsek bez pozadí (vždycky inf, teď něco vidí) je taky popředí
        no_bg = ~np.isfinite(bg)
        closer = np.isfinite(bg) & (r < (bg - margin))
        return valid & (closer | no_bg)

    def _clusters(self, r, bg, angle_inc):
        cfg = self.cfg
        mask = self._foreground_mask(r, bg)
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            return []
        ang = self._angles[idx]
        rr = r[idx]
        xs = rr * np.cos(ang)
        ys = rr * np.sin(ang)

        out = []
        start = 0
        for k in range(1, idx.size + 1):
            split = True
            if k < idx.size:
                gap = math.hypot(xs[k] - xs[k - 1], ys[k] - ys[k - 1])
                # práh roste s dosahem — ve 8 m jsou sousední body 6 cm od sebe
                thr = cfg['cluster_gap_m'] + cfg['cluster_gap_k'] * rr[k] * angle_inc
                # díra v indexech (mezi body je nezasažený paprsek) shluk taky dělí,
                # pokud je větší než pár paprsků
                split = (gap > thr) or ((idx[k] - idx[k - 1]) > 4)
            if not split:
                continue
            sl = slice(start, k)
            start = k
            n = k - sl.start
            cx = float(np.mean(xs[sl]))
            cy = float(np.mean(ys[sl]))
            size = float(math.hypot(xs[sl][-1] - xs[sl][0], ys[sl][-1] - ys[sl][0]))
            rng = float(np.mean(rr[sl]))
            if not (cfg['min_size_m'] <= max(size, 0.0) <= cfg['max_size_m']):
                # jednobodové "vlnění" (size==0) tady padá spolu se zdí (size>1.2 m)
                continue
            if n < self._min_points(rng, angle_inc):
                continue
            out.append({'x': cx, 'y': cy, 'size': size, 'range': rng, 'n': n})
        return out

    def _min_points(self, rng, angle_inc):
        cfg = self.cfg
        # kolik paprsků by mělo trefit nejmenší hledaný objekt v této vzdálenosti
        span = max(1e-6, rng * angle_inc)
        expect = cfg['min_size_m'] / span
        need = int(round(0.5 * expect))
        need = max(cfg['min_points_floor'], min(need, 12))
        if rng < cfg['near_range_m']:
            need = max(need, cfg['min_points_near'])
        return need

    # -- sledování --------------------------------------------------------
    def _track(self, clusters, stamp):
        cfg = self.cfg
        free = list(clusters)
        for t in self.tracks:
            best, bestd = None, cfg['assoc_dist_m']
            for c in free:
                d = math.hypot(c['x'] - t.x, c['y'] - t.y)
                if d < bestd:
                    best, bestd = c, d
            if best is not None:
                free.remove(best)
                t.update(best['x'], best['y'], best['size'], stamp, cfg['speed_alpha'])
            else:
                t.missed += 1
        # UDÁLOSTI: potvrzený objekt, který zmizel, se ohlásí jako „odešel"
        # (s poslední polohou a délkou sledování) — majitel se ptá „kdo tu
        # byl", ne jen „kdo tu je".
        for t in self.tracks:
            if t.missed > cfg['max_missed'] and t.announced:
                self.events.append({'type': 'left', 'id': t.id,
                                    'cls': self.classify(t),
                                    'x': round(t.x, 2), 'y': round(t.y, 2),
                                    'seen_s': round(t.age_s, 1),
                                    'stamp': round(stamp, 2)})
        self.tracks = [t for t in self.tracks if t.missed <= cfg['max_missed']]
        for c in free:
            self.tracks.append(Track(c['x'], c['y'], c['size'], stamp))

    def classify(self, t):
        cfg = self.cfg
        w = t.size
        if t.speed > cfg['max_speed_mps']:
            return 'unknown'
        if (t.size_max - t.size_min) > 0.6:
            # rozměr mezi scany lítá — nevíme, co to je
            return 'unknown'
        if w < cfg['cls_small_max_m']:
            # rychlý drobek bývá spíš pes v běhu než kočka
            return 'medium' if t.speed > 1.0 else 'small'
        if w < cfg['cls_medium_max_m']:
            return 'medium'
        if w <= cfg['max_size_m']:
            return 'large'
        return 'unknown'

    def confirmed(self):
        cfg = self.cfg
        out = []
        for t in self.tracks:
            if t.hits < cfg['confirm_frames'] or t.missed > 0:
                continue
            if not t.announced:
                t.announced = True
                self.events.append({'type': 'appeared', 'id': t.id,
                                    'cls': self.classify(t),
                                    'x': round(t.x, 2), 'y': round(t.y, 2),
                                    'stamp': round(t.last_seen, 2)})
            out.append({
                'id': t.id,
                'cls': self.classify(t),
                'x': round(t.x, 3),
                'y': round(t.y, 3),
                'size': round(t.size, 3),
                'speed': round(t.speed, 2),
                'age_s': round(t.age_s, 1),
                'bearing_deg': round(math.degrees(math.atan2(t.y, t.x)), 1),
                'range_m': round(math.hypot(t.x, t.y), 2),
                # pohyb
                'motion': t.motion(cfg.get('still_mps', 0.15)),
                'heading_deg': t.heading_deg,
                'radial_mps': round(t.radial, 2),
                'path': t.path[-10:],
            })
        out.sort(key=lambda o: o['range_m'])
        return out


# --------------------------------------------------------------------------
# ROS obal
# --------------------------------------------------------------------------

class LidarObjectsNode(object):

    def __init__(self):
        cfg = {}
        for key, dflt in DEFAULTS.items():
            cfg[key] = rospy.get_param('~' + key, dflt)
        self.scan_topic = rospy.get_param('~scan_topic', '/scan')
        self.odom_topic = rospy.get_param('~odom_topic', '/mobile_base_controller/odom')
        self.publish_rate = float(rospy.get_param('~publish_rate', 5.0))
        self.target_frame = rospy.get_param('~target_frame', 'base_link')
        self.publish_markers = bool(rospy.get_param('~publish_markers', True))

        self.det = ObjectDetector(cfg)
        self.lock = threading.Lock()
        self.objects = []
        self.events = []             # appeared/left, vyzvedne cb_publish
        self.moving = False
        self.last_scan_stamp = None
        self.scan_frame = None
        self._tf = (0.0, 0.0, 0.0)     # base_link ← scan_frame (x, y, yaw)
        self._tf_ok = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub = rospy.Publisher('/safety/lidar_objects', String, queue_size=1)
        self.pub_markers = rospy.Publisher('/safety/lidar_objects_markers',
                                           MarkerArray, queue_size=1)
        rospy.Subscriber(self.scan_topic, LaserScan, self.cb_scan, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.cb_odom, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / max(0.5, self.publish_rate)), self.cb_publish)
        rospy.Timer(rospy.Duration(10.0), self.cb_master)
        rospy.loginfo('lidar_objects: scan=%s odom=%s bg_window=%.0fs confirm=%d',
                      self.scan_topic, self.odom_topic,
                      cfg['bg_window_s'], cfg['confirm_frames'])

    # -- vstupy -----------------------------------------------------------
    def cb_odom(self, msg):
        v = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        w = abs(msg.twist.twist.angular.z)
        moving = (v > self.det.cfg['motion_speed_mps']) or (w > 0.10)
        with self.lock:
            if moving and not self.moving:
                # Za jízdy je "statické pozadí" nesmysl — radši nic než 117 duchů.
                self.det.reset()
                self.objects = []
            self.moving = moving

    def cb_scan(self, msg):
        with self.lock:
            if self.moving:
                self.last_scan_stamp = msg.header.stamp.to_sec()
                return
            if self.scan_frame != msg.header.frame_id:
                self.scan_frame = msg.header.frame_id
                self._tf_ok = False
            self._lookup_tf()
            stamp = msg.header.stamp.to_sec()
            objs = self.det.process(msg.ranges, msg.angle_min,
                                    msg.angle_increment, stamp,
                                    range_max=msg.range_max)
            self.objects = [self._to_base_link(o) for o in objs]
            self.events.extend(self.det.take_events())
            if len(self.events) > 50:
                del self.events[:-50]
            self.last_scan_stamp = stamp

    def _lookup_tf(self):
        if self._tf_ok or not self.scan_frame:
            return
        if self.scan_frame == self.target_frame:
            self._tf, self._tf_ok = (0.0, 0.0, 0.0), True
            return
        try:
            tr = self.tf_buffer.lookup_transform(self.target_frame, self.scan_frame,
                                                 rospy.Time(0), rospy.Duration(0.2))
            q = tr.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self._tf = (tr.transform.translation.x, tr.transform.translation.y, yaw)
            self._tf_ok = True
        except Exception:
            pass    # zkusí se znovu při dalším scanu, mezitím jedeme v rámu lidaru

    def _to_base_link(self, o):
        tx, ty, yaw = self._tf
        c, s = math.cos(yaw), math.sin(yaw)
        x = c * o['x'] - s * o['y'] + tx
        y = s * o['x'] + c * o['y'] + ty
        o = dict(o)
        o['x'] = round(x, 3)
        o['y'] = round(y, 3)
        o['bearing_deg'] = round(math.degrees(math.atan2(y, x)), 1)
        o['range_m'] = round(math.hypot(x, y), 2)
        return o

    # -- výstup -----------------------------------------------------------
    def cb_master(self, _evt):
        """Hlídání masteru.  rospy se po restartu ROS masteru znovu
        NEZAREGISTRUJE: uzel běžel 12,5 h, `Publishers: None` (2. 9. 14:00,
        stack se mezitím restartoval).  Tři neúspěšné dotazy → konec s kódem
        3, systemd (Restart=always) uzel zvedne a ten se přihlásí znovu."""
        try:
            rospy.get_master().getSystemState()
            self._master_fail = 0
        except Exception:                                   # noqa: BLE001
            self._master_fail = getattr(self, '_master_fail', 0) + 1
            if self._master_fail >= 3:
                rospy.logerr('lidar_objects: master nedostupný 3×, končím (systemd restartuje)')
                os._exit(3)

    def cb_publish(self, _evt):
        with self.lock:
            objs = list(self.objects)
            events, self.events = self.events, []
            moving = self.moving
            ready = self.det.background_ready
            frame = self.target_frame if self._tf_ok else (self.scan_frame or '')
        now = rospy.Time.now()
        payload = {
            'stamp': round(now.to_sec(), 3),
            'count': len(objs),
            'objects': objs,
            'events': events,        # appeared / left od minulé zprávy
            'frame_id': frame,
            'moving': bool(moving),
            'background_ready': bool(ready),
        }
        self.pub.publish(String(data=json.dumps(payload, separators=(',', ':'))))
        if self.publish_markers:
            self.pub_markers.publish(self._markers(objs, frame, now))

    def _markers(self, objs, frame, now):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        colors = {'small': (0.2, 0.8, 1.0), 'medium': (1.0, 0.7, 0.1),
                  'large': (1.0, 0.2, 0.2), 'unknown': (0.6, 0.6, 0.6)}
        for i, o in enumerate(objs):
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = now
            m.ns = 'lidar_objects'
            m.id = i * 2
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position = Point(o['x'], o['y'], 0.25)
            m.pose.orientation.w = 1.0
            d = max(0.2, o['size'])
            m.scale.x = m.scale.y = d
            m.scale.z = 0.5
            r, g, b = colors.get(o['cls'], colors['unknown'])
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.6
            m.lifetime = rospy.Duration(1.0)
            arr.markers.append(m)

            t = Marker()
            t.header = m.header
            t.ns = 'lidar_objects_text'
            t.id = i * 2 + 1
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position = Point(o['x'], o['y'], 0.7)
            t.pose.orientation.w = 1.0
            t.scale.z = 0.25
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = '%s #%d %.1fm %.1fm/s' % (o['cls'], o['id'],
                                               o['range_m'], o['speed'])
            t.lifetime = rospy.Duration(1.0)
            arr.markers.append(t)
        return arr


def main():
    rospy.init_node('lidar_objects')
    LidarObjectsNode()
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
