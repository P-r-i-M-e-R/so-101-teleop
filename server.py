"""SO-101 teleop recorder: local web UI (stdlib HTTP server, no extra framework)."""
import argparse
import glob
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import applog
import cameras as cam_mod
from applog import log
from engine import JOINTS, Engine, MAX_SESSION_SECONDS
from recorder import Recorder

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / 'config.json'
DEFAULTS = {
    'cameras': [
        {'name': 'front', 'device': 'realsense'},
        {'name': 'wrist', 'device': '/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0'},
    ],
    'width': 640, 'height': 480, 'camera_fps': 30,
    'fps': 30,
    'dataset_dir': str(Path.home() / 'lerobot_datasets'),
    'dataset_name': 'so101_teleop',
    'task': 'pick up the object and place it in the box',
    'session_seconds': 300,
    'speed': 400,
    'extras': True,
    'auto_release_s': 0,
    'save_partial_on_error': False,
    'append_existing': False,
}


def load_config():
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        cfg.update(json.loads(CONFIG_PATH.read_text()))
    return cfg


class App:
    def __init__(self):
        self.cfg = load_config()
        self.engine = Engine(fps=self.cfg['fps'])
        self.engine.auto_release = self.cfg['auto_release_s']
        self.cameras = {}
        self.recorder = Recorder(self.cameras)
        self.engine.on_sample = self.recorder.push
        self.lock = threading.Lock()
        self.ep_lock = threading.RLock()     # serialises episode start/stop from HTTP threads and the supervisor
        self.start_cameras()
        threading.Thread(target=self._supervise, daemon=True).start()

    def save_cfg(self):
        CONFIG_PATH.write_text(json.dumps(self.cfg, indent=2))

    def start_cameras(self):
        for c in self.cameras.values():
            c.stop()
        self.cameras.clear()
        for spec in self.cfg['cameras']:
            c = cam_mod.make_camera(spec['name'], spec['device'], self.cfg['width'], self.cfg['height'],
                               self.cfg['camera_fps'])
            self.cameras[spec['name']] = c
            c.start()

    def finish_episode(self, save):
        """End the episode and log which warnings/collisions happened during it."""
        with self.ep_lock:
            return self._finish_episode(save)

    def _finish_episode(self, save):
        r = self.recorder
        t0 = getattr(self, '_ep_start_ts', time.time())
        e = self.engine
        reason = e.message if e.state == 'error' else None
        frames, elapsed = r.frames, round(time.monotonic() - r.started, 1)
        if not save and e.state == 'error' and self.cfg.get('save_partial_on_error') and frames >= 30:
            save = True                      # keep the truncated episode (flagged in the failure log)
        status = r.end(save)
        if reason or status.startswith('discarded') or r.error:
            rec = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'status': status, 'engine_state': e.state,
                   'reason': reason or r.error, 'frames_recorded': frames, 'seconds': elapsed,
                   'saved_partial': status.startswith('saved')}
            log.error('EPISODE NOT CLEAN: %s', rec)
            try:
                with open(Path(r.root) / 'episode_failures.jsonl', 'a') as f:
                    f.write(json.dumps(rec) + '\n')
            except OSError:
                pass
            r.last_saved = f'EPISODE {"SAVED TRUNCATED" if rec["saved_partial"] else "DISCARDED"} after {elapsed} s: {rec["reason"]}'
        if status.startswith('saved'):
            hits = [a for a in self.engine.alerts if a['ts'] >= t0 and a['level'] in ('warn', 'crit')]
            idx = r.num_episodes - 1
            with open(Path(r.root) / 'alerts_per_episode.jsonl', 'a') as f:
                f.write(json.dumps({'episode_index': idx, 'n_warn': sum(a['level'] == 'warn' for a in hits),
                                    'n_crit': sum(a['level'] == 'crit' for a in hits),
                                    'frame_stats': dict(r.frame_stats), 'alerts': hits}) + '\n')
            bad = {k: x for k, x in r.frame_stats.items() if x['duplicate'] or x['skipped']}
            if bad:
                r.last_saved = f'{status} - camera frame issues: {bad}'
            elif hits:
                r.last_saved = f'{status} - {len(hits)} warning(s)/alert(s), see alerts_per_episode.jsonl'
        return status

    def health_gate(self):
        """Refuse to start an episode that is likely to be broken by a known, avoidable problem."""
        e, r = self.engine, self.recorder
        snap = e.snapshot()
        problems = []
        bad_cams = [c.name for c in self.cameras.values() if not c.status()['ok']]
        bad_cams += [f'{c.name}_depth' for c in self.cameras.values() if c.has_depth and c.depth_fps < 20]
        if bad_cams:
            problems.append(f'camera(s) not delivering frames: {bad_cams}')
        crit = [f'{j}:{k}' for j, a in (snap.get('active_alerts') or {}).items() for k, lv in a.items() if lv == 'crit']
        if crit:
            problems.append(f'critical alerts active: {crit}')
        hot = {j: h['temp'] for j, h in (snap.get('follower') or {}).items() if h['temp'] >= HOT_BLOCK}
        if hot:
            problems.append(f'follower too hot to risk an episode (stops at 45 C): {hot} - let it cool')
        retries = {k: v for k, v in (snap.get('comm') or {}).items() if k in ('leader', 'follower') and sum(v.values()) >= 4}
        if retries:
            problems.append(f'serial bus unstable (retries in last 10 s: {retries}) - check cables/power')
        if r.status()['backlog'] > 60:
            problems.append('encoder still catching up with the previous episode - wait a few seconds')
        if r.status()['disk_free_gb'] < 5:
            problems.append('less than 5 GB free disk')
        if r.preview_errors:
            log.warning('depth preview problems earlier: %s', r.preview_errors)
        if problems:
            raise RuntimeError('not starting episode: ' + '; '.join(problems))

    def _supervise(self):
        """Close an episode as soon as teleop ends for any reason."""
        seen = {}
        while True:
            time.sleep(0.1)
            for c in list(self.cameras.values()):
                st = c.status()
                key = (st['ok'], st['error'])
                if seen.get(c.name) != key:
                    seen[c.name] = key
                    (log.error if st['error'] else log.info)('camera %s: ok=%s error=%s fps=%s', c.name, st['ok'], st['error'], st['fps'])
            r, e = self.recorder, self.engine
            if r.recording and not r.saving and (e.state != 'teleop' or e.aligning):
                threading.Thread(target=self.finish_episode, args=(e.state == 'monitor',), daemon=True).start()

    # ------------------------------------------------------------------ API actions
    def status(self):
        e = self.engine.snapshot()
        out = {
            'engine': {k: e.get(k) for k in ('state', 'message', 'aligning', 'gripper_contact', 'loop_hz',
                                              'teleop_left', 'release_in', 'holding', 'retries', 'read_ms')},
            'alerts': e.get('alerts', []), 'alert_total': e.get('alert_total', 0),
            'active_alerts': e.get('active_alerts', {}),
            'joints': JOINTS,
            'leader': e.get('leader_units'), 'follower': e.get('follower_units'),
            'goal': e.get('goal_units'),
            'health': {n: {k: v[k] for k in ('temp', 'voltage', 'load', 'current', 'speed')}
                       for n, v in (e.get('follower') or {}).items()},
            'leader_health': {n: {k: v[k] for k in ('temp', 'voltage', 'current', 'speed')}
                              for n, v in (e.get('leader') or {}).items()},
            'recorder': self.recorder.status(),
            'cameras': [c.status() for c in self.cameras.values()] + [
                {'name': f'{c.name}_depth', 'ok': c.has_depth and c.depth_fps > 1, 'error': c.error,
                 'fps': round(c.depth_fps, 1), 'size': c.depth_size, 'depth_view': True}
                for c in self.cameras.values() if c.has_depth],
            'cfg': {k: self.cfg[k] for k in ('fps', 'dataset_dir', 'dataset_name', 'task', 'session_seconds',
                                              'speed', 'extras', 'auto_release_s', 'save_partial_on_error', 'append_existing')},
            'max_session': MAX_SESSION_SECONDS,
        }
        return out

    def act(self, name, body):
        cfg, e, r = self.cfg, self.engine, self.recorder
        if name == 'config':
            for k in ('dataset_dir', 'dataset_name', 'task'):
                if k in body:
                    cfg[k] = str(body[k]).strip()
            for k, lo, hi in (('fps', 5, 60), ('session_seconds', 10, MAX_SESSION_SECONDS), ('speed', 65, 600), ('auto_release_s', 0, 120)):
                if k in body:
                    cfg[k] = max(lo, min(hi, int(body[k])))
            if 'append_existing' in body:
                cfg['append_existing'] = bool(body['append_existing'])
            if 'save_partial_on_error' in body:
                cfg['save_partial_on_error'] = bool(body['save_partial_on_error'])
            if 'extras' in body:
                cfg['extras'] = bool(body['extras'])
            e.auto_release = cfg['auto_release_s']
            self.save_cfg()
            return 'saved'
        if name == 'connect':
            return e.connect(resume_holding=bool(body.get('resume_holding')))
        if name == 'disconnect':
            if r.recording:
                self.finish_episode(True)
            return e.disconnect() or 'ok'
        if name == 'teleop_start':
            if e.state != 'monitor':
                raise RuntimeError('connect first')
            if not body.get('confirmed'):
                raise RuntimeError('safety confirmation missing')
            e.fps = cfg['fps']
            return e.start_teleop(cfg['session_seconds'], cfg['speed'])
        if name == 'teleop_stop':
            if r.recording:
                self.finish_episode(True)
            return e.stop_teleop()
        if name == 'cancel_release':
            return e.command('cancel_release')
        if name == 'disable_torque':
            if not body.get('confirmed'):
                raise RuntimeError('support the arm and confirm')
            return e.command('disable_torque')
        if name == 'open_dataset':
            if r.recording:
                raise RuntimeError('episode in progress')
            if not cfg['task']:
                raise RuntimeError('enter a task description')
            base = Path(cfg['dataset_dir']).expanduser()
            root = base / cfg['dataset_name']
            exists = root.exists() and any(root.iterdir())
            if exists and not cfg.get('append_existing'):
                # never silently add to old data: start a NEW, empty dataset next to it
                root = base / f"{cfg['dataset_name']}_{time.strftime('%Y%m%d_%H%M%S')}"
                log.info('dataset %s already exists -> creating new dataset %s', cfg['dataset_name'], root.name)
            n = r.open(root, cfg['task'], cfg['fps'], cfg['extras'])
            log.info('dataset opened: %s (%d episodes)', root, n)
            return (f'NEW empty dataset: {root}' if n == 0 else
                    f'APPENDING to existing dataset {root} ({n} episodes already in it)')
        if name == 'record_start':
            if e.state != 'teleop' or e.aligning:
                raise RuntimeError('start teleop and wait for POSE ALIGNED first')
            if r.ds is None:
                self.act('open_dataset', {})
            r.task = cfg['task']
            self.health_gate()
            with self.ep_lock:
                if r.recording or r.saving:
                    raise RuntimeError('previous episode is still saving' if r.saving else 'already recording')
                self._ep_start_ts = time.time()
                r.begin()
            return 'recording'
        if name == 'record_stop':
            return self.finish_episode(bool(body.get('save', True)))
        if name == 'cameras_restart':
            if r.recording:
                raise RuntimeError('episode in progress')
            self.start_cameras()
            return 'cameras restarted'
        if name == 'cameras_scan':
            return self.scan_cameras()
        if name == 'cameras_set':
            if r.recording:
                raise RuntimeError('episode in progress')
            if r.ds is not None:
                raise RuntimeError('close the dataset (restart server) before changing cameras')
            cfg['cameras'] = [{'name': c['name'], 'device': c['device']} for c in body['cameras']]
            self.save_cfg()
            self.start_cameras()
            return 'ok'
        raise RuntimeError(f'unknown action {name}')

    def scan_cameras(self):
        out = []
        for p in sorted(glob.glob('/dev/v4l/by-id/*')):
            real = os.path.realpath(p)
            if any(c.device == p for c in self.cameras.values()):
                out.append({'device': p, 'in_use': True})
                continue
            ok, info = cam_mod.probe(p, self.cfg['width'], self.cfg['height'], 1.0)
            out.append({'device': p, 'ok': ok, 'info': info, 'node': real})
        return out

    def close(self):
        try:
            if self.recorder.recording:
                self.finish_episode(True)
            self.recorder.close()
        finally:
            try:
                self.engine.disconnect()
            except Exception:   # noqa: BLE001
                pass


APP = None
HOT_BLOCK = 43   # deg C: do not START an episode if a follower motor is already this hot


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json'):
        if not isinstance(body, (bytes, str)):
            body = json.dumps(body)
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            return self._send(200, (HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        if self.path.startswith('/api/log'):
            return self._send(200, applog.tail(300), 'text/plain; charset=utf-8')
        if self.path == '/api/status':
            return self._send(200, APP.status())
        if self.path.startswith('/stream/'):
            sname = self.path[8:].split('.')[0]
            depth = sname.endswith('_depth')
            cam = APP.cameras.get(sname[:-6] if depth else sname)
            if cam is None or (depth and not cam.has_depth):
                return self._send(404, {'error': 'no such camera'})
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            last = -1
            try:
                while True:
                    count = cam.latest_depth()[2] if depth else cam.latest()[2]
                    jpg = (cam.depth_jpeg() if depth else cam.jpeg()) if count != last else None
                    if jpg is None:
                        time.sleep(0.015)
                        continue
                    last = count
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(jpg))
                    self.wfile.write(jpg + b'\r\n')
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                return
        self._send(404, {'error': 'not found'})

    def do_POST(self):
        if not self.path.startswith('/api/'):
            return self._send(404, {'error': 'not found'})
        n = int(self.headers.get('Content-Length') or 0)
        try:
            body = json.loads(self.rfile.read(n) or b'{}')
            result = APP.act(self.path[5:], body)
            self._send(200, {'ok': True, 'result': result})
        except Exception as e:      # noqa: BLE001
            log.error('API %s failed: %s: %s', self.path, type(e).__name__, e,
                      exc_info=None if isinstance(e, RuntimeError) else e)
            self._send(200, {'ok': False, 'error': f'{type(e).__name__}: {e}'})


def main():
    global APP
    applog.setup()
    import logging
    import signal
    logging.getLogger('lerobot.utils.import_utils').setLevel(logging.ERROR)   # torchcodec->pyav fallback notice
    sys.setswitchinterval(0.0005)   # shorter GIL hand-off: less jitter for camera/robot threads while encoding
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8765)
    args = ap.parse_args()
    APP = App()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f'SO-101 recorder: http://{args.host}:{args.port}   (Ctrl+C to quit; follower is left HOLDING)', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        APP.close()


if __name__ == '__main__':
    main()
