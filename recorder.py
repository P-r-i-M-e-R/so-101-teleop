"""Writes teleop episodes straight into a LeRobotDataset (v3 format, videos encoded live by lerobot).

After every saved episode the dataset is finalized and re-opened, so the files on disk are always a
complete, loadable dataset (no metadata stuck in lerobot's buffer if the process dies)."""
import json
import queue
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from applog import log
from engine import BUNDLE, FOLLOWER_CAL, JOINTS, LEADER_CAL

MIN_FRAMES = 10
MAX_BACKLOG = 900          # frames waiting for the encoder before we abort the episode (~30 s @30fps)

TELEMETRY = [   # (feature key, source arm, motor field)
    ('observation.follower.velocity', 'follower', 'speed'),
    ('observation.follower.load', 'follower', 'load'),
    ('observation.follower.current', 'follower', 'current'),
    ('observation.follower.temperature', 'follower', 'temp'),
    ('observation.follower.voltage', 'follower', 'voltage'),
    ('observation.follower.moving', 'follower', 'moving'),
    ('observation.follower.status', 'follower', 'status'),
    ('observation.leader.velocity', 'leader', 'speed'),
    ('observation.leader.load', 'leader', 'load'),
    ('observation.leader.current', 'leader', 'current'),
    ('observation.leader.temperature', 'leader', 'temp'),
    ('observation.leader.voltage', 'leader', 'voltage'),
    ('observation.leader.status', 'leader', 'status'),
]


class PreviewWriter:
    """Writes a normal, playable H.264 colour-mapped depth video (0-6.3 m) next to the dataset.
    The dataset itself keeps the raw 12-bit depth; this file is only for looking at it."""

    def __init__(self, path, w, h, fps):
        import av
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.out = av.open(str(self.path), 'w')
        self.stream = self.out.add_stream('libx264', rate=fps)
        self.stream.width, self.stream.height, self.stream.pix_fmt = w, h, 'yuv420p'
        self.stream.options = {'preset': 'ultrafast', 'crf': '22'}

    def write(self, depth_mm):
        import av
        vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_mm, alpha=0.04), cv2.COLORMAP_JET)
        vis[depth_mm == 0] = 0
        frame = av.VideoFrame.from_ndarray(vis, format='bgr24').reformat(format='yuv420p')
        for p in self.stream.encode(frame):
            self.out.mux(p)

    def close(self):
        try:
            for p in self.stream.encode():
                self.out.mux(p)
        finally:
            self.out.close()


def vec(names, fn):
    return np.array([fn(n) for n in names], dtype=np.float32)


class Recorder:
    def __init__(self, cameras):
        self.cameras = cameras            # name -> Camera
        self.ds = None
        self.root = None
        self.task = ''
        self.fps = 30
        self.extras = True
        self.recording = False
        self.frames = 0
        self.started = 0.0
        self.error = None
        self.last_saved = None
        self.frame_stats = {}             # per episode: duplicate / skipped camera frames
        self._last_count = {}
        self._q = queue.Queue()
        self._worker = None
        self._n_eps = 0
        self.preview_errors = []
        self._ep_id = 0                   # every episode gets an id; the worker drops frames of any other id
        self.saving = False
        self.depth_preset = 'ultrafast'

    # ---------------------------------------------------------------- dataset lifecycle
    def stream_names(self):
        """Every video-like stream, in a fixed order: colour streams then native depth streams."""
        names = list(self.cameras)
        names += [f'{n}_depth' for n, c in self.cameras.items() if c.has_depth]
        return names

    def features(self, extras=True):
        names = [f'{j}.pos' for j in JOINTS]
        f = {
            'observation.state': {'dtype': 'float32', 'shape': (6,), 'names': names},
            'action': {'dtype': 'float32', 'shape': (6,), 'names': names},
        }
        for cam in self.cameras.values():
            f[f'observation.images.{cam.name}'] = {
                'dtype': 'video', 'shape': (cam.height, cam.width, 3), 'names': ['height', 'width', 'channels']}
            if cam.has_depth:    # native, unaligned depth, uint16 -> stored/returned in millimetres
                f[f'observation.images.{cam.name}_depth'] = {
                    'dtype': 'video', 'shape': (cam.depth_size[1], cam.depth_size[0], 1),
                    'names': ['height', 'width', 'channels'], 'info': {'is_depth_map': True}}
            if cam.has_imu:
                f[f'observation.{cam.name}.imu'] = {
                    'dtype': 'float32', 'shape': (6,), 'names': ['ax', 'ay', 'az', 'gx', 'gy', 'gz']}
        if extras:
            raw = [f'{j}.raw' for j in JOINTS]
            f['observation.follower.raw_position'] = {'dtype': 'float32', 'shape': (6,), 'names': raw}
            f['action.leader.raw_position'] = {'dtype': 'float32', 'shape': (6,), 'names': raw}
            f['action.goal_sent'] = {'dtype': 'float32', 'shape': (6,), 'names': names}
            for key, _, _ in TELEMETRY:
                f[key] = {'dtype': 'float32', 'shape': (6,), 'names': JOINTS}
            streams = self.stream_names()
            f['timing.unix_time'] = {'dtype': 'float64', 'shape': (1,), 'names': None}
            f['timing.read_ms'] = {'dtype': 'float32', 'shape': (1,), 'names': None}
            f['timing.camera_age_ms'] = {'dtype': 'float32', 'shape': (len(streams),), 'names': streams}
            f['timing.camera_frame_id'] = {'dtype': 'float64', 'shape': (len(streams),), 'names': streams}
        return f

    def _open_ds(self, create):
        from lerobot.configs.video import DepthEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        repo_id = f'local/{self.root.name}'
        # Depth default is lossless 12-bit HEVC at a slow preset, which starves the camera threads of CPU;
        # a fast preset keeps the same (lossless) coding with much less load.
        enc = dict(streaming_encoding=True, depth_encoder=DepthEncoderConfig(preset=self.depth_preset))
        if create:
            return LeRobotDataset.create(repo_id, self.fps, self.features(self.extras), root=self.root,
                                         robot_type='so101_follower', use_videos=True,
                                         metadata_buffer_size=1, **enc)
        ds = LeRobotDataset.resume(repo_id, root=self.root, **enc)
        ds.meta._metadata_buffer_size = 1
        return ds

    def open(self, root, task, fps, extras=True):
        if self.recording:
            raise RuntimeError('stop the current episode first')
        self.close()
        # depth/IMU features are decided by what the cameras deliver: wait for the RealSense instead of
        # silently creating an RGB-only dataset
        end = time.time() + 30
        rs = [c for c in self.cameras.values() if c.device.startswith('realsense')]
        while time.time() < end and any(not c.has_depth for c in rs):
            time.sleep(0.3)
        if any(not c.has_depth for c in rs):
            raise RuntimeError('RealSense depth stream is not running (see camera status) - fix the camera first')
        self.root, self.task, self.fps, self.extras = Path(root).expanduser(), task, int(fps), extras
        if (self.root / 'meta' / 'info.json').exists():
            self.ds = self._open_ds(False)
            missing = set(self.features(extras)) - set(self.ds.features)
            if missing or self.ds.fps != self.fps:
                self.ds.finalize()
                self.ds = None
                raise RuntimeError(f'existing dataset is not compatible (fps/features differ: {sorted(missing)[:3]}). '
                                   f'Use a new dataset name.')
        else:
            self.ds = self._open_ds(True)
        self._n_eps = self.ds.num_episodes
        self._write_calibration()
        self._worker = threading.Thread(target=self._work, name='recorder', daemon=True)
        self._worker.start()
        self.error = None
        return self.num_episodes

    def _write_calibration(self):
        """Camera intrinsics/extrinsics/depth scale + both arms' calibration, next to the dataset."""
        info = {'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                'cameras': {n: {'device': c.device, 'color_size': [c.width, c.height], 'fps': c.fps,
                                'calibration': c.calib} for n, c in self.cameras.items()},
                'joint_units': 'observation.state/action: degrees about centre (2048 counts), gripper 0-100 % of range',
                'depth': 'observation.images.<cam>_depth: native (unaligned) depth, returned in millimetres'}
        (self.root / 'camera_info.json').write_text(json.dumps(info, indent=2))
        (self.root / 'robot_calibration.json').write_text(json.dumps(
            {'leader': LEADER_CAL, 'follower': FOLLOWER_CAL,
             'teleop_config': json.loads((BUNDLE / 'teleop_config.json').read_text())}, indent=2))

    @property
    def num_episodes(self):
        return self._n_eps

    def close(self):
        if self._worker:
            self._q.put(None)
            self._worker.join(timeout=120)
            self._worker = None
        if self.ds is not None:
            self.ds.finalize()
            self.ds = None
        self._n_eps = 0

    # ---------------------------------------------------------------- episode control
    def begin(self):
        if self.ds is None:
            raise RuntimeError('dataset not opened')
        if self.recording:
            return
        self.frames, self.error = 0, None
        self.frame_stats = {n: {'duplicate': 0, 'skipped': 0} for n in self.stream_names()}
        self._last_count = {}
        self.started = time.monotonic()
        self._ep_id += 1
        self._q.put(('begin', self._ep_id))
        log.info('EPISODE %d BEGIN (dataset %s, %d episodes already in it)', self._ep_id, self.root.name, self._n_eps)
        self.recording = True

    def _abort(self, msg):
        log.error('EPISODE ABORTED by recorder: %s (frames so far %d)', msg, self.frames)
        self.error = msg
        self.recording = False
        self._q.put(('discard', self._ep_id))

    def push(self, snap):
        """Called from the engine loop; must be fast (the cameras already hold copied frames)."""
        if not self.recording:
            return
        if self._q.qsize() > MAX_BACKLOG:
            return self._abort('encoder backlog too large: episode aborted')
        streams, ages, ids = {}, [], []
        now = time.monotonic()
        imu = {}
        for name, cam in self.cameras.items():
            fr, ts, cnt = cam.latest()
            if fr is None:
                return self._abort(f'camera {name} has no frames: episode aborted')
            streams[name] = fr
            ages.append((now - ts) * 1000)
            ids.append(cnt)
        for name, cam in self.cameras.items():
            if cam.has_depth:
                dm, ts, cnt, imu_v = cam.latest_depth()
                if dm is None:
                    return self._abort(f'camera {name} has no depth frames: episode aborted')
                streams[f'{name}_depth'] = dm
                ages.append((now - ts) * 1000)
                ids.append(cnt)
                imu[name] = imu_v
        for name, cnt in zip(self.stream_names(), ids):
            last = self._last_count.get(name)
            if last is not None:
                self.frame_stats[name]['duplicate'] += int(cnt == last)
                self.frame_stats[name]['skipped'] += max(0, cnt - last - 1)
            self._last_count[name] = cnt
        self._q.put(('frame', (self._ep_id, snap, streams, ages, ids, imu, time.time())))
        self.frames += 1

    def end(self, save=True):
        """Stop the episode. Returns once the writer has finished (may take seconds for encoding)."""
        was = self.recording
        self.recording = False
        if not was and self.error is None:
            return 'not recording'
        done = threading.Event()
        res = {}
        self.saving = True
        t0 = time.monotonic()
        log.info('EPISODE %d END requested (save=%s, %d frames pushed, backlog %d)', self._ep_id,
                 save and self.error is None, self.frames, self._q.qsize())
        self._q.put(('end', (self._ep_id, save and self.error is None, done, res)))
        done.wait()
        self.saving = False
        log.info('EPISODE %d END done in %.1f s: %s', self._ep_id, time.monotonic() - t0, res.get('status'))
        if 'error' in res:
            self.error = res['error']
        return res.get('status', 'error')

    # ---------------------------------------------------------------- worker thread
    def _frame(self, snap, streams, ages, ids, imu, unix_time):
        L, F = snap['leader'], snap['follower']
        frame = {'task': self.task,
                 'observation.state': vec(JOINTS, lambda n: snap['follower_units'][n]),
                 'action': vec(JOINTS, lambda n: snap['leader_units'][n])}
        for name, arr in streams.items():
            if name.endswith('_depth') and name[:-6] in self.cameras and self.cameras[name[:-6]].has_depth:
                frame[f'observation.images.{name}'] = arr[..., None]           # uint16 (H, W, 1)
            else:
                frame[f'observation.images.{name}'] = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        for name, v in imu.items():
            if self.cameras[name].has_imu:
                frame[f'observation.{name}.imu'] = v.astype(np.float32)
        if self.extras:
            frame['observation.follower.raw_position'] = vec(JOINTS, lambda n: F[n]['position'])
            frame['action.leader.raw_position'] = vec(JOINTS, lambda n: L[n]['position'])
            frame['action.goal_sent'] = vec(JOINTS, lambda n: snap['goal_units'][n])
            for key, arm, field in TELEMETRY:
                src = F if arm == 'follower' else L
                frame[key] = vec(JOINTS, lambda n, s=src, f=field: s[n][f])
            frame['timing.unix_time'] = np.array([unix_time], dtype=np.float64)
            frame['timing.read_ms'] = np.array([snap.get('read_ms', 0.0)], dtype=np.float32)
            frame['timing.camera_age_ms'] = np.array(ages, dtype=np.float32)
            frame['timing.camera_frame_id'] = np.array(ids, dtype=np.float64)
        return frame

    def _save(self, expected_frames):
        """save_episode, then finalize + re-open and VERIFY the files on disk match what was recorded."""
        before_eps, before_frames = self.ds.meta.total_episodes, self.ds.meta.total_frames
        self.ds.save_episode()
        n = self.ds.num_episodes
        self.ds.finalize()
        self.ds = self._open_ds(False)
        self._n_eps = n
        got_eps = self.ds.meta.total_episodes - before_eps
        got_frames = self.ds.meta.total_frames - before_frames
        if got_eps != 1 or got_frames != expected_frames:
            raise RuntimeError(f'saved dataset inconsistent: +{got_eps} episodes, +{got_frames} frames '
                               f'(expected +1 / +{expected_frames})')
        log.info('episode %d saved and verified (%d frames)', n - 1, expected_frames)
        return n - 1

    def _preview(self, name, arr):
        """The preview video is a convenience: any failure here disables it for the rest of the episode
        and is logged, but never affects the dataset episode."""
        if name in self._preview_failed:
            return
        try:
            w = self._previews.get(name)
            if w is None:
                d = self.root / 'depth_preview'
                w = self._previews[name] = PreviewWriter(d / f'.recording_{name}.mp4', arr.shape[1], arr.shape[0], self.fps)
            w.write(arr)
        except Exception as e:      # noqa: BLE001
            log.error('depth preview failed for %s (episode continues, preview disabled)', name, exc_info=e)
            self._preview_failed.add(name)
            self.preview_errors.append(f'{name}: {e!r}')

    def _finish_previews(self, keep, episode_index=None):
        for name, w in list(self._previews.items()):
            try:
                w.close()
                if keep and episode_index is not None and name not in self._preview_failed:
                    w.path.replace(w.path.with_name(f'episode_{episode_index:06d}_{name}.mp4'))
                else:
                    w.path.unlink(missing_ok=True)
            except Exception as e:      # noqa: BLE001
                log.error('closing depth preview %s failed', name, exc_info=e)
        self._previews = {}
        self._preview_failed = set()

    def _work(self):
        cur = None                  # id of the episode the writer is currently building
        in_episode = 0
        self._previews = {}
        self._preview_failed = set()
        dropped = 0
        while True:
            item = self._q.get()
            if item is None:
                return
            kind, payload = item
            try:
                if kind == 'begin':
                    if in_episode or cur is not None:       # leftovers of an unfinished episode: never keep them
                        log.warning('begin: clearing %d leftover frames of episode %s', in_episode, cur)
                        self._finish_previews(False)
                        self.ds.clear_episode_buffer()
                    cur, in_episode, dropped = payload, 0, 0
                elif kind == 'frame':
                    ep_id = payload[0]
                    if ep_id != cur:
                        dropped += 1
                        if dropped in (1, 10, 100):
                            log.warning('dropped %d frame(s) of episode %s (writer is on episode %s)', dropped, ep_id, cur)
                        continue
                    frame_args = payload[1:]
                    self.ds.add_frame(self._frame(*frame_args))
                    for sname, arr in frame_args[1].items():
                        if sname.endswith('_depth'):
                            self._preview(sname, arr)
                    in_episode += 1
                elif kind == 'discard':
                    if payload == cur:
                        self._finish_previews(False)
                        self.ds.clear_episode_buffer()
                        cur, in_episode = None, 0
                elif kind == 'end':
                    ep_id, save, done, res = payload
                    try:
                        if ep_id != cur:
                            res['status'] = 'discarded (episode already aborted)'
                        elif save and in_episode >= MIN_FRAMES:
                            # the writer's own frame counter must agree with ours, otherwise the episode would
                            # contain frames that are not ours -> never write such an episode
                            buf = self.ds.writer.episode_buffer['size']
                            if buf != in_episode:
                                raise RuntimeError(f'internal frame count mismatch: writer buffer {buf} vs counted '
                                                   f'{in_episode}; episode discarded instead of saved')
                            idx = self._save(in_episode)
                            self._finish_previews(True, idx)
                            res['status'] = f'saved episode {idx} ({in_episode} frames)'
                            self.last_saved = res['status']
                        else:
                            self._finish_previews(False)
                            self.ds.clear_episode_buffer()
                            res['status'] = 'discarded' if not save else f'discarded (only {in_episode} frames)'
                    except Exception as e:      # noqa: BLE001
                        log.error('SAVE FAILED (%d frames lost)', in_episode, exc_info=e)
                        res['error'] = f'save failed: {e!r}'
                        res['status'] = 'error'
                        try:
                            self._finish_previews(False)
                            self.ds.clear_episode_buffer()
                        except Exception:       # noqa: BLE001
                            log.error('clear_episode_buffer after failed save also failed', exc_info=True)
                    cur, in_episode = None, 0
                    done.set()
            except Exception as e:      # noqa: BLE001
                log.error('recorder worker error (kind=%s)', kind, exc_info=e)
                self.error = f'recorder: {e!r}'
                self.recording = False
                if kind == 'end':
                    payload[2].set()

    def status(self):
        bad = {k: v for k, v in self.frame_stats.items() if v['duplicate'] or v['skipped']}
        return {
            'open': self.ds is not None, 'root': str(self.root) if self.root else None,
            'episodes': self.num_episodes, 'recording': self.recording, 'saving': self.saving, 'frames': self.frames,
            'elapsed': round(time.monotonic() - self.started, 1) if self.recording else 0,
            'backlog': self._q.qsize(), 'error': self.error, 'last_saved': self.last_saved,
            'task': self.task, 'fps': self.fps, 'frame_issues': bad,
            'streams': self.stream_names(), 'preview_errors': list(getattr(self, 'preview_errors', []))[-3:],
            'disk_free_gb': round(shutil.disk_usage(self.root if self.root and self.root.exists() else '.').free / 1e9, 1),
        }
