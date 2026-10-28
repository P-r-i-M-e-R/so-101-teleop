"""V4L2 camera capture: one thread per camera, always keeps the newest frame."""
import glob
import os
import threading
import time

import cv2
import numpy as np


def _v4l_index(device):
    """/dev/v4l/by-id/... or /dev/videoN -> integer index (OpenCV warns when given a path)."""
    real = os.path.realpath(device)
    base = os.path.basename(real)
    return int(base[5:]) if base.startswith('video') and base[5:].isdigit() else device


def list_video_nodes():
    """Return [{'path','name','index'}] for /dev/video* capture-capable nodes (deduplicated by by-id order)."""
    nodes = []
    for p in sorted(glob.glob('/dev/video*'), key=lambda s: int(s[10:] or 0)):
        base = os.path.basename(p)
        try:
            name = open(f'/sys/class/video4linux/{base}/name').read().strip()
            idx = int(open(f'/sys/class/video4linux/{base}/index').read().strip())
        except OSError:
            continue
        nodes.append({'path': p, 'name': name, 'index': idx})
    return nodes


def probe(path, width=640, height=480, timeout=2.0):
    """Try to grab one colour frame. Returns (ok, shape or error)."""
    cap = cv2.VideoCapture(_v4l_index(path), cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            return False, 'cannot open'
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        end = time.time() + timeout
        while time.time() < end:
            ok, frame = cap.read()
            if ok and frame is not None:
                if frame.ndim == 3 and frame.shape[2] == 3:
                    return True, list(frame.shape)
                return False, f'not colour: {frame.shape}'
        return False, 'no frames'
    finally:
        cap.release()


class Camera:
    def __init__(self, name, device, width=640, height=480, fps=30):
        self.name, self.device = name, device
        self.width, self.height, self.fps = width, height, fps
        self._lock = threading.Lock()
        self._frame = None          # BGR ndarray
        self._ts = 0.0              # monotonic time of newest frame
        self._count = 0
        self._jpeg = None
        self._jpeg_count = -1
        self.error = None
        self._stop = threading.Event()
        self._thread = None
        self.measured_fps = 0.0
        self.has_depth = False
        self.has_imu = False
        self.depth_size = None
        self.calib = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f'cam-{self.name}', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self):
        cap = cv2.VideoCapture(_v4l_index(self.device), cv2.CAP_V4L2)
        if not cap.isOpened():
            self.error = f'cannot open {self.device}'
            return
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)   # 3 = auto (V4L2 aperture priority); settings persist in the camera
        self.error = None
        window = []
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                now = time.monotonic()
                if not ok or frame is None:
                    self.error = 'read failed'
                    time.sleep(0.05)
                    continue
                self.error = None
                with self._lock:
                    self._frame, self._ts = frame, now
                    self._count += 1
                window.append(now)
                window = [t for t in window if now - t < 2]
                if len(window) > 1:
                    self.measured_fps = (len(window) - 1) / (window[-1] - window[0])
        finally:
            cap.release()

    def latest(self):
        """(frame_bgr, monotonic_ts, counter) — frame must be treated read-only."""
        with self._lock:
            return self._frame, self._ts, self._count

    def latest_depth(self):
        """(depth_uint16_mm_HxW, monotonic_ts, counter, imu[6]) or (None, 0, 0, None) for cameras without depth."""
        return None, 0.0, 0, None

    def jpeg(self, quality=70):
        frame, _, count = self.latest()
        if frame is None:
            return None
        with self._lock:
            if self._jpeg_count == count and self._jpeg is not None:
                return self._jpeg
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        with self._lock:
            self._jpeg, self._jpeg_count = buf.tobytes(), count
            return self._jpeg

    def status(self):
        frame, ts, _ = self.latest()
        return {'name': self.name, 'device': self.device, 'ok': frame is not None and self.error is None,
                'error': self.error, 'fps': round(self.measured_fps, 1),
                'age_ms': round((time.monotonic() - ts) * 1000) if ts else None,
                'size': [self.width, self.height], 'depth': self.has_depth, 'imu': self.has_imu,
                'depth_fps': round(getattr(self, 'depth_fps', 0.0), 1)}


class RealSenseCamera(Camera):
    """Intel RealSense via pyrealsense2: colour + NATIVE depth (unaligned, uint16, depth_scale units) and,
    when the udev rule for the motion module is installed, accelerometer + gyroscope.
    Everything is copied in the capture thread so nothing is dropped by the recorder.
    device = 'realsense' (first device) or 'realsense:<serial>'."""

    DEPTH_W, DEPTH_H = 848, 480

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._depth = None
        self._depth_ts = 0.0
        self._depth_count = 0
        self._imu = np.zeros(6, dtype=np.float32)
        self.depth_fps = 0.0
        self.depth_scale = 0.001
        self.imu_error = None

    def latest_depth(self):
        with self._lock:
            return self._depth, self._depth_ts, self._depth_count, self._imu.copy()

    def depth_jpeg(self, quality=70):
        """Colour-mapped depth preview (0-6.3 m); preview only, the recording keeps raw uint16."""
        dm, _, count, _ = self.latest_depth()
        if dm is None:
            return None
        with self._lock:
            if getattr(self, '_dj_count', -1) == count:
                return self._dj
        vis = cv2.applyColorMap(cv2.convertScaleAbs(dm, alpha=0.04), cv2.COLORMAP_JET)
        vis[dm == 0] = 0
        ok, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        with self._lock:
            self._dj, self._dj_count = buf.tobytes(), count
        return self._dj

    def _start_pipeline(self, rs, serial_no):
        ctx = rs.context()
        devs = [d for d in ctx.query_devices() if serial_no in (None, d.get_info(rs.camera_info.serial_number))]
        if not devs:
            raise RuntimeError('RealSense not found')
        sn = devs[0].get_info(rs.camera_info.serial_number)
        last = None
        for imu in (True, False):          # IMU needs the udev rule; fall back to colour+depth only
            cfg = rs.config()
            cfg.enable_device(sn)
            cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            cfg.enable_stream(rs.stream.depth, self.DEPTH_W, self.DEPTH_H, rs.format.z16, self.fps)
            if imu:
                cfg.enable_stream(rs.stream.accel)
                cfg.enable_stream(rs.stream.gyro)
            pipe = rs.pipeline(ctx)
            try:
                prof = pipe.start(cfg)
                pipe.wait_for_frames(5000)
                self.has_imu = imu
                self.imu_error = None if imu else (str(last)[:120] if last else None)
                return pipe, prof, devs[0]
            except Exception as e:      # noqa: BLE001
                last = e
                try:
                    pipe.stop()
                except Exception:       # noqa: BLE001
                    pass
                if not imu:
                    raise
        raise RuntimeError(str(last))

    def _calibration(self, rs, prof):
        color = prof.get_stream(rs.stream.color).as_video_stream_profile()
        depth = prof.get_stream(rs.stream.depth).as_video_stream_profile()
        intr = lambda p: (lambda i: {'width': i.width, 'height': i.height, 'fx': i.fx, 'fy': i.fy,
                                      'ppx': i.ppx, 'ppy': i.ppy, 'model': str(i.model), 'coeffs': list(i.coeffs)}
                          )(p.get_intrinsics())
        ex = depth.get_extrinsics_to(color)
        dev = prof.get_device()
        return {
            'device': dev.get_info(rs.camera_info.name), 'serial': dev.get_info(rs.camera_info.serial_number),
            'firmware': dev.get_info(rs.camera_info.firmware_version),
            'depth_scale_m_per_unit': self.depth_scale,
            'color_intrinsics': intr(color), 'depth_intrinsics': intr(depth),
            'depth_to_color_extrinsics': {'rotation_3x3_col_major': list(ex.rotation), 'translation_m': list(ex.translation)},
            'depth_format': 'uint16 z16, NOT aligned to colour (native depth stream); metres = value * depth_scale',
            'imu': {'accel': 'm/s^2', 'gyro': 'rad/s'} if self.has_imu else None,
        }

    def _run(self):
        import pyrealsense2 as rs
        serial_no = self.device.split(':', 1)[1] if ':' in self.device else None
        pipe = None
        for attempt in range(2):
            try:
                pipe, prof, dev = self._start_pipeline(rs, serial_no)
                break
            except Exception as e:      # noqa: BLE001
                self.error = f'RealSense start failed ({e}); resetting USB device'
                pipe = None
                if attempt == 0:
                    try:
                        for d in rs.context().query_devices():
                            d.hardware_reset()
                    except Exception:   # noqa: BLE001
                        pass
                    time.sleep(6)
        if pipe is None:
            return
        try:
            self.depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
        except Exception:               # noqa: BLE001
            pass
        self.has_depth = True
        self.depth_size = [self.DEPTH_W, self.DEPTH_H]
        self.calib = self._calibration(rs, prof)
        self.error = None
        win_c, win_d = [], []
        try:
            while not self._stop.is_set():
                try:
                    frames = pipe.wait_for_frames(2000)
                except Exception:       # noqa: BLE001
                    self.error = 'read failed'
                    continue
                now = time.monotonic()
                for f in frames:
                    if f.is_motion_frame():
                        d = f.as_motion_frame().get_motion_data()
                        with self._lock:
                            if f.get_profile().stream_type() == rs.stream.accel:
                                self._imu[0:3] = (d.x, d.y, d.z)
                            else:
                                self._imu[3:6] = (d.x, d.y, d.z)
                if not frames.is_frameset():
                    continue
                fs = frames.as_frameset()
                color, depth = fs.get_color_frame(), fs.get_depth_frame()
                if color:
                    frame = np.asanyarray(color.get_data()).copy()
                    with self._lock:
                        self._frame, self._ts = frame, now
                        self._count += 1
                    win_c.append(now)
                    win_c = [t for t in win_c if now - t < 2]
                    if len(win_c) > 1:
                        self.measured_fps = (len(win_c) - 1) / (win_c[-1] - win_c[0])
                    self.error = None
                if depth:
                    dm = np.asanyarray(depth.get_data()).copy()
                    with self._lock:
                        self._depth, self._depth_ts = dm, now
                        self._depth_count += 1
                    win_d.append(now)
                    win_d = [t for t in win_d if now - t < 2]
                    if len(win_d) > 1:
                        self.depth_fps = (len(win_d) - 1) / (win_d[-1] - win_d[0])
        finally:
            try:
                pipe.stop()
            except Exception:           # noqa: BLE001
                pass


def make_camera(name, device, width, height, fps):
    cls = RealSenseCamera if device.startswith('realsense') else Camera
    return cls(name, device, width, height, fps)
