"""SO-101 leader/follower engine.

Re-implements the control loop of bundle/teleop_so101.py as a long-lived, command driven
state machine (disconnected -> monitor -> teleop -> monitor ...) and reads the *full* motor
telemetry block of every servo of both arms on every tick, so a recorder can log it.

The safety logic (preflight asserts, TrackingWatchdog, absolute_target, bounded_target,
temperature confirmation, load/voltage limits, stale telemetry, leader jump check) is taken
unchanged from the bundle.  Nothing here moves a motor unless start_teleop() is called.
"""
import collections
import json
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / 'bundle'
sys.path.insert(0, str(BUNDLE))

import serial  # noqa: E402
from serial_protocol import exchange  # noqa: E402
from teleop_so101 import (TrackingWatchdog, absolute_target, bounded_target)  # noqa: E402
from applog import log  # noqa: E402

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
CONFIG = json.loads((BUNDLE / 'teleop_config.json').read_text())
LEADER_CAL = json.loads((BUNDLE / 'leader_setup/so101_leader_calibration.json').read_text())
FOLLOWER_CAL = json.loads((BUNDLE / 'so101_follower_calibration.json').read_text())

MAX_SESSION_SECONDS = 300      # bundle rule: bounded sessions only
TEMP_LIMIT, LOAD_LIMIT = 45, 850
# Alert thresholds (warnings only; the hard limits above stop teleop as in the bundle)
VOLT_RANGE = {'follower': (105, 135), 'leader': (40, 90)}   # 0.1 V; the leader runs on its own ~5.5 V supply
STALE_WARN_S = 0.4        # loop gap that is reported
STALE_STOP_S = 1.5        # loop gap that stops teleop (the follower keeps holding its last goal meanwhile)
TEMP_CONFIRM_READS = 5    # extra reads used to tell a real overheat from a glitching temperature sensor
TEMP_GLITCH_MAX = 30      # consecutive glitchy ticks before teleop is stopped
LAG_SUSTAIN_S = 0.7       # lag must persist this long before it is reported
LEADER_ERR_FATAL_S = 3.0  # a leader status/error flag must persist this long to stop teleop
TEMP_WARN = 40            # deg C
LOAD_WARN = 600           # 0.1 % of max torque -> 60 %
CURRENT_WARN = 500        # counts (~6.5 mA each, ~3.2 A)
COLLISION_LOAD = 500      # load that together with a large tracking error suggests a blocked joint
COLLISION_ERR = 80        # counts between goal and position
LAG_WARN = 250            # counts of follower lag behind the goal (sustained)
ALERT_LOG = HERE / 'alerts.jsonl'
STATUS_BITS = {0: 'input voltage', 1: 'angle sensor', 2: 'overheat', 3: 'overcurrent', 4: 'overload'}


def parse_block(d):
    """Feetech STS3215 registers 56..70."""
    u16 = lambda a: d[a] | (d[a + 1] << 8)
    sp, ld = u16(2), u16(4)
    return {
        'position': u16(0),
        'speed': (sp & 0x7FFF) * (-1 if sp & 0x8000 else 1),   # counts/s
        'load': (ld & 1023) * (-1 if ld & 1024 else 1),        # 0.1 % of max torque
        'voltage': d[6],                                       # 0.1 V
        'temp': d[7],                                          # deg C
        'status': d[9],
        'moving': d[10],
        'current': u16(13),                                    # ~6.5 mA / count
    }


def to_units(name, raw, cal):
    """Degrees around the calibrated centre (2048); gripper as 0..100 of its calibrated range."""
    if name == 'gripper':
        lo, hi = cal['range_min'], cal['range_max']
        return 100.0 * (raw - lo) / (hi - lo)
    return (raw - 2048) * 360.0 / 4096.0


class Bus:
    def __init__(self, path, baud, events, name):
        self.name = name
        self.port = serial.Serial(path, baud, timeout=.1, exclusive=True)
        self.lock = threading.Lock()
        self.events = events
        self.retry_log = collections.deque(maxlen=200)   # (monotonic, motor id)
        self.retry_total = collections.Counter()
        self.temp_glitch = collections.Counter()
        self.last_temp = {}

    def _on_retry(self, event):
        event = {**event, 'bus': self.name}
        self.events.append(event)
        self.retry_log.append((time.monotonic(), event.get('id')))
        self.retry_total[event.get('id')] += 1
        log.warning('comm retry on %s: %s', self.name, event)

    def recent_retries(self, window=10.0):
        now = time.monotonic()
        c = collections.Counter(i for t, i in self.retry_log if now - t <= window)
        return dict(c)

    def close(self):
        try:
            self.port.close()
        except Exception:
            pass

    def tx(self, i, cmd, params, size):
        try:
            with self.lock:
                return exchange(self.port, i, cmd, params, size, on_retry=self._on_retry)
        except RuntimeError as e:
            raise RuntimeError(f'{self.name}: {e}') from None

    def block_tolerant(self, i):
        """Read registers 56..70 but keep the data when the servo reports a status/error byte
        (e.g. a voltage flag). Returns (parsed_block, error_byte). Used for the passive leader only."""
        body = [i, 4, 2, 56, 15]
        for attempt in range(3):
            with self.lock:
                self.port.reset_input_buffer()
                self.port.write(bytes([255, 255, *body, (~sum(body)) & 255]))
                self.port.flush()
                r = self.port.read(21)
            if (len(r) == 21 and r[:3] == bytes([255, 255, i]) and r[3] == 17 and sum(r[2:]) % 256 == 255):
                return parse_block(r[5:-1]), r[4]
            self._on_retry({'id': i, 'attempt': attempt + 1, 'response_hex': r.hex(), 'reason': 'invalid or incomplete packet'})
            time.sleep(.01)
        raise RuntimeError(f'{self.name}: ID {i}: no valid response after 3 attempts')

    def rd(self, i, address, size=1):
        return int.from_bytes(self.tx(i, 2, [address, size], size + 6), 'little')

    def wr(self, i, address, value, size=1):
        self.tx(i, 3, [address, *value.to_bytes(size, 'little')], 6)
        assert self.rd(i, address, size) == value, f'ID {i}: write verify failed @ {address}'

    def block(self, i):
        data = self.tx(i, 2, [56, 15], 21)
        if len(data) != 15:
            raise RuntimeError(f'ID {i}: invalid telemetry block length')
        return parse_block(data)

    def read_all(self, cal, follower=False):
        out = {}
        for name in JOINTS:
            i = cal[name]['id']
            if not follower:
                b, err = self.block_tolerant(i)
                b['err'] = err
                out[name] = b
                continue
            b = self.block(i)
            if b['temp'] >= TEMP_LIMIT or b['temp'] < 5:
                b = self._confirm_temp(name, i, b)
            else:
                self.temp_glitch[name] = 0
            self.last_temp[name] = b['temp']
            out[name] = b
        return out

    def _confirm_temp(self, name, i, first):
        """A temperature >= limit (or an impossible one) is re-read several times. Real overheating is
        consistently high -> stop. Scattered garbage values (seen on this hardware: 33,42,57,52,33,47...)
        are a sensor glitch -> use the last good temperature and keep recording; persistent glitches stop."""
        vals = [first['temp']]
        for _ in range(TEMP_CONFIRM_READS):
            time.sleep(.01)
            vals.append(self.block(i)['temp'])
        hot = sum(v >= TEMP_LIMIT for v in vals)
        if hot >= len(vals) - 1:
            raise RuntimeError(f'{self.name}: ID {i} ({name}): OVERHEATING, temperature readings {vals}')
        good = sorted(v for v in vals if 5 <= v < TEMP_LIMIT)
        self.temp_glitch[name] += 1
        log.warning('%s/%s temperature sensor glitch: readings %s (glitch #%d in a row)',
                    self.name, name, vals, self.temp_glitch[name])
        if self.temp_glitch[name] > TEMP_GLITCH_MAX or not good:
            raise RuntimeError(f'{self.name}: ID {i} ({name}): temperature sensor unreliable, readings {vals}')
        first = dict(first)
        first['temp'] = self.last_temp.get(name, good[len(good) // 2])
        first['temp_glitch'] = True
        self.events.append({'bus': self.name, 'joint': name, 'temperature_glitch': vals})
        return first

    def write_goals(self, cal, pose):
        params = [42, 2]
        for name in JOINTS:
            c = cal[name]
            v = round(pose[name])
            # A joint resting on its stop can read a few counts outside the range (bundle tolerance: 25).
            # Clamp such goals into the calibrated range; anything further out is still an error.
            assert c['range_min'] - 25 <= v <= c['range_max'] + 25, f'{name}: goal {v} outside range'
            v = max(c['range_min'], min(c['range_max'], v))
            params.extend([c['id'], v & 255, v >> 8])
        body = [254, len(params) + 2, 131, *params]
        with self.lock:
            self.port.write(bytes([255, 255, *body, (~sum(body)) & 255]))
            self.port.flush()


class Engine:
    def __init__(self, fps=30):
        self.fps = fps
        self.state = 'disconnected'       # disconnected | monitor | teleop | error
        self.message = ''
        self.aligning = False
        self.gripper_contact = False
        self.teleop_deadline = None
        self.teleop_speed = CONFIG['speed_counts_per_second']
        self.events = []
        self.leader = self.follower = None
        self.holding = False              # follower torque is on (holding a pose)
        self.latest = None                # last published telemetry snapshot
        self.loop_hz = 0.0
        self.on_sample = None             # callback(sample) from the loop thread while teleop aligned
        self._cmds = queue.Queue()
        self._thread = None
        self._stop = threading.Event()
        self._pool = ThreadPoolExecutor(2)
        self._lock = threading.Lock()
        self.auto_release = 0             # seconds after teleop ends before torque is released (0 = never)
        self.release_at = None
        self._lag_since = {}
        self._leader_err_since = {}
        self.alerts = collections.deque(maxlen=300)
        self.alert_total = 0
        self.active_alerts = {}           # joint -> {kind: level} currently true
        self._alert_seen = {}

    # ------------------------------------------------------------------ public API (thread safe)
    def command(self, name, timeout=30, **kw):
        done = threading.Event()
        box = {}
        self._cmds.put((name, kw, done, box))
        if not done.wait(timeout):
            raise TimeoutError(f'{name}: timed out')
        if 'error' in box:
            raise RuntimeError(box['error'])
        return box.get('result')

    def connect(self, resume_holding=False):
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name='engine', daemon=True)
            self._thread.start()
        return self.command('connect', resume_holding=resume_holding)

    def disconnect(self):
        if self._thread and self._thread.is_alive():
            self.command('disconnect')

    def start_teleop(self, seconds=60, speed=None):
        return self.command('start_teleop', seconds=seconds, speed=speed or self.teleop_speed)

    def stop_teleop(self):
        return self.command('stop_teleop')

    def alert(self, level, joint, kind, text, every=5.0):
        """Record an alert (level: info|warn|crit); the same (joint, kind) is re-logged at most every `every` s."""
        now = time.monotonic()
        key = (joint, kind)
        if now - self._alert_seen.get(key, -1e9) < every:
            return
        self._alert_seen[key] = now
        row = {'time': time.strftime('%H:%M:%S'), 'ts': time.time(), 'level': level,
               'joint': joint, 'kind': kind, 'text': text, 'state': self.state}
        self.alerts.append(row)
        self.alert_total += 1
        {'info': log.info, 'warn': log.warning, 'crit': log.error}[level]('ALERT %s %s: %s', joint, kind, text)
        try:
            with open(ALERT_LOG, 'a') as f:
                f.write(json.dumps(row) + '\n')
        except OSError:
            pass

    def _scan_alerts(self, lstate, fstate):
        """Heating, overload, current, voltage, hardware-status and collision heuristics for both arms."""
        active = {}
        goal = self._t['previous'] if self.state == 'teleop' and not self.aligning else None
        for arm, st in (('follower', fstate), ('leader', lstate)):
            for n in JOINTS:
                h = st[n]
                j = f'{arm}/{n}'
                a = active.setdefault(j, {})
                if h['temp'] >= TEMP_LIMIT:
                    a['temp'] = 'crit'
                    self.alert('crit', j, 'temp', f'{j} OVERHEATING {h["temp"]} C (limit {TEMP_LIMIT})')
                elif h['temp'] >= TEMP_WARN:
                    a['temp'] = 'warn'
                    self.alert('warn', j, 'temp', f'{j} is hot: {h["temp"]} C', every=30)
                if arm == 'follower':
                    if abs(h['load']) >= LOAD_LIMIT:
                        a['load'] = 'crit'
                        self.alert('crit', j, 'load', f'{j} OVERLOAD {h["load"] / 10:.0f} % torque')
                    elif abs(h['load']) >= LOAD_WARN and n != 'gripper':
                        a['load'] = 'warn'
                        self.alert('warn', j, 'load', f'{j} high load {h["load"] / 10:.0f} % - blocked or heavy contact?')
                    if h['current'] >= CURRENT_WARN:
                        a['current'] = 'warn'
                        self.alert('warn', j, 'current', f'{j} high current {h["current"] * 6.5 / 1000:.1f} A')
                    if goal is not None:
                        err = goal[n] - h['position']
                        if (n != 'gripper' and abs(h['load']) >= COLLISION_LOAD and abs(err) >= COLLISION_ERR):
                            a['collision'] = 'crit'
                            self.alert('crit', j, 'collision',
                                       f'POSSIBLE COLLISION / BLOCKED: {j} load {h["load"] / 10:.0f} %, '
                                       f'lag {abs(err):.0f} counts')
                        elif abs(err) >= LAG_WARN and n != 'gripper':
                            t0 = self._lag_since.setdefault(n, time.monotonic())
                            if time.monotonic() - t0 >= LAG_SUSTAIN_S:
                                a['lag'] = 'warn'
                                self.alert('warn', j, 'lag',
                                           f'{j} lagging {abs(err):.0f} counts behind the goal for >{LAG_SUSTAIN_S} s')
                        else:
                            self._lag_since.pop(n, None)
                    else:
                        self._lag_since.pop(n, None)
                lo, hi = VOLT_RANGE[arm]
                if not lo <= h['voltage'] <= hi:
                    a['voltage'] = 'warn'
                    self.alert('warn', j, 'voltage',
                               f'{j} supply voltage {h["voltage"] / 10:.1f} V out of {lo / 10:.1f}-{hi / 10:.1f} V')
                if arm == 'leader' and h.get('err'):
                    a['status'] = 'warn'
                    self.alert('warn', j, 'status', f'{j} reports error byte {h["err"]} (data still read)')
                if h['status']:
                    bits = [v for b, v in STATUS_BITS.items() if h['status'] >> b & 1] or [str(h['status'])]
                    a['status'] = 'crit' if arm == 'follower' else 'warn'
                    self.alert('crit' if arm == 'follower' else 'warn', j, 'status',
                               f'{j} hardware error: {", ".join(bits)}')
                if not a:
                    del active[j]
        if self.gripper_contact:
            active.setdefault('follower/gripper', {})['contact'] = 'info'
        self.active_alerts = active

    def snapshot(self):
        with self._lock:
            s = dict(self.latest or {})
        s.update(state=self.state, message=self.message, aligning=self.aligning,
                 gripper_contact=self.gripper_contact, loop_hz=round(self.loop_hz, 1),
                 teleop_left=(max(0, self.teleop_deadline - time.monotonic())
                              if self.teleop_deadline else None),
                 release_in=(max(0, self.release_at - time.monotonic()) if self.release_at else None),
                 holding=self.holding, retries=len(self.events),
                 comm={'leader': self.leader.recent_retries() if self.leader else {},
                       'follower': self.follower.recent_retries() if self.follower else {},
                       'leader_total': dict(self.leader.retry_total) if self.leader else {},
                       'follower_total': dict(self.follower.retry_total) if self.follower else {}},
                 alerts=list(self.alerts)[-40:], alert_total=self.alert_total,
                 active_alerts=self.active_alerts)
        return s

    # ------------------------------------------------------------------ loop thread
    def _run(self):
        period = 1.0 / self.fps
        nxt = time.monotonic()
        rates = []
        while not self._stop.is_set():
            try:
                while True:
                    name, kw, done, box = self._cmds.get_nowait()
                    try:
                        box['result'] = getattr(self, '_cmd_' + name)(**kw)
                    except BaseException as e:   # noqa: BLE001
                        box['error'] = f'{type(e).__name__}: {e}'
                        log.error('command %s failed: %s', name, box['error'], exc_info=e)
                        if self.state == 'teleop':
                            self._fail(e)
                    finally:
                        done.set()
            except queue.Empty:
                pass
            if self.state in ('monitor', 'error'):
                self._maybe_release()
            if self.state in ('monitor', 'teleop'):
                try:
                    self._step()
                except BaseException as e:      # noqa: BLE001
                    self._fail(e)
            now = time.monotonic()
            rates.append(now)
            rates = [t for t in rates if now - t < 2]
            if len(rates) > 1:
                self.loop_hz = (len(rates) - 1) / (rates[-1] - rates[0])
            nxt += period
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.monotonic()
        self._close_buses()

    def _schedule_release(self):
        if self.auto_release > 0 and self.follower is not None:
            self.release_at = time.monotonic() + self.auto_release
            self.alert('warn', 'system', 'release',
                       f'Follower torque will be RELEASED in {self.auto_release} s - support the arm (cancel to keep holding)',
                       every=0)

    def _maybe_release(self):
        if self.release_at is not None and time.monotonic() >= self.release_at:
            self.release_at = None
            try:
                self._cmd_disable_torque()
                self.alert('info', 'system', 'release', 'Follower torque released (arm is limp).', every=0)
            except Exception as e:      # noqa: BLE001
                self.alert('crit', 'system', 'release', f'Torque release FAILED: {e!r} - support the arm', every=0)

    def _fail(self, exc):
        """Any teleop error: freeze at present pose (hold), report, go to error state."""
        self.message = f'{type(exc).__name__}: {exc}'
        log.error('ENGINE FAILURE in state %s: %s', self.state, self.message, exc_info=exc)
        self.events.append({'error': self.message})
        self.alert('crit', 'system', 'stop', f'STOPPED: {self.message}', every=0)
        was_teleop = self.state == 'teleop'
        self.state = 'error'
        self.aligning = False
        self.teleop_deadline = None
        if was_teleop:
            self._hold()
            self._schedule_release()

    def _hold(self):
        try:
            hold = {n: self.follower.rd(FOLLOWER_CAL[n]['id'], 56, 2) for n in JOINTS}
            self.follower.write_goals(FOLLOWER_CAL, {
                n: max(FOLLOWER_CAL[n]['range_min'], min(FOLLOWER_CAL[n]['range_max'], p))
                for n, p in hold.items()})
            self.holding = True
        except Exception as e:      # noqa: BLE001
            log.error('HOLD update failed', exc_info=e)
            self.events.append({'hold_error': repr(e)})
            self.message += ' | HOLD UPDATE FAILED, support the arm'

    def _close_buses(self):
        for b in (self.leader, self.follower):
            if b:
                b.close()
        self.leader = self.follower = None
        self.state = 'disconnected'

    # ------------------------------------------------------------------ commands
    def _cmd_connect(self, resume_holding=False):
        if self.state != 'disconnected' and self.state != 'error':
            return 'already connected'
        self._close_buses()
        self.message = ''
        self.events.clear()
        try:
            self.leader = Bus(CONFIG['leader_port'], CONFIG['baudrate'], self.events, 'leader')
            self.follower = Bus(CONFIG['follower_port'], CONFIG['baudrate'], self.events, 'follower')
            self._preflight(resume_holding)
        except BaseException:
            self._close_buses()
            raise
        self.state = 'monitor'
        self.message = 'Connected (read-only: no motor enabled).'
        return 'ok'

    def _cmd_disconnect(self):
        if self.state == 'teleop':
            self._cmd_stop_teleop()
        if self.auto_release > 0 and self.release_at is not None and self.follower is not None:
            self.release_at = 0       # release now: we are about to close the port
            self._maybe_release()
        self._close_buses()
        self.message = 'Disconnected. Follower torque unchanged.'
        return 'ok'

    def _preflight(self, resume_holding):
        for bus, cal, is_follower in ((self.leader, LEADER_CAL, False), (self.follower, FOLLOWER_CAL, True)):
            who = 'follower' if is_follower else 'leader'
            for name, c in cal.items():
                i = c['id']
                if bus.rd(i, 3, 2) != 777:
                    raise RuntimeError(f'{who}/{name}: unexpected motor model')
                if bus.rd(i, 5) != i or bus.rd(i, 33) != 0:
                    raise RuntimeError(f'{who}/{name}: ID/mode mismatch')
                torque = bus.rd(i, 40)
                if not (torque == 0 or (is_follower and resume_holding and torque == 1)):
                    raise RuntimeError(f'{who}/{name}: torque already enabled '
                                       f'(tick "follower already holding" if that is intended)')
                if torque:
                    if abs(bus.rd(i, 42, 2) - bus.rd(i, 56, 2)) > 65:
                        raise RuntimeError(f'{who}/{name}: not settled in hold')
                    self.holding = True
                off = c['homing_offset']
                if bus.rd(i, 31, 2) != (abs(off) | (2048 if off < 0 else 0)):
                    raise RuntimeError(f'{who}/{name}: homing offset differs from calibration file')
                if is_follower:
                    if bus.rd(i, 9, 2) != c['range_min'] or bus.rd(i, 11, 2) != c['range_max']:
                        raise RuntimeError(f'{who}/{name}: EEPROM angle limits differ from calibration file')
        for bus, cal in ((self.leader, LEADER_CAL), (self.follower, FOLLOWER_CAL)):
            for name, c in cal.items():
                p = bus.rd(c['id'], 56, 2)
                if not c['range_min'] - 25 <= p <= c['range_max'] + 25:
                    raise RuntimeError(f'{name}: encoder outside calibrated range: {p}')
        self.follower.read_all(FOLLOWER_CAL, follower=True)  # temp / health sanity

    def _check_health(self, fs):
        for name, h in fs.items():
            if h['temp'] >= TEMP_LIMIT or abs(h['load']) >= LOAD_LIMIT or not 105 <= h['voltage'] <= 135:
                raise RuntimeError(f'{name}: unsafe telemetry {h}')

    def _cmd_start_teleop(self, seconds, speed):
        if self.state != 'monitor':
            raise RuntimeError(f'cannot start teleop from state {self.state}')
        if not 0 < seconds <= MAX_SESSION_SECONDS:
            raise RuntimeError(f'seconds must be 1..{MAX_SESSION_SECONDS}')
        if not 65 <= speed <= 600:
            raise RuntimeError('speed must be 65..600 counts/s')
        self.release_at = None
        fol, cal = self.follower, FOLLOWER_CAL
        fs = {n: fol.rd(cal[n]['id'], 56, 2) for n in JOINTS}
        for n, p in fs.items():
            if not cal[n]['range_min'] - 25 <= p <= cal[n]['range_max'] + 25:
                raise RuntimeError(f'{n}: follower outside calibrated range: {p}')
        self._check_health(fol.read_all(cal, follower=True))
        if self.holding:       # restart from our own hold: it must be settled
            for n in JOINTS:
                if abs(fol.rd(cal[n]['id'], 42, 2) - fs[n]) > 65:
                    raise RuntimeError(f'{n}: not settled in hold')
        ls = {n: self.leader.rd(LEADER_CAL[n]['id'], 56, 2) for n in JOINTS}
        for name, c in cal.items():
            i = c['id']
            fol.wr(i, 41, 5)
            fol.wr(i, 44, 0, 2)
            fol.wr(i, 46, 65, 2)
            if name == 'gripper':
                fol.wr(i, 48, min(200, fol.rd(i, 48, 2)), 2)
        fol.write_goals(cal, fs)
        for c in cal.values():
            fol.wr(c['id'], 40, 1)
        self.holding = True
        clamped = {n: max(cal[n]['range_min'], min(cal[n]['range_max'], p)) for n, p in fs.items()}
        self._t = {
            'speed': speed, 'previous': dict(clamped), 'fs': fs, 'ls': ls, 'last_leader': dict(ls),
            'start': time.monotonic(), 'last': time.monotonic(),
            'watch': TrackingWatchdog(fs, time.monotonic()),
        }
        self.teleop_speed = speed
        self.teleop_deadline = self._t['start'] + seconds
        self.aligning = True
        self.gripper_contact = False
        self.state = 'teleop'
        self.message = 'TELEOP ACTIVE: slow initial alignment (40 counts/s). Keep the leader still.'
        return 'ok'

    def _cmd_stop_teleop(self):
        if self.state != 'teleop':
            return 'not running'
        self.state = 'monitor'
        self.aligning = False
        self.teleop_deadline = None
        self._hold()
        self._schedule_release()
        self.message = 'Teleop stopped. Follower HOLDING (torque on). Support it before disabling torque.'
        return 'ok'

    def _cmd_cancel_release(self):
        self.release_at = None
        self.message = 'Torque release cancelled: follower keeps holding.'
        return 'ok'

    def _cmd_disable_torque(self):
        """Same as bundle/disable_torque.py --supported; only when not teleoperating."""
        if self.state not in ('monitor', 'error'):
            raise RuntimeError('stop teleop first')
        for i in range(1, 7):
            self.follower.wr(i, 40, 0)
        self.holding = False
        self.release_at = None
        self.message = 'Follower torque DISABLED (arm is limp).'
        return 'ok'

    # ------------------------------------------------------------------ per-tick work
    def _read_both(self):
        fl = self._pool.submit(self.leader.read_all, LEADER_CAL, False)
        ff = self._pool.submit(self.follower.read_all, FOLLOWER_CAL, True)
        return fl.result(), ff.result()

    def _step(self):
        tick = time.monotonic()
        lstate, fstate = self._read_both()
        now = time.monotonic()
        snap = {
            't': now,
            'leader': lstate, 'follower': fstate,
            'leader_units': {n: to_units(n, lstate[n]['position'], LEADER_CAL[n]) for n in JOINTS},
            'follower_units': {n: to_units(n, fstate[n]['position'], FOLLOWER_CAL[n]) for n in JOINTS},
            'goal': None,
        }
        try:
            self._scan_alerts(lstate, fstate)
        except Exception as e:      # noqa: BLE001
            log.error('alert scan failed', exc_info=e)
            self.events.append({'alert_scan_error': repr(e)})
        if self.state == 'teleop':
            self._teleop_tick(snap, lstate, fstate, now)
        with self._lock:
            self.latest = snap
        snap['read_ms'] = (time.monotonic() - tick) * 1000

    def _teleop_tick(self, snap, lstate, fstate, now):
        t = self._t
        cal, lc = FOLLOWER_CAL, LEADER_CAL
        lp = {n: lstate[n]['position'] for n in JOINTS}
        for n in JOINTS:
            if not lc[n]['range_min'] - 25 <= lp[n] <= lc[n]['range_max'] + 25:
                raise RuntimeError(f'{n}: leader encoder outside calibrated range: {lp[n]}')
        self._check_health(fstate)
        for n in JOINTS:
            if lstate[n].get('err') or lstate[n]['status']:
                since = self._leader_err_since.setdefault(n, now)
                if now - since >= LEADER_ERR_FATAL_S:
                    raise RuntimeError(f'leader/{n}: status/error flag persisted {LEADER_ERR_FATAL_S:.0f} s '
                                       f'(err={lstate[n].get("err")}, status={lstate[n]["status"]})')
            else:
                self._leader_err_since.pop(n, None)
        gap = now - t['last']
        if gap > STALE_STOP_S:
            raise RuntimeError(f'Control loop stalled {gap:.2f} s (> {STALE_STOP_S} s); stopping')
        if gap > STALE_WARN_S:
            self.alert('warn', 'system', 'stall', f'control loop stalled {gap * 1000:.0f} ms (follower held its last goal)',
                       every=1)
            log.warning('control loop stall %.0f ms (retries last 10 s: leader %s follower %s)', gap * 1000,
                        self.leader.recent_retries(), self.follower.recent_retries())
        for n in JOINTS:
            if abs(lp[n] - t['last_leader'][n]) > 250:
                raise RuntimeError(f'{n}: abrupt leader encoder change')
            actual = fstate[n]['position']
            if not cal[n]['range_min'] - 25 <= actual <= cal[n]['range_max'] + 25:
                raise RuntimeError(f'{n}: follower outside calibrated range: {actual}')
            contact = t['watch'].check(n, actual, t['previous'][n], now)
            if n == 'gripper':
                if contact and not self.gripper_contact:
                    self.events.append({'gripper_contact': True, 'position': actual})
                    self.alert('info', 'follower/gripper', 'contact', 'gripper closing stopped (object contact?)')
                self.gripper_contact = contact
        desired = {n: absolute_target(n, lp[n], lc[n], cal[n]) for n in JOINTS}
        if self.aligning and all(
                abs(fstate[n]['position'] - desired[n]) <= 25 or
                (n == 'gripper' and self.gripper_contact and desired[n] < fstate[n]['position'])
                for n in JOINTS):
            self.aligning = False
            for c in cal.values():
                self.follower.wr(c['id'], 46, t['speed'], 2)
            self.message = f'POSE ALIGNED. Following the leader (motor speed {t["speed"]} counts/s).'
        target = {n: bounded_target(t['fs'][n], t['ls'][n], lp[n], cal[n], t['previous'][n], now - t['last'],
                                    True, desired[n], self.aligning, direct=True) for n in JOINTS}
        self.follower.write_goals(cal, target)
        snap['goal'] = dict(target)
        snap['goal_units'] = {n: to_units(n, target[n], cal[n]) for n in JOINTS}
        t['previous'], t['last_leader'], t['last'] = target, lp, now
        if self.teleop_deadline and now >= self.teleop_deadline:
            self.message = 'TIME LIMIT REACHED. Teleop stopped, follower holding.'
            self.alert('info', 'system', 'timeout', self.message, every=0)
            self.state = 'monitor'
            self.aligning = False
            self.teleop_deadline = None
            self._hold()
            self._schedule_release()
            return
        if not self.aligning and self.on_sample:
            self.on_sample(snap)
