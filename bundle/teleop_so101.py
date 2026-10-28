"""SO-101 absolute pose teleoperation; default is a read-only preflight."""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT/'teleop_config.json').read_text())
LEADER = CONFIG['leader_port']
FOLLOWER = CONFIG['follower_port']


class TrackingWatchdog:
    """Allow travel time, but stop if a commanded joint makes no progress."""
    def __init__(self, positions, now):
        self.anchor = dict(positions)
        self.last_progress = {n: now for n in positions}
        self.direction = {n: 0 for n in positions}

    def check(self, name, actual, goal, now):
        error = goal-actual
        direction = 1 if error > 0 else -1
        progress = (actual-self.anchor[name])*direction
        if abs(error) <= 65 or direction != self.direction[name] or progress >= 3:
            self.anchor[name] = actual
            self.last_progress[name] = now
        elif now-self.last_progress[name] >= 2:
            if name == 'gripper' and error < -65:
                # Closing can legitimately stop on an object. The current/torque
                # limits and full health checks remain active. Opening must move.
                return True
            raise RuntimeError(f'{name}: no progress toward goal for 2 s; '
                               f'actual={actual}, goal={round(goal)}, error={round(error)}')
        self.direction[name] = direction if abs(error) > 65 else 0
        return False


def absolute_target(name, leader_now, leader_limits, follower_limits):
    if name == 'gripper':
        fraction = (leader_now-leader_limits['range_min']) / (leader_limits['range_max']-leader_limits['range_min'])
        desired = follower_limits['range_min'] + fraction * (follower_limits['range_max']-follower_limits['range_min'])
    else:
        # Both encoders use the calibrated pose at 2047 and 4096 counts/turn.
        desired = leader_now
    return max(follower_limits['range_min'], min(follower_limits['range_max'], desired))


def bounded_target(start, leader_start, leader_now, limits, previous, dt, full=False,
                   absolute=None, aligning=False, direct=False):
    desired = absolute if absolute is not None else start + (1 if full else .35) * (leader_now - leader_start)
    desired = max(limits['range_min'], min(limits['range_max'], desired))
    if not full and absolute is None:
        desired = max(start - 180, min(start + 180, desired))
    if direct and not aligning:
        return desired
    step = (160 if full and not aligning else 40) * min(dt, .15)
    return max(previous - step, min(previous + step, desired))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--full-range', action='store_true', help='Use full calibrated range and faster tracking after alignment')
    parser.add_argument('--relative', action='store_true', help='Use the previous relative mode without initial pose alignment')
    parser.add_argument('--smooth-tracking', action='store_true', help='Retain software slew limiting after initial alignment')
    parser.add_argument('--resume-holding', action='store_true', help='Resume from a previously held follower pose')
    parser.add_argument('--seconds', type=float, default=CONFIG['session_seconds'])
    parser.add_argument('--speed', type=int, default=CONFIG['speed_counts_per_second'], help='Full-range motor speed, encoder counts/s (65..600)')
    args = parser.parse_args()
    if not 0 <= args.seconds <= 300:
        parser.error('seconds must be 0..300')
    if not 65 <= args.speed <= 600:
        parser.error('speed must be 65..600 counts/s')
    import serial
    from serial_protocol import exchange
    from verified_telemetry import read_health
    lc = json.loads((ROOT/'leader_setup/so101_leader_calibration.json').read_text())
    fc = json.loads((ROOT/'so101_follower_calibration.json').read_text())
    log = {'started': time.time(), 'execute': args.execute, 'events': []}
    log_path = ROOT/'leader_setup/teleop_last_run.json'
    active = []
    with serial.Serial(LEADER, CONFIG['baudrate'], timeout=.1, exclusive=True) as leader, serial.Serial(FOLLOWER, CONFIG['baudrate'], timeout=.1, exclusive=True) as follower:
        def tx(port, i, cmd, params, size):
            def retry(event):
                log['events'].append(event)
            return exchange(port, i, cmd, params, size, on_retry=retry)
        def rd(port, i, address, size=1):
            return int.from_bytes(tx(port, i, 2, [address, size], size+6), 'little')
        def wr(i, address, value, size=1):
            tx(follower, i, 3, [address, *value.to_bytes(size, 'little')], 6)
            assert rd(follower, i, address, size) == value
        def positions(port, cal):
            result = {}
            for name, c in cal.items():
                p = rd(port, c['id'], 56, 2)
                if not c['range_min']-25 <= p <= c['range_max']+25:
                    raise RuntimeError(f'{name}: encoder outside calibrated range: {p}')
                result[name] = p
            return result
        def goals(pose):
            params = [42, 2]
            for name, c in fc.items():
                v = round(pose[name])
                assert c['range_min'] <= v <= c['range_max']
                params.extend([c['id'], v & 255, v >> 8])
            body = [254, len(params)+2, 131, *params]
            follower.write(bytes([255, 255, *body, (~sum(body)) & 255]))
            follower.flush()
        def health():
            result = {}
            for name, c in fc.items():
                h = read_health(c['id'], lambda *a: tx(follower, *a))
                if 'confirmed_temperature_outlier' in h:
                    event = {'joint': name, 'temperature_outlier': h['confirmed_temperature_outlier'],
                             'confirmations': h['temperature_confirmations']}
                    log['events'].append(event)
                    print('TEMPERATURE RECHECK:', event, flush=True)
                if h['temp'] >= 45 or h['load'] >= 850 or not 105 <= h['voltage'] <= 135:
                    raise RuntimeError(f'{name}: unsafe telemetry {h}')
                result[name] = h
            return result
        try:
            for port, cal in ((leader, lc), (follower, fc)):
                for name, c in cal.items():
                    i = c['id']
                    assert rd(port, i, 3, 2) == 777
                    assert rd(port, i, 5) == i and rd(port, i, 33) == 0
                    torque = rd(port, i, 40)
                    assert torque == 0 or (port is follower and args.resume_holding and torque == 1), f'{name}: torque already enabled'
                    if torque:
                        assert abs(rd(port, i, 42, 2)-rd(port, i, 56, 2)) <= 65, f'{name}: not settled in hold'
                    off = c['homing_offset']
                    assert rd(port, i, 31, 2) == (abs(off) | (2048 if off < 0 else 0))
                    if port is follower:
                        assert rd(port, i, 9, 2) == c['range_min']
                        assert rd(port, i, 11, 2) == c['range_max']
            ls = positions(leader, lc)
            fs = positions(follower, fc)
            assert all(fc[n]['range_min'] <= p <= fc[n]['range_max'] for n, p in fs.items())
            log.update(leader_start=ls, follower_start=fs, health=health())
            print(json.dumps(log, indent=2), flush=True)
            if not args.execute:
                print('PREFLIGHT OK. No motors enabled.', flush=True)
                return
            for name, c in fc.items():
                wr(c['id'], 41, 5)
                wr(c['id'], 44, 0, 2)
                wr(c['id'], 46, args.speed if args.full_range and args.relative else 65, 2)
                if name == 'gripper':
                    wr(c['id'], 48, min(200, rd(follower, c['id'], 48, 2)), 2)
            goals(fs)
            for name, c in fc.items():
                if name not in active:
                    active.append(name)
                wr(c['id'], 40, 1)
            previous = dict(fs)
            last_leader = dict(ls)
            aligning = not args.relative
            start = last = time.monotonic()
            tracking = TrackingWatchdog(fs, start)
            gripper_contact = False
            print('TELEOP ACTIVE: '+('relative' if args.relative else 'absolute pose; slow initial alignment at 40 counts/s')+'. Ctrl+C freezes and holds.', flush=True)
            while args.seconds == 0 or time.monotonic()-start < args.seconds:
                tick = time.monotonic()
                lp = positions(leader, lc)
                h = health()
                now = time.monotonic()
                log['last_telemetry'] = h
                log['last_goals'] = dict(previous)
                log.setdefault('recent_tracking', []).append({
                    'elapsed': now-start, 'goals': dict(previous),
                    'actual': {n: h[n]['position'] for n in fc}})
                log['recent_tracking'] = log['recent_tracking'][-30:]
                if now-last > .4:
                    raise RuntimeError('Telemetry stale; stopping')
                for name in fc:
                    if abs(lp[name]-last_leader[name]) > 250:
                        raise RuntimeError(f'{name}: abrupt leader encoder change')
                    actual = h[name]['position']
                    if not fc[name]['range_min']-25 <= actual <= fc[name]['range_max']+25:
                        raise RuntimeError(f'{name}: follower outside calibrated range: {actual}')
                    contact = tracking.check(name, actual, previous[name], now)
                    if name == 'gripper':
                        if contact and not gripper_contact:
                            print('GRIPPER CONTACT: closing stopped; keeping limited gripping effort.', flush=True)
                            log['events'].append({'gripper_contact': True, 'position': actual,
                                                  'goal': previous[name], 'elapsed': now-start})
                        gripper_contact = contact
                desired = {n: absolute_target(n, lp[n], lc[n], fc[n]) for n in fc}
                if aligning and all(abs(h[n]['position']-desired[n]) <= 25 or
                                    (n == 'gripper' and gripper_contact and desired[n] < h[n]['position'])
                                    for n in fc):
                    aligning = False
                    if args.full_range:
                        for c in fc.values():
                            wr(c['id'], 46, args.speed, 2)
                    print(f'POSE ALIGNED. Following joystick; motor speed={args.speed if args.full_range else 65} counts/s.', flush=True)
                target = {n: bounded_target(fs[n], ls[n], lp[n], fc[n], previous[n], now-last,
                          args.full_range, None if args.relative else desired[n], aligning,
                          direct=not args.smooth_tracking) for n in fc}
                goals(target)
                previous, last_leader, last = target, lp, now
                time.sleep(max(0, .1-(time.monotonic()-tick)))
            print('TIME LIMIT REACHED. Stopping teleop.', flush=True)
        except BaseException as e:
            log['error'] = repr(e)
            print('STOP:', repr(e), flush=True)
            raise
        finally:
            if active:
                try:
                    hold = positions(follower, fc)
                    goals({n: max(fc[n]['range_min'], min(fc[n]['range_max'], p)) for n, p in hold.items()})
                    log['held_pose'] = hold
                    print('HOLDING. Support follower before disabling torque/power.', flush=True)
                except Exception as e:
                    log['hold_error'] = repr(e)
                    print('Hold update failed. Motors may retain previous goals; support arm.', flush=True)
            log['active_motors'] = active
            log_path.write_text(json.dumps(log, indent=2)+'\n')


if __name__ == '__main__':
    main()
