"""Confirm anomalous servo temperatures using separate full telemetry reads."""
import time


def read_health(motor_id, exchange, limit=45):
    def block():
        data = exchange(motor_id, 2, [56, 15], 21)
        if len(data) != 15:
            raise RuntimeError(f'ID {motor_id}: invalid telemetry block length')
        result = {
            'position': int.from_bytes(data[0:2], 'little'),
            'load': int.from_bytes(data[4:6], 'little') & 1023,
            'voltage': data[6],
            'temp': data[7],
            'status': data[9],
        }
        if result['status'] != 0:
            raise RuntimeError(f'ID {motor_id}: hardware status {result["status"]}')
        return result

    first = block()
    if first['temp'] < limit:
        return first
    confirmations = []
    consecutive_hot = 1
    deadline = time.monotonic() + .25
    for _ in range(8):
        # The caller sends no new movement goals during confirmation.
        time.sleep(.02)
        row = block()
        confirmations.append(row)
        if any(not 105 <= r['voltage'] <= 135 or r['load'] >= 850
               for r in [first, row]):
            raise RuntimeError(f'ID {motor_id}: unsafe voltage or load during confirmation')
        consecutive_hot = consecutive_hot + 1 if row['temp'] >= limit else 0
        temperatures = [r['temp'] for r in confirmations]
        if consecutive_hot >= 3 or time.monotonic() > deadline:
            break
        tail = temperatures[-3:]
        if len(tail) == 3 and max(tail) < limit and max(tail)-min(tail) <= 2:
            result = row
            result['confirmed_temperature_outlier'] = first['temp']
            result['temperature_confirmations'] = temperatures
            return result
    raise RuntimeError(
        f'ID {motor_id}: temperature anomaly not cleared: '
        f'{first["temp"]}, confirmations={[r["temp"] for r in confirmations]}'
    )
