"""Validated Feetech requests with bounded retries for read/idempotent motion writes."""
import time


def exchange(port, motor_id, command, parameters, response_length, on_retry=None):
    retryable = command == 2 or (
        command == 3 and parameters and parameters[0] in {40, 41, 42, 44, 46, 48, 55})
    attempts = 3 if retryable else 1
    for attempt in range(attempts):
        port.reset_input_buffer()
        body = [motor_id, len(parameters) + 2, command, *parameters]
        port.write(bytes([255, 255, *body, (~sum(body)) & 255]))
        port.flush()
        response = port.read(response_length)
        valid = (len(response) == response_length
                 and response[:3] == bytes([255, 255, motor_id])
                 and response[3] == response_length - 4
                 and sum(response[2:]) % 256 == 255)
        if valid:
            if response[4] != 0:
                raise RuntimeError(f'ID {motor_id}: hardware error {response[4]}')
            return response[5:-1]
        event = {'id': motor_id, 'command': command, 'attempt': attempt + 1,
                 'response_hex': response.hex(), 'reason': 'invalid or incomplete packet'}
        if on_retry:
            on_retry(event)
        if attempt + 1 < attempts:
            time.sleep(.01)
    raise RuntimeError(f'ID {motor_id}: no valid response after {attempts} attempts; last={response.hex()}')
