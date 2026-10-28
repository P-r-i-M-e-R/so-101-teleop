"""Disable follower torque only after the arm/object are mechanically supported."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--supported',action='store_true')
    a=p.parse_args()
    if not a.supported:p.error('Support the follower and held object, then add --supported')
    import serial
    from serial_protocol import exchange
    root=Path(__file__).resolve().parent
    config=json.loads((root/'teleop_config.json').read_text())
    with serial.Serial(config['follower_port'],config['baudrate'],timeout=.3,exclusive=True) as bus:
        for i in range(1,7):
            exchange(bus,i,3,[40,0],6)
            assert exchange(bus,i,2,[40,1],7)==bytes([0])
    print('Follower torque disabled for all 6 motors.')

if __name__=='__main__':main()
