#!/usr/bin/env bash
# Installs Intel's official librealsense udev rules (needed for the D455 IMU: accel/gyro).
# Needs sudo; run it yourself, then replug the camera. Your user must be in the plugdev group.
set -euo pipefail
dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
sudo install -m 0644 "$dir/99-realsense-libusb.rules" /etc/udev/rules.d/99-realsense-libusb.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
printf '%s\n' 'Rules installed. Unplug and replug the RealSense, then restart ./run.sh (IMU appears in the dataset).'
