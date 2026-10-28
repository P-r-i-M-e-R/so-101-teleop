#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
sudo install -m 0644 "$bundle_dir/99-so101-teleop.rules" /etc/udev/rules.d/99-so101-teleop.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty
printf '%s\n' 'Rules installed. Reconnect both USB controllers if necessary.'
