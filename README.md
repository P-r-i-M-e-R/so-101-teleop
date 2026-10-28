# SO-101 teleop recorder (LeRobot dataset, for pi0.5)

Web UI: two live cameras, Connect / Start teleop / Record, writes a **LeRobotDataset v3** directly
(`lerobot` 0.6, videos encoded live). Control logic and safety checks come from
`bundle/` (your SO101_teleop_export_2026-10-23, unchanged copies).

## Run
```bash
./run.sh            # then open http://127.0.0.1:8765
```
Python deps live only in `.venv` (`pip install -r requirements.txt` inside it).

## One-time USB access (needs sudo, not done automatically)
```bash
bash bundle/install_usb_access.sh      # then replug both USB controllers
```
(or the temporary `setfacl` commands in `bundle/README_RU.md`).

## Workflow
1. **Connect** – read-only preflight (models, IDs, offsets, limits, torque, temps). No motor moves.
   Tick *follower already holding* if the follower still has torque from a previous run.
2. Fill task prompt / dataset name → **Apply & open dataset** (resumes an existing dataset with the same name).
3. **Start teleop…** → confirm. Follower aligns slowly to the leader, then follows ("POSE ALIGNED").
4. **Space** = start / stop+save episode, **Backspace** = discard, **Esc** = stop teleop (hold).
   Teleop sessions are limited to 300 s (bundle rule); an episode in progress is saved when it ends,
   discarded if teleop ended by an error. Re-start teleop for the next session.

## Recorded features (per frame, `fps` = 30)
| key | content |
|---|---|
| `observation.state` | follower joints, degrees about centre (2048 cnt); gripper 0–100 % of calibrated range |
| `action` | leader joints, same units |
| `observation.images.front` / `.wrist` | 640x480 RGB video (front = RealSense colour, wrist = USB cam) |
| `observation.images.front_depth` | **native RealSense depth**, 848x480, unaligned, uint16 -> returned in millimetres (lossless-quantised HEVC) |
| `observation.front.imu` | accel+gyro (only if `realsense_udev/install_realsense_access.sh` was run) |
| `timing.camera_frame_id`, `timing.camera_age_ms`, `timing.unix_time` | which camera frame each row used (detect duplicates), its age, wall clock |
| `observation.follower.raw_position`, `action.leader.raw_position` | raw encoder counts |
| `observation.follower.{velocity,load,current,temperature,voltage,moving}` | per-motor telemetry |
| `observation.leader.{velocity,load,current,temperature,voltage}` | per-motor telemetry |
| `action.goal_sent` | clamped goal actually written to the follower |
| `timing.read_ms`, `timing.camera_age_ms` | loop read time, age of each camera frame |

Extras can be switched off in the UI (then only the standard state/action/images are written).
Camera names / devices: `config.json` (`device` = `realsense`, `realsense:<serial>` or a `/dev/v4l/by-id/...` path).

## Train
Point lerobot at the dataset folder, e.g. `--dataset.repo_id=local/<name> --dataset.root=<dataset_dir>/<name>`
with `--policy.type=pi05`. Use the same task prompt at inference.

## Notes
- Stopping never cuts power: the follower keeps holding. Use *Disable follower torque…* only with the arm supported.
- If the RealSense hangs, **Restart cameras** (the app also hardware-resets it on a failed start).

## Depth, calibration, frame accounting
- Depth is shown live in the UI (colour-mapped preview) and saved as its own video stream per episode.
- `<dataset>/camera_info.json`: colour/depth intrinsics, depth->colour extrinsics, depth scale, firmware.
  `<dataset>/robot_calibration.json`: both arms' calibration. Align depth to colour / make point clouds offline.
- The UI shows duplicate / skipped camera frames live; `alerts_per_episode.jsonl` stores them per episode.
- After every saved episode the dataset is finalized and re-opened, so it is always loadable
  (LeRobot v3 keeps all episodes of a camera in one `file-000.mp4`; that is normal).
- IMU needs one sudo step: `bash realsense_udev/install_realsense_access.sh`, then replug the camera.
- An existing dataset without depth cannot be extended; use a new dataset name.
- Normal players cannot open LeRobot's depth video (12-bit HEVC, log-quantised). For viewing, every saved episode also gets
  a playable colour-mapped depth video: `<dataset>/depth_preview/episode_000000_front_depth.mp4` (H.264, 0-6.3 m).
  Training uses the raw depth stored in the dataset, not these previews.

## Error tracing and protection against broken episodes
- Everything is logged with full tracebacks to `logs/so101.log` (also: UI button **Show log**). Uncaught thread errors too.
- Per dataset: `episode_failures.jsonl` (why an episode was discarded/truncated), `alerts_per_episode.jsonl` (warnings,
  frame duplicates/skips per episode).
- A garbage temperature reading (single servo returning e.g. 33,42,57,52,33) no longer stops teleop; consistently high
  readings (real overheating) still do. A control-loop stall only stops teleop after 1.5 s (was 0.4 s).
- Recording refuses to start when: a camera/depth stream is down, a critical alert is active, a follower motor is >= 43 C,
  the serial bus is retrying, the encoder is still catching up, or < 5 GB free.
- A failing depth-preview video never affects the episode; every save is verified against the files on disk.
- Optional: keep a truncated episode when teleop stops on an error (UI checkbox).
