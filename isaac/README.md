# Stage 3 · Isaac Sim

Target machine (confirmed 2026-10-02): HP Z2, RTX 4000 Ada 20 GB, driver 595.84, Ubuntu 24.04.4, **Isaac Sim 6.0.1**
(standalone at `~/isaacsim`), ROS 2 Jazzy, ROS 2 bridge 5.1.2 using the system `rclpy`.

## Plan

| Step | What | Question it answers |
|---|---|---|
| **3a (now)** | Render the simulated office from the drone's viewpoints with a ZED-like camera (1280×720, ~110° FOV, pitched 15° down). No flight dynamics yet. | Does the line-of-sight visibility model behind Stages 1–2 match what a rendered camera actually sees? |
| 3b | Real perception on rendered frames: open-vocabulary detector + depth → 3D objects → IDs → DINOv2 re-identification. Publish over ROS 2. | Do the Stage 2 conclusions hold with real perception errors and latency? |
| 3c | PX4 flight via Pegasus Simulator (once it supports Isaac Sim 6.0), real ZED extension if compatible. | Same, with real flight dynamics. |

**Pre-registered rule for 3a:** if the render and the model agree on fewer than **80 %** of (viewpoint, object) pairs, the
Stage 1–2 visibility model is revised and Stage 2 is re-run before anything is built on it.

## Run 3a on the Z2

All three scripts run with Isaac Sim's Python, so nothing else needs installing.

```bash
# 0. get the code
git clone https://github.com/kiran-11013/Rov.git ~/rov      # or: cd ~/rov && git pull
cd ~/rov && git checkout claude/robotics-paper-audit-9h35eg
powerprofilesctl set performance                            # avoid the CPU power-save warning

# 1. export the scene (geometry, objects at 11:00 on a test weekday, 90 viewpoints, model visibility)
~/isaacsim/python.sh isaac/export_scene.py --layout booth --out isaac/out/scene_spec.json

# 2. quick check: 8 frames with RGB saved, then look at isaac/out/frames_test/rgb_*.png
~/isaacsim/python.sh isaac/capture.py --spec isaac/out/scene_spec.json --out isaac/out/frames_test \
    --headless --max-frames 8 --rgb-every 1

# 3. full capture: 360 frames (a few minutes)
~/isaacsim/python.sh isaac/capture.py --spec isaac/out/scene_spec.json --out isaac/out/frames --headless

# 4. compare render vs model
~/isaacsim/python.sh isaac/analyze_visibility.py --spec isaac/out/scene_spec.json --frames isaac/out/frames
```

Step 4 writes `isaac/out/frames/VISIBILITY.md`. Send that file back (or paste it), plus 2–3 of the RGB images from step 2.

### What to check in the step-2 images
- Walls, desks, cubicle walls and coloured proxy objects are visible and lit (not black).
- The view is level and looks forward and slightly down. If it points at the ceiling or the floor, the camera
  rotation convention differs in your version; send a screenshot.
- The log line `frame 8/8 ...: N objects visible` shows N > 0. If it is 0 everywhere, the instance-segmentation
  id→prim map uses a different key in 6.0; run step 2 again with `--save-seg` and send one `seg_00000.npz`.

## Notes
- The ROS 2 topics `/robot_1…3` and `/coop_robot_team` on this machine come from another project. From step 3b on we
  use `export ROS_DOMAIN_ID=42` in every terminal to keep our topics separate.
- IOMMU warning: harmless with a single NVIDIA GPU. If images show corruption or Isaac Sim crashes randomly,
  disable VT-d in the BIOS.
- Objects are simple proxies (boxes and cylinders sized like chairs, mugs, laptops…). Real assets come in 3b, when
  a detector has to recognise them.

## Stage 3b (replay): watch the drone fly an episode

A drone built from plain USD shapes replays one command exactly as the simulator flew it. This is kinematic,
with no flight physics yet; PX4 comes in Stage 3c. It shows trust-then-search, ours at 0.4 m only, and ours.
Each video frame has two panels:
- **left:** the room from above, with the trail, the stale map position and every look;
- **right:** the onboard ZED-like camera, with the commanded object tinted green whenever it is in view.

```bash
cd ~/rov && git pull
sudo apt install -y ffmpeg                                   # once, for the .mp4 files
python3 isaac/export_episode.py --layout open --out isaac/out/episode.json
~/isaacsim/python.sh isaac/fly_episode.py --episode isaac/out/episode.json --out isaac/out/fly --headless --max-frames 20   # quick test
~/isaacsim/python.sh isaac/fly_episode.py --episode isaac/out/episode.json --out isaac/out/fly --headless
```

`export_episode.py` picks a command where the object moved and "ours" gains the most, so it is an illustration,
not evidence. The statistics are in `sim/results/`. Use `--cmd N` to pick a specific command, and `--speed`
and `--fps` on `fly_episode.py` to change playback speed.
