# Train a tiny model, replace slow MARS

## Short answer (read this first)

**Best plan: train a small, fast model that copies MARS, then run it instead of MARS.** The best first choice is a small YOLO pose model ("YOLO26n-pose"). It finds both mice and all 14 points in one step. The second choice is a "top-down" model called RTMPose. Top-down means: first find each mouse, then cut out a small picture of each mouse, then find the points in that picture.

No one has published a speed test for two mice from above on your laptop. So every speed number for your computers below is a **guess from other tests**. I tell you which is which.

The guesses are good news. A small model of this type is about 10 to 50 times faster than 0.85 s per frame. The small YOLO model needs about 0.04 s on a slow server CPU for a big 640 pixel picture, so the first test is close to your 0.033 s limit ([Ultralytics pose docs](https://docs.ultralytics.com/tasks/pose/)). Smaller pictures and Intel's OpenVINO tool should make it faster (my guess). On the M5 Mac, the Neural Engine should give a few milliseconds (my guess).

A pretrained mouse model that is fast does **not** exist. The one pretrained top-view mouse model I found (DeepLabCut SuperAnimal) is heavy, it already failed on your video, and it is for non-commercial use only ([model card](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)). But **you do have data to train on**. There are 15,000 hand-labeled mouse pictures from the MARS team, free for non-commercial use ([Caltech record 2011](https://data.caltech.edu/records/2011)).

What to do first: train the small model on a few thousand pictures. Then measure speed on the laptop. The test that counts: **30 frames per second or more (33 ms or less per frame) for the whole step.** If the laptop cannot do it, the M5 Mac probably can. If not, there is a fallback (Part 6).

Note: the M5 MacBook Air is not a new product in October 2026. Apple announced it in March 2026 ([Gigazine](https://gigazine.net/gsc_news/en/20260304-apple-m5-macbook-air)). Your date is just your delivery day.

---

## Part 1. Why MARS is slow, and why a new model fixes it

MARS runs two kinds of networks. First, two "detectors" find the boxes around the mice. Then a big network called a "stacked hourglass" finds the points. From your own tests, the detectors take about 0.32 s and the points step takes about 0.54 s.

MARS itself is not fast, even for its authors. Their paper says it ran at **about 11 frames per second** on a strong 8-core computer with a Titan XP graphics card ([MARS paper](https://elifesciences.org/articles/63720)). So 30 fps was never the goal of MARS.

The pose network has **8 hourglass stacks** and works on 256 by 256 pixel pictures ([MARS_Developer config](https://github.com/neuroethology/MARS_Developer/blob/master/_template/pose/config_train.yaml)). That is much more work than a small modern model needs.

We could also try to speed up MARS itself. This gives a smaller gain. My estimate from the notes: skip the detectors on most frames, stop at stack 4 of 8, run pose every second frame, and use Intel OpenVINO. That gives about **100 to 200 ms per frame, which is 5 to 10 fps**. This is an estimate, not a test. It is not enough for 30 fps. It is only a good "step 0" or fallback. One more thing: MARS's own CoreML code checks for macOS 10.15, and new Macs say "26", so on the M5 it would quietly use the slow path ([MARS_pose_machinery.py](https://github.com/neuroethology/MARS/blob/master/mars_v1_8/MARS_pose_machinery.py)).

---

## Part 2. The options, compared

| Option | What it is | Speed facts (cited) | Speed guess for you | License |
|---|---|---|---|---|
| **YOLO26n-pose** (Ultralytics) | One network, finds both mice and all points | 40.3 ms on a 2 GHz server CPU, 640 px picture. 2.9 million parameters ([docs](https://docs.ultralytics.com/tasks/pose/)). On iPhone 17 Pro: 12 ms CPU, 4 ms with Neural Engine ([CoreML docs](https://docs.ultralytics.com/integrations/coreml/)) | Laptop about 15 to 25 ms with OpenVINO and a wide, short picture. M5: 5 to 15 ms | AGPL-3.0 |
| **RTMPose-t / s** + a box finder | Top-down. Two small crops | 3.2 ms (t) and 4.5 ms (s) per crop on a fast 8-core desktop i7-11700 ([MMPose](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/README.md)) | Your 4-core laptop is maybe 2 to 3 times slower. Two crops: about 16 to 26 ms. Box finder adds time | Apache-2.0 (rtmlib) |
| **sleap-nn** (SLEAP) | Full toolkit with training | No CPU speed found. CPU is "much slower"; Mac uses MPS (Apple GPU) ([FAQ](https://nn.sleap.ai/latest/help/faq/)) | Unknown | BSD-3 |
| **DeepLabCut 3** | Full toolkit | No mouse CPU speed found. Older small model: 15 to 19 fps at 459 by 349 px on a fast CPU, one animal ([DLC-Live](https://elifesciences.org/articles/61909)) | Unknown | LGPL-3.0 (SuperAnimal weights: non-commercial) |
| **DeepLabCut SuperAnimal** (pretrained) | Ready mouse model, 27 points | Heavy (HRNet-w32). 5.8 s per frame on your video, and it did poorly (your earlier test; the notes say "31/150 frames", I am not sure what that count means) | Too slow | Non-commercial only |
| **Faster MARS** | Same MARS, tricks | MARS: 11 Hz on a Titan XP GPU | 5 to 10 fps (guess) | Code: no license file found |

Other tools do not fit. MediaPipe and MoveNet are for people and cannot be retrained for mice, based on forum replies (unofficial) ([Google forum](https://discuss.ai.google.dev/t/transfer-learning-using-mediapipe-model-maker-for-blazeface-face-detector/167371)). YOLO-NAS-pose is not maintained and its weights are non-commercial ([Deci docs](https://docs.deci.ai/super-gradients/YOLONAS-POSE.html)). ViTPose and Sapiens are far too big for a CPU (general knowledge, no source). Lightning Pose needs an NVIDIA GPU on Linux ([GitHub](https://github.com/paninski-lab/lightning-pose)).

**Why I pick YOLO26n-pose first:**
1. One model, one step, one export command. Easy to build.
2. It works for two mice at once. The cost does not grow with the number of mice (my reasoning).
3. It can export to OpenVINO for Intel and CoreML for Mac with one line each ([OpenVINO docs](https://docs.ultralytics.com/integrations/openvino/), [CoreML docs](https://docs.ultralytics.com/integrations/coreml/)).
4. SLEAP's own docs say bottom-up (one-step) models are better when animals touch often ([nn.sleap.ai](https://nn.sleap.ai/latest/reference/models/)). Fighting mice touch a lot.

**Why RTMPose is the second choice:** it has the best published CPU speed and a free license. But it needs a box finder. The box finder takes 8 to 10 ms in the human pipeline ([RTMPose README](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/README.md)). Training it also needs the MMPose tool set, which may be hard on Windows with Python 3.12 (not checked).

**A cheap box finder (my idea, not tested):** one mouse is black and one is white. Simple thresholding (dark blob and bright blob) could find the boxes in about 1 ms. But when the mice touch, the blobs join. That is exactly when fights happen. So use it only as a helper, not as the only finder.

---

## Part 3. Speed: laptop and M5

### What is known (cited)

1. YOLO26n-pose: 40.3 ms on a 2.0 GHz Xeon server CPU with ONNX, at 640 px ([Ultralytics](https://docs.ultralytics.com/tasks/pose/)).
2. On new Intel Core Ultra chips, OpenVINO ran YOLO26n **detection** 3 to 7 times faster than normal PyTorch CPU (for example 29.28 ms to 4.09 ms) ([OpenVINO docs](https://docs.ultralytics.com/integrations/openvino/)). This is detection, not pose, and not your chip.
3. INT8 (smaller numbers) helped little on the newest Intel chips and lost a little accuracy (mAP 0.4762 to 0.4634) ([same page](https://docs.ultralytics.com/integrations/openvino/)). Skip INT8 at first.
4. iPhone 17 Pro: 12.0 ms CPU, 4.0 ms CPU plus Neural Engine for YOLO26n-pose at 640 px ([CoreML docs](https://docs.ultralytics.com/integrations/coreml/)). That is a burst average; long use is slower.
5. A MacBook Air M3 ran YOLO11n (detection) in about 15.6 ms with CoreML ([Elixir benchmark](https://hexdocs.pm/yolo/macbook_air_m3.html)).
6. The M5 has a 16-core Neural Engine and GPU "Neural Accelerators". Apple says over 4 times the peak GPU AI compute of the M4 ([Apple](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/)). I found **no independent M5 test for any pose model**.

### What is guessed (not cited as fact)

1. Your i7-1165G7 is probably close to the 2 GHz Xeon. So the small YOLO model at 640 by 640 may take about 40 ms. That is a bit too slow.
2. A wide picture such as 640 by 384 has about 0.6 times the pixels. With OpenVINO this may give **15 to 25 ms**. This is my scaling from FLOPs (work amount). No one measured it.
3. The M5 should do the small model in about 5 to 15 ms. The iPhone result supports this, but a phone chip is not an M5.
4. The Air has no fan. After many minutes it may slow down. I did not find data.

**Honest summary:** 30 fps looks possible on the M5 and possible, but not sure, on the laptop. You must test.

---

## Part 4. Accuracy, data and legal rules

### Accuracy

A student model that copies MARS can at best be as good as MARS. If the student learns from MARS output, it also learns MARS's mistakes. To avoid this, train on **hand-labeled** data (below).

I found **no published accuracy numbers** for a small YOLO or RTMPose model on 7-point mouse data. I found no paper that shows "copy MARS into a small model" with results either. So accuracy for black and white mice is unknown until you measure it.

What helps: the hand-labeled MARS set has the same look as your video: black resident, white intruder, top view, 1024 by 570, 30 fps ([Caltech record 2011](https://data.caltech.edu/records/2011)). I did not check how many frames show each mouse color.

How to measure: use the 7 points as test data. Compare the new model with MARS on CalMS21 frames. Count the share of points within a few pixels.

### Data you can use

| Data | What it has | License | Use it? |
|---|---|---|---|
| **MARS pose annotation data** (record 2011) | 15,000 hand-labeled top pictures (1024x570). 5 workers per point, middle value used. 1.5 GB for top pictures. 9 top points, 7 are the MARS ones | CC BY-NC 4.0 (non-commercial) | **Yes. Best data.** |
| **CalMS21** (record 1991) | About 1 million frames with MARS points, plus videos (28.3 GB for task 1). The points are MARS output, not hand-made | CC BY 4.0 on the record. The paper may say CC BY-NC-SA (I did not read it) | Yes, as extra "pseudo-labels". Use the stricter license |
| **TopViewMouse-5K** | 26-point set, many mouse looks | Academic, non-commercial | Maybe later. Points need mapping |
| **MABe 2022** | 3 mice, 12 points, different arena | License not found | Skip at first |

Sources: [record 2011](https://data.caltech.edu/records/2011), [CalMS21 readme](https://data.caltech.edu/records/s0vdx-0k302/preview/readme.md), [record 1991](https://data.caltech.edu/records/1991), [Zenodo 10618947](https://zenodo.org/records/10618947), [AIcrowd MABe22](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022/problems/mabe-2022-mouse-triplets).

How many pictures to use (guess): 2,000 to 10,000, including touching-mouse frames. Toolkit guides say 200 to 500 pictures is "good" for one case and 1000+ for hard cases ([sleap-nn FAQ](https://nn.sleap.ai/latest/help/faq/)). Our case has two similar mice and fights, so use more. Nobody tested the exact number.

### Legal rules for a public GitHub repo (not legal advice)

1. **Do not put the images or labels in your repo.** The data is non-commercial. Write download steps in the README instead. (My advice.)
2. **Put your trained weights in the repo only after you think about it.** I found no text that says if weights trained on non-commercial data are a "derived work". This is a gray area. For a class project it is low risk, but say so in the README. (Inference.)
3. **YOLO (Ultralytics) uses AGPL-3.0.** Ultralytics says this fits coursework and research if your project is open ([license page](https://www.ultralytics.com/es/license)). A public repo is fine. A closed product would need a paid license. Check the LICENSE file yourself.
4. **RTMPose through rtmlib is Apache-2.0** ([rtmlib](https://github.com/Tau-J/rtmlib)). It is the most free choice.
5. **MARS code has no LICENSE file** that I could find ([MARS_Developer](https://github.com/neuroethology/MARS_Developer)). The MARS models are CC BY-NC 4.0 (from your own log). Keep attribution.

---

## Part 5. Setup effort and tool risks

| Step | Effort (my guess) | Risk |
|---|---|---|
| Install `ultralytics` | Low. One pip command. Works on Windows and Mac | Low |
| Make data in YOLO format (7 points per mouse, 2 mice) | Medium. Need a small script. Points have a visibility flag | Left/right ear and hip swaps. The raw MARS labels are not fixed for left/right swaps ([record 2011](https://data.caltech.edu/records/2011)). Use the median file |
| Train | Medium. No training time found for any hardware | Unknown time. Use a free Colab GPU or a Mac GPU (MPS) if needed |
| Export (OpenVINO, CoreML) | Low. One line each | CoreML needs macOS to run. The iOS 27.0.1 cpuOnly crash was only for large pose models ([docs](https://docs.ultralytics.com/integrations/coreml/)) |
| Replace MARS in the app | Medium. The app expects MARS output (7 points per mouse, mouse 0 black, mouse 1 white) | Mouse ID. Use brightness to keep black/white identity (my idea) |
| Camera on Mac | Low | OpenCV camera permission belongs to Terminal or the IDE. Set it in System Settings, Privacy and Security, Camera ([OpenCV forum](https://forum.opencv.org/t/opencv-not-authorized-to-capture-video-status-0-requesting/8512)) |

**Avoid TensorFlow on the Mac.** `tensorflow-metal` is fragile. It works with TF 2.18.x and fails above 2.18.1 according to forum reports ([Apple forums](https://developer.apple.com/forums/thread/803658)). A PyTorch or CoreML model avoids this.

---

## Part 6. Recommendation and step-by-step plan

### Recommendation

Train **YOLO26n-pose** (or YOLO11n-pose) on the hand-labeled MARS data, plus some CalMS21 pseudo-labels. Export to OpenVINO for the laptop and to CoreML for the M5. Keep MARS as the "teacher" and the "check". Keep RTMPose as the second choice.

### Plan

1. **Today, measure what you have.** Run the speed test with a pretrained small YOLO pose model (Test 1 below). This tells you the speed floor on your laptop in about 10 minutes.
2. **Get the data.** Download `raw_images_top.zip` (1.5 GB) and `MARS_keypoints_top.json` (21.6 MB) from record 2011. Keep them outside the repo.
3. **Convert labels.** Write a script that makes YOLO files from the JSON: use the 7 points, make one box per mouse from the points (add a margin), and set the mouse class or order. Check some pictures by drawing the points.
4. **Train a small model.** Start from the COCO-pretrained weights. Use a wide input like 640 by 384. Train on Colab or any GPU. Hold out 10 to 15 percent of the pictures as a test set.
5. **Measure accuracy.** Compare with MARS on the test pictures and on CalMS21 frames. Also look at touching-mouse frames by eye.
6. **Export and measure speed** on the laptop with OpenVINO (Test 2). Later, on the M5 with CoreML (Test 3).
7. **Plug into the live app.** Keep the same output shape as MARS. Keep MARS as a switch (old path) for comparison.
8. **If accuracy is weak,** add CalMS21 pseudo-labels, then hand-fix a few hundred of your own frames (your camera-filming-an-iPad setup differs from the original).
9. **Update `SESSION_LOG.md` and save each small step to GitHub.**

### Fallback if 30 fps is not reached

Pick the first one that works:
1. **Run pose on every 2nd frame** and fill the gap by interpolation (guess: little loss, except in very fast moves). This halves the work.
2. **Use a smaller picture** (for example 480 by 288), then check accuracy.
3. **Use the M5 only for live,** and keep the laptop for offline analysis.
4. **Switch to the top-down RTMPose-t** with boxes from the previous frame's points, and refresh boxes every few frames. DeepLabCut-Live used the same idea and got 75 percent more speed with small error when the margin was 25 to 50 px ([DLC-Live](https://elifesciences.org/articles/61909.pdf)).
5. **Last fallback: faster MARS** (skip detectors, fewer stacks, OpenVINO). My estimate is 5 to 10 fps, which is "near live" at best. For live alerts, a delay of a second or two with smoothing may still be fine.

---

## Part 7. Gaps and what to test first

### Gaps (things I could not find)

1. No speed test of any pose model on an i7-1165G7 or Intel Iris Xe.
2. No independent speed test of any pose model on an M-series Mac, and none for the M5.
3. No published accuracy for a small YOLO or RTMPose model on 7-point two-mouse data.
4. No paper showing "copy MARS into a small model" with numbers.
5. No training times for any hardware.
6. CalMS21 license conflict (CC BY vs CC BY-NC-SA). I did not read the paper text.
7. No statement about whether weights trained on non-commercial data are free to share.
8. M5 Air long-run slowdown (no fan) not researched.
9. Python 3.12 support for MMPose/RTMPose training not checked.
10. Several papers (SLEAP, SuperAnimal) were blocked, so some numbers are missing.

### Tests, in order

| # | Test | Command or action | Success means |
|---|---|---|---|
| 1 | Laptop speed floor, no training | `pip install ultralytics openvino`, then `yolo export model=yolo26n-pose.pt format=openvino imgsz=384,640` and time 200 frames with `yolo predict model=yolo26n-pose_openvino_model source=<your video>` (use the speed lines it prints) | **33 ms or less total per frame** (pre-process + model + post-process). Under 25 ms is comfortable |
| 2 | Laptop speed of the trained model | Same export on the trained weights, run on your 1024x570 video, time 500 frames | **33 ms or less** including camera reading and drawing |
| 3 | M5 speed | On the Mac: `yolo export model=<trained>.pt format=coreml nms=True imgsz=384,640`, then predict on the same video for 5 minutes | **33 ms or less, still true after 5 to 10 minutes** (to check slowdown). Under 15 ms is good |
| 4 | Accuracy vs MARS | Run both on 500 held-out frames. Compute the share of points within 5 px of the hand label (or of MARS) | New model **within a few percent of MARS** on the points. Also no more mouse-ID swaps on touching frames |
| 5 | Wide vs square picture | Repeat Test 1 with `imgsz=640` and `imgsz=384,640` | The wide one is clearly faster (about 30 to 40 percent) |
| 6 | Cheap box finder | Threshold the video (dark blob, bright blob) and count frames with 2 clear blobs | At least 90 percent of frames have 2 clean boxes, and time is 2 ms or less. If blobs merge in many fight frames, do not use it alone |
| 7 | Convert MARS (fallback only) | Try `ov.convert_model("<mars_pose>.pb", input=[2,256,256,3])` (OpenVINO), then `benchmark_app -d CPU` and `-d GPU` | Pose step under 200 ms for two crops. This only matters for the fallback |

---

## Final thought

The big discovery is that you probably do **not** need to make MARS faster. You can replace it with a small model that learns from the same public data MARS used. That data is large, hand-labeled, and free for school work. The one real risk is not speed, it is **accuracy when the two mice touch**. So test touching frames early, and keep MARS next to the new model as a judge.

---

## Appendix: Test 1, measured on this laptop (2026-10-25)

This is the first real measurement. It is **speed only**. The model file is the small YOLO pose model that knows people (17 keypoints), not mice. A mouse-trained model has the same network shape (only 7 keypoints, so a little less work), so it should run at about the same speed. We have **no accuracy result** for mice yet.

Setup: Intel i7-1165G7 laptop (4 cores, Intel Iris Xe graphics, no NVIDIA), Python 3.12, ultralytics 8.4.189, OpenVINO 2026.4.1, `yolo26n-pose` (2.9 million parameters). Input: 110 frames of the 1024 x 570 demo video; the first 10 frames are warm-up; the numbers are medians. The time includes preparing the picture, the model, and the post-processing.

| Test | Time per frame | Frames per second |
|---|---|---|
| PyTorch on the CPU, 384 x 640 | 64.4 ms | about 15 |
| PyTorch on the CPU, 640 x 640 | 62.5 ms | about 16 |
| OpenVINO on the CPU, 384 x 640 | 61.1 ms | about 16 |
| OpenVINO on the CPU, 640 x 640 | 182.8 ms (not understood; the export folder was reused, so ignore this number) | about 5 |
| **OpenVINO on the Intel Iris Xe graphics chip, 384 x 640** | **16.5 ms** | **about 60** |

**Sustained run (4 minutes, Iris Xe, 384 x 640):** 14,258 frames, **average 59 frames per second**. The median stayed between 14.8 and 18.4 ms in every 10-second window. In each window, 99% of the frames took under 20 to 36 ms. There was no big slowdown in 4 minutes. A longer run was not tested.

What this means:
1. The target is 33 ms or less per frame. The Iris Xe graphics chip meets it with about two times to spare, **for this one step**. The CPU alone does not (about 61 to 64 ms).
2. The rest of the live program (reading the camera, Model B scoring, the monitor, the page) also needs time. Nobody measured the total yet.
3. The unknown that matters most is **accuracy after training on mice**, especially when the two mice touch.

Scripts: `scripts/bench_yolo_pose_speed.py` and `scripts/bench_yolo_pose_sustained.py` (run them in an environment with `ultralytics` and `openvino`).
