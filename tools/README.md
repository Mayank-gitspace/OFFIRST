# tools

- `user_test/`: quick ad-hoc runs of the v3 (`hires1440`) segmenter on single images (`run.py` on a user image, `run55.py` on IDRiD_55 with crops and overlay, `runh.py` on a healthy eye). Run from inside the `lesion_detector` code folder (`lesion_segmenter/code/`) with the checkpoint at `kaggle_runs/hires1440/ckpt/best.pth`.
- `quality_guard_poc/`: earlier proof of concept for an image-quality guard (EyeQ quality classifier, synthetic degradations, INT8 export, small app). See its README. No images included.

Paths are relative to the project root; no dataset images are stored in this repo.
