# chess-cv-recognition


## 1. Prerequisites
 - Python 3.12
 - pip/conda
 - A USB webcam (Tested on Logitech C920; other cams may require more tweaking)


## 2. Git Clone
```bash
git clone https://github.com/jrcalgo/chess-cv-recognition.git
cd chess-cv-recognition
```


## 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```


## 4. Stockfish Engine
1. Downlioad the Stockfish binary for your platform from the [Stockfish website](https://stockfishchess.org/download/)
2. Unzip it somewhere (best in chess-cv-recognition directory)
3. Open `src/config.json` and set `stockfish_exe_path`:
```jsonc
    "cv": {
        "_comment": "Use absolute path to stockfish exe",
        "stockfish_exe_path": "<absolute_path_to_stockfish_exe>",
    }
```
`yolo_model` in `src/config.json` has two options:
 - `best_v1.pt`: was trained on ~3k images (overfitted)
 - `best_v2.pt`: was trained on a a larger, cumulative ~30k image dataset with use of data augmentation


## 5. Verify Model Files
Make sure your `chess-cv-recognition/models/` directory contains:
 - best_v1.pt
 - best_v2.pt


## 6. Run the Game
From the `chess-cv-recognition/src/` directory, run:
```bash
python demo_app.py
```
Or, if you're importing:
```python
from demo_app import main
main()
```
This initialization will:
1. Loads `src/config.json` keys and values
2. Spawns a opencv calibration window, and then
3. Spawns Pygame and opencv windows for gameplay


## Additional Notes
 - Gameplay configuration: `game` in `src/config.json` stores values for adjusting white's time (human), black's time (Stockfish), and Stockfish's ELO rating. Adjust as needed.
 - Camera Assignment: `cv` in `src/config.json` has a `video_capture_device` parameter for choosing the active camera used by opencv.
 - Camera limitations: Expect to tweak `bounding_box_bottom_ratio` in `src/config.json` in relation to the angle of your capture device. The current value .95 is tested and works well with most angles.
 - Performance barriers: Ultralytics YOLO + Stockfish move inference is resource intensive; consider closing other apps if necessary.
