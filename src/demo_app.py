import os
from pathlib import Path

import pygame

from chess.cv_chess_game2 import RealtimeChessCV
from utils.configuration import load_json_config

this_file = Path(__file__).resolve()
project_root = this_file.parent.parent
YOLO_MODEL_DIR = os.path.join(project_root, 'models')
CONFIG_PATH = os.path.join(this_file.parent, 'config.json')


def main(stats=None):
    config = load_json_config(CONFIG_PATH)
    yolo_model_path = os.path.join(YOLO_MODEL_DIR, config['cv']['yolo_model'])
    video_capture_device = config['cv']['video_capture_device']
    capture_orientation = str(config['cv']['capture_orientation']).lower()

    pygame.init()
    game = RealtimeChessCV(yolo_model_path, video_capture_device)
    game.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=bool, default=False)
    args = parser.parse_args()
    stats = args.stats

    main(stats=args.stats)
