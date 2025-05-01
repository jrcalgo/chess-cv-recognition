import os
from pathlib import Path

import pygame

from chess.cv_chess_game2 import RealtimeChessCV
from utils.configuration import load_json_config

this_file = Path(__file__).resolve()
project_root = this_file.parent.parent
YOLO_MODEL_DIR = os.path.join(project_root, 'models')
CONFIG_PATH = os.path.join(this_file.parent, 'config.json')


def main():
    config = load_json_config(CONFIG_PATH)

    yolo_model_path = os.path.join(YOLO_MODEL_DIR, config['cv']['yolo_model'])
    video_capture_device = int(config['cv']['video_capture_device'])
    stockfish_exe_path = str(config['cv']['stockfish_exe_path'])
    bounding_box_bottom_ratio = float(config['cv']['bounding_box_bottom_ratio'])

    white_minutes = int(config['game']['white_minutes'])
    black_minutes = int(config['game']['black_minutes'])
    stockfish_elo = int(config['game']['stockfish_elo'])

    pygame.init()
    game = RealtimeChessCV(yolo_model_path, video_capture_device, stockfish_exe_path,
                           bounding_box_bottom_ratio, white_minutes, black_minutes, stockfish_elo)
    game.run()


if __name__ == "__main__":
    main()
