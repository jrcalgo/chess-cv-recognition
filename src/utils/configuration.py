import json


def load_json_config(config_path: str):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except ():
        return
