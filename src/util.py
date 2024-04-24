import json
import numpy as np


def convert(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {convert(key): convert(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert(item) for item in obj]
    else:
        return obj


def write_json(data: list or dict,
               path: str) -> None:
    print(convert(data))
    with open(path, "w+", encoding="utf8") as outfile:
        json.dump(convert(data), outfile, separators=(",", ":"), indent=4, ensure_ascii=False)


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as json_file:
        return json.load(json_file)
