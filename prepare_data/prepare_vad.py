import argparse
import shutil
import os
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vad_path", type=str, default="data/vad", help="Path to the VAD dataset")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    train_path = os.path.join(args.vad_path, "train/bad")
    low_shot_train_fn = os.path.join(args.vad_path, "low_shot_train.json")

    with open(low_shot_train_fn, "r") as file:
        low_shot_train = json.load(file)

    for key, filenames in low_shot_train.items():
        new_path = os.path.join(args.vad_path, "low_shot_train", str(key), "bad")
        os.makedirs(new_path, exist_ok=True)
        for fn in filenames:
            shutil.copy(os.path.join(train_path, fn), os.path.join(new_path, fn))
