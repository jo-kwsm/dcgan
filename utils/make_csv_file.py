import argparse
import glob, json, os, sys
from typing import Dict, List

import pandas as pd

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="make csv files for GANImg dataset"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset/img78/",
        help="path to a dataset dirctory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./csv",
        help="a directory where csv files will be saved",
    )

    return parser.parse_args()

def main() -> None:
    args = get_arguments()

    data: Dict[str, List[str]] = {
        "image_path": [],
    }

    for idx in range(200):
        img_path = os.path.join(args.dataset_dir, "img_%s_%s.jpg")
        data["image_path"].append(img_path % (7, idx))
        data["image_path"].append(img_path % (8, idx))

    # list を DataFrame に変換
    df = pd.DataFrame(
        data,
        columns=["image_path"],
    )

    # 保存ディレクトリがなければ，作成
    os.makedirs(args.save_dir, exist_ok=True)

    # 保存
    train_df.to_csv(os.path.join(args.save_dir, "train.csv"), index=None)
    val_df.to_csv(os.path.join(args.save_dir, "val.csv"), index=None)
    test_df.to_csv(os.path.join(args.save_dir, "test.csv"), index=None)

    print("Finished making csv files.")


if __name__ == "__main__":
    main()
