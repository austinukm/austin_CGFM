import os
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import pathlib
from typing import Tuple, List

def check_normalized(features: np.ndarray) -> bool:
    print(features.shape)
    norms = np.linalg.norm(features, axis=1)
    true_or_false = np.allclose(norms, 1.0)
    print(f"feature_shape: {features.shape} normalized to 1 {true_or_false}")
    return true_or_false

def load_npz_features(npz_path: str):
    """Load npz and return dict-like object (np.load result)."""
    logging.info("Loading features from %s", npz_path)
    data = np.load(npz_path, allow_pickle=True)
    logging.info("Found arrays: %s", data.files)
    return data

def load_ratings(rating_path: str, good_rating: float) -> pd.DataFrame:
    """Load ratings file and add binary label column."""
    logging.info("Loading ratings from %s", rating_path)
    ratings = pd.read_csv(
        rating_path,
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
    ).sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)
    ratings["label"] = (ratings["rating"] >= good_rating).astype(int)
    return ratings

def split_user_ratings(
    ratings: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    train_threshold: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ratings per user into train/val/test. Users with fewer than train_threshold
    ratings are left entirely to train set (original behaviour)."""
    logging.info("Filtering users with at least %d ratings", train_threshold)
    user_count = ratings["user_id"].value_counts()
    active_usrs = user_count[user_count >= train_threshold].index
    ratings = ratings[ratings["user_id"].isin(active_usrs)].sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)

    train_list, val_list, test_list = [], [], []
    for uid, df in tqdm(ratings.groupby("user_id")):
        n = len(df)
        if n < train_threshold:
            train_list.append(df)
        else:
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            train_list.append(df[:train_end])
            val_list.append(df[train_end:val_end])
            test_list.append(df[val_end:])
    keep_cols = ['user_id', 'movie_id', 'rating', 'timestamp', 'label']
    train_df = pd.concat(train_list)[keep_cols].reset_index(drop=True) if train_list else pd.DataFrame(columns=keep_cols)
    val_df = pd.concat(val_list)[keep_cols].reset_index(drop=True) if val_list else pd.DataFrame(columns=keep_cols)
    test_df = pd.concat(test_list)[keep_cols].reset_index(drop=True) if test_list else pd.DataFrame(columns=keep_cols)
    all_ratings = ratings
    return train_df, val_df, test_df, all_ratings

def save_splits(save_dir: pathlib.Path, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,all_df: pd.DataFrame):
    save_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(save_dir / f"train.csv", index=False)
    val_df.to_csv(save_dir / "val.csv", index=False)
    test_df.to_csv(save_dir / "test.csv", index=False)
    all_df.to_csv(save_dir / "all_ratings.csv", index=False)
    logging.info("Saved train/val/test to %s", str(save_dir))

def main(
    img_npz: str,
    text_npz: str,
    rating_path: str,
    save_path: str,
    good_rating: float = 3,
    train_ratio: float = 0.8,
    train_threshold: int = 5,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    # Load and check features
    if img_npz:
        img_data = load_npz_features(img_npz)
        if "features" in img_data:
            check_normalized(img_data["features"])
    if text_npz:
        text_data = load_npz_features(text_npz)
        if "features" in text_data:
            check_normalized(text_data["features"])

    ratings = load_ratings(rating_path, good_rating)

    train_df, val_df, test_df, all_df = split_user_ratings(
        ratings, train_ratio=train_ratio, val_ratio=(1 - train_ratio) / 2, train_threshold=train_threshold
    )

    save_splits(pathlib.Path(save_path), train_df, val_df, test_df, all_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split ratings dataset into train/val/test per user.")
    parser.add_argument("--img-npz", default=r"D:\work_space\work\austin_paper_coding\austin_paper1\feature_output\movie_img_features_20251103_1457.npz")
    parser.add_argument("--text-npz", default=r"D:\work_space\work\austin_paper_coding\austin_paper1\feature_output\movie_text_features_20251103_1457.npz")
    parser.add_argument("--rating-path", default=r"..\data\ratings.dat")
    parser.add_argument("--save-path", default=str(pathlib.Path("../data/train_val_test_dir")))
    parser.add_argument("--good-rating", type=float, default=4.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--train-threshold", type=int, default=5)
    args = parser.parse_args()
    main(
        img_npz=args.img_npz,
        text_npz=args.text_npz,
        rating_path=args.rating_path,
        save_path=args.save_path,
        good_rating=args.good_rating,
        train_ratio=args.train_ratio,
        train_threshold=args.train_threshold,
    )



