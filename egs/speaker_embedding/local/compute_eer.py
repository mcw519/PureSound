import argparse
import sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from tqdm import tqdm


def get_acc(
    trial_file: str,
    emb_folder_path: str,
    trial_score_file: str,
    prefix: str = "",
    suffix: str = "",
):
    trial_score = open(trial_score_file, "w")
    all_scores = []
    all_keys = []

    # for each trials in trial file
    with open(trial_file, "r") as f:
        tmp_file = f.readlines()
        for line in tqdm(tmp_file):
            line = line.strip()
            truth, x_speaker, y_speaker = line.split()

            x_speaker = x_speaker.split("/")
            x_speaker = "-".join(x_speaker).replace(".wav", "")
            x_speaker = f"{prefix}{x_speaker}{suffix}.txt"

            y_speaker = y_speaker.split("/")
            y_speaker = "-".join(y_speaker).replace(".wav", "")
            y_speaker = f"{prefix}{y_speaker}{suffix}.txt"

            X = np.loadtxt(f"{emb_folder_path}/{x_speaker}")
            Y = np.loadtxt(f"{emb_folder_path}/{y_speaker}")

            score = np.dot(X, Y) / ((np.dot(X, X) * np.dot(Y, Y)) ** 0.5)
            score = (score + 1) / 2

            all_scores.append(score)
            trial_score.write(str(score) + "\t" + truth)
            truth = int(truth)
            all_keys.append(truth)
            trial_score.write("\n")

    trial_score.close()

    y_score = np.asarray(all_scores)
    y = np.asarray(all_keys)

    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    sys.stdout.write("EER={0:.2f}\n".format(eer * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_file", help="path to the trial file", type=str)
    parser.add_argument("emb_folder", help="path to numpy file of embeddings", type=str)
    parser.add_argument("score_file", help="path to save scoring file", type=str)
    parser.add_argument(
        "--prefix",
        help="prefix of embedding name",
        type=str,
        required=False,
        default="",
    )
    parser.add_argument(
        "--suffix",
        help="suffix of embedding name",
        type=str,
        required=False,
        default="",
    )
    args = parser.parse_args()
    get_acc(
        trial_file=args.trial_file,
        emb_folder_path=args.emb_folder,
        trial_score_file=args.score_file,
        prefix=args.prefix,
        suffix=args.suffix,
    )
