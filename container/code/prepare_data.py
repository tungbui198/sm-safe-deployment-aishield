import sys

import numpy as np
import pandas as pd
from pathlib import Path

from .utils import list_dir, logger, pre_process


class PrepareDataConfig:
    ROOT_DIR = Path("/opt/ml/processing")
    INPUT_DIR = ROOT_DIR / "input"
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    IN_TRAIN_DIR = INPUT_DIR / "train"
    IN_TRAIN_AUDIO_DIR = IN_TRAIN_DIR / "audio_files"
    IN_TRAIN_METADATA_CSV = IN_TRAIN_DIR / "metadata.csv"
    IN_TRAIN_MEDICAL_CONDITION_CSV = IN_TRAIN_DIR / "medical_condition.csv"

    IN_TEST_DIR = INPUT_DIR / "test"
    IN_TEST_AUDIO_DIR = IN_TEST_DIR / "audio_files"
    IN_TEST_RESULTS_CSV = IN_TEST_DIR / "results.csv"

    OUT_TRAIN_DIR = ROOT_DIR / "train"
    OUT_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TRAIN_CSV = OUT_TRAIN_DIR / "train.csv"

    OUT_TEST_DIR = ROOT_DIR / "test"
    OUT_TEST_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TEST_CSV = OUT_TEST_DIR / "test.csv"


def inspect_input():
    logger.info(f"Start inspect_input")
    files = list_dir(PrepareDataConfig.INPUT_DIR)
    logger.info(f"{PrepareDataConfig.INPUT_DIR.as_posix()}: {files}")

    files = list_dir(PrepareDataConfig.IN_TRAIN_DIR)
    logger.info(f"{PrepareDataConfig.IN_TRAIN_DIR.as_posix()}: {files}")

    files = list_dir(PrepareDataConfig.IN_TEST_DIR)
    logger.info(f"{PrepareDataConfig.IN_TEST_DIR.as_posix()}: {files}")


def prepare_train_dataset():
    logger.info(f"Start extract training data")
    train_meta_df = pd.read_csv(PrepareDataConfig.IN_TRAIN_METADATA_CSV)
    train_meta_df = train_meta_df.loc[train_meta_df["audio_noise_note"].isnull(
    )]

    train_df = pd.DataFrame()
    train_df["fname"] = train_meta_df["uuid"].apply(lambda x: f"{x}.wav")
    train_df["label"] = train_meta_df["assessment_result"]
    train_df["silence"] = 0

    xmfcc = []
    for index, fname in enumerate(train_df["fname"]):
        fpath = str(PrepareDataConfig.IN_TRAIN_AUDIO_DIR / fname)
        try:
            mfcc = pre_process(fpath)
            xmfcc.append(mfcc)
        except:
            train_df.at[index, "silence"] = 1
    mfcc_df = pd.DataFrame(xmfcc)
    train_df = train_df.loc[train_df["silence"] == 0]
    train_df.reset_index(drop=True, inplace=True)
    mfcc_df["label"] = train_df["label"]

    mfcc_df.to_csv(PrepareDataConfig.OUT_TRAIN_CSV, index=False)


def prepare_test_dataset():
    logger.info(f"Start extract test data")
    test_meta_df = pd.read_csv(PrepareDataConfig.IN_TEST_RESULTS_CSV)
    test_meta_df["fname"] = test_meta_df["uuid"].apply(lambda x: f"{x}.wav")

    xmfcc = []
    for fname in test_meta_df["fname"]:
        fpath = str(PrepareDataConfig.IN_TEST_AUDIO_DIR / fname)
        try:
            mfcc = pre_process(fpath)
            xmfcc.append(mfcc)
        except:
            mfcc = np.zeros(2808)
            mfcc = mfcc.reshape(-1,)
            xmfcc.append(mfcc)

    mfcc_df = pd.DataFrame(xmfcc)
    mfcc_df["label"] = test_meta_df["assessment_result"]
    mfcc_df.to_csv(PrepareDataConfig.OUT_TEST_CSV, index=False)


def inspect_output():
    logger.info(f"Start inspect_output")
    files = list_dir(PrepareDataConfig.OUTPUT_DIR)
    logger.info(f"{PrepareDataConfig.OUTPUT_DIR.as_posix()}: {files}")


if __name__ == "__main__":
    sys.path.append("/opt/program")
    inspect_input()
    prepare_train_dataset()
    prepare_test_dataset()
    inspect_output()
    sys.exit(0)
