import argparse
from numpy.lib.twodim_base import diag

from tqdm import tqdm
import pickle
import os
import json
import pandas as pd
import numpy as np
# import pydub

import joyful

# def write(f, sr, x, normalized=False):
#     """numpy array to MP3"""
#     channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
#     if normalized:  # normalized array - each item should be a float in [-1, 1)
#         y = np.int16(x * 2 ** 15)
#     else:
#         y = np.int16(x)
#     song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
#     song.export(f, format="mp3", bitrate="320k")


log = joyful.utils.get_logger()
# This script appears to be a data preprocessing script for different datasets, including IEMOCAP and MOSEI. Here's a breakdown of what it does:

# 1. **Import Libraries**: The script imports necessary libraries and modules such as `argparse`, `tqdm`, `pickle`, `os`, `json`, `pandas`, `numpy`, and `joyful`.

# 2. **Define Functions**:
#    - `get_iemocap()`: This function loads the IEMOCAP dataset features from a pickle file, splits them into train, dev, and test sets, and returns them.
#    - `get_iemocap_split(split_utterances)`: This function is similar to `get_iemocap()` but splits each dialogue into smaller segments or utterances based on the `split_utterances` parameter.
#    - `get_mosei_from_tbje(args)`: This function loads the MOSEI dataset features from a pickle file, processes them, and returns train, dev, and test sets.
#    - `get_mosei_from_tbje_emotion(args)`: This function is similar to `get_mosei_from_tbje(args)` but is specific to extracting emotion-related features.
#    - `get_mosei()`: This function loads the MOSEI dataset features, splits them into train, dev, and test sets, and returns them.
   
# 3. **Main Function**:
#    - The `main()` function parses command-line arguments using `argparse`, calls the appropriate functions based on the dataset specified, and saves the preprocessed data into pickle files.

# 4. **Script Execution**:
#    - Finally, the script checks if it's being run directly (`__name__ == "__main__"`) and executes the `main()` function with the provided command-line arguments.

# This script seems well-structured for preprocessing different datasets, allowing flexibility in handling different data formats and processing requirements.

def get_iemocap():
    joyful.utils.set_seed(args.seed)
    feature_path= "./data/iemocap/IEMOCAP_features.pkl"
    if args.dataset == "iemocap":
        (
            video_ids,
            video_speakers,
            video_labels,
            video_text,
            video_audio,
            video_visual,
            video_sentence,
            trainVids,
            test_vids,
        ) = pickle.load(
            open(feature_path, "rb"), encoding="latin1"
        )
    elif args.dataset == "iemocap_4":
        (
            video_ids,
            video_speakers,
            video_labels,
            video_text,
            video_audio,
            video_visual,
            video_sentence,
            trainVids,
            test_vids,
        ) = pickle.load(
            open("./data/iemocap_4/IEMOCAP_features_4.pkl", "rb"), encoding="latin1"
        )
    elif args.dataset == "meld":
        (
            video_ids,
            video_speakers,
            video_labels,
            video_text,
            video_audio,
            video_visual,
            video_sentence,
            trainVids,
            test_vids,
            aaa
        ) = pickle.load(
            open("./data/MELD/MELD_features_final.pkl", "rb"), encoding="latin1"
        )
        trainVids= list(trainVids)
        test_vids= list(test_vids)
    print(type(video_sentence), len(video_sentence), video_sentence.keys())
    # print(len(video_ids), type(video_ids), video_ids.keys())
    # print(video_ids['Ses02F_impro05'])
    # print(video_ids['Ses02M_impro05'])
    train, dev, test = [], [], []
    dev_size = int(len(trainVids) * 0.1)
    train_vids, dev_vids = trainVids[dev_size:], trainVids[:dev_size]
    print(len(train_vids), len(test_vids))
    flag= 1
    rows= 0
    for vid in tqdm(train_vids, desc="train"):
        # rows+= len(video_sentence[vid])
        if(flag==1):
            print(
                # replace 51 with length of dialogue. 
            #     len(video_speakers[vid]), # this is a list of the length 51 ['M', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'F', 'F', 'F', 'F', 'M', 'F', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'M', 'M', 'F', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'M', 'F'] 
            #     video_labels[vid],  # this is a list of the form [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 2, 2, 5, 2, 2, 5, 5, 2, 2, 2, 3, 5, 2, 2, 3, 2, 2, 5, 3, 2, 5, 5, 2, 2, 5, 5, 2, 5, 5, 2, 3, 2, 2, 3, 5, 5, 5, 5]
            #     video_text[vid], # list of length 51, each element of list is of length 100. 
            #     type(video_text[vid]), np.array(video_text[vid]).shape,
            #     np.array(video_audio[vid]).shape, # (51, 100)
            #      np.array(video_visual[vid]).shape, #  (51, 512)
            #     video_sentence[vid]
            )
            flag=0
        train.append(
            joyful.Sample(
                vid,
                video_speakers[vid],
                video_labels[vid],
                video_text[vid],
                video_audio[vid],
                video_visual[vid],
                video_sentence[vid],
            )
        )
    # print("Train rows are ", rows) : 5146. 
    for vid in tqdm(dev_vids, desc="dev"):
        rows+= len(video_sentence[vid])
        dev.append(
            joyful.Sample(
                vid,
                video_speakers[vid],
                video_labels[vid],
                video_text[vid],
                video_audio[vid],
                video_visual[vid],
                video_sentence[vid],
            )
        )
    # print("Dev rows are ", rows) : 664
    test_rows= 0
    for vid in tqdm(test_vids, desc="test"):
        test_rows+= len(video_sentence[vid])
        if(flag==0):
            print(
                # replace 51 with length of dialogue. 
                video_ids[vid][0],
                len(video_speakers[vid]), # this is a list of the length 51 ['M', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'F', 'F', 'F', 'F', 'M', 'F', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'M', 'M', 'F', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'M', 'F'] 
                # video_labels[vid],  # this is a list of the form [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 2, 2, 5, 2, 2, 5, 5, 2, 2, 2, 3, 5, 2, 2, 3, 2, 2, 5, 3, 2, 5, 5, 2, 2, 5, 5, 2, 5, 5, 2, 3, 2, 2, 3, 5, 5, 5, 5]
                # video_text[vid], # list of length 51, each element of list is of length 100. 
                # type(video_text[vid]), np.array(video_text[vid]).shape,
                # np.array(video_audio[vid]).shape, # (51, 100)
                #  np.array(video_visual[vid]).shape, #  (51, 512)
                video_sentence[vid]
            )
            flag=1
        test.append(
            joyful.Sample(
                vid,
                video_speakers[vid],
                video_labels[vid],
                video_text[vid],
                video_audio[vid],
                video_visual[vid],
                video_sentence[vid],
            )
        )
    # print("Test rows are ", test_rows): 1623 : diff 538 
    log.info("unsorted train vids:")
    log.info((train_vids))
    log.info("train vids:")
    log.info(sorted(train_vids))
    log.info("dev vids:")
    log.info(sorted(dev_vids))
    log.info("test vids:")
    log.info(sorted(test_vids))

    return train, dev, test


def get_iemocap_split(split_utterances):
    joyful.utils.set_seed(args.seed)

    if args.dataset == "iemocap":
        (
            video_ids,
            video_speakers,
            video_labels,
            video_text,
            video_audio,
            video_visual,
            video_sentence,
            trainVids,
            test_vids,
        ) = pickle.load(
            open("./data/iemocap/IEMOCAP_features.pkl", "rb"), encoding="latin1"
        )
    elif args.dataset == "iemocap_4":
        (
            video_ids,
            video_speakers,
            video_labels,
            video_text,
            video_audio,
            video_visual,
            video_sentence,
            trainVids,
            test_vids,
        ) = pickle.load(
            open("./data/iemocap_4/IEMOCAP_features_4.pkl", "rb"), encoding="latin1"
        )
    elif args.dataset == "meld":
        (
            video_ids,
            video_speakers,
            video_labels,
            video_text,
            video_audio,
            video_visual,
            video_sentence,
            trainVids,
            test_vids,
            aaa
        ) = pickle.load(
            open("./data/MELD/MELD_features_final.pkl", "rb"), encoding="latin1"
        )
    trainVids= list(trainVids)
    test_vids= list(test_vids)
    train, dev, test = [], [], []
    dev_size = int(len(trainVids) * 0.1)
    train_vids, dev_vids = trainVids[dev_size:], trainVids[:dev_size]

    for vid in tqdm(train_vids, desc="train"):
        for split_i in range(len(video_text[vid]) // split_utterances):
            train.append(
                joyful.Sample(
                    vid,
                    video_speakers[vid][split_i : split_i + split_utterances],
                    video_labels[vid][split_i : split_i + split_utterances],
                    video_text[vid][split_i : split_i + split_utterances],
                    video_audio[vid][split_i : split_i + split_utterances],
                    video_visual[vid][split_i : split_i + split_utterances],
                    video_sentence[vid][split_i : split_i + split_utterances],
                )
            )
    for vid in tqdm(dev_vids, desc="dev"):
        for split_i in range(len(video_text[vid]) // split_utterances):
            dev.append(
                joyful.Sample(
                    vid,
                    video_speakers[vid][split_i : split_i + split_utterances],
                    video_labels[vid][split_i : split_i + split_utterances],
                    video_text[vid][split_i : split_i + split_utterances],
                    video_audio[vid][split_i : split_i + split_utterances],
                    video_visual[vid][split_i : split_i + split_utterances],
                    video_sentence[vid][split_i : split_i + split_utterances],
                )
            )
    for vid in tqdm(test_vids, desc="test"):
        for split_i in range(len(video_text[vid]) // split_utterances):
            test.append(
                joyful.Sample(
                    vid,
                    video_speakers[vid][split_i : split_i + split_utterances],
                    video_labels[vid][split_i : split_i + split_utterances],
                    video_text[vid][split_i : split_i + split_utterances],
                    video_audio[vid][split_i : split_i + split_utterances],
                    video_visual[vid][split_i : split_i + split_utterances],
                    video_sentence[vid][split_i : split_i + split_utterances],
                )
            )
    log.info("train vids:")
    log.info(sorted(train_vids))
    log.info("dev vids:")
    log.info(sorted(dev_vids))
    log.info("test vids:")
    log.info(sorted(test_vids))

    return train, dev, test


def get_mosei_from_tbje(args):
    def cmumosei_7(a):
        if a < -2:
            res = 0
        if -2 <= a and a < -1:
            res = 1
        if -1 <= a and a < 0:
            res = 2
        if 0 <= a and a <= 0:
            res = 3
        if 0 < a and a <= 1:
            res = 4
        if 1 < a and a <= 2:
            res = 5
        if a > 2:
            res = 6
        return res

    def cmumosei_2(a):
        # binary classification problem- I guess we are returning a real valued a which must be converted to binary. 
        if a < 0:
            return 0
        if a >= 0:
            return 1

    (
        video_ids,
        video_speakers,
        video_labels,
        video_text,
        video_audio,
        video_visual,
        video_sentence,
        trainVids,
        dev_vids,
        test_vids,
    ) = pickle.load(
        open("./data/mosei/tbje_mosei_updated.pkl", "rb"), encoding="latin1"
    )

    train, dev, test = [], [], []

    video_ids = np.array(list(video_ids.items()))

    for dialogue_idx in tqdm(trainVids, desc="train"):
        num_of_utterances = len(video_ids[video_ids[:, 0] == dialogue_idx][0][1])
        audio = [
            np.average(video_audio[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]
        text = [" ".join(i) for i in video_sentence[dialogue_idx]]
        visual = [video_visual[dialogue_idx][i] for i in range(num_of_utterances)]
        visual_size = np.array(
            [len(video_visual[dialogue_idx][i]) for i in range(num_of_utterances)]
        )
        if not all(visual_size == 35):
            breakpoint()

        speakers = ["M" for i in range(num_of_utterances)]
        if args.dataset == "mosei_tbje_7class":
            labels = [
                cmumosei_7(video_labels[dialogue_idx][i][0])
                for i in range(num_of_utterances)
            ]
        elif args.dataset == "mosei_tbje_2class":
            labels = [
                cmumosei_2(video_labels[dialogue_idx][i][0])
                for i in range(num_of_utterances)
            ]
        train.append(
            joyful.Sample(
                "diag_" + str(dialogue_idx), speakers, labels, text, audio, visual, text
            )
        )

    for dialogue_idx in tqdm(dev_vids, desc="dev"):
        num_of_utterances = len(video_ids[video_ids[:, 0] == dialogue_idx][0][1])
        audio = [
            np.average(video_audio[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]
        text = [" ".join(i) for i in video_sentence[dialogue_idx]]
        visual = [video_visual[dialogue_idx][i] for i in range(num_of_utterances)]

        visual_size = np.array(
            [len(video_visual[dialogue_idx][i]) for i in range(num_of_utterances)]
        )
        if not all(visual_size == 35):
            ()

        speakers = ["M" for i in range(num_of_utterances)]
        if args.dataset == "mosei_tbje_7class":
            labels = [
                cmumosei_7(video_labels[dialogue_idx][i][0])
                for i in range(num_of_utterances)
            ]
        elif args.dataset == "mosei_tbje_2class":
            labels = [
                cmumosei_2(video_labels[dialogue_idx][i][0])
                for i in range(num_of_utterances)
            ]

        dev.append(
            joyful.Sample(
                "diag_" + str(dialogue_idx), speakers, labels, text, audio, visual, text
            )
        )

    for dialogue_idx in tqdm(test_vids, desc="test"):
        num_of_utterances = len(video_ids[video_ids[:, 0] == dialogue_idx][0][1])
        audio = [
            np.average(video_audio[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]
        text = [" ".join(i) for i in video_sentence[dialogue_idx]]
        visual = [video_visual[dialogue_idx][i] for i in range(num_of_utterances)]

        visual_size = np.array(
            [len(video_visual[dialogue_idx][i]) for i in range(num_of_utterances)]
        )
        if not all(visual_size == 35):
            breakpoint()

        speakers = ["M" for i in range(num_of_utterances)]
        if args.dataset == "mosei_tbje_7class":
            labels = [
                cmumosei_7(video_labels[dialogue_idx][i][0])
                for i in range(num_of_utterances)
            ]
        elif args.dataset == "mosei_tbje_2class":
            labels = [
                cmumosei_2(video_labels[dialogue_idx][i][0])
                for i in range(num_of_utterances)
            ]

        test.append(
            joyful.Sample(
                "diag_" + str(dialogue_idx), speakers, labels, text, audio, visual, text
            )
        )
    return train, dev, test


def get_mosei_from_tbje_emotion(args):
    (
        video_ids,
        video_speakers,
        video_labels,
        video_text,
        video_audio,
        video_visual,
        video_sentence,
        trainVids,
        dev_vids,
        test_vids,
    ) = pickle.load(
        open("./data/mosei/tbje_mosei_" + args.emotion + ".pkl", "rb"),
        encoding="latin1",
    )

    train, dev, test = [], [], []

    video_ids = np.array(list(video_ids.items()))

    for dialogue_idx in tqdm(trainVids, desc="train"):
        num_of_utterances = len(video_ids[video_ids[:, 0] == dialogue_idx][0][1])
        audio = [
            np.average(video_audio[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]
        text = [" ".join(i) for i in video_sentence[dialogue_idx]]
        visual = [video_visual[dialogue_idx][i] for i in range(num_of_utterances)]
        visual_size = np.array(
            [len(video_visual[dialogue_idx][i]) for i in range(num_of_utterances)]
        )
        if not all(visual_size == 35):
            breakpoint()

        speakers = ["M" for i in range(num_of_utterances)]
        labels = [video_labels[dialogue_idx][i] for i in range(num_of_utterances)]
        train.append(
            joyful.Sample(
                "diag_" + str(dialogue_idx), speakers, labels, text, audio, visual, text
            )
        )

    for dialogue_idx in tqdm(dev_vids, desc="dev"):
        num_of_utterances = len(video_ids[video_ids[:, 0] == dialogue_idx][0][1])
        audio = [
            np.average(video_audio[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]
        text = [" ".join(i) for i in video_sentence[dialogue_idx]]
        visual = [video_visual[dialogue_idx][i] for i in range(num_of_utterances)]

        visual_size = np.array(
            [len(video_visual[dialogue_idx][i]) for i in range(num_of_utterances)]
        )
        if not all(visual_size == 35):
            breakpoint()

        speakers = ["M" for i in range(num_of_utterances)]
        labels = [video_labels[dialogue_idx][i] for i in range(num_of_utterances)]

        dev.append(
            joyful.Sample(
                "diag_" + str(dialogue_idx), speakers, labels, text, audio, visual, text
            )
        )

    for dialogue_idx in tqdm(test_vids, desc="test"):
        num_of_utterances = len(video_ids[video_ids[:, 0] == dialogue_idx][0][1])
        audio = [
            np.average(video_audio[dialogue_idx][i], axis=0)
            for i in range(num_of_utterances)
        ]
        text = [" ".join(i) for i in video_sentence[dialogue_idx]]
        visual = [video_visual[dialogue_idx][i] for i in range(num_of_utterances)]

        visual_size = np.array(
            [len(video_visual[dialogue_idx][i]) for i in range(num_of_utterances)]
        )
        if not all(visual_size == 35):
            breakpoint()

        speakers = ["M" for i in range(num_of_utterances)]
        labels = [video_labels[dialogue_idx][i] for i in range(num_of_utterances)]

        test.append(
            joyful.Sample(
                "diag_" + str(dialogue_idx), speakers, labels, text, audio, visual, text
            )
        )

    return train, dev, test


def get_mosei():

    mosei_path = args.data_dir
    joyful.utils.set_seed(args.seed)

    feature_path = "categorical.p"
    path = os.path.join(mosei_path, feature_path)
    (
        video_ids,
        video_speakers,
        video_labels,
        video_text,
        video_audio,
        video_visual,
        video_sentence,
        trainVids,
        test_vids,
    ) = pickle.load(open(path, "rb"), encoding="latin1")

    label_count = []
    len_count = []
    trainVids = np.array(trainVids)
    test_vids = np.array(test_vids)
    train, dev, test = [], [], []
    dev_size = int(len(trainVids) * 0.1)
    train_vids, dev_vids = trainVids[dev_size:], trainVids[:dev_size]

    for vid in tqdm(train_vids, desc="train"):
        train.append(
            joyful.Sample(
                vid,
                video_speakers[vid],
                video_labels[vid],
                video_text[vid],
                video_audio[vid],
                video_visual[vid],
                video_sentence[vid],
            )
        )
        label_count.extend(video_labels[vid])
        len_count.append(len(video_speakers[vid]))

    for vid in tqdm(dev_vids, desc="dev"):
        dev.append(
            joyful.Sample(
                vid,
                video_speakers[vid],
                video_labels[vid],
                video_text[vid],
                video_audio[vid],
                video_visual[vid],
                video_sentence[vid],
            )
        )
    for vid in tqdm(test_vids, desc="test"):
        test.append(
            joyful.Sample(
                vid,
                video_speakers[vid],
                video_labels[vid],
                video_text[vid],
                video_audio[vid],
                video_visual[vid],
                video_sentence[vid],
            )
        )

    total = len(label_count)
    pos = sum(label_count)
    neg = total - pos

    log.info("train vids:")
    log.info(sorted(train_vids))
    log.info("dev vids:")
    log.info(sorted(dev_vids))
    log.info("test vids:")
    log.info(sorted(test_vids))

    return train, dev, test

def get_mosei_split(split_utterances):
	
    mosei_path = args.data_dir
    joyful.utils.set_seed(args.seed)

    feature_path = "categorical.p"
    path = os.path.join(mosei_path, feature_path)
    (
        video_ids,
        video_speakers,
        video_labels,
        video_text,
        video_audio,
        video_visual,
        video_sentence,
        trainVids,
        test_vids,
    ) = pickle.load(open(path, "rb"), encoding="latin1")

    label_count = []
    len_count = []
    trainVids = np.array(trainVids)
    test_vids = np.array(test_vids)
    train, dev, test = [], [], []
    dev_size = int(len(trainVids) * 0.1)
    train_vids, dev_vids = trainVids[dev_size:], trainVids[:dev_size]

    for vid in tqdm(train_vids, desc="train"):
        for split_i in range(len(video_text[vid]) // split_utterances):
            train.append(
                joyful.Sample(
                    vid,
                    video_speakers[vid][split_i : split_i + split_utterances],
                    video_labels[vid][split_i : split_i + split_utterances],
                    video_text[vid][split_i : split_i + split_utterances],
                    video_audio[vid][split_i : split_i + split_utterances],
                    video_visual[vid][split_i : split_i + split_utterances],
                    video_sentence[vid][split_i : split_i + split_utterances],
                )
            )
    for vid in tqdm(dev_vids, desc="dev"):
        for split_i in range(len(video_text[vid]) // split_utterances):
            dev.append(
                joyful.Sample(
                    vid,
                    video_speakers[vid][split_i : split_i + split_utterances],
                    video_labels[vid][split_i : split_i + split_utterances],
                    video_text[vid][split_i : split_i + split_utterances],
                    video_audio[vid][split_i : split_i + split_utterances],
                    video_visual[vid][split_i : split_i + split_utterances],
                    video_sentence[vid][split_i : split_i + split_utterances],
                )
            )
    for vid in tqdm(test_vids, desc="test"):
        for split_i in range(len(video_text[vid]) // split_utterances):
            test.append(
                joyful.Sample(
                    vid,
                    video_speakers[vid][split_i : split_i + split_utterances],
                    video_labels[vid][split_i : split_i + split_utterances],
                    video_text[vid][split_i : split_i + split_utterances],
                    video_audio[vid][split_i : split_i + split_utterances],
                    video_visual[vid][split_i : split_i + split_utterances],
                    video_sentence[vid][split_i : split_i + split_utterances],
                )
            )

    total = len(label_count)
    pos = sum(label_count)
    neg = total - pos

    log.info("train vids:")
    log.info(sorted(train_vids))
    log.info("dev vids:")
    log.info(sorted(dev_vids))
    log.info("test vids:")
    log.info(sorted(test_vids))

    return train, dev, test

def main(args):
    if args.dataset == "iemocap" and args.split_utterances == -1:
        print("Not splitting iemocap")
        train, dev, test = get_iemocap()
        data = {"train": train, "dev": dev, "test": test}
        joyful.utils.save_pkl(data, "./data/iemocap/data_iemocap_asr.pkl")
    if args.dataset == "iemocap_4" and args.split_utterances == -1:
        train, dev, test = get_iemocap()
        data = {"train": train, "dev": dev, "test": test}
        joyful.utils.save_pkl(data, "./data/iemocap_4/data_iemocap_4.pkl")
    if args.dataset == "iemocap" and args.split_utterances != -1:
        print("Splitting iemocap")
        train, dev, test = get_iemocap_split(args.split_utterances)
        data = {"train": train, "dev": dev, "test": test}
        joyful.utils.save_pkl(
            data,
            "./data/iemocap/data_iemocap_split_"
            + str(args.split_utterances)
            + ".pkl",
        )
    if args.dataset == "iemocap_4" and args.split_utterances != -1:
        train, dev, test = get_iemocap_split(args.split_utterances)
        data = {"train": train, "dev": dev, "test": test}
        joyful.utils.save_pkl(
            data,
            "./data/iemocap_4/data_iemocap_4_split_"
            + str(args.split_utterances)
            + ".pkl",
        )
        print("Saved split dataset")
    
    if args.dataset == "meld" and args.split_utterances == -1:
        print("Not splitting meld")
        train, dev, test = get_iemocap()
        data = {"train": train, "dev": dev, "test": test}
        joyful.utils.save_pkl(data, "./data/MELD/data_meld.pkl")
    if args.dataset == "meld" and args.split_utterances != -1:
        train, dev, test = get_iemocap_split(args.split_utterances)
        data = {"train": train, "dev": dev, "test": test}
        joyful.utils.save_pkl(data,  "./data/MELD/data_meld_split_"
            + str(args.split_utterances)
            + ".pkl",
		)
        
    if args.dataset == "mosei" and args.split_utterances == -1:
        train, dev, test = get_mosei()
        data = {"train": train, "dev": dev, "test": test}
        joyful.utils.save_pkl(data, "./data/mosei/data_mosei.pkl")
        
    if args.dataset == "mosei" and args.split_utterances != -1:
        print("Splitting mosei")
        train, dev, test = get_mosei_split(args.split_utterances)
        data = {"train": train, "dev": dev, "test": test}
        joyful.utils.save_pkl(
            data,
            "./data/mosei/data_mosei_split_"
            + str(args.split_utterances)
            + ".pkl",
        )


    if args.dataset == "mosei_tbje_2class":
        train, dev, test = get_mosei_from_tbje(args)
        data = {"train": train, "dev": dev, "test": test}
        joyful.utils.save_pkl(data, "./data/mosei/data_mosei_2class.pkl")

    if args.dataset == "mosei_tbje_7class":
        train, dev, test = get_mosei_from_tbje(args)
        data = {"train": train, "dev": dev, "test": test}
        joyful.utils.save_pkl(data, "./data/mosei/data_mosei_7class.pkl")

    if args.dataset == "mosei_emotion":
        train, dev, test = get_mosei_from_tbje_emotion(args)
        data = {"train": train, "dev": dev, "test": test}
        joyful.utils.save_pkl(data, "./data/mosei/data_mosei_" + args.emotion + ".pkl")

    log.info("number of train samples: {}".format(len(train)))
    log.info("number of dev samples: {}".format(len(dev)))
    log.info("number of test samples: {}".format(len(test)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["iemocap", "iemocap_4", "mosei","meld"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Dataset directory"
    )
    # parser.add_argument(
    #     "--asr", action="store_true", default=False, help="Whether or not to use ASR text instead of transcribed text for iemocap"
    # )
    parser.add_argument(
        "--use_wave2vec2_audio_features",
        action="store_true",
        default=False,
        help="uses wave2vec2 extracted audio features",
    )
    parser.add_argument(
        "--use_pose_visual_features",
        action="store_true",
        default=False,
        help="uses extracted pose from visual modality",
    )
    parser.add_argument("--split_utterances", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=24, help="Random seed.")
    args = parser.parse_args()

    main(args)
