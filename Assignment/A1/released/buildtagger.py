# python3.8 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import json
import os
import math
import sys
import datetime


def read_file(path):
    with open(path) as f:
        data = f.read()
    f.close()
    return data


def write_file(data, path):
    data_json = json.dumps(data, sort_keys=False, indent=4, separators=(",", ": "))
    with open(path, "w") as f:
        f.write(data_json)
    f.close()


def ProbScale(x: dict, ref: dict or set or list = None, smoothing: str = None):
    out = x
    for i in out.keys():
        sum_i = sum(out[i].values())
        for j in out[i].keys():
            if not smoothing:
                out[i][j] /= sum_i
            elif smoothing == "Witten-Bell":
                out[i][j] /= sum_i + len(out.keys())
        if smoothing == "Witten-Bell":
            for ind in ref:
                if not ind in out[i].keys():
                    out[i][ind] = len(out.keys()) / (
                        len(ref) * (sum_i + len(out.keys()))
                    )
    return out


def update_TProb(tag, prev_POS, TProb):
    # Transition Count
    if not tag in TProb.keys():  # Create new state
        TProb[tag] = {}
    if not tag in TProb[prev_POS].keys():  # a_ij from 0 to 1
        TProb[prev_POS][tag] = 1
    else:
        TProb[prev_POS][tag] += 1  # a_ij from x to x+1 (x > 0)


def update_ObsL(tag, word, ObsL):
    # Observation Likelihood
    if not tag in ObsL.keys():
        ObsL[tag] = {word: 1}
    else:
        if not word in ObsL[tag].keys():
            ObsL[tag][word] = 1
        else:
            ObsL[tag][word] += 1


def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    texts_train = read_file(train_file)
    paragraphs_train = texts_train.split("\n")[:-1]

    PennTreebankPOS = {"[START]": {}}
    WordEmission = {}

    for paragraph_train in paragraphs_train:
        words_train = paragraph_train.split(" ")
        prev_POS = "[START]"

        for word_train in words_train:
            tag = word_train.split("/")[-1]
            word = word_train[: -(len(tag) + 1)]

            update_TProb(tag, prev_POS, PennTreebankPOS)
            update_ObsL(tag, word, WordEmission)

            prev_POS = tag

        # last word to q_f
        if not "[END]" in PennTreebankPOS[prev_POS].keys():
            PennTreebankPOS[prev_POS]["[END]"] = 1
        else:
            PennTreebankPOS[prev_POS]["[END]"] += 1

    Words = set()
    for item in WordEmission.values():
        for word in item.keys():
            Words.add(word)

    Tprob_mat = ProbScale(PennTreebankPOS, WordEmission.keys(), smoothing="Witten-Bell")
    ObsL_mat = ProbScale(WordEmission, Words, "Witten-Bell")

    os.makedirs(model_file, exist_ok=True)
    write_file(Tprob_mat, model_file + "/transition_probability")
    write_file(ObsL_mat, model_file + "/observation_likelihood")

    print("Finished...")


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print("Time:", end_time - start_time)
