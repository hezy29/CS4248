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


def ProbScale(x: dict):
    out = x.copy()
    for i in out.keys():
        sum_i = sum(out[i].values())
        for j in out[i].keys():
            out[i][j] /= sum_i
    return out


def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    with open(train_file) as f:
        texts_train = f.read()
    f.close()
    paragraphs_train = texts_train.split("\n")[-1]

    PennTreebankPOS = {"[START]": {}}
    WordEmission = {}

    for paragraph_train in paragraphs_train:
        words_train = paragraph_train.split(" ")
        prev_POS = "[START]"

        for word_train in words_train:
            tag = word_train.split("/")[-1]
            word = word_train[: -(len(tag) + 1)]

            if not tag in PennTreebankPOS.keys():  # Create new state
                PennTreebankPOS[tag] = {}
            if not tag in PennTreebankPOS[prev_POS].keys():  # a_ij from 0 to 1
                PennTreebankPOS[prev_POS][tag] = 1
            else:
                PennTreebankPOS[prev_POS][tag] += 1  # a_ij from x to x+1 (x > 0)

            # Observation Likelihood
            if not tag in WordEmission.keys():
                WordEmission[tag] = {word: 1}
            else:
                if not word in WordEmission[tag].keys():
                    WordEmission[tag][word] = 1
                else:
                    WordEmission[tag][word] += 1

            prev_POS = tag

    POS_trans_mat = ProbScale(PennTreebankPOS)
    obs_emi_mat = ProbScale(WordEmission)

    os.makedirs(model_file, exist_ok=True)
    write_file(POS_trans_mat, model_file + "/transition_matrix")
    write_file(obs_emi_mat, model_file + "/observation_emission")

    print("Finished...")


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print("Time:", end_time - start_time)
