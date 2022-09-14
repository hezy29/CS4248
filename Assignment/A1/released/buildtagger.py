# python3.8 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

from cgitb import text
import os
import math
import sys
import datetime


def read_file(path):
    with open(path) as f:
        data = f.read()
    f.close()
    return data


def update_prob_transition_matrix(tag, prev_POS, A):
    if not tag in A.keys():  # Create new state
        A[tag] = {}
    if not tag in A[prev_POS].keys():  # a_ij from 0 to 1
        A[prev_POS][tag] = 1
    else:  # a_ij from x to x+1 (x > 0)
        A[prev_POS][tag] += 1


def update_emission_prob_matrix(word, tag, E):
    if not tag in E.keys():  # Create new state and record the first word appears
        E[tag] = {word: 1}
    else:
        if (
            not word in E[tag].keys()
        ):  # Record the first word appears in the existed state
            E[tag][word] = 1
        else:  # Record times of appearance
            E[tag][word] += 1


def ProbScale(x: dict):
    out = x.copy()
    for i in out.keys():
        sum_i = sum(out[i].values())
        for j in out[i].keys():
            out[i][j] /= sum_i
    return out


def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    texts_train = read_file(train_file)
    paragraphs_train = texts_train.split("\n")[-1]

    POS_trans_mat = {"[START]": {}}
    obs_emi_mat = {}

    for paragraph_train in paragraphs_train:
        words_train = paragraph_train.split(" ")
        prev_POS = "[START]"

        for word_train in words_train:
            tag = word_train.split("/")[-1]
            word = word_train[: -(len(tag) + 1)]

            # Transition Probability Matrix
            update_prob_transition_matrix(tag, prev_POS, POS_trans_mat)

            # Observation Likelihood
            update_emission_prob_matrix(word, tag, obs_emi_mat)

            prev_POS = tag

    POS_trans_mat = ProbScale(POS_trans_mat)
    obs_emi_mat = ProbScale(obs_emi_mat)

    print("Finished...")


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print("Time:", end_time - start_time)
