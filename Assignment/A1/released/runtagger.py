# python3.8 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import json
import numpy as np
import os
import math
import sys
import datetime


def read_file(path, isjson: bool = False):
    with open(path) as f:
        if isjson:
            data = json.load(f)
        else:
            data = f.read()
    f.close()
    return data


def get_tprob(A: dict, from_state: str, to_state: str):
    out = 0
    if to_state in A[from_state].keys():
        out = A[from_state][to_state]
    return out


def get_obsl(B: dict, state: str, word: str):
    out = 0.5
    if word in B[state].keys():
        out = B[state][word]
    return out


def update_viterbi(
    viterbi,
    transition_matrix,
    observation_likelihood,
    word,
    backpoint: bool = False,
):
    if not backpoint:
        out = np.array(
            [
                [
                    (
                        viterbi[:, -1]
                        * np.array(
                            [
                                get_tprob(transition_matrix, prev_state, state)
                                for prev_state in observation_likelihood.keys()
                            ]
                        )
                        * get_obsl(observation_likelihood, state, word)
                    ).max()
                    for state in observation_likelihood.keys()
                ]
            ]
        ).T
    else:
        out = np.array(
            [
                [
                    (
                        viterbi[:, -1]
                        * np.array(
                            [
                                get_tprob(transition_matrix, prev_state, state)
                                for prev_state in observation_likelihood.keys()
                            ]
                        )
                        # * get_obsl(observation_likelihood, state, word)
                    ).argmax()
                    for state in observation_likelihood.keys()
                ]
            ]
        ).T

    return out


def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
    texts_test = read_file(test_file)
    paragraphs_test = texts_test.split("\n")[:-1]

    Words = set()
    for paragraph_test in paragraphs_test:
        for word in paragraph_test:
            Words.add(word)

    paragraphs_tagged = []  # To store tagged paragraph

    A = read_file(
        model_file + "/transition_probability", isjson=True
    )  # Transition probability matrix A
    B = read_file(
        model_file + "/observation_likelihood", isjson=True
    )  # Observation likelihood matrix B

    # POS tagging for each paragraph using Viterbi algorithms
    for paragraph_test in paragraphs_test:
        words_test = paragraph_test.split(" ")

        # Intial State
        word = words_test[0]
        viterbi = np.array(
            [
                [
                    get_tprob(A, "[START]", state) * get_obsl(B, state, word)
                    for state in B.keys()
                ]
            ]
        ).T

        backpoint = -np.ones((len(B.keys()), 1))

        for word in words_test[1:]:
            vit_new_col = update_viterbi(viterbi, A, B, word)
            bp_new_col = update_viterbi(viterbi, A, B, word, backpoint=True)
            viterbi = np.column_stack((viterbi, vit_new_col))
            backpoint = np.column_stack((backpoint, bp_new_col))

        # vit_last = (
        #     viterbi[:, -1]
        #     * np.array([get_tprob(A, prev_state, "[END]") for prev_state in B.keys()])
        # ).max()
        bp_last = (
            viterbi[:, -1]
            * np.array([get_tprob(A, prev_state, "[END]") for prev_state in B.keys()])
        ).argmax()

        # Backtracking optimal transition for POS tagger
        backtrace = [bp_last]
        for j in range(len(words_test), 1, -1):
            backtrace.insert(0, int(backpoint[backtrace[0]][j - 1]))

        tag_test = [list(B.keys())[j] for j in backtrace]

        paragraph_tagged = " ".join([a + "/" + b for a, b in zip(words_test, tag_test)])
        paragraphs_tagged.append(paragraph_tagged)

    texts_tagged = (
        "\n".join(paragraphs_tagged) + "\n"
    )  # Add the final "\n" back to where it belongs

    with open(out_file, "w") as f:
        f.write(texts_tagged)
    f.close()

    print("Finished...")


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print("Time:", end_time - start_time)
