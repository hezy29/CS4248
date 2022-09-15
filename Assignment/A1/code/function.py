import json
import numpy as np


def read_file(path, isjson: bool = False):
    with open(path) as f:
        if isjson:
            data = json.load(f)
        else:
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


def get_tprob(A: dict, from_state: str, to_state: str):
    out = 0
    if to_state in A[from_state].keys():
        out = A[from_state][to_state]
    return out


def get_obsl(B: dict, state: str, word: str):
    out = 0
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
