# Language Identifier

> CS4248 Assignment 2

## Introduction

`a2part2.py` implements a neural network to classify different language texts. 

The network uses the character bigram as input, with an output of the class of the text input. In this case, five different languages, i.e., English, German, French, Italian, and Spanish, are given.

## Network

### Architecture

- `Embedding`: The layer embeds input bigrams to vectors with fixed length `embedding_dim`. 

- `FC Layer`: This layer makes a linear projection $xW+b$ of the input vector $x$.

- `ReLU`: This layer makes a non-linear activation $\max{(x_i,0)}$ of the input vector $x$. 

- `Dropout`: This layer randomly masks the input vector $x$ with probability $p$. 

- `Softmax`: This layer conducts a probabilistic projection $\frac{e^{x_i}}{\sum_i{e^{x_i}}}$ for input vector $x$. 

### Criterion

`CrossEntropy`: $-\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^C{y_{ij}\log{(p_{ij})}}$

- $N$: number of samples in a batch

- $C=5$: number of classes

### Optimizer

`Adam`

## Environment

- `Python 3.10.8` + `PyTorch 1.12.1`

- `Python 3.8.10` + `PyTorch 1.9.0` + `CUDA 11.3`

## Command

### Train model

```shellag-0-1gfhs598fag-1-1gfhs598fag-0-1gfhs598fag-1-1gfhs598fag-0-1gfhs598fag-1-1gfhs598f
python3 a2part2.py --train --text_path x_train.txt --label_path y_train.txt --model_path model.pt
```

### Test

```shell
python3.8 a2part2.py --test --text_path x_test.txt --model_path model.pt --output_path out.txt
```

### Evaluating Model Performance

```shell
python3.8 eval.py out.txt y_test.txt
```

## Results

| Type  | Sample Size | Number of Correct Classification | Accuracy |
| ----- | ----------- | -------------------------------- | -------- |
| Train | 2000        | 1988                             | 0.994    |
| Test  | 500         | 496                              | 0.992    |
