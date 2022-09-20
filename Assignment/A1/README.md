# PennTree POS Tagging

> First assignment for CS4248 22 Fall

## Data

## Environment

> All fundamental experiments are conducted on `macOS 11.6.8` and successfully re-implemented on `Ubuntu 20.04.4`

- `Python 3.10.6 (macOS)`/`Python 3.8.10 (Linux)`

## Command

### Training

```shell
python3 buildtagger.py sents.train model-file
```

### Tagging test set

```shell
python3 runtagger.py sents.test model-file sents.out
```

### Evaluating tagging performance

```shell
python3 eval.py sents.out sents.answer
```

## Results

| A\B              | No smoothing | Add-One | Witten-Bell |
| ---------------- | ------------ | ------- | ----------- |
| **No smoothing** | 86.86%       | 92.77%  | 95.34%      |
| **Add-One**      | 86.87%       | 92.78%  | 95.35%      |
| **Witten-Bell**  | 86.87%       | 92.77%  | 95.35%      |

> Notes:
> 
> 1. Observation Likelihood Scaling is much more time-consuming than Transition Probability. 
> 
> 2. Smoothing on Observation Likelihood is much more effective than that on Transition Probability due to OOV words. 
