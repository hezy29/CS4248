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
| **No smoothing** | 5.53%        | 92.77%  | 95.34%      |
| **Add-One**      | \            | 92.78%  | 95.35%      |
| **Witten-Bell**  | \            | 92.77%  | 95.35%      |

> Notes:
> 
> 1. Observation Likelihood Scaling is much more time-consuming than Transition Probability
> 
> 2. Add-One smoothing methodology has little effect on Observation Likelihood
