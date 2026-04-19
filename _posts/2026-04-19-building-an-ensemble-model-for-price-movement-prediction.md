---
layout: post
title: "Building an ensemble-based model for price movement prediction"
date: 2026-04-19 00:00:00 -0000
categories:
    - python
    - gradient-boosting
    - optimization
tags: ["pytorch", "mlflow"]
---

# Building an ensemble-based model for price movement prediction

In [my previous post](https://mmalek06.github.io/python/gradient-boosting/optimization/2026/02/10/building-a-tree-based-model-training-pipeline-with-scoring.html) I described 95% of the pipeline I used to obtain a set of gradient-boosted models that would let me see into the stock market future. In that post I mentioned that trying to train a neural network gave me terrible results so I dropped the idea entirely. Turns out I made a couple of errors when trying to train NN the first time. I also noticed that the tree models were pretty highly correlated which means that they were making similar errors. I also trained them on data coming from a single, rather stable regime, so while I managed to earn some money in January, the moment USA/Israel war with Iran started was the moment when NASDAQ predictions started becoming more and more useless. I've been working on fixing that since February and while I had some success with GBM and extended dataset, what gave me the biggest boost was actually adding a neural net to the ensemble.

## The differences

In that past post I kept mentioning not wanting to disclose my secret-feature-recipe, although I guess pulling the courtain a bit won't harm. So the tree models all used basic technical indicators, like RSI or MACD. Obviously I used many more and the ones I just mentioned were not even in the top-5 most important features in any of the trained models, but they were there. However, I figured that for my neural network this may not be the best idea. My reasoning was that even if the underlying model architectures are different, the correlation may remain similar because of using similar features.

Another difference is that back then I decided to use a two-step model. The first step was supposed to be recall-oriented, the second one precision-oriented. It made sense back then, with the feature set I used, but in this iteration I simplified it as much as I could, and so I arrived at the idea of having Optuna search for hyperparameters that directly maximize precision@K per day. Not a loss function - the models still train on their usual internal losses (binary cross-entropy for the trees, `BCEWithLogitsLoss` with a `pos_weight` for the network) because P@K is not differentiable. It is the metric that drives the hyperparameter search on top of that training:

```python
def precision_at_k_per_day(
    dates: np.ndarray, 
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    k: int,
) -> float:
    """Mean of (top-K picks by prob that are positives) / K, averaged over days."""
    precisions = []

    for d in np.unique(dates):
        mask = dates == d
        day_prob = y_prob[mask]
        day_true = y_true[mask]
    
        if len(day_prob) < k:
            continue
    
        topk_idx = np.argsort(-day_prob)[:k]
        precisions.append(day_true[topk_idx].mean())
    
    return float(np.mean(precisions)) if precisions else 0.0
```

The last difference was that in the previous post I didn't actually achieve a proper meta-learner. 8/10 times, after retraining the base models, simple averaging over their predictions gave me better results than applying sklearn's `LogisticRegression` over them.

## Neural network architecture

Let's jump straight into the code:

```python
import torch
import torch.nn as nn

LOOKBACK = 20


class TemporalClassifier(nn.Module):
    def __init__(self, n_seq: int, n_static: int,
                 d_model: int = 64, n_head: int = 4, n_layers: int = 2, dim_ff: int = 128,
                 static_hidden: int = 64, head_hidden: int = 64,
                 dropout: float = 0.2, lookback: int = LOOKBACK):
        super().__init__()

        self.input_proj = nn.Linear(n_seq, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, lookback, d_model))
        
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.static_mlp = nn.Sequential(
            nn.Linear(n_static, static_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(static_hidden, static_hidden // 2),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model + static_hidden // 2),
            nn.Linear(d_model + static_hidden // 2, head_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(seq) + self.pos_embedding
        h = self.encoder(h)
        h_seq = h.mean(dim=1)
        h_stat = self.static_mlp(static)
        
        return self.head(torch.cat([h_seq, h_stat], dim=-1)).squeeze(-1)
```

### What each piece does

The model takes two inputs per sample:
- `seq` of shape `(batch, 20, n_seq)` - 20 days of history, a handful of raw-ish per-day signals.
- `static` of shape `(batch, n_static)` - today's cross-sectional features that either do not belong in a sequence or the network should not re-derive from returns (things like cross-sectional percentiles, event flags, and regressor outputs).

**`input_proj`** is a plain linear layer that projects the sequence features into the transformer's working dimension (`d_model`). Transformers internally work in a single hidden space for residual-connection bookkeeping, so raw features have to be lifted into that space first.

**`pos_embedding`** is a learnable `(1, 20, d_model)` tensor added to every sequence. Without it, self-attention has no idea which day came first because attention is permutation-equivariant. I chose a learnable one instead of sinusoidal because 20 positions is tiny and the network can memorize a better layout than a fixed scheme. Initializing with truncated normal of std 0.02 is the usual recipe from the original GPT-2 initialization so gradients at step zero stay sane.

**`TransformerEncoder`** is the PyTorch built-in. Two layers, 4 heads, feed-forward dimension of 128, GELU activation, pre-norm (`norm_first=True`) which makes training more stable for deep transformers. At each layer: multi-head self-attention lets every day attend to every other day within the 20-day window, and the FFN gives the network some per-position non-linearity. Dropout of 0.2 applies to attention weights and to the residual connections. `batch_first=True` just swaps dimensions so I do not have to transpose everything.

**Global average pool** (`h.mean(dim=1)`) collapses the sequence into a single `(batch, d_model)` summary. I tried attention pooling and a CLS token but they did not help and added complexity. Mean pool turned out to be fine.

**`static_mlp`** takes the static features through a 2-layer MLP with GELU and dropout. Output is `static_hidden // 2 = 32` dimensions. This is the "wide" side of a wide-and-deep style architecture - its role is to carry today's context that the sequence encoder should not have to re-derive.

**`head`** concatenates the sequence summary (`d_model = 64` dims) with the static summary (32 dims), normalizes, passes through one more MLP, and produces a single logit. I intentionally do not sigmoid here because the loss function is `BCEWithLogitsLoss` which is numerically more stable in logit space.

One design choice worth flagging: I deliberately kept the classical rolling technical indicators (the RSI/MACD family) OUT of the static channel. Those features are exactly what a transformer over a 20-day window should re-derive on its own, and I did not want to confuse the model by handing it both the raw series and the cooked version. Correlation with the tree models (which DO consume those indicators) stayed around 0.72-0.80 - low enough for the stacker to find diverse signal, high enough to tell me the network still learns the same underlying process.

I also experimented with a per-ticker embedding (an `nn.Embedding` lookup on ticker id concatenated to the static features, with a UNK row for tickers unseen in training). The intuition was that some tickers have persistent idiosyncratic behaviour the model could memorize. In practice it gave me essentially no gain on validation P@5, and on the test set it actually nudged correlation with the tree models slightly upward rather than down. I suspect it was just too much capacity for a dataset where the target is so rare and the regime shifts quickly. I dropped it from the final architecture, which is why you do not see it in the code above.

## The stacking story: correlations and LR weights

This is the part that actually made the ensemble work. After training the three trees plus the transformer I checked how similar their predictions were on the 2025 test set (917k ticker-day rows). The Pearson correlations between their probability outputs looked like this:

| pair | correlation |
|---|---|
| XGB vs CatBoost | 0.891 |
| XGB vs LightGBM | 0.886 |
| CatBoost vs LightGBM | 0.894 |
| NN vs XGB | 0.795 |
| NN vs CatBoost | 0.721 |
| NN vs LightGBM | 0.718 |
| NN vs avg-of-trees | 0.784 |

The trees are around 0.89 with each other - which is exactly why no learned meta-learner could beat simple averaging in the old pipeline. The transformer, by contrast, sits at 0.72-0.80 against each tree. Still borderline from a textbook stacking perspective (you would like < 0.7) but far enough away that the logistic regression has something to work with.

Even more interesting is the top-K overlap - what fraction of each model's top-K daily picks the other model also picks. At K=5 on the same test set:

| pair | top-5 overlap |
|---|---|
| LightGBM vs NN (K=20 trained) | 0.506 |
| CatBoost vs NN | 0.518 |
| XGB vs NN | 0.454 (with a K=5-retargeted variant) |
| CatBoost vs LightGBM | 0.585 |
| XGB vs CatBoost | 0.636 |
| XGB vs LightGBM | 0.638 |

So even though the NN vs XGB Pearson correlation is 0.82 (which looks "redundant" on paper), only 45% of their top-5 daily picks actually coincide. The network and the tree agree on what a probable winner looks like on average, but they disagree on 2-3 tickers out of 5 every day. That is the structural independence that the stacker exploits.

And here is how the logistic-regression meta-learner weighted things (on standardized inputs, so the magnitudes are comparable):

**Trees only (LR over the three tree models)**:

| feature | weight | absolute share |
|---|---|---|
| prob_xgb | +1.060 | 82% |
| prob_catboost | +0.072 | 6% |
| prob_lgbm | -0.160 | 12% (negative - contrarian) |

XGB dominated, CatBoost was basically dead weight, LightGBM got a small negative coefficient (the stacker actively subtracts probability when LightGBM rates a ticker high but the others do not).

**Trees + NN (the ensemble I actually deploy at K=5)**:

| feature | weight | absolute share |
|---|---|---|
| prob_xgb | +0.215 | 15% |
| prob_catboost | +0.060 | 4% |
| prob_lgbm | -0.071 | 5% (still negative) |
| prob_nn | +1.087 | 76% |

The transformer immediately took over as the dominant signal. XGB dropped from 82% of the weight to 15% - not because XGB got worse but because NN offered the same "main signal" with lower correlation to CatBoost and LightGBM, so the stacker could lean on NN without sacrificing the diversity the other models provided. CatBoost kept its near-zero weight. LightGBM kept its tiny negative role. This redistribution was very stable across 5-fold cross-validation on the validation set - coefficient-of-variation of every weight was below 0.1.

On the actual trading metric the numbers came out like this on 2025 test:

| approach | P@5 | basket mean | worst day | Sharpe-like |
|---|---|---|---|---|
| xgb solo | 0.516 | +8.96% | -16.77% | 0.999 |
| avg of 3 trees | 0.515 | +8.89% | -12.94% | 0.984 |
| LR over 3 trees | 0.523 | +9.19% | -16.39% | 0.911 |
| NN solo (K=5 retargeted) | 0.499 | +9.51% | -17.20% | 0.999 |
| **LR over 3 trees + NN** | **0.529** | **+10.10%** | **-14.54%** | **1.030** |

The last row is the one that matters. Better precision, better basket return, cleaner tail, best risk-adjusted return. Adding one transformer and one 4-input logistic regression on top pulled both metrics up in a way that any single base model could not.

I also re-ran the whole pipeline on 2026-Q1 data as a real out-of-sample test. That quarter was not a kind one (Iran war, tariff announcements, NASDAQ deeply red year-to-date). Both the K=5 and K=20 recipes held their Sharpe thresholds comfortably. The correlations even drifted slightly in our favour - the NN dropped another few percentage points of correlation against the trees under the hostile regime. Whatever the transformer is looking at, it is not a pure 2024-2025 artifact.

## Summary

I don't know about you, but this was rather unexpected for me, especially so, since my first NN-based experiment was that bad. Turns out that Transformers can be helpful in financial markets analysis too :)
