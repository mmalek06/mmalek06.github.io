---
layout: post
title: "Building a tree-based model training pipeline with scoring"
date: 2026-02-10 00:00:00 -0000
categories:
    - python
    - gradient-boosting
    - optimization
tags: ["python", "xgboost", "lightgbm", "catboost", "optuna", "mlflow"]
---

# Building a tree-based model training pipeline with scoring

In [my previous post](https://mmalek06.github.io/python/apache-arrow/data-engineering/optimization/2026/02/10/avoiding-python-in-python.html) I mentioned that I have been involved in algotrading software that is supposed to predict price movements in the stock market. Despite what I've seen some people say on the internet, it's not an effort completely doomed to fail.

#1 Obviously, if it were easy, everyone would already be doing it, and I've stepped on metaphorical LEGO bricks more than 99,999 times so far, but I also managed to discover that there are things in algotrading that work surprisingly well.

#2 Obviously, the degree of your success will also be heavily dependent on the quality and amount of data that you use - writing web scrapers, if you don't want to pay for a commercially available API, is another huge subdomain in algotrading.

#3 Maybe less obviously, but still - finding the correct target for your models can also be a huge effort, and defining it as simply as "I will short like crazy, so give me 10,000 tickers that are about to implode tomorrow, so I can sell a truckload of outrageously overpriced calls and pretend margin requirements are just a social construct" won't work and will make you lose your house, your wife, dog, and kids. Don't try this at home!

#4 obviously I won't share the details about what features I use, or other technicalities, because if you do the same, my strategy may stop working! :)

## Requirements

Requirements can be laid out simply as building an ML training pipeline that will find the best types/sets of types/hierarchies of models. Before I even started working on this specific piece, I experimented quite a lot in Jupyter Notebooks, exploring how far I could get with XGBoost, LightGBM, and CatBoost on their own. Then I figured out that ensemble methods might work better, since all three are similar in spirit but different in implementation, and perhaps they make different mistakes.

And then, when I saw that my results were basically shit, I sat down and thought even more to find a better target variable. When I came up with one (let's say it was simply 0 for the price going down the next day vs. 1 for it going up) and experimented some more, it became clear that I needed a two-stage model. One (or an ensemble) would aim for a decent 70% recall at 30% precision, while the second one would optimize only for precision.

I also used the stacking technique, where a meta-model at the end learns how to assess the three models' predictions if an ensemble has been used; however, it has yet to give me a substantial boost in performance. It's still a work in progress, so I'll just describe it as it is now.

## Why those specific models?

Well, you might have already noticed that I'm a neural network nerd, so why would I even look at gradient boosting on decision trees techniques, right? I'm pretty sure there are some rich people out there, owning IT companies that hire people way smarter than me have figured out how to train neural nets on stock market data. But me - I haven't. I tried but my results were so horrible that I would be better off just flipping a coin. 

Then I did my research and saw that a lot of people in the quant/algotrading space use XGBoost for their work. Perhaps they are as secretive as I am and they use a lot more (maybe even neural nets or witchcraft, who knows!), but the most prevalent information that I managed to find was screaming XGB at me. You could say that the internet terrorized me into using gradient boosting in this project.

<b>Side note:</b> I was also thinking for a second about using sklearn, but since they don't support GPU training it wasn't a viable candidate.

## What are the differences?

As it goes for ML, when you see similar libraries, the differences between them are usually very technical and nuanced, and it's no different this time, so I won't really go into the details. The thing is, those details usually don't matter much when you apply the techniques. With dataset A, you'll get great results with XGBoost. With a similar but slightly different dataset B, you'll get better results using either of the other two, and so on. So how do I find the best model for my dataset? I run Optuna over a rich search space until it finds something that meets the requirements. Everybody does it :)

But let's lay out at least some minimal technical details. It is said that XGBoost heavily regularizes its trees to avoid overfitting, and it builds them level-wise, which means it builds all the nodes on a given level before moving on. In other words, it tries to find all the splits on one level, thus avoiding missing something important. XGB trees are more balanced, possibly less deep. LightGBM, on the other hand, builds its trees leaf-wise which means that on each iteration it picks a leaf with the greatest gain and it only splits on that one. Thus, LGBM trees may be unbalanced and very deep. It's also said this often leads to faster training and better overall results. But in my case, it didn't train faster, nor did it lead to great results. In my case the standalone LightGBM loses most of the time when pitted against XGBoost or CatBoost; however, it still manages to bring some value to the table. As for CatBoost - I just needed a third model that would serve as a tie breaker. Turns out, I kept losing some predictions in ensemble methods, because XGB and LGBM were not agreeing quite often and it got better after mixing in the third model.

<b>Side note:</b> ok, first and foremost I'm a curious programmer, so I couldn't leave out one technical detail I only hinted at above: the regularization and tree balancing. So the thing is that XGB implements L1 and L2 regularization in its classic form and it's used on the level of a tree. You can use these params for regularization:

- `lambda (L2)`
- `alpha (L1)`
- `gamma (min split loss)`
- `min_child_weight`
- `max_depth`

L1 regularization is turned off by default and L2 is turned on. What it all leads to is: XGB trees are often more balanced than LGBM trees. XGB may decide not to split on leaf A, while it still splits on leaf B, but in general it will happen less often than in the case of LGBM. This behavior is an artifact of the architectures both use.

LGBM mostly uses the same regularization math, however it calls some params differently, and also adds some of its own:
 
- `lambda_l1`
- `lambda_l2`
- `min_data_in_leaf`
- `min_gain_to_split`
- `max_depth`
- `num_leaves`

L1 and L2 are turned off by default. The `min_gain_to_split` plays a role that is similar to XGBoost's `gamma` - if the value is too low, there's no split, however, it's also turned off by default. Also, since it builds its trees leaf-wise, regularization relies more on architectural constraints by default - you can set the `num_leaves` and `min_data_in_leaf` params exactly for that, but even if you switch on all the regularization knobs, you'll be getting less balanced trees most of the time.

But how do these two algorithms even decide whether or not to split (XGB) or where to split (LGBM). They both use the gain metric for that. That's also another big and important difference between the two. I can summarize it like this: XGB uses the gain metric to limit the splits while LGBM uses the same metric to find the best leaf (the one with the highest gain) to split on.

Anyway, the end effect is that an LGBM tree may look like this:

```plaintext
        [root]
       /      \
     [A]      [B]
    /   \       
  [C]   [D]
         |
        [E]
```

...and XGB tree can look more like this:

```plaintext
        [root]
       /      \
     [A]      [B]
    /   \    /   \
  [C]  [D] [E]  [F]
```

As for CatBoost and its tree building approach: it uses a strategy called "symmetric trees" (or "oblivious trees") by default. This means that at each level of the tree, the same split condition is used for all nodes. Sounds limiting, but it leads to very fast inference and acts as a natural form of regularization:

```plaintext
        [age > 30?]
        /         \
   [yes]          [no]
   /    \         /   \
 [inc>50k?]     [inc>50k?] <- same split across the entire level
```

Looking at the graph, it looks kind of similar to the one for XGB; however, the difference is in the split condition.

As for regularization, CatBoost offers a similar set of parameters:

- `l2_leaf_reg` (equivalent to lambda/lambda_l2)
- `min_data_in_leaf`
- `max_depth` (defaults to 6, same as XGBoost)
- `num_trees`

There's no direct equivalent of `gamma`/`min_gain_to_split`. Instead, CatBoost relies more on symmetric trees and a technique called "ordered boosting", which prevents prediction shift (a type of overfitting specific to gradient boosting).

One more thing worth noting: CatBoost defaults to `grow_policy='SymmetricTree'`, but you can switch to `Depthwise` (like XGBoost) or `Lossguide` (like LightGBM). So technically you can get the behavior of all three libraries in one, it's just the default setting that differs.

Ok, that was a huge side note, so now let's look at the code, kind of...

## Time series split

There's so much stuff in this project of mine that I didn't know where to start, so I just picked the first topic at random. It's something specific to making predictions based on time series: splitting the data so that there's no look-ahead bias/error (sometimes known as temporal leakage).

This one is pretty easy to understand. Time series are time-ordered; therefore, the process of splitting the data into train–val–test datasets needs to be time-aware. In my case, I'm taking the last N months of data for training, then 1/3 * N of that for validation, and the same amount for the test set. To illustrate: I could use all 12 months from 2024 as my training dataset, then use the first 4 months of 2025 as the validation dataset and the next 4 (so beginning in May 2025) as the test dataset.

However, there's a subtlety hidden in the numbers - one that I didn't realize until I passed my code through an LLM :)

Let's say I'm training my models on the last 7 days of log returns and predicting the returns on the 8th day. Then some of the information from the last days of my training set will leak into the first days of my validation set. Specifically, the first week of the validation set overlaps with the last week of the training set. Luckily, there's a well-known strategy for dealing with this issue: using a purge gap. It's quicker to visualize than to describe:

```plaintext
|------ TRAIN ------|-- 7 days --|------ VAL ------|-- 7 days --|------ TEST ------| 
```

However, this leads to another subtle issue. What if your features are not as simple as log returns? What if you use features that need 90 days of prior data? Then the purge gap would have to be equal to 90. If you have a limited dataset, that would mean losing a lot of data, so it may be better to split it by ticker - tickers that go into the training dataset don't go into validation or test at all. It's called a cross-sectional split; however, you need to be aware that it still contains some temporal elements - market microstructure effects and cross-correlation between tickers.

But that doesn't really solve the issue with small datasets - you may simply not be able to afford this. So there's another - some may say questionable - strategy: accept the overlap and the fact that the evaluation metrics may be overly optimistic. You may use a slightly smaller purge gap, accept the risk, and understand that when doing empirical tests (like paper trading), your model may slowly start to "degenerate" over time because the data it sees now (for the long features) starts to diverge from what was embedded in the training set.

## Preparing the data for cross validation

Did I mention, we're working on time-series data? :)

Cross-validation is a very popular technique for model evaluation (called K-Fold CV). However, it would wreak havoc in this project, because it would break the temporal ordering and suddenly my models would be able to access data from the future. For that some smart people invented what's called Walk-Forward CV:

```plaintext
Fold 1: |---- TRAIN ----|purge|-- VAL --|
Fold 2: |------- TRAIN -------|purge|-- VAL --|
Fold 3: |---------- TRAIN ----------|purge|-- VAL --|
```

Each subsequent fold is larger to make sure the model performs well on all possible windows, not only one. Also, as you can see, there's no mixing of the temporally unavailable data that would occur had I used K-Fold:

```plaintext
Data:    [----1----|----2----|----3----|----4----|----5----]                                                        
                
Fold 1:  [   VAL   |  TRAIN  |  TRAIN  |  TRAIN  |  TRAIN  ]                                                        
Fold 2:  [  TRAIN  |   VAL   |  TRAIN  |  TRAIN  |  TRAIN  ]                                                        
Fold 3:  [  TRAIN  |  TRAIN  |   VAL   |  TRAIN  |  TRAIN  ]
```

Finally the code:

```python
class WalkForwardFold(NamedTuple):
    fold_idx: int
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def create_walk_forward_splits(
    df: pd.DataFrame,
    n_folds: int = DEFAULT_WF_FOLDS,
    val_months: int = DEFAULT_WF_VAL_MONTHS,
    purge_days: int = PURGE_GAP_DAYS,
    date_column: str = "target_date",
) -> list[WalkForwardFold]:
    """
    Create walk-forward validation splits from a DataFrame.

    Walk-forward validation trains on expanding windows and validates on
    subsequent periods, mimicking real-world model deployment:

    Fold 1: |---- TRAIN ----|purge|-- VAL --|
    Fold 2: |------- TRAIN -------|purge|-- VAL --|
    Fold 3: |---------- TRAIN ----------|purge|-- VAL --|
    ...

    Args:
        df: DataFrame with date column, sorted by date
        n_folds: Number of walk-forward folds
        val_months: Length of each validation period in months
        purge_days: Gap between train and val to prevent leakage
        date_column: Name of the date column

    Returns:
        List of WalkForwardFold tuples with train/val DataFrames and date ranges
    """
    df = df.copy()
    df["_wf_date"] = pd.to_datetime(df[date_column])
    df = df.sort_values("_wf_date").reset_index(drop=True)
    min_date = df["_wf_date"].min()
    max_date = df["_wf_date"].max()
    total_days = (max_date - min_date).days
    val_days = val_months * 30
    purge_td = pd.Timedelta(days=purge_days)
    val_td = pd.Timedelta(days=val_days)
    total_val_space = n_folds * val_days + n_folds * purge_days
    min_train_days = 180  # At least 6 months of training data for first fold

    if total_val_space + min_train_days > total_days:
        raise ValueError(
            f"Not enough data for {n_folds} folds with {val_months} month validation periods. "
            f"Total days: {total_days}, required: {total_val_space + min_train_days}"
        )

    folds = []
    # Calculate fold boundaries working backwards from max_date
    # Each fold's validation period is val_months long
    # Folds are spread evenly across the available validation space
    available_for_val = total_days - min_train_days
    step_days = (available_for_val - val_days) // (n_folds - 1) if n_folds > 1 else 0

    for i in range(n_folds):
        val_end = max_date - pd.Timedelta(days=(n_folds - 1 - i) * step_days)
        val_start = val_end - val_td + pd.Timedelta(days=1)
        train_end = val_start - purge_td - pd.Timedelta(days=1)
        train_start = min_date
        train_mask = df["_wf_date"] <= train_end
        val_mask = (df["_wf_date"] >= val_start) & (df["_wf_date"] <= val_end)
        df_train = df[train_mask].drop(columns=["_wf_date"]).reset_index(drop=True)
        df_val = df[val_mask].drop(columns=["_wf_date"]).reset_index(drop=True)

        if len(df_train) < 100 or len(df_val) < 50:
            continue

        folds.append(WalkForwardFold(
            fold_idx=i,
            df_train=df_train,
            df_val=df_val,
            train_start=pd.Timestamp(train_start),
            train_end=pd.Timestamp(train_end),
            val_start=pd.Timestamp(val_start),
            val_end=pd.Timestamp(val_end),
        ))

    return folds
```

It first makes sure the dataframe is sorted properly and selects the first and last date that it contains along with some other variables used lated in the loop. It also calculates `step_days` - the loop will use its value to move the window. And the loop itself goes from the max date backwards, building folds in the process. That if statement it contains is an example of defensive programming. Even though the date ranges are calculated in a way that prevents creating folds that are too small in terms of those ranges, a fold can still contain not enough data just because the data itself may be less dense in that range.

## The training, finally!

Before I jump on the code, I'll show you how the experiment is saved in mlflow - that will give you a better outlook of why I'm doing certain things the way I'm doing them:

<p>
    <img src="https://mmalek06.github.io/images/mlflow_with_models_collapsed.png">
    <br />
    <img src="https://mmalek06.github.io/images/mlflow_with_models_expanded.png">
</p>

The first image shows all the steps and the second one includes the model training substeps. There are two stages in the training, like I mentioned in the requirements section of this post, but what I didn't mention there is that the second stage also has two branches. The dependent model trains on the outputs from s1 model and the independent one, well, it's independent so it trains on the same data that's used by s1 training, but it optimizes for a different metric.

After all base models are trained, I'm also training some stacking meta-models that are supposed to further improve the initial predictions.

Now the main training function:

```python
async def train_full_pipeline(
        exchange: str,
        months_of_data: int,
        val_months: int = 12,
        test_months: int = 12,
        n_trials_s1: int = 350,
        n_trials_s2: int = 350,
        data_to: date | None = None,
) -> PipelineResult:
    run_name = f"full_pipeline_{exchange}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(dict(
            months_of_data=months_of_data,
            val_months=val_months,
            test_months=test_months,
            n_trials_s1=n_trials_s1,
            n_trials_s2=n_trials_s2,
            exchange=exchange,
        ))
        if data_to is not None:
            mlflow.log_param("data_to", str(data_to))

        trained_models, df_test = await _train_models_only(
            months_of_data=months_of_data,
            val_months=val_months,
            test_months=test_months,
            n_trials_s1=n_trials_s1,
            n_trials_s2=n_trials_s2,
            exchange=exchange,
            parent_run_id=run.info.run_id,
            data_to=data_to,
        )

        with mlflow.start_run(run_name="evaluation", nested=True):
            all_evaluations = evaluate_models_on_data(trained_models, df_test)
            best = select_best_pipeline(all_evaluations)

            _log_evaluation_to_mlflow(all_evaluations, best, trained_models)

    return PipelineResult(
        trained_models=trained_models,
        best_s1_mode=best.s1_mode,
        best_s2_mode=best.s2_mode,
        best_s2_training_mode=best.s2_training_mode,
        all_evaluations=all_evaluations,
        best_metrics=best.metrics,
        best_validation=best.validation,
    )
```

I think it's rather cleanly written, but just for completeness, the most important parts are:

1. start mlflow run under which all will be put
2. train the models
3. run evaluation
  3.1 evaluate the models on test data
  3.2 save the best models (which I'm calling pipeline here)
