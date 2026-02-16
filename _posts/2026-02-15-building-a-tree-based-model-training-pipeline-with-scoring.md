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

#3 Maybe less obviously, but still — finding the correct target for your models can also be a huge effort, and defining it as simply as "I will short like crazy, so give me 10,000 tickers that are about to implode tomorrow, so I can sell a truckload of outrageously overpriced calls and pretend margin requirements are just a social construct" won't work and will make you lose your house, your wife, dog, and kids. Don't try this at home!

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

But how do these two algorithms even decide whether or not to split (XGB) or where to split (LGBM). They both use the gain metric for that. That's also another big and important difference between the two. I can summarize it like this: XGB uses the gain metric to limit the splits while LGBM uses the same metric to find the leaf to split on.

As for CatBoost and its tree building approach: it uses a strategy called "symmetric trees" (or "oblivious trees") by default. This means that at each level of the tree, the same split condition is used for all nodes. Sounds limiting, but it leads to very fast inference and acts as a natural form of regularization:

```plaintext
        [age > 30?]
        /         \
   [yes]          [no]
   /    \         /    \
[inc>50k?]      [inc>50k?] <- same split across the entire level
```

As for regularization, CatBoost offers a similar set of parameters:

- `l2_leaf_reg` (equivalent to lambda/lambda_l2)
- `min_data_in_leaf`
- `max_depth` (defaults to 6, same as XGBoost)
- `num_trees`

There's no direct equivalent of `gamma`/`min_gain_to_split`. Instead, CatBoost relies more on symmetric trees and a technique called "ordered boosting", which prevents prediction shift (a type of overfitting specific to gradient boosting).

One more thing worth noting: CatBoost defaults to `grow_policy='SymmetricTree'`, but you can switch to `Depthwise` (like XGBoost) or `Lossguide` (like LightGBM). So technically you can get the behavior of all three libraries in one, it's just the default setting that differs.

Ok, that was a huge side note, so let's look at the code.
