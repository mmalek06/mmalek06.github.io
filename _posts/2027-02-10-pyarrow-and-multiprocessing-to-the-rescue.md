---
layout: post
title: "PyArrow and multiprocessing to the rescue!"
date: 2027-09-12 00:00:00 -0000
categories:
    - python
    - apache-arrow
    - data-engineering
    - optimization
tags: ["python", "apache arrow", "pyarrow", "multiprocessing", "ipc"]
---

# PyArrow and multiprocessing to the rescue!

In the past few months, I switched my focus to the algotrading domain. Not that I expect to beat all the bots thrashing around on the stock market - it's just really interesting: both probing the stock market, seeing how my code can actually extract actionable information from raw price or volume signals, and working on the core algorithms that have to be implemented in every respectable algotrading app. For the one I'm creating right now, I chose the Python programming language, for better or worse. Worse being: god damn, how slow is that thing!

I mean, it doesn't surprise me - scripting languages have a notorious reputation for being slow. However, in my case, I couldn't just say "it is what it is" and move on, because that would mean hours or days spent waiting for the script to finish. And so I arrived at the idea of using Apache Arrow + Python multiprocessing to squeeze every last bit of performance from my financial feature engineering scripts.

## Why even bother?

Obviously, no one made me choose Python, so I might as well have built the features in a different language known to me - e.g. C#. I would love that; however, Python already has a ton of well-maintained and well-documented packages that my feature engineering scripts use extensively (NumPy, pandas, statsmodels - to name the three I use most often), and I didn't want to waste time reimplementing them.

## The context

Ok, so before I go into the technical details, a little bit of context. I got my hands on a huge intraday prices dataset for NASDAQ and NYSE (and many others; it's just that I mostly use these two). After the time I spent researching the algotrading topic, I came up with a few features I would build and feed into my models to learn from. A good magician never tells his tricks, so let's focus on only one feature group I use - one based on RSI/MACD indicators. Building them for the 5k++ NASDAQ tickers only takes around 40 seconds on my 28-core CPU, so it would be totally fine to run the logic without anything as advanced around it as Apache Arrow Arrow; however:

**I need to illustrate the point somehow, right? :)**

Also, this feature group is one of many - if I ran the "quick" ones synchronously, it would all add up to a few minutes, and the "slow" ones take more than an hour, each.

Besides that, I wanted to be able to run the feature-engineering logic both in a background job of my fastapi-app and in jupyter notebook, for quick'n'dirty experimentation purposes. The only way to achieve that was to put each feature-building logic into a separate python file and run it using the subprocessing module. That way the background job code could do something like this to call it:

```python
proc = await asyncio.create_subprocess_exec(
    sys.executable,
    "compute_rsi_macd.py",
    env=env,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
)
```

...and the notebook could just use the subprocess-blocking call directly.

Here comes the problem that's the reason for writing this blog post.

## How to pass big amounts of data between processes?

In case of small bits of data, python's pickle module, or just plain, old `json.dumps`/`json.load` would do. However, in this case, I noticed that my code is wasting a lot of time waiting for serialization/deserialization to finish. The problem grew the more data I tried to pass through.

<b>Side note:</b> you may be surprised, after all even full set of raw NASDAQ price data would only be a few MBs, The thing is that I have to pre-engineer it. I take the raw data and built something else out of it - something that's substantially bigger and used by all the second step feature engineering logic in the pipeline.

The only other way I could find to make it faster was to pass it via database or stick to a "local" way of doing it. I chose the "local" way, thus opening the door for Apache Arrow. From the docs:

> Apache Arrow defines a language-independent columnar memory format for flat and nested data, 
> organized for efficient analytic operations on modern hardware like CPUs and GPUs. 
> The Arrow memory format also supports zero-copy reads for lightning-fast data 
> access without serialization overhead.

The second sentence sounds like it could fit my use case perfectly - indeed, zero-copy and no serialization overhead is what I was after. Let's take a look next.


