---
layout: post
title: "Avoiding Python in Python!"
date: 2026-02-10 00:00:00 -0000
categories:
    - python
    - apache-arrow
    - data-engineering
    - optimization
tags: ["python", "apache arrow", "pyarrow", "polars", "ipc"]
---

# Avoiding Python in Python

I think that in 2026 everyone knows how slow Python can be. Sure, its rich ecosystem and convenient syntax allow for fast prototyping. It's fast enough for most web apps too, and I've also seen it being used in larger microservices-based architectures. If you use a smart caching strategy and are careful about your app's performance, you may never even get into trouble. If you're cloud-crazy and your client doesn't mind the cost of horizontal scaling, that can also save your job. However, there's one popular Python use case where you can optimize every whitespace in your code and still end up with bad performance - and that is data engineering (ok, you can do it on Databricks, but that costs a ton of gold for bigger workloads).

In my case, I was building a classic machine learning pipeline with a dozen scripts being run by the multiprocessing module, and I noticed that for some of the more complex features, building them takes even more time than actually training my XGB, LGBM, and CatBoost models. To be completely honest, it was mostly my fault, because I used Python's pickle module for passing the data between feature engineering steps, which might not be the most optimal way of doing it. How I was doing it, from a high level:

1. Load the data using the `psycopg` driver in the main script
2. Save it to the disk with `pickle`
3. Load it in the feature engineering scripts into `pandas` dataframes
4. Do magic

Turns out that's not the best-performing strategy one can come up with. In my case, I was also constrained by the hardware - it was an experiment I was running on my PC, so offloading it to Databricks was not possible.

<b>Side note #1:</b> I didn't gather the times it took to write and read using `pickle`, so I'll only pit the alternatives against themselves.
<b>Side note #2:</b> in the next few examples you'll see something silly - loading the data from db, then immediately writing it to the disk or shared memory - that's only to illustrate my point. In the original code I wrote it wasn't so straightforward. There were some initial data transformations that justified loading the data in the main script, then saving it for the other ones to process.
<b>Side note #3:</b> each time measurement I present in this post was executed several times to ensure I'm not believing the outliers; I just didn't put that looping code.
<b>Side note #4:</b> every code snippet here expects imports like this to be present:

```python
import os
import time
from multiprocessing import shared_memory

import adbc_driver_postgresql.dbapi
import dotenv
import numpy as np
import pandas as pd
import polars as pl
import psycopg2
import pyarrow as pa
import pyarrow.parquet as pq
```

## How about Pandas + parquet?

I was using Pandas anyway, so why even bother loading into some immediately value if it can all be put into a `DataFrame`? 

```python
t0 = time.perf_counter()

with adbc_driver_postgresql.dbapi.connect(os.environ["PG_URL"]) as conn:
    df = pd.read_sql(QUERY, conn)

n_rows = len(df)
t_query = time.perf_counter()
parquet_path = "experiment_pandas.parquet"

df.to_parquet(parquet_path, index=False)

t_write = time.perf_counter()
file_size = os.path.getsize(parquet_path)

del df

print(f"Rows:          {n_rows:,}")
print(f"Output:        {file_size / 1024**2:.1f} MB (Parquet)")
print(f"---")
print(f"DB query:      {t_query - t0:.2f}s")
print(f"to_parquet:    {t_write - t_query:.2f}s")
print(f"---")
print(f"Total:         {t_write - t0:.2f}s")
```

What's `adbc_driver_postgresql` you ask? It's `Arrow Database Connectivity`. Since Apache Arrow can be used as Pandas engine in some cases, why not use it for loading? Doing that makes sure that the data goes straight from libpq -> Arrow buffers (C++). No Python code involved, except for the glue code. The result?

```plaintext
Rows:          10,114,549
Output:        260.8 MB (Parquet)
---
DB query:      24.08s
to_parquet:    4.56s
---
Total:         28.64s
```

The `to_parquet` function call takes 4.56s which may not be a lot in this specific case, but in my code it was still too slow. Can it be made better? Initially I thought that since the data pulled in from db is Arrow, Pandas will be smart enough to use it for writing too, but it was not, as proven by this change:

```python
df.to_parquet(parquet_path, index=False, engine="pyarrow")
```

The result:

```plaintext
Rows:          10,114,549
Output:        260.8 MB (Parquet)
---
DB query:      24.60s
to_parquet:    1.83s
---
Total:         26.43s
```

But can it be better?

## How about pure Arrow?

You can transform your data when it's kept in the pure Arrow form too - you don't need Pandas for that. That is, for some operations. `pyarrow` doesn't implement rolling windows and many other complex ones. But in my case, I only did some simple transformations in the pre-feature-engineering step, so keeping everything in Arrow on this stage was a viable option for me.

```python
t0 = time.perf_counter()

with adbc_driver_postgresql.dbapi.connect(os.environ["PG_URL"]) as conn:
    with conn.cursor() as cur:
        cur.execute(QUERY)
        table = cur.fetch_arrow_table()

n_rows = table.num_rows
t_query = time.perf_counter()
parquet_path = "experiment_arrow.parquet"

pq.write_table(table, parquet_path)

t_write = time.perf_counter()
file_size = os.path.getsize(parquet_path)

del table

print(f"Rows:          {n_rows:,}")
print(f"Output:        {file_size / 1024**2:.1f} MB (Parquet)")
print(f"---")
print(f"DB query:      {t_query - t0:.2f}s")
print(f"Parquet write: {t_write - t_query:.2f}s")
print(f"---")
print(f"Total:         {t_write - t0:.2f}s")
```

The result:

```plaintext
Rows:          10,114,549
Output:        260.8 MB (Parquet)
---
DB query:      30.61s
Parquet write: 1.40s
---
Total:         32.01s
```

1.83s VS 1.40s - a very small difference, but can it be made even better?

## How about Arrow + SHM? (when no compression is needed)

SHM stands for shared memory. Python's `multiprocessing.shared_memory` module gives you access to it. What it allows you to do is to save the data not to the disk, but directly to RAM and then read it from there. Technically, it mounts RAM as part of the file system. The thing with RAM is that normally it's not shared between processes. So if you need some scripts that are running separately to share some data - like the feature engineering scripts in my case - you have to put it in a place that's accessible by all, e.g. a database. However, there's a simpler, more performant option (at least on Linux): shared memory.

```python
SHM_NAME = "experiment_shm"
QUERY = (
    "SELECT ticker_code, exchange, date, "
    "open_::float8, avg_high::float8, avg_low::float8, close::float8, avg_volume::float8 "
    "FROM daily_prices WHERE exchange = 'NASDAQ'"
)
t0 = time.perf_counter()

with adbc_driver_postgresql.dbapi.connect(os.environ["PG_URL"]) as conn:
    with conn.cursor() as cur:
        cur.execute(QUERY)
        table = cur.fetch_arrow_table()

n_rows = table.num_rows
t_query = time.perf_counter()
sink = pa.BufferOutputStream()

with pa.ipc.new_stream(sink, table.schema) as writer:
    writer.write_table(table)

buf = sink.getvalue()
buf_bytes = buf.to_pybytes()
buf_size = len(buf_bytes)
t_serialize = time.perf_counter()
shm = shared_memory.SharedMemory(create=True, size=buf_size, name=SHM_NAME)
shm.buf[:buf_size] = buf_bytes
t_write = time.perf_counter()

del buf_bytes, table
shm.close()

print(f"Rows:          {n_rows:,}")
print(f"Output:        {buf_size / 1024**2:.1f} MB (Arrow IPC)")
print(f"---")
print(f"DB query:      {t_query - t0:.2f}s")
print(f"IPC serialize: {t_serialize - t_query:.2f}s")
print(f"SHM write:     {t_write - t_serialize:.2f}s")
print(f"---")
print(f"Total:         {t_write - t0:.2f}s")
```

This saves raw bytes to the RAM. The results:

```plaintext
Rows:          10,114,549
Output:        597.6 MB (Arrow IPC)
---
DB query:      25.89s
IPC serialize: 0.62s
SHM write:     0.22s
---
Total:         26.72s
```

Saving the data goes below one second. Pretty awesome if you ask me. BUT CAN IT BE MADE ANY BETTER??? 

No :)

At least I couldn't make it any better, so I'll move on to the reading part.

## A little bit of context

Like I mentioned in the beginning - I was trying to calculate some features for my ML pipeline - and I was doing it in separate scripts. One such feature group was RSI and MACD indicators which are well known for anyone interested in technical analysis of the stock market. These are my base implementations:

```python
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def compute_rsi_series(prices: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = prices.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    avg_gain = gains.ewm(span=period, adjust=False).mean()
    avg_loss = losses.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_macd_series(prices: pd.Series) -> pd.DataFrame:
    ema_fast = prices.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = prices.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    macd_hist = macd_line - signal_line

    return pd.DataFrame(dict(macd_line=macd_line, macd_signal=signal_line, macd_hist=macd_hist), index=prices.index)
```

They are used after reading the data with code like this. Here I should mention that this only applies to reading from the shared memory buffer written in the last write step. As you can suspect - getting the data this way also performs best, so I omitted the other two ways of doing it:

```python
t0 = time.perf_counter()
shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
reader = pa.ipc.open_stream(shm.buf)
table = reader.read_all()
t_read = time.perf_counter()
df = table.to_pandas()

del table, reader

t_convert = time.perf_counter()

df.sort_values(["ticker_code", "date"], inplace=True, ignore_index=True)

grouped = df.groupby("ticker_code")["close"]
df["rsi"] = grouped.transform(compute_rsi_series)
macd = grouped.apply(compute_macd_series)
df[["macd_line", "macd_signal", "macd_hist"]] = macd.reset_index(level=0, drop=True)
t_compute = time.perf_counter()

print(f"Rows:          {len(df):,}")
print(f"---")
print(f"SHM read:      {t_read - t0:.2f}s")
print(f"to_pandas:     {t_convert - t_read:.2f}s")
print(f"RSI+MACD:      {t_compute - t_convert:.2f}s")
print(f"---")
print(f"Total:         {t_compute - t0:.2f}s")
shm.close()
```
```plaintext
Rows:          10,114,549
---
SHM read:      0.00s
to_pandas:     0.14s
RSI+MACD:      6.66s
---
Total:         6.79s
```

It's not an awful result, and the read itself is not the bottleneck - performing the calculations is. But why? 

Well, the thing is that it involves significant Python-level overhead - the groupby/apply/transform dispatch runs in Python, is bound by the GIL, and processes each group sequentially in a single thread. And it runs over data sitting in Python's process memory.

## How about Polars?

And so we arrived at a moment when I thought about a library I heard about a long time ago and almost completely forgot about it, deeming it a novelty. Turns out, it can make all the difference in the world. 

First off: Arrow promises zero-copy operations in some cases. Zero-copy means that data can be passed between libraries without copying the underlying memory buffers. <i>And it's almost true in this case</i> :) `pa.ipc.open_stream(shm.buf) + reader.read_all()` deserializes the data from raw bytes, so there's some copying involved, but then the `pl.from_arrow(table)` is zero-copy, so the subsequent operations run over that data deserialized a step earlier.

Second off: it runs its stuff with Rust speed, leveraging multithreading.

So the whole pipeline is native, with Python only instructing the underlying, highly-performant mechanisms:

```plaintext
ADBC (C/C++) -> Arrow buffers -> zero-copy -> Polars (Rust) -> compute
```

```python
t0 = time.perf_counter()
shm = shared_memory.SharedMemory(name=SHM_NAME, create=False)
reader = pa.ipc.open_stream(shm.buf)
table = reader.read_all()
t_read = time.perf_counter()
df = pl.from_arrow(table)

del table, reader

t_convert = time.perf_counter()
df = (df.sort("ticker_code", "date")
.with_columns(
    rsi_expr().over("ticker_code"),
    macd_line_expr().over("ticker_code"),
).with_columns(
    macd_signal_expr().over("ticker_code"),
).with_columns(
    (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_hist"),
))
t_compute = time.perf_counter()

print(f"Rows:          {len(df):,}")
print(f"---")
print(f"SHM read:      {t_read - t0:.2f}s")
print(f"to_polars:     {t_convert - t_read:.2f}s")
print(f"RSI+MACD:      {t_compute - t_convert:.2f}s")
print(f"---")
print(f"Total:         {t_compute - t0:.2f}s")
df.head()
```

The results? 6x faster in terms of RSI+MACD calculations:

```plaintext
Rows:          10,114,549
---
SHM read:      0.00s
to_polars:     0.11s
RSI+MACD:      1.01s
---
Total:         1.12s
```

However, to make it work that well, a bunch of changes to the function definitions were required.

```python
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def rsi_expr(col: str = "close", period: int = RSI_PERIOD) -> pl.Expr:
    delta = pl.col(col).diff()
    gains = pl.when(delta > 0).then(delta).otherwise(0.0)
    losses = pl.when(delta < 0).then(-delta).otherwise(0.0)
    avg_gain = gains.ewm_mean(span=period, adjust=False)
    avg_loss = losses.ewm_mean(span=period, adjust=False)

    return (100 - 100 / (1 + avg_gain / avg_loss)).alias("rsi")


def macd_line_expr(col: str = "close") -> pl.Expr:
    ema_fast = pl.col(col).ewm_mean(span=MACD_FAST, adjust=False)
    ema_slow = pl.col(col).ewm_mean(span=MACD_SLOW, adjust=False)

    return (ema_fast - ema_slow).alias("macd_line")


def macd_signal_expr(col: str = "macd_line") -> pl.Expr:
    return pl.col(col).ewm_mean(span=MACD_SIGNAL, adjust=False).alias("macd_signal")


def macd_hist_expr() -> pl.Expr:
    return (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_hist")
```

I guess this requires some explanation, especially in the code that does the reads. Notice the call to the `.over` method? That's like SQL's `OVER (PARTITION BY ticker_code)`. What it means is it partitions/groups the original dataframe by `ticker_code`, then it evaluates whatever expressions come before it independently, within each partition; then it aggregates the results. So basically expressions like this `rsi_expr().over("ticker_code")` would translate to pandas: `groupby("ticker_code")["close"].transform(compute_rsi_series)`. Polars adds multithreading to the whole process and some other fanciness that is invisible to us but makes all the difference (precomputed execution trees).

I already know that I won't forget about using Polars in my upcoming projects if only I can. And that's all folks!
