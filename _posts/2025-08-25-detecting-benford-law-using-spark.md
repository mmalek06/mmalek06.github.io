---
layout: post
title: "Detecting Conformance to Benford's Law Using Apache Spark"
date: 2025-08-25 00:00:00 -0000
categories:
    - Python
    - Spark
    - Statistics
tags: ["python", spark", "scipy", "testing"]
---

# Detecting Conformance to Benford's Law Using Apache Spark

Before diving back into neural network topics, I figured I needed a bit of space to breathe - away from AI, transformers, and multiple-hour-long wait times. For that, I picked a simple idea: checking if a number series aligns with a certain statistical distribution. On its own, that would be a pretty basic task, so I decided to complicate it in two ways: by using Apache Spark and by going with a distribution that's not as widely known as, say, the normal distribution. Enter Benford's distribution - and Benford's law it's built upon.

Why Apache Spark and not Pandas? The reason is simple - laziness. I already had most of the code ready from one of my older passion projects. The only thing I had to change was swapping in Benford's distribution instead of the one I used back then.

<b>Side note</b>: from what I've seen on the internet, this is often used for fraud detection in banking, though it's by no means limited to that domain. In this post, I'll use some synthetic data to demonstrate the calculations.

## WTH is Benford's Law?!

First, some theory. Benford's Law states that if you take the first digit from each number in a series, the digits should appear with certain precalculated probabilities. For example, the digit 1 should appear about 30% of the time, digit 2 about 17.5%, all the way up to digit 9, which according to Benford's Law should appear only around 4.5% of the time. This corresponds to the Benford distribution known in statistics. . It's represented by this formula:

$$\begin{aligned}
P(d) = \log_{10}\left(1 + \frac{1}{d}\right), \quad d = 1, 2, \dots, 9
\end{aligned}$$

And looks like this:
<br />
<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/benford_distribution.png" /><br />

Values for that plot can be generated using this function:

```python
def generate_benford_values(n_days: int = 365 * 4) -> list[int]:
    multiplier = np.log(100)

    return list(
        np.exp(multiplier * np.random.rand(n_days)).astype(int)
    )
```

It actually produces a log-uniform distribution, but the leading digits conform to Benford's Law, that's why I used it.

<b>Side note</b>: obviously not all natural phenomena will produce number series conforming to this law, but some will (like cash flows being inspected in fraud detection algorithms).

## Testing

In data analysis looking at the data is the first step. In this case I didn't have any data, so I generated some for the experiment using these functions:

```python
def generate_uniform_values(n_data_points: int = 365 * 4) -> list[int]:
    values = []
    magnitudes = [4, 5, 6, 7]  # 10^4 to 10^7 range

    for _ in range(n_data_points):
        # Pick a random first digit with uniform probability
        first_digit = random.randint(1, 9)
        # Pick a random magnitude
        magnitude = random.choice(magnitudes)
        # Generate a number starting with that digit
        # For digit d and magnitude m: range is [d * 10^(m-1), (d+1) * 10^(m-1) - 1]
        min_val = first_digit * (10 ** (magnitude - 1))
        max_val = (first_digit + 1) * (10 ** (magnitude - 1)) - 1
        value = random.randint(min_val, max_val)

        values.append(value)

    return value
```

Why the 365 * 4 multiplication? In this experiment, I wanted a year's worth of data where each day produces 4 data points. As for the loop, it iterates over the given range, picks a random number between 1 and 9, chooses a magnitude (whether it's 10,000 or 10,000,000 doesn't really matter - everything could've just been within a fixed range like [1, 1000]), selects limits that should constrain the randomly chosen number, and appends it to a list.

Then, to visualize what's been produced I use this code:

```python
def get_first_digits(values: list[int]) -> tuple[list[int], list[int], list[int]]:
    first_digits = list(map(lambda x: int(str(x)[0]), values))
    digit_occurences = dict(Counter(first_digits))
    keys = list(digit_occurences.keys())
    values = list(digit_occurences.values())

    return first_digits, keys, values


u_values = generate_uniform_values()
u_first_digits, u_keys, u_values = get_first_digits(u_values)

plt.bar(u_keys, u_values)
```

I also created three more functions to generate variable set of test cases that I will later use:

```python
import numpy as np
import random

def generate_biased_values(n_days: int = 365 * 4) -> list[int]:
    scales = [1e4, 1e5, 1e6, 1e7]
    values = []

    for _ in range(n_days):
        scale = random.choice(scales)

        if random.random() < 0.7:  # natural distribution
            value = int(np.random.lognormal(mean=np.log(scale), sigma=1.5))
        else:  # biased branch
            magnitude = int(np.log10(scale))
            r = random.random()

            if r < 0.4:  # overrepresent digit 5
                value = random.randint(5_000_000, 5_999_999)
            elif r < 0.7:  # underrepresent digit 1 favoring 2, 3, 4
                digit = random.choice([2, 3, 4])
                value = random.randint(digit * 10 ** magnitude,
                                       (digit + 1) * 10 ** magnitude - 1)
            else:  # overrepresent digit 9
                value = random.randint(9 * 10 ** (magnitude - 1),
                                       10 ** magnitude - 1)

        value = max(10_000, min(100_000_000, value))

        values.append(value)

    return values


def generate_lognormal_values(n_days: int = 365 * 4) -> list[int]:
    return list(
        np.random.lognormal(mean=np.log(5000), sigma=0.25, size=n_days).astype(int)
    )


def generate_benford_values(n_days: int = 365 * 4) -> list[int]:
    multiplier = np.log(100)

    return list(
        np.exp(multiplier * np.random.rand(n_days)).astype(int)
    )
```

`generate_benford_values` has been described earlier, so I'll skip it here. As for the lognormal value generator - I chose lognormal because it generates strictly positive values. I could have just used a normal distribution, but it can produce negatives, even with a very large `loc` parameter. Since we'll be considering `str(x)[0]` in the first digit selection function, it's more intuitive to use something strictly positive from the start.

`generate_biased_values` produces values from a lognormal distribution 70% of the time, and 30% of the time it overrepresents certain numbers. The magnitude calculations are there only to make the overall number range more spread out, but it could just as well be simplified to use a fixed range (like in the `generate_uniform_values` function).

This is how the biased distribution can look like:

<br />
<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/biased_distribution.png" /><br />

I also defined a function to check what does the `scipy.stats.chisquare` function think about them:

```python
def get_chisquare(first_digits: list[int]) -> tuple[float, float]:
    benford_probs = get_benford_probabilities()
    observed_counts = np.zeros(9)

    for digit in first_digits:
        observed_counts[digit - 1] += 1

    expected_counts = benford_probs * len(first_digits)
    chi_stat, p_value = chisquare(
        f_obs=observed_counts,
        f_exp=expected_counts
    )

    return chi_stat, p_value
```

After feeding it first digits for each distribution I got these results:

- Uniform: (np.float64(701.2662324951431), np.float64(3.820331560700687e-146))
- Biased: (np.float64(329.04613868697993), np.float64(2.673073644629964e-66))
- Lognormal: (np.float64(2250.8987402440307), np.float64(0.0))
- Benford: (np.float64(10.862339416528645), np.float64(0.2096219090178658))

The `chisquare` test produced very high `chi_stat` and very low `p_value`'s for the first three ones which means that there's almost zero probability for them to come from benford distribution, contrary to the last one.

Since this whole idea came from one of my old apps, I also wanted to show a unit test I wrote, that utilizes the code show earlier. There will be some references to components that I haven't defined yet - I'll do that shortly:

```python
class MockSharedSparkSession(SharedSparkSession):
    def get(self, **kwargs) -> SparkSession:
        return (
            SparkSession
            .builder
            .master("local[*]")
            .appName("unit-test")
            .config("spark.sql.shuffle.partitions", "1")
            .config("spark.sql.adaptive.enabled", "false")
            .getOrCreate()
        )


spark_session = MockSharedSparkSession()


@pytest.fixture
def sut():
    return BenfordAnalysisCommand(spark_session)


def create_data_frame(volumes: list[int]) -> DataFrame:
    start_date = datetime.now(timezone.utc) - timedelta(days=365)
    data = []
    current_date = start_date

    for day in range(len(volumes) // 4):
        date_only = current_date.date()
        times = [
            current_date.replace(hour=10),
            current_date.replace(hour=12),
            current_date.replace(hour=14),
            current_date.replace(hour=16),
        ]
        day_values = values[day * 4:(day + 1) * 4]

        for ts_dt, value in zip(times, day_values):
            ts = int(ts_dt.timestamp() * 1000)
            data.append((
                date_only,
                ts,
                ts_dt,
                Decimal(str(value))
            ))

        current_date += timedelta(days=1)

    schema = StructType([
        StructField("date", DateType(), False),
        StructField("ts", LongType(), False),
        StructField("datetime_", TimestampType(), False),
        StructField("volume", DecimalType(20, 0), True),
    ])

    return spark_session.get().createDataFrame(data, schema)


@pytest.mark.parametrize(
    "data_frame,expected_label",
    [
        (create_data_frame(generate_uniform_volumes()), ticker, "high"),
        (create_data_frame(generate_biased_volumes()), ticker, "high"),
        (create_data_frame(generate_lognormal_volumes()), ticker, "high"),
        (create_data_frame(generate_benford_volumes()), ticker, "low"),
    ]
)
def test_benford_analysis_command_threshold(
        sut: BenfordAnalysisCommand,
        data_frame: DataFrame,
        expected_label: str,
        monkeypatch
):
    monkeypatch.setattr(sut, "_get_df_", lambda t: data_frame)

    result = sut.run(ticker)
    chi_series = result[0]["chi_sq_series"]
    chi_stat = float(chi_series[0])

    if expected_label == "high":
        assert chi_stat > 15, f"Chi-square {chi_stat} unexpectedly low for {ticker.code}"
    else:
        assert chi_stat < 15, f"Chi-square {chi_stat} unexpectedly high for {ticker.code}"

```
