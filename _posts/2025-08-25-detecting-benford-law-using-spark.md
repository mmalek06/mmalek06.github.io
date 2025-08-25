---
layout: post
title: "Detecting Conformance to Benford's Law Using Apache Spark"
date: 2025-08-25 00:00:00 -0000
categories:
    - Python
    - Spark
    - Statistics
tags: ["python", spark", "scipy"]
---

# Detecting Conformance to Benford's Law Using Apache Spark

Before diving back into neural network topics, I figured I needed a bit of space to breathe - away from AI, transformers, and multiple-hour-long wait times. For that, I chose an idea I came across somewhere: checking whether a series of numbers conforms to Benford's Law. From what I've seen on the internet, this is often used for fraud detection in banking, though it's by no means limited to that domain. In this post, I'll use some synthetic data to demonstrate the calculations.

## WTH is Benford's Law?!

First, some theory. Benford's Law states that if you take the first digit from each number in a series, the digits should appear with certain precalculated probabilities. For example, the digit 1 should appear about 30% of the time, digit 2 about 17.5%, all the way up to digit 9, which according to Benford's Law should appear only around 4.5% of the time. This corresponds to the Benford distribution known in statistics. It's represented by this formula:

$$\begin{aligned}
P(d) = \log_{10}\left(1 + \frac{1}{d}\right), \quad d = 1, 2, \dots, 9
\end{aligned}$$

And looks like this:
<br />
<img style="display: block; margin: 0 auto;" src="https://mmalek06.github.io/images/benford_distribution.png" /><br />
