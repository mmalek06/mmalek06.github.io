---
layout: post
title: "Denoising autoencoders, Part 1"
date: 2026-11-27 00:00:00 -0000
categories:
    - python
    - computer-vision
tags: ["python", "pytorch", "transfer learning", "computer vision", "math"]
---

# Denoising autoencoders, Part 1

I think the DAE concept was one of the first things I saw in deep learning that made me so interested in it. If you do a quick google search, you'll find many articles describing it, so the topic should be exhausted already, right? Well, most of what I found were very superficial articles - their authors based on the MNIST dataset, which is ok if you just want to learn the concept. In this post I go beyond that. I found out I learn best on medium to hard examples, so for this experiment I picked a dataset I've already been using - the crack dataset I used in the still unfinished crack detection series.
