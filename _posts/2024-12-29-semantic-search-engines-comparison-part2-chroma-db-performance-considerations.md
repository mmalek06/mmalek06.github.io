---
layout: post
title: "Semantic search engines comparison, Part 2 - ChromaDB performance considerations"
date: 2025-12-29 00:00:00 -0000
categories: Transformers, Python, Semantic Search, Vector Database
tags: ["python", "transformers", "semantic search", "chromadb", "vector database"]
---

# Semantic search engines comparison, Part 2 - ChromaDB performance considerations

The [last post](https://mmalek06.github.io/transformers,/python,/semantic/search/2024/12/28/semantic-search-engines-comparison-part1.html) got quite long, so I decided that splitting the rest of the series into smaller parts will be both: more readable and easier for me to do. So here it is, a very short post about ChromaDB. I will mention couple of facts from their official documentation for my future reference + make some experiments just for fun.

## Excerpts from the docs

You may remember what I said about ChromaDB - it's a simple tech in its current state. It's been designed for a single task, that is the vector search, and it does that task very well.

In the previous post you might have noticed the `hnsw` term used in the `create_collection` method call. `hnsw` (Hierarchical Navigable Small World Graphs) is an algorithm that ChromaDB (and Elasticsearch, not yet sure what PostgreSQL does, but I bet it's the same) uses to search over embedding vectors. [In their doc](https://docs.trychroma.com/production/administration/performance) they give a [link to a lib](https://github.com/nmslib/hnswlib) they incorporated into their product; that lib implements the `hnsw` algorithm. I could try explaining it, but when I googled it, I got the best search result, something way better than I could write, so [here it is](https://www.youtube.com/watch?v=77QH0Y2PYKg). TLDR; it's `O(log N)`. Pretty cool, huh?

ChromaDB's documentation states that the index data must be kept in RAM. Additionally, it performs processing using the CPU, not the GPU, which came as a surprise to me. I haven't delved deeply into the details, but my guess is that there's nothing inherent in `hnsw` preventing it from running on a GPU. However, GPUs generally have less memory compared to RAM, and since we're dealing with a database, space consumption is a significant factor. I hypothesize that moving data in batches between RAM and GPU memory, along with coordinating the search, would likely reduce the algorithm's performance compared to running it directly on the CPU with RAM.

More than that, ChromaDB is single-threaded! That doesn't have to be a very bad thing though, as Redis example teaches us (it's also single threaded, but performant enough to be used as caching technology). It's mentioned in the docs directly:

> Although aspects of HNSW’s algorithm are multithreaded internally, only one thread can read or write to a given index at a time. For the most part, 
> single-node Chroma is fundamentally single threaded. If a operation is executed while another is still in progress, it blocks until the first one is 
> complete.

## The code



## Summary
