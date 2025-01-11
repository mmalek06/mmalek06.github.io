---
layout: post
title: "Semantic search engines comparison, Part 3 - Milvus"
date: 2025-01-11 00:00:00 -0000
categories: 
    - Transformers
    - Python
    - Semantic Search
    - Vector Database
tags: ["linear algebra", "transformers", "semantic search", "chromadb", "milvus", "elasticsearch", "postgresql", "ponyorm", "fastapi", "vector database", "embedding database", "rag"]
---

# Semantic search engines comparison, Part 3 - Milvus

With this short article, I'd like to finish this short series. It will be devoted to the Milvus embedding database, but I'll also need to revisit the performance testing part, because on the new PC I bought recently (i7-14700F, RTX 4070 ti Super) the performance measurement result changed. 

## Milvus - Excerpts from the docs

I chose this database as a second, purely embedding-oriented solution alongside ChromaDB, expecting similar performance results and tuning options. However, I discovered that Milvus is a much more mature solution. I deployed it using [the provided docker compose file](https://milvus.io/docs/install_standalone-docker-compose.md) in standalone mode. A notable feature of this software is its simple management web UI. In standalone mode, the UI doesn’t show much, but I suspect that in a multi-node environment, it would offer cluster management capabilities. In single-node mode, the most important screen is likely the one listing all collections. It provides basic insights into the databases, collections, and indexes managed by the engine. For this post, that aspect isn’t particularly relevant, as my primary goal was to explore integrating this technology from a programmer's perspective and compare its performance to the other three engines.

What sets this database apart, and makes it potentially better suited for enterprise-level solutions than ChromaDB, is its support for three alternative indexing methods: in-memory, on-disk, and GPU-based. I only used the in-memory method with the general HNSW index, but [the documentation contains a lot of useful information](https://milvus.io/docs/index.md?tab=floating) about the other index types. For example, like Elasticsearch, Milvus’s HNSW index can be further optimized through quantization. If your company handles a large volume of data, the DiskANN algorithm might be preferable, as it allows all data to reside on fast SSD drives. For scenarios with extremely high request pressure, GPU-based indexes could be an ideal solution.

This engine also allows for partitioning the data, so that the searches are even faster (because they operate on subsets of the data contained in partitioned collections) and offers various consistency levels, including strong consistency.

I only explored a small subset of what Milvus offers, but when I have more time, I’ll definitely dive deeper into its features.

## The code

```python
def _define_collection(client: MilvusClient, name: str) -> None:
    if name in client.list_collections():
        client.drop_collection(name)

    id_field = FieldSchema(
        name="id",
        dtype=DataType.VARCHAR,
        is_primary=True,
        max_length=32,
    )
    family_field = FieldSchema(
        name="family",
        dtype=DataType.INT32,
    )
    vector_field = FieldSchema(
        name="vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=768,
    )
    text_field = FieldSchema(
        name="text",
        dtype=DataType.VARCHAR,
        max_length=20_000,
    )
    schema = CollectionSchema(
        fields=[id_field, family_field, vector_field, text_field],
    )
    index_params = IndexParams()

    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        M=16,
        efConstruction=200
    )
    client.create_collection(collection_name=name, schema=schema,
                             description="Semantic search collection with manual string IDs")
    client.create_index(collection_name=name, index_params=index_params)
    client.load_collection(collection_name=name)
```

This snippet shows one way of defining a collection. What's worth noting is that not all fields need to be present in the schema, but following the rule that explicit is better than implicit I decided to define each explicitly. Besides, it's pretty simple and somewhat resembles some python ORM's notation. 

That was the only piece of code I wrote for this engine that was different from the other code fragments; the rest is more or less the same, so let's get to the performance testing:

<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/vectors-testing2.png" /><br />

You may notice two things here:

1. As compared to the previous post, PostgreSQL has been beaten by ChromaDB
2. Milvus is faster than the other three solutions

## Summary

Out of the four solutions, Milvus seems to be the most performant one, but performance is not the only factor that comes into play when deciding on a storage engine. For example you may be building a big product with a small team that knows other technologies better - ES or PGSQL. In that case the slight performance improvement over ES is probably not worth the headache of learning how to administer Milvus cluster (even if it's quite simple and well documented). Still, it's good to know that nowadays, the world offers a plethora of possibilities for storing embeddings and that even though under the hood they use the same algorithms, the engines authors still manage to squeeze every last bit of performance from them.
