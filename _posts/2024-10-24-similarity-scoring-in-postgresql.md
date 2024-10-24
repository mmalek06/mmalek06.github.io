---
layout: post
title: "Similarity scoring in PostgreSQL"
date: 2024-24-10 00:00:00 -0000
categories: SQL
tags: ["sql", "levenshtein", "similarity score", "jsonb"]
---

# Similarity scoring in PostgreSQL

Recently, I was tasked with writing functionality to calculate a similarity score between data in two JSONB columns located in two different tables. Imagine a structure like this in one of them:

```sql
{ "lastName": "Smith" }
```

...and this kind of structure in the other one:

```sql
{ "surname": "Smith", "anotherProperty": "whatever" }
{ "sn": "Smyth" }
```

The similarity score between the first structure and the first row of the second structure should be 1, and it should be lower for the second row.

## The code

Turns out this is very easily achievable with `pg_trgm` PostgreSQL extension:

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;

SELECT
    t1.id AS table1_id,
    t2.id AS table2_id,
    t1.payload ->> 'lastName' AS table1_lastname,
    COALESCE(t2.payload ->> 'sn', t2.payload ->> 'surname') AS table2_name,
    similarity(
        COALESCE(t1.payload ->> 'lastName', ''),
        COALESCE(t2.payload ->> 'sn', t2.payload ->> 'surname', '')
    ) AS similarity_ratio
FROM
    genie_user_revision t1,
    user_revision t2
WHERE
    t1.id = 1 AND
    t1.payload ? 'lastName' AND
    (t2.payload ? 'longUsername' OR t2.payload ? 'surname')
ORDER BY
    similarity_ratio DESC
LIMIT 10;
```

The first `COALESCE` call selects the properties that the query will use for comparison, then the similarity function runs the comparisons. And that's it! Btw. there's also the `fuzzystrmatch` pg extension that provides the `levenshtein` function for distance calculation.
