---
layout: post
title: "Semantic search engines comparison, Part 1 - ChromaDB, Elasticsearch and PostgreSQL"
date: 2024-12-29 00:00:00 -0000
categories: Transformers, Python, Semantic Search
tags: ["linear algebra", "transformers", "semantic search", "chromadb", "milvus", "elasticsearch", "postgresql", "ponyorm", "fastapi"]
---

# Semantic search engines comparison, Part 1 - ChromaDB, Elasticsearch and PostgreSQL

When I wrote [the last post](https://mmalek06.github.io/linear/algebra/2024/12/25/attention-mechanisms-explained-again.html) I realized I didn't know much about how modern similarity search engines work. It's time to rectify that. In this two-part series, I'll explore two modern vector databases used for semantic search and two "classic" engines that have been updated in recent years to handle this scenario more effectively. These engines are ChromaDB, Milvus, Elasticsearch, and PostgreSQL. I'll talk about Milvus in the next post, as it needs to run under a Linux environment and emulating that with Docker gives me a lot of headache, and I didn't take my Linux machine with me on this Christmas break. I won't really go into any details describing the other features (like hybrid search), as I only want to focus on a single one - semantic search.

## Requirements

There's just a single requirement that I set for myself: to compare the engines functionally - how easy it is to index and query them, how many configuration is required to set up the indexes etc.

## The code

For this experiment I generated a FastAPI backend and some Jupyter Notebooks. I might have as well just done everything in Jupyter Notebooks, but I also wanted to refresh my FastAPI knowledge, and using this framework doesn't really complicate things, as it's so simple. Below you'll find a screenshot of the project layout:

<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/vectors-layout.png" /><br />

I'll go into details of each route in the section dedicated to that route but first I'd like to create a general overview. Even though Elasticsearch and Postgres (with the pgml extension) offer ways to generate embeddings on the fly, I decided not to use that option. The reason is simple - configuration. With the `pgml` extension [there's a way](https://postgresml.org/docs/open-source/pgml/developers/gpu-support) to configure the database to use GPU, but I didn't want to get buried in the configuration. For Elasticsearch I couldn't really find anything similar. ChromaDB and Milvus on the other hand are GPU-native.

### Notebooks

As shown in the screenshot above, the `src` folder contains several notebooks. The one named `seed` downloads data from kagglehub. The data comes from two sources: a list of news articles and abstracts of scientific articles. The code is fairly straightforward:

```python
import chardet
import kagglehub
import os
import pandas as pd
from pony.orm import db_session, commit

from src.database import db
from src.entities import Article


db.generate_mapping(create_tables=True)

news_articles_path = kagglehub.dataset_download("asad1m9a9h6mood/news-articles")
research_articles_path = path = kagglehub.dataset_download("arashnic/urban-sound")

def convert_file_encoding(file_path: str, original_encoding: str, target_encoding: str) -> None:
    with open(file_path, "r", encoding=original_encoding) as file:
        content = file.read()
    with open(file_path, "w", encoding=target_encoding) as file:
        file.write(content)


def get_dataframe(path: str) -> pd.DataFrame:
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    dfs = []

    for file in csv_files:
        with open(os.path.join(path, file), "rb") as rawdata:
            result = chardet.detect(rawdata.read(100000))
            encoding = result["encoding"]

        convert_file_encoding(os.path.join(path, file), encoding, "utf-8")
        dfs.append(pd.read_csv(os.path.join(path, file), encoding="utf-8"))

    combined = pd.concat(dfs)

    return combined


with db_session:
    for index, row in news_dataframe.iterrows():
        Article(
            title=row["Heading"],
            txt=row["Article"]
        )
        commit()
    for index, row in research_dataframe.iterrows():
        Article(
            title=row["TITLE"],
            txt=row["ABSTRACT"]
        )
        commit()
```

The other one contains code that I created to make it easier to interact with the api. Even though FastAPI generates OpenAPI docs that are usable on their own, and PyCharm generated those test_*.http files, nothing beats a proper interface, more so when it's so easy to create:

```python
import itertools
import time

import pandas as pd
import requests
from IPython.core.display_functions import clear_output
from ipywidgets import widgets, HTML


search_engine_box = widgets.Dropdown(
    options=["chroma", "elasticsearch", "postgres"],
    value="chroma",
    description="Pick the search engine:",
    disabled=False,
)
model_box = widgets.Dropdown(
    options=["all-mpnet-base-v2", "multi-qa-mpnet-base-cos-v1"],
    value="all-mpnet-base-v2",
    description="Pick the model type:",
    disabled=False,
)
text_input = widgets.Text(
    value="",
    placeholder="Enter query",
    description="Enter query:",
    disabled=False,
)
submit = widgets.Button(
    description="Submit",
    button_style="success",
)
time_label = widgets.Label(
    value="Request time: -- s", 
    layout={'margin': '10px 0px 10px 0px'}
)
results_output = widgets.Output()


def render_results(data):
    results_html = ""
    
    for item in data:
        doc_id = item.get("id", "N/A")
        distance = item.get("distance", "N/A")
        similarities = item.get("similarities", [])
        
        results_html += f"""
        <div style='border: 1px solid #ccc; padding: 10px; margin: 10px 0; border-radius: 5px;'>
            <strong>Doc ID:</strong> {doc_id}<br>
            <strong>Distance:</strong> {distance:.4f}<br>
            <hr>
        """
        for sim in similarities:
            text, score, strength = sim
            color = dict(weak="red", mid="yellow", strong="green").get(strength, "black")
            results_html += f"""
            <p style='color: {color}; margin: 5px 0;'>
                {text} (score: {score:.4f}, strength: {strength})
            </p>
            """
        results_html += "</div>"
    
    return HTML(results_html)


def run_request(engine: str, model: str, query: str) -> tuple[str, int]:
    start_time = time.time()
    response = requests.get(
        f"http://127.0.0.1:8000/api/v1/{engine}/search/{model}?query_string={query}"
    )
    end_time = time.time()    
    elapsed_time = end_time - start_time
    response_data = response.json()
    
    return response_data, elapsed_time


def on_button_clicked(b: widgets.Button):
    with results_output:
        clear_output()
        selected_engine = search_engine_box.value
        selected_model = model_box.value
        query_string = text_input.value
        response_data, elapsed_time = run_request(selected_engine, selected_model, query_string)
        time_label.value = f"Request time: {elapsed_time:.4f} s"      
        
        display(render_results(response_data))


submit.on_click(on_button_clicked)
display(search_engine_box, model_box, text_input, submit, time_label, results_output)
```

This is the first part of the notebook. Its goal is to display a set of controls that allow me to select the engine, model variation, and input a query to send to the backend. The most interesting part here is the `render_results` function. If you examine it closely, you'll notice that I'm coloring parts of the text differently based on how they were tagged by the backend. The backend assigns three labels to text segments: strong, mid, and weak. These labels correspond to green, yellow, and red highlights in the interface. I didn't just want to retrieve the documents most similar to the query string - I also wanted to see which parts of the text were the most relevant.

This notebook also contains a part dedicated to measuring the performance of each engine+model combination:

```python
engines = search_engine_box.options
models = model_box.options
queries = [
    "drunk football hooligans",
    "manifold hypothesis",
    "fuel prices growing",
    "climate change",
    "south america football news"
]
results = []

for engine, model, query in itertools.product(engines, models, queries):
    for i in range(5):
        try:
            _, elapsed_time = run_request(engine, model, query)
            
            results.append({
                "engine": engine,
                "model": model,
                "query": query,
                "elapsed_time": elapsed_time
            })
        except Exception as e:
            print(f"Request failed for ({engine}, {model}, {query}): {e}")

df = pd.DataFrame(results)
summary_df = df.groupby(["engine", "model"]).agg(
    mean_time=("elapsed_time", "mean"),
    std_time=("elapsed_time", "std")
).reset_index()

summary_df
```

Checking for the most performant combination was not really the goal here but I wrote the code for it anyway, out of curiosity. I'll present the results at the end of this post.

Now let's dive into the shared components of this project starting with entities:

```python
from pony.orm import Required, Optional, Set

from .database import db


class Article(db.Entity):
    title = Required(str, max_len=1024)
    txt = Required(str)
    all_mpnet_embeddings = Set("ArticleEmbeddingAllMpnetBaseV2")
    multi_qa_mpnet_embeddings = Set("ArticleEmbeddingMultiQaMpnetBaseCosV1")


class ArticleEmbeddingAllMpnetBaseV2(db.Entity):
    article = Required(Article)
    sentences = Required(str, max_len=8192)
    embedding = Optional(str, sql_type="vector(768)")


class ArticleEmbeddingMultiQaMpnetBaseCosV1(db.Entity):
    article = Required(Article)
    sentences = Required(str, max_len=8192)
    embedding = Optional(str, sql_type="vector(768)")
```

I really like `Ponyorm`. I haven't used it in a commercial project yet because usually the team would choose the more popular option, which is `SQLAlchemy` but since this one is a non-commercial one, I can pick whatever I want and I decided that `pony` will make my life easier. And so, above you can see a standard `ponyorm` entity definitions with one twist - it doesn't natively support the vector type (but that's not an issue, vectors will only be used on the database level, not in python). Out of these three entities "shared" is only the first one - it was used in the `seed` notebook to populate the set of articles. They will be selected and processed by each of the routes I describe next.

Now let's talk about the `text_manipulation.py` file starting with the contained `chunk_text` function and the accompanying `_ensure_punkt_downloaded` one:

```python
from functools import singledispatch
from typing import Iterable

import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import MPNetTokenizerFast


def chunk_text(text: str, tokenizer: MPNetTokenizerFast) -> Iterable[str]:
    """
    Splits text into chunks that respect the model's token limit,
    grouping sentences together when possible.
    """
    _ensure_punkt_downloaded()

    sentences = sent_tokenize(text)
    current_chunk = []
    current_tokens = 0
    chunks = []

    for sentence in sentences:
        """
        Side note, the below would generate:

        Token indices sequence length is longer than the specified maximum sequence length for this model (1127 > 384).
        Running this sequence through the model will result in indexing errors

        If it was called on full text. 
        """
        sentence_tokens = tokenizer.tokenize(sentence)
        sentence_token_count = len(sentence_tokens)

        if current_tokens + sentence_token_count > tokenizer.model_max_length:
            if current_chunk:
                chunk_text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(" ".join(current_chunk)))

                chunks.append(chunk_text)

            current_chunk = [sentence]
            current_tokens = sentence_token_count
        else:
            current_chunk.append(sentence)

            current_tokens += sentence_token_count

    if current_chunk:
        chunk_text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(" ".join(current_chunk)))

        chunks.append(chunk_text)

    return chunks


def _ensure_punkt_downloaded():
    """
    NLTK - punkt is capable of splitting text into sentences based on punctuation, capitalization, and
    linguistic patterns which will come in handy to properly split the texts.
    """
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("Downloading 'punkt' tokenizer...")
        nltk.download("punkt")
        nltk.download("punkt_tab")
        print("'punkt' has been successfully downloaded.")
```

Transformers (or any NLP models, really) can only process chunks of text. Remember the semantic vectors from [the last post](https://mmalek06.github.io/linear/algebra/2024/12/25/attention-mechanisms-explained-again.html)? In reality, these are vectors that encapsulate meaning within those chunks. The sentence transformer models I selected use a context window of 384 tokens. Very long chunks of text would inevitably exceed this token limit, which is why splitting is necessary.

The first step in this code is calling the `_ensure_punkt_downloaded` function to download the necessary nltk toolkit components for proper text splitting. However, it's not perfect. Sometimes it splits every sentence; other times, it leaves clusters of sentences intact. I suspect this inconsistency is due to the data downloaded by the seed notebook. In particular, the article contents are not very clean, and even I had difficulty identifying clear splitting points while reading through them. Additionally, the presence of HTML-like artifacts in the text doesn't help.

Once the text is initially tokenized (in the nltk sense, not in the transformer sense - these are two very different tokenization processes), a loop runs to perform transformer-level tokenization on the chunks. In the end, the tokens are glued together to form sentences.

Why do it this way? Well, it's the model's responsibility to tokenize and encode the sentences. You can see this in the `sentence_transformers` source code (file: SentenceTransformer.py, lines 576 or 591, as of December 28, 2024). However, for the model to tokenize and encode effectively for semantic search, I first need to provide it with chunks of text it can properly process. That's why this preprocessing step is essential.

The last function in this file is `tag`:

```python
@singledispatch
def tag(data):
    raise NotImplementedError


@tag.register(str)
def _(
        query: str,
        contents: list[str],
        model: SentenceTransformer
) -> Iterable[tuple[str, float, str]]:
    encoded_query = model.encode(query)
    encoded_sentences = model.encode(contents)
    similarities = model.similarity(encoded_query, encoded_sentences)
    labeled_results = []

    for chunk, similarity in zip(contents, similarities.tolist()[0]):
        label = tag(similarity)

        labeled_results.append((chunk, similarity, label))

    return labeled_results


@tag.register(float)
def _(similarity: float) -> str:
    if similarity > .6:
        return "strong"
    elif similarity >= .3:
        return "mid"
    else:
        return "weak"
```

<b>Side note:</b> I'm implementing a poor man's version function overloading - `singledispatch` decorator allows function behavior to change based on the type of the first argument.

As you might have already guessed, this is the piece of code responsible for tagging the sentences after the search is performed. There's one important but not immediately visible detail here. There are a few methods of calculating vector similarity: l2, euclidean distance and cosine similarity. The sentence transformer models I picked use the last one as the default, so when I run `model.similarity`, a cosine similarity is being calculated. It's very important to use the same metric on the vector storage level, otherwise the numbers returned from the database and the ones obtained in python will differ.

As for the `shared_dependencies.py` file, it contains functions that are used by every route:

```python
from pony.orm.core import DBSessionContextManager, db_session
from sentence_transformers import SentenceTransformer


def get_explicit_default_transformer() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_all_mpnet_base_v2_transformer() -> SentenceTransformer:
    return SentenceTransformer("all-mpnet-base-v2")


def get_multi_qa_mpnet_base_cos_v1_transformer() -> SentenceTransformer:
    return SentenceTransformer("multi-qa-mpnet-base-cos-v1")


def get_db_session() -> DBSessionContextManager:
    return db_session
```

### ChromaDB

The pattern you'll notice in each route is that there are two functions for indexing the data, defined like this:

```python
@chroma_router.post("/vectorize/all-mpnet-base-v2")
def vectorize_using_all_mpnet_base_v2_transformer(
        db_session: DBSessionContextManager = Depends(get_db_session),
        client: chromadb.ClientAPI = Depends(get_chroma_client),
        model: SentenceTransformer = Depends(get_all_mpnet_base_v2_transformer)
) -> None:
    _vectorize_with(db_session, client, model, GENERAL_PURPOSE_COL)


@chroma_router.post("/vectorize/multi-qa-mpnet-base-cos-v1")
def vectorize_using_multi_qa_cos_v1_transformer(
        db_session: DBSessionContextManager = Depends(get_db_session),
        client: chromadb.ClientAPI = Depends(get_chroma_client),
        model: SentenceTransformer = Depends(get_multi_qa_mpnet_base_cos_v1_transformer)
) -> None:
    _vectorize_with(db_session, client, model, SEMANTIC_SEARCH_COL)
```

And two other ones that are used for searching:

```python
@chroma_router.get("/search/all-mpnet-base-v2", response_model=Iterable[dict])
def all_mpnet_base_v2_search(
        query_string: str,
        db_session: DBSessionContextManager = Depends(get_db_session),
        client: chromadb.ClientAPI = Depends(get_chroma_client),
        model: SentenceTransformer = Depends(get_all_mpnet_base_v2_transformer)
) -> Iterable[dict]:
    return _search(query_string, db_session, client, model, GENERAL_PURPOSE_COL)


@chroma_router.get("/search/multi-qa-mpnet-base-cos-v1", response_model=Iterable[dict])
def multi_qa_mpnet_base_cos_v1_search(
        query_string: str,
        db_session: DBSessionContextManager = Depends(get_db_session),
        client: chromadb.ClientAPI = Depends(get_chroma_client),
        model: SentenceTransformer = Depends(get_multi_qa_mpnet_base_cos_v1_transformer)
) -> Iterable[dict]:
    return _search(query_string, db_session, client, model, SEMANTIC_SEARCH_COL)
```

### Elasticsearch

### Postgresql
