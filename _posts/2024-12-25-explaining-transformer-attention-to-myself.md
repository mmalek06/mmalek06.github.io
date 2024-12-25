---
layout: post
title: "Explaining transformer attention to myself"
date: 2024-12-26 00:00:00 -0000
categories: Linear algebra
tags: ["linear algebra", "math", "python", "numpy", "matplotlib"]
---

# Explaining transformer attention to myself

While still experimenting with the ancient R-CNN architecture, I often find myself exploring different territories - mainly because of the long wait times while models finish training. This time, my thoughts drifted toward the transformer architecture. It might seem funny, given that the whole world seems to understand and use it already - and so did I - until I realized I don't fully grasp the attention mechanism at its core. That's a problem since attention is the essence of transformers.

It's been a long time since I last crammed transformer knowledge into my brain, and much of it has faded over time. So, since explaining concepts is one of the best ways to reinforce and deepen understanding, I'll attempt to do just that in this post.

## The code, the math

In his short book Helgoland, Carlo Rovelli write (not a direct quote) that "it's all in the relations". Elements of matter don't possess intrinsic properties on their own; rather, they exhibit certain properties through interactions with other elements of matter. Reality is relational, and to a big extent, the same applies to language, as the transformer architecture demonstrates.

The first component of the transformer encoder module is the embedding layer combined with positional encoding. Explaining these in detail isn't the goal of this post, so I'll cover them in general terms. Let's assume a sentence has passed through this initial layer and is now represented by six vectors:

```python
v1 = np.array([1, 0, 0])
v2 = np.array([0.95, 0.1, 0])
v3 = np.array([0, 1, 0])
v4 = np.array([0, 0.95, 0.1])
v5 = np.array([-1, 0, 0])
v6 = np.array([0.5, 0.5, 0.5])
semantic_vectors = np.array([v1, v2, v3, v4, v5, v6])
```

If you plot them, you'll see that some of them align with others better than the rest:

```python
semantic_labels = ["v1", "v2", "v3", "v4", "v5", "v6"]
semantic_colors = ["r", "r", "b", "b", "g", "orange"]

%matplotlib notebook
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection="3d")

for i in range(len(semantic_vectors)):
    ax1.quiver(
        0, 
        0, 
        0, 
        semantic_vectors[i][0], 
        semantic_vectors[i][1], 
        semantic_vectors[i][2],
        color=semantic_colors[i], 
        label=semantic_labels[i]
    )
    
ax1.set_xlim([-1.5, 1.5])
ax1.set_ylim([-1.5, 1.5])
ax1.set_zlim([-1.5, 1.5])
ax1.set_title("Semantic Vectors")
ax1.legend()
plt.tight_layout()
plt.show()
```

<b>Side note</b>: I might as well used 2d vector representations, but I needed some fanciness in my life and adding another dimensions doesn't make this probelm harder to understand.

The two blue ones point mostly in the same direction, as do the red ones. The yellow one is somewhere inbetween and the green one points to the direction oposite to the red ones, and is orthogonal to the blue ones:

<div style="height: 400px">
    <img style="width: 360px; float: left" src="https://mmalek06.github.io/images/semantic-vectors.png" />
    <img style="width: 360px; float: right" src="https://mmalek06.github.io/images/semantic-vectors-different-perspective.png" />
</div>

Think: semantic meaning. The vectors in blue and red pairs may be sharing meaning (vector to vector, not pair to pair), like: (v1, v2) = (dog, puppy) and (v3, v4) = (forest, tree). The v6, yellow vector, is supposed to be kind of a "bridge" - something that holds meaning common to the two pairs, but I can't come up with any word to illustrate that :) The v5 vector is supposed to be the oposite to the red ones - I'm also unable to come up with a good example, but I think this setup is so easy to understand that examples are not really necessary. All those relations can be mathematically described with cosine similarity function - it will be used later.

The semantic meaning is obtained, but the transformer architecture also uses positional information and they encode that with function similar to this one:

```python
def positional_encoding(position: int, d_model: int) -> np.ndarray:
    PE = np.zeros((position, d_model))
    
    for pos in range(position):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** ((i + 1) / d_model)))
                
    return PE
```

I also define two helper functions:

```python
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_similarities(vectors: np.ndarray):
    similarity_results = {}

    for i, j in combinations(range(6), 2):
        pair = f"v{i + 1} and v{j + 1}"
        similarity_results[pair] = cosine_similarity(
            vectors[i],
            vectors[j]
        )
    
    print("Cosine Similarity between vector pairs:")
    
    for pair, similarity in similarity_results.items():
        print(f"{pair}: {similarity:.2f}")
```

I initially treated positional encodings as a technical detail that helps transformers perform better, but I never really questioned why the function is defined the way it is. When I started Googling and ChatGPT-ing for answers, every explanation seemed to deepen my confusion. I began to suspect that I may be missing the mathematical intuition needed to analyze this like a pro. So, being as lazy as I am, I opted for an empirical approach. I asked myself two questions:

1. Why can't either sine or cosine function be used? Why use both?
2. Can I come up with another function that would encode positional information like this one does?

The first question can be explained easily if you make the step range function argument equal to 2 and you comment out the if statement. This is the result:

<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/positional-encodings.png" /><br />

There are three pairs, and vectors in each of them overlap so it seems they don't encode positional information very well. However, when this function is used, it gets better:

<div style="height: 400px">
    <img style="width: 360px; float: left" src="https://mmalek06.github.io/images/positional-encodings-sin-cos-1.png" />
    <img style="width: 360px; float: right" src="https://mmalek06.github.io/images/positional-encodings-sin-cos-2.png" />
</div>

They still overlap to some extent. However, note that they are less orthogonal. To check how this looks like in 512 dimensions I just run the `get_similarities` function for positional encodings generated for 512 dimensions. Then I did the same for 3d vectors and these are the cosine similarity results between each pair of vectors:

```plaintext
Cosine Similarity between vector pairs (3d):
v1 and v2: 0.76
v1 and v3: 0.74
v1 and v4: 0.99
v1 and v5: 0.79
v1 and v6: 0.71
v2 and v3: 1.00
v2 and v4: 0.85
v2 and v5: 0.21
v2 and v6: 0.09
v3 and v4: 0.83
v3 and v5: 0.17
v3 and v6: 0.05
v4 and v5: 0.70
v4 and v6: 0.61
v5 and v6: 0.99

Cosine Similarity between vector pairs (512d):
v1 and v2: 0.97
v1 and v3: 0.91
v1 and v4: 0.83
v1 and v5: 0.77
v1 and v6: 0.74
v2 and v3: 0.97
v2 and v4: 0.91
v2 and v5: 0.83
v2 and v6: 0.77
v3 and v4: 0.97
v3 and v5: 0.91
v3 and v6: 0.83
v4 and v5: 0.97
v4 and v6: 0.91
v5 and v6: 0.97
```

In higher dimensions, positional information becomes more clustered and lacks abrupt jumps. Stability is a quality most AI architecture designers strive for, and this method of positional encoding seems to provide consistently stable values.

As for the second question (<i>"Can I come up with another function that would encode positional information like this one does?"</i>) - there are likely many possible functions, but I lack the mathematical tools to identify them. So, my answer is no. :)

One idea I considered was dividing the word's position number (e.g., 1, 2, etc.) by the dimensionality value (e.g., 1/512, 2/512) and then adding that fraction to each dimension. However, this approach would result in a linear function, which might not be desirable here. Additionally, the values would become unevenly distributed at the edges of the range: tiny for the early words and excessively large at the end. In an extreme scenario with a sequence of 512 tokens, the final token would require adding 1 (one) to each dimension.

In contrast, the sine and cosine combination offers a much more constrained and stable representation. That said, this is merely a hypothesis - I'm aware it's more hand-waving than a rigorous explanation.

Let's take another step. The embeddings are ready and the positional encodings are ready as well. The transformer architecture adds them to obtain the vectors that will be passed into the attention mechanism. I could attach a screenshot of the vectors ofter summing them, but that wouldn't provide any insight. It suffices to say that the vectors change, which is kind of obvious :)

Enter: attention mechanism. The algorithm is as follows:

1. Matrices $$Q$$ and $$K^T$$ are multiplied to form the so called attention scores - they represent how important each token is to another token.
2. Their values are scaled down by dividing them by $$\sqrt{d_k}$$ (the dimensionality of each query and key vector).
3. That's done in case the dot products created in the first step are very large - that could make the softmax function return skewed probability values (softmax is similar to the sigmoid function - it also has the S shape - imagine getting only values at the far ends)
4. Finally, the result of the third step is multiplied by the value matrix $$V$$.

An obvious question to ask is: what are these matrices, where do they come from? Well, as almost everything in AI, they are learned. To be precise, the weight matrices are learned, and the Q, K, V ones are created by multiplying weights with the input vectors, and the weight matrices are the ones that were learned.
