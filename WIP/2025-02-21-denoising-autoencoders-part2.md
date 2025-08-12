---
layout: post
title: "Denoising autoencoders, Part 2 - synthetic data generation"
date: 2025-01-30 00:00:00 -0000
categories:
    - python
    - computer-vision
tags: ["python", "computer vision", "math"]
---

# Denoising autoencoders, Part 2 - synthetic data generation

One of the projects I'm working on requires me to generate a synthetic dataset for my neural network to learn from. Whether or not it will make it run well on the actual data is something I will find out in the coming days. The thing is, synthetic datasets can be helpful as long as the programmer creating them is able to make them similar enough to real-world samples, and this, by far, is not a trivial task. I did read about some unsupervised learning approaches that, purely hypothetically, could work in my case, but I have yet to try them out.

So, given that my real-world project gave me an idea and inspiration for the next articles in this series, I thought it would be beneficial to describe the first step I had to take separately. Even more so since the code wasn’t easy to write.

As a reminder [here's the previous post](https://mmalek06.github.io/python/computer-vision/2025/01/30/denoising-autoencoders-part1.html), however, this one won't have anything to do with it.

## The code

The data I will run my model on is a set of scanned documents. The quality of some of them is very low, as they contain various scanner artifacts, faded-out letters, dirt, etc. The most prevalent issue is letter fading. Some letters, even though they are fairly recognizable to the human eye, make it difficult for ready-made solutions like `tesseract` or `EasyOCR` to process them accurately.

The code I'm attaching here will attempt to recreate the problem on randomly generated text samples.

```python
def get_random_font(size: int) -> ImageFont:
    fonts = [
        "arial.ttf",
        "DejaVuSans.ttf",
        "LiberationSerif-Regular.ttf",
        "times.ttf",
        "cour.ttf",
        "calibri.ttf",
        "comic.ttf"
    ]
    chosen = random.choice(fonts)
    
    try:
        return ImageFont.truetype(chosen, size)
    except IOError:
        return ImageFont.load_default()

def generate_random_text() -> str:
    words = []
    num_words = random.randint(20, 70)

    for _ in range(num_words):
        word_length = random.randint(3, 10)
        word = "".join(random.choices(string.ascii_lowercase, k=word_length))
        words.append(word)

    return " ".join(words)
```

These two functions are pretty much self-explanatory. The first one randomly chooses a font and the other one generates random strings of characters. The next one is much more interesting:

```python
def generate_clear_text_image(width: int, height: int, bg_color=255, text_color=0) -> Image:
    img = Image.new("L", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    num_lines = random.randint(3, 6)
    lines = [generate_random_text() for _ in range(num_lines)]
    line_details = []
    total_text_height = 0

    for line in lines:
        font_size = random.randint(8, 20)
        font = get_random_font(font_size)
        bbox = draw.textbbox((0, 0), line, font=font)
        text_height = bbox[3] - bbox[1]
        line_spacing = random.randint(1, 5)

        line_details.append((line, font, text_height, line_spacing))

        total_text_height += text_height + line_spacing

    total_text_height -= line_details[-1][3]

    if height - total_text_height > 0:
        current_y = random.randint(0, height - total_text_height)
    else:
        current_y = 0

    for line, font, text_height, line_spacing in line_details:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]

        if width - text_width > 0:
            x = random.randint(0, width - text_width)
        else:
            x = 0

        draw.text((x, current_y), line, fill=text_color, font=font)

        current_y += text_height + line_spacing

        if current_y > height:
            break

    return img
```

This one is the first big step in my synthetic data generation framework. First it uses the first two functions to get random texts and fonts (one font per line). Then it also varies where the text will be drawn by measuring the total text height and offsetting it by a random value on the y axis. Then it draws and returns an image.

Now, big brain time:

```python
def erode_letters(img: Image, iterations=1, cluster_prob=0.25, dilation_iters=2) -> Image:
    eroded = img.copy()

    for _ in range(iterations):
        eroded = eroded.filter(ImageFilter.MaxFilter(3))

    original_np = np.array(img)
    eroded_np = np.array(eroded)
    height, width = original_np.shape[:2]
    non_white_mask = np.any(original_np != 255)
    seed_mask = (np.random.rand(height, width) < cluster_prob) & non_white_mask
    cluster_mask = binary_dilation(seed_mask, iterations=dilation_iters)
    result_np = eroded_np.copy()
    result_np[cluster_mask] = original_np[cluster_mask]

    return Image.fromarray(result_np)
```
