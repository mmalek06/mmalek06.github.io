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

The DAE concept was one of the first things I encountered in deep learning that truly captivated me. A quick Google search will reveal countless articles explaining it, so the topic must be thoroughly covered by now, right? Well, most of what I found were superficial guides - authors limited their examples to the MNIST dataset and used small, simple networks composed purely of dense layers. While this works for grasping the basics, this post dives deeper. I’ve found I learn best with medium-to-hard examples, so for this experiment, I chose a dataset I’ve already been exploring: the crack dataset from my [still-unfinished crack detection series](https://mmalek06.github.io/python/computer-vision/2024/11/23/multiple-bounding-box-detection-part3.html).

## What's a denoising autoencoder?

In short: it's a neural network that is capable of reconstructing a clean output from a corrupted input. Putting more words to it: this term is best defined through its use cases. Denoising autoencoders (DAEs), for example, are often used to enhance image quality. Imagine working with a dataset of images plagued by visual artifacts. Traditionally, you might tackle this using a library like `OpenCV`: first, identify the type of noise present in the dataset, then apply a pre-built function like `GaussianBlur` (if something resembling gaussian noise is present on the images). While this approach can reduce some artifacts after extensive trial and error, the results are often limited. Denoising autoencoders were developed to achieve better outcomes.

A DAE architecture is - on a high level - very similar across models. There's the encoder module, the decoder module and a bottleneck placed inbetween them:

<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/denoising-autoencoder.png" /><br />

I intentionally chose a diagram that’s highly generic. As you’ll see, even a seemingly simple architecture can take many forms. However, no matter which variant you select, the core architecture remains consistent with the diagram above: an encoder module that compresses the input into a compact bottleneck layer, followed by a decoder module that reconstructs the original data from this compressed representation. The goal of the encoding process is to force the neural network to learn a latent representation - one that retains only the essential features while discarding noise.

## The code - baseline

I didn't want to spend too much time learning "the old way" of denoising, so I used the basic techniques leveraging the `OpenCV` library:

```python
def calculate_average_mse(original_folder, noised_folder, kernel_size, sigma):
    original_files = [f for f in os.listdir(original_folder) if f.endswith('_raw.jpg')]
    noised_files = [f for f in os.listdir(noised_folder) if f.endswith('_noised.jpg')]
    original_files.sort()
    noised_files.sort()
    mse_list = []

    for original_file, noised_file in zip(original_files, noised_files):
        original_path = os.path.join(str(original_folder), original_file)
        noised_path = os.path.join(str(noised_folder), noised_file)
        original_img = cv2.imread(original_path).astype('float32') / 255.0
        noised_img = cv2.imread(noised_path).astype('float32') / 255.0

        if original_img is None or noised_img is None:
            print(f'Error reading files: {original_file} or {noised_file}')
            continue

        denoised_img = cv2.GaussianBlur(noised_img, kernel_size, sigma)
        mse = mean_squared_error(original_img.flatten(), denoised_img.flatten())
        mse_list.append(mse)

    avg_mse = np.mean(mse_list) if mse_list else float('inf')

    return avg_mse, mse_list


base = Path('/') / 'home' / 'marek' / 'datasets' / 'denoising-nn' / '56x56_tiled'
original_folder = base / 'test'
noised_folder = base / 'test-gaussian'
kernel_sizes = [(3, 3), (5, 5), (7, 7), (9, 9)]
sigmas = [0, 1, 2, 3]
best_mse = float('inf')
best_config = None
mse_lists = None

for kernel_size in kernel_sizes:
    for sigma in sigmas:
        avg_mse, mse_list = calculate_average_mse(original_folder, noised_folder, kernel_size, sigma)

        print(f'Kernel: {kernel_size}, Sigma: {sigma}, Avg MSE: {avg_mse}')

        if avg_mse < best_mse:
            best_mse = avg_mse
            best_config = (kernel_size, sigma)
            mse_lists = mse_list

print(f'Best configuration: Kernel: {best_config[0]}, Sigma: {best_config[1]}, Avg MSE: {best_mse}')
```

The attached code shows the process of searching for the best-fitting `GaussianBlur` parameters. Its output is this:

```text
Kernel: (3, 3), Sigma: 0, Avg MSE: 0.00562406936660409
Kernel: (3, 3), Sigma: 1, Avg MSE: 0.005374730098992586
Kernel: (3, 3), Sigma: 2, Avg MSE: 0.00519159808754921
Kernel: (3, 3), Sigma: 3, Avg MSE: 0.0051943715661764145
Kernel: (5, 5), Sigma: 0, Avg MSE: 0.004354353528469801
Kernel: (5, 5), Sigma: 1, Avg MSE: 0.004459990654140711
Kernel: (5, 5), Sigma: 2, Avg MSE: 0.004235366825014353
Kernel: (5, 5), Sigma: 3, Avg MSE: 0.004387643188238144
Kernel: (7, 7), Sigma: 0, Avg MSE: 0.0040468499064445496
Kernel: (7, 7), Sigma: 1, Avg MSE: 0.004397430922836065
Kernel: (7, 7), Sigma: 2, Avg MSE: 0.0043433653190732
Kernel: (7, 7), Sigma: 3, Avg MSE: 0.004775661509484053
Kernel: (9, 9), Sigma: 0, Avg MSE: 0.004193980246782303
Kernel: (9, 9), Sigma: 1, Avg MSE: 0.004395665600895882
Kernel: (9, 9), Sigma: 2, Avg MSE: 0.004441110882908106
Kernel: (9, 9), Sigma: 3, Avg MSE: 0.005127414129674435
Best configuration: Kernel: (7, 7), Sigma: 0, Avg MSE: 0.0040468499064445496
```

Visual inspection shows that the achieved result is highly unsatisfactory:

<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/baseline-dae.png" /><br />

`GaussianBlur` does exactly what its name says: it blurs the image. Maybe with some experimentation this solution could have been made better, but let's not waste time and move to the first DAE implementation.

## The code - basic DAE

The first model I run was almost 1:1 code I copied from some blog post I found. That's also why I'm not showing its code - it would be just noise. "Almost" - because my dataset consists of 56x56 images and that blog post was for the MNIST dataset (28x28 images). I also experimented with different layer sizes, but the best result that I achieved was this (with average test MSE across the dataset == 0.0088):

<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/basic-dae.png" /><br />

That's... Pretty underwhelming. However, this screenshot points to something - network is capable of learning something that is currently represented by mostly correct coloring without gaussian blur interference. At that moment I remembered that there exist some widely-known and simple techniques to make networks more capable of learning important features and one of them is using skip connections. Let's illustrate that with code:

```python
class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()

        self.input_size = size * size * 3
        self.channel_size = self.input_size // 3
        self.encoder1_out = (self.input_size - int(self.input_size * 0.1)) // 3 * 3

        encoder2_out = self.encoder1_out // 3 - int(self.encoder1_out // 3 * 0.2)
        encoder3_out = self.encoder1_out // 3 - int(self.encoder1_out // 3 * 0.4)

        def create_module():
            return nn.Sequential(
                nn.Linear(self.encoder1_out // 3, encoder2_out),
                nn.ReLU(),
                nn.Linear(encoder2_out, encoder3_out),
                nn.ReLU(),
                nn.Linear(encoder3_out, encoder2_out),
                nn.ReLU(),
                nn.Linear(encoder2_out, self.encoder1_out // 3),
            )

        self.module1 = create_module()
        self.module2 = create_module()
        self.module3 = create_module()
        self.fc1 = nn.Linear(self.input_size, self.encoder1_out)
        self.fc6 = nn.Linear(self.encoder1_out, self.input_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, self.input_size)
        encoded = self.relu(self.fc1(x))
        split_size = self.encoder1_out // 3
        x1, x2, x3 = torch.split(encoded, [split_size, split_size, split_size], dim=1)
        x1 = self.module1(x1)
        x2 = self.module2(x2)
        x3 = self.module3(x3)
        decoded = torch.cat((x1, x2, x3), dim=1)
        skip_connection = self.fc6(decoded) + x
        output = self.sigmoid(skip_connection)

        return output.view(-1, 3, size, size)
```

Skip connections don’t alter the core architecture, as they’re only applied within the `forward` method. The goal here was to combine the flattened raw-input with the final decoder layer. The initial results performed poorly because the latent representation compressed the input too aggressively. A natural question arises: why not increase the bottleneck size? The answer is that doing so risks the network memorizing training patterns, leading to underfitting. But won’t directly adding inputs to the outputs cause overfitting or force the model to memorize noisy data in the final layer? In this case, while it didn’t lead to overfitting, but the model began retaining some noise - though it also improved at extracting finer details (with average test MSE across the dataset == 0.0070):

<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/basic-dae2.png" /><br />

It was a good direction but can this be improved further?

As the final upgrade I decided to use `AdamW` + `Lookahead` optimizers combination hoping to squeeze every last bit of accuracy from the model:

```python
criterion = nn.MSELoss()
base_optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
```

<b>Side note:</b> I didn't show it in any of the previous snippets, but I've been using `MSELoss` and `Adam` optimizer so far.

How did I come up with this idea, since using the standard `Adam` optimizer is such a popular choice? Well, at first I tried sticking with it and improving the model itself, but no changes to the layers or the number of neurons would help. I also tried augmenting the dataset itself with this code:

```python
custom_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=dataset_mean, std=dataset_std)
])
```

I took a few samples of the images and it looked like most of them are rather grey (mostly images of concrete cracks). However when I dug deep I saw that a substantial amount of them is not like that at all. Some were rather blue, others were almost completely black. Probably with some more data engineering I could have split them into groups and calculate their means and standard deviations within those groups, but I decided to change the angle and search for a simpler solution, so I started looking at the available optimizers. My suspicion was that the model underperforms because with `Adam` it easily settles for local minima (because the images within groups are so similar and some groups are more prevalent than others).

When I sampled a few images, most appeared gray (primarily concrete cracks). However, upon closer inspection, a significant portion looked entirely different - some were blue, while others were nearly black. With more data engineering, I could have grouped them and calculated group-specific means and standard deviations. Instead, I shifted angle and started thinking about using another optimizer. I suspected the model underperformed because Adam caused it to settle in local minima, as images within groups were highly similar and some groups dominated the dataset. To expand on that last sentence: Adam is generally less prone to local minima than non-adaptive optimizers (like SGD without momentum) due to its momentum and adaptive learning rates. However, in cases of low gradient diversity (e.g., repetitive/overrepresented patterns), it can converge too quickly to flat regions. That's basically the case with this dataset. Many images are similar, add on top of that the gaussian noise which also introduces some pattern overrepresentation.

Now, a short explanation of the `Adam` optimizer, and then I'll get to the `AdamW` and `Lookahead` combination. 

`Adam` optimizer is an adaptive algorithm. [This article contains a very good and detailed explanation](https://towardsdatascience.com/why-adamw-matters-736223f31b5d), so I won't repeat that here. I'll just outline the main points:

1. It's adaptive, meaning that it changes the amount by which it updates model's weights by looking at the the second moment which is the average of squared gradients (so it represents the variance of gradients). That means if there's not a lot of variation in the gradients it will allow for bigger steps, and if there's a lot of variation, it will be more careful.
2. It pushes the model to learn smaller weight values, because it was observed that model with smaller weights generalize better. 

First let's look at the former; And btw. here's [the source of](https://arxiv.org/abs/1711.05101) the `AdamW` idea and here's [a distilled description of it](. I'll try distilling the distilled version even more, so hold on :) 


