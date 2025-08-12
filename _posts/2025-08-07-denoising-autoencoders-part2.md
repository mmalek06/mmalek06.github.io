---
layout: post
title: "Denoising autoencoders, Part 2 - transformer-based denoising"
date: 2025-08-07 00:00:00 -0000
categories:
    - python
    - computer-vision
tags: ["python", "pytorch", "transfer learning", "computer vision", "transformer"]
---

# Denoising autoencoders, Part 2 - transformer-based denoising

In this post, I'd like to turn to what the whole world seems to have embraced already: transformer-based architectures. The Conv-based U-Net did impressively well, but I really wanted to see how a transformer-based network would fare in this competition. I trained my U-Net on 56x56 images, and at this point, I can't really remember why I didn't just use 224x224px versions. It must have had something to do with the limited GPU resources of my old workstation.

Anyway, I retrained that network on larger images and ended up with a pretty decent MSE of 0.0017. I'll use that number as a starting point.

## The code - the dataset class

Before I say anything about this simple dataset class, a bit of theory. Transformers were originally designed for sequences - more concretely: word sequences. However, images are not sequences, although with a patchification process in place, they can be treated as if they were.

What this class does is it loads the clean and noisy versions of an image and turns the noisy one into a tensor of flat patches. <i>Flat</i> meaning that each patch is a 1D vector where the channels have been stacked on top of each other.

```python
class TransformerDenoisingDataset(Dataset):
    def __init__(self, dir: Path, suffix: str = 'gaussian'):
        self.root_dir = dir
        self.noisy_dir = f'{str(dir)}-{suffix}'
        self.clean_dir = dir
        self.image_filenames = [f for f in os.listdir(self.clean_dir) if os.path.isfile(os.path.join(self.clean_dir, f))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, str]:
        img_name = os.path.join(self.clean_dir, self.image_filenames[idx])
        noisy_img_name = os.path.join(self.noisy_dir, self.image_filenames[idx])
        noisy_img_name = noisy_img_name.replace('_raw', '_noised')
        clean_image = np.array(Image.open(img_name)).astype(np.float32) / 255.0
        clean_image = torch.from_numpy(clean_image).permute(2, 0, 1).float()
        noisy_image = np.array(Image.open(noisy_img_name)).astype(np.float32) / 255.0
        noisy_patches = patchify(noisy_image)

        return noisy_patches, clean_image, img_name
```

```python
def patchify(image: np.ndarray) -> torch.Tensor:
    patch_size = 8
    H, W, C = image.shape
    shape = (
        H // patch_size,
        W // patch_size,
        patch_size,
        patch_size,
        C
    )
    # this may be hard to grasp; especially the first two calculations
    # essentially it's about moving to the next row and col in the space that
    # contains the 3d tensors of patches
    # clean_image.strides[0] would move through one row in the image, so
    # since each patch is 8x8, one of those 8's has to be used
    strides = (
        image.strides[0] * patch_size,
        image.strides[1] * patch_size,
        image.strides[0],
        image.strides[1],
        image.strides[2]
    )
    patches = as_strided(image, shape=shape, strides=strides)
    # the below is (784, 8, 8, 3) for patch_size = 8
    # or # (196, 16, 16, 3) for patch_size = 16
    patches = patches.reshape(-1, patch_size, patch_size, C)
    num_patches = (H // patch_size) * (W // patch_size)
    patches = patches.reshape(num_patches, -1)

    return torch.from_numpy(patches).float()
```

With all those calculations in the `patchify` function it's getting a little crazy, especially when compared to how simple the basic arithmetic was with the U-Net. But basically the process is this (step #2 is only mental):

1. Define a patch size - it will be 8 by 8 pixels
2. Define patch dim - the flat vector size - it's 8 * 8 * 3 = 192
3. How many such patches will there be needed for a 224x224 image? It will be 28 (224 / 8) in the x and y directions, so 28 * 28 = 784 in total
4. `image.strides` tells us how much in memory does the pointer have to move to get to another row or column (I guess the underlying representation is just a 1D array). If you use `PIL` to open an 224x224px image, then printing out it's strides would give `(672, 3, 1)` - that 672 value at index 0 comes from multiplying 224 by 3 (the number of channels). So, to get an 8x8x3 patch, we have to move by `image.strides[0] * patch_size` - same for columns 
5. All this could have been done using torch function calls if the image has been converted to a tensor as step #0, but I figured that numpy code is easier to understand for me, so I left it like this

## The code - positional encodings

Before I get to the actual neural network I have to admit that I had literally zero success with a full transformer architecture that included both encoder and decoder. It barely trained and after several attempts, I just gave up on that idea. I also read that [modern neural networks for computer vision](https://en.wikipedia.org/wiki/Vision_transformer) based on transformers are actually encoder-only, so I followed that approach.

Let's first tackle the positional encoding part. In modern day transformer architectures positional encodings are learned, but since I'm just starting to learn the architecture, I wanted to also get to know all of the original ideas. First the formula:

$$\mathrm{PE}_{(pos,\,2i)}   = \sin\!\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)\qquad$$<br /><br />
$$\mathrm{PE}_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$<br /><br />
$$\quad\text{for } i = 0,1,\dots,\left\lfloor \frac{d_{\text{model}}}{2}\right\rfloor\!-\!1.$$<br />

```python
class OptimizedPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        i_values = torch.arange(0, d_model, 2).float()
        denominators = 10000 ** (i_values / d_model)
        pe[:, 0::2] = torch.sin(position / denominators)
        pe[:, 1::2] = torch.cos(position / denominators)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
```

A word of explanation why did I paste in these two classes here. The first one is something that I found on some torch forum. It uses mathematical tricks, like `a^b = exp(b * log(a))` identity and it multiplies a number by some other number's reciprocal instead of just dividing the two. What I assume is that the author did it because using exponentials and logs may be faster than raising to a power and that it provides a higher level of numerical stability. But for this post I don't care about that; I just want to understand so I'll use the simpler form.

<b>Side note:</b> I think I made a mistake [in my previous post about attention mechanisms](https://mmalek06.github.io/linear-algebra/transformers/python/2024/12/25/attention-mechanisms-explained-again.html) while I was trying to explain positional encodings. I mean, my general intuition was correct, but the details were shaky because I mixed the data vectors with the positional vectors on the plots, and then presented cosine similarity results calculated using those mixed vectors instead of just using the ones that the positional encoding calculation had given me. I'm not correcting that post, because this whole blog is supposed to be the log of my increasing (hopefully!) knowledge.

So to supplement and correct what I said back then: each row in the `pe` matrix will represent a unique position of a token in a sequence. Next the `position` column-vector is calculated so that it contains values in the [0, max_len) range. After that, `i_values` is declared to be half the `d_model`, so if `d_model = 16`, `i_values` will be a float vector: `[0, 2, 4, 6, 8, 10, 12, 14]`. Then the `denominators` are calculated for the upcoming sin and cos function calls. So far it's been quite a simple algebra, but now it gets interesting. Seeing this equation I asked myself "Why two trigonometric functions? Why not just one (either)?". Actually a single sin or cos would do, iff it would be ok to encode tokens in a smaller space. Much smaller space. Look at the [sine function plot](https://www.desmos.com/calculator/nqfu5lxaij?lang=pl). It gives unique values only between `0 and PI/2`, then between `PI and PI * 3/2` etc. It works similarly for the cosine function. However, if we take tuples of numbers ([really, take a look at this page :D](https://www.desmos.com/calculator/nqfu5lxaij?lang=pl)) it becomes obvious that the only non-unique tuple-values happen to sit right after each period. So `sin x = cos x` only at `PI * 1/5` and then at `2PI + PI * 1/5`. So mixing these two functions really broadens the possible space for positional encodings. Now, the actual embeddings are not tuples, but high dimensional vectors. That means that since each dimension holds a number related to a different sin/cos frequence, using that simple equation we managed to generate a bunch of unique vectors with values constrained to the `[0, 1]` range. That's pretty ingenious if you ask me. Also, all this stuff has a useful property. The authors wanted to encode the notion of relative distance between the tokens/positions. So: token 1 and 2 are the same distance apart from each other as tokens 10 and 11, but token 1 is very far away from token 11. Let's think, vectors, distances... Hmmmmm... If one treats distance as a similarity metric (which is in fact one of the typical ways of looking at this category of problems in math), then every AI nerd knows there's a tool for this and it's called the dot product. What I mean by that is that we can now check if the distance/similarity assumption holds:

<b>Here I should post a code snippet for the experiment, but since I wrote it in a hurry and I wasn't really trying to make it clean (as in: clean code) AND I don't really have time to fix that, I'll just post my results</b>:

```text
Vector similarity (dot product) between positions:
==================================================
Position 0 vs Position 1: 0.970214
Position 0 vs Position 2: 0.896592
Position 0 vs Position 3: 0.815410
Position 0 vs Position 4: 0.759157
Position 1 vs Position 2: 0.970214
Position 1 vs Position 3: 0.896592
Position 1 vs Position 4: 0.815410
Position 2 vs Position 3: 0.970214
Position 2 vs Position 4: 0.896592
Position 3 vs Position 4: 0.970214
```

And that's it. Dot product between position 0 and 1 equals ~0.97, the same as between positions 1 and 2, and at the same time it's clear that as the position number grows, the dot product is getting lower and lower, until two positions that are far away have dot product saying "close to zero similarity".

One last thing before I move to the neural nets - the original paper used sine and cosine functions but if one wanted to have an even broader range of unique values, no one says a third function couldn't be incorporated. I guess they settled for those two, because the achieved range is satisfying.

## The code - the most basic network

The most basic version consists of two custom elements and one that has been conveniently defined for us by the PyTorch creators.

```python
class OutputHead(torch.nn.Module):
    def __init__(self, d_model: int, patch_dim: int):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, patch_dim)
        )

    def forward(self, x):
        return self.proj(x)
```

The `OutputHead`'s purpose is to map the transformer encoder's output back into the patch dimension - the length of the vector that results from stacking patch rows (across channels). In the case of 8x8 patches and three channels, that's 192. After that, all the patches are converted back into an image (`unpatchify` function handles that).

```python
class TransformerDenoiser(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        encoder_depth: int,
        img_size: int = 224,
        patch_size: int = 8,
        img_channels: int = 3
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.img_channels = img_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * img_channels
        self.patch_embed = torch.nn.Linear(self.patch_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, self.num_patches)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model,
                nhead=16,
                batch_first=True,
                activation=torch.nn.functional.gelu,
            ),
            num_layers=encoder_depth,
            mask_check=False
        )
        self.output_head = OutputHead(d_model, self.patch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, num_patches, d_model]
        Returns:
            Tensor of shape [B, img_channels, img_size, img_size]
        """
        x = self.patch_embed(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.output_head(x)
        x = unpatchify(x, self.img_size, self.img_channels, self.patch_size)

        return x
```

I guess it's best to start from the hardest element - the `TransformerEncoder` class. The first parameter I'm passing in is called `d_model`. I already talked about it in the positional encoding chapter - that's the length of a vector that encodes a token. I could have used patch_dim as `d_model` (mentally calculated previously), however, I empirically checked that this architecture, for this specific problem trains best when `d_model` != `patch_dim`, specifically when `d_model` is smaller which contradicts what I saw on the internet: people using `d_model` values that are bigger than `patch_dim`. It should give model more expressive power, but in this case the bigger I made it, the worse it did, so I tried that unorthodox approach.

As I mentioned near the beginning, I also tried using a full encoder-decoder architecture, but that turned out to be a dead end. No surprise there - decoders are trained a bit differently (with masking), so that part of the architecture didn't really match what I was trying to accomplish. As for the results - I hoped they would blow my mind, but they were worse than those of a U-Net. This model achieved MSE of 0.0022 (the U-Net's result was 0.0017). Is that good or bad? It depends. Each epoch of training a U-Net took around a minute and a half, while training this model took only 22 seconds per epoch. The model also has a smaller memory footprint, so I guess that under certain conditions a programmer might be tempted to favor it over the U-Net.

<b>Side note</b>: the best `d_model` + `encoder_depth` param combination was 64 and 2. Especially when I made the encoder deeper it started loosing it's mind. With great power comes great responsibility to solve huge problems. I guess this one is too tiny to fully utilize a deep encoder.

## The code - an improvement

...then I became stuck for a few hours. I really wanted the transformer-based model to beat the archaic U-Net architecture but no param permutation or improving the output head was capable of making a positive impact on the model's performance. One thing helped though, but I tried it last: I switched to a fully conv-based output head:

```python
class OutputHead(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up1 = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.GELU(),
        )
        self.up2 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )
        self.residual_proj = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.residual_up = torch.nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

    def forward(self, x):
        main = self.up1(x)
        main = self.conv1(main)
        main = self.up2(main)
        main = self.conv2(main)
        res = self.residual_proj(x)
        res = self.residual_up(res)
        out = main + res

        return torch.sigmoid(out)
```

This also required some adjustments in the `forward` method of the main network:

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        B = x.shape[0]
        x = x.view(B, self.h_patches, self.w_patches, -1).permute(0, 3, 1, 2).contiguous()
        x = self.output_head(x)

        return x
```

I tried running it several times and on some attemps it achieved results better than the previous approach (almost unsignificantly: 0.0022 -> 0.0021). I didn't think it was worth it, because finding a good hyperparameter permutation was a time-consuming and boring job, also the forward method looked more complicated. Still, this seemed to be a step in the right direction. I decided to give a shot to a hybrid based approach - one that would be similarly simple to the basic one:

```python
class OutputHead(torch.nn.Module):
    def __init__(self, d_model, patch_dim, img_channels, img_size, patch_size):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 2),
            torch.nn.GELU(),
            torch.nn.Linear(d_model * 2, patch_dim),
        )
        self.img_size = img_size
        self.img_channels = img_channels
        self.patch_size = patch_size
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(img_channels, 64, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, img_channels, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.mlp(x)
        x = unpatchify(x, self.img_size, self.img_channels, self.patch_size)
        x = self.conv(x)

        return x
```

As you can see I threw away all that upsampling and reshaping code and did the simplest thing possible:

1. OutputHead now maps encoder outputs back to the `patch_dim`
2. Then it rebuilds an image from them
3. That image is post processed by a conv-based subnetwork

The results? A huge improvement: 0.0022 -> 0.0018. Now, something interesting. I actually think this version of my model could have easily beaten the U-Net architecture, but because of the resource constraints I started encountering (and being tired and bored because the training was taking so long), I used `GradScaler`. Best. Decision. Ever. Especially in a POC-type project. It cut each epoch down to 2 seconds, which allowed me to iterate quickly and test a large number of hyperparameter combinations (and yes, I know I could have used something automated to do that for me). Why do I think the results could have been better? Well, I trained in the fp16 space, not fp32, so my hypothesis is that with more precision, a better end result would come.

Using it is also quite simple:

```python
scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## The code - the final boss

Even though I think that last architecture could have been the best one if trained with fp32 precision, I decided not to go that route. Instead, I wanted to use transformers with transfer learning to see how that would do. Again, I started from the wrong end. At first, I tried something like a U-Net on top of [ViT](https://huggingface.co/docs/transformers/model_doc/vit). I built a network with skip connections between some encoder layers and the ones in the output head, and it was... Horrible! I'm not even putting that code here, as that would be pointless.

I played with that approach for a bit, but it was another dead end, and being as impatient as I am, I started thinking about calling it a day. However, following my previous trail of thought, I reasoned that since ViT is known to be a good feature extractor and my U-Net did so well, maybe using both is a good idea after all - but I shouldn't use one after the other; instead, I should simply mix their outputs (well, ViT output and U-Net's encoder output to be exact, then use U-Net's decoder on that)?

```python
class HybridViTUNet(nn.Module):
    def __init__(self, freeze_vit=True):
        super(HybridViTUNet, self).__init__()

        self.vit = transformers.ViTModel.from_pretrained(
            'google/vit-base-patch16-224',
            add_pooling_layer=False
        )

        if freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.encoder1 = self._conv_block(3, 64)
        self.encoder2 = self._conv_block(64, 128)
        self.encoder3 = self._conv_block(128, 256)
        self.vit_projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.bottleneck = self._conv_block(256 + 512, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self._conv_block(256 + 256, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self._conv_block(128 + 128, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self._conv_block(64 + 64, 64)
        self.output_layer = nn.Conv2d(64, 3, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, x_vit: torch.Tensor) -> torch.Tensor:
        vit_outputs = self.vit(pixel_values=x_vit, output_hidden_states=True)
        vit_features = vit_outputs.last_hidden_state[:, 1:, :]
        batch_size = vit_features.shape[0]
        vit_features = vit_features.reshape(batch_size, 14, 14, 768)
        vit_features = vit_features.permute(0, 3, 1, 2)
        vit_features_pooled = torch.nn.functional.adaptive_avg_pool2d(vit_features, (28, 28))
        vit_features_pooled = vit_features_pooled.permute(0, 2, 3, 1)
        vit_features_projected = self.vit_projection(vit_features_pooled)
        vit_features_projected = vit_features_projected.permute(0, 3, 1, 2)
        e1 = self.encoder1(x)
        p1 = self.pool(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool(e3)
        b_input = torch.cat([p3, vit_features_projected], dim=1)
        b = self.bottleneck(b_input)
        up3 = self.upconv3(b)
        d3 = self.decoder3(torch.cat([up3, e3], dim=1))
        up2 = self.upconv2(d3)
        d2 = self.decoder2(torch.cat([up2, e2], dim=1))
        up1 = self.upconv1(d2)
        d1 = self.decoder1(torch.cat([up1, e1], dim=1))

        return self.output_layer(d1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
```

And this is what this code does. The `forward` method is provided with two inputs - one for ViT, one for U-Net. Then ViT pretrained network runs to extract features (I'm slicing from `1:` because the first token is a CLS token - something of no interest for this implementation). Then after some dimension juggling a ViT projection is constructed in such a way that is U-Net-compliant:

1. The initial `vit_features` are of `(batch, 196, 768)` shape. An extra dimension needs to be added so after that first reshape operation, `vit_features` are `(batch, 14, 14, 768)`
2. Enter the permutation dance: I permute to use pooling, then back so that linear layers work well, then again for concatenation with conv bottleneck module. TBH, I'm not proud of this and if time allows, I'll try changing it to something simpler
3. The projected features are concatenated with the encoded features and put through the bottleneck module

The result is pretty decent: 0.0017 -> 0.0015, however this time I also trained with half precision, to save some time, so my hypothesis is that it can get even better!

## Summary

This post certainly doesn't cover all the potential solutions. Especially after reaching a good result with a hybrid ViT + U-Net approach, I think I could squeeze something more out of this architecture, but to be honest, most of what I do here is learn 80% of what's required to be fluent in the technique I'm investigating and leave the remaining 20% for when I have time for more experiments. Therefore, for now, I conclude this two-part series and move on to something else!

<b>Side note</b>: I didn't put any example outputs after denoising because the visual differences were much smaller than the ones visible in [the previous post](https://mmalek06.github.io/python/computer-vision/2025/01/30/denoising-autoencoders-part1.html). Instead, I decided to focus on MSE numbers.
