---
layout: post
title: "Denoising autoencoders, Part 2 - improving the denoising quality with a custom loss function"
date: 2025-08-07 00:00:00 -0000
categories:
    - python
    - computer-vision
tags: ["python", "pytorch", "transfer learning", "computer vision", "transformer", "gradcam"]
---

# Denoising autoencoders, Part 2 - improving the denoising quality with a custom loss function

In the previous post, I said I probably wouldn't come back to this topic soon, but I couldn't stop thinking that this series wouldn't be complete until I investigated how my final architecture "thinks" and, depending on what I saw, whether there might still be some low-hanging fruit to harvest in this project. So I arrived at the idea of using the `pytorch-grad-cam` library to see what activates my network, which in turn guided my next steps.

## So... WHAT IS THAT?!

Glad you asked! From what I read on the internet, explainability has been one of the hottest topics in AI for several years now. We - the neural networks architects - love how our code turns impossible problems into solutions, but explaining why a certain architecture made certain decision has been a problem since the dawn of time. Obviously looking at how single neurons work is not really productive, as the "thinking" of neural network happens when they cooperate. That means they need to be analyzed on a higher level. That's why some explainability frameworks have been introduced. One of them is the one I mentioned in the introduction - `pytorch-grad-cam`. The idea is simple - give it a trained model instance, layers that you want to inspect and sample images. It will give back a heat map representing the image fragments that activated the inspected layers the most.

There's one caveat - the library wants its user to give it some targets. That's easy when the task at hand is about classification, object detection, or semantic segmentation, because the targets are available. In the case of a denoising task, there's no such natural target because the network's goal is not to find something on an image but rather make the image better-looking. That's why an artificial target needs to be passed in (gradients have to be calculated with respect to something). The result of that is that there's some manual work to be done around using various "cameras" exposed by this lib.

## The code - inspecting the activations

As the winner architecture from [the last post](https://mmalek06.github.io/python/computer-vision/2025/08/07/denoising-autoencoders-part2.html) consisted of two subnetworks, I was interested in learning what does the ViT network thinks is important and then comparing that with the autoencoder module. So here comes the first of those manual steps - wrapping ViT in another module:

```python
class ViTWrapper(nn.Module):
    def __init__(self, vit_model):
        super().__init__()

        self.vit = vit_model

    def forward(self, pixel_values):
        # outputs = self.vit(pixel_values=pixel_values)
        # patch_tokens = outputs.last_hidden_state[:, 1:, :]
        #
        # return torch.norm(patch_tokens, dim=2).mean(dim=1, keepdim=True)
        outputs = self.vit(pixel_values=pixel_values)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]

        return patch_tokens.mean(dim=[1, 2], keepdim=True)
```

Why do I even do it? Well, that's all because of that missing natural target. If there's no object on the image that would be of interest to the training process and instead it's the image as a whole that is the object of interest, I had to decide what general trait of an image I want to focus on to later show via the camera. In that code snippet you can see two alternatives that would result in very different visual outcomes. The commented-out one is the L2 norm. Using it would make strong features stand out. Strong features could be the cracks that are visible on my images. That was my expectation, but in reality big parts of the image were glowing red, which means it was not underlining anything in particular that would be of interest for the network. I switched to mean calculation and it made a world of difference (mean answers the question: "what contributes to the average activation?").

Before I show the results, the rest of the code:

```python
def run_cam(vit_input: torch.Tensor, targets: list[torch.nn.Module], cam: BaseCAM, name: str, do_filter: bool = False) -> None:
    cam_data = cam(input_tensor=vit_input, targets=targets)
    lower_threshold = 0.4
    upper_threshold = 0.9

    if do_filter:
        cam_data = cam_data[0]
        cam_data_filtered = np.where(
            (cam_data >= lower_threshold) & (cam_data <= upper_threshold),
            cam_data,
            0
        )

        if cam_data_filtered.max() > 0:
            non_zero_mask = cam_data_filtered > 0
            min_val = cam_data_filtered[non_zero_mask].min()
            max_val = cam_data_filtered[non_zero_mask].max()
            cam_data_final = np.zeros_like(cam_data_filtered)
            cam_data_final[non_zero_mask] = (cam_data_filtered[non_zero_mask] - min_val) / (max_val - min_val + 1e-8)
        else:
            cam_data_final = cam_data_filtered
    else:
        cam_data_final = cam_data[0]

    img_np = noisy_img.permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    overlay = show_cam_on_image(img_np.astype(np.float32), cam_data_final, use_rgb=True)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Input Noisy Image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(cam_data_final, cmap='jet', vmin=0, vmax=1)
    plt.title(f"{name} Heatmap range")
    plt.colorbar()
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"{name} Overlay on ViT")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
```

That if statement looks interesting, right? I mentioned this two times already - there's no natural target here. No natural target -> the full image will be considered as the target by the cam. However, even in that case, some parts will stand out anyway, but given the small color difference, it's not easy to see them. That's why I decided to use some color thresholding - I noticed that objects in that range were not picked randomly. 

```python
with EigenCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=vit_reshape_transform) as eigen_cam:
    with KPCA_CAM(model=wrapped_model, target_layers=target_layers, reshape_transform=vit_reshape_transform) as kpca_cam:

        for cnt in range(0, 10):
            vit_input = vit_noisy[cnt].unsqueeze(0).to(device)
            noisy_img = noisy[cnt].cpu()

            vit_input.requires_grad_()

            targets = [ClassifierOutputTarget(0)]

            run_cam(vit_input, targets, eigen_cam, name="Eigen")
            run_cam(vit_input, targets, kpca_cam, name="KPCA")
            print("=" * 100)
```

Why `EigenCAM` and `KPCA_CAM`? Actually, I used all of them and those two were showing something interesting. What's even more interesting is that their results are - to a degree - complementary:

<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/eigen_map_vs_kpca_map.png" />
<br />

I gotta be honest - I cherry-picked this example, because most of the other images were rather random (but that's expected!). But even with this cherry picking, it shows that ViT itself can "see" some bigger image traits, even though it was pretrained on a much different dataset. However, it gets more interesting when the autoencoder activations are visualized:

<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/eigen_map_vs_kpca_map2.png" />
<br />

They are not as bright, and not complementary anymore. Also, it seems that the autoencoder didn't really pay attention to the cracks.

This is the moment that I finished my visual analysis and summed it up with:
"Ok, so ViT attends to the fragments that show cracks, then the CNN module tunes those features down. Is that good or bad? Well, the task was to denoise the images, not denoise the cracks and even though this is kind of weird, the overall process makes the result better. But what would happen if I push the network to focus on the features that ViT was activated by the most?"

## The code - focusing on ViT features

After I scrolled through several images it became clear that ViT attends to cracks, but what makes them so special? Considering them on atomic level, the image regions where the cracks are present stand out because of different coloring. So what if I could incorporate a loss function component just for that? I couldn't find a loss function that would represent that idea (however, I didn't really search for it thoroughly :) ), but I figured that I could write one myself. Another learning oportunity.

```python

```
