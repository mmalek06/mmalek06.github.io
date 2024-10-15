---
layout: post
title: "Multiple bounding box detection, Part 2 - searching for a backbone network"
date: 2024-08-29 00:00:00 -0000
categories: Python
tags: ["python", "pytorch", "transfer learning", "image vision"]
---

# Multiple bounding box detection, Part 2 - searching for a backbone network for R-CNN architecture

One of the key components in the R-CNN architecture is the so-called "backbone network." It sits between the region proposal module and the final module responsible for bounding box regression. Its purpose is to extract features from the region proposals and feed them into the final module. The [original R-CNN paper](https://arxiv.org/pdf/1311.2524v5) mentioned using [the Caffe implementation of CNN](https://caffe.berkeleyvision.org/). However, there are now many pretrained classifier networks that can serve as backbone networks in the R-CNN architecture. 

In this article, I'll explore two approaches:

1. Training a small, custom-made neural network to serve as a baseline. I don't expect it to be a part of the final solution, but out of sheer curiosity I'd like to see how it does in the context of this problem when compared for something better.
2. Using a pretrained `resnext50_32x4d` model from `torchvision.models`.

## Requirements

The single requirement is to see which one of the two networks performs better. The original paper mentioned using a NN trained on a dataset different than the one used for testing the full R-CNN idea implementation, so I'll do the same.

## The code

I didn't want to spend too much time finding the best architecture because the assumption was that, no matter how much time I spent, I still wouldn't be able to outperform any of the pretrained networks. That's why I created this small network that includes two modules:

1. `feature_extractor`: As the name suggests, its sole purpose is to find kernels capable of extracting features useful in crack detection.
2. `classifier`: Although crack detection is technically a regression problem, I needed a way to score how well my `feature_extractor` performs. To do this, I translated the regression problem back into a classification problem. If the classifier detects cracks in the image, it indicates that the previous module did a good job extracting features.

```python
import torch
import torch.nn as nn


class CustomClassifier(nn.Module):
    def __init__(
            self,
            input_shape: tuple[int, int, int] = (3, 224, 224),
            conv_out_shapes: tuple[int, int] = (64, 128),
            linear_layers_features: int = 512
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, conv_out_shapes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(conv_out_shapes[0], conv_out_shapes[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._get_feature_size(input_shape), linear_layers_features),
            nn.ReLU(inplace=True),
            nn.Linear(linear_layers_features, linear_layers_features),
            nn.ReLU(inplace=True),
            nn.Linear(linear_layers_features, 1)
        )

    def _get_feature_size(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            features = self.feature_extractor(dummy_input)

            return features.numel()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)

        return x
```

After defining the architecture, I created the typical training-and-validation loop code, but I won't include it here as it's quite long and self-explanatory. If you're interested, you can see everything [in my repository](https://github.com/mmalek06/crack-detection/blob/main/0.3_custom_classifier.ipynb).

Since this architecture is small and trains quickly, I decided to use four sets of hyperparameters to search for the best combination. Additionally, to save time if I want to run the notebook multiple times, I implemented a checkpoint system. If a checkpoint file is present for a given hyperparameter combination, it will be loaded and used instead of rerunning the training.

```python
model_results = []
param_combinations = [
    {
        "conv_out_shapes": (64, 128),
        "linear_layers_features": 256,
    },
    {
        "conv_out_shapes": (128, 256),
        "linear_layers_features": 512,
    },
    {
        "conv_out_shapes": (64, 128),
        "linear_layers_features": 512,
    },
    {
        "conv_out_shapes": (64, 128),
        "linear_layers_features": 1024,
    },
]

for param_combination in param_combinations:
    model_custom_path = "_".join(map(str, param_combination["conv_out_shapes"]))
    model_custom_path = f"{param_combination['linear_layers_features']}_{model_custom_path}"
    checkpoint_path = os.path.join("checkpoints", f"custom_classifier_{model_custom_path}.pt")
    history_path = os.path.join("checkpoints", f"history_{model_custom_path}.pkl")

    if os.path.exists(checkpoint_path):
        with open(history_path, "rb") as history_file:
            history = pickle.load(history_file)

        model, _, criterion, _, device = get_loop_objects(
            param_combination["conv_out_shapes"], 
            param_combination["linear_layers_features"]
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint)

        _, valid_loader = get_loaders()
        valid_accuracy, valid_loss = validate(model, valid_loader, criterion, {"valid_loss": []})
        
        model_results.append((checkpoint_path, history, valid_accuracy))
        print(f"Checkpoint already exists for {checkpoint_path}. Skipping...")
    else:
        history, validation_acc = run_training_loop(
            param_combination["conv_out_shapes"], 
            param_combination["linear_layers_features"], 
            checkpoint_path,
            history_path
        )
    
        model_results.append((checkpoint_path, history, validation_acc))
        torch.cuda.empty_cache()

max_tuple = max(model_results, key=lambda x: x[2])

print(f"Max model accuracy is: {max_tuple[2]}, checkpoint path: {max_tuple[0]}")
```

Using this code revealed that the best hyperparameter combination includes `1024` units in the linear layers of the classifier and `64` and `128` out_channels for the two `Conv2d` layers. I also like to inspect the network activations to visually confirm that the kernels found are actually guiding the network in the desired direction.

<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/filters_level1_1.png" /><br />
<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/filters_level1_2.png" />

Some activations are good and make the crack stand out more, while others make it slightly less visible. Let's take a look at the activations from the second `Conv2d` layer:

<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/filters_level2_1.png" />

Visual inspection reveals that more of the second layer's filters "made sense" - more of them made the crack stand out, which can be considered evidence that this architecture managed to converge for good reasons. As for the achieved accuracy, it was equal to `96.4%`. For me it was surprisingly good for such a small model and the complexity of the problem at hand. Or maybe I was mistaken, and the problem is not so hard :)

For the pretrained network, I decided to go with the `resnext50_32x4d` model from `torchvision.models`. I chose this model because it was one of the newer options available in the package. The architecture definition changed slightly:

```python
import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models import ResNeXt50_32X4D_Weights


class Resnext50BasedClassifier(nn.Module):
    def __init__(
            self,
            input_shape: tuple[int, int, int] = (3, 224, 224),
            linear_layers_features: int = 512
    ):
        super().__init__()

        self.feature_extractor = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._get_feature_size(input_shape), linear_layers_features),
            nn.ReLU(inplace=True),
            nn.Linear(linear_layers_features, linear_layers_features),
            nn.ReLU(inplace=True),
            nn.Linear(linear_layers_features, 1)
        )

    def _get_feature_size(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            features = self.feature_extractor(dummy_input)

            return features.numel()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)

        return x
```

You may notice that I'm not freezing the `feature_extractor` weights, which is a common practice when using transfer learning. I chose not to do this because I noticed a slight accuracy improvement when allowing the weights to be trained. The model stopped improving after 7 epochs, and the final result was nearly perfect - a validation accuracy of `99.94%` with a validation loss of `0.0001`.


## Summary and next steps

It's clear that the pretrained architecture outperformed the custom one, so that's the one I'll use as the backbone network in the next article. However, it's important to consider that this might not always be the default approach for every problem. In the case of a smaller problem, it might be better to use a small, custom architecture, even if it's slightly less accurate. The reason is that a smaller network could offer faster inference times, which could be a significant advantage if inference speed is a critical factor.
