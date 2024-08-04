---
layout: post
title: "Bounding box detection for HAM10000 dataset"
date: 2024-07-13 12:00:00 -0000
categories: ComputerVision
tags: ["computer vision", "pytorch", "python", "ai"]
---

# Bounding box detection for HAM10000 dataset

In my opinion, tasks like bounding box detection and image segmentation are among the most satisfying applications of computer vision. A model that can accurately pinpoint an interesting item in an image seems even more magical to me than one that performs classification across millions of classes. In this post, I'll describe the first part of a project aimed at creating a bounding box detection model based on the HAM10000 dataset.

---

## Requirements

1. The standard format for these types of tasks, such as COCO, should be used.
2. Various model variations, including pretrained models, should be trained.

## The code

### Loading the data

I'm not sharing all of the code I used to build the train/valid/test images folder or the annotation files because this post would become infinitely long. Once I'm done with torturing the HAM10000 dataset with bounding boxes, I'll share a link to the repository containing the whole solution.

```python
root = os.path.join("data", "train_images")
ann_file = os.path.join("data", "train_coco_annotations.json")
transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = CocoDetection(
    root=root,
    annFile=ann_file,
    transform=transform
)
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    prefetch_factor=16
)
```

What surprised me in this code is that setting num_workers to a high number, like the number of processor cores in my PC, or even just something greater than two, actually slows down the data loading process. This is peculiar, and I haven't yet discovered the reason for it. For now, I'll stick to using the number 2 as a magical incantation that makes it all work.

### Loss function

For this variation (and the next one as well), I decided to use the basic SmoothL1Loss function. It's always good to start with something simple and see if we can improve upon it by later swapping it with something more complex.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModelVariation().to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Early stopping

I think I could have used Keras, but since I'm learning PyTorch, I didn't want any additional tools to get in the way. Because of that, I decided to create an EarlyStopping class myself. Its functionality is similar to the one found in the Keras library - if the metric value doesn't improve for certain amount of epochs the `early_stop` parameter gets set to `True` and the training loop will stop.

```python
import torch


class EarlyStopping:
    def __init__(
        self, 
        patience: int = 7, 
        verbose: bool = False, 
        delta: float = 0
    ):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(
        self, 
        val_loss: float, 
        model: torch.nn.Module, 
        path: str
    ) -> None:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score

            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

            self.save_checkpoint(val_loss, model, path)

    def save_checkpoint(
        self, 
        val_loss: float, 
        model: torch.nn.Module, 
        path: str
    ) -> None:
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")

        torch.save(model.state_dict(), path)

        self.val_loss_min = val_loss

```

### Basic variation

The code of the basic variation is... Well, very basic:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundingBoxModel(nn.Module):
    def __init__(self):
        super(BoundingBoxModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = self._initialize_fc1()
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._run_first_layers(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def _initialize_fc1(self) -> nn.Linear:
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 150, 200)
            x = self._run_first_layers(dummy_input)
            input_size = x.size(1)

            return nn.Linear(input_size, 128)

    def _run_first_layers(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)

        return x
```

It's the kind of standard code you would see on other blogs or that ChatGPT would create for you, given a simple problem. While building this project, I wasn't sure if this architecture would achieve at least 50% accuracy, especially since I had used the HAM10000 dataset before to train a classifier model with much larger architectures. As it turned out, it performs nicely. For this dataset and as a proof of concept, I'd say it does just fine. I don't think it's all worth describing because of the overall simplicity, except for the _initialize_fc1 method.

To create the fc1 layer, I would have to manually calculate the shape of the last layer before this one. If I wanted to change things inside, I'd have to rerun those calculations repeatedly. It's much better to do a "dry run" of the first layer and let PyTorch calculate that number instead.

### The training loop

This is also something you could easily find on the internet. However, one thing to note is the `extract_bboxes` call. That function, which will be included in the repository I'll share later, is used to simplify the extraction of bounding box information from the COCO file.

```python
early_stopping = EarlyStopping(patience=7, verbose=True)
num_epochs = 25
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training")

    for images, targets in train_loader_tqdm:
        images = images.to(device)
        bboxes = extract_bboxes(targets)
        bboxes = torch.stack(bboxes).to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, bboxes)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        train_loader_tqdm.set_postfix({"Train Loss": running_loss / len(train_loader)})

    epoch_train_loss = running_loss / len(train_loader)

    train_losses.append(epoch_train_loss)
    print(f"Epoch {epoch + 1}, Loss: {epoch_train_loss}")
    model.eval()

    val_loss = 0.0
    valid_loader_tqdm = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation")
    
    with torch.no_grad():
        for images, targets in valid_loader_tqdm:
            images = images.to(device)
            bboxes = extract_bboxes(targets)
            bboxes = torch.stack(bboxes).to(device)
            outputs = model(images)
            loss = criterion(outputs, bboxes)
            val_loss += loss.item()
            
            valid_loader_tqdm.set_postfix({"Val Loss": val_loss / len(valid_loader)})

    epoch_val_loss = val_loss / len(valid_loader)
    
    val_losses.append(epoch_val_loss)
    print(f"Validation Loss: {epoch_val_loss}")

    early_stopping(
        epoch_val_loss,
        model,
        path=os.path.join("checkpoints", f"checkpoint_1_bigger_basic_run_{RUN_NUMBER}.pt")
    )

    if early_stopping.early_stop:
        print("Early stopping")
        break

print("Training complete")
```

## Summary and next steps

In the [previous post](mmalek06.github.io/2024-07-05-running-jupyter-notebook-in-a-loop.html), I described the process of running notebooks multiple times. I employed that approach for running this model. It has been run 20 times, and the best model achieved a loss of 5.89 (and a CIoU loss of 0.23). Those numbers alone don't say much, so to give more context—most lesions visible in the images from the test set were correctly boxed.

In the next post, I'll describe a larger variation of this model, as well as the CIoU loss function used as an alternative to SmoothL1Loss.
