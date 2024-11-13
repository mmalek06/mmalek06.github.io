---
layout: post
title: "Multiple bounding box detection, Part 3 - fine tuning the backbone network"
date: 2024-11-02 00:00:00 -0000
categories: Python
tags: ["python", "pytorch", "transfer learning", "image vision"]
---

# Multiple bounding box detection, Part 3 - fine tuning the backbone network

In the previous posts I focused on preparing the training data: 

1. [Here](https://mmalek06.github.io/python/2024/08/04/multiple-bounding-box-detection-part1.html) I data-engineered the full images from the original dataset.
2. [And here](https://mmalek06.github.io/python/2024/11/02/multiple-bounding-box-detection-part2.html) I run the region proposal algorithm to obtain 224x224 regions to be fed to a network that I will train in this post.

<b>Side note:</b> You may wonder why I didn't choose a more modern network as the backbone of my model, like VisionTransformer or newer architectures. The reason is that I tried to stay true to the original R-CNN architecture, with one exception - they used AlexNet, and I'm using a variation of ResNet, but both are CNNs. Additionally, I'm not very familiar with Vision Transformer architectures yet, so to limit the amount of new information I'd need to take in, I decided to go with something I already know.

## Requirements

1. Train ResNet-backed feature extractor model.
2. Experiment with different sampling strategies.

## The code

No elaborate logic here. The `__init__` method constructs image transformation pipelines in accordance with the `is_train` flag - the pipeline should be different for validation, because for that step no augmentation is needed. Another thing to note is the last line of the transformation pipeline declaration - the one with `v2.Normalize` - that's something that ResNet architecture expects. It would work without it too, but the performance could be degraded. As for the `__getitem__` method - it's returning the image name and iou score for debugging. As for the iou score and label - I talked about it in the previous post, but repeating that info here won't do any harm: a sample is considered to represent the positive class (there's a crack) if the iou score is > 0.5. That's in line with the R-CNN paper author's approach for this phase.

```python
class CrackDataset(Dataset):
    backbone_mean = [0.485, 0.456, 0.406]
    backbone_std = [0.229, 0.224, 0.225]
    
    def __init__(self, directory: str, image_files: list[str], is_train: bool):
        self.directory = directory
        self.image_files = image_files

        if is_train:
            self.transform = v2.Compose([
                v2.ColorJitter(brightness=0.5),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(180),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.backbone_mean, std=self.backbone_std)
            ])
        else:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.backbone_mean, std=self.backbone_std)
            ])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, float, int]:
        image_name = self.image_files[idx]
        iou_score, label = CrackDataset.parse_filename(image_name)
        image_path = os.path.join(self.directory, image_name)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, image_name, iou_score, label

    @staticmethod
    def parse_filename(filename: str) -> tuple[float, int]:
        parts = filename.split(".")
        iou_score = float(parts[2].replace("_", "."))
        label = int(parts[3])
        
        return iou_score, label
```

The model itself is quite simple as well. The last layer of ResNet is `(fc): Linear(in_features=2048, out_features=1000, bias=True)`, so there's no need to flatten anything - I could pass the feature extractor output straight to the ReLU layer. 

At this stage, the feature extractor backbone network is intentionally not frozen. The goal here is to fine-tune the already performant network to the specific problem at hand. In the next post, this retrained network will be reused, this time in a frozen state.

```python
class Resnext50BasedClassifier(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int] = (3, 224, 224)):
        super().__init__()

        self.feature_extractor = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self._get_feature_size(input_shape), 1),
            nn.Sigmoid()
        )

    def _get_feature_size(self, shape: tuple[int, int, int]) -> int:
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            features = self.feature_extractor(dummy_input)

            return features.numel()

    def forward(self, x):
        features = self.feature_extractor(x)
        class_scores = self.classifier(features)

        return class_scores
```

The next sample is slightly longer than the previous ones. Since I kept running the notebook multiple times, and since the data volume (proposals generated in the second part of this series) is quite big, it made sense to cache the image paths to speed up the overall process. Also, you'll notice that I limited the number of negative samples. I did it to speed up the training. If I allowed the network to go through the whole dataset it was very slow on my RTX3060 graphics card. Apart from that <b>I'm not aiming to train the best possible bounding box detection model</b>, only to learn this particular architecture and have some fun with it. If this was a real world project obviously I would be much more thorough and strict(then again, I would probably have something better than RTX3060 on board ;) ).

```python
def parse_image_metadata(image_name: str):
    parts = image_name.split('.')
    iou_score = float(parts[2].replace('_', '.'))
    label = int(parts[3])

    return iou_score, label


def filter_and_sort_images(image_names: list[str], validation_set: bool) -> list[str]:
    label_0_images = []
    label_1_images = []

    for image_name in image_names:
        iou_score, label = parse_image_metadata(image_name)
        
        if label == 1:
            label_1_images.append(image_name)
        elif label == 0:
            label_0_images.append((image_name, iou_score))
            
    set_size = 800 if not validation_set else 200
    label_0_images_sorted = [img for img, _ in sorted(label_0_images, key=lambda x: x[1], reverse=True)[:set_size]]
    filtered_images = label_1_images + label_0_images_sorted

    return filtered_images


def get_image_paths_from_file(file_path: str, images_dir: str, validation_set: bool) -> list[str]:
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            print(f"Loading image paths from {file_path}")
            
            image_paths = json.load(f)
    else:
        print(f"{file_path} not found. Reading image paths from directory.")
        
        image_paths = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
        
        with open(file_path, 'w') as f:
            json.dump(image_paths, f)

    filtered_images = filter_and_sort_images(image_paths, validation_set)

    return filtered_images


def get_loaders() -> tuple[DataLoader, DataLoader]:
    train_images_dir = DATASETS_PATH / "train"
    valid_images_dir = DATASETS_PATH / "valid"
    train_paths_file = DATASETS_PATH / "train_image_paths.json"
    valid_paths_file = DATASETS_PATH / "valid_image_paths.json"
    train_images_paths = get_image_paths_from_file(train_paths_file, train_images_dir, validation_set=False)

    print(f"Finished reading train images. Total: {len(train_images_paths)} images.")

    valid_images_paths = get_image_paths_from_file(valid_paths_file, valid_images_dir, validation_set=True)

    print(f"Finished reading valid images. Total: {len(valid_images_paths)} images.")

    train_dataset = CrackDataset(train_images_dir, train_images_paths, is_train=True)
    valid_dataset = CrackDataset(valid_images_dir, valid_images_paths, is_train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_dataloader, valid_dataloader
```

And finally, below you'll find the training loop. Its code will change slightly from notebook to notebook, as I experiment with sampling and weighing, so I'll just put it here without any further comment, as its code is very basic.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Resnext50BasedClassifier()
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
early_stopping = EarlyStopping(patience=3, verbose=True)
train_loader, valid_loader = get_loaders()

model.to(device)

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

for epoch in range(NUM_EPOCHS):
    model.train()

    running_loss = 0.0
    correct_train = 0
    total_train = 0
    epoch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]", unit="batch")

    for images, names, scores, labels in epoch_progress:
        images, labels = images.to(device), labels.float().to(device)
        
        optimizer.zero_grad()

        outputs = model(images).squeeze()
        predictions = (outputs >= 0.5).float()
        correct_train += (predictions == labels).sum().item()
        total_train += labels.size(0)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        epoch_progress.set_postfix(
            loss=running_loss / total_train,
            accuracy=100.0 * correct_train / total_train
        )

    train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = 100.0 * correct_train / total_train
    
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
    model.eval()
    
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        valid_progress = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]", unit="batch")
        
        for valid_images, valid_names, valid_scores, valid_labels in valid_progress:
            valid_images, valid_labels = valid_images.to(device), valid_labels.float().to(device)
            outputs = model(valid_images).squeeze()
            predictions = (outputs >= 0.5).float()
            correct_val += (predictions == valid_labels).sum().item()
            total_val += valid_labels.size(0)

            loss = criterion(outputs, valid_labels)
            val_loss += loss.item() * valid_images.size(0)

    val_loss /= len(valid_loader.dataset)
    val_accuracy = 100.0 * correct_val / total_val

    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    early_stopping(val_loss, model, SAVE_PATH)

    if early_stopping.early_stop:
        print("Early stopping")
        break
```

When the model is done training, it's time for testing. First let's generate the classification report to see how well it does with the two categories it was supposed to distinguish.


## to use with next notebooks

The most important part is found in the last lines of the `get_loaders` function. `class_weights` is a dictionary where each class label maps to its calculated weight: `num_samples / count`. This weighting inversely scales with the frequency of each class, meaning classes with fewer examples get higher weights. `sample_weights` is a list that maps the weight for each image's label. Images from underrepresented classes get higher weights, making them more likely to be chosen. `WeightedRandomSampler` uses these weights to sample images with replacement, ensuring that each mini-batch contains a balanced representation across classes. This is particularly useful for imbalanced datasets because it forces the model to train on all classes equally(-ish), improving performance on minority classes. Or is it...? The thing is that I've set the `replacement` argument to `True`. What that means is that each sample can potentially be selected multiple times, and this in turn may lead to overfitting. Whether or not that will happen will be visible after the training is completed. There are also two alternative approaches I used for sampling and weighing and I'll describe those later in this post.
