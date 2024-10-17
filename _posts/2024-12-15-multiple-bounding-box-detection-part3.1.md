---
layout: post
title: "Multiple bounding box detection, Part 3.1 - finding the right region proposal algorithm"
date: 2024-12-15 00:00:00 -0000
categories: Python
tags: ["python", "pytorch", "transfer learning", "image vision", "selective search", "opencv"]
---

# Multiple bounding box detection, Part 3.1 - finding the right region proposal algorithm

The original R-CNN paper mentions using the selective search algorithm for finding regions of interest to be fed into the network. At the time of R-CNN's inception, one common approach was to use sliding windows, but this came with several challenges. For instance, how much should the sliding window move? Should the movement be the same along both axes? How can we ensure that windows capture relevant visual information, like shape, color, or light changes? All of these issues were addressed by the selective search algorithm, which is available in the `open-cv` library.

I won't go into the technical details of how it works, as it would take us off-topic. However, while reading about it, I became curious whether I could create something similar - not as advanced, but using similar mechanisms. For example, after closely examining the dataset, I noticed that most of the cracks could be described by color; they were either brighter or darker than the rest of the image and were also larger than other similarly colored elements.

I figured that even if a network using my custom region proposal algorithm turns out to be less efficient than one using canonical selective search, I would still have fun and gain knowledge in the process. I'll describe this custom algorithm in the next blog post.

## The justification

If I knew AI and computer vision techniques better, I would have probably jumped straight into building an R-CNN network. In fact, that's what I initially did, but it turned out to be a frustrating mistake. I didn’t get any good results, so I decided to take a step back and approach things incrementally.

I started by focusing on finding the right combination of IoU threshold and proposal set size. I measured the "goodness of fit" of these parameters by observing how the model's accuracy changed. The idea was that if the classifier performed better for a given combination, then the bounding box regressor would also be able to learn more effectively.

## Requirements

1. Find the best proposal set size / IoU threshold combination - this one is to optimize time VS accuracy ratio - I'm not attempting to reach over 90% accuracy, I just want the end effect to be able to pinpoint most cracks.
2. Build a dataset class to be reused with the actual R-CNN network (possibly with minimal changes).
3. Experiment with frozen/unfrozen `feature_extractor` module to see how the accuracy changes.

## The code

R-CNN trains two modules in parallel: a bounding box regressor and a classifier. The losses from both modules are combined, and the loss information is then backpropagated. Using this approach, I wasn't able to make much progress - neither the classifier nor the regressor was being trained properly and I couldn't tell if that because I made a mistake architecting the network itself, or was it because of the SS algorithm, or something else. So, I decided to create a separate classifier and run it over a set of region proposals to gain more insight into what was happening. 

The first piece of code I created was this dataset (I removed some methods from the snippet, because they are so simple and lengthy, I just didn't want to describe them anyway):

```python
class CrackDatasetForClassificationWithProposals(Dataset):
    def __init__(
            self,
            selective_search_runner: Callable,
            coco_file_path: str,
            images_dir: str
    ):
        with open(coco_file_path, "r") as f:
            self.coco_data = json.load(f)

        self.selective_search_runner = selective_search_runner
        self.images_dir = images_dir
        self.image_data = [
            img
            for img in self.coco_data["images"]
            if CrackDatasetForClassificationWithProposals._load_image(os.path.join(self.images_dir, img["file_name"])) is not None
        ]
        self.annotations = self._group_annotations_by_image(self.coco_data["annotations"])
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        return len(self.image_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_info = self.image_data[idx]
        image_path = os.path.join(self.images_dir, image_info["file_name"])
        image = CrackDatasetForClassificationWithProposals._load_image(image_path)
        image = self.transform(image)
        proposals = self.selective_search_runner(image.permute(1, 2, 0).numpy(), image_path)
        ground_truth_boxes = torch.tensor(
            self._parse_annotations(self.annotations.get(image_info["id"], [])),
            dtype=torch.float32)

        if ground_truth_boxes.numel() == 0:
            return image, proposals, torch.zeros(proposals.size(0))

        ious = torchvision.ops.box_iou(proposals, ground_truth_boxes)
        labels = (ious.max(dim=1)[0] > 0.01).float()

        return image, proposals, labels
```

Most of the action happens inside the `__getitem__` method. First, an image is loaded from disk, then it's transformed using the transforms composed in the `__init__` method earlier, and finally, the image is passed into the selective search algorithm. The last lines of the method are responsible for creating a tensor of labels - if there's any intersection between a ground truth bounding box and a region proposed by the selective search (SS) algorithm, the label will be set to 1, and 0 otherwise.

This code snippet compares the `ious` (and IoU calculation is the same as used [here](https://mmalek06.github.io/computervision/2024/07/13/bounding-box-detection-with-bigger-model-and-ciou.html) - it's the bounding boxes overlap divided by the union so the sum of their areas) tensor to a fairly small threshold of `0.01`, which is something I plan to tune later. The repository will contain the final value for this parameter.

As for the SS algorithm, I wrapped it in a helper function designed to limit the number of proposals. The programmer has no control over how many proposals will be produced by default, but to speed up training, this number can be adjusted after running the OpenCV code (since the SS algorithm can generate thousands of proposals for some images).

```python
proposal_cache = defaultdict(list)
SELECTIVE_SEARCH_BATCH_SIZE = 70


def perform_selective_search(image: np.ndarray, image_path: str, batch_size: int = SELECTIVE_SEARCH_BATCH_SIZE) -> torch.Tensor:
    if image_path in proposal_cache:
        random.shuffle(proposal_cache[image_path])
        
        return torch.tensor(proposal_cache[image_path], dtype=torch.float32)
    else:
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()

        rects = ss.process()
        boxes = []

        for (x, y, w, h) in rects:
            area = w * h

            boxes.append((x, y, x + w, y + h, area))

        boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, area in boxes]
        num_proposals = min(len(boxes), 2 * batch_size)
        top_proposals = boxes[:num_proposals]

        random.shuffle(top_proposals)

        proposal_cache[image_path] = top_proposals

        return torch.tensor(top_proposals, dtype=torch.float32)
```

Although this function doesn't return batched results, the `SELECTIVE_SEARCH_BATCH_SIZE` will also be used later in the training loop. That constant is also something that I tuned while working with this part of the project. Apart from that the `if` statement at the beginning is something very important. Since the Python implementation of the selective search (SS) algorithm runs on the CPU, it can be quite slow, and because it's part of the training loop, the overall training time is significantly impacted during the first epoch. However, since the SS results are cached by image path, subsequent epochs run much faster. 

This is the difference between first two epochs when the weights of the `feature_extractor` are not frozen:

<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/caching_proof1.png" />

And this is the difference between them when the weights are frozen:

<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/caching_proof2.png" /><br />

As for the rest of the function - it's tailored to the dataset and since the cracks are big, the code is sorting the SS proposals by their area, descending, and return `2 * batch_size` of them.

The next piece of code are two functions that are used in the training and validation loops:

```python
def chunkify(x: Iterable | Sized) -> Iterable[Iterable]:
    for i in range(0, len(x), SELECTIVE_SEARCH_BATCH_SIZE):
        yield x[i:i + SELECTIVE_SEARCH_BATCH_SIZE]


def process_and_calculate_loss(
        image: torch.Tensor, 
        proposals: torch.Tensor, 
        labels: Iterable[int]
) -> tuple[torch.Tensor, int, int]:
    cropped_proposals_with_labels = []

    for idx, proposal in enumerate(proposals):
        label = labels[idx]
        x_min, y_min, x_max, y_max = proposal.int()
        cropped_region = image[:, y_min:y_max, x_min:x_max]
        resized_region = resize(cropped_region, [224, 224]) # stretch the proposal so that it matches the NN expected input shape

        cropped_proposals_with_labels.append((label, resized_region))

    batch_loss = 0.0
    correct = 0
    total = 0

    for chunk in chunkify(cropped_proposals_with_labels):
        labels_batch, proposals_batch = zip(*chunk)
        labels_batch = torch.stack(labels_batch).to(device)
        regions = torch.stack(proposals_batch).to(device)
        predictions = model(regions).squeeze(1)
        loss = criterion(predictions, labels_batch)
        batch_loss += loss.item()

        if model.training:
            loss.backward()

        predicted = (predictions > 0.5).float()
        correct += (predicted == labels_batch).sum().item()
        total += labels_batch.numel()

    return batch_loss, correct, total
```

The above is part of a larger training loop. You might notice that I'm backpropagating the loss after each model run. I could have pushed this to the outer training loop and returned aggregated results from the function, but for some reason, that approach caused a huge spike in memory utilization on my PC. I even moved all the data (except `regions`, which go into the model) to the `CPU` device, but that didn't help. The `GPU shared memory` kept getting drained over and over again, so the current approach is the only viable option.

There's no harm in doing it this way—the loss is simply backpropagated fragment by fragment, rather than in a single large chunk.

I'll skip the wrapping training loop and go straight into testing to see which bounding boxes are considered as containing cracks and which are not:

```python

```
