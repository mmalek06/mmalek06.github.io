---
layout: post
title: "Multiple bounding box detection, Part 3.1 - finding the right region proposal algorithm"
date: 2024-12-15 00:00:00 -0000
categories: Python
tags: ["python", "pytorch", "transfer learning", "image vision", "selective search", "opencv"]
---

# Multiple bounding box detection, Part 3.1 - finding the right region proposal algorithm

The original R-CNN paper mentions using the selective search algorithm for finding regions of interest to be fed into the network. At the time of R-CNN's inception, one common approach was to use sliding windows, but this came with several challenges. For instance, how much should the sliding window move? Should the movement be the same along both axes? How can we ensure that windows capture relevant visual information, like shape, color, or light changes? All of these issues were addressed by the selective search algorithm, which is available in the `open-cv` library.

I won't go into the technical details of how it works, as it would take us off-topic. However, while reading about it, I became curious whether I could create something similar—not as advanced, but using similar mechanisms. For example, after closely examining the dataset, I noticed that most of the cracks could be described by color; they were either brighter or darker than the rest of the image and were also larger than other similarly colored elements.

I figured that even if a network using my custom region proposal algorithm turns out to be less efficient than one using canonical selective search, I would still have fun and gain knowledge in the process. I’ll describe this custom algorithm in the next blog post.

## Requirements

1. Build a classifier NN utilizing the canonical selective search algorithm.
2. Check if there's a need to use all of the selective search (SS) results, or should they be capped - what impact on the accuracy would it have?

## The code

I didn't jump straight into the R-CNN implementation - although I initially did, and it turned out to be a mistake. R-CNN trains two modules in parallel: a bounding box regressor and a classifier. The losses from both modules are combined, and the loss information is then backpropagated. Using this approach, I wasn't able to make much progress - neither the classifier nor the regressor was being trained properly. So, I decided to create a separate classifier and run it over a set of region proposals to gain more insight into what was happening. 

The first piece of code I created was this dataset:

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

    @staticmethod
    def _group_annotations_by_image(annotations: list[dict]) -> dict[int, list[dict]]:
        grouped_annotations = {}

        for annotation in annotations:
            image_id = annotation["image_id"]

            if image_id not in grouped_annotations:
                grouped_annotations[image_id] = []

            grouped_annotations[image_id].append(annotation)

        return grouped_annotations

    @staticmethod
    def _parse_annotations(annotations: list[dict]) -> np.ndarray:
        """
        Returns:
        - bboxes: A list of bounding boxes in the format [x_min, y_min, x_max, y_max].
        """
        bboxes = []

        for annotation in annotations:
            bbox = annotation.get("bbox", [])

            if not bbox or len(bbox) != 4:
                continue

            x_min, y_min, width, height = map(int, bbox)

            if width <= 0 or height <= 0:
                continue

            x_max, y_max = x_min + width, y_min + height

            bboxes.append([x_min, y_min, x_max, y_max])

        if not bboxes:
            return np.zeros((0, 4), dtype=np.float32)

        return np.array(bboxes, dtype=np.float32)

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        return image
```

Most of the action happens inside the `__getitem__` method. First, an image is loaded from disk, then it's transformed using the transforms composed in the `__init__` method earlier, and finally, the image is passed into the selective search algorithm. The last lines of the method are responsible for creating a tensor of labels - if there's any intersection between a ground truth bounding box and a region proposed by the selective search (SS) algorithm, the label will be set to 1, and 0 otherwise.

This code snippet compares the `ious` tensor to a fairly small threshold of `0.01`, which is something I plan to tune later. The repository will contain the final value for this parameter.

As for the SS algorithm, I wrapped it in a helper function designed to limit the number of proposals. The programmer has no control over how many proposals will be produced by default, but to speed up training, this number can be adjusted after running the OpenCV code (since the SS algorithm can generate thousands of proposals for some images).

```python
proposal_cache = defaultdict(list)
SELECTIVE_SEARCH_BATCH_SIZE = 70


def perform_selective_search(image: np.ndarray, image_path: str, batch_size: int = SELECTIVE_SEARCH_BATCH_SIZE) -> torch.Tensor:
    if image_path in proposal_cache:
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

Although this function doesn't return batched results, the `SELECTIVE_SEARCH_BATCH_SIZE` will also be used later in the training loop.
