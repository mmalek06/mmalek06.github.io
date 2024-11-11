---
layout: post
title: "Multiple bounding box detection, Part 2 - preparing region proposals for the fine tuning phase"
date: 2024-11-02 00:00:00 -0000
categories: Python
tags: ["python", "pytorch", "transfer learning", "image vision", "selective search", "opencv"]
---

# Multiple bounding box detection, Part 2 - preparing region proposals for the fine tuning phase

The second part of the title may be a bit surprising to some. Didn't we just prepare the data in [this post](https://mmalek06.github.io/python/2024/08/04/multiple-bounding-box-detection-part1.html)? The thing is, that was only phase one. But where's the initial training phase, you may ask? Well, the authors of the [original R-CNN paper](https://arxiv.org/pdf/1311.2524v5) mention pretraining a CNN on an auxiliary dataset (ILSVRC2012 - this is noted on page 3 of the linked PDF). However, [this youtube video](https://youtu.be/5DvljLV4S1E?t=831) says that they only fine-tuned a pretrained AlexNet.

For my work, it doesn’t make much difference. I don't have enough time to train a feature extractor from scratch, so I'll use a good, pretrained CNN and fine-tune its weights. Additionally, I plan to see if I can optimize the chosen CNN for this specific task. In AI lore, the layers closer to the network output detect more general features, like noses, eyes, or doors - large, recognizable elements. For this task, however, this may not be necessary since it's theoretically simple: there's only one class, and the entities are essentially lines and curves.

## Requirements

- Experiment with custom region proposal algorithm for the sake of learning
- Use selective search for finding good region proposals
- Save the proposals on disk for later use

## The code

Fortunately, there's no need to implement selective search (SS) from scratch, as the god-like authors of the OpenCV library have already done it for us. I experimented with implementing parts of what I understood from the selective search description for fun and even achieved some surprisingly good results. However, my attempts didn't come close to the quality of the ready-made algorithm.

<b>Side note</b>: for those interested, my custom algorithm focused mainly on detecting color differences - one of the key aspects of SS. I'll attach it here for completeness.

```python
def calculate_average_brightness(image):
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image

    average_brightness = np.mean(image_gray)

    return average_brightness

def adaptive_thresholding(image, average_brightness, threshold_offsets=[40, 80, 120]):
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image

    thresholded_images = []

    for offset in threshold_offsets:
        if average_brightness + offset < 255:
            _, thresh_image = cv2.threshold(image_gray, average_brightness + offset, 255, cv2.THRESH_BINARY)
            thresholded_images.append(thresh_image)

    for offset in threshold_offsets:
        if average_brightness - offset > 0:
            _, thresh_image = cv2.threshold(image_gray, average_brightness - offset, 255, cv2.THRESH_BINARY_INV)
            thresholded_images.append(thresh_image)

    # Return all threshold variations separately
    return thresholded_images


def apply_morphological_closing(image, kernel_size=(5, 5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return closed_image

def detect_and_filter_components(image, min_area):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    filtered_components = []

    for i in range(1, num_labels):  # Start from 1 to skip the background
        x, y, w, h, area = stats[i]

        if area >= min_area:
            filtered_components.append((x, y, w, h))

    return filtered_components


def find_adaptive_regions(image, min_area=250):
    average_brightness = calculate_average_brightness(image)
    thresholded_images = adaptive_thresholding(image, average_brightness)

    all_regions = []

    for thresh_image in thresholded_images:
        closed_image = apply_morphological_closing(thresh_image)
        regions = detect_and_filter_components(closed_image, min_area)
        all_regions.extend(regions)

    return all_regions
```

- `calculate_average_brightness` is kind of self explanatory, so I'll omit its description.
- `adaptive_thresholding` is more interesting function. The idea here is to expand the input image into a set of black and white images where each would capture brighter (the first if statement) and darker (the second one) regions. If you go through the dataset, you'll see that the cracks stand out - they are either brighter or darker, so this function tries to capture that trait of the dataset.
- `apply_morphological_closing` the body of the function looks similar to what I did in the previous post in this series. It tries to merge regions that are very close to each other - the `adaptive_thresholding` function helps with that by making the image BW.
- `detect_and_filter_components` is essential for isolating and focusing on significant regions within the image while discarding smaller, potentially irrelevant ones. The function uses OpenCV's `cv2.connectedComponentsWithStats` to perform a connected component analysis on the input binary image. This method groups connected pixels into distinct labeled components, effectively identifying each "region" in the image.
- `find_adaptive_regions` orchestrates all the operations and returns region proposals.

<div style="height: 390px">
    <img style="width: 360px; float: left" src="https://mmalek06.github.io/images/custom_ss_1.png" />
    <img style="width: 360px; float: right" src="https://mmalek06.github.io/images/custom_ss_2.png" />
</div>
<div style="height: 390px">
    <img style="width: 360px; float: left" src="https://mmalek06.github.io/images/custom_ss_3.png" />
    <img style="width: 360px; float: right" src="https://mmalek06.github.io/images/custom_ss_4.png" />
</div>
<br />
It's not half bad, isn't it? The thing is that this outputs small regions for certain images and for some it's not even proposing any regions, where there should be at least a few. SS does much better and obviously it's use is much less involved:

```python
def perform_selective_search(image: np.ndarray) -> Iterable[int]:
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()

    rects = ss.process()
    boxes = []
    image_area = image.shape[0] * image.shape[1]

    for (x, y, w, h) in rects:
        area = w * h

        if area == image_area:
            continue

        boxes.append((x, y, x + w, y + h, area))

    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, area in boxes]

    return boxes
```

The R-CNN paper authors mention that in their case they got around 2k proposals per image. However, if you try running this function on the crack dataset I'm using, you'll be lucky if it produces a thousand proposals. It's understandable. Crack images are usually very gray, don't show varying structures like people, cars, or buildings. Below I attach two outputs (I also limited the number of boxes to 100):

<div style="height: 360px">
    <img style="width: 360px; float: left" src="https://mmalek06.github.io/images/premade_ss_1.png" />
    <img style="width: 360px; float: right" src="https://mmalek06.github.io/images/premade_ss_2.png" />
</div>
<br />
The last piece of the puzzle is the proposal builder functionality. You'll notice that I'm capping the number of negative proposals saved to disk at 400 - this is for performance reasons. The proposal generation process already takes hours, and if I saved every 20x20px proposal image, it would quickly consume all my disk space. However, because positive proposals are very rare, I'm not limiting those.

I'm also sorting the proposals by area in descending order to obtain the most relevant ones. This is based on the dataset's characteristics - it doesn't contain many small cracks, so there's no point in considering the smallest proposals as good candidates. Finally, resizing is necessary since I've chosen a feature extractor CNN with an expected input size of 224x224.

```python
class DatasetKind(StrEnum):
    TRAIN = "train"
    VALID = "valid"


class ProposalBuilder:
    def __init__(
            self,
            coco_file_path: str,
            images_dir: str
    ):
        with open(coco_file_path, "r") as f:
            self.coco_data = json.load(f)

        self.original_dataset_dir = images_dir
        self.image_data = [
            img
            for img in self.coco_data["images"]
            if ProposalBuilder._load_image(os.path.join(self.original_dataset_dir, img["file_name"])) is not None
        ]
        self.annotations = self._group_annotations_by_image(self.coco_data["annotations"])

    def build_proposal_set(self, kind: DatasetKind):
        # need to put the proposal set outside the current repo, because some tooling goes crazy if it sees too many files
        # for example DataSpell tries to reindex everything even though I excluded that folder
        proposal_set_path = os.path.join("..", "..", "datasets", "transformed", "crack-detection", "proposals", kind.value)
        os.makedirs(proposal_set_path, exist_ok=True)
    
        transforms = v2.Compose([
            v2.ToPILImage(),
            v2.Resize((224, 224))
        ])
        positive_count = 0
        negative_count = 0
        total_count = -1
    
        for idx, row in enumerate(self.coco_data["images"]):
            image_path = os.path.join(self.original_dataset_dir, row["file_name"])
            image = ProposalBuilder._load_image(image_path)
            proposals = perform_selective_search(image)
            proposal_areas = [(proposal, proposal[2] - proposal[0]) * (proposal[3] - proposal[1]) for proposal in proposals]
            sorted_proposals = sorted(proposal_areas, key=lambda x: x[1], reverse=True)
            sorted_proposals = torch.Tensor([proposal[0] for proposal in sorted_proposals])
            ground_truth_boxes = torch.tensor(
                self._parse_annotations(self.annotations.get(row["id"], [])),
                dtype=torch.float32
            )

            if sorted_proposals.size(0) == 0 or ground_truth_boxes.size(0) == 0:
                continue

            ious = torchvision.ops.box_iou(sorted_proposals, ground_truth_boxes)
            labels = (ious.max(dim=1)[0] > 0.5).float()
            original_filename, file_extension = row["file_name"].split(".")
    
            for counter, (proposal, iou, label) in enumerate(zip(sorted_proposals, ious, labels)):
                if total_count < 3000000:
                    total_count += 1
                    
                    continue
                
                cropped_image = image[int(proposal[1]):int(proposal[3]), int(proposal[0]):int(proposal[2])]
                cropped_image = torch.tensor(cropped_image).permute(2, 0, 1)  # Change (H, W, C) to (C, H, W)
                resized_proposal = transforms(cropped_image)
                iou_score = round(iou.max().item(), 2)
                proposal_filename = f"{original_filename}.{counter}.{str(iou_score).replace('.', '_')}.{int(label.item())}.{file_extension}"
                proposal_filepath = os.path.join(proposal_set_path, proposal_filename)
    
                resized_proposal.save(proposal_filepath)

                if label.item() > 0.5:
                    resized_proposal.save(proposal_filepath)

                    positive_count += 1
                else:
                    if counter <= 400:
                        resized_proposal.save(proposal_filepath)
                        
                        negative_count += 1
    
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} images...")
                print(f"Positive proposals in last 100 images: {positive_count}")
                print(f"Negative proposals in last 100 images: {negative_count}")
    
                positive_count = 0
                negative_count = 0

    def __len__(self) -> int:
        return len(self.image_data)

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

## Summary and next steps

The proposals have been generated. The next step is to run a so called backbone network (so a pretrained one) on them, so that it's tuned for extracting features from the given dataset better.
