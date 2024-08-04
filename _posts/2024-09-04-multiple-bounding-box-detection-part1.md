---
layout: post
title: "Multiple bounding box detection, Part 1"
date: 2024-07-05 00:00:00 -0000
categories: Python
tags: ["python", "jupyter notebooks", "opencv", "flood fill"]
---

# Multiple bounding box detection, Part 1

This post is the first one in a series that I long planned to write. At the time of writing this I don't know how many parts will there be, or their specific topics, but what I can be sure of is that I will try to implement big chunks of R-CNN, Fast(er) R-CNN, and Mask R-CNN. The initial idea I had was that I will try to train my networks using [this kaggle dataset](https://www.kaggle.com/datasets/lakshaymiddha/crack-segmentation-dataset), and then test them on [something else](https://www.kaggle.com/datasets/dataclusterlabs/cracked-screen-dataset). However, seeing how different the other dataset is, I may substitute this idea with training on a subset of the first dataset, and testing on the left over part of it. Let's see how it goes :)

This part will focus on performing some data engineering steps to prepare the data.

## Requirements

Some of the segmentation masks provided along with the dataset are of bad quality. Take a look at two examples of such images:

<br />
<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/cracktree.jpg" /><br />
<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/rissbilder" /><br />

If you look very closely at the first one, you'll notice it consists of clusters of white pixels, very often not connected to each other. That means that had I used this mask directly, I would detect tens of separate bounding boxes. The other example is of a different nature. It's hard to put into words what's visible on that image - some strange artifacts are present surrounding the actual correct masks - that also will have to be taken care of.

So to sum up, the requirement I came up for the problem of preprocessing the images to obtain bounding boxes is to preprocess the images in such a way that will limit the number of bounding boxes in case small, disconnected clusters of pixels are detected.

## The code

There are very few necessary imports that need to be present.

```python
import os
import cv2
import json
import numpy as np
```

Apart from that some initial variables need to be set up for the later code.

```python
train_dir = os.path.join("data", "train")
valid_dir = os.path.join("data", "valid")
images_dir_train = os.path.join(train_dir, "images")
images_dir_valid = os.path.join(valid_dir, "images")
masks_dir_train = os.path.join(train_dir, "masks")
masks_dir_valid = os.path.join(valid_dir, "masks")
```

For storing the data about the images I again choose the COCO format. The below function returns a template dictionary to be filled in and extended in the code presented next.

```python
def get_coco_tpl() -> dict:
    return {
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "crack",
            "supercategory": "defect"
        }]
    }
```

The below function satisfies the requirement mentioned in the previous section. In each iteration it takes a box from the top of the bboxes list. Then it iterates over the remaining boxes, at each step checking if the current bounding box lies in proximity (regulated using the threshold param) to another one. Then it builds a bounding box that will contain both of them. That new bounding box substitutes the original one and is used in the next iteration step for comparison with another bounding box present on the original list. This way bigger and bigger bounding boxes will be built, and the final count will be sufficiently small.


```python
def merge_adjacent_bboxes(bboxes: list[list[int]], threshold: int = 10) -> list[list[int]]:
    merged_bboxes = []

    while bboxes:
        current_bbox = bboxes.pop(0)
        merged = True

        while merged:
            merged = False
            
            for i, bbox in enumerate(bboxes):
                if (current_bbox[0] - threshold < bbox[0] + bbox[2] and
                    current_bbox[0] + current_bbox[2] + threshold > bbox[0] and
                    current_bbox[1] - threshold < bbox[1] + bbox[3] and
                    current_bbox[1] + current_bbox[3] + threshold > bbox[1]):

                    x_min = min(current_bbox[0], bbox[0])
                    y_min = min(current_bbox[1], bbox[1])
                    x_max = max(current_bbox[0] + current_bbox[2], bbox[0] + bbox[2])
                    y_max = max(current_bbox[1] + current_bbox[3], bbox[1] + bbox[3])
                    current_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    bboxes.pop(i)
                    merged = True
                    break

        merged_bboxes.append(current_bbox)

    return merged_bboxes
```

The above function is then used in the below one:

```python
def find_boxes(images_dir: str, masks_dir: str, coco_format: dict, closing_kernel_size: int = 15) -> None:
    annotation_id = 1
    image_id_mapping = {}
    kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)

    for mask_filename in os.listdir(masks_dir):
        image_id = os.path.splitext(mask_filename)[0]

        if "cracktree" in image_id.lower():
            continue
        
        mask_path = os.path.join(masks_dir, mask_filename)
        image_path = os.path.join(images_dir, f"{image_id}.jpg")

        if not os.path.exists(image_path):
            continue

        image_id_mapping[image_id] = len(image_id_mapping) + 1
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None or np.sum(mask) == 0:
            annotation_entry = {
                "id": annotation_id,
                "image_id": image_id_mapping[image_id],
                "category_id": 1,
                "bbox": [],
                "area": 0,
                "segmentation": [],
                "iscrowd": 0,
                "label": f"no crack {annotation_id}"
            }
            coco_format["annotations"].append(annotation_entry)
            annotation_id += 1

        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        _, binary_image = cv2.threshold(closed_mask, 127, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

        if image_id not in coco_format["images"]:
            image_entry = {
                "id": image_id_mapping[image_id],
                "file_name": os.path.basename(image_path),
                "width": mask.shape[1],
                "height": mask.shape[0],
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            }
            coco_format["images"].append(image_entry)

        bboxes = []
        segmentations = []

        for j in range(1, num_labels):
            if stats[j, cv2.CC_STAT_AREA] > 20:
                bbox = [
                    int(stats[j, cv2.CC_STAT_LEFT]),
                    int(stats[j, cv2.CC_STAT_TOP]),
                    int(stats[j, cv2.CC_STAT_WIDTH]),
                    int(stats[j, cv2.CC_STAT_HEIGHT])
                ]
                mask_region = (labels == j).astype(np.uint8)
                contours, _ = cv2.findContours(mask_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                segmentation = [contour.flatten().tolist() for contour in contours if contour.size >= 6]

                bboxes.append(bbox)
                segmentations.append(segmentation)

        merged_bboxes = merge_adjacent_bboxes(bboxes)

        for bbox, segmentation in zip(merged_bboxes, segmentations):
            annotation_entry = {
                "id": annotation_id,
                "image_id": image_id_mapping[image_id],
                "category_id": 1,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "segmentation": segmentation,
                "iscrowd": 0,
                "label": f"crack {annotation_id}"
            }
            coco_format["annotations"].append(annotation_entry)
            annotation_id += 1

    print(f"Entries count: {len(coco_format['annotations'])}")
```

It's quite a lengthy algorithm, however, it's not hard to understand. Third line also satisfies the "limit bounding box number" requirement. 
