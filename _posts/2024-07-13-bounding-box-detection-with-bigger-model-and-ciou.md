# Bounding box detection for HAM10000 dataset with bigger model and CIoU loss function

This time I'll go straight into the code, because the most of this notebook has already been covered in [previous post](https://mmalek06.github.io/2024/07/13/bounding-box-detection.html).

## The code

### Bigger variation

If you read the previous post, you'll notice that the only difference between the architecture used there and this one is that the `conv2`, and `conv3` layers `in_channels` param gets a higher value.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundingBoxModel(nn.Module):
    def __init__(self):
        super(BoundingBoxModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
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

### SmoothL1Loss (Huber loss) formula and explanation

$$\begin{aligned}
\text{SmoothL1Loss}(x, y) =
\begin{cases} 
(x - y)^2 & \text{if } |x - y| < 1 \\
|x - y| & \text{otherwise}
\end{cases}
\end{aligned}$$

This function combines two other ones that are widely known MAE and MSE. If the error is small, as given by the first branch of the formula, MSE is used, if it's big, MAE is used. The main selling point of this function, at least to me, is that because of the use of MAE, Huber Loss function prevents very large prediction errors to disproportionatelly affect the overall loss - in certain extreme cases it could slow the model convergence greatly.
Also, the number `1` you see at the end of the first branch is a threshold parameter that you can pick to have different value. The default is `1` though.

### CIoU loss

The motivation for creation this loss function is that is more specific to the problem at hand. First the code:

```python
import torch
import torch.nn as nn


def bbox_iou(
    bboxes1: torch.Tensor, 
    bboxes2: torch.Tensor, 
    eps: float
) -> torch.Tensor:
    """
    :param bboxes1: Expected to be already in transposed format -> (4, N)
    :param bboxes2: Expected to be already in transposed format -> (4, N)
    :param eps: Param for preventing zero-division errors
    :return:
    """

    b1_x1, b1_y1, b1_x2, b1_y2 = bboxes1
    b2_x1, b2_y1, b2_x2, b2_y2 = bboxes2
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area + eps
    iou = inter_area / union_area

    return iou


def bbox_ciou(
    bboxes1: torch.Tensor, 
    bboxes2: torch.Tensor, 
    eps: float = 1e-7
) -> torch.Tensor:
    """
    bboxes1, bboxes2 should be tensors of shape (N, 4), with each box in (x1, y1, x2, y2) format
    """

    # transpose both to get xs and ys as vectors (below)
    bboxes1 = bboxes1.t()
    bboxes2 = bboxes2.t()
    b1_x1, b1_y1, b1_x2, b1_y2 = bboxes1
    b2_x1, b2_y1, b2_x2, b2_y2 = bboxes2
    iou = bbox_iou(bboxes1, bboxes2, eps)
    b1_center_x = (b1_x1 + b1_x2) / 2
    b1_center_y = (b1_y1 + b1_y2) / 2
    b2_center_x = (b2_x1 + b2_x2) / 2
    b2_center_y = (b2_y1 + b2_y2) / 2
    center_distance = (b1_center_x - b2_center_x) ** 2 + (b1_center_y - b2_center_y) ** 2
    enclose_x1 = torch.min(b1_x1, b2_x1)
    enclose_y1 = torch.min(b1_y1, b2_y1)
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)
    enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    b1_w, b1_h = b1_x2 - b1_x1, b1_y2 - b1_y1
    b2_w, b2_h = b2_x2 - b2_x1, b2_y2 - b2_y1
    aspect_ratio = 4 / (torch.pi ** 2) * torch.pow(torch.atan(b1_w / (b1_h + eps)) - torch.atan(b2_w / (b2_h + eps)), 2)
    v = aspect_ratio / (1 - iou + aspect_ratio + eps)
    ciou = iou - (center_distance / (enclose_diagonal + eps) + v)

    return ciou


class CIoULoss(nn.Module):
    def __init__(self):
        super(CIoULoss, self).__init__()

    def forward(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        ciou = bbox_ciou(preds, targets)
        loss = 1 - ciou

        return loss.mean()

```

Although I try to not include too much math in my ML/AI efforts, sometimes it's unavoidable. And certainly if I see the above blob of code in 3 months, I won't remember what does it do, and mathematical formulation of the above can be helpful:

$$\begin{aligned}
$$
\text{IoU} = \frac{\text{Intersection Area}}{\text{Union Area}} = \frac{|B_p \cap B_g|}{|B_p \cup B_g|}
$$
$$
\text{CIoU} = \text{IoU} - \left( \frac{\rho^2(\mathbf{b}, \mathbf{b}^g)}{c^2} + \alpha v \right)
$$
$$
\rho^2(\mathbf{b}, \mathbf{b}^g) = (b_x - b_x^g)^2 + (b_y - b_y^g)^2
c^2 = (c_x - c_x^g)^2 + (c_y - c_y^g)^2
$$
$$
v = \frac{4}{\pi^2} \left( \arctan \frac{w^g}{h^g} - \arctan \frac{w}{h} \right)^2
$$
$$
\alpha = \frac{v}{(1 - \text{IoU}) + v}
$$
\end{aligned}$$
