# mmdetection-for-chest-X-ray-imaging

# Overview

This project focuses on object detection for chest X-ray imaging using the “Chest ImaGenome Dataset,” particularly the “Gold Data Set” of 1000 manually annotated chest radiographs.

## Dataset

Size: Average image size is 2700 x 2800 pixels; average organ size is 650 x 580 pixels.
Annotations: Over 250 relation annotations and 5000 local comparisons.
Organs: Largest (left lung, right lung, cardiac silhouette) and smallest (cavoatrial junction, aortic arch, carina).

## Experimental Setup

Library: MMDetection (based on PyTorch).
Data Format: Converted to COCO format.
Augmentation: Applied rotation and translation to enhance generalization.

## Performance

IoU 0.5: Faster R-CNN (mAP ~ 89%), SSD (mAP ~ 87%), YOLO (mAP ~ 84%).
IoU 0.75: Faster R-CNN (mAP ~ 58%), SSD (mAP ~ 45%), YOLO (mAP ~ 40%).

## Data Augmentation Impact

Minimal impact, with slight improvements seen in Faster R-CNN with translation.

## Organ-Specific Performance

Higher Accuracy: Larger organs (e.g., lungs, cardiac silhouette).
Lower Accuracy: Smaller organs (e.g., carina, cavoatrial junction), especially at higher IoU thresholds.

## Bounding Box Analysis

Large Organs: Generally accurate predictions.
Small Organs: Often oversized bounding boxes, leading to low IoU values. Adjusting anchor box size showed no significant improvement.


## Conclusion

Faster R-CNN showed the best performance overall. Future work will focus on improving detection for small organs and optimizing anchor box sizes.



## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
##training
python custom_training.py

##show result(example)
python tools/test.py ./configs/yolo/yolov3_d53_mstrain-608_273e_coco.py ./checkpoints/yolo3.pth --out ./result/result.pkl  --work-dir ./model_eval --gpu-id 3 --eval bbox --options "classwise=True"
```
<img src="https://github.com/Juliuslyh/mmdetection-for-chest-X-ray-imaging/blob/main/analysis/screenshot4.png" width="800" height="400">

```python
##visualisation
python custom_testing_scale.py
```
<img src="https://github.com/Juliuslyh/mmdetection-for-chest-X-ray-imaging/blob/main/analysis/screenshot1.png" width="200" height="200"> <img src="https://github.com/Juliuslyh/mmdetection-for-chest-X-ray-imaging/blob/main/analysis/screenshot2.png" width="200" height="200"> <img src="https://github.com/Juliuslyh/mmdetection-for-chest-X-ray-imaging/blob/main/analysis/screenshot3.png" width="200" height="200">
