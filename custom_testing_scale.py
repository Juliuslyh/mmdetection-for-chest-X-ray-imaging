import mmcv
from mmdet.datasets import build_dataset
import numpy as np
from mmdet.apis import init_detector, inference_detector
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the configuration file for the detection model
config_file = './configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'

# Load the pre-trained weights of the model
checkpoint_file = 'checkpoints/epoch_60_faster_rcnn_hybrid.pth'
device = 'cuda:3'  # or 'cpu'
model = init_detector(config_file, checkpoint_file, device=device)

# load the test data using the built-in dataset function
cfg = mmcv.Config.fromfile(config_file)
test_dataset = build_dataset(cfg.data.test)

# specify the class indices you want to visualize
class_idxs = [0, 1]  # change this to the desired class indices

# loop over the test dataset and visualize the bounding boxes for each image
for i in range(len(test_dataset)):
    # get the i-th sample
    img_info = test_dataset.coco.loadImgs(test_dataset.img_ids[i])[0]
    img_path = test_dataset.img_prefix + '/' + img_info['file_name']
    img = mmcv.imread(img_path)

    # get the ground truth bounding boxes for the desired classes
    ann_ids = test_dataset.coco.getAnnIds(imgIds=img_info['id'], catIds=class_idxs)
    anns = test_dataset.coco.loadAnns(ann_ids)
    gt_bboxes = []
    for ann in anns:
        if ann['category_id'] in class_idxs:
            gt_bboxes.append(ann['bbox'])
    gt_bboxes = np.array(gt_bboxes)

    # run inference on the image to get the predicted bounding boxes
    result = inference_detector(model, img)

    # get the predicted bounding boxes for the desired classes
    pred_bboxes = []
    for class_idx in class_idxs:
        pred_bboxes.append(result[class_idx])
    pred_bboxes = np.concatenate(pred_bboxes)

    # undo the scaling of the predicted bounding boxes to get the coordinates in the original image space
    img_h, img_w, _ = img.shape
    img_scale_h, img_scale_w = cfg.data.test.pipeline[1].img_scale
    scale_factor = np.array([img_w, img_h, img_w, img_h]) / np.array([img_scale_w, img_scale_h, img_scale_w, img_scale_h])
    pred_bboxes[:, :4] = pred_bboxes[:, :4] * scale_factor[:4]

    # Round the coordinates to integers
    pred_bboxes = pred_bboxes.round().astype(np.int32)

    # visualize the ground truth bounding boxes in blue and the predicted bounding boxes in green
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for bbox in gt_bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
    for bbox in pred_bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    plt.show()
