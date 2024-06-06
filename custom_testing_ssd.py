import mmcv
from mmdet.datasets import build_dataset
import numpy as np
from mmdet.apis import init_detector, inference_detector
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the configuration file for the detection model
config_file = './configs/ssd/ssd512_coco.py'
#config_file_yolo = './configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'
#config_file_ssd = './configs/ssd/ssd512_coco.py'


# Load the pre-trained weights of the model
checkpoint_file = 'checkpoints/epoch_50_ssd_hybrid.pth'
#checkpoint_file_yolo = 'checkpoints/epoch_40_yolo3_hybrid.pth'
#checkpoint_file_ssd = 'checkpoints/epoch_50_ssd_hybrid.pth'
device = 'cuda:3'  # or 'cpu'
#device_yolo = 'cuda:1'  # or 'cpu'
#device_ssd = 'cuda:2'  # or 'cpu'
model = init_detector(config_file, checkpoint_file, device=device)
#model_yolo = init_detector(config_file_yolo, checkpoint_file_yolo, device=device_yolo)
#model_ssd = init_detector(config_file_ssd, checkpoint_file_ssd, device=device_ssd)

# load the test data using the built-in dataset function
cfg = mmcv.Config.fromfile(config_file)
test_dataset = build_dataset(cfg.data.test)

# specify the class index you want to visualize
class_idxs = [0, 1, 25, 5, 9, 12]  # change this to the desired class index

# loop over the test dataset and visualize the bounding boxes for each image
for i in range(len(test_dataset)-70):
    # get the i-th sample
    img_info = test_dataset.coco.loadImgs(test_dataset.img_ids[i])[0]
    img_path = test_dataset.img_prefix + '/' + img_info['file_name']
    img = mmcv.imread(img_path)

    # get the ground truth bounding boxes for the desired class
    ann_ids = test_dataset.coco.getAnnIds(imgIds=img_info['id'])
    anns = test_dataset.coco.loadAnns(ann_ids)
    gt_bboxes = []
    gt_class_names = []
    for ann in anns:
        if ann['category_id'] in class_idxs:
            gt_bboxes.append(ann['bbox'])
            gt_class_names.append(test_dataset.CLASSES[ann['category_id']])
    gt_bboxes = np.array(gt_bboxes)
    #gt_bboxes = np.array([ann['bbox'] for ann in anns if ann['category_id'] == class_idx])

    # run inference on the image to get the predicted bounding boxes
    result = inference_detector(model, img)
    #result_yolo = inference_detector(model_yolo, img)
    #result_ssd = inference_detector(model_ssd, img)

    # get the predicted bounding boxes for the desired class
    pred_bboxes = []
    for class_idx in class_idxs:
        if len(result[class_idx]) != 0:
            pred_bbox = result[class_idx]
        # Sort the bounding box predictions based on the score in descending order
            sorted_bboxes = pred_bbox[pred_bbox[:, 4].argsort()[::-1]]
        # Take the bounding box with the highest score
            highest_score_bbox = sorted_bboxes[0]

            pred_bboxes.append([highest_score_bbox])
    pred_bboxes = np.concatenate(pred_bboxes)
    #pred_bboxes = result[class_idx]

    # pred_bboxes_yolo = []
    # for class_idx in class_idxs:
    #     pred_bboxes = result_yolo[class_idx]
    #     # Sort the bounding box predictions based on the score in descending order
    #     sorted_bboxes = pred_bboxes[pred_bboxes[:, 4].argsort()[::-1]]
    #     # Take the bounding box with the highest score
    #     highest_score_bbox = sorted_bboxes[0]
    #
    #     pred_bboxes.append(highest_score_bbox)
    # pred_bboxes_yolo = np.concatenate(pred_bboxes_yolo)

    # pred_bboxes_ssd = []
    # for class_idx in class_idxs:
    #     pred_bboxes = result_ssd[class_idx]
    #     # Sort the bounding box predictions based on the score in descending order
    #     sorted_bboxes = pred_bboxes[pred_bboxes[:, 4].argsort()[::-1]]
    #     # Take the bounding box with the highest score
    #     highest_score_bbox = sorted_bboxes[0]
    #
    #     pred_bboxes_ssd.append(highest_score_bbox)
    # pred_bboxes_ssd = np.concatenate(pred_bboxes_ssd)

    # undo the scaling of the predicted bounding boxes to get the coordinates in the original image space
    #img_h, img_w, _ = img.shape
    #img_scale_h, img_scale_w = cfg.data.test.pipeline[1].img_scale
    #scale_factor = np.array([img_w, img_h, img_w, img_h]) / np.array(
    #    [img_scale_w, img_scale_h, img_scale_w, img_scale_h])
    #pred_bboxes[:, :4] = pred_bboxes[:, :4] / scale_factor[:4]

    # Round the coordinates to integers
    #pred_bboxes = pred_bboxes.round().astype(np.int32)


    # visualize the ground truth bounding boxes in blue and the predicted bounding boxes in green
    class_colors = {'left lung': 'r',
                    'right lung': 'g',
                    'cardiac silhouette': 'b',
                    'cavoatrial junction': 'c',
                    'aortic arch': 'm',
                    'carina': 'y'}
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    #for bbox in gt_bboxes:
    #    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='k', facecolor='none')
    #    ax.add_patch(rect)
    for bbox, class_name in zip(pred_bboxes, gt_class_names):
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor=class_colors[class_name], facecolor='none')
        ax.add_patch(rect)
        #ax.text(bbox[0], bbox[1] - 10, class_name, fontsize=10, color='b')
    #for bbox in pred_bboxes_yolo:
    #    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
    #    ax.add_patch(rect)
    #for bbox in pred_bboxes_ssd:
    #    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='b', facecolor='none')
    #    ax.add_patch(rect)

    plt.show()
    #fig.savefig('./analysis/cavoatrial junction_ssd/output'+str(i)+'.jpg')
