import mmcv
from mmdet.datasets import build_dataset
import numpy as np
from mmdet.apis import init_detector, inference_detector
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the configuration file for the detection model
config_file = './configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'

# Load the pre-trained weights of the model
#checkpoint_file = 'checkpoints/epoch_20_yolo3_hybrid.pth'
#device = 'cuda:2'  # or 'cpu'
#model = init_detector(config_file, checkpoint_file, device=device)

# load the test data using the built-in dataset function
cfg = mmcv.Config.fromfile(config_file)
test_dataset = build_dataset(cfg.data.test)

# specify the class indices you want to visualize
# [0, 1, 25, 5, 9, 12]
class_idxs = [0, 1, 25, 5, 9, 12]  # change this to the desired class indices

# loop over the test dataset and visualize the bounding boxes for each image
for i in range(len(test_dataset)-70):
    # get the i-th sample
    img_info = test_dataset.coco.loadImgs(test_dataset.img_ids[i])[0]
    img_path = test_dataset.img_prefix + '/' + img_info['file_name']
    img = mmcv.imread(img_path)

    # get the ground truth bounding boxes for the desired classes
    ann_ids = test_dataset.coco.getAnnIds(imgIds=img_info['id'], catIds=class_idxs)
    anns = test_dataset.coco.loadAnns(ann_ids)
    gt_bboxes = []
    gt_class_names = []
    for ann in anns:
        if ann['category_id'] in class_idxs:
            gt_bboxes.append(ann['bbox'])
            gt_class_names.append(test_dataset.CLASSES[ann['category_id']]) # subtract 1 because category IDs start from 1
    gt_bboxes = np.array(gt_bboxes)

    # run inference on the image to get the predicted bounding boxes
    #result = inference_detector(model, img)

    # get the predicted bounding boxes for the desired classes
    #pred_bboxes = []
    #for class_idx in class_idxs:
    #    pred_bboxes.append(result[class_idx])
    #pred_bboxes = np.concatenate(pred_bboxes)
    #pred_bboxes = []
    #pred_class_names = []
    #for class_idx in class_idxs:
    #    pred_bbox = result[class_idx]
    #    # Sort the bounding box predictions based on the score in descending order
    #    sorted_bboxes = pred_bbox[pred_bbox[:, 4].argsort()[::-1]]
    #    # Take the bounding box with the highest score
    #    highest_score_bbox = sorted_bboxes[0]

    #    pred_bboxes.append([highest_score_bbox])
    #    pred_class_names.append(test_dataset.CLASSES[class_idx])  # subtract 1 because category IDs start from 1
    #pred_bboxes = np.concatenate(pred_bboxes)

    # visualize the ground truth bounding boxes in blue and the predicted bounding boxes in green
    #'r': red
    #'g': green
    #'b': blue
    #'c': cyan
    #'m': magenta
    #'y': yellow
    #'k': black
    #'w': white
    class_colors = {'left lung':'r',
                    'right lung':'g',
                    'cardiac silhouette':'b',
                    'cavoatrial junction':'c',
                    'aortic arch':'m',
                    'carina':'y'}
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for bbox, class_name in zip(gt_bboxes, gt_class_names):
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor=class_colors[class_name], facecolor='none')
        ax.add_patch(rect)
        #ax.text(bbox[0], bbox[1] - 10, class_name, fontsize=10, color=class_colors[class_name])
    #for bbox, class_name in zip(pred_bboxes, pred_class_names):
    #    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='g', facecolor='none')
    #    ax.add_patch(rect)
    #    ax.text(bbox[0], bbox[1] - 10, class_name, fontsize=10, color='g')
    plt.show()
    #fig.savefig('./analysis/carina/output'+str(i)+'.jpg')

    #3:trachea
    #7:left clavicle
    #15:right clavicle
    #2:left costophrenic angle
    #4:right costophrenic angle
    #1:'left lung':'r',
    #0:'right lung':'g',
    #25:'cardiac silhouette':'b',
    #5:'cavoatrial junction':'c',
    #9:'aortic arch':'m',
    #12:'carina':'y'