import mmcv
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# Define the configuration file for the detection model
config_file = './configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'

# Load the pre-trained weights of the model
checkpoint_file = 'checkpoints/epoch_80_faster_rcnn.pth'
device = 'cuda:3'  # or 'cpu'
model = init_detector(config_file, checkpoint_file, device=device)

# Load the image
image = mmcv.imread('img1.jpg')

# Run the detection model on the image
result = inference_detector(model, image)

# Display the predicted bounding boxes and class labels on the image
show_result_pyplot(model, image, result, score_thr=0.3)
