import mmcv
import copy
from mmcv import Config
import os.path as osp
import mmdet
from mmdet.datasets.builder import DATASETS
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')
#cfg = Config.fromfile('./configs/yolo/yolov3_d53_mstrain-608_273e_coco.py')
#cfg = Config.fromfile('./configs/ssd/ssd512_coco.py')

mmdet.datasets.coco.CocoDataset.CLASSES=('trachea', 'right atrium', 'left upper lung zone', 'left hilar structures',
    'left apical zone', 'left clavicle', 'cavoatrial junction', 'right lower lung zone',
    'left hemidiaphragm', 'right hemidiaphragm', 'right clavicle',
    'left costophrenic angle', 'aortic arch', 'right apical zone', 'left mid lung zone',
    'right lung', 'right mid lung zone', 'upper mediastinum', 'left lower lung zone',
    'cardiac silhouette', 'svc', 'left lung', 'right costophrenic angle',
    'carina', 'right hilar structures', 'right upper lung zone')

# If we need to finetune a model based on a pre-trained detector, we need to
# use load_from to set the path of checkpoints.
cfg.load_from = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'
#cfg.load_from = 'checkpoints/yolov3_d53_fp16_mstrain-608_273e_coco_20210517_213542-4bc34944.pth'
#cfg.load_from = 'checkpoints/ssd512_coco_20210803_022849-0a47a1ca.pth'

# Set up working dir to save files and logs.
#cfg.work_dir = './tutorial_exps'
#cfg.work_dir = './yolo3_shift'
#cfg.work_dir = './ssd_shift'
cfg.work_dir = './faster_rcnn_new_c'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
#cfg.optimizer.lr = 0.02 / 8
#cfg.optimizer.lr = 0.0002 / 8
#cfg.lr_config.warmup = None
cfg.log_config.interval = 50

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'bbox'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 200
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 5

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
#device = torch.device('cuda:3')
cfg.device = 'cuda'
cfg.gpu_ids = [4]

# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

##add val
cfg.workflow = [('train',1),('val',1)]

# We can initialize the logger for training and have a look
# at the final config used for training
#print(f'Config:\n{cfg.pretty_text}')

# Build dataset
#datasets = [build_dataset(cfg.data.train),build_dataset(cfg.data.val)]
datasets = [build_dataset(cfg.data.train)]
if len(cfg.workflow) == 2:
    val_dataset = copy.deepcopy(cfg.data.val)
    val_dataset.pipeline = cfg.data.train.pipeline
    datasets.append(build_dataset(val_dataset))
# Build the detector
model = build_detector(cfg.model)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

##cuda
#device = torch.device('cuda:3')
#model.to(device)


# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)