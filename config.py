import argparse
from sympy import root

import torch

LAMBDA_DICT_IMG_INPAINTING = {
    'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0
}
LAMBDA_DICT_PR_INPAINTING = {
    'SSL-OUT': 1.0,  # 'SSL-OUT-COMP': 1.0, 'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0
}

data_type = None
evaluation_dir = None
partitions = None
infill = None
create_images = None
create_video = None
create_report = None
log_dir = None
snapshot_dir = None
data_root_dir = None
mask_dir = None
resume = None
device = None
batch_size = None
n_threads = None
finetune = None
lr = None
lr_finetune = None
max_iter = None
log_interval = None
save_model_interval = None
prev_next = None
lstm_steps = None
encoding_layers = None
pooling_layers = None
image_size = None
image_dir = None
mask_dir = None
image_name = None
mask_name = None
depth = None
mode = None
root_dir = None
save_dir = None
mask_year = None
im_year = None
vis_interval = None
in_channels = None
resume_iter = None
image_size = None
mode = None
attributes = None

def set_train_args():
    arg_parser = argparse.ArgumentParser()
    # training options
    arg_parser.add_argument('--data-type', type=str, default='tas')
    arg_parser.add_argument('--root_dir', type=str, default='/')
    arg_parser.add_argument('--mask_dir', type=str, default='../Asi_maskiert/original_masks/')
    arg_parser.add_argument('--image_dir', type=str, default='../Asi_maskiert/original_image/')
    arg_parser.add_argument('--save_part', type=str, default='part_1')
    arg_parser.add_argument('--save_dir', type=str, default='../Asi_maskiert/results/')
    arg_parser.add_argument('--log_dir', type=str, default='./logs/default')
    arg_parser.add_argument('--device', type=str, default='cuda')
    arg_parser.add_argument('--mask_year', type=str, default='2020')
    arg_parser.add_argument('--im_year', type=str, default='tho_r8_12')
    arg_parser.add_argument('--finetune', action='store_true')
    arg_parser.add_argument('--lr', type=float, default=2e-4)
    arg_parser.add_argument('--lr_finetune', type=float, default=5e-5)
    arg_parser.add_argument('--max_iter', type=int, default=800000)
    arg_parser.add_argument('--batch_size', type=int, default=4)
    arg_parser.add_argument('--n_threads', type=int, default=18) 
    arg_parser.add_argument('--save_model_interval', type=int, default=50000)
    arg_parser.add_argument('--vis_interval', type=int, default=50000)
    arg_parser.add_argument('--log_interval', type=int, default=50)
    arg_parser.add_argument('--image_size', type=int, default=256)
    arg_parser.add_argument('--in_channels', type=int, default=3)
    arg_parser.add_argument('--depth', action='store_true')
    arg_parser.add_argument('--resume_iter', type=str)
    arg_parser.add_argument('--prev-next', type=int, default=0)
    arg_parser.add_argument('--lstm-steps', type=int, default=0)
    arg_parser.add_argument('--encoding-layers', type=int, default=3)
    arg_parser.add_argument('--pooling-layers', type=int, default=0)
    args = arg_parser.parse_args()


    global data_type
    global root_dir
    global log_dir
    global mask_dir
    global image_dir
    global save_dir
    global mask_year
    global im_year
    global finetune
    global lr
    global lr_finetune
    global max_iter
    global device
    global batch_size
    global n_threads
    global vis_interval
    global log_interval
    global image_size
    global in_channels
    global depth
    global resume_iter
    global save_model_interval
    global prev_next
    global lstm_steps
    global encoding_layers
    global pooling_layers

    root_dir = args.root_dir
    mask_year = args.mask_year
    im_year = args.im_year
    data_type = args.data_type
    log_dir = args.log_dir
    mask_dir = args.mask_dir
    image_dir = args.image_dir
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    batch_size = args.batch_size
    n_threads = args.n_threads
    finetune = args.finetune
    lr = args.lr
    lr_finetune = args.lr_finetune
    max_iter = args.max_iter
    vis_interval = args.vis_interval
    log_interval = args.log_interval
    save_model_interval = args.save_model_interval
    prev_next = args.prev_next
    lstm_steps = args.lstm_steps
    encoding_layers = args.encoding_layers
    pooling_layers = args.pooling_layers
    image_size = args.image_size
    depth = args.depth
    resume_iter = args.resume_iter
    in_channels = args.in_channels



def set_evaluation_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-type', type=str, default='tas')
    arg_parser.add_argument('--evaluation-dir', type=str, default='evaluation/')
    arg_parser.add_argument('--snapshot-dir', type=str, default='snapshots/')
    arg_parser.add_argument('--data-root-dir', type=str, default='../data/')
    arg_parser.add_argument('--mask-dir', type=str, default='masks/')
    arg_parser.add_argument('--device', type=str, default='cuda')
    arg_parser.add_argument('--partitions', type=int, default=1)
    arg_parser.add_argument('--prev-next', type=int, default=0)
    arg_parser.add_argument('--lstm-steps', type=int, default=0)
    arg_parser.add_argument('--encoding-layers', type=int, default=3)
    arg_parser.add_argument('--pooling-layers', type=int, default=0)
    arg_parser.add_argument('--image-size', type=int, default=72)
    arg_parser.add_argument('--infill', type=str, default=None)
    arg_parser.add_argument('--create-images', type=str, default='2017-07-12-14:00,2017-07-12-14:00')
    arg_parser.add_argument('--create-video', action='store_true')
    arg_parser.add_argument('--create-report', action='store_true')
    args = arg_parser.parse_args()

    global data_type
    global evaluation_dir
    global snapshot_dir
    global data_root_dir
    global mask_dir
    global device
    global partitions
    global prev_next
    global lstm_steps
    global encoding_layers
    global pooling_layers
    global image_size
    global infill
    global create_images
    global create_video
    global create_report

    data_type = args.data_type
    evaluation_dir = args.evaluation_dir
    snapshot_dir = args.snapshot_dir
    data_root_dir = args.data_root_dir
    mask_dir = args.mask_dir
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    partitions = args.partitions
    prev_next = args.prev_next
    lstm_steps = args.lstm_steps
    encoding_layers = args.encoding_layers
    pooling_layers = args.pooling_layers
    image_size = args.image_size
    infill = args.infill
    create_images = args.create_images
    create_video = args.create_video
    create_report = args.create_report


def set_preprocessing_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--image_size', type=int, default=128)
    arg_parser.add_argument('--image_dir', type=str, default='../Asi_maskiert/original_image/')
    arg_parser.add_argument('--mask_dir', type=str, default='../Asi_maskiert/original_masks/')
    arg_parser.add_argument('--image_name', type=str, default='Image_r10_newgrid')
    arg_parser.add_argument('--mask_name', type=str, default='Maske_1970_1985_newgrid')
    arg_parser.add_argument('--depth', type=int, default=3)
    arg_parser.add_argument('--mode', type=str, default='image')
    arg_parser.add_argument('--attributes', type=str, default='_newgrid_anomalies')
    args = arg_parser.parse_args()


    global image_size
    global image_dir
    global mask_dir
    global image_name
    global mask_name
    global depth
    global mode
    global attributes

    image_size = args.image_size
    image_dir = args.image_dir
    mask_dir = args.mask_dir
    image_name = args.image_name
    mask_name = args.mask_name
    depth = args.depth
    mode = args.mode
    attributes = args.attributes





