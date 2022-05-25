"""
import cv2
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
"""
"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.RibFracDataLoader_1cls import PartNormalDataset
import data_utils.data_trans as d_utils
import torchvision
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import gc
gc.collect()
torch.cuda.empty_cache()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


# seg_classes = {'rib':[0,1]}
# we adapted the PointNet++ code from https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git, 
# to keep the consistency, we simply let the number of classes be the same as the source code.
# For experiments, the only classes used will be 'rib': 0 ,1.

seg_classes = {'rib':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],'Earphone': [44, 45], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [29], 'Laptop': [28 ], 'Cap': [26], 'Skateboard': [46],
                            'Mug': [36], 'Guitar': [39, 40], 'Bag': [27], 'Lamp': [25],
                            'Table': [47], 'Airplane': [48], 'Pistol': [38],
                            'Chair': [37], 'Knife': [49]}


seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg', help='model name [default: pointnet2_part_seg_msg]')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default:16]')
    parser.add_argument('--epoch',  default=251, type=int, help='Epoch to run [default: 251]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=30000, help='Point Number [default: 2048]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--step_size', type=int,  default=20, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.5, help='Decay rate for lr decay [default: 0.5]')

    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('part_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'C:/data/pn/'

    transforms = torchvision.transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            # d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
            d_utils.PointcloudScale(),
            d_utils.PointcloudTranslate(),
            d_utils.PointcloudJitter(),
        ]
    )

    TRAIN_DATASET = PartNormalDataset(root = root, npoints=args.npoint, split='trainval',transforms=transforms, normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size,shuffle=True, num_workers=0)
    TEST_DATASET = PartNormalDataset(root = root, npoints=args.npoint, split='test', transforms=None,normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, num_workers=0)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" %  len(TEST_DATASET))
    num_classes = 16 #1
    num_part = 50 #2
    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet_util.py', str(experiment_dir))

    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()


    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        mean_correct = []
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))

        '''learning one epoch'''
        for i, data in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points, label, target = data
            # print(points.shape,label.shape,target.shape)
            points = points.data.numpy()
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(),label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            optimizer.zero_grad()
            classifier = classifier.train()
            
            print(points.size())
            #print('label:',label,label.size())
            a=to_categorical(label, num_classes)
            #print('ssss:',a,a.size())
            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            print('seg_pred',seg_pred.shape)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            loss = criterion(seg_pred, target, trans_feat)
            loss.backward()
            optimizer.step()
        train_instance_acc = np.mean(mean_correct)
        log_string('Train accuracy is: %.5f' % train_instance_acc)

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                classifier = classifier.eval()
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32) 
                target = target.cpu().data.numpy()
                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
            for cat in sorted(shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)


        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
                 epoch+1, test_metrics['accuracy'],test_metrics['class_avg_iou'],test_metrics['inctance_avg_iou']))
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s'% savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        print('Dice:',loss)
        log_string('Best accuracy is: %.5f'%best_acc)
        log_string('Best class avg mIOU is: %.5f'%best_class_avg_iou)
        log_string('Best inctance avg mIOU is: %.5f'%best_inctance_avg_iou)
        global_epoch+=1
if __name__ == '__main__':
    args = parse_args()
    main(args)

"""        
def  infer(model, image_tensor):
    predictions =model.predict(np.expand_dims((image_tensor),axis=0))
    predictions=np.squeeze(predictions)
    predictions=np.argmax(predictions, axis=2)
    return predictions
def decode_segmentation_masks(mask, colormap, n_classes):
    r=np.zeros_like(mask).astype(np.uint8)
    g=np.zeros_like(mask).astype(np.uint8)
    b=np.zeros_like(mask).astype(np.uint8)
    for i in range(0, n_classes):
        idx = mask == i
        r[idx] = colormap[i,0]
        g[idx] = colormap[i,1]
        b[idx] = colormap[i,2]
    rgb = np.stack([r,g,b], axis=2)
    return rgb
def get_overlay(image, colored_mask):
    image=tf.keras.preprocessing.image.array_to_img(image)
    image=np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay
def plot_samples_matplotlib(display_list, figsize=(5,3)):
    _, axes=plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1]==3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.show()       
#def plot_predictions(image_list, colormap, model):
#    for image_file in images_list:
#        image_tensor=read_image(image_file)
#       prediction_mask=infer(image_tensor=image_tensor,model=model)
#        prediction_colormap=decode_segmentation_masks(prediction_mask, colormap,20)
#        overlay = get_overlay(image_tensor, prediction_colormap)
#        plot_samples_matplotlib([image_tensor, overlay,prediction_colormap],figsize=(18,14))
#    plot_predictions(train_images[:4], colormap, model=model)
"""


