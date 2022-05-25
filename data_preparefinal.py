import argparse
import os
import nibabel as nib
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np
nii_train1 = []

for data in [x for x in os.listdir('C:/data/ribfrac/ribfrac-train-images-1/Part1/')]:
    if str(data[:-12]+'rib-seg.nii') in os.listdir('C:/data/RibSeg/nii/'):
        nii_train1.append(data)
print(nii_train1)
for data in nii_train1:
    source = nib.load('C:/data/ribfrac/ribfrac-train-images-1/Part1/'+data)
    source = source.get_fdata()
    source[source >= 200] = 1
    source[source != 1] = 0

    label = nib.load('C:/data/RibSeg/nii/'+data[:-12]+'rib-seg.nii')
    label = label.get_fdata()

    temp = np.argwhere(source == 1)
    #choice = np.random.choice(temp.shape[0], 30000, replace=False)
    ##downsample
    #points = temp[choice, :]

    label_selected_points = []
    for i in temp:
        label_selected_points.append(label[i[0]][i[1]][i[2]])
    label_selected_points = np.array(label_selected_points)
    np.save('C:/data/pn/data_pn/train'+data[:-13], temp)
    np.save('C:/data/pn/label_pn/train' + data[:-13], label_selected_points)
nii_train2 = []

for data in [x for x in os.listdir('C:/data/ribfrac/ribfrac-train-images-2/Part2/')]:
    if str(data[:-12]+'rib-seg.nii') in os.listdir('C:/data/RibSeg/nii/'):
        nii_train2.append(data)
print(nii_train2)
for data in nii_train2:
    source = nib.load('C:/data/ribfrac/ribfrac-train-images-2/Part2/'+data)
    source = source.get_fdata()
    source[source >= 200] = 1
    source[source != 1] = 0

    label = nib.load('C:/data/RibSeg/nii/'+data[:-12]+'rib-seg.nii')
    label = label.get_fdata()

    temp = np.argwhere(source == 1)
#    choice = np.random.choice(temp.shape[0], 30000, replace=False)
#    # downsample
#    points = temp[choice, :]

    label_selected_points = []
    for i in temp:
        label_selected_points.append(label[i[0]][i[1]][i[2]])
    label_selected_points = np.array(label_selected_points)
    np.save('C:/data/pn/data_pn/train'+data[:-13], temp)
    np.save('C:/data/pn/label_pn/train'+ data[:-13], label_selected_points)
nii_val = []

for data in [x for x in os.listdir('C:/data/ribfrac/ribfrac-val-images/')]:
    if str(data[:-12]+'rib-seg.nii') in os.listdir('C:/data/RibSeg/nii/'):
        nii_val.append(data)
print(nii_val)
for data in nii_val:
    source = nib.load('C:/data/ribfrac/ribfrac-val-images/' + data)
    source = source.get_fdata()
    source[source >= 200] = 1
    source[source != 1] = 0

    label = nib.load('C:/data/RibSeg/nii/' + data[:-12] + 'rib-seg.nii')
    label = label.get_fdata()

    temp = np.argwhere(source == 1)
#    choice = np.random.choice(temp.shape[0], 30000, replace=False)
#    # downsample
#    points = temp[choice, :]

    label_selected_points = []
    for i in temp:
        label_selected_points.append(label[i[0]][i[1]][i[2]])
    label_selected_points = np.array(label_selected_points)
    np.save('C:/data/pn/data_pn/val' + data[:-13], temp)
    np.save('C:/data/pn/label_pn/val'+ data[:-13], label_selected_points)
nii_test = []

for data in [x for x in os.listdir('C:/data/ribfrac/ribfrac-test-images/')]:
    if str(data[:-12]+'rib-seg.nii') in os.listdir('C:/data/RibSeg/nii/'):
        nii_test.append(data)
print(nii_test)
for data in nii_test:
    source = nib.load('C:/data/ribfrac/ribfrac-test-images/'+data)
    source = source.get_fdata()
    source[source >= 200] = 1
    source[source != 1] = 0

    label = nib.load('C:/data/RibSeg/nii/'+data[:-12]+'rib-seg.nii')
    label = label.get_fdata()

    temp = np.argwhere(source == 1)
#    choice = np.random.choice(temp.shape[0], 30000, replace=False)
#    # downsample
#    points = temp[choice, :]

    label_selected_points = []
    for i in temp:
        label_selected_points.append(label[i[0]][i[1]][i[2]])
    temp = np.array(temp)
    np.save('C:/data/pn/data_pn/test'+data[:-13], temp)
    np.save('C:/data/pn/label_pn/test'  + data[:-13], label_selected_points)
