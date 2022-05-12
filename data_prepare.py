"""
Author: Benny
Date: Nov 2019
"""
#NIFTI(NeuroImaging Informatics Technology Initiative)
#Nibabel:Brain MRI 영상등을 표현할 때 자주 쓰이는 NIFTI형식의 파일(.nii.gz)을 다룰 때에는 이 패키지를 많이 사용한다.
import argparse
import os
import nibabel as nib
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np

def main():
    #for data in [x for x in os.listdir('./data/ribfrac/ribfrac-train-images-1/Part1/')]:
    for data in [x for x in os.listdir('C:/data/Part1/')]: #nibabel에 이미지가 있다.
        #source = nib.load('./data/ribfrac/ribfrac-train-images-1/Part1/'+data)
        source = nib.load('C:/data/Part1/'+data)
        source = source.get_fdata() #전체image array 불러오기
        source[source >= 200] = 1
        source[source != 1] = 0

        #label = nib.load('./data/RibSeg/nii/'+data[:-12]+'rib-seg.nii.gz')
        #label = nib.load('./data/RibSeg/nii/'+data[:-12]+'c-rib-seg.nii')
        #label = nib.load('C:/data/RibSeg/nii/'+data[:-9]+'rib-seg.nii')#nibabel 이미지 객체를 생성하기 위해 파일을 로드한다. #nifti 이미지 읽기
        label = nib.load('C:/data/RibSeg/nii/MainForm/'+data[:-9]+'rib-seg.nii')
        label = label.get_fdata()

        temp = np.argwhere(source == 1)
#         choice = np.random.choice(temp.shape[0], 30000, replace=False)
##         downsample
#         points = temp[choice, :]

        label_selected_points = []
        for i in temp:
            label_selected_points.append(label[i[0]][i[1]]) 
            #Error: index 332 is out of bounds for axis 0 with size 325: 크기가 325인 축 0에 대한 인덱스 332가 범위를 벗어났습니다.([i[0]][i[1]][i[2]])
        label_selected_points = np.array(label_selected_points)
        np.save('C:/data/pn/data_pn/train'+ data[:-13], temp)
        #np.save('./data/pn/data_pn/train'+data[:-13], temp)
        np.save('C:/data/pn/label_pn/train' + data[:-13], label_selected_points)
        #np.save('./data/pn/label_pn/train' + data[:-13], label_selected_points)

    for data in [x for x in os.listdir('C:/data/Part2/')]:
    #for data in [x for x in os.listdir('./data/ribfrac/ribfrac-train-images-2/Part2/')]:
        source = nib.load('C:/data/Part2/'+data)
        #source = nib.load('./data/ribfrac/ribfrac-train-images-2/Part2/'+data)
        source = source.get_fdata()
        source[source >= 200] = 1
        source[source != 1] = 0

        label = nib.load('C:/data/RibSeg/nii/Mainform/' +data[:-9]+'rib-seg.nii')
#       label = nib.load('C:/data/RibSeg/nii/' +data[:-9]+'rib-seg.nii')
#       label = nib.load('./data/RibSeg/nii/'+data[:-12]+'rib-seg.nii.gz')
        label = label.get_fdata()

        temp = np.argwhere(source == 1)
#         choice = np.random.choice(temp.shape[0], 30000, replace=False)
#         # downsample
#         points = temp[choice, :]

        label_selected_points = []
        for i in temp:
            #label_selected_points.append(label[i[0]][i[1]][i[2]])
            label_selected_points.append(label[i[0]][i[1]])
        label_selected_points = np.array(label_selected_points)
        np.save('C:/data/pn/data_pn/train'+data[:-13], temp)
#        np.save('./data/pn/data_pn/train'+data[:-13], temp)
        np.save('C:/data/pn/label_pn/train' + data[:-13], label_selected_points)
#        np.save('./data/pn/label_pn/train' + data[:-13], label_selected_points)

    for data in [x for x in os.listdir('C:/data/ribfrac-val-images/')]:
#    for data in [x for x in os.listdir('./data/ribfrac/ribfrac-val-images/')]:
        source = nib.load('C:/data/ribfrac-val-images/' + data)
#        source = nib.load('./ribfrac/ribfrac-val-images/' + data)
        source = source.get_fdata()
        source[source >= 200] = 1
        source[source != 1] = 0
        label=nib.load('C:/data/Ribseg/nii/Mainform/'+data[:-9]+ 'rib-seg.nii')
        #label=nib.load('C:/data/Ribseg/nii/'+data[:-12]+ 'rib-seg.nii')
#        label = nib.load('./data/RibSeg/nii/' + data[:-12] + 'rib-seg.nii.gz')
        label = label.get_fdata()

        temp = np.argwhere(source == 1)
#         choice = np.random.choice(temp.shape[0], 30000, replace=False)
#         # downsample
#         points = temp[choice, :]

        label_selected_points = []
        for i in temp:
            label_selected_points.append(label[i[0]][i[1]])
            #label_selected_points.append(label[i[0]][i[1]][i[2]])
        label_selected_points = np.array(label_selected_points)
        np.save('C:/data/pn/data_pn/val' + data[:-13], temp)
        #np.save('./data/pn/data_pn/val' + data[:-13], temp)
        np.save('C:/data/pn/label_pn/val' + data[:-13], label_selected_points)
        #np.save('./data/pn/label_pn/val' + data[:-13], label_selected_points)

    for data in [x for x in os. listdir('C:/data/ribfrac-test-images/')]:
#    for data in [x for x in os.listdir('./data/ribfrac/ribfrac-test-images/')]:
        source = nib.load('C:/data/ribfrac-test-images/'+data)
#        source = nib.load('./data/ribfrac/ribfrac-test-images/'+data)
        source = source.get_fdata()
        source[source >= 200] = 1
        source[source != 1] = 0
        label = nib.load('C:/data/Ribseg/nii/Mainform/'+data[:-9]+'rib-seg.nii')
#        label = nib.load('./data/RibSeg/nii/'+data[:-12]+'rib-seg.nii.gz')
        label = label.get_fdata()

        temp = np.argwhere(source == 1)
#         choice = np.random.choice(temp.shape[0], 30000, replace=False)
#         # downsample
#         points = temp[choice, :]

        label_selected_points = []
        for i in temp:
            label_selected_points.append(label[i[0]][i[1]])
        temp = np.array(temp)
        np.save('C:/data/pn/data_pn/test'+data[:-13], temp)
#        np.save('./data/pn/data_pn/test'+data[:-13], temp)
        np.save('C:/data/pn/label_pn/test' + data[:-13], label_selected_points)

#        np.save('./data/pn/label_pn/test' + data[:-13], label_selected_points)

if __name__ == '__main__':
    main()

