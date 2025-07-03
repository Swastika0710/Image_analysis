
import glob
import os
import random
import sys
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
import re 
import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from torch.utils.data import Dataset

from utils import digi_mean, digi_std

import shutil



class Digitalrock_sub1(Dataset):
    def __init__(self, root, split='train', crop_size=None, mean=digi_mean, std=digi_std, ignore_label=255):

        self.split = split
        if split == 'train':
            assert crop_size is not None
            self.crop_h, self.crop_w = crop_size
        self.mean, self.std = mean, std
        self.ignore_label = ignore_label

        search_images = os.path.join(root, 'images', '*DR.png')
        search_labels = os.path.join(root, 'labels', '*L.png')
        search_grads = os.path.join(root, 'images', '*grad.png')
        search_boundaries = os.path.join(root, 'labels', '*boundary.png')
        self.images = glob.glob(search_images)
        
        self.labels = glob.glob(search_labels)
        self.grads = glob.glob(search_grads)
        self.boundaries = glob.glob(search_boundaries)
        self.images.sort()
        self.labels.sort()
        self.grads.sort()
        self.boundaries.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]
        grad_path = self.grads[index]
        boundary_path = self.boundaries[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        grad = cv2.imread(grad_path, cv2.IMREAD_GRAYSCALE)
        boundary = cv2.imread(boundary_path, cv2.IMREAD_GRAYSCALE)
        name = image_path.split('/')[-1]

        # random resize, multiple scale training
        if self.split == 'train':
            f_scale = random.choice([0.5, 1.0, 2.0])
            image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
            grad = cv2.resize(grad, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
            boundary = cv2.resize(boundary, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)
        grad = np.asarray(grad, np.float32)
        boundary = np.asarray(boundary, np.float32)
        
        # change to RGB
        image = image[:, :, ::-1]
        # normalization
        image = image / 255.0

        boundary = boundary / 255
        label = label/50
        
        image = image - self.mean
        image = image / self.std
        grad = grad / 255

        # random crop
        if self.split == 'train':
            img_h, img_w = label.shape
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                             pad_w, cv2.BORDER_CONSTANT,
                                             value=(0.0, 0.0, 0.0))
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                               pad_w, cv2.BORDER_CONSTANT,
                                               value=(self.ignore_label,))
                grad_pad = cv2.copyMakeBorder(grad, 0, pad_h, 0,
                                              pad_w, cv2.BORDER_CONSTANT,
                                              value=(0,))
                boundary_pad = cv2.copyMakeBorder(boundary, 0, pad_h, 0,
                                                  pad_w, cv2.BORDER_CONSTANT,
                                                  value=(0,))
            else:
                img_pad, label_pad, grad_pad, boundary_pad = image, label, grad, boundary

            img_h, img_w = label_pad.shape
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)
            image = img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
            label = label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
            grad = grad_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
            boundary = boundary_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
        # HWC -> CHW
        image = image.transpose((2, 0, 1))
        label = np.asarray(label, np.float32)

        # random horizontal flip
        if self.split == 'train':
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
            grad = grad[:, ::flip]
            boundary = boundary[:, ::flip]

        return image.copy(), label.copy(), np.expand_dims(grad, axis=0).copy(), boundary.copy(), name
class Digitalrock_sub2(Dataset):
    def __init__(self, root, split='train', crop_size=None, mean=digi_mean, std=digi_std, ignore_label=255):

        self.split = split
        if split == 'train':
            assert crop_size is not None
            self.crop_h, self.crop_w = crop_size
        self.mean, self.std = mean, std
        self.ignore_label = ignore_label

        search_images = os.path.join(root, 'images', '*DR.png')
        search_labels = os.path.join(root, 'labels', '*L.png')
        search_grads = os.path.join(root, 'images', '*grad.png')
        search_boundaries = os.path.join(root, 'labels', '*boundary.png')
        search_xy = os.path.join(root, 'XY', '*xy.png')
        self.images = glob.glob(search_images)
        
        self.labels = glob.glob(search_labels)
        self.grads = glob.glob(search_grads)
        self.boundaries = glob.glob(search_boundaries)
        self.xy = glob.glob(search_xy)
        self.images.sort()
        self.labels.sort()
        self.grads.sort()
        self.boundaries.sort()
        self.xy.sort()
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]
        grad_path = self.grads[index]
        boundary_path = self.boundaries[index]
        xy_path = self.xy[index]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        grad = cv2.imread(grad_path, cv2.IMREAD_GRAYSCALE)
        boundary = cv2.imread(boundary_path, cv2.IMREAD_GRAYSCALE)
        xy = cv2.imread(xy_path, cv2.IMREAD_GRAYSCALE)
        name = image_path.split('/')[-1]

        # random resize, multiple scale training
        if self.split == 'train':
            f_scale = random.choice([0.5, 1.0, 2.0])
            image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
            grad = cv2.resize(grad, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
            boundary = cv2.resize(boundary, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
            xy = cv2.resize(xy, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        image = np.asarray(image, np.float32)
        grad = np.asarray(grad, np.float32)
        boundary = np.asarray(boundary, np.float32)
        xy =  np.asarray(xy, np.float32)
        
        # change to RGB
        #image = image[:, :, ::-1]
        # normalization
        image = image / 255.0

        boundary = boundary / 255
        label = label/50
        xy = xy / 100
        image = image - 0.24
        image = image / 0.029
        grad = grad / 255

        image = np.stack((image, xy), axis=2)
        # random crop
        if self.split == 'train':
            img_h, img_w = label.shape
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                             pad_w, cv2.BORDER_CONSTANT,
                                             value=(0.0, 0.0, 0.0))
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                               pad_w, cv2.BORDER_CONSTANT,
                                               value=(self.ignore_label,))
                grad_pad = cv2.copyMakeBorder(grad, 0, pad_h, 0,
                                              pad_w, cv2.BORDER_CONSTANT,
                                              value=(0,))
                boundary_pad = cv2.copyMakeBorder(boundary, 0, pad_h, 0,
                                                  pad_w, cv2.BORDER_CONSTANT,
                                                  value=(0,))
            else:
                img_pad, label_pad, grad_pad, boundary_pad = image, label, grad, boundary

            img_h, img_w = label_pad.shape
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)
            image = img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
            label = label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
            grad = grad_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
            boundary = boundary_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
        # HWC -> CHW
        image = image.transpose((2, 0, 1))
        label = np.asarray(label, np.float32)

        # random horizontal flip
        if self.split == 'train':
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
            grad = grad[:, ::flip]
            boundary = boundary[:, ::flip]

        return image.copy(), label.copy(), np.expand_dims(grad, axis=0).copy(), boundary.copy(), name

def generate_grad(image_name, total_num):

    dst = image_name.replace('DR', 'grad')
    image_data = cv2.imread(image_name)[:,:,1]
    image_data = np.uint8(image_data)


    grad_image = cv2.Canny(np.uint8(image_data*255), 30, 30)
    
    # create the output filename


    cv2.imwrite(dst, grad_image)

    sys.stdout.flush()


def generate_boundary(image_name, num_classes, ignore_label, total_num):
    # create the output filename
    dst = image_name.replace('L', 'boundary')
    # do the conversion
    semantic_image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)/255
    onehot_image = np.array([semantic_image == i for i in range(num_classes)]).astype(np.uint8)
    # change the ignored label to 0
    onehot_image[onehot_image == ignore_label] = 0
    boundary_image = np.zeros(onehot_image.shape[1:])
    # for boundary conditions
    onehot_image = np.pad(onehot_image, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    for i in range(num_classes):
        dist = distance_transform_edt(onehot_image[i, :]) + distance_transform_edt(1.0 - onehot_image[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > 2] = 0
        boundary_image += dist
    boundary_image = (boundary_image > 0).astype(np.uint8)
    cv2.imwrite(dst, boundary_image*255)
    sys.stdout.flush()


def creat_dataset_xy(root, num_classes=3, ignore_label=255, type='train'):
    try:
        shutil.rmtree('./data/')
    except:
        pass
    os.mkdir('./data/')
    os.mkdir('./data/images')
    os.mkdir('./data/labels')

    col = None
    row = None
    zx = None



    water_label = 0
    rock_label = 1
    oil_label = 2

    images = None
    labels = None

    index_water = np.where(labels==1)
    index_rock = np.where(labels==2)
    index_oil = np.where(labels==0)
    index_none = np.where(labels==-1)

    labels[index_water] = water_label
    labels[index_rock] = rock_label
    labels[index_oil] = oil_label
    labels[index_none] = rock_label
    images = images.reshape(col, row, zx,order = 'F')
    labels = labels.reshape(col, row, zx,order = 'F')
    if type == 'train': 
        print(type)
        profile_xy_num = np.arange(1,zx,5)
        profile_xz_num = np.arange(1,row,5)
        print('profile xy',profile_xy_num)
        print('profile xz',profile_xz_num)
    else:
        print(type)
        profile_xy_num = np.arange(1,zx-1,1)
        profile_xz_num = np.arange(1,row-1,1)
        print('profile xy',profile_xy_num)
        print('profile xz',profile_xz_num)
    
    min_value = np.min(images)
    max_value = np.max(images)
    images = (images - min_value) / (max_value - min_value)*255
    image_flag = 1


    # xy train data process
    print('------------------ process xy ------------------------')
    for i in profile_xy_num:

        images_profile = images[:,:,i-1:i+2]


        file_name = str(image_flag)+'DR'+'.png'
        line_name = 'data/images/'+ file_name
        cv2.imwrite(line_name, images_profile)
        
        label_profile = labels[:,:,i]
        file_name = str(image_flag) + 'L' + '.png'
        line_name = 'data/labels/' + file_name
        cv2.imwrite(line_name, label_profile*50)
        image_flag = image_flag +1
    print('flag is',image_flag) 
    
    search_path = os.path.join(root, 'images', '*DR.png')
    if not glob.glob(search_path):
        if not os.path.exists(root):
            os.makedirs(root)

    search_path = os.path.join(root, 'labels', '*L.png')
    if not glob.glob(search_path):
        os.environ['DIGIROCK_DATASET'] = root
        os.system('csCreateTrainIdLabelImgs')
    search_path = os.path.join(root, 'images', '*grad.png')
    if not glob.glob(search_path):
        search_path = os.path.join(root, 'images',  '*DR.png')
        files = glob.glob(search_path)
        files.sort()
        # use multiprocessing to generate grad images
        pool = ThreadPool()
        pool.map(partial(generate_grad, total_num=len(files)), files)
        pool.close()
        pool.join()

    search_path = os.path.join(root, 'labels', '*boundary.png')
    if not glob.glob(search_path):
        search_path = os.path.join(root, 'labels', '*L.png')
        files = glob.glob(search_path)
        files.sort()

        pool = ThreadPool()
        pool.map(partial(generate_boundary, num_classes=num_classes, ignore_label=ignore_label, total_num=len(files)),
                 files)
        pool.close()
        pool.join()
        
def creat_dataset_xz(root, num_classes=3, ignore_label=255,type='train'):
    try:
        shutil.rmtree('./data/')
    except:
        pass
    os.mkdir('./data/')
    os.mkdir('./data/images')
    os.mkdir('./data/labels')

    col = None
    row = None
    zx = None



    water_label = 0
    rock_label = 1
    oil_label = 2

    images = None
    labels = None
    
    images_index = np.where(images<1)
    images[images_index] = 25000
    print("image size：",images.size)  
    print('label max',np.max(labels))
    print('label min',np.min(labels))
    index_water = np.where(labels==1)
    index_rock = np.where(labels==2)
    index_oil = np.where(labels==0)
    index_none = np.where(labels==-1)

    labels[index_water] = water_label
    labels[index_rock] = rock_label
    labels[index_oil] = oil_label
    labels[index_none] = rock_label
    images = images.reshape(col, row, zx,order = 'F')
    labels = labels.reshape(col, row, zx,order = 'F') 

    if type =='train':
        print(type)
        profile_xy_num = np.arange(1,zx-1,5)
        profile_xz_num = np.arange(1,row-1,5)
        print('profile xy',profile_xy_num)
        print('profile xz',profile_xz_num)

    else:
        print(type)
        profile_xy_num = np.arange(1,zx-2,1)
        profile_xz_num = np.arange(1,row-2,1)
        print('profile xy',profile_xy_num)
        print('profile xz',profile_xz_num)
    
    min_value = np.min(images)
    max_value = np.max(images)
    images = (images - min_value) / (max_value - min_value)*255
    image_flag = 1
    xy_name = np.load('./predict/name_list.npy',allow_pickle=True)
    predcit_xy = np.load('./predict/logits.npy',allow_pickle=True)
    print(predcit_xy[-1].shape)
    logits_xy = np.zeros([508,predcit_xy[-1].shape[1],predcit_xy[-1].shape[2],predcit_xy[-1].shape[3]])
    for i in range(len(xy_name)):
        for j in range(len(xy_name[i])):

            index = int(re.findall(r'\d+', xy_name[i][j])[0])
            logits_xy[index-1,:,:,:] = predcit_xy[i][j,:,:,:].numpy().copy()
    final_logit_xy = logits_xy[:,:,:,1:-1]
    prediction_xy = np.argmax(final_logit_xy,axis=1)
    print('xy_prediction shape is',prediction_xy.shape)
    # xy train data process
    print('------------------ process xz ------------------------')
    for i in profile_xz_num:

        images_profile = images[:,i-1:i+2,1:509].transpose(0,2,1)


        file_name = str(image_flag)+'DR'+'.png'
        line_name = 'data/images/'+ file_name
        cv2.imwrite(line_name, images_profile)
        
        label_profile = labels[:,i,1:509]
        file_name = str(image_flag) + 'L' + '.png'
        line_name = 'data/labels/' + file_name
        cv2.imwrite(line_name, label_profile*50)
        
        images_profile = prediction_xy[:,:,i]
        file_name = str(image_flag)+'xy'+'.png'
        line_name = 'data/XY/'+ file_name
        cv2.imwrite(line_name, images_profile.T*50)
        image_flag = image_flag +1
    print('flag is',image_flag) 
    
    search_path = os.path.join(root, 'images', '*DR.png')
    if not glob.glob(search_path):
        if not os.path.exists(root):
            os.makedirs(root)

    search_path = os.path.join(root, 'labels', '*L.png')
    if not glob.glob(search_path):
        # config the environment variable
        os.environ['DIGIROCK_DATASET'] = root
        # generate pixel labels
        os.system('csCreateTrainIdLabelImgs')
    search_path = os.path.join(root, 'images', '*grad.png')
    if not glob.glob(search_path):
        search_path = os.path.join(root, 'images',  '*DR.png')
        files = glob.glob(search_path)
        files.sort()
        # use multiprocessing to generate grad images
        pool = ThreadPool()
        pool.map(partial(generate_grad, total_num=len(files)), files)
        pool.close()
        pool.join()

    search_path = os.path.join(root, 'labels', '*boundary.png')
    if not glob.glob(search_path):
        search_path = os.path.join(root, 'labels', '*L.png')
        files = glob.glob(search_path)
        files.sort()
        # use multiprocessing to generate boundary images
        pool = ThreadPool()
        pool.map(partial(generate_boundary, num_classes=num_classes, ignore_label=ignore_label, total_num=len(files)),
                 files)
        pool.close()
        pool.join()
