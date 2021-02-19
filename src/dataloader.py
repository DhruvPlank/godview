import os
import xml.etree.ElementTree as Et

import numpy as np
import cv2
from torch.utils.data import Dataset



class ALOVDataset(Dataset):
    ''' handler for the ALOV tracking dataset '''

    def __init__(self, root_dir, target_dir, transform=None, input_size=227):
        super(ALOVDataset, self).__init__()

        self.exclude = ['01-Light_video00016',
                        '01-Light_video00022',
                        '01-Light_video00023',
                        '02-SurfaceCover_video00012',
                        '03-Specularity_video00003',
                        '03-Specularity_video00012',
                        '10-LowContrast_video00013']

        self.root_dir = root_dir
        self.target_dir = target_dir
        self.input_size = input_size
        self.transform = transform
        self.x, self.y = self._parse_data(root_dir, target_dir)
        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample, _  = self.get_sample(idx)
        if (self.transform):
            sample = self.transform(sample)
        return sample

    def _parse_data(self, root_dir, target_dir):
        ''' 
        Note : Parse ALOV dataset and return tuples of (template, search region)
               from annotated frames
        '''

        x = []
        y = []

        t_dir = os.listdir(target_dir)
        num_annot = 0
        print('Parsing ALOV dataset ...')

        for _file in t_dir:
            vids = os.listdir(root_dir + _file)
            for vid in vids:
                if vid in self.exclude:
                    continue

                vid_src = self.root_dir + _file + "/" + vid
                vid_ann = self.target_dir + _file + "/" + vid + ".ann"
                frames = os.listdir(vid_src)
                frames.sort()
                # frames as the entire path 
                frames = [vid_src + "/" + frame for frame in frames]
                frames = np.array(frames)

                f = open(vid_ann, "r")
                annotations = f.readlines()
                f.close()
                frame_idxs = [int(ann.split(' ')[0])-1 for ann in annotations]
                num_annot += len(annotations)

                for i in range(len(frame_idxs)-1):
                    curr_idx = frame_idxs[i]
                    next_idx = frame_idxs[i+1]

                    x.append([frames[curr_idx], frames[next_idx]])
                    y.append([annotations[curr_idx], annotations[next_idx]])


        x, y = np.array(x), np.array(y)
        self.len = len(y)
        print('ALOV dataset parsing done.')
        print(f'Total annotations in ALOV dataset : {num_annot}')

        return x, y


    def get_sample(self, idx):
        '''
        Get sample without doing any transformation for visualization.

        Sample consists of resized previous and current frame with target which
        is passed to the network. Bounding box values are normalized between 0 and 1
        with respect to the target frame and then scaled by factor of 10.
        '''
        opts_curr = {}
        curr_sample = {}
        curr_img = self.get_orig_sample(idx, 1)['image']
        currbb = self.get_orig_sample(idx, 1)['bb']
        prevbb = self.get_orig_sample(idx, 0)['bb']

        bbox_curr_shift = BoundingBox(prevbb[0],
                                      prevbb[1],
                                      prevbb[2],
                                      prevbb[3])

        rand_search_reg, rand_search_loc, edge_spacing_x, edge_spacing_y = cropPadImage(bbox_curr_shift, curr_img)

        bbox_curr_gt = BoundingBox(currbb[0],
                                   currbb[1],
                                   currbb[2],
                                   currbb[3])
        bbox_gt_recenter = BoundingBox(0, 0, 0, 0)
        bbox_gt_recenter = bbox_curr_gt.recenter(rand_search_loc, edge_spacing_x, edge_spacing_y, bbox_gt_recenter)

        curr_sample['image'] = rand_search_region
        curr_sample['bb'] = bbox_gt_recenter.get_bb_list()

        # options for visualization
        opts_curr['edge_spacing_x'] = edge_spacing_x
        opts_curr['edge_spacing_y'] = edge_spacing_y
        opts_curr['search_location'] = rand_search_loc
        opts_curr['search_region'] = rand_search_reg

        # build prev sample
        prev_sample = self.get_orig_sample(idx, 0)
        preV_sample, opts_prev = crop_sample(prev_sample)

        # scale
        scale = Rescale((self.input_size, self.input_size))
        scaled_curr_obj = scale(curr_sample, opts_curr)
        scaled_prev_obj = scale(prev_sample, opts_prev)

        training_sample = {'previmg' : scaled_prev_obj['image'],
                           'currimg' : scaled_curr_obj['image'],
                           'currbb'  : scaled_curr_obj['bb']}

        return training_sample, opts_curr


    def get_orig_sample(self, idx, i=1):
        '''
        Returns original iamge with bounding box at a specific index.
        Range of valid index : [0, self.len - 1]
        '''

import time
time.












