#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 16:30:46 2021

@author: brijeshlakkad
"""
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import random
import xml.etree.ElementTree as ET

data_folder = './face_mask_detection/'
root_folder = '/Users/brijeshlakkad/Downloads/VMI\ face\ detection/'


def get_objects(xml_file):
    annotations = ET.parse(xml_file)
    root = annotations.getroot()
    objects = []
    for obj in root.findall('object'):
      new_object = {'name': obj.find('name').text}
      objects.append(new_object)
    return objects


size_of_the_dataset = 853

indexes = list(range(size_of_the_dataset))
random.shuffle(indexes)

label_stat = {'with_mask': 0, 'without_mask': 0,
              'mask_weared_incorrect': 0, 'num_bbox': 0}

indexes = list(range(size_of_the_dataset))
for i in indexes:
    objects = get_objects(os.path.join(data_folder, 'annotations', 'maksssksksss' + str(i) + '.xml'))
    for d in objects:
        label_stat[d['name']] += 1
        label_stat['num_bbox'] += 1


label_stat['num_bbox_per_image'] = label_stat['num_bbox']/size_of_the_dataset
print(label_stat)