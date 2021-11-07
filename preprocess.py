#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 16:30:46 2021

@author: brijeshlakkad
"""
import math
import os

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

rootDir = './data'
if not os.path.isdir(rootDir):
    os.makedirs(rootDir)


dataFolder = "./data/face-mask-detection-dataset"
inputFile = 'train.csv'

columns = ["name", "x1", "x2", "y1", "y2", "classname"]

train_df = pd.read_csv(os.path.join(dataFolder, inputFile),
                       skiprows=1, names=columns)

iterator = train_df.iterrows()

def preview_classes():
    classes = []
    preselectedClasses = {
        "face_with_mask": 0,
        "mask_colorful": 0,
        "face_no_mask": 0,
        "face_with_mask_incorrect": 0,
        "mask_surgical": 0
    }
    for row_index in range(len(train_df)):
        className = train_df[columns[5]][row_index]
        if className not in classes:
            classes.append(className)
        if className in preselectedClasses.keys():
            preselectedClasses[className] += 1
    return classes, preselectedClasses


preselectedClasses = [
    "face_with_mask",
    "mask_colorful",
    "face_no_mask",
    "mask_surgical",
]

def saveCSV(dirPath, fileName):
    if not os.path.isdir(dirPath):
        os.makedirs(dirPath)

    selectedClassIndexes = []
    for row_index in range(len(train_df)):
        className = train_df[columns[5]][row_index]
        if className in preselectedClasses:
            selectedClassIndexes.append(row_index)

    df = train_df.iloc[selectedClassIndexes, :]
    df.to_csv(os.path.join(dirPath, fileName))
    print(f'Saved file {fileName} successfully!')


saveCSV(os.path.join(rootDir, 'preprocessed'), "data.csv")
