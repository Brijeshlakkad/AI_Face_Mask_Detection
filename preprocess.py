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

trainDF = pd.read_csv(os.path.join(dataFolder, inputFile),
                      skiprows=1, names=columns)

iterator = trainDF.iterrows()

ouputCSVFile = []

preselectedClasses = [
    "face_with_mask",
    "mask_colorful",
    "face_no_mask",
    "face_with_mask_incorrect",
    "mask_surgical",
]

for rowIndex in range(len(trainDF)):
    className = trainDF[columns[5]][rowIndex]
    if className in preselectedClasses:
        ouputCSVFile.append(rowIndex)


def saveCSV(dirPath, fileName, rows):
    if not os.path.isdir(dirPath):
        os.makedirs(dirPath)
    df = trainDF.iloc[rows, :]
    df.to_csv(os.path.join(dirPath, fileName))
    print(f'Saved file {fileName} successfully!')


saveCSV(os.path.join(rootDir, 'preprocessed'), "data.csv", ouputCSVFile)
