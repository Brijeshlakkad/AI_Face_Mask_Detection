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

rootDir = './data/preprocessed'
if not os.path.isdir(rootDir):
    os.makedirs(rootDir)

trainCSVFileName = "./data/face-mask-detection-dataset/train.csv"
#testCSVFileName = pd.read_csv("./data/face-mask-detection-dataset/submission.csv")

columns = ["name", "x1", "x2", "y1", "y2", "classname"]

trainDF = pd.read_csv(trainCSVFileName, skiprows=1, names=columns)

iterator = trainDF.iterrows()

ouputCSVFiles = {}

for rowIndex in range(len(trainDF)):
    className = trainDF[columns[5]][rowIndex]
    if className not in ouputCSVFiles:
        ouputCSVFiles[className] = []
    ouputCSVFiles[className].append(rowIndex)


def saveCSV(fileName, rows):
    # for i, row in enumerate(rows):
    #     record = trainDF[:][rowIndex]

    df = trainDF.iloc[rows, :]
    df.to_csv(os.path.join(rootDir, fileName+".csv"))

for fileName in list(ouputCSVFiles.keys()):
    print(f'Creating file.. {fileName}')
    saveCSV(fileName, ouputCSVFiles[fileName])