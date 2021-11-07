#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 16:30:46 2021

@author: brijeshlakkad
"""
import os
import pandas as pd

rootDir = './data'
if not os.path.isdir(rootDir):
    os.makedirs(rootDir)


data_folder = "./data/face-mask-detection-dataset"
input_file = 'train.csv'

columns = ["name", "x1", "x2", "y1", "y2", "classname"]

train_df = pd.read_csv(os.path.join(data_folder, input_file),
                       skiprows=1, names=columns)

iterator = train_df.iterrows()


def preview_classes():
    classes = []
    preselected_class_count = {
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
        if className in preselected_class_count.keys():
            preselected_class_count[className] += 1
    return classes, preselected_class_count


preselected_classes = [
    "face_with_mask",
    "mask_colorful",
    "face_no_mask",
    "mask_surgical",
]


def make_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def save_preselected_class_data(dir_path, file_name):
    make_dir(dir_path)

    selectedClassIndexes = []
    for row_index in range(len(train_df)):
        className = train_df[columns[5]][row_index]
        if className in preselected_classes:
            selectedClassIndexes.append(row_index)

    df = train_df.iloc[selectedClassIndexes, :]
    df.to_csv(os.path.join(dir_path, file_name))
    print(f'Saved file {file_name} successfully!')


def extract_class_data(src_dir, dir_path, p_class_name):
    from shutil import copyfile

    class_dir_path = os.path.join(dir_path, p_class_name)
    
    make_dir(class_dir_path)
    copied = 0
    for row_index in range(len(train_df)):
        class_name = train_df[columns[5]][row_index]
        if class_name == p_class_name:
            copyfile(os.path.join(src_dir, train_df[columns[0]][row_index]), os.path.join(class_dir_path, train_df[columns[0]][row_index]))
            copied+=1
    
    print(f'{copied} files to {class_dir_path}')

    


# save_preselected_class_data(os.path.join(rootDir, 'preprocessed'), "data.csv")
# extract_class_data(os.path.join(data_folder, 'images'), os.path.join(rootDir, 'preprocessed'), "face_with_mask")
