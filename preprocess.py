#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 16:30:46 2021

@author: brijeshlakkad
"""
import os
import pandas as pd

root_dir = './data'
data_folder = "./data/face-mask-detection-dataset"
columns = ["name", "x1", "x2", "y1", "y2", "classname"]

if not os.path.isdir(root_dir):
    os.makedirs(root_dir)


def get_df():
    input_file = "train.csv"
    return pd.read_csv(os.path.join(data_folder, input_file),
                       skiprows=1, names=columns)


def get_preprocessed_df():
    return pd.read_csv(os.path.join(root_dir, "preprocessed", "data.csv"),
                       skiprows=1, names=["filename", "classname"])


def make_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def save_preselected_class_data(dir_path, file_name):
    make_dir(dir_path)

    preselected_classes = [
        "face_with_mask",
        "mask_colorful",
        "face_no_mask",
        "mask_surgical",
    ]

    selectedClassIndexes = []
    train_df = get_df()
    for row_index in range(len(train_df)):
        className = train_df[columns[5]][row_index]
        if className in preselected_classes:
            selectedClassIndexes.append(row_index)

    df = train_df.iloc[selectedClassIndexes, :]
    df.to_csv(os.path.join(dir_path, file_name))
    print(f'Saved file {file_name} successfully!')


def extract_same_class_files(src_dir, dir_path, p_class_name):
    from shutil import copyfile

    class_dir_path = os.path.join(dir_path, p_class_name)

    make_dir(class_dir_path)
    copied = 0
    train_df = get_df()
    for row_index in range(len(train_df)):
        class_name = train_df[columns[5]][row_index]
        if class_name == p_class_name:
            copyfile(os.path.join(src_dir, train_df[columns[0]][row_index]), os.path.join(
                class_dir_path, train_df[columns[0]][row_index]))
            copied += 1

    print(f'{copied} files to {class_dir_path}')


def move_files_using_file(src_dir, class_dir_path, file_name):
    make_dir(class_dir_path)
    from shutil import copyfile

    # Opening file
    input_file = open(file_name, 'r')
    # Using for loop
    for file in input_file:
        file = file.strip('\n')
        copyfile(os.path.join(src_dir, file),
                 os.path.join(class_dir_path, file))

    # Closing files
    input_file.close()


def move_files_using_list(src_dir, class_dir_path, l):
    make_dir(class_dir_path)
    from shutil import copyfile

    for file in l:
        copyfile(os.path.join(src_dir, file),
                 os.path.join(class_dir_path, file))


def is_file(file_name):
    return file_name != '.DS_Store'


def create_dataset(src_dir, dest_dir, external_data):
    import csv
    from shutil import copyfile
    make_dir(dest_dir)
    seed = 1000

    train_df = get_df()
    preselected_classes = [
        "mask_colorful",
        "face_no_mask",
        "mask_surgical",
    ]
    extract_count = 400
    class_dict = []
    class_count_tracker = {

    }
    processed_files = []

    file_index = seed
    for row_index in range(len(train_df)):
        current_class_name = train_df[columns[5]][row_index]
        if current_class_name not in preselected_classes:
            continue
        if current_class_name not in class_count_tracker:
            class_count_tracker[current_class_name] = 0
        elif class_count_tracker[current_class_name] >= extract_count:
            continue
        file_name = train_df[columns[0]][row_index]
        if not is_file(file_name) or file_name in processed_files:
            continue
        class_count_tracker[current_class_name] += 1
        file_index += 1
        new_file_name = str(file_index)+'.'+file_name.split(".")[-1]
        copyfile(os.path.join(src_dir, file_name),
                 os.path.join(dest_dir, new_file_name))
        class_dict.append({"filename": new_file_name,
                          "classname": current_class_name})
        processed_files.append(file_name)

    if external_data is not None:
        data_path = external_data["path"]
        classname = external_data["classname"]
        for file_name in os.listdir(data_path):
            if not is_file(file_name):
                continue
            file_index += 1
            new_file_name = str(file_index)+'.'+file_name.split(".")[-1]
            copyfile(os.path.join(data_path, file_name),
                     os.path.join(dest_dir, new_file_name))
            class_dict.append(
                {"filename": new_file_name, "classname": classname})

    # print(class_dict)
    with open(os.path.join(dest_dir, "..", "data.csv"), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename', 'classname'])
        writer.writeheader()
        writer.writerows(class_dict)

    print("File created: ", (file_index-seed))


def preview_classes():
    classes = []
    preselected_class_count = {
        "mask_colorful": 0,
        "face_no_mask": 0,
        "ffp2_mask": 0,
        "mask_surgical": 0
    }
    train_df = get_preprocessed_df()
    for row_index in range(len(train_df)):
        className = train_df[columns[5]][row_index]
        if className not in classes:
            classes.append(className)
        if className in preselected_class_count.keys():
            preselected_class_count[className] += 1
    print(classes, preselected_class_count)


def deleteNotRequiredImages():
    """Removes images that doesn't belong to new_dataset/train.csv"""
    root_dir = os.path.join("data", "new_dataset")
    dataframe = pd.read_csv(os.path.join(root_dir, "train.csv"), skiprows=1, names=[
                            "name", "x1", "x2", "y1", "y2", "classname"])
    deleted_files = 0
    file_list = dataframe[columns[0]].to_list()
    from shutil import copyfile
    image_dir = os.path.join(root_dir, "images")
    for file_name in os.listdir(image_dir):
        if file_name not in file_list:
            file_path = os.path.join(image_dir, file_name)
            copyfile(file_path, os.path.join(
                root_dir, "not-required", file_name))
            os.remove(file_path)
            deleted_files += 1
    print("Moved and deleted %s files" % deleted_files)


def removeDuplicateRows():
    import json
    root_dir = os.path.join("data", "new_dataset")
    dataframe = pd.read_csv(os.path.join(root_dir, "train.csv"), skiprows=1, names=[
                            "name", "x1", "x2", "y1", "y2", "classname"])

    duplicate = 0
    duplicate_rows = {}
    for row1_index in range(len(dataframe)):
        for row2_index in range(len(dataframe)):
            if row1_index == row2_index:
                continue

            if dataframe[columns[0]][row1_index] != dataframe[columns[0]][row2_index]:
                continue

            if dataframe[columns[0]][row1_index] not in duplicate_rows:
                duplicate_rows[dataframe[columns[0]][row1_index]] = []
                duplicate_rows[dataframe[columns[0]][row1_index]].append({
                    "index": row1_index,
                    "classname": dataframe[columns[5]][row1_index]
                })
            else:
                l = duplicate_rows[dataframe[columns[0]][row1_index]]
                indices = [i['index'] for i in l]
                if row2_index in indices:
                    continue

            duplicate_rows[dataframe[columns[0]][row1_index]].append({
                "index": row2_index,
                "classname": dataframe[columns[5]][row2_index]
            })
            duplicate += 1

    print("Duplicates: %s" % duplicate)

    with open(os.path.join(root_dir, 'class.json'), 'w') as fp:
        json.dump(duplicate_rows, fp,  indent=4)


def removeImagesWithSameClassname():
    import json
    root_dir = os.path.join("data", "new_dataset")
    class_data = None
    with open(os.path.join(root_dir, 'class_v1.json')) as f_in:
        class_data = json.load(f_in)

    ignore_images = []
    for image_d in class_data:
        p = None
        consider = True
        for img_r in class_data[image_d]:
            if p == None:
                p = img_r['classname']
                continue
            
            if img_r['classname'] != p:
                consider = False
                break
        
        if consider:
            ignore_images.append(image_d)
    
    for ignore_image in ignore_images:
        del class_data[ignore_image]
    
    with open(os.path.join(root_dir, 'class.json'), 'w') as fp:
        json.dump(class_data, fp,  indent=4)
    

# move_files_using_list(os.path.join(rootDir, 'preprocessed', 'face_with_mask'), os.path.join(rootDir, 'preprocessed', 'face_with_ff92_mask'), l)
# move_files_using_file(os.path.join(data_folder, 'images'), os.path.join(rootDir, 'preprocessed', 'face_with_ff92_mask'),'face_with_ff92_mask.txt')
# save_preselected_class_data(os.path.join(rootDir, 'preprocessed'), "data.csv")
# extract_same_class_files(os.path.join(data_folder, 'images'), os.path.join(rootDir, 'preprocessed'), "face_with_mask")

# preview_classes()

# create_dataset(os.path.join(data_folder, 'images'),
#     os.path.join(root_dir, 'preprocessed', 'images'),
#     {"path": os.path.join(root_dir, 'ffp2'), "classname": 'ffp2_mask'})


removeImagesWithSameClassname()
