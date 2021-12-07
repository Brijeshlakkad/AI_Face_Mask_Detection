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
                       skiprows=1, names=["filename", "classname", "gender", "age"])


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
        "face_with_cloth_mask": 0,
        "face_no_mask": 0,
        "face_with_ffp2_mask": 0,
        "face_with_surgical_mask": 0
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


def previewClassJSON():
    import json
    root_dir = os.path.join("data", "new_dataset")
    class_data = None
    with open(os.path.join(root_dir, 'class.json')) as f_in:
        class_data = json.load(f_in)

    print(len(class_data.keys()))


def moveImagesWithMultipleFaceMask():
    import json
    root_dir = os.path.join("data", "new_dataset")
    class_data = None
    with open(os.path.join(root_dir, 'class.json')) as f_in:
        class_data = json.load(f_in)

    from shutil import copyfile
    deleted_files = 0
    image_dir = os.path.join(root_dir, "images")
    for image_d in class_data:
        file_path = os.path.join(image_dir, image_d)
        copyfile(file_path, os.path.join(
            root_dir, "face-with-multiple-mask", image_d))
        os.remove(file_path)
        deleted_files += 1
    print("Moved and deleted %s files" % deleted_files)


def createCSVFile():
    new_dataset_dir = os.path.join("data", "new_dataset")
    new_image_dir = os.path.join(new_dataset_dir, "images")
    new_dataset_df = pd.read_csv(os.path.join(
        new_dataset_dir, "train.csv"), skiprows=1, names=columns)
    file_list = os.listdir(new_image_dir)
    delete_rows = []
    for row_index in range(len(new_dataset_df)):
        if new_dataset_df[columns[0]][row_index] not in file_list:
            delete_rows.append(row_index)
    new_dataset_df.drop(new_dataset_df.index[delete_rows], inplace=True)
    new_dataset_df.to_csv(os.path.join(new_dataset_dir, "data.csv"))

def deleteDuplicateRows():
    new_dataset_dir = os.path.join("data", "new_dataset")
    new_dataset_df = pd.read_csv(os.path.join(
        new_dataset_dir, "data.csv"), skiprows=1, names=columns)

    delete_rows = []
    file_list = []
    for row_index in range(len(new_dataset_df)):
        if new_dataset_df[columns[0]][row_index] not in file_list:
            file_list.append(new_dataset_df[columns[0]][row_index])
        else:
            delete_rows.append(row_index)
            
    new_dataset_df.drop(new_dataset_df.index[delete_rows], inplace=True)
    new_dataset_df.to_csv(os.path.join(new_dataset_dir, "new_data.csv"))


def create_dataset_v2():
    new_columns = ["filename", "classname"]
    dest_dataset_dir = os.path.join("data", "preprocessed")
    dest_image_dir = os.path.join(dest_dataset_dir, "images")
    make_dir(dest_dataset_dir)
    make_dir(dest_image_dir)

    new_dataset_dir = os.path.join("data", "new_dataset")
    new_dataset_df = pd.read_csv(os.path.join(
        new_dataset_dir, "new_data.csv"), skiprows=1, names=columns)
    new_image_dir = os.path.join(new_dataset_dir, "images")

    dataset_dir = os.path.join("data", "dataset")
    dataset_df = pd.read_csv(os.path.join(
        dataset_dir, "data.csv"), skiprows=1, names=new_columns)
    image_dir = os.path.join(dataset_dir, "images")

    from shutil import copyfile

    seed = 1000
    file_index = seed

    # move files
    for row_index in range(len(dataset_df)):
        file_name = dataset_df[new_columns[0]][row_index]
        file_index += 1
        copyfile(os.path.join(image_dir, file_name),
                 os.path.join(dest_image_dir, file_name))

    for row_index in range(len(new_dataset_df)):
        file_name = new_dataset_df[columns[0]][row_index]
        file_index += 1
        new_file_name = str(file_index)+'.'+file_name.split(".")[-1]
        copyfile(os.path.join(new_image_dir, file_name),
                 os.path.join(dest_image_dir, new_file_name))
        new_row = {}
        new_row[new_columns[0]] = new_file_name
        new_row[new_columns[1]] = new_dataset_df[columns[5]][row_index]
        dataset_df = dataset_df.append(new_row, ignore_index = True)

    dataset_df.replace("mask_colorful", "face_with_cloth_mask", inplace=True)
    dataset_df.replace("mask_surgical", "face_with_surgical_mask", inplace=True)
    dataset_df.replace("ffp2_mask", "face_with_ffp2_mask", inplace=True)
    dataset_df.to_csv(os.path.join(dest_dataset_dir, 'data.csv'))

def exctraFilesFromSubDir():
    from shutil import copyfile
    dest_dir = os.path.join("data", "new_new_ffp2")
    make_dir(dest_dir)
    source_dir = os.path.join('data', 'ffp2_folder')
    make_dir(source_dir)
    file_index = 1000
    for folder in os.listdir(source_dir):
        if folder == '.DS_Store':
            continue
        s_dir = os.path.join(source_dir, folder)
        for file_name in os.listdir(s_dir):
            new_file_name = str(file_index)+'.'+file_name.split(".")[-1]
            copyfile(os.path.join(s_dir, file_name), os.path.join(dest_dir, new_file_name))
            file_index+=1


def deleteFFP2Data():
    from shutil import copyfile
    new_columns = ["filename", "classname"]
    dest_dataset_dir = os.path.join("data", "preprocessed")
    make_dir(dest_dataset_dir)
    dataset_dir = os.path.join("data", "before_removing_ffp2")
    image_dir = os.path.join(dataset_dir, "images")
    ffp2_dir = os.path.join(dataset_dir, 'removed_ffp2')
    make_dir(ffp2_dir)

    dataset_df = pd.read_csv(os.path.join(
        dataset_dir, "data.csv"), skiprows=1, names=new_columns)

    ffp2_rows = []
    for row_index in range(len(dataset_df)):
        if dataset_df[new_columns[1]][row_index] == 'face_with_ffp2_mask':
            file_name = dataset_df[new_columns[0]][row_index]
            file_path = os.path.join(image_dir, file_name)
            copyfile(file_path, os.path.join(ffp2_dir, file_name))
            os.remove(file_path)
            ffp2_rows.append(row_index)
    
    dataset_df.drop(dataset_df.index[ffp2_rows], inplace=True)
    dataset_df.to_csv(os.path.join(dest_dataset_dir, 'data.csv'))

def create_dataset_v3():
    import csv
    from shutil import copyfile

    new_columns = ["filename", "classname"]
    dest_dataset_dir = os.path.join("data", "preprocessed")
    dest_image_dir = os.path.join(dest_dataset_dir, "images")
    make_dir(dest_image_dir)

    dataset_dir = os.path.join("data", "before_adding_ffp2")
    image_dir = os.path.join(dataset_dir, "images")
    dataset_df = pd.read_csv(os.path.join(dataset_dir, "data.csv"), skiprows=1, names=new_columns)
    
    ffp2_dir = os.path.join("data", 'ffp2')

    seed = 1000
    file_index = seed

    csv_data = []
    for row_index in range(len(dataset_df)):
        file_index+=1
        file_name = dataset_df[new_columns[0]][row_index]
        new_file_name = str(file_index)+'.'+file_name.split(".")[-1]

        file_path = os.path.join(image_dir, file_name)
        copyfile(file_path, os.path.join(dest_image_dir, new_file_name))

        new_row = {}
        new_row[new_columns[0]] = new_file_name
        new_row[new_columns[1]] = dataset_df[new_columns[1]][row_index]
        csv_data.append(new_row)
    
    for file_name in os.listdir(ffp2_dir):
        if not is_file(file_name):
            continue
        file_index += 1
        new_file_name = str(file_index)+'.'+file_name.split(".")[-1]
        copyfile(os.path.join(ffp2_dir, file_name),
                    os.path.join(dest_image_dir, new_file_name))
        new_row = {}
        new_row[new_columns[0]] = new_file_name
        new_row[new_columns[1]] = 'face_with_ffp2_mask'
        csv_data.append(new_row)
    
    with open(os.path.join(dest_dataset_dir, "data.csv"), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=new_columns)
        writer.writeheader()
        writer.writerows(csv_data)

    print("File created: ", (file_index-seed))

def move_class_data(num, class_name):
    from shutil import copyfile
    new_columns = ["filename", "classname"]
    dest_dataset_dir = os.path.join("data", "preprocessed")
    make_dir(dest_dataset_dir)
    
    dataset_dir = os.path.join("data", "before_removing_class")
    image_dir = os.path.join(dataset_dir, "images")
    dataset_df = pd.read_csv(os.path.join(dataset_dir, "data.csv"), skiprows=1, names=new_columns)
    other_image_dir = os.path.join(dataset_dir, class_name)
    make_dir(other_image_dir)

    other_class_rows = []
    count = 0
    for row_index in range(len(dataset_df)):
        if dataset_df[new_columns[1]][row_index] == class_name:
            file_name = dataset_df[new_columns[0]][row_index]
            file_path = os.path.join(image_dir, file_name)
            copyfile(file_path, os.path.join(other_image_dir, file_name))
            os.remove(file_path)
            other_class_rows.append(row_index)
            count+=1
        
        if count >= num:
            break

    dataset_df.drop(dataset_df.index[other_class_rows], inplace=True)
    dataset_df.to_csv(os.path.join(dest_dataset_dir, 'data.csv'))

def find_file():
    from shutil import copyfile
    l = [
    ]
    dataset_dir = os.path.join("data", "preprocessed")
    image_dir = os.path.join(dataset_dir, "images")
    dest_image_dir = os.path.join(dataset_dir, "crop")
    make_dir(dest_image_dir)

    for file_name in os.listdir(image_dir):
        x_file_name = file_name.split(".")[0]
        if x_file_name in l:
            copyfile(os.path.join(image_dir, file_name), os.path.join(dest_image_dir, file_name))


def create_dataset_v4(class_name, class_dir_name):
    import csv
    from shutil import copyfile

    new_columns = ["filename", "classname", "gender", "age"]
    dest_dataset_dir = os.path.join("data", "preprocessed")
    dest_image_dir = os.path.join(dest_dataset_dir, "images")
    make_dir(dest_image_dir)

    dataset_dir = os.path.join("data", "before_processing")
    image_dir = os.path.join(dataset_dir, "images")
    dataset_df = pd.read_csv(os.path.join(dataset_dir, "data.csv"), skiprows=1, names=new_columns)

    class_dir = os.path.join(dataset_dir, class_dir_name)
    make_dir(class_dir)

    seed = 1000
    file_index = seed

    csv_data = []
    for row_index in range(len(dataset_df)):
        file_index+=1
        file_name = dataset_df[new_columns[0]][row_index]
        new_file_name = str(file_index)+'.'+file_name.split(".")[-1]

        file_path = os.path.join(image_dir, file_name)
        copyfile(file_path, os.path.join(dest_image_dir, new_file_name))

        new_row = {}
        new_row[new_columns[0]] = new_file_name
        new_row[new_columns[1]] = dataset_df[new_columns[1]][row_index]
        new_row[new_columns[2]] = dataset_df[new_columns[2]][row_index]
        new_row[new_columns[3]] = dataset_df[new_columns[3]][row_index]
        csv_data.append(new_row)
    
    for file_name in os.listdir(class_dir):
        if not is_file(file_name):
            continue
        file_index += 1
        new_file_name = str(file_index)+'.'+file_name.split(".")[-1]
        copyfile(os.path.join(class_dir, file_name),
                    os.path.join(dest_image_dir, new_file_name))
        new_row = {}
        new_row[new_columns[0]] = new_file_name
        new_row[new_columns[1]] = class_name
        new_row[new_columns[2]] = ''
        new_row[new_columns[3]] = ''
        csv_data.append(new_row)
    
    with open(os.path.join(dest_dataset_dir, "data.csv"), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=new_columns)
        writer.writeheader()
        writer.writerows(csv_data)

    print("File created: ", (file_index-seed))


def deleteRows():
    """Removes images that doesn't belong to new_dataset/train.csv"""
    dataset_dir = os.path.join("data", "preprocessed")
    new_columns = ["filename", "classname", "gender", "age"]
    dataset_df = pd.read_csv(os.path.join(dataset_dir, "before.csv"), skiprows=1, names=new_columns)
    image_dir = os.path.join(dataset_dir, "images")                        
    deleted_files =[]
    file_list = os.listdir(image_dir)
    for row_index in range(len(dataset_df[new_columns[0]])):
        if dataset_df[new_columns[0]][row_index] not in file_list:
            deleted_files.append(row_index)

    dataset_df.drop(dataset_df.index[deleted_files], inplace=True)
    dataset_df.to_csv(os.path.join(dataset_dir, 'data.csv'))
    print("Deleted files: ", len(deleted_files))


# create_dataset_v4('face_with_surgical_mask', 'surgical')

# move_class_data(663, 'face_no_mask')
# move_files_using_list(os.path.join(rootDir, 'preprocessed', 'face_with_mask'), os.path.join(rootDir, 'preprocessed', 'face_with_ff92_mask'), l)
# move_files_using_file(os.path.join(data_folder, 'images'), os.path.join(rootDir, 'preprocessed', 'face_with_ff92_mask'),'face_with_ff92_mask.txt')
# save_preselected_class_data(os.path.join(rootDir, 'preprocessed'), "data.csv")
# extract_same_class_files(os.path.join(data_folder, 'images'), os.path.join(rootDir, 'preprocessed'), "face_with_mask")
# create_dataset(os.path.join(data_folder, 'images'),
#     os.path.join(root_dir, 'preprocessed', 'images'),
#     {"path": os.path.join(root_dir, 'ffp2'), "classname": 'ffp2_mask'})


# preview_classes()
