#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd

root_dir = './data'
data_folder = "./data/face-mask-detection-dataset"


# In[2]:


columns = ["filename", "classname"]
def get_preprocessed_df():
    return pd.read_csv(os.path.join(root_dir, "preprocessed", "data.csv"),
                        skiprows=1, names=columns)


# In[3]:


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
        className = train_df[columns[1]][row_index]
        if className not in classes:
            classes.append(className)
        if className in preselected_class_count.keys():
            preselected_class_count[className] += 1
    print(preselected_class_count)


# In[4]:


preview_classes()


# In[5]:


train_df = get_preprocessed_df()

train_df.head()

