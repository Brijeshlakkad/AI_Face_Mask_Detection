import os
from shutil import copyfile

def make_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

root = "./data/preprocessed/"
dir_name_1 = "face_with_ff92_mask"
dir_name_2 = "face_with_mask"

dir1_files = os.listdir(os.path.join(root, dir_name_1))
dir2_files = os.listdir(os.path.join(root, dir_name_2))

output = "output"
output = os.path.join(root, output)
make_dir(output)

for file in dir2_files:
    if file not in dir1_files:
        copyfile(os.path.join(root,dir_name_2, file), os.path.join(output, file))
        print(f'{file} copied')