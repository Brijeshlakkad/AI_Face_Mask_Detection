# https://sempioneer.com/python-for-seo/how-to-compress-images-in-python/

from PIL import Image
import os
from tqdm.notebook import tqdm

def make_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

root = "./data/preprocessed"
image_dir = os.path.join(root, "./original_images")
compressed_dir = os.path.join(root, 'images')
make_dir(compressed_dir)

# images = [file for file in os.listdir(image_dir) if file.endswith(('jpeg'))]

for image in tqdm(os.listdir(image_dir)):
    # 1. Open the image
    img = Image.open(os.path.join(image_dir, image))
    # 2. Compressing the image
    img.save(os.path.join(compressed_dir, image),
             optimize=True,
             quality=10)