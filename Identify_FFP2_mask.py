#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms

data_folder = "./data/face-mask-detection-dataset/images"
root_folder = "./data"


# In[12]:


data_files = [
    {"file": '1804.jpg', "label": 0},
    {"file": '1828.jpg', "label": 0},
    {"file": '1850.jpg', "label": 0},
    {"file": '1874.jpg', "label": 0},
    {"file": '2042.jpg', "label": 0},
    {"file": '1974.jpg', "label": 0},
    {"file": '1980.jpg', "label": 0},
    {"file": '1989.jpg', "label": 0},
    {"file": '2000.jpg', "label": 0},
    {"file": '2010.jpg', "label": 0},
    {"file": '2015.jpg', "label": 0},
    {"file": '2021.jpg', "label": 0},
    {"file": '2066.jpg', "label": 0},
    {"file": '2081.jpg', "label": 0},
    {"file": '2109.jpg', "label": 0},
    {"file": '2113.jpg', "label": 0},
    {"file": '2117.jpg', "label": 0},
    {"file": '2133.jpg', "label": 0},
    {"file": '2141.jpg', "label": 0},
    {"file": '2159.png', "label": 0},
    {"file": '2188.png', "label": 0},
    {"file": '2190.png', "label": 0},
    {"file": '2189.png', "label": 0},
    {"file": '2213.png', "label": 0},
    {"file": '2254.png', "label": 0},
    {"file": '2263.png', "label": 0},
    {"file": '2288.png', "label": 0},
    {"file": '2323.png', "label": 0},
    {"file": '2325.png', "label": 0},
    {"file": '2360.png', "label": 0},
    {"file": '2366.png', "label": 0},
    {"file": '2407.png', "label": 0},
    {"file": '2472.png', "label": 0},
    {"file": '2500.png', "label": 0},
    {"file": '2501.png', "label": 0},
    {"file": '2537.png', "label": 0},
    {"file": '2551.png', "label": 0},
    {"file": '2578.png', "label": 0},
    {"file": '2584.png', "label": 0},
    {"file": '2593.png', "label": 0},
    {"file": '2613.png', "label": 0},
    {"file": '2610.png', "label": 0},
    {"file": '2635.png', "label": 0},
    {"file": '2639.png', "label": 0},
    {"file": '2668.png', "label": 0},
    {"file": '2670.png', "label": 0},
    {"file": '2686.png', "label": 0},
    {"file": '2688.png', "label": 0},
    {"file": '2705.png', "label": 0},
    {"file": '2733.png', "label": 0},
    {"file": '2751.png', "label": 0},
    {"file": '2752.png', "label": 0},
    {"file": '2763.png', "label": 0},
    {"file": '2762.png', "label": 0},
    {"file": '2769.png', "label": 0},
    {"file": '2829.png', "label": 0},
    {"file": '2843.png', "label": 0},
    {"file": '2842.png', "label": 0},
    {"file": '2881.png', "label": 0},
    {"file": '2894.png', "label": 0},
    {"file": '2898.png', "label": 0},
    {"file": '2906.png', "label": 0},
    {"file": '2908.png', "label": 0},
    {"file": '2947.png', "label": 0},
    {"file": '2980.png', "label": 0},
    {"file": '2983.png', "label": 0},
    {"file": "3851.png", "label": 0},
    {"file": "3871.png", "label": 0},
    {"file": "3909.png", "label": 0},
    {"file": "3931.png", "label": 0},
    {"file": "3985.png", "label": 0},
    {"file": "3994.png", "label": 0},
    {"file": "4014.png", "label": 0},
    {"file": "4051.png", "label": 0},
    {"file": "4050.png", "label": 0},
    {"file": "4065.png", "label": 0},
    {"file": "4081.png", "label": 0},
    {"file": "4082.png", "label": 0},
    {"file": "4101.png", "label": 0},
    {"file": "4163.png", "label": 0},
    {"file": "4178.png", "label": 0},
    {"file": "4193.png", "label": 0},
    {"file": "4209.png", "label": 0},
    {"file": "4211.png", "label": 0},
    {"file": "4247.png", "label": 0},
    {"file": "4261.png", "label": 0},
    {"file": "4274.png", "label": 0},
    {"file": "4299.png", "label": 0},
    {"file": "4337.png", "label": 0},
    {"file": "4338.png", "label": 0},
    {"file": "4381.png", "label": 0},
    {"file": "4399.png", "label": 0},
    {"file": "4407.png", "label": 0},
    {"file": "4413.png", "label": 0},
    {"file": "4428.png", "label": 0},
    {"file": "4452.png", "label": 0},
    {"file": "4480.png", "label": 0},
    {"file": "4482.png", "label": 0},
    {"file": "4501.png", "label": 0},
    {"file": "4523.png", "label": 0},
    {"file": "4607.png", "label": 0},
    {"file": "4614.png", "label": 0},
    {"file": "4696.png", "label": 0},
    {"file": "4829.png", "label": 0},
    {"file": "4842.png", "label": 0},
    {"file": "4846.png", "label": 0},
    {"file": "4935.png", "label": 0},
    {"file": "4937.png", "label": 0},
    {"file": "4964.png", "label": 0},
    {"file": "4975.png", "label": 0},
    {"file": "4995.jpg", "label": 0},
    {"file": "4997.jpg", "label": 0},
    {"file": "5028.jpg", "label": 0},
    {"file": "5037.jpg", "label": 0},
    {"file": "5077.jpg", "label": 0},
    {"file": "5110.jpg", "label": 0},
    {"file": "5194.jpg", "label": 0},
    {"file": "5218.jpg", "label": 0},
    {"file": "5224.jpg", "label": 0},
    {"file": "5251.jpg", "label": 0},
    {"file": "5369.jpeg", "label": 0},
    {"file": "5582.jpg", "label": 0},

    {"file": '1802.jpg', "label": 1},
    {"file": '1805.jpg', "label": 1},
    {"file": '1806.jpg', "label": 1},
    {"file": '1810.jpg', "label": 1},
    {"file": '1812.jpg', "label": 1},
    {"file": '1818.jpg', "label": 1},
    {"file": '1821.jpg', "label": 1},
    {"file": '1822.jpg', "label": 1},
    {"file": '1823.jpg', "label": 1},
    {"file": '1825.jpg', "label": 1},
    {"file": '1826.jpg', "label": 1},
    {"file": '1827.jpg', "label": 1},
    {"file": '1829.jpg', "label": 1},
    {"file": '1833.jpg', "label": 1},
    {"file": '1835.jpg', "label": 1},
    {"file": '1836.jpg', "label": 1},
    {"file": '1837.jpg', "label": 1},
    {"file": '1838.jpg', "label": 1},
    {"file": '1839.jpg', "label": 1},
    {"file": '1840.jpg', "label": 1},
    {"file": '1841.jpg', "label": 1},
    {"file": '1843.jpg', "label": 1},
    {"file": '1844.jpg', "label": 1},
    {"file": '1845.jpg', "label": 1},
    {"file": '1847.jpg', "label": 1},
    {"file": '1848.jpg', "label": 1},
    {"file": '1849.jpg', "label": 1},
    {"file": '1851.jpg', "label": 1},
    {"file": '1852.jpg', "label": 1},
    {"file": '1854.jpg', "label": 1},
    {"file": '1858.jpg', "label": 1},
    {"file": '1864.jpg', "label": 1},
    {"file": '1867.jpg', "label": 1},
    {"file": '1870.jpg', "label": 1},
    {"file": '1873.png', "label": 1},
    {"file": '1880.jpg', "label": 1},
    {"file": '1881.jpg', "label": 1},
    {"file": '1882.jpg', "label": 1},
    {"file": '1885.jpg', "label": 1},
    {"file": '1886.jpg', "label": 1},
    {"file": '1873.png', 'label': 1},
    {"file": '1900.png', 'label': 1},
{"file": '1915.png', 'label': 1},
{"file": '1961.png', 'label': 1},
{"file": '1962.png', 'label': 1},
{"file": '1963.png', 'label': 1},
{"file": '2153.png', 'label': 1},
{"file": '2154.png', 'label': 1},
{"file": '2155.png', 'label': 1},
{"file": '2158.png', 'label': 1},
{"file": '2159.png', 'label': 1},
{"file": '2162.png', 'label': 1},
{"file": '2163.png', 'label': 1},
{"file": '2165.png', 'label': 1},
{"file": '2166.png', 'label': 1},
{"file": '2167.png', 'label': 1},
{"file": '2168.png', 'label': 1},
{"file": '2169.png', 'label': 1},
{"file": '2170.png', 'label': 1},
{"file": '2171.png', 'label': 1},
{"file": '2172.png', 'label': 1},
{"file": '2173.png', 'label': 1},
{"file": '2174.png', 'label': 1},
{"file": '2175.png', 'label': 1},
{"file": '2176.png', 'label': 1},
{"file": '2177.png', 'label': 1},
{"file": '2178.png', 'label': 1},
{"file": '2179.png', 'label': 1},
{"file": '2180.png', 'label': 1},
{"file": '2181.png', 'label': 1},
{"file": '2182.png', 'label': 1},
{"file": '2183.png', 'label': 1},
{"file": '2184.png', 'label': 1},
{"file": '2185.png', 'label': 1},
{"file": '2187.png', 'label': 1},
{"file": '2188.png', 'label': 1},
{"file": '2189.png', 'label': 1},
{"file": '2190.png', 'label': 1},
{"file": '2191.png', 'label': 1},
{"file": '2192.png', 'label': 1},
{"file": '2193.png', 'label': 1},
{"file": '2194.png', 'label': 1},
{"file": '2195.png', 'label': 1},
{"file": '2202.png', 'label': 1},
{"file": '2203.png', 'label': 1},
{"file": '2204.png', 'label': 1},
{"file": '2205.png', 'label': 1},
{"file": '2213.png', 'label': 1},
{"file": '2214.png', 'label': 1},
{"file": '2215.png', 'label': 1},
{"file": '2216.png', 'label': 1},
{"file": '2219.png', 'label': 1},
{"file": '2220.png', 'label': 1},
{"file": '2221.png', 'label': 1},
{"file": '2222.png', 'label': 1},
{"file": '2223.png', 'label': 1},
{"file": '2225.png', 'label': 1},
{"file": '2226.png', 'label': 1},
{"file": '2227.png', 'label': 1},
{"file": '2228.png', 'label': 1},
{"file": '2229.png', 'label': 1},
{"file": '2230.png', 'label': 1},
{"file": '2233.png', 'label': 1},
{"file": '2234.png', 'label': 1},
{"file": '2235.png', 'label': 1},
{"file": '2236.png', 'label': 1},
{"file": '2244.png', 'label': 1},
{"file": '2245.png', 'label': 1},
{"file": '2251.png', 'label': 1},
{"file": '2252.png', 'label': 1},
{"file": '2254.png', 'label': 1},
{"file": '2255.png', 'label': 1},
{"file": '2257.png', 'label': 1},
{"file": '2258.png', 'label': 1},
{"file": '2259.png', 'label': 1},
{"file": '2260.png', 'label': 1},
{"file": '2261.png', 'label': 1},
{"file": '2262.png', 'label': 1},
{"file": '2263.png', 'label': 1},
{"file": '2264.png', 'label': 1},
{"file": '2265.png', 'label': 1},
{"file": '2266.png', 'label': 1},
{"file": '2267.png', 'label': 1},
{"file": '2268.png', 'label': 1},
{"file": '2269.png', 'label': 1},
{"file": '2271.png', 'label': 1},
{"file": '2275.png', 'label': 1},
{"file": '2276.png', 'label': 1},
{"file": '2277.png', 'label': 1},
{"file": '2279.png', 'label': 1},
{"file": '2281.png', 'label': 1},
{"file": '2282.png', 'label': 1},
{"file": '2283.png', 'label': 1},
{"file": '2284.png', 'label': 1},
{"file": '2285.png', 'label': 1},
{"file": '2286.png', 'label': 1},
{"file": '2288.png', 'label': 1},
{"file": '2289.png', 'label': 1},
{"file": '2290.png', 'label': 1},
{"file": '2293.png', 'label': 1},
{"file": '2294.png', 'label': 1},
{"file": '2295.png', 'label': 1},
{"file": '2296.png', 'label': 1},
{"file": '2300.png', 'label': 1},
{"file": '2309.png', 'label': 1},
{"file": '2314.png', 'label': 1},
{"file": '2315.png', 'label': 1},
{"file": '2316.png', 'label': 1},
{"file": '2317.png', 'label': 1},
{"file": '2320.png', 'label': 1},
{"file": '2321.png', 'label': 1},
{"file": '2323.png', 'label': 1},
{"file": '2324.png', 'label': 1},
{"file": '2325.png', 'label': 1},
{"file": '2326.png', 'label': 1},
{"file": '2327.png', 'label': 1},
{"file": '2328.png', 'label': 1},
{"file": '2329.png', 'label': 1},
{"file": '2331.png', 'label': 1},
{"file": '2333.png', 'label': 1},
{"file": '2334.png', 'label': 1},
{"file": '2335.png', 'label': 1},
{"file": '2336.png', 'label': 1},
{"file": '2337.png', 'label': 1},
{"file": '2339.png', 'label': 1},
{"file": '2340.png', 'label': 1},
{"file": '2341.png', 'label': 1},
{"file": '2342.png', 'label': 1},
{"file": '2344.png', 'label': 1},
{"file": '2345.png', 'label': 1},
{"file": '2350.png', 'label': 1},
{"file": '2351.png', 'label': 1},
{"file": '2352.png', 'label': 1},
{"file": '2353.png', 'label': 1},
{"file": '2355.png', 'label': 1},
{"file": '2356.png', 'label': 1},
{"file": '2357.png', 'label': 1},
{"file": '2359.png', 'label': 1},
{"file": '2360.png', 'label': 1},
{"file": '2361.png', 'label': 1},
{"file": '2362.png', 'label': 1},
{"file": '2363.png', 'label': 1},
{"file": '2365.png', 'label': 1},
{"file": '2366.png', 'label': 1},
{"file": '2367.png', 'label': 1},
{"file": '2368.png', 'label': 1},
{"file": '2371.png', 'label': 1},
{"file": '2372.png', 'label': 1},
{"file": '2373.png', 'label': 1},
{"file": '2375.png', 'label': 1},
{"file": '2381.png', 'label': 1},
{"file": '2382.png', 'label': 1},
{"file": '2385.png', 'label': 1},
{"file": '2386.png', 'label': 1},
{"file": '2387.png', 'label': 1},
{"file": '2388.png', 'label': 1},
{"file": '2389.png', 'label': 1},
{"file": '2394.png', 'label': 1},
{"file": '2395.png', 'label': 1},
{"file": '2397.png', 'label': 1},
{"file": '2398.png', 'label': 1},
{"file": '2399.png', 'label': 1},
{"file": '2400.png', 'label': 1},
{"file": '2401.png', 'label': 1},
{"file": '2403.png', 'label': 1},
{"file": '2405.png', 'label': 1},
{"file": '2406.png', 'label': 1}
]


# In[13]:




def getImageData(folder, image_file):
    # Load the image
    return Image.open(os.path.join(folder, image_file)).convert('RGB')


# In[14]:


class FaceMaskDataset(Dataset):
    dataset = []
    conversion = None

    def __init__(self, indexes, conversion=transforms.ToTensor()):
        self.conversion = conversion
        for rowIndex in indexes:
            sample = {}
            sample['image'] = getImageData(data_folder, data_files[rowIndex]["file"])
            sample['target'] = data_files[rowIndex]["label"]
            self.dataset.append(sample)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index]['image']
        if self.conversion is not None:
            image = self.conversion(image)
        return image, self.dataset[index]['target']


# In[15]:


train_split_percentage = 0.8
val_split_percentage = 0.1
test_split_percentage = 0.1
size_of_the_dataset = len(data_files)

batch_size = 5
learning_rate = 0.001
num_epochs = 15

x_direction = 150
y_direction = 150

indexes = list(range(size_of_the_dataset))
random.shuffle(indexes)


train_indexes = indexes[:int(train_split_percentage*len(indexes))]
val_indexes = indexes[int(train_split_percentage*len(indexes))                      :int((train_split_percentage + val_split_percentage)*len(indexes))]
test_indexes = indexes[int(
    (train_split_percentage + val_split_percentage)*len(indexes)):]


print(f"Effective train split = {len(train_indexes)/len(indexes)*100}%")
print(f"Effective val split = {len(val_indexes)/len(indexes)*100}%")
print(f"Effective test split = {len(test_indexes)/len(indexes)*100}%")


# In[16]:



transform = transforms.Compose(
    [transforms.Resize((x_direction, y_direction)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


print("Loading training set")
train_dataset = FaceMaskDataset(train_indexes, conversion=transform)
print("Loading validation set")
val_dataset = FaceMaskDataset(val_indexes, conversion=transform)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# In[17]:




class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        new_x_direction = x_direction+50
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=x_direction,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(x_direction),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=x_direction,
                      out_channels=x_direction, kernel_size=3, padding=1),
            nn.BatchNorm2d(x_direction),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.Conv2d(in_channels=x_direction,
                      out_channels=new_x_direction, kernel_size=3, padding=1),
            nn.BatchNorm2d(new_x_direction),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=new_x_direction,
                      out_channels=new_x_direction, kernel_size=3, padding=1),
            nn.BatchNorm2d(new_x_direction),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        num_of_classes = 2
        self.fc_layer = nn.Sequential(nn.Dropout(p=0.1), nn.Linear(37 * 37 * new_x_direction, 1000), nn.ReLU(
            inplace=True), nn.Linear(1000, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.1), nn.Linear(512, num_of_classes))

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
#         print(x.shape)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)
        return x


# In[18]:


model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[19]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = model.to(device)


# In[ ]:




total_step = len(train_loader)
loss_list = []
acc_list = []

from tqdm.notebook import tqdm

for epoch in range(num_epochs):
    for (images, labels) in tqdm(train_loader):  # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Train accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch +
          1, num_epochs, loss.item(), (correct / total) * 100))
    torch.save(model.state_dict(), os.path.join(
        root_folder, 'model', 'epoch' + str(epoch) + '.pt'))


# In[11]:



test_dataset = FaceMaskDataset(test_indexes, conversion=transform)
print("Loading test set")
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model on the {} test images: {} %' .format(len(test_indexes), (correct / total) * 100))
    


# In[12]:


from shutil import copyfile

working_folder = "./data/preprocessed/face_with_mask"
output_folder = './data/preprocessed/face_with_ff92_mask'

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

files = os.listdir(working_folder)

count=0
batch = []
current_files =[]
for i, file in enumerate(files):
    if count == 2:
        batch = torch.Tensor(batch)
        labels = model(batch)
        _, predicted = torch.max(labels.data, 1)
        print("Prediction: ", predicted)
        for i, prediction in enumerate(predicted):
            if prediction == 0:
                # copy this file
                copyfile(os.path.join(working_folder, current_files[i]), os.path.join(output_folder, current_files[i]))
                
        current_files = []
        batch = []
        count = 0
    if not file.endswith('png'):
        continue
    image_data = transform(getImageData(working_folder, file))
    image_data = image_data.numpy()
    batch.append(image_data)
    current_files.append(file)
    count+=1


# In[ ]:




