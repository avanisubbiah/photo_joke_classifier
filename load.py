import torch
import torchvision
import torchvision.transforms as transforms
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ('photos', 'memes')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

class MemesPhotosDataset(Dataset):
  def __init__(self, path):
    self.data_path = path
    file_list = glob.glob(self.data_path + "*")
    print(file_list)
    self.data = []
    for class_path in file_list:
      class_name = class_path.split("/")[-1]
      for img_path in glob.glob(class_path + "/*.jpg"):
        self.data.append([img_path, class_name])
    print(self.data)
    self.class_map = {"photos" : 0, "memes" : 1}
    self.img_dim = (512, 512)

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    img_path, class_name = self.data[idx]
    img = cv2.imread(img_path)
    img = np.flip(img, 2)
    img = cv2.resize(img, self.img_dim)
    class_id = self.class_map[class_name]
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute(2, 0, 1)
    class_id = torch.tensor([class_id])
    # class_id = class_id.view(-1,)
    # print(class_id)
    return img_tensor.float(), class_id

dataset = MemesPhotosDataset("photos_memes_dataset/")
print(len(dataset))
from torch.utils.data import random_split
# train, test = random_split(dataset, [1119, 373])
trainloader = DataLoader(dataset, batch_size=4, shuffle=True)
testloader = DataLoader(dataset, batch_size=4, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*125*125, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

PATH = './photomeme_net.pth'
net = Net()
net.load_state_dict(torch.load(PATH))
net.to(device)

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = npimg.astype(np.uint8)
    # plt.imshow(npimg)
    # print(npimg.shape)
    # print(np.transpose(npimg, (1, 2, 0)).shape)
    # print(npimg)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(testloader)
images, labels = dataiter.next()
classes = ('photos', 'memes')
# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
correct = 0
total = 0

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
                correct += 1
            total_pred[classes[label]] += 1
            total += 1


# print accuracy for each class
print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))



