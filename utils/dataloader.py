import os
import csv
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import random
import matplotlib.image as mpimg
from PIL import Image

class ISLES2015SISSDataset(Dataset):
    def __init__(self, data_root, test=False):
        self.data_root = data_root
        self.test = test
        self.files = os.listdir(data_root)  # list containing names of all files in data_root directory
        self.images = self.files  # list containing names strictly the image files
        self.lists = []

        file_path = '/dbfs/tcdh-isles/lists_export.csv'  # path to csv containing lists of image file names
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.lists.append(row)
                
        self.DWI_Input = self.lists[0]  # get list of DWI images in ISLES 2015 SISS
        self.expert = self.lists[4]  # get list of corresponding labelled images
    
    def __len__(self):
        return len(self.DWI_Input)  # return length of dataset

    def __getitem__(self, idx):
        DWI_image_name = self.DWI_Input[idx]
        expert_mask_name = self.expert[idx]

        DWI_image_path = os.path.join(self.data_root, DWI_image_name)  # construct full paths to DWI image
        expert_mask_path = os.path.join(self.data_root, expert_mask_name)  # construct full paths to expert masks

        DWI_image = Image.open(DWI_image_path)
        expert_mask = Image.open(expert_mask_path)

        transform = transforms.Compose([
            transforms.ToTensor()])  # define pipeline which converts PIL Image or np array to PyTorch tensor

        DWI_image_tensor = transform(DWI_image)
        expert_mask_tensor = transform(expert_mask)
        
        return DWI_image_tensor, expert_mask_tensor  # return pairs of DWI images and their corresponding expert masks

    def get_img_size(self):
        first_DWI_image = self.DWI_Input[0]
        img = mpimg.imread(first_DWI_image)
        H, W, _ = img.shape
        return H, W  # return dimensions of input images

if __name__ == "__main__":
    a = np.array([1,2])
    print(a.dtype)
