import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
from net.att_unet import AttUNet
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tag = "dice7_bce3_resnet_att_16"
model_path = "/dbfs/tcdh-isles/net/Model_Params_att_16.pth"  # use saved network with attention
#model_path = ""  # use saved network without attention
data_root = "/dbfs/tcdh-isles/All_Images_Test/Normalized/"  # specify directory hosting normalized test data subset
save_root = "/dbfs/tcdh-isles/output/{}/".format(tag)  # directory where output images will be saved
log_root = os.path.join(save_root, "logs")  # directory where logs will be saved
net_type = "ResNet50"  # specify type of network being used
out_image = True  # ask to output images

if not os.path.exists(model_path):
    raise ValueError("Model not found")  # check if specified model exists

if not os.path.exists(log_root):
    os.makedirs(log_root)  # specify if log directory exists

net = AttUNet(net_type, is_train=False).to(device)  # set to evaluation mode
net.load_state_dict(torch.load(model_path))  # load parameters of trained model (with or without attention)
print("{} loaded".format(model_path))

data_transform = transforms.Compose([
    transforms.ToTensor(),
])  # configure data transformation pipeline which transforms input images into tensors

#image_files = [os.path.join(data_root, file) for file in os.listdir(data_root) if file.endswith(('.jpg', '.png', '.jpeg'))]  # all modalities
image_files = [os.path.join(data_root, file) for file in os.listdir(data_root) if file.endswith(('.jpg', '.png', '.jpeg')) and "DWI" in file]  # DWI only

for idx, image_path in enumerate(image_files):  # 'idx' being the index of the test image file & 'image_path' the path to it
    img = Image.open(image_path)  # open image
    img_tensor = data_transform(img).unsqueeze(0).to(device)  # transform image to tensor

    with torch.no_grad():  # disable gradient calculation during inference to reduce memory consumption
        output = net(img_tensor)  # execute trained network on test image tensor
        output = output.repeat(1, 3, 1, 1)  # repeat output image tensor along channel dimension 3 times (for compatibility)
        onehot_output = (output > 0.68).float()  # define pixel threshold (trial and error)

    for i in range(onehot_output.shape[1]):
        output_np = onehot_output[0, i, :, :].detach().cpu().numpy()
        output_filename = "output_{}_DWI.jpg".format(os.path.basename(image_path))  # generate filename for output image (DWI analysis)

        if not os.path.exists(save_root):
            os.makedirs(save_root)

        plt.imsave(os.path.join(save_root, output_filename), output_np, cmap='gray')  # save output np array as image file in save_root
