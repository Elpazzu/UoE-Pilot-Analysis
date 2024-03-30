# Databricks notebook source
# MAGIC %md
# MAGIC #### Mount ISLES data ####

# COMMAND ----------

client_id = dbutils.secrets.get(scope ='spad-kv-weu-prd01-bld006', key ='UCB-SPAD-TCDH-POS-001-SPN-ID')
client_secret = dbutils.secrets.get(scope ='spad-kv-weu-prd01-bld006', key ='UCB-SPAD-TCDH-POS-001-SPN')
configs = {
  "fs.azure.account.auth.type":"OAuth",
  "fs.azure.account.oauth.provider.type":"org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
  "fs.azure.account.oauth2.client.id": client_id,
  "fs.azure.account.oauth2.client.secret": client_secret,
  "fs.azure.account.oauth2.client.endpoint": "https://login.microsoftonline.com/237582ad-3eab-4d44-8688-06ca9f2e613b/oauth2/token"
}

dbutils.fs.mount(source="abfss://raw@tcdhsaweuprd01rwe.dfs.core.windows.net/ISLES/",
                  mount_point="/mnt/tcdh-isles",
                  extra_configs=configs)

folder_location = "dbfs:/mnt/tcdh-isles"  # mount ISLES data which sits in ADLSgen2 storage to this location in DBFS

# COMMAND ----------

# MAGIC %md
# MAGIC #### Unzip ISLES data ####

# COMMAND ----------

import os
import zipfile
from io import BytesIO
from azure.storage.filedatalake import FileSystemClient, DataLakeServiceClient

account_name = "tcdhsaweuprd01rwe"
access_key = "/IAzuhbuo+xwZQmSulRwPLifTkx8wXzAkrMSl2LOJKMrxbarr6ec6Z51i+g49nhsGM45FxhHj6PB+ASt5c/raQ=="
file_system_name = "raw"
base_directory = "ISLES"

dbfs_base_path = "/dbfs/tcdh-isles/"

def list_directory_contents(file_system_client, directory_name: str):
    paths = file_system_client.get_paths(directory_name)

    for path in paths:
        if not path.is_directory and path.name.lower().endswith('.zip'):
            print(f"Unzipping: {path.name}")
            with file_system_client.get_file_client(path.name) as file_client:
                zipped_download = file_client.download_file()
                zipped_data = zipped_download.readall()
                
                original_file_name = os.path.splitext(path.name)[0]
                relative_path = os.path.relpath(path.name, base_directory)
                dbfs_path = os.path.join(dbfs_base_path, relative_path)
                dbfs_path = os.path.splitext(dbfs_path)[0]  # Rename the file extension
                
                os.makedirs(os.path.dirname(dbfs_path), exist_ok=True)

                with zipfile.ZipFile(BytesIO(zipped_data), 'r') as zip_ref:
                    zip_ref.extractall(dbfs_path)
                    print(f"Unzipped files saved to: {dbfs_path}")
                
service_client = DataLakeServiceClient(account_url=f"https://{account_name}.dfs.core.windows.net", credential=access_key)
file_system_client = service_client.get_file_system_client(file_system_name)
list_directory_contents(file_system_client, base_directory)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Build lists of images ####

# COMMAND ----------

import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

base_dir = "/dbfs/tcdh-isles/ISLES_2015/ISLES_2015/SISS2015_Training"

def process_patient(patient_folder):
    patient_path = os.path.join(base_dir, patient_folder)
    img_names = []
    
    for modality_folder in os.listdir(patient_path):
        modality_path = os.path.join(patient_path, modality_folder)
        
        for file_name in os.listdir(modality_path):
            if file_name.endswith(".nii"):
                file_path = os.path.join(modality_path, file_name)
                data = nib.load(file_path).get_fdata().T
                non_black_slices = [i for i in range(data.shape[0]) if np.sum(data[i]) != 0]

                for i in non_black_slices:
                    img = data[i]
                    img = img[19:211, 19:211]
                    img_name = f"{patient_folder}_{modality_folder}_{file_name}_{i}.jpg"
                    img_path = os.path.join("/dbfs/tcdh-isles/All_Images", img_name)
                    plt.imsave(img_path, img, cmap='gray', format='jpeg')
                    img_names.append(img_name)
    return img_names

def process_all_patients(base_dir, output_dir):
    all_img_names = []
    
    for patient_folder in tqdm(sorted(os.listdir(base_dir), key=lambda x: int(x))):
        img_names = process_patient(patient_folder)
        all_img_names.extend(img_names)
    
    np.save(os.path.join(output_dir, "resulting_image_list.npy"), np.array(all_img_names))
    
output_directory = "/dbfs/tcdh-isles/All_Images"
os.makedirs(output_directory, exist_ok=True)

process_all_patients(base_dir, output_directory)

# COMMAND ----------

import re
import numpy as np

npy_file_path = "/dbfs/tcdh-isles/All_Images/resulting_image_list.npy"
image_list = np.load(npy_file_path)

image_list_input_DWI_unpr = [item for item in image_list if "OT" not in item and "DWI" in item] # all DWI images
image_list_input_Flair_unpr = [item for item in image_list if "OT" not in item and "Flair" in item] # all Flair images
image_list_input_T1_unpr = [item for item in image_list if "OT" not in item and "T1" in item] # all T1 images
image_list_input_T2_unpr = [item for item in image_list if "OT" not in item and "T2" in item] # all T2 images

image_list_DWI_input = []
for item in image_list_input_DWI_unpr:
    image_list_DWI_input.append(item) # for test set
        
image_list_Flair_input = []
for item in image_list_input_Flair_unpr:
    image_list_Flair_input.append(item) # for test set

image_list_T1_input = []
for item in image_list_input_T1_unpr:
    image_list_T1_input.append(item) # for test set

image_list_T2_input = []
for item in image_list_input_T2_unpr:
    image_list_T2_input.append(item) # for test set

print(len(image_list_DWI_input), image_list_DWI_input[:3])
print(len(image_list_Flair_input), image_list_Flair_input[:3])
print(len(image_list_T1_input), image_list_T1_input[:3])
print(len(image_list_T2_input), image_list_T2_input[:3])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Perform data augmentation ####

# COMMAND ----------

from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt

data_gen_args = dict(
    rescale=1.0 / 255,
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2,
    zoom_range=0.1
)

def augment_data(images, save_dir, b_size, seed, num_augmentations=3):
    image_datagen = ImageDataGenerator(**data_gen_args)

    augmented_images = []
    for img_relative_path in images:
        img_path = "/dbfs/tcdh-isles/All_Images/" + img_relative_path
        img = np.expand_dims(plt.imread(img_path), 0)

        flow = image_datagen.flow(img, batch_size=b_size, seed=seed, save_to_dir=save_dir, save_prefix=f'aug_{img_relative_path[:-4]}', save_format='jpg')

        for i in range(num_augmentations):
            augmented_batch = next(flow)
            
            for j, augmented_img in enumerate(augmented_batch):
                filename = f'aug_{img_relative_path[:-4]}_{i}_{j}.jpg'
                plt.imsave(os.path.join(save_dir, filename), augmented_img)
                augmented_images.append(os.path.join(save_dir, filename))

    return augmented_images

augmentation_seed = 1337

save_dir_all = "/dbfs/tcdh-isles/All_Images/Augmented_v2/DWI"
os.makedirs(save_dir_all, exist_ok=True)
augmented_DWI_input = augment_data(image_list_DWI_input, save_dir_all, 3, augmentation_seed, num_augmentations=3)

save_dir_all = "/dbfs/tcdh-isles/All_Images/Augmented_v2/Flair"
os.makedirs(save_dir_all, exist_ok=True)
augmented_Flair_input = augment_data(image_list_Flair_input, save_dir_all, 3, augmentation_seed, num_augmentations=3)

save_dir_all = "/dbfs/tcdh-isles/All_Images/Augmented_v2/T1"
os.makedirs(save_dir_all, exist_ok=True)
augmented_T1_input = augment_data(image_list_T1_input, save_dir_all, 3, augmentation_seed, num_augmentations=3)

save_dir_all = "/dbfs/tcdh-isles/All_Images/Augmented_v2/T2"
os.makedirs(save_dir_all, exist_ok=True)
augmented_T2_input = augment_data(image_list_T2_input, save_dir_all, 3, augmentation_seed, num_augmentations=3)

print("Augmented DWI Input:", augmented_DWI_input[:5])
print("Augmented Flair Input:", augmented_Flair_input[:5])
print("Augmented T1 Input:", augmented_T1_input[:5])
print("Augmented T2 Input:", augmented_T2_input[:5])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Save lists in csv file ####

# COMMAND ----------

import csv

lists = [augmented_DWI_input, augmented_Flair_input, augmented_T1_input, augmented_T2_input, augmented_expert]

file_path = '/dbfs/tcdh-isles/lists_export.csv'
with open(file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    for lst in lists:
        writer.writerow(lst)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Normalize Test Data ####

# COMMAND ----------

from PIL import Image
import numpy as np
import os

original_directory = "/dbfs/tcdh-isles/All_Images_Test/"
normalized_directory = "/dbfs/tcdh-isles/All_Images_Test/Normalized/"

os.makedirs(normalized_directory, exist_ok=True)

image_files = [f for f in os.listdir(original_directory)[:50] if os.path.isfile(os.path.join(original_directory, f))]  # subset
print(image_files)

for filename in image_files:
    original_img_path = os.path.join(original_directory, filename)
    print(original_img_path)
    img = Image.open(original_img_path)    
    img_array = np.array(img)
    img_normalized = img_array / 255.0
    
    normalized_img_path = os.path.join(normalized_directory, filename)
    print(normalized_img_path)
    Image.fromarray((img_normalized * 255).astype(np.uint8)).save(normalized_img_path)

# COMMAND ----------


