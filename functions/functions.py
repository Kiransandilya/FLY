def imshow(alpha, cropped, grid=False,rotate=0):
    for i in range(alpha):
        file_list = sorted(os.listdir(cropped))
        z = random.choice([f for f in file_list if f.endswith("tif")])
        img = mh.imread(os.path.join(cropped, z))
        img = ndimage.rotate(img, rotate)
        print(z)
        if not grid:
            fig = plt.figure()
            fig.set_size_inches(30, 30)
            plt.imshow(img)
        else:
            fig = plt.figure()
            fig.set_size_inches(30, 30)
            ax = plt.gca()
            ax.imshow(img)
            ax.xaxis.set_major_locator(MultipleLocator(200))
            ax.yaxis.set_major_locator(MultipleLocator(200))
            plt.grid(True, which='major', axis='both', linestyle='-', color='r')


def unzip(path, file_name):
    zip_file_path = f"{path}/{file_name}.zip"
    extract_to_path = f"{path}/{file_name}"
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all the contents into the target directory
        zip_ref.extractall(extract_to_path)

    print(f"Successfully extracted to {extract_to_path}")
    
    
# update with new dimensions
def cropC1(test):
    for z in tqdm(sorted(os.listdir(test))):
        if (z.endswith("tif")): # checking the file ends with tif
            # Read in the image
            img = tifffile.imread(os.path.join(test, z))
            img_cropped = img[1000:2500, 2500:4500]
            tifffile.imsave(os.path.join(test, z), img_cropped)
            #print(z)
            #hint: crop(spath) - To crop all images
            
            
###############################################functions
start_time = 0  # Define start_time in the global scope

def starttime():
    global start_time  # Use the global keyword to access the global start_time variable
    start_time = time.time()
    #hint: starttime() - To start timer.
    
def endtime():
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    #hint: endtime() - To end timer

def check(path):
    # Create the directory if it does not exist
    os.makedirs(path, exist_ok=True)

    # Remove the directory and all its contents
    shutil.rmtree(path)

    # Create a new empty directory
    os.mkdir(path)
    #hint: check(path) - To recreate a particular path
from PIL import Image 

def heavycrop(test):
    starttime()
    for z in tqdm(sorted(os.listdir(test))):
        if z.endswith("tif"):
            # Read in the image
            try:
                img = mh.imread(os.path.join(test, z))
            except Exception as e:
                try:
                    img = Image.open(os.path.join(test, z))
                except Exception as e:
                    print(f"An error occurred: {e}, image.openn isusue ############################")
            width, height = img.size
            # Calculate the number of crops in each dimension
            num_crops_y = height // 512
            num_crops_x = width // 512

            for i in range(num_crops_y):
                for j in range(num_crops_x):
                    # Crop the image
                    start_y = i * 512
                    start_x = j * 512
                    img_cropped = img.crop((start_x, start_y, start_x + 512, start_y + 512))

                    # Create a new file name for the cropped image
                    file_name, file_ext = os.path.splitext(z)
                    new_file_name = f"{file_name}_{i}_{j}{file_ext}"

                    # Save only if cropped image has size (512, 512)
                    if img_cropped.size == (512, 512):
                        try:
                            img_cropped.save(os.path.join(test, new_file_name))
                        except Exception as e:
                            print(f"An error occurred: {e}")
                        
                    else:
                        print(f"Warning: Cropped image has unexpected size {img_cropped.size}")
                        
            endtime()
            # Remove original image file after cropping is done.
            os.remove(os.path.join(test,z))
            #hint: heavycrop(spath) - To heavy crop all images 512x512
            
def npyconversion(tif_dir, npy_path):
    tif_files = [f for f in os.listdir(tif_dir) if f.endswith('.tif')]
    tif_files.sort()
    data = []
    for tif_file in tqdm(tif_files):
        img = Image.open(os.path.join(tif_dir, tif_file))
        data.append(np.array(img))
    np.save(npy_path, data)
    #hint: npyconversion(path , npy + '/filename.npy' ) -To create NPY files
            
def a2bcopy(path1, path2):
    for z in tqdm(sorted(os.listdir(path1))):
        if z.endswith("tif"):
            shutil.copy(os.path.join(path1, z), os.path.join(path2, z))
            #hint: a2bcopy(sorce path, dest path) - To copy all images
            
def crop(test):
    for z in tqdm(sorted(os.listdir(test))):
        if (z.endswith("tif")): # checking the file ends with tif
            # Read in the image
            img = mh.imread(os.path.join(test, z))
            img_cropped = img[1000:2500, 2500:4500]
            mh.imsave(os.path.join(test, z), img_cropped)
            print(z)
            #hint: crop(spath) - To crop all images
    
def a2brandom(src_dir, dst_dir, number):
    # Get a list of all image files in the source directory
    image_files = [f for f in tqdm(os.listdir(src_dir)) if f.endswith('.tif')]
    # Randomly select 10 images from the list
    selected_images = random.sample(image_files, number)
    # Copy the selected images to the destination directory
    for image in selected_images:
        src_path = os.path.join(src_dir, image)
        dst_path = os.path.join(dst_dir, image)
        shutil.copy2(src_path, dst_path)
    # Print a message when done
    print('Copied 10 random images to', dst_dir)
    #hint: a2brandom(sorce path, dest path, random number) - To copy n random images
    
def count_files(dir_path):
    if os.path.isdir(dir_path):
        file_count = 0
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
            #print(f'{file_name} - Size: {file_size_mb} MB')
            file_count += 1
        return file_count
    else:
        print(f"{dir_path} is not a valid directory")
        return 0    
        #hint: count_files(path) - To count number of files in that path
    
def norm(path):
    for z in tqdm(sorted(os.listdir(path))):# added interactive progressbar to decrease the uncertanity and to increase curiosity :)
        if (z.endswith("tif")): # checking the file ends with tif 
            img = mh.imread(os.path.join(path, z))            
            # Normalize the image
            img = img.astype(np.float64)
            img /= img.max()
            img *= 255            
            # Save the processed image back to the temporary directory
            mh.imsave(os.path.join(path, z), img)
            #hint: norm(path) - To normalize all the images in the path
            
def shape(raw):
    for z in tqdm(sorted(os.listdir(raw))):
        if (z.endswith("tif")):
            img = mh.imread(os.path.join(raw, z))
            print (img.shape)
            #hint: shape(path) _ To print shape of all the images in the path
            
def refresh(experiment: str, directories: dict):
    for key in directories:
        if os.path.exists(directories[key]):
            shutil.rmtree(directories[key])
        os.makedirs(directories[key])
        #hint: refresh("experiment name", directories) - to recreate all directories in that dict
        
def paths(directories):
    for key, value in directories.items():
        globals()[key] = value
    return directories
    #hint: paths(directories) - To call the directories outside the dictionary
        
def help():
    functions = [value for key, value in globals().items() if inspect.isfunction(value)]
    headers = ["Function", "Hint", "Used for"]
    data = []
    for func in functions:
        source = inspect.getsource(func)
        lines = source.split("\n")
        hint_line = [line for line in lines if line.strip().startswith("#hint:")]
        if hint_line:
            hint_parts = hint_line[0].split("#hint:")[1].strip().split(" - ")
            hint_text = hint_parts[0]
            usage_text = hint_parts[1] if len(hint_parts) > 1 else ""
            data.append([f"{func.__name__}()", hint_text, usage_text])
    print(tabulate(data, headers=headers))
    
def readpaths(directories):
    for key, path in directories.items():
        globals()[key] = path
            
            
def loadnpz(location, filename, p=False):
    # Load the data
    data = np.load(os.path.join(location, filename))
    # Extract the file and names
    file = data['data']
    names = data['names']
    file = file.reshape(file.shape[0], file.shape[1], file.shape[2], 1)
    if p:
        # Randomly permute the file if p is True
        p = np.random.permutation(len(file))
        file = file[p]
    # Normalize the file to range [0, 1]
    file = file.astype(np.float64)
    for i in range(file.shape[0]):
        file[i] = (file[i] - file[i].min()) / (file[i].max() - file[i].min())
    print(f'File shape: {file.shape}')
    return file, names


def unet_load(weight):
    model = custom_unet(
    input_shape=(512, 512, 1),
    use_batch_norm=False,
    num_classes=1,
    filters=32,
    dropout=0.5,
    output_activation='sigmoid')
    
    opt = keras.optimizer_v1.Adam(lr=0.01)
    
    model.compile(optimizer = 'Adam',    
              loss='binary_crossentropy', 
              metrics=[iou, iou_thresholded])
    
    model.load_weights(weight)
    
    return model

def unet_create():
    model = custom_unet(
    input_shape=(512, 512, 1),
    use_batch_norm=False,
    num_classes=1,
    filters=32,
    dropout=0.5,
    output_activation='sigmoid')
    
    opt = keras.optimizer_v1.Adam(lr=0.01)
    
    model.compile(optimizer = 'Adam',    
              loss='binary_crossentropy', 
              metrics=[iou, iou_thresholded])
    
    #model.load_weights(weight)
    
    return model


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, jaccard_score, f1_score
import tifffile

def load_tiff(file_path):
    return tifffile.imread(file_path)

def normalize_image(image):
    # Scale the image values to be in the range [0, 255]
    return (image * 255).astype(np.uint8)

def calculate_metrics1(gt_image, pred_image):
    gt_flat = gt_image.flatten()
    pred_flat = pred_image.flatten()

    pixel_accuracy = accuracy_score(gt_flat, pred_flat)
    iou = jaccard_score(gt_flat, pred_flat, pos_label=255)
    dice_coefficient = f1_score(gt_flat, pred_flat, pos_label=255)
    return pixel_accuracy, iou, dice_coefficient

#[None, 'micro', 'macro', 'weighted']
def calculate_metrics2(gt_image, pred_image):
    gt_flat = gt_image.flatten()
    pred_flat = pred_image.flatten()

    pixel_accuracy = accuracy_score(gt_flat, pred_flat)
    iou = jaccard_score(gt_flat, pred_flat, average='micro', pos_label=255)
    dice_coefficient = f1_score(gt_flat, pred_flat, average='micro', pos_label=255)
    return pixel_accuracy, iou, dice_coefficient

def plot_images(gt_image, pred_image):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(gt_image, cmap='gray')
    plt.title('Ground Truth')

    plt.subplot(1, 3, 2)
    plt.imshow(pred_image, cmap='gray')
    plt.title('Predicted')

    plt.subplot(1, 3, 3)
    diff_image = np.abs(  pred_image - gt_image   )
    plt.imshow(diff_image, cmap='gray')
    plt.title('Difference')

    plt.show()

import pandas as pd
import numpy as np


def validation2(phase,pred_folder, gt_folder):
    gt_files = [f for f in os.listdir(gt_folder) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
    pred_files= [f for f in os.listdir(pred_folder) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]

    metrics_data = []

    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_image = load_tiff(os.path.join(gt_folder, gt_file))
        pred_image = load_tiff(os.path.join(pred_folder, pred_file))
        
        gt_image = normalize_image(gt_image)
        pred_image = normalize_image(pred_image)
        #calculate metrics
        accuracy, iou, dice = calculate_metrics2(gt_image, pred_image)
        #print results
        print(f'Pixel Accuracy: {accuracy:.4f}')
        print(f'Intersection over Union (IoU): {iou:.4f}')
        print(f'Dice Coefficient: {dice:.4f}')
        # Plot images
        plot_images(gt_image, pred_image)
        # Bar plot for metrics
        metrics_names = ['Pixel Accuracy', 'IoU', 'Dice Coefficient']
        metrics_values = [accuracy, iou, dice]

        plt.bar(metrics_names, metrics_values)
        plt.ylabel('Metric Value')
        plt.title('Metrics Comparison')
        plt.show()
        
              
        
        metrics_data.append([gt_file, accuracy, iou, dice])

    metrics_df = pd.DataFrame(metrics_data, columns=['Slice', 'Pixel Accuracy', 'IoU', 'Dice'])
#     print(metrics_df)
    print(tabulate(metrics_df, headers='keys', tablefmt='pretty'))
    # Calculate average and standard deviation
    avg_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()

    summary_df = pd.DataFrame([avg_metrics, std_metrics], index=['Average', 'Std Dev'])
    print(summary_df)
    phase =phase
    # Save to CSV
    metrics_df.to_csv(phase, index=False)



def validation1(phase,pred_folder, gt_folder):
    gt_files = [f for f in os.listdir(gt_folder) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
    pred_files = [f for f in os.listdir(pred_folder) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]

    metrics_data = []

    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_image = load_tiff(os.path.join(gt_folder, gt_file))
        pred_image = load_tiff(os.path.join(pred_folder, pred_file))
        
        gt_image = normalize_image(gt_image)
        pred_image = normalize_image(pred_image)
        #calculate metrics
        accuracy, iou, dice = calculate_metrics1(gt_image, pred_image)
        #print results
        print(f'Pixel Accuracy: {accuracy:.4f}')
        print(f'Intersection over Union (IoU): {iou:.4f}')
        print(f'Dice Coefficient: {dice:.4f}')
        # Plot images
        plot_images(gt_image, pred_image)
        # Bar plot for metrics
        metrics_names = ['Pixel Accuracy', 'IoU', 'Dice Coefficient']
        metrics_values = [accuracy, iou, dice]

        plt.bar(metrics_names, metrics_values)
        plt.ylabel('Metric Value')
        plt.title('Metrics Comparison')
        plt.show()
        
        
#         fig = go.Figure(data=[go.Bar(x=metrics_names, y=metrics_values)])
#         fig.update_layout(title_text='Metrics Comparison')
#         fig.show()
        
        metrics_data.append([gt_file, accuracy, iou, dice])

    metrics_df = pd.DataFrame(metrics_data, columns=['Slice', 'Pixel Accuracy', 'IoU', 'Dice'])
#     print(metrics_df)
    print(tabulate(metrics_df, headers='keys', tablefmt='pretty'))
    # Calculate average and standard deviation
    avg_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()

    summary_df = pd.DataFrame([avg_metrics, std_metrics], index=['Average', 'Std Dev'])
    print(summary_df)
    phase =phase
    # Save to CSV
    metrics_df.to_csv(phase, index=False)

 #############################################################################   
def validation(pred_folder,gt_folder):
    gt_files = [f for f in os.listdir(gt_folder) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
    pred_files = [f for f in os.listdir(pred_folder) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]

    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_image = load_tiff(os.path.join(gt_folder, gt_file))
        pred_image = load_tiff(os.path.join(pred_folder, pred_file))
        
        gt_image = normalize_image(gt_image )
        pred_image = normalize_image(pred_image)
        #calculate metrics
        accuracy, iou, dice = calculate_metrics(gt_image, pred_image)
        #print results
        print(f'Pixel Accuracy: {accuracy:.4f}')
        print(f'Intersection over Union (IoU): {iou:.4f}')
        print(f'Dice Coefficient: {dice:.4f}')
        # Plot images
        plot_images(gt_image, pred_image)
        # Bar plot for metrics
        metrics_names = ['Pixel Accuracy', 'IoU', 'Dice Coefficient']
        metrics_values = [accuracy, iou, dice]

        plt.bar(metrics_names, metrics_values)
        plt.ylabel('Metric Value')
        plt.title('Metrics Comparison')
        plt.show()

#def del(path):
def convert_to_binary(source_folder, destination_folder):
    # Get a list of all the image files in the source folder
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith('.tif')]

    # Check if destination folder exists, if not, create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for image_file in image_files:
        # Load the image
        image = Image.open(os.path.join(source_folder, image_file))

        # Display the original image
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.show()

        # Check if the image is already binary
        if len(np.unique(image)) > 2:
            # Convert the image to grayscale
            image = image.convert('L')

            # Convert the grayscale image to binary
            binary_image = image.point(lambda x: 0 if x<128 else 255, '1')
        else:
            binary_image = image

        # Display the binary image
        plt.imshow(binary_image, cmap='gray')
        plt.title('Binary Image')
        plt.show()

        # Save the binary image
        binary_filename =  image_file
        binary_image.save(os.path.join(destination_folder, binary_filename))


import zipfile
import os
import cv2
import time
import shutil
import random
import inspect
import keras
import numpy as np
from PIL import Image
from tabulate import tabulate
from tqdm import tqdm
from scipy import ndimage
import mahotas as mh

from skimage import measure, filters, exposure
from matplotlib.ticker import MultipleLocator
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import keras.optimizers
import skimage.filters as filters
import skimage.measure as measure
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Conv2DTranspose, Dense, MaxPooling2D
from keras.metrics import MeanIoU
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras_unet.losses import jaccard_distance
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.models import custom_unet
from keras_unet.utils import plot_imgs, plot_segm_history
from sklearn.metrics import classification_report

import tifffile
# import plotly.graph_objects as go

import zipfile
import os
import cv2
import time
import shutil
import random
import inspect
import keras
import numpy as np
from PIL import Image
from tabulate import tabulate
from tqdm import tqdm
from scipy import ndimage
import mahotas as mh

from skimage import measure, filters, exposure
from matplotlib.ticker import MultipleLocator
from matplotlib import image as mpimg
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import keras.optimizers
import skimage.filters as filters
import skimage.measure as measure
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Conv2DTranspose, Dense, MaxPooling2D
from keras.metrics import MeanIoU
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras_unet.losses import jaccard_distance
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.models import custom_unet
from keras_unet.utils import plot_imgs, plot_segm_history
from sklearn.metrics import classification_report

import tifffile

from PIL import Image
import matplotlib.pyplot as plt


# Import necessary libraries
import os
import cv2
import time
import shutil
import random
import inspect
#import imageio as im
import numpy as np
import mahotas as mh
from PIL import Image
from tabulate import tabulate
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
#from skimage import measure, filters

         
import keras.optimizers
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance    