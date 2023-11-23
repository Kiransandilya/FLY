

import os
from PIL import Image
import tifffile
import numpy as np
from tqdm import tqdm
import mahotas as mh

from functions.functions import *

from functions.phase3 import *


def normalize_image(img):
    arr = np.array(img) 
    arr = arr.astype(np.float32)/255
    return Image.fromarray(arr)
#(img, x1=1130, x2=2154,y1=2400, y2=3936):
def crop_image(img, x1=1130, x2=2400,y1=2154, y2=3936):
    arr = np.array(img)
    cropped = arr[x1:x2, y1:y2] 
    return Image.fromarray(cropped)

class ImageCropper:
    def __init__(self):
        pass
    
    def crop_initial_images(self, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        files = os.listdir(input_folder)

        for file_name in files:
            if file_name.lower().endswith(".tif") or file_name.lower().endswith(".tiff"):
                input_image_path = os.path.join(input_folder, file_name)
                original_image = Image.open(input_image_path)
                # Normalize the image
                
                
                normalize = normalize_image(original_image)
#                 crop_coords= crop_coords
                # Crop the image
                cropped_image = crop_image(normalize)
                os.makedirs(output_folder, exist_ok=True)

                output_image_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.tif")
                cropped_image.save(output_image_path)
    
    def only_crop_initial_images(self, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        files = os.listdir(input_folder)

        for file_name in files:
            if file_name.lower().endswith(".tif") or file_name.lower().endswith(".tiff"):
                input_image_path = os.path.join(input_folder, file_name)
                original_image = Image.open(input_image_path)
                # Normalize the image
                
                
                #normalize = normalize_image(original_image)
#                 crop_coords= crop_coords
                # Crop the image
                cropped_image = crop_image(original_image)
                os.makedirs(output_folder, exist_ok=True)

                output_image_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.tif")
                cropped_image.save(output_image_path)

    def crop_512x512_tiles(self, input_folder, output_folder):
         # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        files = os.listdir(input_folder)

        for file_name in files:
            if file_name.lower().endswith(".tif") or file_name.lower().endswith(".tiff"):
                input_image_path = os.path.join(input_folder, file_name)
                original_image = Image.open(input_image_path)

                tile_size = 512
                tile_count = 0

                for y in range(0, original_image.height, tile_size):
                    for x in range(0, original_image.width, tile_size):
                        box = (x, y, x + tile_size, y + tile_size)
                        tile = original_image.crop(box)

                        base_name = os.path.splitext(file_name)[0]
                        output_tile_filename = f"{output_folder}/{base_name}_{x}_{y}_count_{tile_count}.tif"
                        tile.save(output_tile_filename)

                        tile_count += 1

    def stitch_tiles(self, input_folder, output_folder):
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        files = os.listdir(input_folder)

        # Group the tiles by base name
        base_names = set('_'.join(f.split('_')[:-4]) for f in files)
        for base_name in base_names:
            base_files = [f for f in files if f.startswith(base_name)]

            folder_tiles = []
            for filename in base_files:
                if filename.endswith((".tif", ".tiff")):
                    parts = filename.split('_')
                    x, y = int(parts[-4]), int(parts[-3])
                    tile_image = Image.open(os.path.join(input_folder, filename)).convert("RGBA")
                    folder_tiles.append((x, y, tile_image))

            folder_tiles.sort(key=lambda tile: (tile[1], tile[0]))

            max_x = max(tile[0] + tile[2].width for tile in folder_tiles)
            max_y = max(tile[1] + tile[2].height for tile in folder_tiles)

            stitched_image = Image.new("RGBA", (max_x, max_y), (255, 255, 255, 0))

            for tile in folder_tiles:
                x, y, tile_image = tile
                # Adjust the coordinates to be within the bounds of the stitched image
                x = max(0, min(x, stitched_image.width - tile_image.width))
                y = max(0, min(y, stitched_image.height - tile_image.height))
                mask = Image.new("L", tile_image.size, 255)
                stitched_image.paste(tile_image, (x, y), mask)

            output_image_path = os.path.join(output_folder, f"{base_name}.tif")
            stitched_image.save(output_image_path)
            print(f"Image for base name '{base_name}' reconstructed successfully.")

    def invert_colors_in_folder(self, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        # Iterate over each subfolder in the input folder
        for subfolder_name in os.listdir(input_folder):
            subfolder_path = os.path.join(input_folder, subfolder_name)

            # Check if the item in the folder is a subfolder
            if os.path.isdir(subfolder_path):
                # Create the output subfolder
                output_subfolder_path = os.path.join(output_folder, subfolder_name)
                os.makedirs(output_subfolder_path, exist_ok=True)

                # Iterate over each file in the subfolder
                for file_name in os.listdir(subfolder_path):
                    # Check if the file is an image
                    if file_name.lower().endswith((".tiff", ".tif")):
                        # Construct the full path for the input image
                        input_image_path = os.path.join(subfolder_path, file_name)

                        # Open the image
                        image = Image.open(input_image_path)

                        # Invert colors (black to white, white to black)
                        inverted_image = Image.eval(image, lambda x: 255 - x)

                        # Construct the full path for the output image
                        output_image_path = os.path.join(output_subfolder_path, file_name)

                        # Save the inverted image
                        inverted_image.save(output_image_path)

    def overlay_images_with_transparent_blacks(self, image_with_markings, base_image, output_path):
        # Open the images
        markings = Image.open(image_with_markings)
        base = Image.open(base_image)

        # Ensure both images have an alpha channel
        markings = markings.convert("RGBA")
        base = base.convert("RGBA")

        # Create a new image with transparent blacks
        transparent_blacks = Image.new("RGBA", markings.size, (0, 0, 0, 0))
        transparent_blacks.paste(markings, (0, 0), markings)

        # Composite the images
        result = Image.alpha_composite(base, transparent_blacks)

        # Save the result
        result.save(output_path)


def norm(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for z in tqdm(sorted(os.listdir(input_folder))):
        if z.endswith((".tif", ".tiff")):
            img_path = os.path.join(input_folder, z)
            img = tifffile.imread(img_path)

            # Normalize the image
            img = img.astype(np.float64)
            img /= img.max()
            img *= 255

            # Save the processed image to the normalized directory
            output_path = os.path.join(output_folder, z)
            tifffile.imsave(output_path, img)


phase = ImageCropper()
def saveimage(n, img_path, data, steps, z):
        step = f'step{n}'
        step_path = os.path.join(steps, step)
        if not os.path.exists(step_path):
            os.makedirs(step_path)
        print(img_path)
        
        new_path = os.path.join(step_path, z)
        
        
        
        print(new_path)  
        shutil.copy(img_path, new_path)
        tifffile.imsave(new_path, data)
        new_image = tifffile.imread(os.path.join(step_path, z))
        print('saved-image')
        plt.figure(figsize=(10,10))
        plt.imshow(new_image, cmap='gray')
        print(f'Saved at {new_path}')
        plt.figure(figsize=(10,10))
        plt.imshow(data, cmap='gray')
        return new_path

    
def phase12(filename,thres,steps,source1,destination):   
    z = filename
    img_path = (os.path.join(source1, z)) #tifffile.imread
    print(z)
    success_count=0 
    print(img_path)
    print('source-1',source1)
    img = tifffile.imread(os.path.join(source1,z))
    print (z)
    plt.imshow(img)
    #plt.show()

    # Apply a Gaussian filter to the image
    c = img.copy()
    #b = mh.gaussian_filter(b, sigma=3)

    # Set values below 100 to 0
    for a in range(150, 0, -1):
        starttime()
        b1= img.copy()
        b1 = mh.gaussian_filter(b1, sigma=3)
        b=b1.copy()
        b[b < a] = 0
        #print (a)      
        #b = exposure.equalize_hist(b)
        # Label the regions in the filtered image
        labeled, number = mh.label(b)


        # filter based on labeled region size
        sizes = mh.labeled.labeled_size(labeled)

        # Remove the regions that are less than 1000
        too_small = np.where(sizes < 1500)
        labeled_only_big = mh.labeled.remove_regions(labeled, too_small)


        plt.imshow(labeled_only_big)
        plt.show()

        #too_large = np.where(sizes > 20500)
        #labeled_only_big = mh.labeled.remove_regions(labeled, too_large)
        #for debug
        #plt.imshow(labeled_only_big)
        #plt.show()


        # Create a binary mask from the filtered labeled regions
        binary_mask = labeled_only_big.copy()
        binary_mask[binary_mask > 0] = 1
        labeled1, number_1 = mh.label(binary_mask)

        plt.imshow(labeled1)
        plt.show()


         # Close the regions in the binary mask
        binary_mask_closed = mh.morph.close(binary_mask)



        plt.imshow(binary_mask_closed)
        plt.show()


        plt.figure(figsize=(10,10))
        #plt.imshow(binary_mask_closed)
        #plt.show() 

        # Set a threshold for the minimum region size           
        min_region_size = 3000

        # Initialize a variable to count the number of regions above the minimum size
        large_regions = 0

        # Get the sizes of the labeled regions
        region_sizes = measure.regionprops(labeled, intensity_image=binary_mask_closed)

        # Iterate over the region sizes and count the number of large regions
        for region in region_sizes:
            if region.area > min_region_size:
                 large_regions += 1


        threshold = filters.threshold_otsu(binary_mask_closed) 
        binary_image = binary_mask_closed > threshold

        plt.imshow(binary_image)
        plt.show()
        
        print (number_1) 
        print (large_regions)
        print (threshold)


        print('time taken for iteration',a,'image',z ,'is:')
        endtime()
#             if number_1>= 90:                
#                 print (z)
#                 plt.imshow(binary_image)
#                 plt.show()
#                 print (number_1)
#                 print (threshold)
#                 print(large_regions)

        if number_1 <=110 and number_1 >=85:
            if large_regions <=20:       # 20 is ideal value 
                print ("######################################################################")
                print(z)
                #plt.figure(figsize=(10,10))
                print("The image has clear segmentation.")
                #plt.imshow(binary_image)
                #plt.show()
                print (number_1)
                print (threshold)
                print(large_regions)


                #step-4 Smoothed data
                saveimage(3, img_path, b1, steps, z)                    
                #step -5 Thresholding
                saveimage(4, img_path, b, steps, z)                 
                #step -6 labelled
                saveimage(5, img_path, labeled, steps, z)                
                #step -7 removing small islands
                saveimage(6, img_path, labeled_only_big, steps, z)                  
                #step-8 binary mask open
                saveimage(7, img_path, labeled1, steps, z)                
                #step-9 close binary mask
                saveimage(8, img_path, binary_mask_closed, steps, z)                
                # final image =
                saveimage(9, img_path, binary_image, steps, z)
                

                os.makedirs(destination, exist_ok=True)
                output = os.path.join(destination, z)
                shutil.copy(img_path, output)
                tifffile.imsave(output, binary_image)


                #print (sizes)
                print ("######################################################################")
                success_count+=1
                print (success_count)
                break
    print (success_count)
    print ('######################################### DONE        ############################################')


    
    
#########################################################################################################    
def phase1prediction(thres,original_data,final_output):
    thres =thres
    crop1 = '/raid/mpsych/RISTERLAB/kiran/flyem/K_Experiments/ISBI/images/Phase-1/steps/crop1'
    steps= '/raid/mpsych/RISTERLAB/kiran/flyem/K_Experiments/ISBI/images/Phase-1/steps/'
    phase.crop_initial_images(original_data,crop1)
    print('crop is done')
    file_list = sorted(os.listdir(crop1))
    print('phase12 ##########  ####### start' )
    steps = steps
    destination = final_output
    for filename in file_list:
        if filename.endswith(".tif"):            
            phase12(filename,thres,steps,crop1,destination)
            print('phase12')
    print('phase12 ################# end' )        
###############################################################################################################"" THe  above funtion here will finish from reading image,normalising,croping and last prediction"""
# This includes few functions which are used above
# Fix this in oops concept and also, write functions for validations as well
# phase1validation(prediction_folder, Groundtruth_folder)
# validation should consist of  IOU and Dice co-efficient functions to validate the scores.
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


def npycon(tif_dir, npy_path):
    tif_files = [f for f in os.listdir(tif_dir) if f.lower().endswith(('.tif', '.tiff'))]
    data = []
    image_names = []
    for tif_file in tqdm(tif_files):
        img = Image.open(os.path.join(tif_dir, tif_file))
        data.append(np.array(img))
        image_names.append(tif_file)
    np.savez(npy_path, data=data, names=image_names)
    
    #npycon(source , destination + '/filename.npz')
    
    
    
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

def phase2prediction(model,npz_test,npz_name,length_npz,prediction):
    model =model
    y_pred = model.predict(npz_test)
    for i in range(length_npz):
        z = npz_name[i]
        binary_mask = y_pred[i]
#         print(z)
        threshold = filters.threshold_otsu(binary_mask) 
        binary_mask = binary_mask > threshold  
        new_path = os.path.join(prediction, z)    
        tifffile.imsave(new_path, binary_mask)
#         print(f'Saved at {new_path}')
#         plt.figure(figsize=(15, 5))
#         plt.imshow(binary_mask,cmap='gray')
#         plt.show()
    

def phase2processing(source,prediction,destination,weight):
    os.makedirs(destination, exist_ok=True)
    os.makedirs(prediction, exist_ok=True)
    prediction = prediction
    destination = destination
    temp = 'temp_folder'
    os.makedirs(temp, exist_ok=True)
    phase.crop_initial_images(source,temp)
    images_in_temp = len(os.listdir(temp))
    print('initial crop is done',images_in_temp)
    temp2 = 'temp2_folder'
    os.makedirs(temp2, exist_ok=True)
    phase.crop_512x512_tiles(temp,temp2)
    images_in_temp2 = len(os.listdir(temp2))
    print('512x512 crop is done',images_in_temp2)
    #delete temp folder
    shutil.rmtree(temp)
    #converting into npz file
    npycon(temp2 , temp2 + '/prediction.npz')
    npz_file,npz_name = loadnpz(temp2, 'prediction.npz', p=False)
    model = unet_load(weight)
    npz_images = npz_file
    npz_names = npz_name
    length_npz = len(npz_images)
    npz_test = npz_images[0:length_npz]    
    print('phase2-prediction-begin')
    phase2prediction(model,npz_test,npz_name,length_npz,prediction)   #function
    print('phase-2 stitch back begin')
    phase.stitch_tiles(prediction,destination)                        #function
    shutil.rmtree(temp2)
    print ('phase-2 success')

###########################################################################################################    
def phase3retrain(original_weight,orignal_folder,masks_location,predict_folder,temp_folder, destination=False):
    predict_originals = predict_folder
    original_weight = original_weight
    model_unet_untrained = modelload_unet(original_weight)
    orignal_folder = orignal_folder
    lamda =  generate_random_folder_name()
    temp_folder=os.path.join(temp_folder, f"{lamda}")
    os.makedirs(temp_folder, exist_ok=True)
    
    rawimages_location= generate_random_folder_name()
    rawimages_location=os.path.join(temp_folder, f"{rawimages_location}")
    os.makedirs(rawimages_location, exist_ok=True)
    raw = rawimages_location
    phase.only_crop_initial_images(orignal_folder,raw)
    mask = masks_location
    destination = destination
    if not destination:
        destination = False        
    print ('temp - folder',temp_folder)       
    model_unet_trained,weight_path,temp_dest = modeltraining(raw, mask,temp_folder, model_unet_untrained)
    print ('model trained')
    modelpredict(model_unet_trained,weight_path,temp_folder,predict_originals,destination=destination)
#     shutil.rmtree(temp_dest)
    print (temp_dest)
    print (weight_path)
    shutil.rmtree(rawimages_location)
    print ('model predicted')    
    print ('phase-3- re-train- completed')
############################################################################################################               
def phase3(orignal_folder,masks_location,predict_folder,temp_folder, destination=False):
    predict_originals = predict_folder
    model_unet_untrained = modelcreate_unet()
    orignal_folder = orignal_folder
    lamda =  generate_random_folder_name()
    temp_folder=os.path.join(temp_folder, f"{lamda}")
    os.makedirs(temp_folder, exist_ok=True)
    
    rawimages_location= generate_random_folder_name()
    rawimages_location=os.path.join(temp_folder, f"{rawimages_location}")
    os.makedirs(rawimages_location, exist_ok=True)
    raw = rawimages_location
    phase.only_crop_initial_images(orignal_folder,raw)
    mask = masks_location
    destination = destination
    if not destination:
        destination = False        
    print ('temp - folder',temp_folder)       
    model_unet_trained,weight_path,temp_dest = modeltraining(raw, mask,temp_folder, model_unet_untrained)
    print ('model trained')
    modelpredict(model_unet_trained,weight_path,temp_folder,predict_originals,destination=destination)
#     shutil.rmtree(temp_dest)
    print (temp_dest)
    print (weight_path)
    shutil.rmtree(rawimages_location)
    print ('model predicted')    
    print ('phase-3 completed')
           



# if __name__ == "__main__":
#     input_folder = "ground_truth"
#     output_cropped_folder = 'output_cropped_images'
#     output_tiles_folder = 'output_tiles'
#     output_stitched_folder = 'output_stitched_images'

#     cropper = ImageCropper()

#     # Normalize the images in the input folder
#     #uncomment to normalise, but normalise fucntion not owrking propery, it corrupts the input images
#     #norm("output_cropped_images", "output_norm")
#     # Call the initial crop function

#     cropper.invert_colors_in_folder("invert_test", "inverted")
#     print("images inverted")

#     cropper.crop_initial_images("grayimages", "initial_crop_output")
#     print("Images initially cropped successfully.")

#     # Call the 512x512 tile crop function
#     cropper.crop_512x512_tiles("initial_crop_output", "tile_output")
#     print("Images cropped into 512x512 tiles successfully.")

#     # Call the stitch tiles function
#     cropper.stitch_tiles("inverted", "stitched_outputs")
#     print("Images stitched back together successfully.")

#     #cropper.overlay_images_with_transparent_blacks("stitched_outputs/tile_1024_0_count_2 (1)_reconstructed.tif", "initial_crop_output/355_cropped.tif","overllllay.tif")


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



from sklearn.metrics import jaccard_score, f1_score, confusion_matrix
from keras.models import load_model
import keras.optimizers
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance