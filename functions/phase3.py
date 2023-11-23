import os
import random
import cv2
import mahotas as mh
import skimage.morphology as skm
import numpy as np
import random
import string

from functions.functions import *
from functions.pipelines import *
from functions.modules import *
import tifffile
def npycon(tif_dir, npy_path):
    tif_files = [f for f in os.listdir(tif_dir) if f.lower().endswith(('.tif', '.tiff'))]
    data = []
    image_names = []
    for tif_file in tqdm(tif_files):
        img = Image.open(os.path.join(tif_dir, tif_file))
        data.append(np.array(img))
        image_names.append(tif_file)
    np.savez(npy_path, data=data, names=image_names)
    
def normalize_image(img):
    arr = np.array(img) 
    arr = arr.astype(np.float32)/255
    return Image.fromarray(arr)

def crop_image(img, x1=1130, x2=2154,y1=2400, y2=3936):
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
                    tile_image = Image.open(os.path.join(input_folder, filename))#.convert("RGBA")
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

##############################################################################

phase = ImageCropper()

def generate_random_folder_name():
    letters = ''.join(random.choice(string.ascii_lowercase) for _ in range(7))
    
    return letters

def modelcreate_unet():
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
    print ('done')

def modelload_unet(original_weight):
    model = custom_unet(
    input_shape=(512, 512, 1),
    use_batch_norm=False,
    num_classes=1,
    filters=32,
    dropout=0.5,
    output_activation='sigmoid')
    
    opt = keras.optimizer_v1.Adam(lr=0.001)
    
    model.compile(optimizer = 'Adam',    
              loss='binary_crossentropy', 
              metrics=[iou, iou_thresholded])
    
    model.load_weights(original_weight)
    
    return model
    print ('done')
    
def modeltraining(raw, mask,temp_folder, model):
    # Check if destination path exists, if not create one
    temp_dest = generate_random_folder_name()
    temp_dest=os.path.join(temp_folder, f"{temp_dest}")
    os.makedirs(temp_dest, exist_ok=True)
    desti = temp_dest    
    
    # Create temp folders
    raw_images = os.path.join(desti, 'raw_images')
    mask_images = os.path.join(desti, 'mask_images')
    os.makedirs(raw_images, exist_ok=True)
    os.makedirs(mask_images, exist_ok=True)   
    # Copy images
    for filename in os.listdir(mask):
        if filename.lower().endswith(('.tif', '.tiff')):
            shutil.copy(os.path.join(mask, filename), mask_images)
            
            if filename in os.listdir(raw):
                shutil.copy(os.path.join(raw, filename), raw_images)
            shutil.copy(os.path.join(mask, filename), mask_images) 
    print ('mask images=',len(mask_images))        
    print ('mask images=',len(mask_images))
    print ('raw_masks',raw_images)
    print ('mask_images',mask_images)
    heavycrop(raw_images) 
    heavycrop(mask_images)
    print(raw_images)
    print(mask_images)
    print('######################################HEAVY CROP DONE ########################')
    #Now creating data
    npycon(raw_images, desti +'/images.npz')
    npycon(mask_images, desti +'/masks.npz')
    print ('dest',desti)
    #Now loading the data
    npz_images,images_names =loadnpz(desti, 'images.npz', p=False)
    npz_labels,labels_names =loadnpz(desti, 'masks.npz', p=False)
    length_npz = len(npz_images)
    
    # Define your percentages
    train_percent = 0.67  # 67%
    val_percent = 0.15  # 15%
    test_percent = 0.18  # 18%
    
    # Calculate indices
    train_index = int(length_npz * train_percent)
    val_index = train_index + int(length_npz * val_percent)
    
    # Split the data
    X_train = npz_images[0:train_index]
    y_train = npz_labels[0:train_index]
    X_val = npz_images[train_index:val_index]
    y_val = npz_labels[train_index:val_index]
    X_test = npz_images[val_index:]
    y_test = npz_labels[val_index:]
    model = model
    batch_size = 50
    epochs = 110
    history = model.fit(X_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_val, y_val))
    #a = generate_random_folder_name()
    weight_path = f'{desti}/{batch_size}_{epochs}_UNET.h5'
    model.save_weights(weight_path)
    return model,weight_path,temp_dest
#     shutil.rmtree(raw_images)
#     shutil.rmtree(mask_images)
    print ('weight_path',weight_path)
    print ('done')
    

def modelpredict(model,weight,temp_folder,orignal_folder,destination=False):
    #Now creating data
    temp_minicrop = generate_random_folder_name()
    temp_heavycrop = generate_random_folder_name()
    
    temp_minicrop=os.path.join(temp_folder, f"{temp_minicrop}")
    temp_heavycrop=os.path.join(temp_folder, f"{temp_heavycrop}")
    
    os.makedirs(temp_minicrop, exist_ok=True)
    os.makedirs(temp_heavycrop, exist_ok=True)
    
    phase.crop_initial_images(orignal_folder,temp_minicrop)
    phase.crop_512x512_tiles(temp_minicrop,temp_heavycrop)
    
    
    
    npycon(temp_heavycrop, temp_heavycrop +'/test.npz')    
    npz_test,npz_names =loadnpz(temp_heavycrop, 'test.npz', p=False)
    
    length_npz = len(npz_test)
    model = unet_load(weight)
    y_pred = model.predict(npz_test)
    for i in range(length_npz):
        z = npz_names[i]
        binary_mask = y_pred[i]
        print(z)
        plt.imshow(binary_mask,cmap='gray')
        plt.show()
        threshold = filters.threshold_otsu(binary_mask) 
        binary_mask = binary_mask > threshold
        temp_512 = generate_random_folder_name()
        temp_514 = generate_random_folder_name()
        tiffle_save = 'tifflesave'         
        if destination:            
            temp_512=os.path.join(temp_folder, f"{temp_512}")
            os.makedirs(temp_512, exist_ok=True)           
            if not os.path.exists(destination):
                os.makedirs(destination, exist_ok=True)               
            temp_new_path = os.path.join(temp_512, z)
            temp_new_path1 = os.path.join(temp_folder, 'tifflesave')

            # Ensure the directory exists before trying to save the file
            os.makedirs(temp_new_path1, exist_ok=True)

            height, width = binary_mask.shape[:2]
            binary_mask = binary_mask.reshape((height, width))

            binary_mask1 = binary_mask.copy()         
            binary_mask = Image.fromarray(binary_mask.astype(np.uint8))
            binary_mask.save(temp_new_path)

            tiffle_save1 = os.path.join(temp_new_path1,z)
            binary_mask.save(tiffle_save1) # this is to create a tif file and then rewrite with tifffile.imsave
            tifffile.imsave(tiffle_save1,binary_mask1)
            os.makedirs(temp_514, exist_ok=True)
            binary_save_2 = os.path.join(destination, temp_514)
    if destination:
        phase.stitch_tiles(temp_512,destination)
        phase.stitch_tiles(temp_new_path1,binary_save_2)
# stitch_tiles(tiffle_save,binary_save_2)
            
        print('tiffle_save_1',tiffle_save1)
        print ('temp_new_path',temp_new_path)
        print ('binary_save_2',binary_save_2)
        print ('destination',destination)
        print ('temp_path',temp_new_path)
        print('1')
    print ('done')
    
#     shutil.rmtree(temp_minicrop)
#     shutil.rmtree(temp_heavycrop)
#     shutil.rmtree(temp_512)