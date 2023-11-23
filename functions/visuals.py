import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np



from functions.modules import *


def imagescomparisions(raw, groundtruth, phase1, phase2, phase3):
    folders = [raw, groundtruth, phase1, phase2, phase3]
    folder_names = ['Raw', 'Groundtruth', 'Phase-1 Results', 'Phase-2 Results', 'Phase-3 Results']
    
    # Filter out the False folders
    folders = [(folder, name) for folder, name in zip(folders, folder_names) if folder]
    
    # Assuming all folders have the same number of images with the same names
    image_names = [name for name in os.listdir(groundtruth) if name.endswith('.tif') or name.endswith('.tiff')]
    
    for image_name in image_names:
        fig, axs = plt.subplots(1, len(folders), figsize=(60, 40))
        
        for i, (folder, name) in enumerate(folders):
            try:
                image_path = os.path.join(folder, image_name)
                image = Image.open(image_path).convert('L')  # Convert image to grayscale
            except FileNotFoundError:
                # If the image file is not found, create a black image of the same size
                image = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))  # Adjust the size as needed
            
            axs[i].imshow(image, cmap='gray')
            axs[i].set_title(name, fontsize=40)  # Increase the font size
            axs[i].axis('off')
            
        plt.show()


import pandas as pd
import matplotlib.pyplot as plt

def plotsvisualisation(data1, data2, data3=None):
    # Assuming 'data1' and 'data2' are your DataFrames for the two phases
    data1 = pd.read_csv(data1)
    data2 = pd.read_csv(data2)

    # Calculate the mean and standard deviation for each phase
    mean1 = data1.mean()
    std1 = data1.std()
    mean2 = data2.mean()
    std2 = data2.std()

    # If data3 is provided, calculate its mean and standard deviation
    if data3 is not None:
        data3 = pd.read_csv(data3)
        mean3 = data3.mean()
        std3 = data3.std()

    # Create a DataFrame for the mean and standard deviation values
    stats = pd.DataFrame({
        'Phase 1': pd.concat([mean1, std1], keys=['Mean', 'Std. Dev']),
        'Phase 2': pd.concat([mean2, std2], keys=['Mean', 'Std. Dev']),
        'Phase 3': pd.concat([mean3, std3], keys=['Mean', 'Std. Dev']) if data3 is not None else None
    })

    # Print the DataFrame in a table format
    print(stats.to_markdown())

    # Create a bar plot for each metric
    metrics = ['Pixel Accuracy', 'IoU', 'Dice']
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        plt.bar(data1['Slice'], data1[metric], color='b', alpha=0.5, label='Phase 1')
        plt.bar(data2['Slice'], data2[metric], color='r', alpha=0.5, label='Phase 2')
        if data3 is not None:
            plt.bar(data3['Slice'], data3[metric], color='g', alpha=0.5, label='Phase 3')
        plt.xlabel('Slice')
        plt.ylabel(metric)
        plt.legend()
        plt.show()
    
    # Create a line plot for each metric
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        plt.plot(data1['Slice'], data1[metric], color='b', marker='o', label='Phase 1')
        plt.plot(data2['Slice'], data2[metric], color='r', marker='o', label='Phase 2')
        if data3 is not None:
            plt.plot(data3['Slice'], data3[metric], color='g', marker='o', label='Phase 3')
        plt.xlabel('Slice')
        plt.ylabel(metric)
        plt.legend()
        plt.show()

    # Create an overview plot for all metrics and phases
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        plt.plot(data1['Slice'], data1[metric], marker='o', label=f'Phase 1 - {metric}')
        plt.plot(data2['Slice'], data2[metric], marker='o', label=f'Phase 2 - {metric}')
        if data3 is not None:
            plt.plot(data3['Slice'], data3[metric], marker='o', label=f'Phase 3 - {metric}')
    plt.xlabel('Slice')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(csv_files):
    phases = ['CV', 'UNET', 'PP']
    metrics = ['Pixel Accuracy', 'IoU', 'Dice']
    
    fig, ax = plt.subplots()
    
    for i, csv_file in enumerate(csv_files):
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Calculate the mean and standard deviation of the metrics
        means = df[metrics].mean()
        stds = df[metrics].std()
        
        # Create the bar plot for the means
        x = np.arange(len(metrics))
        ax.bar(x + i/len(csv_files), means, yerr=stds, width=1/len(csv_files), align='edge', label=phases[i])
        
        # Create the line plot for the means
        ax.plot(x + (i+0.5)/len(csv_files), means, color='black')
        
    ax.set_xticks(x + 0.5)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    plt.show()

# Call the function with the paths to your CSV files
plot_metrics(['path_to_csv_file1', 'path_to_csv_file2', 'path_to_csv_file3'])

