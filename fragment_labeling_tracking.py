import cv2 as cv
import numpy as np
from skimage import morphology, measure, color, util
from skimage.filters import threshold_multiotsu
import pandas as pd
import matplotlib.pyplot as plt

# Function to preprocess the image
def process_image(image_path, size1, size2, area_threshold):
    # Read image
    im = cv.imread(image_path)
    # Convert to grayscale
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    # Apply Gaussian filter
    fil_im = gaussian_filter(imgray - gaussian_filter(imgray.astype(float), 101), 1)
    # Apply adaptive thresholding
    thresh = cv.adaptiveThreshold(fil_im.astype('uint8'), 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 501, 1)
    return imgray, fil_im, thresh

# Function to detect delaminated area
def detect_delaminated_area(thresh, fil_im, size, area_threshold):
    # Perform binary closing
    thresh_closing = morphology.binary_closing(thresh, np.ones((size, size)))
    # Label connected components
    labeled_image = measure.label(thresh_closing.astype('uint8'), background=1)
    # Extract properties of labeled regions
    props = measure.regionprops_table(labeled_image, fil_im, properties=['label', 'area', 'perimeter', 'eccentricity'])
    props_table = pd.DataFrame(props)
    # Apply area threshold
    threshold_labels = props_table['label'] * (props_table['area'] > area_threshold)
    # Map labels
    new_labels = map_array(labeled_image, np.asarray(props_table['label']), np.asarray(threshold_labels))
    # Generate overlay image
    return label2rgb(new_labels, fil_im)

# Function to detect laminated area
def detect_laminated_area(fil_im, size, area_threshold):
    # Perform multi-level Otsu thresholding
    thresh1 = threshold_multiotsu(fil_im, classes=3)
    # Digitize image based on thresholds
    regions = 2 + util.invert(np.digitize(fil_im, thresh1))
    # Perform binary opening
    thresh_opening = morphology.binary_opening(regions, np.ones((size, size)))
    # Label connected components
    labeled_image = measure.label(thresh_opening.astype('uint8'), background=0)
    # Extract properties of labeled regions
    props = measure.regionprops_table(labeled_image, fil_im, properties=['label', 'area', 'perimeter', 'eccentricity'])
    props_table = pd.DataFrame(props)
    # Apply area threshold
    threshold_labels = props_table['label'] * (props_table['area'] > area_threshold)
    # Map labels
    new_labels = map_array(labeled_image, np.asarray(props_table['label']), np.asarray(threshold_labels))
    # Generate overlay image
    return label2rgb(new_labels, fil_im)

# Main function
def main():
    # Define start and end frames
    imfrag = 100
    imstart = 102
    imend = 106 
    # Generate image paths
    image_paths = [r'D:\ESPCI\Experiments\Product+PDS\20220516\VCXU-32M_700002909206_220516-174043\20220516_t_z{:04d}.tif'.format(image) for image in [imstart, imend]]
    # Define parameters
    size1 = 11
    size2 = 21
    area_threshold = 5000
    
    # Create figure and axis objects
    fig, axs = plt.subplots(2, 3, figsize=(24, 15))
    
    # Iterate over each image
    for i, image_path in enumerate(image_paths):
        # Preprocess the image
        imgray, fil_im, thresh = process_image(image_path, size1, size2, area_threshold)
        
        # Determine the size based on frame number
        if i < imfrag:
            size = size2
            im_overlay_1 = detect_delaminated_area(thresh, fil_im, size, area_threshold)
        else:
            size = size1
            im_overlay_1 = detect_delaminated_area(thresh, fil_im, size, area_threshold)
            
        # Detect laminated area
        im_overlay_2 = detect_laminated_area(fil_im, size2, area_threshold)
        
        # Display images
        axs[i, 0].imshow(imgray, origin='lower')
        axs[i, 1].imshow(im_overlay_1, origin='lower')
        axs[i, 2].imshow(im_overlay_2, origin='lower')
        
        # Turn off axes
        for ax in axs.flatten():
            ax.axis('off')
        
    # Set titles
    axs[0, 1].set_title("delaminated", fontsize=18)
    axs[0, 2].set_title("laminated", fontsize=18)
    
    # Adjust layout
    plt.tight_layout()
    # Save figure
    plt.savefig('labeled_image.pdf')

if __name__ == "__main__":
    main()
