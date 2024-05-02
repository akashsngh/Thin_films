import cv2 as cv
import numpy as np
from skimage import morphology, measure, color, util
from skimage.filters import threshold_multiotsu
import pandas as pd
import matplotlib.pyplot as plt

def process_image(image_path, size1, size2, area_threshold):
    im = cv.imread(image_path)
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    fil_im = gaussian_filter(imgray - gaussian_filter(imgray.astype(float), 101), 1)
    thresh = cv.adaptiveThreshold(fil_im.astype('uint8'), 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 501, 1)
    return imgray, fil_im, thresh

def detect_delaminated_area(thresh, fil_im, size, area_threshold):
    thresh_closing = morphology.binary_closing(thresh, np.ones((size, size)))
    labeled_image = measure.label(thresh_closing.astype('uint8'), background=1)
    props = measure.regionprops_table(labeled_image, fil_im, properties=['label', 'area', 'perimeter', 'eccentricity'])
    props_table = pd.DataFrame(props)
    threshold_labels = props_table['label'] * (props_table['area'] > area_threshold)
    new_labels = map_array(labeled_image, np.asarray(props_table['label']), np.asarray(threshold_labels))
    return label2rgb(new_labels, fil_im)

def detect_laminated_area(fil_im, size, area_threshold):
    thresh1 = threshold_multiotsu(fil_im, classes=3)
    regions = 2 + util.invert(np.digitize(fil_im, thresh1))
    thresh_opening = morphology.binary_opening(regions, np.ones((size, size)))
    labeled_image = measure.label(thresh_opening.astype('uint8'), background=0)
    props = measure.regionprops_table(labeled_image, fil_im, properties=['label', 'area', 'perimeter', 'eccentricity'])
    props_table = pd.DataFrame(props)
    threshold_labels = props_table['label'] * (props_table['area'] > area_threshold)
    new_labels = map_array(labeled_image, np.asarray(props_table['label']), np.asarray(threshold_labels))
    return label2rgb(new_labels, fil_im)

def main():
    image_paths = [r'D:\ESPCI\Experiments\Product+PDS\20220516\VCXU-32M_700002909206_220516-174043\20220516_t_z{:04d}.tif'.format(image) for image in [102, 106]]
    size1 = 11
    size2 = 21
    area_threshold = 5000
    
    fig, axs = plt.subplots(2, 3, figsize=(24, 15))
    
    for i, image_path in enumerate(image_paths):
        imgray, fil_im, thresh = process_image(image_path, size1, size2, area_threshold)
        
        if i < 100:
            size = size2
            im_overlay_1 = detect_delaminated_area(thresh, fil_im, size, area_threshold)
        else:
            size = size1
            im_overlay_1 = detect_delaminated_area(thresh, fil_im, size, area_threshold)
            
        im_overlay_2 = detect_laminated_area(fil_im, size2, area_threshold)
        
        axs[i, 0].imshow(imgray, origin='lower')
        axs[i, 1].imshow(im_overlay_1, origin='lower')
        axs[i, 2].imshow(im_overlay_2, origin='lower')
        
        for ax in axs.flatten():
            ax.axis('off')
        
    axs[0, 1].set_title("delaminated", fontsize=18)
    axs[0, 2].set_title("laminated", fontsize=18)
    
    plt.tight_layout()
    plt.savefig('labeled_image.pdf')

if __name__ == "__main__":
    main()
