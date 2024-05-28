### This program is for picking two point or object manually by clicking in an image and saving the points in a txt file.
### The code iterates over multiple image file and hence we can have a txt file  containing position of the object/points tracked over time
### This can be used to measure global strain in region of interest
### An improvement of this code can be using machine learning tools to detect the object. However, it did not work in the first attempt

from datetime import datetime
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
import math
import cv2 as cv
import time
import os
import argparse

def track_points(image_path_template, ti, tf, delt, delI, scale_percent, output_file):
    num_images = int((tf - ti) / delI)

    # Variables to store the selected points
    points_1 = np.zeros((num_images + 1, 2))
    points_2 = np.zeros((num_images + 1, 2))
    point = []

    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONUP:
            point.append((x, y))
            if len(point) >= 2:
                cv.destroyAllWindows()

    # Iterate over the image series, taking every delI-th image
    for i, k in enumerate(np.arange(ti, tf, delI)):
        # Load the image
        image_path = image_path_template.format(k + 1)
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Failed to load image at {image_path}")
            continue

        # Resize the image to fit within the window
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv.resize(image, dim, interpolation=cv.INTER_AREA)

        # Draw circles for the previously selected points
        if i > 0:
            cv.circle(image, tuple(map(int, points_1[i - 1])), 1, (0, 255, 0), -1)  # Green circle
            cv.circle(image, tuple(map(int, points_2[i - 1])), 1, (0, 0, 255), -1)  # Red circle

        # Create a window and bind the mouse callback function
        window_name = 'Select Points_{}'.format(i)
        cv.namedWindow(window_name)
        cv.setMouseCallback(window_name, mouse_callback)

        # Display the image and wait for points to be selected
        cv.imshow(window_name, image)
        cv.waitKey(0)

        # Save the coordinates
        points_1[i] = point[0]
        points_2[i] = point[1]
        
        # Reset the points list for the next image
        point = []

        # Print the mean coordinates
        print(f"The x points for image {i + 1}: ({points_1[i][0]}, {points_2[i][0]})")

    # Save the points to a file
    np.savetxt(output_file, np.column_stack((points_1, points_2)))
    print(f"Points saved to {output_file}")

# Example usage
image_path_template = r"D:\ESPCI\Experiments\mechanoconfocal\20231020\20231019_sbr_r2o2_2.5_400_60_min\VCXU-32M_700002909206_231020-163611\sample_20231019_sbrnew_2.5r2o2_film400um_pinkpdms_60min_1{:04d}.tif"
ti = 190
tf = 525
delt = 1
delI = 3
scale_percent = 50
output_file = r"D:\ESPCI\Experiments\mechanoconfocal\20231020\20231019_points_tracking_sbr_r2o2_400_60_min_point__subs_zone_5_1.txt"

###track_points(image_path_template, ti, tf, delt, delI, scale_percent, output_file)


