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

def load_and_resize_image(image_path, scale_percent=50):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def draw_previous_points(image, points_1, points_2, i):
    if i > 0:
        cv2.circle(image, tuple(map(int, points_1[i - 1])), 1, (0, 255, 0), -1)  # Green circle
        cv2.circle(image, tuple(map(int, points_2[i - 1])), 1, (0, 0, 255), -1)  # Red circle

def select_points(image):
    points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONUP:
            points.append((x, y))
            if len(points) >= 2:
                cv2.destroyAllWindows()

    cv2.namedWindow('Select Points')
    cv2.setMouseCallback('Select Points', mouse_callback)

    cv2.imshow('Select Points', image)
    cv2.waitKey(0)

    return np.array(points)

def process_images(image_path, ti, tf, delI, save_path, scale_percent=50):
    num_images = int((tf - ti) / delI)

    points_1 = np.zeros((num_images + 1, 2))
    points_2 = np.zeros((num_images + 1, 2))

    for i, k in enumerate(np.arange(ti, tf, delI)):
        image_path = image_path.format(k + 1)

        image = load_and_resize_image(image_path, scale_percent)
        draw_previous_points(image, points_1, points_2, i)

        selected_points = select_points(image)

        points_1[i] = selected_points[0]    
        points_2[i] = selected_points[1]

        # Print the x coordinates
        print("The x points for image {}: ({}, {})".format(i + 1, points_1[i][0], points_2[i][0]))

    np.savetxt(save_path, np.column_stack((points_1, points_2)))

### Example usage
image_path = r"D:\ESPCI\Experiments\mechanoconfocal\20231020\20231019_sbr_r2o2_2.5_400_60_min\VCXU-32M_700002909206_231020-163611\sample_20231019_sbrnew_2.5r2o2_film400um_pinkpdms_60min_1{:04d}.tif"
save_path = r"D:\ESPCI\Experiments\mechanoconfocal\20231020\20231019_sbr_r2o2_2.5_400_60_min\20231019_points_tracking_sbr_r2o2_400_60_min_point__subs_zone_5_1.txt"
process_images(image_path = image_path, ti=190, tf=525, delI=3, save_path= save_path)
