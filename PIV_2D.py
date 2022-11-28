from openpiv import tools, pyprocess, validation, filters, scaling
import numpy as np
import openpiv.filters
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
import cv2 as cv
import h5py
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
import time
import os

def PIV_film_2D (image_dir,im_ref_name,dt, ws,ol, sas,ti,tf, px, im_folder, h5_file):
    os.chdir(image_dir)
    start = time.time()
    #dt = 10
    #ws = 128
    #ol = 32
    #sas = 128
    #ti = 1
    #tf = 200
    #px = np.array([1.76*1e-6,1.76*1e-6]) #Assuming image size is 3.6x2.7 mm
    im1 = cv.imread(im_ref_name+ '{:04d}.tif'.format(1))
    #im_folder = 'VCXU-32M_700002909206_220516-174043'
    args = {'time_resolution': dt, 'window_size': ws, 'overlap_size' : ol, 'sas' : sas, 'first_frame' :ti, 'last_frame' : tf, 'image_file' : im_folder}
    imgray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    x0,y0  = pyprocess.get_coordinates(imgray1.shape,search_area_size=sas, overlap=ol)
    with h5py.File('h5_file.h5', "a") as h5file:
        #prepare the output array on disk
        flow = h5file.require_dataset(
            "FlowField",
            (tf-ti, x0.shape[0],x0.shape[1],2),
            dtype='float32')
        grad_field = h5file.require_dataset(
            "gradient_field",
            (tf-ti, x0.shape[0],x0.shape[1],2,2),
            dtype='float32')
        #save all analysis parameters
        for k,v in args.items():
            flow.attrs[k] = v
        for i in range(ti, tf):
            im1 = cv.imread(im_ref_name + '{:04d}.tif'.format(i))
            imgray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
            intensity_profile1 = imgray1.max()
            imgray1 = imgray1*(255/intensity_profile1)[None,None].astype(np.uint8)
            fil_im1 = gaussian_filter(imgray1-gaussian_filter(imgray1.astype(float), 101),1)
            im2 = cv.imread(im_ref_name + '{:04d}.tif'.format(i+1))
            imgray2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
            fil_im2 = gaussian_filter(imgray2-gaussian_filter(imgray2.astype(float), 101),1)
            fil_im1 = (fil_im1 - fil_im1.min())
            fil_im1 = fil_im1*(fil_im1.max()/255)
            fil_im2 = (fil_im2 - fil_im2.min())
            fil_im2 = fil_im2*(fil_im2.max()/255)
            u, v, sig2noise = pyprocess.extended_search_area_piv(fil_im1,fil_im2, window_size=ws, overlap = ol,dt = dt, search_area_size=sas,sig2noise_method='peak2peak')
            u_f, v_f = openpiv.filters.replace_outliers(u, v, method='localmean', kernel_size=1)
            u_s = gaussian_filter(u_f,1)
            v_s= gaussian_filter(v_f,1)
            x,y  = pyprocess.get_coordinates(fil_im1.shape,search_area_size=sas, overlap=ol)
            for j in range(flow[i-ti].shape[0]):
                for k in range(flow[i-ti].shape[1]):
                    flow[i-ti,j,k] = u_s[j,k]*px[0]*dt, v_s[j,k]*px[1]*dt ### I am storing the displacement and not the velocities
                    #flow[i-ti,j,k]  = gaussian_filter(flow[i-ti,j,k], axis = (0,1),
            grad = np.gradient(flow[i-ti], axis = (0,1),*(px*np.array([ws-ol,ws-ol])))
            gd_maar = np.zeros(( x0.shape[0],x0.shape[1],2,2))
            for j in range(2):
                for k in range(2):
                    gd_maar[..., j,k] = grad[j][...,k]
                    gd_maar = np.roll(gd_maar, shift = 1, axis = 2)
            grad_field[i-ti] = gd_maar
            #fig, axs = plt.subplots(1,1, figsize = (16,12))
            #axs.imshow(fil_im1, origin = 'lower')
            #axs.imshow(np.dstack((scalevalues(fil_im1,0,100),scalevalues(fil_im2,0,100), np.zeros(imgray1.shape)))[...,[0,1,0]], origin = 'lower')
            #axs.quiver(x,y,u_s,v_s,color='white', scale = 0.4, scale_units='inches', width = 0.0005, headwidth = 8)
            #plt.savefig(r'D:\ESPCI\Experiments\Product+PDS\20220428\sprayed_substrate_pink\VCXU-32M_700002909206_220428-110021\piv_w64_ol0_sas_64\t_{:04d}'.format(i))
            print('\r%d (%.01f %%)'%(i, 100*(i-ti)/flow.shape[0]), sep=' ', end=' ', flush=True)
        print("\ndone!")
    end = time.time()
    print(end - start)

## See if git workss
