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
import argparse

def scalevalues(im, m=None, M=None):
    if m is None:
        m = im.min()
    if M is None:
        M = im.max()
    return (im-m)/(M-m)

if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Apply PIV between two stack of images at different timelapse')
    parser.add_argument('image_dir', help = 'Location of the image directory')
    parser.add_argument('im_ref_name',  help = 'Name of the mage files without the numbering')
    parser.add_argument('h5_file', type = str, help ='name of the h5py file to store the data')
    parser.add_argument('im_folder', type = str, help = 'The folder in which to save the image')
    parser.add_argument('image', type = bool, help='Boolean to specify if the PIV should output the plot of image with quivers')
    parser.add_argument('--dt', '--time_step', default=10, type=float, help  = 'the time gap between two images from the camera')
    parser.add_argument('--ti', '--time_initial', default=1, type=int, help  = 'the first time frame for applying the PIV')
    parser.add_argument('--tf', '--time_final', default=100, type=int, help  = 'the first time frame for applying the PIV')
    parser.add_argument('--gauss_ns', '--noise_gaussian', default=1, type=int, help  = 'the gaussian filter size to remoove the noise')
    parser.add_argument('--gauss_bg', '--background_gaussian', default=101, type=int, help  = 'the gaussian filter size to remoove the background')
    parser.add_argument('--ws','--window_size',default=96, type=int, help = 'size  of the PIV window in the first image')
    parser.add_argument('--ol','--overlap',default=48, type=int,  help='number of pixel that can be overlapped between 2 windows')
    parser.add_argument('--sas','--search_area_size',type=int, default=192,
    help=' The area in which the window is allowed for image correlation in the 2nd image')
    parser.add_argument('--px','--pixel_size',  nargs="+", default = [1.76*1e-6,1.76*1e-6],
    help = ' The pixel dimension assuming its equal in x and y')
    args=parser.parse_args()

    os.chdir(args.image_dir)
    #print(os. getcwd())
    start = time.time()
    im1 = cv.imread(args.im_ref_name+ '{:04d}.tif'.format(1))
    args_h5 = {'time_resolution': args.dt, 'window_size': args.ws, 'overlap_size' : args.ol, 'sas' : args.sas, 'first_frame' :args.ti,
    'last_frame' : args.tf, 'image_file' : args.im_folder}
    imgray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    x0,y0  = pyprocess.get_coordinates(imgray1.shape,search_area_size=args.sas, overlap=args.ol)

    with h5py.File(args.h5_file, "a") as h5file:
        #prepare the output array on disk
        flow = h5file.require_dataset(
            "FlowField",
            (args.tf-args.ti, x0.shape[0],x0.shape[1],2),
            dtype='float32')
        grad_field = h5file.require_dataset(
            "gradient_field",
            (args.tf-args.ti, x0.shape[0],x0.shape[1],2,2),
            dtype='float32')
        #save all analysis parameters
        for k,v in args_h5.items():
            flow.attrs[k] = v

        for i in range(args.ti, args.tf):
            im1 = cv.imread(args.im_ref_name + '{:04d}.tif'.format(i))
            imgray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
            intensity_profile1 = imgray1.max()
            imgray1 = imgray1*(255/intensity_profile1)[None,None].astype(np.uint8)
            fil_im1 = gaussian_filter(imgray1-gaussian_filter(imgray1.astype(float), args.gauss_bg),args.gauss_ns)
            im2 = cv.imread(args.im_ref_name + '{:04d}.tif'.format(i+1))
            imgray2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
            fil_im2 = gaussian_filter(imgray2-gaussian_filter(imgray2.astype(float), args.gauss_bg),args.gauss_ns)
            fil_im1 = (fil_im1 - fil_im1.min())
            fil_im1 = fil_im1*(fil_im1.max()/255)
            fil_im2 = (fil_im2 - fil_im2.min())
            fil_im2 = fil_im2*(fil_im2.max()/255)
            u, v, sig2noise = pyprocess.extended_search_area_piv(fil_im1,fil_im2, window_size=args.ws, overlap = args.ol,dt = args.dt,
            search_area_size=args.sas,sig2noise_method='peak2peak')
            u_f, v_f = openpiv.filters.replace_outliers(u, v, method='localmean', kernel_size=1)
            u_s = gaussian_filter(u_f,1)
            v_s= gaussian_filter(v_f,1)
            x,y  = pyprocess.get_coordinates(fil_im1.shape,search_area_size=args.sas, overlap=args.ol)
            for j in range(flow[i-args.ti].shape[0]):
                for k in range(flow[i-args.ti].shape[1]):
                    flow[i-args.ti,j,k] = u_s[j,k]*args.px[0]*args.dt, v_s[j,k]*args.px[1]*args.dt ### I am storing the displacement and not the velocities
                    #flow[i-ti,j,k]  = gaussian_filter(flow[i-ti,j,k], axis = (0,1),
            grad = np.gradient(flow[i-args.ti], axis = (0,1),*(args.px*np.array([args.ws-args.ol,args.ws-args.ol])))
            gd_maar = np.zeros(( x0.shape[0],x0.shape[1],2,2))
            for j in range(2):
                for k in range(2):
                    gd_maar[..., j,k] = grad[j][...,k]
                    gd_maar = np.roll(gd_maar, shift = 1, axis = 2)
            grad_field[i-args.ti] = gd_maar
            if args.image == True:
                fig, axs = plt.subplots(1,1, figsize = (16,12))
                axs.imshow(fil_im1, origin = 'lower')
                axs.quiver(x,y,u_s,v_s,color='white', scale = 0.4, scale_units='inches', width = 0.0005, headwidth = 8)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(args.im_folder + 't_{:04d}.tif'.format(i))
                plt.close()
            print('\r%d (%.01f %%)'%(i, 100*(i-args.ti)/flow.shape[0]), sep=' ', end=' ', flush=True)
        print("\ndone!")
    end = time.time()
    print(end - start)
