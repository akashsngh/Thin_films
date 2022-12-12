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



def disp_grad_2D_time(im_ref_name,dt,ws,ol, sas,ti,tf, im_folder, image, gauss_bg, gauss_ns, px, cut = False):
    # The function generates the displacement field from PIV for consecutive image of a timelapse and later assuming small strain approximation generates the 2D strain field
    ## Initialize the h5py file which generates the specified memory in the disk
    ### Picking a sample image to extract the size of the file
    im1 = cv.imread(im_ref_name+ '{:04d}.png'.format(1))
    args_h5 = {'time_resolution': dt, 'window_size': ws, 'overlap_size' : ol, 'sas' : sas, 'first_frame' :ti,
    'last_frame' : tf, 'image_file' : im_folder}
    imgray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    if cut == True:
        xlim1 = input("The x coordinate of the first point: ")
        ylim1 = input("The y coordinate of the first point: ")
        xlim2 = input("The x coordinate of the second point: ")
        ylim2 = input("The y coordinate of the second point: ")
        #plt.imshow(imgray1)
        #points = plt.ginput(2)
        #plt.close()
        #xlim1, ylim1, xlim2, ylim2 = np.array(points).ravel()
        print('Point1 is {},{}; Point 2 is {},{}'.format(xlim1, ylim1, xlim2, ylim2))
        imgray1 = imgray1[int(ylim1):int(ylim2),int(xlim1):int(xlim2)]
    x0,y0  = pyprocess.get_coordinates(imgray1.shape,search_area_size=sas, overlap=ol)

    with h5py.File(args.h5_file, "a") as h5file:
        flow = h5file.require_dataset(
            "FlowField",
            (tf-ti, x0.shape[0],x0.shape[1],2),
            dtype='float32')
        grad_field = h5file.require_dataset(
            "gradient_field",
            (tf-ti, x0.shape[0],x0.shape[1],2,2),
            dtype='float32')
        #save all analysis parameters
        for k,v in args_h5.items():
            flow.attrs[k] = v

        ### Run PIV and store the PIV displacments in the flowfield PIV
        for i in range(ti, tf):
            im1 = cv.imread(im_ref_name + '{:04d}.png'.format(i))
            imgray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
            intensity_profile1 = imgray1.max()
            imgray1 = imgray1*(255/intensity_profile1)[None,None].astype(np.uint8)
            fil_im1 = gaussian_filter(imgray1-gaussian_filter(imgray1.astype(float), gauss_bg),gauss_ns)
            im2 = cv.imread(im_ref_name + '{:04d}.png'.format(i+1))
            imgray2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
            fil_im2 = gaussian_filter(imgray2-gaussian_filter(imgray2.astype(float), gauss_bg),gauss_ns)
            fil_im1 = (fil_im1 - fil_im1.min())
            fil_im1 = fil_im1*(fil_im1.max()/255)
            fil_im2 = (fil_im2 - fil_im2.min())
            fil_im2 = fil_im2*(fil_im2.max()/255)

            if cut== True:
                fil_im1 = fil_im1[int(ylim1):int(ylim2),int(xlim1):int(xlim2)]
                fil_im2 = fil_im2[int(ylim1):int(ylim2),int(xlim1):int(xlim2)]


            u, v, sig2noise = pyprocess.extended_search_area_piv(fil_im1,fil_im2, window_size=ws, overlap = ol,dt = dt,
            search_area_size=sas,sig2noise_method='peak2peak')
            u_f, v_f = openpiv.filters.replace_outliers(u, v, method='localmean', kernel_size=1)
            u_s = gaussian_filter(u_f,1)
            v_s= gaussian_filter(v_f,1)
            x,y  = pyprocess.get_coordinates(fil_im1.shape,search_area_size=sas, overlap=ol)
            for j in range(flow[i-ti].shape[0]):
                for k in range(flow[i-ti].shape[1]):
                    flow[i-ti,j,k] = u_s[j,k]*px[0]*dt, v_s[j,k]*px[1]*dt

            ### Calculate the gradient
            grad = np.gradient(flow[i-args.ti], axis = (0,1),*(args.px*np.array([args.ws-args.ol,args.ws-args.ol])))
            gd_maar = np.zeros(( x0.shape[0],x0.shape[1],2,2)) ### Added to adjust to the right tensor coordinates in x y
            for j in range(2):
                for k in range(2):
                    gd_maar[..., j,k] = grad[j][...,k]
                    gd_maar = np.roll(gd_maar, shift = 1, axis = 2)
            grad_field[i-args.ti] = gd_maar

            ### Plotting the quiver overlayed on the image
            if args.image == True:
                fig, axs = plt.subplots(1,1, figsize = (16,12))
                axs.imshow(fil_im2, origin = 'lower')
                axs.quiver(x,y,u_s,v_s,color='white', scale = 0.1, scale_units='inches', width = 0.0005, headwidth = 8)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(args.im_folder + 't_{:04d}.tif'.format(i))
                plt.close()
            print('\r%d (%.01f %%)'%(i, 100*(i-args.ti)/flow.shape[0]), sep=' ', end=' ', flush=True)
        print("\ndone!")


def total_elastic_strain_energy(h5_dir, h5files, ti, tf, Ef, px, imname, plot = True):
    os.chdir(h5_dir)
    energy_released = np.zeros((tf-ti, len(h5files)))
    area_total = np.zeros((tf-ti, len(h5files)))
    net_displacment_area = np.zeros((tf-ti, len(h5files)))
    labels = np.zeros((len(h5files)))
    for j in range(0,energy_released.shape[0]):
        for i, file in enumerate(h5files):
            with h5py.File(file,'r') as h5file:
                ls = list(h5file.keys())
                data = h5file.get('FlowField')
                atts = dict(data.attrs)
                ws = atts['window_size']
                ol = atts['overlap_size']
                labels[i] = (ws-ol)*px[0]*1e3
                grad_array =h5file.get('gradient_field')[j]#[t1-1:t1-1+j]
                #print(grad_array)
                strain_field = 0.5*(grad_array+grad_array.transpose(0,1,3,2))
                flowfield_array = h5file.get('FlowField')[j] #Flowfields at the jth time to get the net displacment beetween j-1th and jth frame
                norm_strain = np.linalg.norm(strain_field, axis=(-2, -1))
                #print(norm_strain.shape)
                energy_released[j,i] = 1.33*0.5*Ef*(((ws-ol)**2)*((px[0])**2))*(norm_strain**2).sum()
                area_total[j-1, i] = (((ws-ol)**2)*((px[0])**2))*np.ones(norm_strain.shape).sum()
    if plot==True:
        fig, axs = plt.subplots(1,1, figsize = (12,8))
        for i in range(energy_released.shape[0]):
            axs.plot(labels, energy_released[i,:], '-o', label = 't = {}, {} s'.format((ti+i)*10,(ti+i+1)*10))
        axs.set_yscale('log')
        axs.set_xlabel(r'resolution (mm) ', fontsize = 18) #($\mu$m)
        axs.set_ylabel(r'0.5$E_f$$\sum$|$\epsilon^2$|$A_{grid}$ (J/m)', fontsize = 18)
        axs.tick_params(axis="x", labelsize=16)
        axs.tick_params(axis="y", labelsize=16)
        axs.legend(fontsize = 16)
        plt.tight_layout()
        plt.savefig(imname)

if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Apply PIV between two stack of images at different timelapse')
    parser.add_argument('image_dir', help = 'Location of the image directory')
    parser.add_argument('im_ref_name',  help = 'Name of the mage files without the numbering')
    parser.add_argument('h5_file', type = str, help ='name of the h5py file to store the data')
    parser.add_argument('im_folder', type = str, help = 'The folder in which to save the image')
    parser.add_argument('image', type = bool, help='Boolean to specify if the PIV should output the plot of image with quivers')
    parser.add_argument('--cut', type = bool, help='Boolean to specify if the the deformation has to be mapped in a maller region of the image which can be choosen from the pop up image')
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
    start = time.time()
    disp_grad_2D_time(im_ref_name = args.im_ref_name,dt = args.dt,ws = args.ws,ol = args.ol, sas = args.sas,ti = args.ti,tf = args.tf, im_folder = args.im_folder, image = args.image, gauss_bg = args.gauss_bg, gauss_ns = args.gauss_ns, px = args.px, cut = args.cut)
    end = time.time()
    print(end - start)
