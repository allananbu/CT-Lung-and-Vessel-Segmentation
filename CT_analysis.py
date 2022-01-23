# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:53:43 2022

@author: Allan
"""

import os
import shutil

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from skimage import measure

import glob
import csv


#Functions
#display CT image
def show_slice(img_slice):
    plt.figure()
    plt.imshow(img_slice.T,cmap='gray',origin='lower')

#Display image based on windowing around a particular intensity level
def slice_window(slice,level,window):
    maxi=level+window/2
    mini=level-window/2
    slice=slice.clip(mini,maxi)
    plt.figure()
    plt.imshow(slice.T,cmap='gray',origin='lower')

#Define the path of the images
basepath='./Images/slice*.nii.gz'
paths=sorted(glob.glob(basepath))

for c,exam_path in enumerate(paths):
    ct_img=nib.load(exam_path)
    ct_np=ct_img.get_fdata()
    
if c==1:
    show_slice(ct_np)
    slice_window(ct_np,50,350)
    slice_window(ct_np,-200,2000)

def find_pixdim(ct_img):
    pix_dim=ct_img.header["pixdim"]
    dim=ct_img.header["dim"]
    max_indx=np.argmax(dim)
    pixdimX=pix_dim[max_indx]
    dim=np.delete(dim,max_indx)
    pix_dim=np.delete(pix_dim,max_indx)
    max_indy=np.argmax(dim)
    pixdimY=pix_dim[max_indy]
    return [pixdimX,pixdimY]    

def intensity_seg(ct_img_np,min,max):
    clipped=ct_img_np.clip(min,max)
    clipped[clipped!=max]=1
    clipped[clipped==max]=0
    return measure.find_contours(clipped,0.95)

def set_is_closed(contour):#if area of contour is less than 1 contour is closed
    if contour_distance(contour)<1:
        return True
    else:
        False
        
def contour_distance(contour):
    dx=contour[0,1]-contour[-1,1]
    dy=contour[0,0]-contour[-1,0]
    return np.sqrt(np.power(dx,2)+np.power(dy,2))
    
def find_lungs(contours):
    body_lung_contour=[]
    vol_contour=[]
    for contour in contours:
        hull=ConvexHull(contour)
        
        if hull.volume>2000 and set_is_closed(contour):
            body_lung_contour.append(contour)
            vol_contour.append(hull.volume)
    
    if len(body_lung_contour)==2:
        return body_lung_contour
    elif len(body_lung_contour)>2:
        vol_contour,body_lung_contour=(list(t) for t in zip(*sorted(zip(vol_contour,body_lung_contour))))
        body_lung_contour.pop(-1)
        return body_lung_contour

def show_contour(image,contours):
    fig,ax=plt.subplots()
    ax.imshow(image.T,cmap=plt.cm.gray)
    
    for contour in contours:
        ax.plot(contour[:,0],contour[:,1],linewidth=1)
    
    ax.set_xticks([])
    ax.set_yticks([])
    #plt.show()

def create_mask_from_polygon(image,contours):
    lung_mask=np.array(Image.new('L',image.shape,0))
    
    for contour in contours:
        x=contour[:,0]
        y=contour[:,1]
        polygon_tuple=list(zip(x,y))
        img=Image.new('L',image.shape,0)
        ImageDraw.Draw(img).polygon(polygon_tuple,outline=0,fill=1)
        mask=np.array(img)
        lung_mask+=mask
    
    lung_mask[lung_mask>1]=1
    
    return lung_mask.T

def compute_area(mask,pixdim):
    mask[mask>=1]=1
    lung_pixels=np.sum(mask)
    return lung_pixels*pixdim[0]*pixdim[1]

def save_nifty(img_np,name,affine):
    img_np[img_np==1]=255
    ni_img=nib.Nifti1Image(img_np,affine)
    nib.save(ni_img,name+'.nii.gz')

def create_vessel_mask(mask,image,denoise=False):
    vessels=mask*image
    vessels[vessels==0]=-1000
    vessels[vessels>=-500]=1
    vessels[vessels<-500]=0
    show_slice(vessels)
    
    if denoise:
        return denoise_vessels(lungs,vessels)
    show_slice(vessels)
    
    return vessels

def denoise_vessels(lungs,vessels):
    ves_x,ves_y=np.nonzero(vessels)
    
    for contour in lungs:
        x_pts=contour[:,0]
        y_pts=contour[:,1]
        
        for co_x,co_y in zip(ves_x,ves_y):
            for x,y in zip(x_pts,y_pts):
                d=euclidean_dist(x-co_x,y-co_y)
                if d<=0.1:
                    vessels[co_x,co_y]=0
    return vessels
def euclidean_dist(dx,dy):
    return np.sqrt(np.power(dx,2)+np.power(dy,2))

def overlay_plot(image,mask):
    plt.figure()
    plt.imshow(image.T,interpolation='none')
    plt.imshow(mask.T,'jet',interpolation='none',alpha=0.5)
    #plt.show()

#Lung Contour segmentation
outpath='./Lungs/'
contour_path='./Contours/'
myFile=open('lung_volume.csv','w')

for c,exam_path in enumerate(paths):
    img_name=exam_path.split("/")[-1].split('.nii')[0]
    out_mask_name = outpath + img_name + "_mask"
    ct_img=nib.load(exam_path)
    
    pix_dim=find_pixdim(ct_img)
    ct_img_np=ct_img.get_fdata()
    
    contours=intensity_seg(ct_img_np,min=-1000,max=-300)
    
    lungs=find_lungs(contours)
    show_contour(ct_img_np,lungs)
    lung_mask=create_mask_from_polygon(ct_img_np,lungs)
    
    lung_area=compute_area(lung_mask,find_pixdim(ct_img))
    
    #Lung vessel segmentation
    vessels_only=create_vessel_mask(lung_mask,ct_img_np,denoise=True)
    #save_nifty(lung_mask,out_mask_name,ct_img.affine) #saves to nifti image format
    overlay_plot(ct_img_np,vessels_only)
