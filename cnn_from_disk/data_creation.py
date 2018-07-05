import pandas as pd
import numpy as np
from PIL import Image
import os
from .proc_utils import *


def split_image_and_save(idf,row,file_col,
                         show=False,save_directory=None,box_wh=(30,30),resize_img=None,
                        v=False,strides=(30,30)):
    """
    Purpose: Given a dataframe load an image and extract windows of size box_wh. 
    
    idf: Dataframe that has in its columns the location of the image file and 
        a column that defines the center of the objects.
    row: row of the DataFrame that will be processed
    file_col: column name in the idf that has the full path to the image file.
    show: set to True if you want to plot with plt.imshow() the images that are beinc created.
    save_directory: Specify the directory where you want to save the images.
        There is no validation if the directory exist.
    box_wh: The box width and height that holds each object and also the window dimensions
        that will be extracted from the original image.
    resize_img: specify if the original image (file_col) will be resized prior the extraction 
        of windows
    v: Verbosity, set to True if you want to see the progress.
    strides: define how many pixels for the X and Y dimensions will be skipped to draw each window.
    
    """
    test_row = idf.iloc[row]
    file_name = test_row[file_col]
    image_name = file_name.split('/')[-1]
    
    timg = get_np_image(file_name,resize=resize_img)
    w,h = box_wh
    
    ylim,xlim = timg.shape[0],timg.shape[1]
    
    limsx, limsy = [],[]
    
    samplex = 0
    for ys in range(0,ylim,strides[1]):
        ye = ys+h
        for xs in range(0,xlim,strides[0]):
            xe = xs + w
            if (xe < xlim) and (ye < ylim):
                img_box = timg[ys:ye,xs:xe]
                if show==True:
                    plt.imshow(img_box)
                    plt.show()
                if type(save_directory)==str:
                    if save_directory[-1]!='/':
                        save_directory+='/'
                    output_format = image_name.split('.')[-1]
                    save_name = save_directory+image_name+'_box_'+str(samplex)+'_'+str(w)+"x"+str(h)+"_"+str(xs)+"x"+str(ys)+"."+output_format
                    pil_img = Image.fromarray(img_box)
                    pil_img.save(save_name)
                    samplex+=1
                    if v==True:
                        print("Saved:",save_name)


def point_in_limits(points,limsx,limsy,final='or'):
    """
    Purpose: Tell if a square defined by 4 points overlaps a list of squares defined by 4 points.
        Returns the number of coincidences. Returns zero if there is no coincidence.
    points: List with a structure of sx,ex,sy,ey. 
        sx: start X pos
        ex: end X pos
        sy: start Y pos
        ey: end Y pos
    limsx: List of limits for the X coordinate with the structure: 
        [
        [lower_limit_0,upper_limit_0],
        [lower_limit_1,upper_limit_1],
        ...
        [lower_limit_n,upper_limit_n]
        ]
    limsy: List of limits for the Y coordinate with the structure: 
        [
        [lower_limit_0,upper_limit_0],
        [lower_limit_1,upper_limit_1],
        ...
        [lower_limit_n,upper_limit_n]
        ]
    final: Two modes, 'or' and 'and'. if 'or': return a number greater than 0 if any of the
        rectangle points is inside the limits. if 'and' return a number greater than 0 if all
        the rectanble points are inside the limits.
    """
    def point_in_limit(point, limits):
        return list(map(lambda limits: (limits[0]<=point) and (point<=limits[1]),limits))
    def a_and_b (a,b):
        return list(map(lambda x,y: x and y,a,b))
    def a_or_b (a,b):
        return list(map(lambda x,y: x or y,a,b))
    r = []
    sxr = point_in_limit(points[0], limsx)
    exr = point_in_limit(points[1], limsx)
    syr = point_in_limit(points[2], limsy)
    eyr = point_in_limit(points[3], limsy)
    bools = [sxr,exr,syr,eyr]
    a,b,c,d = a_and_b(sxr,syr),a_and_b(sxr,eyr),a_and_b(exr,syr),a_and_b(exr,eyr)
    if final=='or':
        rs = a_or_b(a_or_b(a,b),a_or_b(c,d))
    elif final=='and':
        rs = a_and_b(a_and_b(a,b),a_and_b(c,d))
    return sum(rs)

def extract_all_but_box_and_save(idf,row,file_col,center_points_col,
                         show=False,save_directory=None,box_wh=(30,30),resize_img=None,
                        v=False,strides=(30,30),augmentation=True,
                                 window_from_col=False,window_col='window'):
    """
    Purpose: Given a dataframe load an image and extract windows of size box_wh. 
        The windows that are being created won't be inside the ranges of a  
        list that has the center of boxes (center_points_col).
    
    idf: Dataframe that has in its columns the location of the image file and 
        a column that defines the center of the objects.
    row: row of the DataFrame that will be processed
    file_col: column name in the idf that has the full path to the image file.
    center_points_col: column name in the idf that has the center points of the objects in a list
    show: set to True if you want to plot with plt.imshow() the images that are beinc created.
    save_directory: Specify the directory where you want to save the images.
        There is no validation if the directory exist.
    box_wh: The box width and height that holds each object and also the window dimensions
        that will be extracted from the original image.
    resize_img: specify if the original image (file_col) will be resized prior the extraction 
        of windows
    v: Verbosity, set to True if you want to see the progress.
    strides: define how many pixels for the X and Y dimensions will be skipped to draw each window.
    
    """
    test_row = idf.iloc[row]
    file_name = test_row[file_col]
    image_name = file_name.split('/')[-1]
    
    timg = get_np_image(file_name,resize=resize_img)
    points = test_row[center_points_col]
    w,h = box_wh
    if window_from_col==True:
        w,h = test_row[window_col]
    #
    w,h=int(w),int(h)
    ylim,xlim = timg.shape[0],timg.shape[1]
    #
    limsx, limsy = [],[]
    if v==True:
        print("Calculating points")    
    for i,point in enumerate(points):
        cx,cy = point[0],point[1]
        sx,sy = int(max(0,cx-w/2)),int(max(0,cy-h/2))
        ex,ey = int(min(xlim,cx+w/2)),int(min(ylim,cy+h/2))
        limsx.append([sx,ex])
        limsy.append([sy,ey])
    if v==True:
        print("Points calculated")
    samplex = 0
    for ys in range(0,ylim,strides[1]):
        ye = ys+h
        for xs in range(0,xlim,strides[0]):
            xe = xs + w
            points_lims=[xs,xe,ys,ye]
            if (point_in_limits(points=points_lims,limsx=limsx,limsy=limsy)==0):
                if (point_in_limits(points=points_lims,limsx=[[0,xlim]],limsy=[[0,ylim]],final='and')>=1):
                    img_box = timg[ys:ye,xs:xe]
                    if show==True:
                        print()
                        plt.imshow(img_box)
                        plt.show()
                    if type(save_directory)==str:
                        if save_directory[-1]!='/':
                            save_directory+='/'
                        output_format = image_name.split('.')[-1]
                        save_name = save_directory+image_name+'_box_'+str(samplex)+'_'+str(w)+"x"+str(h)+"_."+output_format
                        pil_img = Image.fromarray(img_box)
                        pil_img.save(save_name)
                        if v==True:
                            print("Saved:",save_name)
                        samplex+=1
                        if augmentation == True:
                            for rot in range(1,4):
                                t_img_box = np.rot90(img_box, k=rot)
                                pil_img = Image.fromarray(t_img_box)
                                save_name = save_directory+image_name+'_box_'+str(samplex)+'_'+str(w)+"x"+str(h)+"_."+output_format
                                pil_img.save(save_name)
                                if v==True:
                                    print("Saved:",save_name)
                                samplex+=1
                                t_img_box_2 = np.flipud(t_img_box)                                
                                pil_img = Image.fromarray(t_img_box_2)
                                save_name = save_directory+image_name+'_box_'+str(samplex)+'_'+str(w)+"x"+str(h)+"_."+output_format
                                pil_img.save(save_name)
                                if v==True:
                                    print("Saved:",save_name)
                                samplex+=1



def extract_box_and_save(idf,row,file_col,center_points_col,
                         show=False,save_directory=None,box_wh=(30,30),resize_img=None,
                        v=False,augmentation=True,window_from_col=False,window_col='window'):
    """
    Extract a window of shape *box_wh* given the center defined in the column *center_points_col*. 
    If *augmentation*=True then the window extracted will be rotated three times 90° and with a horizontal reflection
    creating 6 more elements.
    *resize_img* can be set to a two elements list/tuple of width and height to resize the original image before extracting the window
    """
    test_row = idf.iloc[row]
    file_name = test_row[file_col]
    image_name = file_name.split('/')[-1]
    ####
    timg = get_np_image(file_name,resize=resize_img)
    points = test_row[center_points_col]
    w,h = box_wh
    if window_from_col==True:
        w,h = test_row[window_col]
    ylim,xlim = timg.shape[0],timg.shape[1]
    ####
    for i,point in enumerate(points):
        cx,cy = point[0],point[1]
        sx,sy = int(max(0,cx-w/2)),int(max(0,cy-h/2))
        ex,ey = int(min(xlim,cx+w/2)),int(min(ylim,cy+h/2))
        img_box = timg[sy:ey,sx:ex]
        if show==True:
            plt.imshow(img_box)
            plt.show()
        if type(save_directory)==str:
            if save_directory[-1]!='/':
                save_directory+='/'
            output_format = image_name.split('.')[-1]
            pre_save_name = save_directory+image_name+'_box_'+str(i)+'_'+str(w)+"x"+str(h)+"_"
            save_name = pre_save_name+"."+output_format
            ######
            pil_img = Image.fromarray(img_box)
            pil_img.save(save_name)
            if v==True:
                print("Saved:",save_name)
            if augmentation == True:
                samplex = 0
                for rot in range(1,4):
                    t_img_box = np.rot90(img_box, k=rot)
                    pil_img = Image.fromarray(t_img_box)
                    save_name = pre_save_name+str(samplex)+'_.'+output_format
                    pil_img.save(save_name)
                    if v==True:
                        print("Saved_:",save_name)
                    samplex+=1
                    t_img_box_2 = np.flipud(t_img_box)                                
                    pil_img = Image.fromarray(t_img_box_2)
                    save_name = pre_save_name+str(samplex)+'_.'+output_format
                    pil_img.save(save_name)
                    if v==True:
                        print("Saved:",save_name)
                    samplex+=1




def extract_negatives(idf,save_dir,parts=16,part=0,strides=(256,256)):
    si=part
    s,e=si*idf.shape[0]//parts,(si+1)*idf.shape[0]//parts
    for row in range(s,e):
        print(row)
        extract_all_but_box_and_save(idf,row,file_col='fullpath',center_points_col='center',
                         show=False,save_directory=save_dir,box_wh=(348,348),resize_img=None,
                        v=True,strides=strides,augmentation=True,
                                 window_from_col=True,window_col='window')
    


def extract_positives(idf,save_dir,parts=16,part=0):
    si=part
    s,e=si*idf.shape[0]//parts,(si+1)*idf.shape[0]//parts
    for row in range(s,e):
        print(row)
        extract_box_and_save(idf,row,file_col='fullpath',center_points_col='center',
                         show=False,save_directory=save_dir,box_wh=(80,40),
                         resize_img=None,
                        v=True,augmentation=True,
                        window_from_col=True,window_col='window')


def threads_func(ifunc,ifunc_args,threads,ifunc_args_thread_key):
    import threading
    for t in range(threads):
        print(t,"/",threads)
        ifunc_args[ifunc_args_thread_key]=t
        try:
            threading.Thread(target=ifunc,kwargs=ifunc_args).start()
        except:
            print( "Error: unable to start thread",t)


            
