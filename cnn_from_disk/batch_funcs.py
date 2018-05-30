from PIL import Image
import pandas as pd
import numpy as np 
import zipfile
from .proc_utils import *

def batch_pre_proc_from_df(idf,xfunc,yfunc,xfunc_params={},yfunc_params={},x_col='filename',y_col='target',
                           batch_size=4,offset=4,inference=False):
    start_index = offset
    rows = idf.shape[0]
    if start_index >= rows:
        start_index = start_index%rows
    end_index = start_index + batch_size
    
    if end_index>rows:
        end_index = rows
    
    tdf = idf.iloc[start_index:end_index]
    
    _x = np.vstack(tdf[x_col].apply(xfunc,**xfunc_params))
    
    if inference == False:
        _y = np.vstack(tdf[y_col].apply(yfunc,**yfunc_params))
        return _x.copy(),_y.copy()
    else:
        return _x.copy()

def batch_pre_proc_from_df_yf(idf,xfunc,yfunc,yfunc2,xfunc_params={},yfunc_params={},yfunc2_params={},x_col='filename',y_col='target',
                           batch_size=4,offset=4,inference=False):
    """
    This function is similar to batch_pre_proc_from_df except that the function that applies to x
    returns two values, the batch value for x and a value that will later be fed into a second y function
    yfunc2.
    """
    start_index = offset
    rows = idf.shape[0]
    if start_index >= rows:
        start_index = start_index%rows
    end_index = start_index + batch_size
    
    if end_index>rows:
        end_index = rows
    
    tdf = idf.iloc[start_index:end_index]
    
    #_x,_yf = tdf[x_col].apply(xfunc,**xfunc_params)
    _ = tdf[x_col].apply(xfunc,**xfunc_params)
    _x, _yf = np.vstack(list(map(lambda z: z[0],_))),np.vstack(list(map(lambda z: z[1],_)))
    
    if inference == False:
        _y = np.vstack(tdf[y_col].apply(yfunc,**yfunc_params))
        _y = yfunc2(_y,_yf,**yfunc2_params)
        return _x.copy(),_y.copy()
    else:
        return _x.copy()


def df_xy (df,x_label,y_label,batch_size=5,offset=0,resize_wh=(32,32),input_channels=3,toGray=False,zip_file=None):
    """
    Function to load images by batch. Oriented to CLASSIFICATION
    """
    if toGray==True:
        channels=1
    else:
        channels=3
    x_y = df.iloc[offset:offset+batch_size]
    images = x_y[x_label].values
    
    
    imgs = []
    fxa = []
    fya = []
    for _ in range(0,batch_size):
        if type(zip_file)!=type(None):
            with zipfile.ZipFile(zip_file) as zf:
                with zf.open(images[_]) as unzip_img:
                    img, fx,fy = imgOpenResize(unzip_img,resize_wh)
        else:
            img, fx,fy = imgOpenResize(images[_],resize_wh)
        
        fxa.append(fx)
        fya.append(fy)
        imgs.append(img)
           
    
    

    x = np.asarray(imgs)
    if toGray==True:
        x = list(map(npToGray,x))
    x = np.asarray(x)
    x = x.reshape([x.shape[0],x.shape[1],x.shape[2],channels])
    
    if type(y_label)!=(type(None)):
        target = np.concatenate(x_y[y_label].values,0)
        return x,target
    else:
        return x

def df_x_y (df,x_label,xp_label=None,yp_label=None,batch_size=5,offset=0,resize_wh=(32,32),toGray=False,zip_file=None):
    """
    Function to load images by batch. Oriented to LANDMARK REGRESSION
    """
    if toGray==True:
        channels=1
    else:
        channels=3
    x_y = df.iloc[offset:offset+batch_size]
    images = x_y[x_label].values
    imgs = []
    fxa = []
    fya = []
    for _ in range(0,batch_size):
        if type(zip_file)!=type(None):
            with zipfile.ZipFile(zip_file) as zf:
                with zf.open(images[_]) as unzip_img:
                    img, fx,fy = imgOpenResize(unzip_img,resize_wh)
        else:
            img, fx,fy = imgOpenResize(images[_],resize_wh)
        
        fxa.append(fx)
        fya.append(fy)
        imgs.append(img)
    #zz = list(map(imgOpenResize,fx,[resize_wh for _ in range(0,batch_size) ]))
    
    def div_np(num,ilist,batch_size):
        div_m =  np.asarray([ilist]).reshape((batch_size,1))
        td = div_m
        for _n_ in range(0,num.shape[1]-1):
            div_m = np.concatenate([div_m,td],1)
        return num*div_m
        
       
    x = np.asarray(imgs)
    if toGray==True:
        x = list(map(npToGray,x))
    x = np.asarray(x)
    x = x.reshape([x.shape[0],x.shape[1],x.shape[2],channels])
    if type(xp_label)!=type(None):
        xps = div_np(x_y[xp_label],fxa,batch_size)
        yps= div_np(x_y[yp_label],fya,batch_size)
    
        return x,xps.values,yps.values
    else:
        return x

