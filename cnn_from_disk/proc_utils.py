import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np 
import os
import sys
import zipfile


def img_nump_from_file(imgF,w_h = (32,32),resize=True,reshape_batch=True):
    """
    Open Image from file and resize it if specified
    """
    img=Image.open(imgF)
    if resize==True:        
        #xf = w_h[0]/img_size[0]
        #yf = w_h[1]/img_size[1]
        img = img.resize(w_h,Image.ANTIALIAS)
    imnp=np.asarray(img)
    img.close() #Close opened image
    nis = imnp.shape
    if reshape_batch==True:
        nisx = list(nis)
        imnp = imnp.reshape(1,nis[0],nis[1],nis[2])
    return imnp.copy()

y_func = lambda y: np.vstack(y.values)


def string_to_list(y):
    output = y.replace("[","").replace("]","").replace("\n","").strip().split(" ")
    for val in output:
        if len(val)==0:
            output.remove(val)
    output = list(map(lambda x: int(x),output))
    return output

def pass_y (y):
    return y


def acc_sen_spe(tp,tn,fp,fn):
    stats_dic={}
    stats_dic['sen']=(tp/(tp+fn))
    stats_dic['acc']=(tn+tp)/(tp+tn+fp+fn)
    stats_dic['spe']=tn/(tn+fp)
    return stats_dic.copy()


def plot_image_points(df,_index_,fullname,max_point=11,figsize=(8,8)):
    """
    Given a dataframe df load a picture in a given row:_index_ by the fullname: absolute path
    with max_point the number of points/columns that are present as landmarks.
    """
    file = df[fullname].loc[_index_]
    x_labels=['x'+str(_) for _ in range(0,max_point)]
    y_labels=['y'+str(_) for _ in range(0,max_point)]
    xp = df[x_labels].loc[_index_]
    yp = df[y_labels].loc[_index_]
    print(file)
    ii = Image.open(file)
    ni = np.asarray(ii)
    ii.close()
    plt.figure(figsize=figsize)
    plt.imshow(ni)
    plt.scatter(xp,yp,marker='x',c='b')
    plt.show()
    
def plot_prediction_points(imgs_array,predicted_points,_index_=0,figsize=(8,8)):
    pred_xy = predicted_points[_index_]
    points=pred_xy.shape[0]//2
    xps = pred_xy[0:points]
    yps = pred_xy[points:]
    x = imgs_array[_index_]
    plt.figure(figsize=(8,8))
    plt.imshow(x)
    plt.scatter(xps,yps)
    plt.show()

def send_mail(email_origin,email_destination,email_pass,subject="Test report",content="Test"):
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    #Next, log in to the server
    server.login(email_origin,email_pass)
    msg = "Subject:"+subject+" \n\n "+content+"\n" # The /n separates the message from the headers
    server.sendmail(email_origin,email_destination, msg)

def rgbToG(img):
    """
    Color to gray
    """
    npImg=np.asarray(img)
    r=0.2125
    g=0.7154
    b=0.0721
    gsImg=r*npImg[:,:,0]+g*npImg[:,:,1]+b*npImg[:,:,2]
    return gsImg

def npToGray(iNp):
    """
    Open Image and 
    """
    imnp=iNp
    ims=len(imnp.shape)
    if ims == 3:
        imgG=rgbToG(imnp)
    elif ims ==2:
        imgG=imnp
    return imgG

def imgOpenResize(imgF,w_h = (32,32)):
    """
    Open Image and 
    """
    img=Image.open(imgF)
    img_size = img.size
    xf = w_h[0]/img_size[0]
    yf = w_h[1]/img_size[1]

    img = img.resize(w_h,Image.ANTIALIAS)
    imnp=np.asarray(img)
    img.close() #Close opened image
    return imnp,xf,yf

def split_train_test(idf,train_test_proportion=0.8,shuffle=False):
    rows = idf.shape[0]
    train_split=int(rows*train_test_proportion)
    test_split=rows-train_split
    if shuffle ==True:
        ttdf = idf.sample(rows).reset_index()
    train_df = ttdf.head(train_split)
    test_df = ttdf.tail(test_split)
    return train_df,test_df