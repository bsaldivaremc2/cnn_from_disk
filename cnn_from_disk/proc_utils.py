import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np 
import os
import sys
import zipfile
import smtplib

def img_nump_from_file_only_proportion(imgF,w_h = (32,32),resize=True,reshape_batch=True,repeat=8):
    """
    Open Image from file and resize it if specified but return only the proportion of resizing instead
    of the image
    """
    img=Image.open(imgF)
    if resize==True:
        ims = img.size
        xf = w_h[0]/ims[0]
        yf = w_h[1]/ims[1]
        img = img.resize(w_h,Image.ANTIALIAS)
    imnp=np.asarray(img)
    img.close() #Close opened image
    nis = imnp.shape
    if reshape_batch==True:
        nisx = list(nis)
        imnp = imnp.reshape(1,nis[0],nis[1],nis[2])
    return np.hstack([[xf,yf] for _ in range(repeat)])

def get_center(np_array):
    """
    The input is a numpy array of shape [x0,y0,x1,y1,...xn,yn] 
    This function will get all the xs and ys and get the mean 
    of each one returning a pair (x.mean,y.mean)
    """
    _p = np_array.flatten()
    pairs  = _p.shape[0]//2
    points = pd.DataFrame(_p)
    xs = points.iloc[[ x for x in range(pairs)]]
    ys = points.iloc[[ x*2+1 for x in range(pairs)]]
    return [xs.mean(),ys.mean()]




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

def img_nump_from_file_proportion(imgF,w_h = (32,32),resize=True,reshape_batch=True):
    """
    Open Image from file and resize it if specified
    """
    img=Image.open(imgF)
    if resize==True:
        ims = img.size
        xf = w_h[0]/ims[0]
        yf = w_h[1]/ims[1]
        img = img.resize(w_h,Image.ANTIALIAS)
    imnp=np.asarray(img)
    img.close() #Close opened image
    nis = imnp.shape
    if reshape_batch==True:
        nisx = list(nis)
        imnp = imnp.reshape(1,nis[0],nis[1],nis[2])
    return imnp.copy(),[xf,yf]

def string_to_list(y):
    output = y.replace("[","").replace("]","").replace("\n","").strip().split(" ")
    for val in output:
        if len(val)==0:
            output.remove(val)
    output = list(map(lambda x: int(x),output))
    return output

def multiply_y(y,yf):
    yf = np.hstack([yf for _ in range(y.shape[1]//2)])
    return y*yf

def multiply_y_repeat(y,yf,repeat=1):
    op = []
    xf,yf = yf[:,0],yf[:,1]
    for _ in [xf,yf]:
        _ = _.reshape(_.shape[0],1)
        for __ in range(repeat):
            op.append(_)
    f = np.hstack(op)
    return y*f


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

def json_to_points(iJson):
    tdf = pd.read_json(iJson).transpose()
    output_dic = {}
    for image_file in range(tdf.shape[0]):
        image_name = tdf['filename'][image_file]
        points_df = tdf['regions'][image_file]
        points = []
        for k in points_df.keys():
            dic_xy_points = points_df[k]['shape_attributes']
            points.append([dic_xy_points['cx'],dic_xy_points['cy']])
        output_dic[image_name] = np.asarray(points)
    return output_dic.copy()

def get_image_from_zip(ifile,izip,resize=None):
    with zipfile.ZipFile(izip) as tz:
        with tz.open(ifile) as f:
            return get_np_image(f,resize)


def get_np_image(ifile,resize=None):
    _imgx = Image.open(ifile)
    if type(resize)!=type(None):
        _imgx = _imgx.resize((resize[0],resize[1]),Image.ANTIALIAS)
    _img_np = np.asarray(_imgx)
    _imgx.close()
    return _img_np.copy()


def get_image_shape(ifile):
    _imgx = Image.open(ifile)
    _img_np = np.asarray(_imgx)
    _imgx.close()
    return _img_np.shape


def yolo_target_binary_from_zip(idf,izipfile=None,index=0,filename_col='filename',points_col='points',
                                reshape_target=(48,64)):
    img_file_0 = idf['filename'][index]
    if type(izipfile)!=type(None):
        with zipfile.ZipFile(izipfile) as tz:
            with tz.open(img_file_0) as f:
                ims = get_image_shape(f)
    else:
        ims = get_image_shape(img_file_0)
    xf,yf = 1.0*reshape_target[1]/ims[1],1.0*reshape_target[0]/ims[0]
    test_points = idf[points_col][index]
    npz = np.zeros((
        reshape_target[0],reshape_target[1]
    ))
    for _ in range(test_points.shape[0]):
        x,y = test_points[_,:]
        npz[int(np.floor(y*yf)),int(np.floor(x*xf))]=1
    return npz.reshape([1,npz.shape[0]*npz.shape[1]]).copy()


def del_key(dic,key):
    try:
        del dic[key]
    except KeyError:
        pass


def mean_overlap(boxes_list,overlap_th=0.2):
    """
    Given a list of boxes, find those that overlap more than the 
    *overlap_th* value (0.2 is 20%) and keep a box that is the mean 
    of those boxes' coordinates.
    A boxes_list's box has the format:
     [x0,x1,y0,y1] standing for the box coordinates:
      x0,y0; x1,y0; x1,y1; x0,y1
    
    """
    box_overlap={}
    boxes = []
    for _ in range(0,len(boxes_list)):
        box_a = boxes_list[_]
        box_overlap[_]=[np.asarray(box_a)]
        for __ in range(0,len(boxes_list)):
            box_b = boxes_list[__]
            if _!=__:
                x_intersect = max(0,min(box_a[1],box_b[1])-max(box_a[0],box_b[0]))/(box_a[1]-box_a[0])
                y_intersect = max(0,min(box_a[3],box_b[3])-max(box_a[2],box_b[2]))/(box_a[3]-box_a[2])
                xy_intersect = x_intersect*y_intersect
                if (xy_intersect>overlap_th):
                    box_overlap[_].append(np.asarray(box_b))    
        boxes.append(np.mean(np.vstack(box_overlap[_]),0).astype(int))
    return boxes


def clear_directory(folder):
    """
    Remove all files from a directory
    Reference: https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder-in-python
    """
    import os, shutil
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def get_window(ilist):
    """
    The input is a numpy array of shape [x0,y0,x1,y1,...xn,yn] 
    This function will get all the xs and ys and get the middle point
    of each one returning a pair (x,y)
    """
    _l = len(ilist)
    xs = ilist[:_l//2]
    ys = ilist[_l//2:]
    
    _x = max(xs)-min(xs)
    _y = max(ys)-min(ys)
    return [_x,_y]

def get_center(ilist):
    """
    The input is a numpy array of shape [x0,y0,x1,y1,...xn,yn] 
    This function will get all the xs and ys and get the middle point
    of each one returning a pair (x,y)
    """
    _l = len(ilist)
    xs = ilist[:_l//2]
    ys = ilist[_l//2:]
    
    _x = (max(xs)+min(xs))//2
    _y = (max(ys)+min(ys))//2
    return [_x,_y]


def load_dataset(pos_dir, neg_dir,train_test_split=0.8):
    datap = []
    datan = []
    for _ in os.listdir(pos_dir):
        if ".jpg" in _:
            datap.append({'fullpath':pos_dir+"/"+_,'target':np.array([[1,0]])})
    for _ in os.listdir(neg_dir):
        if ".jpg" in _:
            datan.append({'fullpath':neg_dir+"/"+_,'target':np.array([[0,1]])})
    idfp = pd.DataFrame(datap)
    idfn = pd.DataFrame(datan)
    train_pos_df,test_pos_df = split_train_test(idfp,train_test_proportion=train_test_split,shuffle=True)
    train_neg_df,test_neg_df = split_train_test(idfn,train_test_proportion=train_test_split,shuffle=True)
    return train_pos_df,test_pos_df,train_neg_df,test_neg_df

def make_dir(idir):
    if not os.path.exists(idir):
        os.makedirs(idir)


def bryan_image_generation(file_name,file_name_2=None,resample_margin=0.2,output_wh = [256,256],
                           flip_h_prob=0.5,flip_v_prob=0.5,add_noise_prob = 0.5,mult_noise_prob = 0.5,add_shift_prob = 0.5,mult_shift_prob = 0.5,
                            add_noise_std = 16,mult_noise_var = 0.25, shift_add_max = 30, shift_mult_var = 0.125,norm=True,reshape_batch=True,
                           zip_file=None,file_name_to_3d=False,file_name_repeat_3_channels=False,file_name_2_to_3d=False,file_name_2_repeat_3_channels=False
                          ):
  """
  Given an Image file name location, load an image at which apply noise addition, multiplication, 
  color shift addition and multiplication, resampling a piece of the image with a margin *resample_margin*, 
  if norm=True, the image will be divided by 255
  if reshape_batch=True then the ouput will be a numpy array of dimmensions [1,w,h,channels]
  zip_file: if not None, the string name of the zip file that contains the file_name
  """
  
  def correct_limits(iinp):
    inp = iinp.copy()
    inp[inp<0]=0
    inp[inp>255]=255
    return inp.copy()
  import numpy as np
  from PIL import Image
  #For sampling
  resize_dims = list(map(lambda x: int(x*(1+resample_margin)),output_wh))
  margin_wh = [ r-o for r,o in zip(resize_dims,output_wh)]
  sw = np.random.randint(0,margin_wh[0])
  sh = np.random.randint(0,margin_wh[1])
  #Get Bbools for data augmentation
  flip_h = np.random.choice([True,False],size=1,p=[flip_h_prob,1-flip_h_prob])[0]
  flip_v = np.random.choice([True,False],size=1,p=[flip_v_prob,1-flip_v_prob])[0]
  add_noise_bool = np.random.choice([True,False],size=1,p=[add_noise_prob,1-add_noise_prob])[0]
  mult_noise_bool = np.random.choice([True,False],size=1,p=[mult_noise_prob,1-mult_noise_prob])[0]
  add_shift_bool = np.random.choice([True,False],size=1,p=[add_shift_prob,1-add_shift_prob])[0]
  mult_shift_bool = np.random.choice([True,False],size=1,p=[mult_shift_prob,1-mult_shift_prob])[0]
  #Open Image
  def open_zip(zip_file,file_name):
    with zipfile.ZipFile(zip_file) as zf:
      with zf.open(file_name) as fn:
        pil_img = Image.open(fn)
        return pil_img.copy()
  if type(zip_file)==type(None):
    pil_img = Image.open(file_name)
    if type(file_name_2)==str:
      pil_img_2 = Image.open(file_name_2)
  else:
    pil_img = open_zip(zip_file,file_name)
    if type(file_name_2)==str:
      pil_img_2 = open_zip(zip_file,file_name_2)
  def resize_flip_resample(ipil_img,flip_h,flip_v,repeat_3_channels=True,to_3=False):
    o_pil_img = ipil_img.resize(resize_dims,Image.ANTIALIAS)
    #Flipping
    if flip_h:
      o_pil_img = o_pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_v:
      o_pil_img = o_pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    o_np_img = np.asarray(o_pil_img)
    o_pil_img.close()
    ipil_img.close()
    #Resampling from image
    if len(o_np_img.shape)==2:
      if repeat_3_channels==True:
        new_dims = []
        #np_img = np_img.reshape(np_img.shape[0],np_img.shape[1],1)
        for _ in range(3):
          new_dims.append(o_np_img)
        o_np_img = np.dstack(new_dims)
      if to_3==True:
        o_np_img = o_np_img.reshape(o_np_img.shape[0],o_np_img.shape[1],1)
    o_np_img = o_np_img[sw:sw+output_wh[0],sh:sh+output_wh[1],:]
    return o_np_img.copy()
  np_img = resize_flip_resample(pil_img,flip_h,flip_v,repeat_3_channels=file_name_repeat_3_channels,to_3=file_name_to_3d)
  if type(file_name_2)==str:
    clean_np = resize_flip_resample(pil_img_2,flip_h,flip_v,repeat_3_channels=file_name_2_repeat_3_channels,to_3=file_name_2_to_3d)
  if add_noise_bool:
    #print("Additive noise")
    additive_noise = np.random.normal(0,add_noise_std,size=np_img.shape)
    np_img= np_img+additive_noise
    np_img = correct_limits(np_img)
    #np_img[np_img<0]=0
    #np_img[np_img>255]=255
  if mult_noise_bool:
    #print("Multiplicative noise")
    low = 1 - mult_noise_var
    high = 1 + mult_noise_var
    m_noise = np.random.rand(*np_img.shape)*2*mult_noise_var + low
    np_img= np_img*m_noise
    np_img = correct_limits(np_img)
  #Color shift
  if add_shift_bool:
    #print("Additive shift")
    shift_dims_n = np.random.randint(1,np_img.shape[2]+1)
    shift_dims = np.random.choice(np.arange(np_img.shape[2]),shift_dims_n,replace=False)
    tnp = np_img.copy()
    for color_dim in shift_dims:
      color_shift = np.random.randint(0,shift_add_max)
      #print(shift_dims,color_shift)
      tnp[:,:,color_dim]+=color_shift
    np_img = correct_limits(tnp)
  if mult_shift_bool:
    #print("Multiplicative shift")
    shift_dims_n = np.random.randint(1,np_img.shape[2]+1)
    shift_dims = np.random.choice(np.arange(np_img.shape[2]),shift_dims_n,replace=False)
    tnp = np_img.copy()
    for color_dim in shift_dims:
      low = 1 - shift_mult_var
      high = 1 + shift_mult_var
      m_shift = np.random.rand()*2*shift_mult_var + low
      tnp[:,:,color_dim]= tnp[:,:,color_dim] * m_shift
    np_img = correct_limits(tnp)
  #Normalize
  if norm==True:
    np_img = np_img/255
  if reshape_batch==True:
    np_img = np_img.reshape(1,*np_img.shape)
    if type(file_name_2)==str:
      clean_np = clean_np.reshape(1,*clean_np.shape)
  if type(file_name_2)!=str:
    return np_img.copy()
  else:
    return np_img.copy(),clean_np.copy()

def cv_index_no_replace(ilist,n_fold=5):
    n_samples = len(ilist)//n_fold
    folds_list=[]
    picked_items = []
    for _ in range(n_fold-1):
        __ = np.random.choice(ilist,n_samples,replace=False)
        folds_list.append(__)
        picked_items.extend(__)
        ilist = list(filter(lambda x: x not in picked_items,ilist))
    folds_list.append(np.asarray(ilist))
    return folds_list[:]

def train_test_df(idf,test_split=0.25):
    test_n = int(np.ceil(idf.shape[0]*0.25))
    iv = idf.index.values
    test_index = np.random.choice(iv,test_n,replace=False)
    ts_df = idf.iloc[test_index,:].reset_index(drop=True)
    train_index=list(filter( lambda x: x not in test_index,iv))
    tr_df = idf.iloc[train_index,:].reset_index(drop=True)
    return tr_df.copy(),ts_df.copy()


def join_head_tail_numbers(ilist):
    #See if odd or even
    ll = len(ilist)
    ll2 = ll%2
    output_index = []
    while(ll>0):
        output_index.append(ilist[0])
        _ = ilist.pop(0)
        if len(ilist)>0:
            output_index.append(ilist[-1])
            _ = ilist.pop(-1)
        ll = len(ilist)
    return output_index[:]



def train_test_df_balanced(idf,n_fold=5,class_column=None,batch_size=None):
    classes = list(idf[class_column].unique())
    fold_classes = []
    for c in classes:
        _df = idf[idf[class_column]==c].reset_index(drop=True)
        ilist = np.arange(_df.shape[0])
        index_groups = cv_index_no_replace(ilist[:],n_fold=n_fold)
        fold_list = []
        for ig in index_groups:
            fold_list.append(_df.iloc[ig].reset_index(drop=True).copy())
        fold_classes.append(fold_list[:])
    mixed_folds = []
    df_folds_list = []
    for nf in range(n_fold):
        tmp_list = []
        for c in range(len(classes)):
            tmp_list.append(fold_classes[c][nf][:])
        df_folds_list.append(pd.concat(tmp_list[:],0).reset_index(drop=True))
    #Getting the nfolds
    output_folds = []
    for nf in range(n_fold):
        possible_indexes = [_ for _ in range(n_fold)]
        test_index = nf
        _ = possible_indexes.pop(nf)
        train_index = possible_indexes
        train_folds = []
        for _ in train_index:
            train_folds.append(df_folds_list[_].copy())
        #Test
        test_fold = df_folds_list[test_index]
        test_balance_index = join_head_tail_numbers(list(np.arange(test_fold.shape[0])))
        test_fold = test_fold.iloc[test_balance_index].reset_index(drop=True)
        #Train
        train_fold = pd.concat(train_folds,0).reset_index(drop=True)
        train_balance_index = join_head_tail_numbers(list(np.arange(train_fold.shape[0])))
        train_fold = train_fold.iloc[train_balance_index].reset_index(drop=True)
        output_folds.append([train_fold.copy(),test_fold.copy()])
    return output_folds[:]


def sample_box_and_noise(filename,sample_box_kargs={},noise_kargs={}):
    _onp = sample_box(filename,**sample_box_kargs)
    _onp = bryan_noise_generation_inp(_onp,**noise_kargs)
    return _onp.copy()

def sample_box(filename,kernel_width=33):
    from scipy import ndimage 
    from scipy import misc
    kw = kernel_width
    k =np.ones((kw,kw))
    kww = kw//2
    pi = Image.open(filename)
    npi = np.asarray(pi)
    npi2d = npi.sum(2)
    npi2d[npi2d>0]=1
    ik = ndimage.convolve(npi2d, k, mode='constant', cval=0.0)
    max_val = ik.max()
    ik[ik!=kw**2]=0
    ik[ik==kw**2]=1
    possible_patches = list(np.where(ik==1))
    row_patch_pos,col_patch_pos = possible_patches
    patch_pos = np.random.choice(np.arange(row_patch_pos.shape[0]),1)[0]
    ri = row_patch_pos[patch_pos]
    ci = col_patch_pos[patch_pos]
    ris,rie = ri-kww,ri+kww
    cis,cie = ci-kww,ci+kww
    sample = npi[ris:rie,cis:cie,:]
    #sample = misc.imresize(sample,output_wh)
    return sample.copy()

def sample_box_mosaic(filename,kernel_width=32,output_wh=224):
    sample = sample_box(filename,kernel_width)
    repeat_mosaic = int(np.floor(output_wh / kernel_width))
    zeros_pad = output_wh-repeat_mosaic*kernel_width
    cols = []
    for col in range(repeat_mosaic):
        cols.append(sample.copy())
    cols.append(np.zeros((sample.shape[0],zeros_pad,3)))
    output = np.hstack(cols)
    rows = []
    for row in range(repeat_mosaic):
        rows.append(output.copy())
    rows.append(np.zeros((zeros_pad,output.shape[1],3)))
    output = np.vstack(rows)
    return output.copy()


def sample_box_mosaic_and_noise(filename,sample_box_kargs={},noise_kargs={}):
    _onp = sample_box_mosaic(filename,**sample_box_kargs)
    _onp = bryan_noise_generation_inp(_onp,**noise_kargs)
    return _onp.copy()

def bryan_noise_generation_inp(inp,output_wh=[224,224],resize=True,resample_margin=0.05,
                           flip_h_prob=0.5,flip_v_prob=0.5,add_noise_prob = 0.5,mult_noise_prob = 0.5,add_shift_prob = 0.5,mult_shift_prob = 0.5,
                            add_noise_std = 16,mult_noise_var = 0.25, shift_add_max = 30, shift_mult_var = 0.125,norm=True,reshape_batch=True,
                            repeat_3_channels=False,to_3=False
                          ):
  """
  Given an Image file name location, load an image at which apply noise addition, multiplication, 
  color shift addition and multiplication, resampling a piece of the image with a margin *resample_margin*, 
  if norm=True, the image will be divided by 255
  if reshape_batch=True then the ouput will be a numpy array of dimmensions [1,w,h,channels]
  zip_file: if not None, the string name of the zip file that contains the file_name
  """
  
  def correct_limits(iinp):
    inp = iinp.copy()
    inp[inp<0]=0
    inp[inp>255]=255
    return inp.copy()
  import numpy as np
  from PIL import Image
  resize_dims = list(map(lambda x: int(x*(1+resample_margin)),output_wh))
  margin_wh = [ r-o for r,o in zip(resize_dims,output_wh)]
  sw = np.random.randint(0,margin_wh[0])
  sh = np.random.randint(0,margin_wh[1])
  #Get Bbools for data augmentation
  flip_h = np.random.choice([True,False],size=1,p=[flip_h_prob,1-flip_h_prob])[0]
  flip_v = np.random.choice([True,False],size=1,p=[flip_v_prob,1-flip_v_prob])[0]
  add_noise_bool = np.random.choice([True,False],size=1,p=[add_noise_prob,1-add_noise_prob])[0]
  mult_noise_bool = np.random.choice([True,False],size=1,p=[mult_noise_prob,1-mult_noise_prob])[0]
  add_shift_bool = np.random.choice([True,False],size=1,p=[add_shift_prob,1-add_shift_prob])[0]
  mult_shift_bool = np.random.choice([True,False],size=1,p=[mult_shift_prob,1-mult_shift_prob])[0]
  #Open Image
  def resize_flip_resample(ipil_img,flip_h,flip_v,repeat_3_channels=True,to_3=False):
    o_pil_img = ipil_img
    if resize==True:
      o_pil_img = ipil_img.resize(resize_dims,Image.ANTIALIAS)
    #Flipping
    if flip_h:
      o_pil_img = o_pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_v:
      o_pil_img = o_pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    o_np_img = np.asarray(o_pil_img)
    o_pil_img.close()
    ipil_img.close()
    #Resampling from image
    if len(o_np_img.shape)==2:
      if repeat_3_channels==True:
        new_dims = []
        #np_img = np_img.reshape(np_img.shape[0],np_img.shape[1],1)
        for _ in range(3):
          new_dims.append(o_np_img)
        o_np_img = np.dstack(new_dims)
      if to_3==True:
        o_np_img = o_np_img.reshape(o_np_img.shape[0],o_np_img.shape[1],1)
    if resize==True:
      o_np_img = o_np_img[sw:sw+output_wh[0],sh:sh+output_wh[1],:]
    return o_np_img.copy()
  pil_img = Image.fromarray(inp)
  np_img = resize_flip_resample(pil_img,flip_h,flip_v,repeat_3_channels=repeat_3_channels,to_3=to_3)
  if add_noise_bool:
    #print("Additive noise")
    additive_noise = np.random.normal(0,add_noise_std,size=np_img.shape)
    np_img= np_img+additive_noise
    np_img = correct_limits(np_img)
    #np_img[np_img<0]=0
    #np_img[np_img>255]=255
  if mult_noise_bool:
    #print("Multiplicative noise")
    low = 1 - mult_noise_var
    high = 1 + mult_noise_var
    m_noise = np.random.rand(*np_img.shape)*2*mult_noise_var + low
    np_img= np_img*m_noise
    np_img = correct_limits(np_img)
  #Color shift
  if add_shift_bool:
    #print("Additive shift")
    shift_dims_n = np.random.randint(1,np_img.shape[2]+1)
    shift_dims = np.random.choice(np.arange(np_img.shape[2]),shift_dims_n,replace=False)
    tnp = np_img.copy()
    for color_dim in shift_dims:
      color_shift = np.random.randint(0,shift_add_max)
      #print(shift_dims,color_shift)
      tnp[:,:,color_dim]+=color_shift
    np_img = correct_limits(tnp)
  if mult_shift_bool:
    #print("Multiplicative shift")
    shift_dims_n = np.random.randint(1,np_img.shape[2]+1)
    shift_dims = np.random.choice(np.arange(np_img.shape[2]),shift_dims_n,replace=False)
    tnp = np_img.copy()
    for color_dim in shift_dims:
      low = 1 - shift_mult_var
      high = 1 + shift_mult_var
      m_shift = np.random.rand()*2*shift_mult_var + low
      tnp[:,:,color_dim]= tnp[:,:,color_dim] * m_shift
    np_img = correct_limits(tnp)
  #Normalize
  if norm==True:
    np_img = np_img/255
  if reshape_batch==True:
    np_img = np_img.reshape(1,*np_img.shape)
  return np_img.copy()

