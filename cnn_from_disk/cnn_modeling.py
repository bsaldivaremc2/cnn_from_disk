import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
from .tf_layers import *
from .proc_utils import *

def create_model(ixs,iys,model=None,opt_mode='classification'):    
    """
    Version 0.107
    """
    if model==None:
        print("No model specified")
        return 0
    tf.reset_default_graph()
    xi = tf.placeholder(tf.float32, shape=ixs,name='x')
    y_ = tf.placeholder(tf.float32, shape=iys,name='y')
    train_bool=tf.placeholder(bool,name='train_test')
    learning_rate = tf.placeholder(tf.float32)
    #Define the model here--DOWN
    x = xi
    layers=['conv','bn','relu','max_pool','drop_out','fc','res_131','conv_t']
    conv_layers = ['conv','res_131','conv_t']
    type_layers = ['fc']
    type_layers.extend(conv_layers)
    types_dic = {}
    for l in layers:
        types_dic[l]=0
    last_type = None
    for i,_ in enumerate(model):
        _type=_[0]
        _input=x
        params={'_input':_input}
        #print(_)
        if len(_)==2:
            _params=_[1]
            #print(params,_params)
            params.update(_params)
        counter=types_dic[_type]
        types_dic[_type]+=1
        name_scope=_type+str(counter)
        #new line
        if 'name_scope' not in _params.keys():
            params['name_scope']=name_scope
        if _type=='conv':    
            x=conv(**params)
        elif _type=='conv_t':    
            x=conv_transpose(**params)
        elif _type=='bn':
            params['is_training']=train_bool
            x = batch_norm(**params)
        elif _type=='relu':
            x = relu(**params)
        elif _type=='max_pool':
            x = max_pool(**params)
        elif _type=='drop_out':
            params['is_training']=train_bool
            x = drop_out(**params)
        elif _type=='fc':
            params['prev_conv']=False
            if i>0:
                if last_type in conv_layers:
                    params['prev_conv']=True
            x = fc(**params)
        elif _type=='res_131':
            params['is_training']=train_bool
            x = res_131(**params)
        if _type in type_layers:
            last_type=_type
    prev_conv_fcl = False
    if last_type in conv_layers:
        prev_conv_fcl=True
    if opt_mode in ['regression','classification']:
        prediction = fc(x,n=iys[1],name_scope="FCL",prev_conv=prev_conv_fcl)
    else:
        prediction = x
    with tf.name_scope('output'):
        prediction = tf.identity(prediction, name="output")
    ##Define the model here--UP
    if opt_mode=='classification':
        y_CNN = tf.nn.softmax(prediction,name='Softmax')        
        class_pred = tf.argmax(y_CNN,1,name='ClassPred')       
        loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]),name="loss")
        acc_,spe_,sen_,tp_,tn_,fp_,fn_ = stats_class(y_CNN,y_)
        stats_dic={'acc':acc_,'spe':spe_,'sen_':sen_,'tp':tp_,'tn':tn_,'fp':fp_,'fn':fn_}
    elif opt_mode in ['regression','yolo','segmentation']:
        loss = tf.reduce_mean(tf.pow(tf.subtract(y_,prediction),2),name='loss')
        stats_dic={'loss':loss}
    #The following three lines are required to make "is_training" work for normalization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return [xi,y_,learning_rate,train_bool,loss,train_step,stats_dic]







def train_model(batch_func,batch_func_params,model=None,iters=10,lr=0.001,
          save_model=True,save_name=None,
          restore_model=False,restore_name=None,
          v=False,opt_mode='classification',log=False,log_file=None):
    """
    Version 0.104
    The train is done using a function to return the x and y values for each batch processing.
    *batch_func: is the function that will output the x and y values that will be loaded each mini-batch
    *batch_func_params: is a dictionary that contains the params for the batch_func, at least if should have
    the following keys:
        * idf: the dataframe that specifies the total size of the data
        * batch_size: the size for the mini_batch
        * offset: this value is used to use the next mini_match starting in the position "offset"
    * model: is a list of lists that defines the model that will be created
    * log: Specify if a log of the training process will be appended to log_file, log_file must exist
    """
    # Define parameters to 
    base_run_params = dict(batch_func_params)
    base_run_params['batch_size']=2
    base_run_params['offset']=0
    rows = base_run_params['idf'].shape[0]
    batch_size = batch_func_params['batch_size']
    batches = int(np.ceil(rows/batch_size))
    
    
    x,y = batch_func(**base_run_params)
    ixs,iys = list(x.shape),list(y.shape)
    ixs[0],iys[0]=None,None
    
    print("Building model...")
    
    xi,y_,learning_rate,train_bool,loss,train_step,stats = create_model(ixs,iys,model=model,opt_mode=opt_mode)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    print("Model created. Training starting ...")
    with tf.Session() as s:
        if restore_model==True:
            if type(restore_name)==type(None):
                print("No model file specified")
                return
            else:
                saver.restore(s,restore_name)
        else:
            s.run(init_op)
        fd={learning_rate:lr,train_bool:True}
        print("Before iters")
        for _ in range(0,iters):            
            #Define parameters to load from disk
            for batch in range(0,batches):
                batch_func_params['offset']=batch*batch_size   
                ix,iy = batch_func(**batch_func_params)
                fdt={xi:ix,y_:iy}
                fd.update(fdt)
                    
                _t,l= s.run([train_step,loss],feed_dict=fd)

                output_string = "'Iter':"+str(_)+",'batch':"+str(batch)+",'batches':"+str(batches)+",'Loss':"+str(l);

                if v==True:
                    print(output_string)
                if (log==True) and (os.path.isfile(log_file)):
                    with open(log_file,"a") as _f:
                        _f.write(output_string+"\n")

            
        if save_model==True:
            if type(save_name)!=type(None):
                save_path = saver.save(s, save_name)
                print("Model saved in file: %s" % save_name)
            else:
                print("No model specified, model not being saved")
                return

def test_model(batch_func,batch_func_params,model_name=None,opt_mode='regression',
               stats_list=['tp','tn','fp','fn','loss','spe','sen','acc']):
    """
    The train is done using a function to return the x and y values for each batch processing.
    *batch_func: is the function that will output the x and y values that will be loaded each mini-batch
    *batch_func_params: is a dictionary that contains the params for the batch_func, at least if should have
    the following keys:
        * idf: the dataframe that specifies the total size of the data
        * batch_size: the size for the mini_batch
    """
    if model_name==None:
        print("No model to load")
        return
    else:
        ## Make the stats for classification useful. Without :0 they can't work
        stats_l = []
        for _ in stats_list:
            stats_l.append(_+":0")
        return_dic ={}
        
        stats_dic = {'regression':'loss:0','classification':stats_l,'yolo':'loss:0'}
        stats_output = stats_dic[opt_mode]
    
        #Start loading the model
        with tf.Session('', tf.Graph()) as s:
            with s.graph.as_default():
                ##Restore model
                saver = tf.train.import_meta_graph(model_name+".meta")
                saver.restore(s,model_name)
                
                #If a dropout layer is present set the values to 1.
                dop_dic = {}
                for x in tf.get_default_graph().get_operations():
                    if x.type == 'Placeholder':
                        if "drop_out" in x.name:
                            dop_dic[x.name+":0"]=1.0
                
                ### Initialize stats output
                fd = dop_dic
                fd={'train_test:0':False}
                fd.update(dop_dic)
                
                batch_output = []
                
                #Define parameters to load from disk
                rows = batch_func_params['idf'].shape[0]
                batch_size = batch_func_params['batch_size']
                batches = int(np.ceil(rows/batch_size))
                
                for batch in range(0,batches):
                    batch_func_params['offset']=batch*batch_size
                    ix,iy = batch_func(**batch_func_params)
                    #Add the values to the dictionary for training
                    fdt={'x:0':ix,'y:0':iy}
                    fd.update(fdt)
                    
                    #Run test
                    stats_result = s.run(stats_output,feed_dict=fd)
                    
                    if opt_mode=='classification':
                        proc_dic = {}
                        for _,sr in enumerate(stats_result):
                            proc_dic[stats_list[_]]=sr
                        print(proc_dic)
                        output = proc_dic
                    elif (opt_mode =='regression') or (opt_mode=='yolo'):
                        print("Loss",stats_result)
                        output = stats_result
                    batch_output.append(output)
    return batch_output 



def infer_model(batch_func,batch_func_params,model_name=None,v=False):
    """
    Function to make an inference based on the trained model
    *batch_func: is the function that will output the x and y values that will be loaded each mini-batch
    *batch_func_params: is a dictionary that contains the params for the batch_func, at least if should have
    the following keys:
        * idf: the dataframe that specifies the total size of the data
        * batch_size: the size for the mini_batch
    """
    if model_name==None:
        print("No model to load")
        return
    else:
        with tf.Session('', tf.Graph()) as s:
            with s.graph.as_default():
                saver = tf.train.import_meta_graph(model_name+".meta")
                saver.restore(s,model_name)
                #Define parameters to load from disk
                rows = batch_func_params['idf'].shape[0]
                batch_size = batch_func_params['batch_size']
                batches = int(np.ceil(rows/batch_size))
                fd={'train_test:0':False}
                output = []
                #########
                for batch in range(0,batches):
                    batch_func_params['offset']=batch*batch_size
                    ix = batch_func(**batch_func_params)
                    fdt={'x:0':ix}
                    fd.update(fdt)
                    fcl = s.run('output/output:0',feed_dict=fd)
                    output.append(fcl)
                    if v==True:
                        print(fcl)
        return output

    
def layer_output(batch_func,batch_func_params,model_name=None,layer_name='FCL/FC:0',v=False):
    """
    Function to return the output of a layer
    *batch_func: is the function that will output the x and y values that will be loaded each mini-batch
    *batch_func_params: is a dictionary that contains the params for the batch_func, at least if should have
    the following keys:
        * idf: the dataframe that specifies the total size of the data
        * batch_size: the size for the mini_batch
    """
    if model_name==None:
        print("No model to load")
        return
    else:
        with tf.Session('', tf.Graph()) as s:
            with s.graph.as_default():
                saver = tf.train.import_meta_graph(model_name+".meta")
                saver.restore(s,model_name)
                #Define parameters to load from disk
                rows = batch_func_params['idf'].shape[0]
                batch_size = batch_func_params['batch_size']
                batches = int(np.ceil(rows/batch_size))
                fd={'train_test:0':False}
                output = []
                #########
                for batch in range(0,batches):
                    batch_func_params['offset']=batch*batch_size
                    ix = batch_func(**batch_func_params)
                    fdt={'x:0':ix}
                    fd.update(fdt)
                    fcl = s.run(layer_name,feed_dict=fd)
                    output.append(fcl)
                    if v==True:
                        print(fcl)
        return output


def stats_class(predicted,ground_truth):
    yi = tf.argmax(ground_truth,1)
    yp = tf.argmax(predicted,1)
    tpi = yp*yi
    tp = tf.reduce_sum(tf.cast(tf.greater(tpi,0),tf.int32),name='tp')
    fni = yi-yp
    fn = tf.reduce_sum(tf.cast(tf.greater(fni,0),tf.int32),name='fn')
    sensitivity = tf.divide(tp,(fn+tp),name='sen')    #sensitivity = tp/(fn+tp)    
    tni = yi+yp
    tn = tf.reduce_sum(tf.cast(tf.equal(tni,0),tf.int32),name='tn')    
    fpi = yp - yi
    fp = tf.reduce_sum(tf.cast(tf.greater(fpi,0),tf.int32),name='fp')
    specificity = tf.divide(tn,(tn+fp),name='spe')#specificity = tn/(tn+fp)
    accuracy = tf.divide((tn+tp),(tn+tp+fn+fp),name='acc')#accuracy = (tn+tp)/(tn+tp+fn+fp)
    return [accuracy,specificity,sensitivity,tp,tn,fp,fn]

def train_model_with_checkpoints(train_model_args,total_iters,iters,model_name_base,start_iter=0,learning_rates={0:0.01},
                                send_email_per_loop_args=None,send_email_end_args=None):
    """
    *total iters: total number of iterations, it is different from *iters* in that 
     after each iters number is reached, the training is stopped, a model is saved, 
     then the saved model is restored and iters are computed until the sum of iters equals total_iters.
    *learning_rates: is a dictionary that each key represent the number of epoch and the learning associated
     to it. if lr={0:0.01,16:0.001}, when the epoch is 16 the learning rate will change from 0.01 to 0.001.
    *model_name_base: is the prefix for the model name file where the model will be saved, e.g. Model01/model_saved_
    *send_email_per_loop_args: is a dictionary that contains the parameters to be send in an email after each model is saved.
    *send_email_end_args: is a dictionary that contains the parameteres to be send in an email after all the iterations are completed.
     type help(send_mail) for information about its parameters
    
    """
    repetitions = total_iters // iters
    lr = learning_rates[0]
    for _ in range(start_iter,repetitions):
        if _ ==0:
            restore_model=False
        else:
            restore_model=True
        restore_iter = (_)*iters
        save_iter = (_+1)*iters
        restore_name = model_name_base+str(restore_iter)+'_.ckpt'
        save_name = model_name_base+str(save_iter)+'_.ckpt'
        if restore_iter in learning_rates.keys():
            lr=learning_rates[restore_iter]
        print("Repetition:",_+1,"Total:",repetitions)
        train_model_args['iters']=iters
        train_model_args['lr']=lr
        train_model_args['save_name']=save_name
        train_model_args['restore_name']=restore_name
        train_model_args['restore_model']=restore_model
        train_model(**train_model_args)
        if type(send_email_per_loop_args)!=type(None):
            send_mail(**send_email_per_loop_args)
    if type(send_email_end_args)!=type(None):
        send_mail(**send_email_end_args)


def train_model_with_checkpoints_unbalanced(train_model_args,total_iters,iters,model_name_base,start_iter=0,learning_rates={0:0.01},
                                send_email_per_loop_args=None,send_email_end_args=None):
    """
    Similar to train_model_with_checkpoints but the argument *train_model_args['batch_func_params']* 
    instead of having a key 'idf' it has 'pos_df' and 'neg_df' representing a dataframe with the positive
    and negative x and y values respectively.
    In addition, this function will get the number of rows of both dataframes and pick the lowest one.
    This number is the number of samples that will be drawn from both dataframes, joined and shuffled for
    each global iteration/checkpoint.
    
    *total iters: total number of iterations, it is different from *iters* in that 
     after each iters number is reached, the training is stopped, a model is saved, 
     then the saved model is restored and iters are computed until the sum of iters equals total_iters.
    *learning_rates: is a dictionary that each key represent the number of epoch and the learning associated
     to it. if lr={0:0.01,16:0.001}, when the epoch is 16 the learning rate will change from 0.01 to 0.001.
    *model_name_base: is the prefix for the model name file where the model will be saved, e.g. Model01/model_saved_
    *send_email_per_loop_args: is a dictionary that contains the parameters to be send in an email after each model is saved.
    *send_email_end_args: is a dictionary that contains the parameteres to be send in an email after all the iterations are completed.
     type help(send_mail) for information about its parameters
    
    """
    pos_df = train_model_args['batch_func_params']['pos_df']
    neg_df = train_model_args['batch_func_params']['neg_df']
    
    del_key(train_model_args['batch_func_params'],'pos_df')
    del_key(train_model_args['batch_func_params'],'neg_df')
    
    pos_rows = pos_df.shape[0]
    neg_rows = neg_df.shape[0]
    sample_rows = min(pos_rows,neg_rows)
    
    xcol = train_model_args['batch_func_params']['x_col']
    ycol = train_model_args['batch_func_params']['y_col']
    
    repetitions = total_iters // iters
    lr = learning_rates[0]
    for _ in range(start_iter,repetitions):
        if _ ==0:
            restore_model=False
        else:
            restore_model=True
        restore_iter = (_)*iters
        save_iter = (_+1)*iters
        restore_name = model_name_base+str(restore_iter)+'_.ckpt'
        save_name = model_name_base+str(save_iter)+'_.ckpt'
        if restore_iter in learning_rates.keys():
            lr=learning_rates[restore_iter]
        print("Repetition:",_+1,"Total:",repetitions)
        train_model_args['iters']=iters
        train_model_args['lr']=lr
        train_model_args['save_name']=save_name
        train_model_args['restore_name']=restore_name
        train_model_args['restore_model']=restore_model
        # Balance dataframe
        _pos_df = pos_df.sample(sample_rows).reset_index()[[xcol,ycol]]
        _neg_df = neg_df.sample(sample_rows).reset_index()[[xcol,ycol]]
        _tdf = pd.concat([_pos_df,_neg_df])
        train_model_args['batch_func_params']['idf'] = _tdf.sample(sample_rows*2).reset_index()[[xcol,ycol]]
        #Start training
        train_model(**train_model_args)
        if type(send_email_per_loop_args)!=type(None):
            send_mail(**send_email_per_loop_args)
    if type(send_email_end_args)!=type(None):
        send_mail(**send_email_end_args)



def list_variables_from_model(model_name):
    saver = tf.train.import_meta_graph(model_name+'.meta')
    imported_graph = tf.get_default_graph()
    graph_op = imported_graph.get_operations()
    _o = []
    for i in graph_op:
        x = i.name
        _o.append(x)
    return sorted(set(_o))


def find_object_from_image(idf,row,batch_func,batch_func_params,file_col='filename',model_name=None,
                           prediction_threshold=0.75,
                           tmp_dir=None,box_wh=(32,32),strides=(32,32),resize_img=None,
                           return_boxes=False,show_result=False,save_dir=None,v=False,
                          box_overlap_loops=2,overlap_th=0.2,box_color='red',prediction_mode='default'):
    """
    *idf: Dataframe that has in the column *file_col* the location of the file to be predicted.
    *batch_func is the function that will pre-process the image in order to make it predictable. 
    see help(batch_pre_proc_from_df) as an example.
    *batch_func_params is a dictionary that holds the kargs for the *batch_func*
    *model_name is the location to the model .ckpt that will be used for inference.
    *prediction_threshold: is the threshold of confidence over which the inference will consider
    a window as a positive class. The model should output a positive when the first columns is 1.
    *tmp_dir: a directory where the input image will be split and used for inference.
    This directory should be empty, because this function will erase all the content when finishing.
    *box_wh: the width and height of the box for inference. This should match the input shape of the model.
    *strides: how many pixels will be taken as a step to analyze another window. 
    *resize_img: If the input image should be resized before taking the windows.
    *return_boxes: if True, the location of the boxes where a positive sample was found will be returned.
    *show_result: plot the input image and show the squares that contain a positive sample.
    *save_dir: If specified used as the directory where to save the input image with the boxes
     for positive windows.
    *v: Verbosity, set to True if you want to see the progress
    *box_overlap_loops: how many times perform mean_overlap function. In case overlapped windows are present
     repeat overlap reduction. 
    overlap_th: overlap threshold percentage. See more information with help(mean_overlap)
    * box_color: specify the color of the box that will be surrounding the positive objects found in the image
    *prediction_mode: change to "old" if using an old version of the inference function where you need
    Softmax
    """
    if type(tmp_dir)==str:
        clear_directory(tmp_dir)
        image_file = idf.iloc[row]['filename'].split('/')[-1]
        if v==True:
            print("Spliting image:",image_file)
        split_image_and_save(idf,row,file_col,show=False,save_directory=tmp_dir,
                             box_wh=box_wh,resize_img=resize_img,v=v,strides=strides)
        
        raw_imgs_predict = []
        for _ in os.listdir(tmp_dir):
            if image_file in _:
                raw_imgs_predict.append({'filename':tmp_dir+_})
        ## 
        raw_predict_df = pd.DataFrame(raw_imgs_predict)
        ### Inference
        batch_func_params['idf']=raw_predict_df
        batch_func_params['yfunc']=None
        if v==True:
            print("Predicting windows")
        ### Here use layer_output instead of infer because it was trained using an old version
        if prediction_mode=='old':
            predictions = layer_output(batch_func,batch_func_params,model_name=model_name,layer_name='Softmax:0',v=False)
        else:
            predictions = infer_model(batch_func,batch_func_params,model_name=model_name,v=False)
        clear_directory(tmp_dir)
        predictions = np.vstack(predictions)
        predictions = predictions.round(2)
        preds_df = pd.DataFrame(columns=['Positive','Negative'],data=predictions)
        xy_pos = []
        for _ in range(0,raw_predict_df.shape[0]):
            x,y = raw_predict_df.iloc[_].values[0].split('_')[-1].split('.')[0].split('x')
            xy_pos.append({'x':int(x),'y':int(y)})
        xy_df = pd.DataFrame(xy_pos)
        pred_df = pd.concat([preds_df,xy_df],1)
        pred_output = pred_df[pred_df['Positive']>prediction_threshold]
        img_file = raw_imgs_df['filename'][row]
        timg = get_np_image(img_file,resize=resize_img)
        ############ Overlap of boxes
        boxes_list = []
        for _ in range(pred_output.shape[0]):
            row = pred_output.iloc[_]
            x0= row['x']
            y0 = row['y']
            boxes_list.append([x0,x0+box_wh[0],y0,y0+box_wh[1]])
        boxes = boxes_list
        for _ in range(box_overlap_loops):
            boxes = mean_overlap(boxes,overlap_th=overlap_th)
        
        if type(save_dir)==str:
            if os.path.isdir(save_dir):
                pil_img = Image.fromarray(timg)
                from PIL import ImageDraw
                draw = ImageDraw.Draw(pil_img)
                for box in boxes:
                    draw.rectangle(
                    ((box[0],box[2]),(box[1],box[3])),outline=box_color
                    )
                if save_dir[-1]!="/":
                    save_dir+="/"
                save_file = save_dir+"prediction_"+image_file+".jpg"
                if v==True:
                    print("saving in",save_file)
                pil_img.save(save_file, "JPEG")
            else:
                print("Directory",save_dir,"does not exist")
        if show_result==True:
            plt.close()
            plt.imshow(timg)
            for _ in range(min(1000,len(boxes))):
                box = boxes[_]
                xs,xe,ys,ye = box
                x_ = [xs,xe,xe,xs,xs]
                y_ = [ys,ys,ye,ye,ys]
                xy = np.vstack([x_,y_]).transpose()
                plt.plot(xy[:,0],xy[:,1],'r--')
            plt.show()
        if return_boxes==True:
            return boxes

    
def find_max_object_from_image(idf,row,batch_func,batch_func_params,file_col='filename',model_name=None,
                           tmp_dir=None,box_wh=(32,32),strides=(32,32),resize_img=None,
                           v=False,prediction_mode='default',prediction_threshold=0.75,
                               reduce_overlap=True,overlap_th=0.2,overlap_method='join',
                                return_only_max=True,reduce_overlap_repetitions=1):
    """
    ==Description==
    *overlap_method* : join or mean
    
    """
    if type(tmp_dir)==str:
        clear_directory(tmp_dir)
        image_file = idf.iloc[row]['filename'].split('/')[-1]
        if v==True:
            print("Spliting image:",image_file)
        #Tmp zone:
        split_image_and_save(idf,row,file_col,show=False,save_directory=tmp_dir,box_wh=box_wh,resize_img=resize_img,v=v,strides=strides)
        #Get images
        raw_imgs_predict = []
        for _ in os.listdir(tmp_dir):
            if image_file in _:
                raw_imgs_predict.append({'filename':tmp_dir+_})
        ## 
        raw_predict_df = pd.DataFrame(raw_imgs_predict)
        ### Inference
        batch_func_params['idf']=raw_predict_df
        batch_func_params['yfunc']=None
        if v==True:
            print("Predicting windows")
        ### Here use layer_output instead of infer because it was trained using an old version
        if prediction_mode=='old':
            predictions = layer_output(batch_func,batch_func_params,model_name=model_name,layer_name='Softmax:0',v=False)
        else:
            predictions = infer_model(batch_func,batch_func_params,model_name=model_name,v=False)
        clear_directory(tmp_dir)
        predictions = np.vstack(predictions)
        predictions = predictions.round(2)
        preds_df = pd.DataFrame(columns=['Positive','Negative'],data=predictions)
        xy_pos = []
        for _ in range(0,raw_predict_df.shape[0]):
            x,y = raw_predict_df.iloc[_].values[0].split('_')[-1].split('.')[0].split('x')
            xy_pos.append({'x':int(x),'y':int(y)})
        xy_df = pd.DataFrame(xy_pos)
        pred_df = pd.concat([preds_df,xy_df],1)
        pred_df = pred_df[pred_df['Positive']>prediction_threshold].reset_index()[['Positive','x','y']]
        print("size of values above threshold",pred_df.shape[0],"threshold:",prediction_threshold)
        pred_df['x1']=pred_df['x']+box_wh[0]
        pred_df['y1']=pred_df['y']+box_wh[1]
        #Initialize box string
        box_cols = ['x','x1','y','y1']
        box_col_name='box'
        score_col="Positive"
        
        pred_df = box_from_coords(idfx=pred_df,box_cols=box_cols,box_col_name=box_col_name)
        rdf = pred_df
        if reduce_overlap==True:
            rdf = reduce_overlap_df(idf=rdf,box_col=box_col_name,box_cols=box_cols,
                      overlap_th=overlap_th,overlap_method=overlap_method,score_col=score_col)
        if type(rdf)==pd.DataFrame:
            if return_only_max==True:
                print("One only")
                rdf = rdf.sort_values(by='Positive',ascending=False).head(1)
            return rdf.copy()
        else:
            return rdf


def reduce_overlap_df(idf,box_col="box",box_cols=['x','x1','y','y1'],
                      overlap_th=0.2,overlap_method='join',score_col="Positive"):
    xc,xc1,yc,yc1=box_cols
    odf = []
    rdf = [] # Initialize dataframe as list for overlapping boxes
    overlap_list = []
    box_num = idf.shape[0]    
    if box_num > 0:
        overlap_list = overlap_df(idf,box_col=box_col,overlap_th=overlap_th)        
        ### Box overlap methods here:
        if len(overlap_list)>0:
            for l in overlap_list:
                wdf = idf.loc[l,:]
                score = np.round(wdf[score_col].mean(),4)
                if overlap_method=='mean':
                    #Method mean
                    xmin,xmax,ymin,ymax = wdf[xc].mean(),wdf[xc1].mean(),wdf[yc].mean(),wdf[yc1].mean()
                elif overlap_method=='min':
                    #Method mean
                    xmin,xmax,ymin,ymax = wdf[xc].min(),wdf[xc1].min(),wdf[yc].min(),wdf[yc1].min()
                elif overlap_method=='max':
                    #Method mean
                    xmin,xmax,ymin,ymax = wdf[xc].max(),wdf[xc1].max(),wdf[yc].max(),wdf[yc1].max()
                else:
                    #Method join
                    xmin,xmax,ymin,ymax = wdf[xc].min(),wdf[xc1].max(),wdf[yc].min(),wdf[yc1].max()    
                odf.append({xc:xmin,xc1:xmax,yc:ymin,yc1:ymax,score_col:score})
            rdf = pd.DataFrame(odf)                
            #get the rows that are not overlapping
            import itertools
            overlap_rows = list(itertools.chain.from_iterable(overlap_list))
            no_overlap_rows = idf.index.difference(pd.Index(overlap_rows))
            no_ov_df = idf.loc[no_overlap_rows,:]
        if type(rdf)==pd.DataFrame:
            if no_ov_df.shape[0]>0:
                rdf = pd.concat([rdf,no_ov_df],0,ignore_index=True)
        else:
            rdf = idf
    return rdf.copy()



def box_from_coords(idfx,box_cols=['x','x1','y','y1'],box_col_name='box'):
    idf = idfx.copy()
    xc,xc1,yc,yc1=box_cols
    idf[box_col_name]=idf[xc].astype(str)
    for l in [xc1,yc,yc1]:
        idf[box_col_name]+=" "+idf[l].astype(str)
    #Box str to ints        
    idf[box_col_name] = idf[box_col_name].apply(lambda x: [ int(s.split(".")[0]) for s in x.strip().split(' ')])
    return idf.copy()




def overlap_box(box1,box2,overlap_th=0.25):
    """
    Returns True if two boxes overlap over a *overlap_th* value. 0.2  = 20%
    """
    overlap=False
    #x_intersect = max(0,min(box1[1],box2[1])-max(box1[0],box2[0]))/(box1[1]-box1[0])
    #y_intersect = max(0,min(box1[3],box2[3])-max(box1[2],box2[2]))/(box1[3]-box1[2])
    x_intersect = max(0,min(box1[1],box2[1])-max(box1[0],box2[0]))/min((box1[1]-box1[0]),(box2[1]-box2[0]))
    y_intersect = max(0,min(box1[3],box2[3])-max(box1[2],box2[2]))/min((box1[3]-box1[2]),(box2[3]-box2[2]))
    xy_intersect = x_intersect*y_intersect
    if (xy_intersect>overlap_th):
        overlap=True
    return overlap


def overlap_df(idf,box_col="box",overlap_th=0.2):
    """
    Returns a list of lists that contain the ids (index) of the boxes of the dataframe that overlaps
    over the *overlap_th* percentage. *overlap_th* value (0.2 is 20%)
    The box column should be specified in *box_col*.  
    A box has the format:
        [x0,x1,y0,y1] standing for the box coordinates:x0,y0; x1,y0; x1,y1; x0,y1
    
    """
    overlap_list = []
    df_index = idf.index
    for _ in df_index :
        rowa = idf.loc[_]
        overlap_boxes = []
        for __ in df_index:
            if _!=__:
                rowb = idf.loc[__]
                box_a = rowa[box_col]
                box_b = rowb[box_col]
                if overlap_box(box_a,box_b,overlap_th):
                    overlap_boxes.append(__)
        if len(overlap_boxes)>0:
            overlap_boxes.append(_)
        overlap_boxes = sorted(overlap_boxes)
        if len(overlap_boxes)>0:
            overlap_list.append(tuple(overlap_boxes))
    overlap_list = list(set(overlap_list))
    return overlap_list[:]


def cascade_max_object(find_max_args,image_proportions=[4,5,6,7,8],prediction_threshold=0.75,
                      return_boxes=False,save_dir=None,v=False,scales_reduce_overlap=True,
                       scales_overlap_method="join",scales_overlap_th=0.2,scales_reduce_overlap_rep=2,
                          box_color='red',text_color=(255,255,0),output_proportion=8,max_per_scale=False,
                      box_cols = ['x','x1','y','y1'], box_col_name='box',score_col="Positive"):
    tdfs = []
    max_evaluation=pd.DataFrame()
    model_window_shape = find_max_args['box_wh']
    #
    for proportion in image_proportions:
        if v==True:
            print("Evaluating proportion",proportion)    
        xp = model_window_shape[0]*proportion
        yp = model_window_shape[1]*proportion
        find_max_args['resize_img']=[xp,yp]
        print(find_max_args['resize_img'])
        #Predict maximun object
        tdf = find_max_object_from_image(**find_max_args)
        if type(tdf)==pd.DataFrame:
            tdf['proportion']=proportion
            tdfs.append(tdf.copy())
    #Merging boxes and proportions
    xc,xc1,yc,yc1=box_cols
    if len(tdfs)>0:
        tdf = pd.concat(tdfs,0,ignore_index=True).dropna(0,how='all')
        tdf['factor']=tdf['proportion'].apply(lambda x: output_proportion/x)
        for c in box_cols:
            tdf[c] = tdf[c]*tdf['factor']
            tdf[c] = tdf[c].astype(int)
        if scales_reduce_overlap==True:
            #print("Scales reduce overlap ON")
            #print("tdf before overlap reduction\n",tdf)
            for _ in range(scales_reduce_overlap_rep):
                tdf = box_from_coords(idfx=tdf,box_cols=box_cols,box_col_name=box_col_name)
                tdf = reduce_overlap_df(idf=tdf,box_col=box_col_name,box_cols=box_cols,
                      overlap_th=scales_overlap_th,
                        overlap_method=scales_overlap_method,score_col=score_col)
            #print("tdf after overlap reduction\n",tdf)
        max_evaluation = tdf.sort_values(by='Positive',ascending=False).dropna(0,how='all')
    #Save option
    if type(save_dir)==str:
        if os.path.isdir(save_dir):
            idf = find_max_args['idf']
            row = find_max_args['row']
            file_col = find_max_args['file_col']
            full_img_name = idf.iloc[row][file_col]
            image_file = full_img_name.split('/')[-1]
            # Get output proportion
            ox = model_window_shape[0]*output_proportion
            oy = model_window_shape[1]*output_proportion
            #Open image
            pil_img = Image.open(full_img_name)
            pil_img = pil_img.resize((ox,oy),Image.ANTIALIAS)           
            from PIL import ImageDraw
            from PIL import ImageFont
            draw = ImageDraw.Draw(pil_img)
            #Search the maximum box per scale
            limit=1
            if max_evaluation.shape[0]>0:
                if max_per_scale==True:
                    limit=max_evaluation.shape[0]
                for _ in range(limit):
                    row_df = max_evaluation.iloc[_]
                    score=str(row_df['Positive'])[:6]
                    x0,x1,y0,y1=row_df[xc],row_df[xc1],row_df[yc],row_df[yc1]
                    #Draw box
                    draw.rectangle(
                        ((x0,y0),(x1,y1)),outline=box_color
                    )
                    #Write score
                    draw.text((x0,y0), str(score), fill=text_color)    
                #End of boxes
            if save_dir[-1]!="/":
                save_dir+="/"
            save_file = save_dir+"prediction_"+image_file+".jpg"
            if v==True:
                print("saving in",save_file)
            pil_img.save(save_file, "JPEG")
        else:
            print("Directory",save_dir,"does not exist")
    if return_boxes==True:
        return max_evaluation.copy()


def make_dir(idir):
    if not os.path.exists(idir):
        os.makedirs(idir)


