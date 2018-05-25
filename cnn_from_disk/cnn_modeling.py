import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
from .tf_layers import *

def create_model(ixs,iys,model=None,opt_mode='classification'):    
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
    types_dic = {'conv':0,'bn':0,'relu':0,'max_pool':0,'drop_out':0,'fc':0,'res_131':0}
    last_type = None
    for i,_ in enumerate(model):
        _type=_[0]
        if _type in ['conv','res_131','fc']:
            last_type=_type
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
        if _type=='conv':    
            x=conv(**params)
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
                if last_type in ['conv','res_131']:
                #if model[i-1][0]=='conv':
                    params['prev_conv']=True
            x = fc(**params)
        elif _type=='res_131':
            params['is_training']=train_bool
            x = res_131(**params)
    prev_conv_fcl = False
    #if model[-1][0] in ['conv','res_131']:
    if last_type in ['conv','res_131']:
        prev_conv_fcl=True
    prediction = fc(x,n=class_output,name_scope="FCL",prev_conv=prev_conv_fcl)
    ##Define the model here--UP
    #y_CNN = tf.nn.softmax(prediction,name='Softmax')
    #class_pred = tf.argmax(y_CNN,1,name='ClassPred')
    #loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]),name="loss")
    if opt_mode=='classification':
        y_CNN = tf.nn.softmax(prediction,name='Softmax')        
        class_pred = tf.argmax(y_CNN,1,name='ClassPred')       
        loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]),name="loss")
        acc_,spe_,sen_,tp_,tn_,fp_,fn_ = stats_class(y_CNN,y_)
        stats_dic={'acc':acc_,'spe':spe_,'sen_':sen_,'tp':tp_,'tn':tn_,'fp':fp_,'fn':fn_}
    elif (opt_mode=='regression') or (opt_mode=='yolo'):
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
          v=False,opt_mode='classification'):
    """
    The train is done using a function to return the x and y values for each batch processing.
    *batch_func: is the function that will output the x and y values that will be loaded each mini-batch
    *batch_func_params: is a dictionary that contains the params for the batch_func, at least if should have
    the following keys:
        * idf: the dataframe that specifies the total size of the data
        * batch_size: the size for the mini_batch
        * offset: this value is used to use the next mini_match starting in the position "offset"
    * model: is a list of lists that defines the model that will be created
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
                if v==True:
                    print("Iter:",_,"batch",batch,"batches",batches,"Loss:",l)
            
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

def infer_model(batch_func,batch_func_params,model_name=None,opt_mode='classification',v=False):
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
                predict_params_d = {'classification':['Softmax:0','ClassPred:0'],'regression':'FCL/FC:0','yolo':'FCL/FC:0'}
                predict_params = predict_params_d[opt_mode]
                output = []
                #########
                for batch in range(0,batches):
                    batch_func_params['offset']=batch*batch_size
                    ix = batch_func(**batch_func_params)
                    fdt={'x:0':ix}
                    fd.update(fdt)
                    fcl = s.run(predict_params,feed_dict=fd)
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