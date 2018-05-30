import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
from .tf_layers import *
from .proc_utils import send_mail

def create_model(ixs,iys,model=None,opt_mode='classification'):    
    """
    Version 0.106
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

def train_model_with_checkpoints(train_model_args,total_iters,iters,model_name_base,learning_rates={0:0.01},
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
    for _ in range(repetitions):
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

