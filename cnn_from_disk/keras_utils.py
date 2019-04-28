import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam,Adadelta
from keras.layers import Flatten
from keras.applications import MobileNet
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet121
from keras.applications.xception import Xception
from .proc_utils import *
from .batch_funcs import *

def sens_spec(tp,tn,fp,fn,epsilon=1e-8):
  sens = tp/(tp+fn+epsilon)
  spec = tn/(tn+fp+epsilon)
  return sens, spec

def sum_bool(ibool):
  return tf.keras.backend.sum(tf.keras.backend.cast(ibool,'int64'))

def tp_tn_fp_tn(y_true,y_pred,ik):
  pred = keras.backend.argmax(y_pred,1)
  truth = keras.backend.argmax(y_true,1)
  ones = tf.keras.backend.ones_like(pred,dtype='int64')
  zeros = tf.keras.backend.zeros_like(pred,dtype='int64')
  tz = tf.keras.backend.greater(truth,zeros)
  pz = tf.keras.backend.greater(pred,zeros)
  #and 
  tpz = tf.keras.backend.stack((tz,pz),axis=0)
  tp = sum_bool(keras.backend.all(tpz,axis=0))
  to = tf.keras.backend.less(truth,ones)
  po = tf.keras.backend.less(pred,ones)
  #and 
  tpo = tf.keras.backend.stack((to,po),axis=0)
  tn = sum_bool(keras.backend.all(tpo,axis=0))
  fp = sum_bool(tf.keras.backend.greater(pred,truth))
  fn = sum_bool(tf.keras.backend.greater(truth,pred))
  return {'tp':tp,'tn':tn,'fp':fp,'fn':fn}[ik]
def tp(y_true, y_pred):
  return tp_tn_fp_tn(y_true,y_pred,'tp')

def tn(y_true, y_pred):
  return tp_tn_fp_tn(y_true,y_pred,'tn')

def fp(y_true, y_pred):
  return tp_tn_fp_tn(y_true,y_pred,'fp')

def fn(y_true, y_pred):
  return tp_tn_fp_tn(y_true,y_pred,'fn')

def parse_metrics(metrics,metric_names):
  metrics_dic = {}
  for metric,metric_name in zip(metrics,metric_names):
    if metric_name=='accuracy':
      metrics_dic[metric_name]=np.mean(metric)
    else:
      metrics_dic[metric_name]=sum(metric)  
  sens,spec = sens_spec(*[metrics_dic[k] for k in ["tp","tn","fp","fn"]])
  metrics_dic['sensibility']=sens
  metrics_dic['specificity']=spec
  metric_str=""
  for k in ["Loss","accuracy","tp","tn","fp","fn","sensibility","specificity"]:
    metric_str+=k+":"+str(np.round(metrics_dic[k],6))+"," 
  return metrics_dic.copy(),metric_str

def append_on_file(ifile,istr):
  with open(ifile,"a") as _f:
    _f.write(istr+" \n")


def keras_train_model(train_df,test_df,save_dir,model_name,save_test_over_th=True,model='InceptionResNetV2',model_kargs={'input_shape':(224,224,3),'include_top':False,'weights':'imagenet'},
	x_col='filename',y_col='target',xfunc=bryan_image_generation,batch_func=batch_pre_proc_from_df,
	xfunc_params={'output_wh':[224,224],'add_noise_std':10,'mult_noise_var':0.10, 'shift_add_max':15, 'shift_mult_var' :0.05},
	test_params = {'output_wh':[224,224],'flip_h_prob':0,'flip_v_prob':0,'add_noise_prob':0,'mult_noise_prob':0,'add_shift_prob':0,'mult_shift_prob':0},
	yfunc_params={},	      
	yfunc=pass_y,batch_size=8,learning_rate=0.1,decay=1e-8,iterations = 128,test_save_each_iter = 2,v=True,sensibility_th=0.99,specificity_th=0.99,
	pred_df=None,predict_each_iter=1,post_pred_func=None,pred_save_dir=None,pred_batch_func,pred_batch_func_args={},save_model_with_pickle=False):
    """
    folds = train_test_df_balanced(dft,class_column='class')
    train_df, test_df = folds[cv]
    model_name='MobileNet'
    save_dir = base_dir+'Cactus_'+model_name+"_CV_"+str(cv+1)+"/"
    modelx = keras_train_model(train_df,test_df,save_dir,model_name,save_test_over_th=True,
                  model=model_name,model_kargs={'input_shape':(224,224,3),'include_top':False,'weights':'imagenet'},
    x_col='file',y_col='target',xfunc=bryan_image_generation,batch_func=batch_pre_proc_from_df,
    xfunc_params={'output_wh':[224,224],'add_noise_std':10,'mult_noise_var':0.10, 'shift_add_max':15, 'shift_mult_var' :0.05,'zip_file':trainz},
    test_params = {'output_wh':[224,224],'flip_h_prob':0,'flip_v_prob':0,'add_noise_prob':0,'mult_noise_prob':0,'add_shift_prob':0,'mult_shift_prob':0,'zip_file':trainz},
    yfunc=pass_y,batch_size=8,learning_rate=0.1,decay=1e-8,iterations = 128,test_save_each_iter = 1,v=True,sensibility_th=0.99,specificity_th=0.99)
    """
    print("Testing functions")
    xdf, ydf = batch_func(train_df,xfunc,yfunc,xfunc_params=xfunc_params,yfunc_params=yfunc_params,x_col=x_col,y_col=y_col,batch_size=batch_size,offset=0,inference=False)
    print("Function passed")
    models = {'MobileNet':MobileNet,'InceptionResNetV2':InceptionResNetV2,'DenseNet121':DenseNet121,'Xception':Xception}
    base_model = models[model](**model_kargs)
    x=base_model.output
    x = Flatten()(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3
    total_classes=ydf.shape[1]
    preds=Dense(total_classes,activation='softmax')(x) #final layer with softmax activation
    model=Model(inputs=base_model.input,outputs=preds)
    #
    model.compile(optimizer=Adadelta(lr=learning_rate, rho=0.95, epsilon=None, decay=decay),loss='categorical_crossentropy', metrics = ['accuracy',tp,tn,fp,fn])
    best_model_name = model_name+"_best"
    import time
    make_dir(save_dir)
    #
    date_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    print("Date time",date_time)
    log_file_name = save_dir+model_name+'_log_'+date_time+'.log'
    with open(log_file_name,"w") as _f:
      _f.write(model_name+" "+date_time+" \n")
    #
    images_n = train_df.shape[0]
    batches = int(np.ceil(images_n / batch_size))
    print("Train number of Batches",batches)
    best_metric_keys =['sensibility','specificity']
    best_dic = {}
    for k in best_metric_keys:
      best_dic[k]=0
    print("Starting Iterations")
    for iterx in range(iterations):
      losss,accs,tps,tns,fps,fns = [],[],[],[],[],[]
      for batch in range(batches):
        offset = batch*batch_size
        #
        xdf,ydf = batch_func(train_df,xfunc,yfunc,xfunc_params=xfunc_params,x_col=x_col,y_col=y_col,batch_size=batch_size,offset=offset,inference=False)
        metrics = model.train_on_batch(xdf,ydf)
        loss = metrics[0]
        for ix,_ in enumerate([losss,accs,tps,tns,fps,fns]):
            _.append(metrics[ix])
        metric_names = ["Loss","accuracy","tp","tn","fp","fn"]
        iter_metric_str = ""
        for _ in range(len(metric_names)):
            iter_metric_str+=metric_names[_]+":"+str(metrics[_])+"."
        log_str = "Iter:"+str(iterx)+"/"+str(iterations)+". batch: "+str(batch)+"/"+str(batches)+"."+iter_metric_str
        if v==True:
            print(log_str)
        append_on_file(log_file_name,log_str)
      metrics = [losss,accs,tps,tns,fps,fns]
      metric_names = ["Loss","accuracy","tp","tn","fp","fn"]
      metric_dic,metric_str = parse_metrics(metrics,metric_names)
      log_str = "Iter:"+str(iterx)+"/"+str(iterations)+","+metric_str
      append_on_file(log_file_name,log_str)
      if v==True:
         print(log_str)
      #Testing
      if iterx%test_save_each_iter==0:
         print("Testing")
         losss,accs,tps,tns,fps,fns = [],[],[],[],[],[]
         save_model(model,save_dir+model_name,save_model_with_pickle)
         #Test model:
         metrics = []
         test_batches = int(np.ceil(test_df.shape[0]/ batch_size))
         for test_batch in range(test_batches):
            offset = test_batch*batch_size
            xdf,ydf = batch_func(test_df,xfunc,yfunc,xfunc_params=test_params,x_col=x_col,y_col=y_col,batch_size=batch_size,offset=offset,inference=False)
            metrics = model.evaluate(xdf,ydf,verbose=0)
            iter_metric_str = ""
            for _ in range(len(metric_names)):
                iter_metric_str+=metric_names[_]+":"+str(metrics[_])+"."
            test_batch_str = "Testing. Batch"+str(test_batch)+"/"+str(test_batches)+"."+iter_metric_str
            if v==True:
                print(test_batch_str)
            append_on_file(log_file_name,test_batch_str )
            for ix,_ in enumerate([losss,accs,tps,tns,fps,fns]):
                _.append(metrics[ix])
         metrics = [losss,accs,tps,tns,fps,fns]
         metric_dic,metric_str = parse_metrics(metrics,metric_names)
         log_str = "Iter:"+str(iterx)+"/"+str(iterations)+". Evaluation:"+metric_str
         append_on_file(log_file_name,log_str)
         if v==True:
           print(log_str)
         if (metric_dic['sensibility']>sensibility_th) and (metric_dic['specificity']>specificity_th):
            save_tail = metric_str.replace(":","_").replace(",","_")
            best_model_save_name = save_dir+best_model_name+save_tail+"_iter_"+str(iterx)+".h5"
            if save_test_over_th==True:
                save_model(model,best_model_save_name,save_model_with_pickle)
                print("Saved a new best model with metric:",best_model_save_name)
    #
    #pred_df=None,predict_each_iter=1,post_pred_func=None,pred_save_dir=None,
        if (type(pred_df)!=type(pred_df)) and (iterx%predict_each_iter==0):
            pred_save_name = 'iter_'+str(iterx)+"_predictions"
            predictions = predict_model(pred_df,model,pred_batch_func,pred_batch_func_args)
            make_dir(pred_save_dir)
            if type(post_pred_func)!=type(None):
                post_pred_func(predictions)
            else:
                if pred_save_dir[-1]!='/':
                    pred_save_dir+='/'
                save_obj(predictions,pred_save_dir+save_name)
    save_model(model,save_dir+model_name,save_model_with_pickle)
    return model

def predict_model(idf,imodel,batch_func,batch_func_args):
  """
  Example of usage
  batch_func_args = {'xfunc':bryan_image_generation,'yfunc':pass_y,
                  'xfunc_params':{'output_wh':[224,224],'flip_h_prob':0,'flip_v_prob':0,'add_noise_prob':0,'mult_noise_prob':0,'add_shift_prob':0,'mult_shift_prob':0,'zip_file':trainz},
                  'yfunc_params':{},'x_col':'file','y_col':'target',
                           'batch_size':8,'offset':0,'inference':True}
  p = predict_model(train_df,modelx,batch_pre_proc_from_df,batch_func_args)
  """
    n = idf.shape[0]
    batch_size = batch_func_args['batch_size']
    batch_func_args['inference']=True
    batches = int(np.ceil(n/ batch_size))
    preds = []
    for batch in range(batches):
        batch_func_args['offset']=batch*batch_size
        xdf = batch_func(idf,**batch_func_args)
        predx = imodel.predict(xdf)
        preds.append(predx.copy())
    preds = np.vstack(preds)
    return preds.copy()

 
def save_obj(obj,name):
  import pickle
  with open(name+'.pkl', 'wb') as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print(name+".pkl saved")

def load_obj(name):
  import pickle
  with open(name+'.pkl', 'rb') as f:
    return pickle.load(f)

def save_model(imodel,model_save_name,save_model_with_pickle=False):
    if save_model_with_pickle==True:
        save_obj(imodel,model_save_name)
    else:
        imodel.save(model_save_name+'.h5')


