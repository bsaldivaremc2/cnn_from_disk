
# Train a CNN with tensorflow from disk  

This repository provides a package to train, test and infer a Convolutional Neural Network  (CNN) given a pre-processing function that feeds the mini-batch processing.

#how to install it
* git clone  https://github.com/bsaldivaremc2/cnn_from_disk.git
* cd cnn_from_disk
* pip install .

## how to use it
```python

from cnn_from_disk import * 

### Create a model

model = [
    ['conv',{'filter_size':7,'layer_depth':2,'strides':[1,2,2,1]}],
    ['bn'],
    ['relu'],
    ['max_pool',{'kernel':[1,3,3,1],'strides':[1,2,2,1],'padding':'SAME'}],
]

b2 = [ ['res_131',{'depths':[2,2,4]}] for _ in range(0,2) ]
b3 = [ ['max_pool',{'kernel':[1,3,3,1],'strides':[1,2,2,1],'padding':'SAME'}] ]
b4 = [['fc',{'n':12}]]

model.extend(b2)
model.extend(b3)

### Define the functions that will be fed for x and y

### idf is a dataframe that has in one column the filepath to the images and one column with the target "y" that will be processed by the yfunc

xfunc_params={'w_h':(16,16),'resize':True}
batch_func = batch_pre_proc_from_df
batch_func_params = {'idf':df,'xfunc':img_nump_from_file,'yfunc':pass_y,'xfunc_params':xfunc_params,'yfunc_params':{},'x_col':'fullpath','y_col':'target','batch_size':4,'offset':4}

train_model(batch_func,batch_func_params,model=model,iters=1,lr=0.1,
          save_model=True,save_name='Model01/test_save.ckpt',
          restore_model=False,restore_name=None,
          v=True,opt_mode='regression')


```

### Testing

```python
batch_func_params = {'idf':df,'xfunc':img_nump_from_file, 'yfunc':pass_y,'xfunc_params':xfunc_params,'yfunc_params':{},'x_col':'fullpath','y_col':'target', 'batch_size':4,'offset':4}  

test_model(batch_func,batch_func_params,model_name='Model01/test_save.ckpt',opt_mode='regression')

```

### Inference

```python
batch_func_params = {'idf':df,'xfunc':img_nump_from_file,'yfunc':None,'xfunc_params':xfunc_params,'yfunc_params':{},'x_col':'fullpath','y_col':None,'batch_size':4,'offset':4,'inference':True}

infer_model(batch_func,batch_func_params,model_name='Model01/test_save.ckpt',opt_mode='regression',v=True)

``` 

