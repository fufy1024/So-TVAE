# So-TVAE
This is the official implementation for paper Sentiment-oriented Transformer-based Variational Autoencoder Network for Live Video Commenting.

## Requirements
* Python 3.7
* Pytorch 1.7.1+cu110 

## Datasets

Go to the corresponding paper project webpage to obtain relevant datasets:

<1> Dataset from [LiveBot: Generating Live Video Comments Based on Visual and Textual Contexts]

<2> Dataset from [VideoIC: A Video Interactive Comments Dataset and Multimodal Multitask Learning for Comments Generation]

The obtained folder should include:

- data_Livebot

 -- dicts-30000.json
 
 -- res18.pkl
 
- data_VideoIC

 -- dict.json
 
 -- image.pkl

<3> Dataset containing sentimental labels can be directly downloaded from [Google Drive](https://drive.google.com/open?id=13hLJ4yCJVJjz02YB0dyugeE_fcl6EI-B) 

The final obtained folder should include:

- data_Livebot

 -- dicts-30000.json
 
 -- res18.pkl

 -- train-context-skep10.json
 
 -- test-candidate-skep10.json
 
 -- dev-candidate-skep10.json
 
- data_VideoIC

 -- dict.json
 
 -- image.pkl

 -- train-all10.json
 
 -- test-all10.json

 -- dev-all10.json

Modify the dataset path in the code and start running.

### Train

## Train in data_Livebot
```
CUDA_VISIBLE_DEVICES=4 python3 tran_sotvae_1.py -mode train -report 3500 -batch_size 128 -dir [save_path] 
```
## Train in data_VideoIC
```
CUDA_VISIBLE_DEVICES=5 python3 tran_sotvae_2.py -mode train -report 7500 -batch_size 210 -dir [save_path]
```

### Test

## Test in data_Livebot
```
CUDA_VISIBLE_DEVICES=5 python3 tran_sotvae_1.py -mode test  -report 3500 -batch_size 1 -restore [save_path]
```
## Test in data_VideoIC
```
CUDA_VISIBLE_DEVICES=0 python3 tran_sotvae_2.py -mode test_all  -report 8000 -batch_size 210 -restore [save_path]
```





