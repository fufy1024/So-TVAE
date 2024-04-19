'''
 @Date  : 8/22/2018
 @Author: Shuming Ma
 @mail  : shumingma@pku.edu.cn
 @homepage: shumingma.com
'''
#####-----------------------------VideoIC dataset version-----------------------
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as Optim
from torch.autograd import Variable
import torchvision
from torchvision.transforms import functional as F

import json
import argparse
import time
import os
import random
import pickle
from PIL import Image
import numpy as np
from metrics import *
from modules_labelvae1 import *

#from coatten_XLAN import *
#from PID import PIDControl
from init_clusters import init_clusters
import evaluation
from evaluation import PTBTokenizer, Cider
from evaluation.bleu import Bleu

from torch.optim.lr_scheduler import LambdaLR 
from coa_MAC1 import coa_MAC
from ADD_BT import obtain_sample_relation_models

parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-n_emb', type=int, default=512, help="Embedding size")
parser.add_argument('-n_hidden', type=int, default=512, help="Hidden size")
parser.add_argument('-d_ff', type=int, default=2048, help="Hidden size of Feedforward")
parser.add_argument('-n_head', type=int, default=8, help="Number of head")
parser.add_argument('-n_block', type=int, default=6, help="Number of block")
parser.add_argument('-batch_size', type=int, default=64, help="Batch size")
parser.add_argument('-vocab_size', type=int, default=30000, help="Vocabulary size")
parser.add_argument('-epoch', type=int, default=50, help="Number of epoch")
parser.add_argument('-report', type=int, default=3000, help="Number of report interval")
parser.add_argument('-lr', type=float, default=1e-4, help="Learning rate")#3e-4
parser.add_argument('-dropout', type=float, default=0.1, help="Dropout rate")
parser.add_argument('-restore', type=str, default='', help="Restoring model path")
parser.add_argument('-mode', type=str, default='train', help="Train or test")
parser.add_argument('-dir', type=str, default='ckpt', help="Checkpoint directory")
parser.add_argument('-max_len', type=int, default=20, help="Limited length for text")
parser.add_argument('-n_img', type=int, default=5, help="Number of input images")
parser.add_argument('-n_com', type=int, default=5, help="Number of input comments")

parser.add_argument('-impatience', type=int, default=12, help='Number of evaluation rounds for early stopping')
parser.add_argument('-output', default='prediction.json', help='Output json file for generation')
#contral VAE:
parser.add_argument('--latent_size', type=int, default=64, help="Embedding size")
parser.add_argument('--exp_kl', type=float, default=1, help="desired KL divergence.")
parser.add_argument('--Kp', type=float, default=0.0, help="Kp for pid.")
parser.add_argument('--Ki', type=float, default=-0.0003, help="Kp for pid.")
parser.add_argument('--z_class', type=int, default=3, help="Kp for pid.")

parser.add_argument('-output_bleu', default='VIC_BS128_lr3e5', help='Output json file for generation')

opt = parser.parse_args()




c_means, c_sigma=init_clusters(num_clusters=opt.z_class, latent_size=opt.latent_size,c_m_file=f'./cluster_means_class{opt.z_class}.pickle')#c_means为随机结果[class,latent_size],c_sigma=0.1
c_means=c_means.cuda()
c_sigma=0.2
print('c_means',c_means)
print('c_sigma',c_sigma)

#对label_mean的优先处理
def to_one_hot(y, n_class):
    return np.eye(n_class)[y]

data_path = '/mnt/data10t/bakuphome20210617/ffy2020/ffy2020/VC/VideoIC/task/comments_generation/processed_data/'#change dataset

if opt.z_class==3:
    label_all=to_one_hot([0,1,2], opt.z_class)#class=3
elif opt.z_class==5:
    label_all=to_one_hot([0,1,2,3,4], opt.z_class)#class=5
elif opt.z_class==10:   
    label_all=to_one_hot([0,1,2,3,4,5,6,7,8,9], opt.z_class)#class=10

train_path, test_path, dev_path = data_path + 'setup/train-all10.json', data_path + 'setup/test-all10.json', data_path + 'setup/dev-all10.json'

vocab_path = data_path + 'dict.json'
img_path = data_path + 'image.pkl'

label_all=torch.tensor(label_all,dtype=torch.float32).cuda() 
label_all_mean=torch.mm(label_all,c_means).unsqueeze(1)#[class,1,64]




vocabs = json.load(open(vocab_path, 'r', encoding='utf8'))['word2id']
rev_vocabs = json.load(open(vocab_path, 'r', encoding='utf8'))['id2word']
opt.vocab_size = len(vocabs)

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


if not os.path.exists(opt.dir):
    os.mkdir(opt.dir)

def load_from_json(fin):
    datas = []
    for line in fin:
        data = json.loads(line)
        datas.append(data)
    return datas

def dump_to_json(datas, fout):
    for data in datas:
        fout.write(json.dumps(data, sort_keys=True, separators=(',', ': '), ensure_ascii=False))
        fout.write('\n')
    fout.close()


class Model(nn.Module):

    def __init__(self, n_emb, n_hidden, vocab_size, dropout, d_ff, n_head, n_block):
        super(Model, self).__init__()
        self.n_emb = n_emb
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.embedding = nn.Sequential(Embeddings(n_hidden, vocab_size), PositionalEncoding(n_hidden, dropout))
        self.video_encoder = VideoEncoder(n_hidden, d_ff, n_head, dropout, n_block)
        self.comment_decoder = CommentDecoder(n_hidden, d_ff, n_head, dropout, n_block)
        
        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=0)

        #co_attiention
        self.coa=coa_MAC(out_size=opt.z_class,LAYER=3,HIDDEN_SIZE=n_hidden,FLAT_MLP_SIZE=n_hidden,FLAT_GLIMPSES=1,DROPOUT_R=dropout,FLAT_OUT_SIZE=1024)
        self.T_enc = nn.LSTM(input_size=n_hidden,hidden_size=n_hidden,num_layers=1,batch_first=True)


        #VAE:
        self.latent_size=opt.latent_size
        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(n_hidden)
        self.z_att = MultiHeadedAttention(n_head, n_hidden)
        self.feed_forward = PositionwiseFeedForward(n_hidden, d_ff, dropout)
        self.z_mean = nn.Linear(n_hidden, self.latent_size )
        self.z_logstd = nn.Linear(n_hidden, self.latent_size )
        self.z_emb = nn.Linear(self.latent_size, n_hidden)

        #label
        self.label_emb = nn.Embedding(opt.z_class, n_hidden, padding_idx=0)
        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(n_hidden)
        self.label_att = MultiHeadedAttention(n_head, n_hidden)
     
        self.loss_c=nn.CrossEntropyLoss(reduction='none')

        self.output_layer = nn.Linear(self.n_hidden*2, self.vocab_size)
        
   
        self.X_linear=nn.Linear(2048, 512)

    def encode_img(self, X):
        out = self.video_encoder(X)
        return out

    def encode_text(self, X):
        T_out = self.embedding(X)#[64, 100, 512]
        T_out,_ = self.T_enc(T_out)
        return T_out

    def maskVAE(self,label,label_mean, c_sigma,seq):
 
        GT_emb=self.embedding(seq) #[bs,100,512] 

        label_emb=self.label_emb(label.long())#[bs,z_class]=>[bs,z_class,512]
        label_att=self.label_att(label_emb,label_emb,label_emb) 
        label_att = self.lnorm2(label_emb + self.dropout2(label_att))
        GT_att=self.z_att(label_att,GT_emb,GT_emb)#[bs, 20, 512]   
        GT_att = self.lnorm1(label_att + self.dropout1(GT_att))
        f_z = self.feed_forward(GT_att)
        Ht = f_z[:,-1,:]#[bs,512]     
        lz_mean = self.z_mean(Ht)#[bs,150]
        lz_logstd = self.z_logstd(Ht)#[bs,150]
        sample =torch.randn(lz_mean.shape).cuda()#[bs,64]
        z=lz_mean + sample*(torch.sqrt((torch.exp(lz_logstd))))#[bs,64]
       
        mask=torch.randperm(self.latent_size).cuda()
        for i_mask in range(20):
            sample_i=torch.randn(z.shape[0]).cuda()
            z[:,mask[i_mask]]=(label_mean[:,mask[i_mask]]+sample_i*c_sigma)

        z_emb=self.z_emb(z).unsqueeze(1)
        return z_emb,lz_mean,lz_logstd
    def decode(self,  x,x_out,y_out,x_out1,y_out1,z_emb, mask,label_emb):
        embs = self.embedding(x)

        out = self.comment_decoder(embs, x_out,y_out,z_emb, mask,label_emb)
        out1 = self.comment_decoder(embs, x_out1,y_out1,z_emb, mask,label_emb)#add _BT
        out = torch.cat([out1, out], dim=-1)
        out = self.output_layer(out)##[ 90, 19, 30005]
        return out

    def decode_test(self,  x,x_out,y_out,z_emb, mask,label_emb):
        embs = self.embedding(x)
        

        out = self.comment_decoder(embs, x_out,y_out,z_emb, mask,label_emb)
        out = torch.cat([out, out], dim=-1)
        out = self.output_layer(out)##[ 90, 19, 30005]
        return out
  

    def forward(self, label,label_mean, c_sigma,X, Y, T):
        #Y:[64, 20]"comment"; T:[64, 100]"context"，X:[64, 5, 2048] "image"
        if X.shape[-1]!=opt.n_hidden :
            X=self.X_linear(X)
        self.T=T
        T = self.encode_text(T)

        #coa_atten
        z_out,x_out,y_out,x_out1,y_out1=self.coa(X,T,self.T)#z_out:[B,z_class]

        #VAE
        z_emb,lz_mean,lz_logstd=self.maskVAE(label,label_mean, c_sigma,Y)

        #label_emb
        label_emb=self.label_emb(label.long())#[bs,z_class]=>[bs,z_class,512]
        
        mask = Variable(subsequent_mask(Y.size(0), Y.size(1)-1), requires_grad=False).cuda()
        outs = self.decode( Y[:,:-1], x_out,y_out,x_out1,y_out1, z_emb, mask,label_emb)#[19, 90, 30005]
        Y = Y.t()
        outs = outs.transpose(0, 1)
       
        loss = self.criterion(outs.contiguous().view(-1, self.vocab_size),
                              Y[1:].contiguous().view(-1))
   
        l=torch.Tensor([torch.argmax(item) for item in label]).long().cuda()
        loss_z=torch.mean(self.loss_c(z_out,l))

        len_vaild=len(torch.nonzero(loss))
        return torch.sum(loss)/len_vaild,lz_mean,lz_logstd,loss_z

    def pred_z(self,z_out):#z_out:[B,z_class]
        _,id_z=torch.max(z_out, dim=-1, keepdim=True)#id_z:[B,1]
        return id_z

    def generate(self, z, label_i,x_out1, y_out1):#label：[1,3]，z:[1,64],
        
        x_out=x_out1.repeat(z.size(0),1,1)
        y_out=y_out1.repeat(z.size(0),1,1)
        
        #VAE
        z_emb=self.z_emb(z).unsqueeze(1)#[1,64]=>[1,64,512]
    
        #label_emb
        label_emb=self.label_emb(label_i.long())#[bs,z_class]=>[bs,z_class,512]

        ys = torch.ones(z.size(0), 1).long()
       
        with torch.no_grad():
            ys = Variable(ys).cuda()
        for i_gen in range(opt.max_len):
            #print('yyyy',ys.shape, x_out.shape,y_out.shape ,z_emb.shape,label_emb.shape)
            out = self.decode_test(ys, x_out,y_out ,z_emb,
                              Variable(subsequent_mask(ys.size(0), ys.size(1))).cuda(),label_emb)#for循环结束shape=[1, 20, 30005]          
            prob = out[:, -1]#[1, 30005]
            #prob=torch.softmax(prob,-1)

            _, next_word = torch.max(prob, dim=-1, keepdim=True)#size均为[1,1]
            
            next_word = next_word.data
            #print('prob',prob.shape,next_word.shape)
            ys = torch.cat([ys, next_word], dim=-1)
        return ys[:, 1:]
   

    
    def z_produce(self,i_emotion):
        label_i=label_all[i_emotion]  
        
        sample =torch.randn([1,opt.latent_size]).cuda()
        z=label_all_mean[i_emotion] + sample*c_sigma#[1,64]
        return z ,label_i.unsqueeze(0)
    
      
      

    def ranking(self, X, Y, T,data,cider):

        if X.shape[-1]!=opt.n_hidden :
            X=self.X_linear(X)
        nums = len(Y)#100

        self.T=T.unsqueeze(0)#[1,100]
        T1 = self.encode_text(T.unsqueeze(0))#[1,100,512]
        X1=X.unsqueeze(0)#[1,5,512]
        T = T1.repeat(nums, 1, 1)#shape 由[a,b]=>[nums,a,b]        
        X = X1.repeat(nums, 1, 1)

        #coa_atten
        z_out1,x_out1,y_out1=self.coa.for_test(X1,T1,self.T)
        x_out=x_out1.repeat(nums, 1, 1)
        y_out=y_out1.repeat(nums, 1, 1)

        #预测label
        label_pred=self.pred_z(z_out1)#[1,1]
        id_gen=label_pred.squeeze()
        #print('id_gen',id_gen)

        mask = Variable(subsequent_mask(Y.size(0), Y.size(1) - 1), requires_grad=False).cuda()
        
        z,label_gen=self.z_produce(id_gen)#[1,64]
        

        z_emb=self.z_emb(z).unsqueeze(0)#[1,64]=>[1,512]=>[1,1,512]
        z_emb=z_emb.repeat(nums,1,1)#[100,1,512]

        #label_emb    
        label_emb=self.label_emb(label_gen.long()).repeat(nums, 1, 1)#[bs,z_class]=>[bs,z_class,512]
       
        outs = self.decode_test(Y[:, :-1], x_out,y_out,z_emb, mask,label_emb)
        
        Y = Y.t()
        outs = outs.transpose(0, 1)
        #outs:[19, 100, 30005] ; Y:[20,100] ; 
        loss = self.criterion(outs.contiguous().view(-1, self.vocab_size),
                              Y[1:].contiguous().view(-1))

        loss_i=[]
        loss = loss.view(-1, nums).transpose(1,0)#[100,19]
        for i in range(loss.shape[0]):
            len_vaild_i=len(torch.nonzero(loss[i]))
            loss_i.append(torch.sum(loss[i])/len_vaild_i)
        loss=torch.stack(loss_i)

        return torch.sort(loss, dim=0, descending=False)[1]#,caps_gen_i  #[100]从小到大排序，并返回对应位置编号
    
    
    def ranking_comment(self, X, Y, T,data,cider,label):
        nums = len(Y)#100

        if X.shape[-1]!=opt.n_hidden :
            X=self.X_linear(X)
    
        self.T=T.unsqueeze(0)#[1,100]
        T1 = self.encode_text(T.unsqueeze(0))#[1,100,512]
        X1=X.unsqueeze(0)#[1,5,512]
        T = T1.repeat(nums, 1, 1)#shape 由[a,b]=>[nums,a,b]        
        X = X1.repeat(nums, 1, 1)

        #coa_atten
        z_out1,x_out1,y_out1=self.coa.for_test(X1,T1,self.T)
        x_out=x_out1.repeat(nums, 1, 1)
        y_out=y_out1.repeat(nums, 1, 1)
        
        mask = Variable(subsequent_mask(Y.size(0), Y.size(1) - 1), requires_grad=False).cuda()

        

        z,label_gen=self.z_produce(label)#[1,64]
        

        z_emb=self.z_emb(z).unsqueeze(0)#[1,64]=>[1,512]=>[1,1,512]
        z_emb=z_emb.repeat(nums,1,1)#[100,1,512]

        #label_emb    
        label_emb=self.label_emb(label_gen.long()).repeat(nums, 1, 1)#[bs,z_class]=>[bs,z_class,512]
       
        outs = self.decode_test(Y[:, :-1], x_out,y_out,z_emb, mask,label_emb)
        
        Y = Y.t()
        outs = outs.transpose(0, 1)
        #outs:[19, 100, 30005] ; Y:[20,100] ; 
        loss = self.criterion(outs.contiguous().view(-1, self.vocab_size),
                              Y[1:].contiguous().view(-1))
  
      
        loss_i=[]
        loss = loss.view(-1, nums).transpose(1,0)#[100,19]
        for i in range(loss.shape[0]):
            len_vaild_i=len(torch.nonzero(loss[i]))
            loss_i.append(torch.sum(loss[i])/len_vaild_i)
        loss=torch.stack(loss_i)

        return torch.sort(loss, dim=0, descending=False)[1]

     

class DataSet(torch.utils.data.Dataset):

    def __init__(self, data_path, vocabs, is_train=True, imgs=None):
        print("starting load...")
        start_time = time.time()
        self.datas = load_from_json(open(data_path, 'r', encoding='utf8'))
        if imgs is not None:
            self.imgs = imgs
        else:
            self.imgs = torch.load(open(img_path, 'rb'))
        print("loading time:", time.time() - start_time)

        self.vocabs = vocabs
        self.vocab_size = len(self.vocabs)
        self.is_train = is_train

    def __len__(self):
        return len(self.datas)
        #return 480

    def __getitem__(self, index):
        data = self.datas[index]
        video_id, video_time = data['video'], data['time']

        X = self.load_imgs(video_id, video_time)
        
        context=data['context']
        context_str=''
        for i,keys in enumerate(context):
              
                i_context=" <&&&> ".join(context[keys])
                if context_str!='':
                    context_str=context_str+" <&&&> "+i_context
                else:
                    context_str=i_context   
       
        T = self.load_comments(context_str)


        comment = data['comment']
        Y = DataSet.padding(comment, opt.max_len)

 
        label_emo=data['skep']
        if opt.z_class==3:
            label_emo=self.label_10to3(label_emo)
        elif    opt.z_class==5:
            label_emo=self.label_10to5(label_emo)
        
        
        label = to_one_hot(label_emo, opt.z_class)

        label=torch.tensor(label,dtype=torch.float32)

        

        return X, Y, T, label

    def get_img_and_candidate(self, index):
        data = self.datas[index]
        video_id, video_time = data['video'], data['time']

        X = self.load_imgs(video_id, video_time)
        
        context=data['context']
        context_str=''
        for i,keys in enumerate(context):
       
                i_context=" <&&&> ".join(context[keys])
                if context_str!='':
                    context_str=context_str+" <&&&> "+i_context
                else:
                    context_str=i_context   
        T=self.padding(context_str, 100)#[len_context]=[50]

        Y = [DataSet.padding(c, opt.max_len) for c in data['candidate']]

        label_emo=data['skep']
        if opt.z_class==3:
            label_emo=self.label_10to3(label_emo)
        elif    opt.z_class==5:
            label_emo=self.label_10to5(label_emo)

        return X, torch.stack(Y), T, data,label_emo

    def load_imgs(self, video_id, video_time):
        if opt.n_img == 0:
            return torch.stack([self.imgs[video_id][video_time].fill_(0.0) for _ in range(5)])

        surroundings = [0, -1, 1, -2, 2, -3, 3, -4, 4]
        X = []
        for t in surroundings:
            if video_time + t >= 0 and video_time + t < len(self.imgs[video_id]):
                i_image=self.imgs[video_id][video_time + t]#[2048]
                X.append(torch.from_numpy(i_image))
                if len(X) == opt.n_img:
                    break
        return torch.stack(X)

    def load_comments(self, context):
        if opt.n_com == 0:
            return torch.LongTensor([1]+[0]*opt.max_len*5+[2])
        return DataSet.padding(context, opt.max_len*opt.n_com)

    @staticmethod
    def padding(data, max_len):
        data = data.split()
        if len(data) > max_len-2:
            data = data[:max_len-2]
        Y = list(map(lambda t: vocabs.get(t, 3), data))
        Y = [1] + Y + [2]
        length = len(Y)
        Y = torch.cat([torch.LongTensor(Y), torch.zeros(max_len - length).long()])
        return Y

    @staticmethod
    def transform_to_words(ids):
        words = []
        for id in ids:
            if id == 2:
                break
            if id.item() not in {0,1,2,3,4}:
                words.append(rev_vocabs[str(id.item())])
        return " ".join(words)

    def label_10to5(self,label):
        if label<2:#0,1
            label_new=0
        elif 2<=label<4:#2,3
            label_new=1
        elif 4<=label<6:#4,5
            label_new=2
        elif 6<=label<8:#6,7
            label_new=3
        else:#8,9
            label_new=4      
        return label_new    

    def label_10to3(self,label):
        if label<3:#0,1,2
            label_new=0
        elif 3<=label<7:#3,4,5,6
            label_new=1
        else:#7,8,9
            label_new=2     
        return label_new        


def lambda_lr(s):
    s=s+1
    if s <= 4:#1,2,3,4
        lr=3e-5
    else:
        lr=opt.lr   
        if s%2==0: 
                lr=lr*3/4#3/4
    return lr

        



def get_dataset(data_path, vocabs, is_train=True, imgs=None):
    return DataSet(data_path, vocabs, is_train=is_train, imgs=imgs)

def get_dataloader(dataset, batch_size, is_train=True):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train)

def save_model(path, model):
    model_state_dict = model.state_dict()
    torch.save(model_state_dict, path,_use_new_zipfile_serialization=False)

def kl_distance(mean,logstd):
    kld=1/2 * (mean**2 + torch.exp(logstd) - logstd - 1) 
    kld=torch.mean(kld,0)
    return torch.sum(kld)


def kl_distance_GMM(lz_mean,lz_logstd,label_mean)    :
    kld=1/2 * ( math.log(c_sigma**2) - lz_logstd + (torch.exp(lz_logstd)+(lz_mean-label_mean)**2)/(c_sigma**2) - 1 ) 
    kld=torch.mean(kld,0)
    return torch.sum(kld)

def train():
    #train_set = get_dataset(train_path, vocabs, is_train=True)
    dev_set = get_dataset(dev_path, vocabs, is_train=False, imgs=train_set.imgs)
    #train_batch = get_dataloader(train_set, opt.batch_size, is_train=True)
    model = Model(n_emb=opt.n_emb, n_hidden=opt.n_hidden, vocab_size=opt.vocab_size,
                  dropout=opt.dropout, d_ff=opt.d_ff, n_head=opt.n_head, n_block=opt.n_block)
    if opt.restore != '':
        model_dict = torch.load(opt.restore)
        model.load_state_dict(model_dict)
    model.cuda()
    optim = Optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=1)
    scheduler = LambdaLR(optimizer=optim, lr_lambda=lambda_lr )
    
    best_score = -1000000
    impatience = 0

    for i in range(opt.epoch):
        model.train()
        report_loss,report_loss1,report_lossvae, start_time, n_samples,report_lossz = 0, 0,0,time.time(), 0, 0
        count, total = 0, len(train_set) // opt.batch_size + 1
        print('train_batch',len(train_batch))
        
        for it,batch in enumerate(train_batch):
            model.zero_grad()
            X, Y, T ,label= batch 
            label=label.cuda()#[bs,class]
            label_mean=torch.mm(label,c_means)#[bs,latent_size]

            X = Variable(X).cuda()
            Y = Variable(Y).cuda()
            T = Variable(T).cuda()
            loss_1 , mean, logstd,loss_z= model(label,label_mean, c_sigma, X, Y, T)
        
                
            loss_vae=kl_distance_GMM(mean,logstd,label_mean)
            loss = loss_1 + loss_vae*2+loss_z*0.3
    
            
            loss.backward()
            optim.step()
           
            report_loss += loss.item()
            report_loss1 += loss_1.item()
            report_lossvae+=loss_vae.item()
            report_lossz+=loss_z.item()
            n_samples += len(X.data)
            if count%200==0:
                print('epoch:',i+1,'count:',count,'report_loss:',report_loss / (count+1),'loss_rc:',report_loss1 / (count+1),'lossvae:',report_lossvae / (count+1),'loss_z',report_lossz/ (count+1))
            
            count += 1
           
            
            if count % opt.report == 0 or count ==len(train_batch):# count == total:
                print('%d/%d, epoch: %d, report_loss: %.3f, time: %.2f'
                      % (count, total, i+1, report_loss / n_samples, time.time() - start_time))
                model.eval()
                score = eval(dev_set, model)
                model.train()
                if score > best_score:
                    best_score = score
                    impatience = 0
                    print('new best_score:',best_score)
                    save_model(os.path.join(opt.dir, 'best_checkpoint.pt'), model)
                else:
                    impatience += 1
                    print('Impatience: ', impatience, 'best score: ', best_score)
                    save_model(os.path.join(opt.dir, 'checkpoint.pt'), model)
                    if impatience > opt.impatience:
                        print('Early stopping!')
                        quit()

                report_loss, start_time, n_samples = 0, time.time(), 0
        opt.lr= float(np.array(scheduler.get_last_lr()))  
        scheduler.step() 
        print('learning rate',i,opt.lr)
    return model


def eval(dev_set, model):
    print("starting evaluating...")
    start_time = time.time()
    model.eval()
    predictions, references = [], []
    dev_batch = get_dataloader(dev_set, opt.batch_size, is_train=False)
    print('dev_batch',len(dev_batch))
    loss = 0
    count_eval=0
    for batch in dev_batch:
        X, Y, T , label= batch
        label=label.cuda()#[bs,class]
        label_mean=torch.mm(label,c_means)#[bs,latent_size]
        #X:[64, 5, 512];Y:[64, 20];T:[64, 100]
        with torch.no_grad(): 
            X = Variable(X).cuda()
            Y = Variable(Y).cuda()
            T = Variable(T).cuda()
            loss_1 , mean, logstd,loss_z= model(label,label_mean, c_sigma, X, Y, T)
            #loss_vae=kl_distance(mean,logstd)
            loss_vae=kl_distance_GMM(mean,logstd,label_mean)

            loss_i=loss_1+loss_vae+loss_z
        loss += loss_i.item()
        count_eval+=1
        if count_eval%200==0:
            print(count_eval)

     
    print(loss)
    print("evaluting time:", time.time() - start_time)

    return -loss


def test(test_set, model,cider_train):
    print("starting testing...")
    start_time = time.time()
    model.eval()
    predictions, references = [], []
    
    outputs=[]
    for i in range(len(test_set)):
        X, Y, T, data, label_emo = test_set.get_img_and_candidate(i)
        
        with torch.no_grad(): 
            X = Variable(X).cuda()
            Y = Variable(Y).cuda()
            T = Variable(T).cuda()
            if True:
                #ids= model.ranking_comment(X, Y, T,data,cider_train, label_emo)#[100] #,cap_gen 
                ids= model.ranking(X, Y, T,data,cider_train)#[100] #,cap_gen 
                #ids,cap_gen = model.ranking_best(X, Y, T,data,cider_train)#[100] 
         
                candidate = []
                comments = list(data['candidate'].keys())#100条候选评论
                for id in ids:
                    candidate.append(comments[id])
                predictions.append(candidate)
                references.append(data['candidate'])
        if i%200==0:
            print('i_test',i)
       

    recall_1 = recall(predictions, references, 1)
    recall_5 = recall(predictions, references, 5)
    recall_10 = recall(predictions, references, 10)
    mr = mean_rank(predictions, references)
    mrr = mean_reciprocal_rank(predictions, references)
    print(recall_1, recall_5, recall_10, mr, mrr)

    print("testing time:", time.time() - start_time)

  

def test_diversity1(test_set, model):
    print("starting testing diversity...")
    model.eval()   
    print('len of test dataset:',len(test_set))
    comemnt_gen,ref={},{}
    N_num=np.arange(opt.z_class)#beam_size
    gen_self,ref_self={},{}
    for i in range(len(test_set)):
        X, Y, T, data,label = test_set.get_img_and_candidate(i)
        X = Variable(X, volatile=True).cuda()
        Y = Variable(Y, volatile=True).cuda()
        T = Variable(T, volatile=True).cuda()
        gen=''
        N_ref=random.choice(N_num)
        n_self=0
        sample=torch.randn(label_all_mean.squeeze(1).shape).cuda()
        z=label_all_mean.squeeze(1) + sample*c_sigma
    
        if X.shape[-1]!=opt.n_hidden :
            X=model.X_linear(X)
        T=T.unsqueeze(0)#[1,100]
        T1 = model.encode_text(T)#[1,100,512]
        X1=X.unsqueeze(0)#[1,5,512]
        #coa_atten
        z_out1,x_out1,y_out1=model.coa.for_test(X1,T1,T)

        out=model.generate(  z,label_all,x_out1, y_out1)#[3,20]
        #print('out',out.shape)
        for i_com in range(opt.z_class):
            com=DataSet.transform_to_words(out[i_com].cpu())
            gen=gen + com +' , '
            if i_com==N_ref:
                for i_v in range(opt.z_class-1):
                    num_gen_self=i*(opt.z_class-1)+i_v
                    gen_self.update({num_gen_self:[com]})
            else:
                num_ref_self=n_self+i*(opt.z_class-1)
                ref_self.update({num_ref_self:[com]})
                n_self+=1
        comemnt_gen.update({i:[gen]})
        ref.update({i:data['comment']}) 
            
        if i%200==0:
            print('i_test',i,'N_ref',N_ref)

       
        if i%10000==0:
            data_for_bleu={'gen_All':comemnt_gen,'ref_All':ref,'gen_self':gen_self,'ref_self':ref_self}
            f_n=opt.output_bleu+'.pkl'
            save_file = open(f_n,"wb")
            pickle.dump(data_for_bleu,save_file)
            save_file.close()
    data_for_bleu={'gen_All':comemnt_gen,'ref_All':ref,'gen_self':gen_self,'ref_self':ref_self}
    f_n=opt.output_bleu+'.pkl'
    save_file = open(f_n,"wb")
    pickle.dump(data_for_bleu,save_file)
    save_file.close()







def test_diversity1_part(test_set, model):
    print("starting testing diversity...")
    model.eval()
    print('len of test dataset:',len(test_set))
    comemnt_gen,ref={},{}
    N_num=np.arange(opt.z_class)#beam_size
    gen_self,ref_self={},{}
    N_num_10=np.arange(4)#beam_size
    for i in range(len(test_set)):
        N_ref_10=random.choice(N_num_10)
        if N_ref_10!=0:
            continue
        X, Y, T, data,label = test_set.get_img_and_candidate(i)
        #X:[5, 512]; Y:[100, 20]对应100条评论，每条len_com=20; T:[100]对应context,长度为n_com*len_com(5*20oi)  
        X = Variable(X, volatile=True).cuda()
        Y = Variable(Y, volatile=True).cuda()
        T = Variable(T, volatile=True).cuda()
        gen=''
        N_ref=random.choice(N_num)#随机挑选作为hypo,剩余两项作为ref
        n_self=0
        sample=torch.randn(label_all_mean.squeeze(1).shape).cuda()
        z=label_all_mean.squeeze(1) + sample*c_sigma
        #print('label_all',label_all.shape)
        #print('z',z.shape)
        #X与T预先处理：
        if X.shape[-1]!=opt.n_hidden :
            X=model.X_linear(X)
        T=T.unsqueeze(0)#[1,100]
        T1 = model.encode_text(T)#[1,100,512]
        X1=X.unsqueeze(0)#[1,5,512]
        #coa_atten
        z_out1,x_out1,y_out1=model.coa.for_test(X1,T1,T)

        out=model.generate(  z,label_all,x_out1, y_out1)#[3,20]
        #print('out',out.shape)
        for i_com in range(opt.z_class):
            com=DataSet.transform_to_words(out[i_com].cpu())
            gen=gen + com +' , '
            if i_com==N_ref:
                for i_v in range(opt.z_class-1):
                    num_gen_self=i*(opt.z_class-1)+i_v
                    gen_self.update({num_gen_self:[com]})
            else:
                num_ref_self=n_self+i*(opt.z_class-1)
                ref_self.update({num_ref_self:[com]})
                n_self+=1
        comemnt_gen.update({i:[gen]})
        ref.update({i:data['comment']})

        if i%200==0:
            print('i_test',i,'N_ref',N_ref)

      
    data_for_bleu={'gen_All':comemnt_gen,'ref_All':ref,'gen_self':gen_self,'ref_self':ref_self}
    f_n=opt.output_bleu+'.pkl'
    save_file = open(f_n,"wb")
    pickle.dump(data_for_bleu,save_file)
    save_file.close()                                                                                    
  



if __name__ == '__main__':
    ref_caps_train={}
    
    cider_train = Cider(ref_caps_train)

    if opt.mode == 'train':
        train()
    elif opt.mode == 'test_div':
        test_set = get_dataset(test_path, vocabs, is_train=False)
        model = Model(n_emb=opt.n_emb, n_hidden=opt.n_hidden, vocab_size=opt.vocab_size,
                  dropout=opt.dropout, d_ff=opt.d_ff, n_head=opt.n_head, n_block=opt.n_block)
        model_dict = torch.load(opt.restore)
        model.load_state_dict(model_dict)
        model.cuda()
        test_diversity1(test_set, model)      

    elif opt.mode == 'test_all':
        test_set = get_dataset(test_path, vocabs, is_train=False)
        model = Model(n_emb=opt.n_emb, n_hidden=opt.n_hidden, vocab_size=opt.vocab_size,
                  dropout=opt.dropout, d_ff=opt.d_ff, n_head=opt.n_head, n_block=opt.n_block)
        model_dict = torch.load(opt.restore)
        model.load_state_dict(model_dict)
        model.cuda()
        test(test_set, model,cider_train)  
        test_diversity1_part(test_set, model)    
          
    else:
        test_set = get_dataset(test_path, vocabs, is_train=False)
        model = Model(n_emb=opt.n_emb, n_hidden=opt.n_hidden, vocab_size=opt.vocab_size,
                  dropout=opt.dropout, d_ff=opt.d_ff, n_head=opt.n_head, n_block=opt.n_block)
        model_dict = torch.load(opt.restore)
        model.load_state_dict(model_dict)
        model.cuda()
        print('test_path',test_path)
        test(test_set, model,cider_train)
        print('test_path',test_path)
        test_path1='/mnt/data10t/bakuphome20210617/ffy2020/ffy/VC/VideoIC/task/comments_generation/processed_data/setup/test-all10_new.json'
        print('test_path1',test_path1)
        test_set = get_dataset(test_path1, vocabs, is_train=False)
        model = Model(n_emb=opt.n_emb, n_hidden=opt.n_hidden, vocab_size=opt.vocab_size,
                  dropout=opt.dropout, d_ff=opt.d_ff, n_head=opt.n_head, n_block=opt.n_block)
        model_dict = torch.load(opt.restore)
        model.load_state_dict(model_dict)
        model.cuda()
        test(test_set, model,cider_train)
        print('test_path1',test_path1)










