import math
from wsgiref.simple_server import WSGIRequestHandler
import torch.nn as nn
import torch
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

#####SA、GA、SGA module

class MHAtt(nn.Module):
    def __init__(self, HIDDEN_SIZE=512,DROPOUT_R=0.1,MULTI_HEAD=8):
        super(MHAtt, self).__init__()
        self.HIDDEN_SIZE_HEAD=int(HIDDEN_SIZE / MULTI_HEAD)
        self.MULTI_HEAD=MULTI_HEAD
        self.HIDDEN_SIZE=HIDDEN_SIZE

        self.linear_v = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_k = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_q = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_merge = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

        self.dropout = nn.Dropout(DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, HIDDEN_SIZE=512,DROPOUT_R=0.1):
        super(FFN, self).__init__()
        self.FF_SIZE = int(HIDDEN_SIZE * 4)

        self.mlp = MLP(
            in_size=HIDDEN_SIZE,
            mid_size=self.FF_SIZE,
            out_size=HIDDEN_SIZE,
            dropout_r=DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, HIDDEN_SIZE=512,DROPOUT_R=0.1):
        super(SA, self).__init__()

        self.mhatt = MHAtt(HIDDEN_SIZE=512,DROPOUT_R=0.1,MULTI_HEAD=8)
        self.ffn = FFN(HIDDEN_SIZE=512,DROPOUT_R=0.1)

        self.dropout1 = nn.Dropout(DROPOUT_R)
        self.norm1 = LayerNorm(HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(DROPOUT_R)
        self.norm2 = LayerNorm(HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, HIDDEN_SIZE=512,DROPOUT_R=0.1):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(HIDDEN_SIZE=512,DROPOUT_R=0.1,MULTI_HEAD=8)
        self.mhatt2 = MHAtt(HIDDEN_SIZE=512,DROPOUT_R=0.1,MULTI_HEAD=8)
        self.ffn = FFN(HIDDEN_SIZE=512,DROPOUT_R=0.1)

        self.dropout1 = nn.Dropout(DROPOUT_R)
        self.norm1 = LayerNorm(HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(DROPOUT_R)
        self.norm2 = LayerNorm(HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(DROPOUT_R)
        self.norm3 = LayerNorm(HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, LAYER=6):#LAYER为所用层数
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA() for _ in range(LAYER)])
        self.dec_list = nn.ModuleList([SGA() for _ in range(LAYER)])

    def forward(self, x, y, x_mask, y_mask):#x对应text,y对应img
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y

class MCA_ST(nn.Module):#stack model
    def __init__(self, LAYER=6):#LAYER为所用层数
        super(MCA_ST, self).__init__()
        self.L=LAYER
        self.enc_list = nn.ModuleList([SA() for _ in range(LAYER)])
        self.dec_list = nn.ModuleList([SGA() for _ in range(LAYER)])

    def forward(self, x, y, x_mask, y_mask):#x对应text,y对应img
        # Get hidden vector
        for iL in range(self.L):
            x = self.enc_list[iL](x, x_mask)
            y = self.dec_list[iL](y, x, y_mask, x_mask)

        return x, y        




class AttFlat(nn.Module):
    def __init__(self, HIDDEN_SIZE=512,FLAT_MLP_SIZE=512,FLAT_GLIMPSES=1,DROPOUT_R=0.1,FLAT_OUT_SIZE=1024):
        super(AttFlat, self).__init__()

        self.mlp = MLP(
            in_size=HIDDEN_SIZE,
            mid_size=FLAT_MLP_SIZE,
            out_size=FLAT_GLIMPSES,
            dropout_r=DROPOUT_R,
            use_relu=True
        )
        self.FLAT_OUT_SIZE=FLAT_OUT_SIZE

        #self.linear_merge = nn.Linear(HIDDEN_SIZE * FLAT_GLIMPSES,FLAT_OUT_SIZE)#1024)

    def forward(self, x, x_mask):#x:[80, m, 512]
        att = self.mlp(x)#[80, m, 1]
        if x_mask is not None:
            att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9)#[80, m, 1] 
        att = F.softmax(att, dim=1)#
        
        x_atted = torch.sum(torch.mul(att,x), 1) #[bs,m,1]*[bs,m,512]=>[bs,m,512] => [bs,512]
        return x_atted.unsqueeze(1)

class coa_MAC(nn.Module):
    def __init__(self, out_size,LAYER,HIDDEN_SIZE=512,FLAT_MLP_SIZE=512,FLAT_GLIMPSES=1,DROPOUT_R=0.1,FLAT_OUT_SIZE=1024):
        super(coa_MAC, self).__init__()

        self.linear_merge_v = nn.Linear(HIDDEN_SIZE * FLAT_GLIMPSES, FLAT_OUT_SIZE )
        self.linear_merge_t = nn.Linear(HIDDEN_SIZE * FLAT_GLIMPSES, FLAT_OUT_SIZE )

        self.backbone = MCA_ED(LAYER)
        self.backbone1 = MCA_ED(LAYER)


        self.attflat_img = AttFlat(HIDDEN_SIZE,FLAT_MLP_SIZE,FLAT_GLIMPSES,DROPOUT_R,FLAT_OUT_SIZE)
        self.attflat_lang = AttFlat(HIDDEN_SIZE,FLAT_MLP_SIZE,FLAT_GLIMPSES,DROPOUT_R,FLAT_OUT_SIZE)
        
        self.proj_norm = LayerNorm(FLAT_OUT_SIZE)
        self.proj = nn.Linear(FLAT_OUT_SIZE, out_size)

        
        self.attflat_img1 = AttFlat(HIDDEN_SIZE,FLAT_MLP_SIZE,FLAT_GLIMPSES,DROPOUT_R,FLAT_OUT_SIZE)
        self.attflat_lang1 = AttFlat(HIDDEN_SIZE,FLAT_MLP_SIZE,FLAT_GLIMPSES,DROPOUT_R,FLAT_OUT_SIZE)
        


    # Masking
    def make_mask(self, feature):#[bs,m,512]=>[bs,m]=>[bs,1,1,m]
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

    def forward(self, img,text,T):#T:[bs,100]
        # Make mask
        lang_feat_mask = self.make_mask(T.unsqueeze(2))
        img_feat_mask = self.make_mask(img)#[B, 1, 1, 5]
       

        lang_feat, img_feat = self.backbone(
            text,
            img,
            lang_feat_mask,
            img_feat_mask
        )
        #lang_feat:[90, 100, 512],img_feat:[90, 5, 512])
        
       
        text_bt,img_bt=text.transpose(1,0),img.transpose(1,0)
        lang_feat_bt, img_feat_bt = self.backbone1(
            text_bt,
            img_bt,
            None,
            None
        )
        #lang_feat_bt:[100, 90, 512]) img_feat_bt:([5, 90, 512]
        lang_feat_bt,img_feat_bt=lang_feat_bt.transpose(1,0),img_feat_bt.transpose(1,0)

        
        lang_feat_bt = self.attflat_lang1(lang_feat_bt, lang_feat_mask)#attend～y，#[B,1,512]
        img_feat_bt = self.attflat_img1(img_feat_bt, img_feat_mask)#attend～x，#[B,1,512]



        lang_feat = self.attflat_lang(lang_feat, lang_feat_mask)#attend～y，#[B,1,512]
        img_feat = self.attflat_img(img_feat, img_feat_mask)#attend～x，#[B,1,512]
        
    
        img_feat1 = self.linear_merge_v(img_feat)#[B,1,1024]
        lang_feat1 = self.linear_merge_t(lang_feat)##[B,1,1024]
        proj_feat = lang_feat1 + img_feat1#[B,1,1024]
        proj_feat = self.proj_norm(proj_feat)#[B,1,1024]
        proj_feat = torch.sigmoid(self.proj(proj_feat)) #[B,1,z_class]
       
        
        return proj_feat.squeeze(1),img_feat,lang_feat,img_feat_bt,lang_feat_bt

    def for_test(self, img,text,T):#T:[bs,100]
        # Make mask
        lang_feat_mask = self.make_mask(T.unsqueeze(2))
        img_feat_mask = self.make_mask(img)#[B, 1, 1, 5]
       

        lang_feat, img_feat = self.backbone(
            text,
            img,
            lang_feat_mask,
            img_feat_mask
        )
        #lang_feat:[90, 100, 512],img_feat:[90, 5, 512])
        
        
        lang_feat = self.attflat_lang(lang_feat, lang_feat_mask)#attend以后的～y，#[B,1,512]
        img_feat = self.attflat_img(img_feat, img_feat_mask)#attend以后的～x，#[B,1,512]
  
        img_feat1 = self.linear_merge_v(img_feat)#[B,1,1024]
        lang_feat1 = self.linear_merge_t(lang_feat)##[B,1,1024]
        proj_feat = lang_feat1 + img_feat1#[B,1,1024]
        proj_feat = self.proj_norm(proj_feat)#[B,1,1024]
        proj_feat = torch.sigmoid(self.proj(proj_feat)) #[B,1,z_class]
       
        
        return proj_feat.squeeze(1),img_feat,lang_feat#,img_feat_bt,lang_feat_bt
        

  

