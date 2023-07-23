import torch
import torch as th
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv
import math
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from conv import myGATConv,slotGATConv

from dgl._ffi.base import DGLError
from torch.profiler import profile, record_function, ProfilerActivity
"""
class DistMult(nn.Module):
    def __init__(self, num_rel, dim):
        super(DistMult, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(size=(num_rel, dim, dim)))
        nn.init.xavier_normal_(self.W, gain=1.414)

    def forward(self, left_emb, right_emb, r_id):
        thW = self.W[r_id]
        left_emb = torch.unsqueeze(left_emb, 1)
        right_emb = torch.unsqueeze(right_emb, 2)
        return torch.bmm(torch.bmm(left_emb, thW), right_emb).squeeze()"""


class DistMult(nn.Module):
    def __init__(self, num_rel, dim):
        super(DistMult, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(size=(num_rel, dim, dim)))
        nn.init.xavier_normal_(self.W, gain=1.414)
    def forward(self, left_emb, right_emb, r_id,slot_num=None,prod_aggr=None,sigmoid="after"):
        if not prod_aggr:
            thW = self.W[r_id]
            left_emb = torch.unsqueeze(left_emb, 1)
            right_emb = torch.unsqueeze(right_emb, 2)
            #return torch.bmm(torch.bmm(left_emb, thW), right_emb).squeeze()
            scores=torch.zeros(right_emb.shape[0]).to(right_emb.device)
            for i in range(int(max(r_id))+1):
                scores[r_id==i]=torch.bmm(torch.matmul(left_emb[r_id==i], self.W[i]), right_emb[r_id==i]).squeeze()
            return scores
        else:
            raise Exception


class Dot(nn.Module):
    def __init__(self):
        super(Dot, self).__init__()
    def forward(self, left_emb, right_emb, r_id,slot_num=None,prod_aggr=None,sigmoid="after"):
        if not prod_aggr:
            left_emb = torch.unsqueeze(left_emb, 1)
            right_emb = torch.unsqueeze(right_emb, 2)
            return torch.bmm(left_emb, right_emb).squeeze()
        else:
            left_emb = left_emb.view(-1,slot_num,int(left_emb.shape[1]/slot_num))
            right_emb = right_emb.view(-1,int(right_emb.shape[1]/slot_num),slot_num)
            x=torch.bmm(left_emb, right_emb)# num_sampled_edges* num_slot*num_slot
            if prod_aggr=="all":
                x=x.flatten(1)
                x=x.sum(1)
                return x
            x=torch.diagonal(x,0,1,2) # num_sampled_edges* num_slot
            if sigmoid=="before":
                x=F.sigmoid(x)
            
            if prod_aggr=="mean":
                x=x.mean(1)
                
            elif prod_aggr=="max":
                x=x.max(1)[0]
            elif prod_aggr=="sum":
                x=x.sum(1)
            else:
                raise Exception()
            return x
























class myGAT(nn.Module):
    def __init__(self,
                 g,
                 edge_dim,
                 num_etypes,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha,
                 decode='distmult',inProcessEmb="True",l2use="True",dataRecorder=None,get_out=None):
        super(myGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.inProcessEmb=inProcessEmb
        self.l2use=l2use
        self.dataRecorder=dataRecorder
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(myGATConv(edge_dim, num_etypes,
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha))
        # output projection
        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        if decode == 'distmult':
            self.decoder = DistMult(num_etypes, num_classes*(num_layers+2))
        elif decode == 'dot':
            self.decoder = Dot()
        self.get_out=get_out

    def l2_norm(self, x):
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        return x / (torch.max(torch.norm(x, dim=1, keepdim=True), self.epsilon))

    def forward(self, features_list, e_feat, left, right, mid):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        emb = [self.l2_norm(h)]
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, e_feat, res_attn=res_attn)
            emb.append(self.l2_norm(h.mean(1)))
            h = h.flatten(1)
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, e_feat, res_attn=res_attn)#None)
        logits = logits.mean(1)
        logits = self.l2_norm(logits)
        #emb.append(logits)
        if self.inProcessEmb=="True":
            emb.append(logits)
        else:
            emb=[logits]
        logits = torch.cat(emb, 1)
        left_emb = logits[left]
        right_emb = logits[right]
        return F.sigmoid(self.decoder(left_emb, right_emb, mid))


       
class slotGAT(nn.Module):
    def __init__(self,
                 g,
                 edge_dim,
                 num_etypes,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha,
                 num_ntype,
                 eindexer,aggregator="average",predicted_by_slot="None", get_out=[""],
                 decode='distmult',inProcessEmb="True",l2BySlot="False",prod_aggr=None,sigmoid="after",l2use="True",SAattDim=128,dataRecorder=None):
        super(slotGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.heads=heads
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        self.num_ntype=num_ntype
        self.num_classes=num_classes
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.predicted_by_slot=predicted_by_slot  
        self.inProcessEmb=inProcessEmb 
        self.l2BySlot=l2BySlot
        self.prod_aggr=prod_aggr
        self.sigmoid=sigmoid
        self.l2use=l2use
        self.SAattDim=SAattDim
        self.dataRecorder=dataRecorder
        
        self.get_out=get_out 
        self.num_etypes=num_etypes
        self.num_hidden=num_hidden
        self.last_fc = nn.Parameter(th.FloatTensor(size=(num_classes*self.num_ntype, num_classes))) ;nn.init.xavier_normal_(self.last_fc, gain=1.414)
        
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
            
        # input projection (no residual)
        self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha,num_ntype=num_ntype,eindexer=eindexer,inputhead=True))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
                num_hidden* heads[l-1] , num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha,num_ntype=num_ntype,eindexer=eindexer))
        # output projection
        self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
            num_hidden* heads[-2] , num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha,num_ntype=num_ntype,eindexer=eindexer))
        self.aggregator=aggregator
        if aggregator=="SA":
            if self.inProcessEmb=="True":
                last_dim=num_hidden*(2+num_layers)
            else:
                last_dim=num_hidden
                
            
            self.macroLinear=nn.Linear(last_dim, self.SAattDim, bias=True);nn.init.xavier_normal_(self.macroLinear.weight, gain=1.414);nn.init.normal_(self.macroLinear.bias, std=1.414*math.sqrt(1/(self.macroLinear.bias.flatten().shape[0])))
            self.macroSemanticVec=nn.Parameter(torch.FloatTensor(self.SAattDim,1));nn.init.normal_(self.macroSemanticVec,std=1)
        

        self.by_slot=[f"by_slot_{nt}" for nt in range(num_ntype)]
        assert aggregator in (["average","last_fc","max","None","SA"]+self.by_slot)
        #self.get_out=get_out
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        if decode == 'distmult':
            if self.aggregator=="None":
                num_classes=num_classes*num_ntype
            self.decoder = DistMult(num_etypes, num_classes*(num_layers+2))
        elif decode == 'dot':
            self.decoder = Dot()


    def forward(self, features_list,e_feat, left, right, mid, get_out="False"): 
        encoded_embeddings=None
        
        h = []
        for nt_id,(fc, feature) in enumerate(zip(self.fc_list, features_list)):
            nt_ft=fc(feature)
            emsen_ft=torch.zeros([nt_ft.shape[0],nt_ft.shape[1]*self.num_ntype]).to(feature.device)
            emsen_ft[:,nt_ft.shape[1]*nt_id:nt_ft.shape[1]*(nt_id+1)]=nt_ft
            h.append(emsen_ft)   # the id is decided by the node types
        h = torch.cat(h, 0)        #  num_nodes*(num_type*hidden_dim)
        
        emb = [self.aggr_func(self.l2_norm(h,l2BySlot=self.l2BySlot))]
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, e_feat,get_out=get_out, res_attn=res_attn)   #num_nodes*num_heads*(num_ntype*hidden_dim)
            emb.append(self.aggr_func(self.l2_norm(h.mean(1),l2BySlot=self.l2BySlot)))
            h = h.flatten(1)#num_nodes*(num_heads*num_ntype*hidden_dim)
            
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, e_feat,get_out=get_out, res_attn=res_attn)#None)   #num_nodes*num_heads*num_ntype*hidden_dim
        
        
        logits = logits.mean(1)
        if self.predicted_by_slot!="None" and self.training==False:
            logits=logits.view(-1,1,self.num_ntype,self.num_classes)
            
            if self.predicted_by_slot=="max":
                if "getMaxSlot" in  get_out:
                    maxSlotIndexesWithLabels=logits.max(2)[1].squeeze(1)
                    logits_indexer=logits.max(2)[0].max(2)[1]
                    self.maxSlotIndexes=torch.gather(maxSlotIndexesWithLabels,1,logits_indexer)
                logits=logits.max(2)[0]
            elif self.predicted_by_slot=="all":
                if "getSlots" in get_out:
                    self.logits=logits.detach()
                logits=logits.view(-1,1,self.num_ntype,self.num_classes).mean(2)

            else:
                target_slot=int(self.predicted_by_slot)
                logits=logits[:,:,target_slot,:].squeeze(2)
        else:
            logits=self.aggr_func(self.l2_norm(logits,l2BySlot=self.l2BySlot))
            
        
        if self.inProcessEmb=="True":
            emb.append(logits)
        else:
            emb=[logits]
        if self.aggregator=="None" and self.inProcessEmb=="True":
            emb=[ x.view(-1, self.num_ntype,int(x.shape[1]/self.num_ntype))   for x in emb]
            o = torch.cat(emb, 2).flatten(1)
        else:
            o = torch.cat(emb, 1)
        if self.aggregator=="SA" :
            o=o.view(-1, self.num_ntype,int(o.shape[1]/self.num_ntype))
            
            slot_scores=(F.tanh( self.macroLinear(o))  @  self.macroSemanticVec).mean(0,keepdim=True)  #num_slots
            self.slot_scores=F.softmax(slot_scores,dim=1)
            o=(o*self.slot_scores).sum(1)  

        left_emb = o[left]
        right_emb = o[right]
        if self.sigmoid=="after":
            logits=self.decoder(left_emb, right_emb, mid,slot_num=self.num_ntype,prod_aggr=self.prod_aggr)
            logits=F.sigmoid(logits)
        elif self.sigmoid=="before":
            
            logits=self.decoder(left_emb, right_emb, mid,slot_num=self.num_ntype,prod_aggr=self.prod_aggr,sigmoid=self.sigmoid)
        elif self.sigmoid=="None":
            left_emb=self.l2_norm(left_emb,l2BySlot=self.l2BySlot)
            right_emb=self.l2_norm(right_emb,l2BySlot=self.l2BySlot)
            logits=self.decoder(left_emb, right_emb, mid,slot_num=self.num_ntype,prod_aggr=self.prod_aggr)
        else:
            raise Exception()
        return logits


    def l2_norm(self, x,l2BySlot="False"):
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        if self.l2use=="True":
            if l2BySlot=="False":
                return x / (torch.max(torch.norm(x, dim=1, keepdim=True), self.epsilon))
            elif l2BySlot=="True":
                x=x.view(-1, self.num_ntype,int(x.shape[1]/self.num_ntype))
                x=x / (torch.max(torch.norm(x, dim=2, keepdim=True), self.epsilon))
                x=x.flatten(1)
                return x
        elif self.l2use=="False":
            return x
        else:
            raise Exception()


    def aggr_func(self,logits):
        if self.aggregator=="average":
            logits=logits.view(-1, self.num_ntype,self.num_classes).mean(1)
        elif self.aggregator=="last_fc":
            logits=logits.view(-1,self.num_ntype,self.num_classes)
            logits=logits.flatten(1)
            logits=logits.matmul(self.last_fc).unsqueeze(1)
        elif self.aggregator=="max":
            logits=logits.view(-1,self.num_ntype,self.num_classes).max(1)[0]
        
        elif self.aggregator=="None" or "SA":
            logits=logits.view(-1, self.num_ntype,self.num_classes).flatten(1)



        else:
            raise NotImplementedError()
        
        return logits

