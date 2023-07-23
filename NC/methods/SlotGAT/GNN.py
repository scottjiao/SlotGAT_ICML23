import torch
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv
import math
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from conv import myGATConv,changedGATConv,slotGATConv
from torch.profiler import profile, record_function, ProfilerActivity
from torch_geometric.typing import OptPairTensor, OptTensor, Size, Tensor
from typing import Callable, Tuple, Union
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from dgl._ffi.base import DGLError
from typing import List, NamedTuple, Optional, Tuple, Union



from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing, GCNConv

class Adj(NamedTuple):
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    size: Tuple[int, int]
    target_size: int

    def to(self, *args, **kwargs):
        return Adj(
            self.edge_index.to(*args, **kwargs),
            self.edge_features.to(*args, **kwargs),
            self.size,
            self.target_size
        )
    




class MLP(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout):
        super(MLP, self).__init__()
        self.num_classes=num_classes
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(nn.Linear(num_hidden, num_hidden, bias=True))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(num_hidden, num_hidden))
        # output layer
        self.layers.append(nn.Linear(num_hidden, num_classes))
        for ly in self.layers:
            nn.init.xavier_normal_(ly.weight, gain=1.414)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features_list, e_feat):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)

        for i, layer in enumerate(self.layers):
            encoded_embeddings=h
            h = self.dropout(h)
            h = layer(h)
            h=F.relu(h) if i<len(self.layers) else h

        return h,encoded_embeddings



class LabelPropagation(nn.Module):
    r"""
    Description
    -----------
    Introduced in `Learning from Labeled and Unlabeled Data with Label Propagation <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.3864&rep=rep1&type=pdf>`_
    .. math::
        \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y},
    where unlabeled data is inferred by labeled data via propagation.
    Parameters
    ----------
        num_layers: int
            The number of propagations.
        alpha: float
            The :math:`\alpha` coefficient.
    """
    def __init__(self, num_layers, alpha):
        super(LabelPropagation, self).__init__()

        self.num_layers = num_layers
        self.alpha = alpha
    
    @torch.no_grad()
    def forward(self, g, labels, mask,get_out="False"):    # labels.shape=(number of nodes of type 0)  may contain false labels, therefore here the mask argument which provides the training nodes' idx is important
        with g.local_scope():
            if labels.dtype == torch.long:
                labels = F.one_hot(labels.view(-1)).to(torch.float32)
            y=torch.zeros((g.num_nodes(),labels.shape[1])).to(labels.device)
            y[mask] = labels[mask]
            
            last = (1 - self.alpha) * y
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(labels.device).unsqueeze(1)

            for _ in range(self.num_layers):
                # Assume the graphs to be undirected
                g.ndata['h'] = y * norm
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                y = last + self.alpha * g.ndata.pop('h') * norm
                y=F.normalize(y,p=1,dim=1)   #normalize y by row with p-1-norm
                y[mask] = labels[mask]
                last = (1 - self.alpha) * y
            
            return y,None





       
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
                 eindexer, aggregator="SA",predicted_by_slot="None", addLogitsTrain="None",  SAattDim=32,dataRecorder=None,targetTypeAttention="False",vis_data_saver=None):
        super(slotGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims]) 
        self.num_ntype=num_ntype
        self.num_classes=num_classes
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.predicted_by_slot=predicted_by_slot 
        self.addLogitsTrain=addLogitsTrain 
        self.SAattDim=SAattDim 
        self.vis_data_saver=vis_data_saver
        self.dataRecorder=dataRecorder
        
        if aggregator=="SA":
            last_dim=num_classes
                
            self.macroLinear=nn.Linear(last_dim, self.SAattDim, bias=True);nn.init.xavier_normal_(self.macroLinear.weight, gain=1.414);nn.init.normal_(self.macroLinear.bias, std=1.414*math.sqrt(1/(self.macroLinear.bias.flatten().shape[0])))
            self.macroSemanticVec=nn.Parameter(torch.FloatTensor(self.SAattDim,1));nn.init.normal_(self.macroSemanticVec,std=1)
            
 
 
        self.last_fc = nn.Parameter(th.FloatTensor(size=(num_classes*self.num_ntype, num_classes))) ;nn.init.xavier_normal_(self.last_fc, gain=1.414)
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha,num_ntype=num_ntype, eindexer=eindexer,inputhead=True, dataRecorder=dataRecorder))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
                num_hidden* heads[l-1] , num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha,num_ntype=num_ntype, eindexer=eindexer, dataRecorder=dataRecorder))
        # output projection
        self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
            num_hidden* heads[-2] , num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha,num_ntype=num_ntype, eindexer=eindexer, dataRecorder=dataRecorder))
        self.aggregator=aggregator
        self.by_slot=[f"by_slot_{nt}" for nt in range(g.num_ntypes)]
        assert aggregator in (["onedimconv","average","last_fc","max","SA"]+self.by_slot)
        if self.aggregator=="onedimconv":
            self.nt_aggr=nn.Parameter(torch.FloatTensor(1,1,self.num_ntype,1));nn.init.normal_(self.nt_aggr,std=1) 
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
    def l2byslot(self,x):
        
        x=x.view(-1, self.num_ntype,int(x.shape[1]/self.num_ntype))
        x=x / (torch.max(torch.norm(x, dim=2, keepdim=True), self.epsilon))
        x=x.flatten(1)
        return x

    def forward(self, features_list,e_feat, get_out="False"):
        with record_function("model_forward"):
            encoded_embeddings=None
            h = []
            for nt_id,(fc, feature) in enumerate(zip(self.fc_list, features_list)):
                nt_ft=fc(feature)
                emsen_ft=torch.zeros([nt_ft.shape[0],nt_ft.shape[1]*self.num_ntype]).to(feature.device)
                emsen_ft[:,nt_ft.shape[1]*nt_id:nt_ft.shape[1]*(nt_id+1)]=nt_ft
                h.append(emsen_ft)   # the id is decided by the node types
            h = torch.cat(h, 0)        #  num_nodes*(num_type*hidden_dim)
            res_attn = None
            for l in range(self.num_layers):
                h, res_attn = self.gat_layers[l](self.g, h, e_feat,get_out=get_out, res_attn=res_attn)   #num_nodes*num_heads*(num_ntype*hidden_dim)
                h = h.flatten(1)    #num_nodes*(num_heads*num_ntype*hidden_dim)
                encoded_embeddings=h
            # output projection
            logits, _ = self.gat_layers[-1](self.g, h, e_feat,get_out=get_out, res_attn=None)   #num_nodes*num_heads*num_ntype*hidden_dim
        
        if self.aggregator=="SA" :
            logits=logits.squeeze(1)
            logits=self.l2byslot(logits)
            logits=logits.view(-1, self.num_ntype,int(logits.shape[1]/self.num_ntype))
            
            if "getSlots" in get_out:
                self.logits=logits.detach()

             
            
            slot_scores=(F.tanh(self.macroLinear(logits))@self.macroSemanticVec).mean(0,keepdim=True)  #num_slots
            self.slot_scores=F.softmax(slot_scores,dim=1)
            logits=(logits*self.slot_scores).sum(1)
            if  self.dataRecorder["meta"]["getSAAttentionScore"]=="True":
                self.dataRecorder["data"][f"{self.dataRecorder['status']}_SAAttentionScore"]=self.slot_scores.flatten().tolist() #count dist


        #average across the ntype info
        if self.predicted_by_slot!="None" and self.training==False:
            with record_function("predict_by_slot"):
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
                    logits=logits.view(-1,1,self.num_ntype,self.num_classes).mean(2)  #average??

                else:
                    target_slot=int(self.predicted_by_slot)
                    logits=logits[:,:,target_slot,:].squeeze(2)
        else:
            #with record_function("slot_aggregation"):
            if self.aggregator=="average":
                logits=logits.view(-1,1,self.num_ntype,self.num_classes).mean(2)
            elif self.aggregator=="onedimconv":
                logits=(logits.view(-1,1,self.num_ntype,self.num_classes)*F.softmax(self.leaky_relu(self.nt_aggr),dim=2)).sum(2)
            elif self.aggregator=="last_fc":
                logits=logits.view(-1,1,self.num_ntype,self.num_classes)
                logits=logits.flatten(1)
                logits=logits.matmul(self.last_fc).unsqueeze(1)
            elif self.aggregator=="max":
                logits=logits.view(-1,1,self.num_ntype,self.num_classes).max(2)[0]
        
            elif self.aggregator=="None":
            
                logits=logits.view(-1,1, self.num_ntype,self.num_classes).flatten(2)
            elif  self.aggregator== "SA":
                logits=logits.view(-1,1, 1,self.num_classes).flatten(2)



            else:
                raise NotImplementedError()
        #average across the heads
        ### logits = [num_nodes *  num_of_heads *num_classes]
        self.logits_mean=logits.flatten().mean()
        logits = logits.mean(1)
        
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits, encoded_embeddings    #hidden_logits






  
class changedGAT(nn.Module):
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
                 eindexer, ):
        super(changedGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims]) 
        #self.ae_drop=nn.Dropout(feat_drop)
        #if ae_layer=="last_hidden":
            #self.lc_ae=nn.ModuleList([nn.Linear(num_hidden * heads[-2],num_hidden, bias=True),nn.Linear(num_hidden,num_ntype, bias=True)])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(changedGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha,num_ntype=num_ntype, eindexer=eindexer))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(changedGATConv(edge_dim, num_etypes,
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha,num_ntype=num_ntype, eindexer=eindexer))
        # output projection
        self.gat_layers.append(changedGATConv(edge_dim, num_etypes,
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha,num_ntype=num_ntype,  eindexer=eindexer))
        self.epsilon = torch.FloatTensor([1e-12]).cuda()

    def forward(self, features_list, e_feat,get_out="False"):

        hidden_logits=None
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))   # the id is decided by the node types
        h = torch.cat(h, 0)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, e_feat, res_attn=res_attn)
            h = h.flatten(1)
            #if self.ae_layer=="last_hidden":
            encoded_embeddings=h
            """for i in range(len(self.lc_ae)):
                _h=self.lc_ae[i](_h)
                if i==0:
                    _h=self.ae_drop(_h)
                    _h=F.relu(_h)
            hidden_logits=_h"""
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, e_feat, res_attn=None)
        logits = logits.mean(1)
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits, encoded_embeddings    #hidden_logits






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
                 alpha, dataRecorder=None):
        super(myGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
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
        self.dataRecorder=dataRecorder

    def forward(self, features_list, e_feat, get_out="False"):


        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))   # the id is decided by the node types
        h = torch.cat(h, 0)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, e_feat, res_attn=res_attn)
            h = h.flatten(1)
            encoded_embeddings=h
        # output projection
        logits, _ = self.gat_layers[-1](self.g, h, e_feat, res_attn=None)
        logits = logits.mean(1)
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits,encoded_embeddings

class RGAT(nn.Module):
    def __init__(self,
                 gs,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual ):
        super(GAT, self).__init__()
        self.gs = gs
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList([nn.ModuleList() for i in range(len(gs))])
        self.activation = activation
        self.weights = nn.Parameter(torch.zeros((len(in_dims), num_layers+1, len(gs))))
        self.sm = nn.Softmax(2)
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        for i in range(len(gs)):
            # input projection (no residual)
            self.gat_layers[i].append(GATConv(
                num_hidden, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation))
            # hidden layers
            for l in range(1, num_layers):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers[i].append(GATConv(
                    num_hidden * heads[l-1], num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
            # output projection
            self.gat_layers[i].append(GATConv(
                num_hidden * heads[-2], num_classes, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, features_list):
        nums = [feat.size(0) for feat in features_list]
        weights = self.sm(self.weights)
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for l in range(self.num_layers):
            out = []
            for i in range(len(self.gs)):
                out.append(torch.split(self.gat_layers[i][l](self.gs[i], h).flatten(1), nums))
            h = []
            for k in range(len(nums)):
                tmp = []
                for i in range(len(self.gs)):
                    tmp.append(out[i][k]*weights[k,l,i])
                h.append(sum(tmp))
            h = torch.cat(h, 0)
        out = []
        for i in range(len(self.gs)):
            out.append(torch.split(self.gat_layers[i][-1](self.gs[i], h).mean(1), nums))
        logits = []
        for k in range(len(nums)):
            tmp = []
            for i in range(len(self.gs)):
                tmp.append(out[i][k]*weights[k,-1,i])
            logits.append(sum(tmp))
        logits = torch.cat(logits, 0)
        return logits

class GAT(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,dataRecorder=None ):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        self.dataRecorder=dataRecorder
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, features_list, e_feat,get_out="False"):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
            encoded_embeddings=h
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits,encoded_embeddings


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout, dataRecorder=None):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        self.dataRecorder=dataRecorder
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input layer
        self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(num_hidden, num_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features_list, e_feat,get_out="False"):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for i, layer in enumerate(self.layers):
            encoded_embeddings=h
            h = self.dropout(h)
            h = layer(self.g, h)
        return h,encoded_embeddings




def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None,norm="D^{-1/2}(A+I)D^{-1/2}",attn_drop=None):

    fill_value = 2. if improved else 1.
    num_nodes = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight
        
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    if norm=="D^{-1/2}(A+I)D^{-1/2}":
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, attn_drop(deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col])
    elif norm=="D^{-1}(A+I)":
        deg_inv_sqrt = deg.pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, attn_drop(deg_inv_sqrt[row] * edge_weight )
    elif norm=="(A+I)D^{-1}":
        deg_inv_sqrt = deg.pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, attn_drop(deg_inv_sqrt[col] * edge_weight )
    elif norm=="(A+I)":
        return edge_index, attn_drop(edge_weight )
    else:
        raise Exception(f"No specified norm: {norm}")




