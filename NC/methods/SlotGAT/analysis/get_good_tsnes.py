
from matplotlib import pyplot as plt
import numpy as np
import os
import json
import sys
import argparse
import time
import math
import random
import torch
import torch.nn as nn
import matplotlib as mpl
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.labelsize']=0
mpl.rcParams['ytick.labelsize']=0
abcd="abcdefghijklmnopqrstuvwxyz"
class vis_data_collector():
    #all data must be simple python objects like int or 'str'
    def __init__(self):
        self.data_dict={}
        self.tensor_dict={}
        #formatting:
        #
        # {"meta":{****parameters and study name},"re-1":{"epoch-0":{"loss":0.1,"w1":1},"epoch-1":{"loss":0.2,"w1":2},...}}

    def save_meta(self,meta_data,meta_name):
        self.data_dict["meta"]={meta_name:meta_data}
        self.data_dict["meta"]["tensor_names"]=[]
        

    def collect_in_training(self,data,name,re,epoch,r=4):
        if f"re-{re}" not in self.data_dict.keys():
            self.data_dict[f"re-{re}" ]={}
        if f"epoch-{epoch}" not in self.data_dict[f"re-{re}" ].keys():
            self.data_dict[f"re-{re}" ][f"epoch-{epoch}"]={}
        if type(data)==float:
            data=round(data,r)
        self.data_dict[f"re-{re}" ][f"epoch-{epoch}"][name]=data

    def collect_in_run(self,data,name,re,r=4):
        if f"re-{re}" not in self.data_dict.keys():
            self.data_dict[f"re-{re}" ]={}
        if type(data)==float:
            data=round(data,r)
        self.data_dict[f"re-{re}" ][name]=data

    def collect_whole_process(self,data,name):
        self.data_dict[name]=data
    def collect_whole_process_tensor(self,data,name):
        self.tensor_dict[name]=data
        self.data_dict["meta"]["tensor_names"].append(name)


    def save(self,fn):
        
        f = open(fn+".json", 'w')
        json.dump(self.data_dict, f, indent=4)
        f.close()

        for k,v in self.tensor_dict.items():

            torch.save(v,fn+"_"+k+".pt")
    
    def load(self,fn):
        f = open(fn, 'r')
        self.data_dict= json.load(f)
        f.close()
        if "meta" in self.data_dict.keys():
            for name in self.data_dict["meta"]["tensor_names"]:
                self.tensor_dict[name]=torch.load(name+".pt")



    def trans_to_numpy(self,name,epoch_range=None):
        data=[]
        re=0
        while f"re-{re}" in self.data_dict.keys():
            data.append([])
            for i in range(epoch_range):
                data[re].append(self.data_dict[f"re-{re}"][f"epoch-{i}"][name])
            re+=1
        data=np.array(data)
        return np.mean(data,axis=0),np.std(data,axis=0)

    def visualize_tsne(self,dn,node_idx_by_ntype,dataset):
        from matplotlib.pyplot import figure

        figure(figsize=(16, 12), dpi=300)
        ncs=["r","b","y","g","chocolate","deeppink"]
        print(dn)
        layers=[]
        heads=[]
        ets=[]
        #"tsne_emb_layer_0_slot_0"
        for k,v in self.data_dict.items():
            if "tsne_emb_layer" in k:
                temp=k.split("_")
                if int(temp[3]) not in layers:
                    layers.append(int(temp[3]))
                if temp[4]=="slot":
                    if int(temp[5]) not in ets:
                        ets.append(int(temp[5]))
        layers,heads,ets=sorted(layers),sorted(heads),sorted(ets)
        print(layers,heads,ets)
        #heads plot
        for layer in layers:
            fig,ax=plt.subplots(2,int((len(node_idx_by_ntype)+1)/2))
            fig.set_size_inches(16,12)
            fig.set_dpi(300)
            nts=list(range(len(node_idx_by_ntype)))
            for nt in nts:
                subax=ax[int((nt)/(len(nts)/2))][int((nt)%(len(nts)/2))]
                subax.cla()
                datas=np.array(self.data_dict[f"tsne_emb_layer_{layer}_slot_{nt}"]).T
                print(datas.shape)
                x_lower,x_upper=0,0
                y_lower,y_upper=0,0
                for nt_j in nts:
                    x=datas[0][ node_idx_by_ntype[nt_j]]
                    y=datas[1][ node_idx_by_ntype[nt_j]]
                   # subax.scatter(x=x,y=y,c=ncs[nt_j],s=0.1,label=f"type {nt_j} with center ({np.mean(x):.2f},{np.mean(y):.2f}) and radius {(np.std(x)+np.std(y))/2:.2f}")
                    #if nt==1:
                    subax.scatter(x=x,y=y,c=ncs[nt_j],s=3,label=f"Type {nt_j}")
                    #else:
                    #    subax.scatter(x=x,y=y,c=ncs[nt_j],s=0.1)
                    #set 90% pencentile of lim
                    xs=sorted(x)
                    ys=sorted(y)
                    x_lower,x_upper=min(xs[int(len(xs)*0.05)],x_lower),max(xs[int(len(xs)*0.95)],x_upper)
                    y_lower,y_upper=min(ys[int(len(ys)*0.05)],y_lower),max(ys[int(len(ys)*0.95)],y_upper)
                #subax.set_xticks(fontsize=12)    
                #subax.set_yticks(fontsize=12)    
                subax.set_xlim(x_lower,x_upper)
                subax.set_ylim(y_lower,y_upper)
                subax.tick_params(direction='in', length=0, width=0)
                #subax.set_xlim(-200,200)
                #subax.set_ylim(-200,200)
                #subax.set_title(f" Layer {layer} Slot {nt}")
                subax.set_xlabel(f"({abcd[nt]}) Slot {nt}",fontsize=55)
                #plt.title(f"Layer {layer} Slot {nt}")
                #if  nt==1:
                    #lgnd=subax.legend(loc="lower right", bbox_to_anchor=(1.3, 0.6))
                    
                    #for lh in lgnd.legendHandles:
                        #lh._sizes=[16]
            #fig.legend()
            handles, labels = subax.get_legend_handles_labels()
            lgnd=fig.legend(handles, labels, loc='right',fontsize=50,bbox_to_anchor=(1.09, 1.04), ncol=4,handletextpad=0.01)
            #lgnd=fig.legend(bbox_to_anchor=(1.3, 0.6))
            #lgnd=fig.legend()
            for lh in lgnd.legendHandles:
                lh._sizes=[100]
            fig.tight_layout() 
            #fig.suptitle(f"Tsne Embedding of Layer {layer}")
            plt.savefig(os.path.join(dn,f"{dataset}_slot_embeddings_layer_{layer}.png"),bbox_inches="tight")




file_names=[] # to specify!!!

for fn,dataset_info_fn in file_names:
    vis=vis_data_collector()
    vis_dataset_info=vis_data_collector()
    vis_dataset_info.load(dataset_info_fn)
    node_idx_by_ntype=vis_dataset_info.data_dict["node_idx_by_ntype"]
    vis.load(fn)
    vis.visualize_tsne(os.path.dirname(fn),node_idx_by_ntype,dataset=dataset_info_fn.split("_")[2])
                