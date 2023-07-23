import torch
import dgl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
import os
import torch.nn.functional as F
import torch.nn as nn
import copy
import json
import pickle
from matplotlib import pyplot as plt




def count_torch_tensor(t):
    t=t.flatten(0).cpu()
    c={}
    for n in t:
        n=n.item()
        if n not in c:
            c[n]=0
        c[n]+=1
    c=sorted(list(c.items()),key=lambda x:x[0])
    return c




def strList(l):
    return [str(x) for x in l]

def writeIntoCsvLogger(dictToWrite,file_name):
    #read file
    to_write_names=sorted(list(dictToWrite.keys()))
    if not os.path.exists(file_name):
        to_write_line=[]
        for n in to_write_names:
            to_write_line.append(dictToWrite[n])
        with open(file_name,"w") as f:
            f.write(  ",".join(strList(to_write_names)) +"\n")
            f.write(  ",".join(strList(to_write_line)) +"\n")
    else:
        with open(file_name,"r") as f:
            rec=[]
            for line in f:
                line=line.strip("\n").split(",")
                rec.append(line)
        #ensure they have same names
        row_names=sorted(rec[0])
        if to_write_names!=row_names:
            collected_names_not_in=[]
            for n in to_write_names:
                if n not in row_names:
                    for i,n_r in enumerate(rec):
                        if i==0:
                            rec[0].append(n)
                        else:
                            rec[i].append("")
                row_names.append(n)
            for n_r in row_names:
                if n_r not in to_write_names:
                    dictToWrite[n_r]=""
                    to_write_names.append(n_r)
            to_write_line=[]
            for n in rec[0]:
                to_write_line.append(dictToWrite[n])
            rec.append(to_write_line)
            with open(file_name,"w") as f:
                for line_list in rec:
                    f.write(  ",".join(strList(line_list)) +"\n")
        else:
            to_write_line=[]
            for n in rec[0]:
                to_write_line.append(dictToWrite[n])
            with open(file_name,"a") as f:
                f.write(  ",".join(strList(to_write_line)) +"\n")
        re_order_csv(file_name)

        
def re_order_csv(file_name):
    with open(file_name,"r") as f:
        rec=[]
        for line in f:
            line=line.strip("\n").split(",")
            rec.append(line)
    row_names=sorted(enumerate(rec[0]),key=lambda x:x[1])
    row_names_idx=[i[0] for i in row_names]    
    row_names_=[i[1] for i in row_names]    
    if row_names_idx==sorted(row_names_idx):
        print("No need to reorder")
        return None
    else:
        print("reordering")
        with open(file_name,"w") as f:
            for line_list in rec:
                to_write_line=[ line_list[row_names_idx[i]]  for i in range(len(line_list))  ]
                f.write(  ",".join(to_write_line) +"\n")













def func_args_parse(*args,**kargs):
    return args,kargs

class blank_profile():
    def __init__(self,*args,**kwargs):
        pass
    def __enter__(self,*args,**kwargs):
        return self
    def __exit__(self,*args,**kwargs):
        pass
    def step(self):
        pass

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

    def visualize_tsne(self,dn,node_idx_by_ntype):
        from matplotlib.pyplot import figure

        figure(figsize=(16, 9), dpi=80)
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
            fig.set_size_inches(16,9)
            fig.set_dpi(100)
            nts=list(range(len(node_idx_by_ntype)))
            for nt in nts:
                subax=ax[int((nt)/(len(nts)/2))][int((nt)%(len(nts)/2))]
                subax.cla()
                datas=np.array(self.data_dict[f"tsne_emb_layer_{layer}_slot_{nt}"]).T
                print(datas.shape)
                for nt_j in nts:
                    x=datas[0][ node_idx_by_ntype[nt_j]]
                    y=datas[1][ node_idx_by_ntype[nt_j]]
                    subax.scatter(x=x,y=y,c=ncs[nt_j],s=0.1,label=f"type {nt_j} with center ({np.mean(x):.2f},{np.mean(y):.2f}) and radius {(np.std(x)+np.std(y))/2:.2f}")
                subax.set_xlim(-100,100)
                subax.set_ylim(-100,100)
                subax.set_title(f"layer_{layer}_slot_{nt}")
                plt.title(f"layer_{layer}_slot_{nt}")
                lgnd=subax.legend()
                for lh in lgnd.legendHandles:
                    lh._sizes=[10]
            fig.suptitle(f"embedding_tsne_layer_{layer}")
            plt.savefig(os.path.join(dn,f"slot_embeddings_layer_{layer}.png"))



                
                    





def single_feat_net(net,*args,**kargs):
    return net(*args,**kargs)




def idx_to_one_hot(idx_arr):
    one_hot = np.zeros((idx_arr.shape[0], idx_arr.max() + 1))
    one_hot[np.arange(idx_arr.shape[0]), idx_arr] = 1
    return one_hot


def kmeans_test(X, y, n_clusters, repeat=10):
    nmi_list = []
    ari_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_pred)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)


def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8), repeat=10):
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list


def evaluate_results_nc(embeddings, labels, num_classes):
    print('SVM test')
    svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels)
    print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(macro_f1_mean, macro_f1_std, train_size) for
                                    (macro_f1_mean, macro_f1_std), train_size in
                                    zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(micro_f1_mean, micro_f1_std, train_size) for
                                    (micro_f1_mean, micro_f1_std), train_size in
                                    zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('K-means test')
    nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes)
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))

    return svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std


def parse_adjlist(adjlist, edge_metapath_indices, samples=None):
    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(' ')))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:
            # sampling neighbors
            if samples is None:
                neighbors = row_parsed[1:]
                result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))
                neighbors = [row_parsed[i + 1] for i in sampled_idx]
                result_indices.append(indices[sampled_idx])
        else:
            neighbors = []
            result_indices.append(indices)
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping


def parse_minibatch(adjlists, edge_metapath_indices_list, idx_batch, device, samples=None):
    g_list = []
    result_indices_list = []
    idx_batch_mapped_list = []
    for adjlist, indices in zip(adjlists, edge_metapath_indices_list):
        edges, result_indices, num_nodes, mapping = parse_adjlist(
            [adjlist[i] for i in idx_batch], [indices[i] for i in idx_batch], samples)

        g = dgl.DGLGraph(multigraph=True)
        g.add_nodes(num_nodes)
        if len(edges) > 0:
            sorted_index = sorted(range(len(edges)), key=lambda i : edges[i])
            g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
            result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
        else:
            result_indices = torch.LongTensor(result_indices).to(device)
        #g.add_edges(*list(zip(*[(dst, src) for src, dst in sorted(edges)])))
        #result_indices = torch.LongTensor(result_indices).to(device)
        g_list.append(g)
        result_indices_list.append(result_indices)
        idx_batch_mapped_list.append(np.array([mapping[idx] for idx in idx_batch]))

    return g_list, result_indices_list, idx_batch_mapped_list


def parse_adjlist_LastFM(adjlist, edge_metapath_indices, samples=None, exclude=None, offset=None, mode=None):
    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(' ')))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:
            # sampling neighbors
            if samples is None:
                if exclude is not None:
                    if mode == 0:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for u1, a1, u2, a2 in indices[:, [0, 1, -1, -2]]]
                    else:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for a1, u1, a2, u2 in indices[:, [0, 1, -1, -2]]]
                    neighbors = np.array(row_parsed[1:])[mask]
                    result_indices.append(indices[mask])
                else:
                    neighbors = row_parsed[1:]
                    result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))
                if exclude is not None:
                    if mode == 0:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for u1, a1, u2, a2 in indices[sampled_idx][:, [0, 1, -1, -2]]]
                    else:
                        mask = [False if [u1, a1 - offset] in exclude or [u2, a2 - offset] in exclude else True for a1, u1, a2, u2 in indices[sampled_idx][:, [0, 1, -1, -2]]]
                    neighbors = np.array([row_parsed[i + 1] for i in sampled_idx])[mask]
                    result_indices.append(indices[sampled_idx][mask])
                else:
                    neighbors = [row_parsed[i + 1] for i in sampled_idx]
                    result_indices.append(indices[sampled_idx])
        else:
            neighbors = [row_parsed[0]]
            indices = np.array([[row_parsed[0]] * indices.shape[1]])
            if mode == 1:
                indices += offset
            result_indices.append(indices)
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping


def parse_minibatch_LastFM(adjlists_ua, edge_metapath_indices_list_ua, user_artist_batch, device, samples=None, use_masks=None, offset=None):
    g_lists = [[], []]
    result_indices_lists = [[], []]
    idx_batch_mapped_lists = [[], []]
    for mode, (adjlists, edge_metapath_indices_list) in enumerate(zip(adjlists_ua, edge_metapath_indices_list_ua)):
        for adjlist, indices, use_mask in zip(adjlists, edge_metapath_indices_list, use_masks[mode]):
            if use_mask:
                edges, result_indices, num_nodes, mapping = parse_adjlist_LastFM(
                    [adjlist[row[mode]] for row in user_artist_batch], [indices[row[mode]] for row in user_artist_batch], samples, user_artist_batch, offset, mode)
            else:
                edges, result_indices, num_nodes, mapping = parse_adjlist_LastFM(
                    [adjlist[row[mode]] for row in user_artist_batch], [indices[row[mode]] for row in user_artist_batch], samples, offset=offset, mode=mode)

            g = dgl.DGLGraph(multigraph=True)
            g.add_nodes(num_nodes)
            if len(edges) > 0:
                sorted_index = sorted(range(len(edges)), key=lambda i : edges[i])
                g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
                result_indices = torch.LongTensor(result_indices[sorted_index]).to(device)
            else:
                result_indices = torch.LongTensor(result_indices).to(device)
            g_lists[mode].append(g)
            result_indices_lists[mode].append(result_indices)
            idx_batch_mapped_lists[mode].append(np.array([mapping[row[mode]] for row in user_artist_batch]))

    return g_lists, result_indices_lists, idx_batch_mapped_lists


class index_generator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0
