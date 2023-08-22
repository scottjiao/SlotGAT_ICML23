
# SlotGAT: Slot-based Message Passing for Heterogeneous Graphs (ICML 2023)

Code and data for SlotGAT: Slot-based Message Passing for Heterogeneous Graphs (ICML 2023) (https://proceedings.mlr.press/v202/zhou23j)

Please cite our paper if you use the code or data.

```@InProceedings{pmlr-v202-zhou23j,
  title = 	 {SlotGAT: Slot-based Message Passing for Heterogeneous Graphs},
  author =       {Zhou, Ziang and Shi, Jieming and Yang, Renchi and Zou, Yuanhang and Li, Qing},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {42644--42657},
  year = 	 {2023},
  volume = 	 {202},
  month = 	 {23--29 Jul},
  publisher =    {PMLR}
}
```


## Data and trained models

The data and trained models could be downloaded in the following link:


* LP: https://drive.google.com/drive/folders/1mFAlrQwQeKLEJ3Tv8fatmHninRn6Fj68?usp=drive_link
* NC: https://drive.google.com/drive/folders/1Ga68xitx5MMxT7XW95xIQiJFs2qgS1z1?usp=drive_link

You should place the data and trained models with the same directory structure as in the google drive link above.


## Scripts

To conduct experiments, you need to do the following steps.

### 1. cd into the sub-directory

For node classification task:
```bash
cd ./NC/methods/SlotGAT
```

For link prediction task:
```bash
cd ./LP/methods/SlotGAT
```

### 2. evaluate the trained model

```bash
python run_use_slotGAT_on_all_dataset.py
```

Then collect the results in the `./NC/methods/SlotGAT/log` or `./LP/methods/SlotGAT/log` directory respectively.

### 3. train the model

If you want to train the model, you can run the following script.


```bash
python run_train_slotGAT_on_all_dataset.py  
```



## Data format

* All ids begin from 0.
* Each node type takes a continuous range of node_id.
* node_id and node_type id are with same order. I.e. nodes with node_type 0 take the first range of node_ids, nodes with node_type 1 take the second range, and so on.
* One-hot node features can be omited.
* For node classification task, the node type of the target node is 0.


## Note

To be consistent with the `PubMed_NC`, the data of `PubMed_LP` is re-organized, which make it different from `PubMed` in HGB, while other all datasets are the same with HGB. Three changes are made:

1. The node type of 0 and 1 are swapped. Since the main type (target node type in node classification task) of PubMed_NC is 0, we swap the node type of 0 and 1 in PubMed_LP to make the main type of PubMed_LP also 0.

2. The id of nodes are re-ordered. According to previous change, the main type of PubMed_LP is 0. We re-order the nodes of PubMed_LP to make the nodes of main type 0 take the first range of node_ids, nodes of main type 1 take the second range, and so on. For example, for type 0 the range is [0, num_of_type_0_nodes), for type 1 the range is [num_of_type_0_nodes, num_of_type_0_nodes + num_of_type_1_nodes), and so on.

3. The corresponding src and dst node id in `links.dat` and `test.dat` are re-mapped according to the new node ids. 

In summary, we only conduct node type swapping, resulted node re-ordering and link re-mapping. The node features and graph structure are not changed. Thus, the performance of SlotGAT on PubMed_LP is the same as SlotGAT on the original PubMed in HGB. 



## Required environment

* python 3.10.9
* pytorch 1.13.1
* dgl 1.0.1+cu117
* pytorch_geometric 2.2.0
* cuda 11.7
* networkx 2.8.4
* scikit-learn 1.2.1
* scipy 1.10.0




