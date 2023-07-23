import time
import subprocess
import multiprocessing
from threading import main_thread
from pipeline_utils import get_best_hypers,run_command_in_parallel,config_study_name,Run,get_tasks,get_tasks_linear_around,get_tasks_for_online
import os
import copy
#time.sleep(60*60*4)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if not os.path.exists("./log"):
    os.mkdir("./log")
if not os.path.exists("./checkpoint"):
    os.mkdir("./checkpoint")
if not os.path.exists("./analysis"):
    os.mkdir("./analysis")
if not os.path.exists("./outputs"):
    os.mkdir("./outputs")

resources_dict={"0":1,"1":1}   #id:load 

start_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

resources=resources_dict.keys()
pool=multiprocessing.Queue( sum([  v  for k,v in resources_dict.items()   ])  )
for i in resources:
    for j in range(resources_dict[i]):
        pool.put(i+str(j)) 
prefix="get_results_use_trained";specified_args=["dataset",  "net", ]

nets=["slotGAT"] 
dataset_restrict=[]

fixed_info_by_net={"slotGAT":
                        {"task_property":prefix,"net":"slotGAT","slot_aggregator":"SA", "verbose":"True",
                        "use_trained":"True",
                        "trained_dir":"outputs",
                        "save_trained":"False",
                        #"save_dir":"outputs",
                        },
}


dataset_and_hypers_by_net={
        "slotGAT":
            {
             ("IMDB",1,5):
                {"search_hidden_dim":128,"search_num_layers":3,"search_lr":0.0001,"search_weight_decay":0.001,"feats-type":1,"num-heads":8,"epoch":300,"SAattDim":3,"dropout_feat":0.8,"dropout_attn":0.2},
            ("ACM",1,5):
                {"search_hidden_dim":64,"search_num_layers":2,"search_lr":0.001,"search_weight_decay":0.0001,"feats-type":1,"num-heads":8,"epoch":300,"SAattDim":32,"dropout_feat":0.8,"dropout_attn":0.8},
            ("DBLP",1,5): 
                {"search_hidden_dim":64,"search_num_layers":4,"search_lr":0.0001,"search_weight_decay":0.001,"feats-type":1,"num-heads":8,"epoch":300,"SAattDim":32,"dropout_feat":0.5,"dropout_attn":0.5},
            ("Freebase",1,5):
                {"search_hidden_dim":16,"search_num_layers":2,"search_lr":0.0005,"search_weight_decay":0.001,"feats-type":2,"num-heads":8,"epoch":300,"SAattDim":8,"dropout_feat":0.5,"dropout_attn":0.5,"edge-feats":"0"},
            ("PubMed_NC",1,5):
                {"search_hidden_dim":128,"search_num_layers":2,"search_lr":0.005,"search_weight_decay":0.001,"feats-type":1,"num-heads":8,"epoch":300,"SAattDim":3,"dropout_feat":0.2,"dropout_attn":0.8,}
            },
    }





def getTasks(fixed_info,dataset_and_hypers):
    
    for k,v in dataset_and_hypers.items():
        for k1,v1 in v.items():
            if "search_" in k1:
                if type(v1)!=str:
                    v[k1]=f"[{v1}]"
        

    tasks_list=[]
    for (dataset,cost,repeat),(task) in dataset_and_hypers.items():
        if len(dataset_restrict)>0 and dataset not in dataset_restrict:
            continue
        args_dict={}
        for dict_to_add in [task,fixed_info]:
            for k,v in dict_to_add.items():
                args_dict[k]=v 
        args_dict['dataset']=dataset
        #args_dict['trial_num']=trial_num
        args_dict['repeat']=repeat
        study_name,study_storage=config_study_name(prefix=prefix,specified_args=specified_args,extract_dict=args_dict)
        args_dict['study_name']=study_name
        #args_dict['study_storage']=study_storage
        args_dict['cost']=cost
        tasks_list.append(args_dict)

    print("tasks_list:", tasks_list)

    return tasks_list 

tasks_list=[]
for net in nets:
    tasks_list.extend(getTasks(fixed_info_by_net[net],dataset_and_hypers_by_net[net]))









sub_queues=[]
items=len(tasks_list)%60
for i in range(items):
    sub_queues.append(tasks_list[60*i:(60*i+60)])
sub_queues.append(tasks_list[(60*items+60):])

if items==0:
    sub_queues.append(tasks_list)

## split the tasks, or it may exceeds of maximal size of sub-processes of OS.
idx=0
tc=len(tasks_list)
for sub_tasks_list in sub_queues:
    process_queue=[]
    for i in range(len(sub_tasks_list)):
        idx+=1
        p=Run(sub_tasks_list[i],idx=idx,tc=tc,pool=pool,start_time=start_time)
        p.daemon=True
        p.start()
        process_queue.append(p)

    for p in process_queue:
        p.join()
    

print('end all')




end_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())