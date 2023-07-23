

import time 
import multiprocessing
from threading import main_thread
from pipeline_utils import config_study_name,Run
import os 
#time.sleep(60*60*4)


if not os.path.exists("outputs"):
    os.mkdir("outputs")
if not os.path.exists("log"):
    os.mkdir("log")
if not os.path.exists("checkpoint"):
    os.mkdir("checkpoint")




resources_dict={"0":1,"1":1}   #id:load
#dataset_to_evaluate=[("pubmed_HNE_LP",1,5)]   # dataset,cost,repeat
prefix="get_results_trained";specified_args=["dataset",   "net"]

start_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

fixed_info_by_selected_keys={
    "slotGAT":{"task_property":prefix,"net":"slotGAT","slot_aggregator":"SA","inProcessEmb":"True",
                        "use_trained":"True",
                        "trained_dir":"outputs",
                        "save_trained":"False",
                        #"save_dir":"outputs",
                        } }



dataset_and_hypers={
    ("PubMed_LP",1,5):
        {"hidden-dim":"[64]","num-layers":"[4]","lr":"[1e-3]","weight-decay":"[1e-4]","feats-type":[2],"num-heads":[2],"epoch":[1000],"decoder":["distmult"],"batch-size":[8192,],"dropout_feat":[0.5],"dropout_attn":[0.5],"residual_att":[0.2],"residual":["True"],"SAattDim":[32]},
    ("LastFM",1,5):
        {"hidden-dim":"[64]","num-layers":"[8]","lr":"[5e-4]","weight-decay":"[1e-4]","feats-type":[2],"num-heads":[2],"epoch":[1000],"decoder":["dot"],"batch-size":[8192,],"SAattDim":[64],"dropout_feat":[0.2],"dropout_attn":[0.9],"residual_att":[0.5],"residual":["True"]}
    }

def getTasks(fixed_info,dataset_and_hypers):
    
    for k,v in dataset_and_hypers.items():
        for k1,v1 in v.items():
            if "search_" in k1:
                if type(v1)!=str:
                    v[k1]=f"[{v1}]"
            # if list str, get the first one
            if  type(v1)==list:
                v[k1]=v1[0]
            if type(v1)==str:
                if v1[0]=="[" and v1[-1]=="]":
                    v[k1]=eval(v1)[0]
        

    tasks_list=[]
    for (dataset,cost,repeat),(task) in dataset_and_hypers.items():
        args_dict={}
        for dict_to_add in [task,fixed_info]:
            for k,v in dict_to_add.items():
                args_dict[k]=v
        net=args_dict['net']
        args_dict['dataset']=dataset
        #args_dict['trial_num']=trial_num
        args_dict['repeat']=repeat
        study_name =config_study_name(prefix=prefix,specified_args=specified_args,extract_dict=args_dict)
        args_dict['study_name']=study_name 
        args_dict['cost']=cost
        tasks_list.append(args_dict)

    print("tasks_list:", tasks_list)

    return tasks_list 

tasks_list=[]
for k in fixed_info_by_selected_keys.keys():
        
    tasks_list.extend(getTasks(fixed_info_by_selected_keys[k],dataset_and_hypers))






resources=resources_dict.keys()
pool=multiprocessing.Queue( sum([  v  for k,v in resources_dict.items()   ])  )
for i in resources:
    for j in range(resources_dict[i]):
        pool.put(i+str(j))


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
print(f"Start time: {start_time}\nEnd time: {end_time}\n")



