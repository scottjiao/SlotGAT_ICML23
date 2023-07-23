
import random
import queue
import time
import subprocess
import multiprocessing
from threading import main_thread
import os
import pandas as pd

import copy

def get_tasks_for_online(dataset_and_hypers):
    pass


def get_tasks_linear_around(task_space,best_hyper):
    tasks=[]
    for param_in_space,param_values in task_space.items():
        assert param_in_space in best_hyper.keys()
        for param_value in param_values:
            temp_t={}
            #copy best hyper except specified param
            for param_in_best,value_in_best in best_hyper.items():
                if "search_" in param_in_best:
                    
                    temp_t[param_in_best]= f"[{value_in_best}]"  if param_in_best!=param_in_space else f"[{param_value}]" 
                else:
                    
                    temp_t[param_in_best]=value_in_best if param_in_best!=param_in_space else param_value
            tasks.append(temp_t)

    
    return tasks


def get_tasks(task_space):
    tasks=[{}]
    for k,v in task_space.items():
        tasks=expand_task(tasks,k,v)
    return tasks

def expand_task(tasks,k,v):
    temp_tasks=[]
    if type(v) is str and type(eval(v)) is list:
        for value in eval(v):
            if k.startswith("search_"):
                value=str([value])
            for t in tasks:
                temp_t=copy.deepcopy(t)
                temp_t[k]=value
                temp_tasks.append(temp_t)
    elif type(v) is list:
        for value in v:
            for t in tasks:
                temp_t=copy.deepcopy(t)
                temp_t[k]=value
                temp_tasks.append(temp_t)
    else:
        for t in tasks:
            temp_t=copy.deepcopy(t)
            temp_t[k]=v
            temp_tasks.append(temp_t)
    return temp_tasks


 

def proc_yes(yes,args_dict):
    temp_yes=[]
    for name in yes:
        temp_yes.append(f"{name}_{args_dict[name]}")
    return temp_yes

def get_best_hypers_from_csv(dataset,net,yes,no,metric="2_valAcc"):
    print(f"yes: {yes}, no: {no}")
    #get search best hypers
    fns=[]
    for root, dirs, files in os.walk("./log", topdown=False):
        for name in files:
            FLAG=1
            if "old" in root:
                continue
            if ".py" in name:
                continue
            if ".txt" in name:
                continue
            if ".csv" not in name:
                continue
            for n in no:
                if n in name:
                    FLAG=0
            for y in yes:
                if y not in name:
                    FLAG=0
            if FLAG==0:
                continue

            if dataset in name:
                name0=name.replace("_GTN","",1) if "kdd" not in name else name
                if net in name0 :

                    fn=os.path.join(root, name)
                    fns.append(fn)
    score_max=0
    print(fns)
    if fns==[]:
        raise Exception
    for fn in fns:

        param_data=pd.read_csv(fn)
        param_data_sorted=param_data.sort_values(by=metric,ascending=False).head(1)
        #print(param_data_sorted.columns)
        param_mapping={"1_Lr":"search_lr",
        "1_Wd":"search_weight_decay",
        "1_featType":"feats-type",
        "1_hiddenDim":"search_hidden_dim",
        "1_numLayers":"search_num_layers",
        "1_numOfHeads":"search_num_heads",}
        score=param_data_sorted[metric].iloc[0]
        if score>score_max:
            print(   f"score:{score}\t {param_data_sorted} bigger than current score {score_max} "  )
            best_hypers={}
            score_max=score
            best_param_data_sorted=param_data_sorted
            for col_name in param_data_sorted.columns:
                if col_name.startswith("1_"):
                    if param_mapping[col_name].startswith("search_"):
                        best_hypers[param_mapping[col_name]]=f"[{param_data_sorted[col_name].iloc[0]}]"
                    else:
                        best_hypers[param_mapping[col_name]]=f"{param_data_sorted[col_name].iloc[0]}"
        print(f"Best Score:{score_max}\t {best_param_data_sorted}")
        

    return best_hypers

def get_best_hypers(dataset,net,yes,no):
    print(f"yes: {yes}, no: {no}")
    #get search best hypers
    best={}
    fns=[]
    for root, dirs, files in os.walk("./log", topdown=False):
        for name in files:
            FLAG=1
            if "old" in root:
                continue
            if ".py" in name:
                continue
            if ".txt" in name:
                continue
            for n in no:
                if n in name:
                    FLAG=0
            for y in yes:
                if y not in name:
                    FLAG=0
            if FLAG==0:
                continue

            if dataset in name:
                name0=name.replace("_GTN","",1) if "kdd" not in name else name
                if net in name0 :

                    fn=os.path.join(root, name)
                    fns.append(fn)
    score_max=0
    print(fns)
    if fns==[]:
        raise Exception
    for fn in fns:
        path=fn
        FLAG0=False
        FLAG1=False
        with open(fn,"r") as f:
            for line in f:
                if "Best trial" in line and FLAG0==False:
                    FLAG0=True
                    FLAG1=False
                    continue
                if FLAG0==True:
                    if "Value" in line:
                        _,score=line.strip("\n").replace(" ","").split(":")
                        score=float(score)
                        continue
                    if "Params:" in line:
                        FLAG1=True
                        count=0
                        continue
                if FLAG1==True and score>=score_max and "    " in line and count<=5:

                    param,value=line.strip("\n").replace(" ","").split(":")
                    best[param]=value
                    score_max=score
                    FLAG0=False
                    count+=1
        print(best)
        best_hypers={}
        for key in best.keys():
            best_hypers["search_"+key]=f"""[{best[key]}]"""
    return best_hypers


class Run( multiprocessing.Process):
    def __init__(self,task,pool=0,idx=0,tc=0,start_time=0):
        super().__init__()
        self.task=task
        self.log=os.path.join(task['study_name'])
        self.idx=idx
        self.pool=pool
        self.device=None
        self.tc=tc
        self.start_time=start_time
        #self.pbar=pbar
    def run(self):
        print(f"{'*'*10} study  {self.log} no.{self.idx} waiting for device")
        count=0
        device_units=[]
        while True:
            if len(device_units)>0:
                try:
                    unit=self.pool.get(timeout=10*random.random())
                except queue.Empty:
                    for unit in device_units:
                            self.pool.put(unit)
                    print(f"Hold {str(device_units)} and waiting for too long! Throw back and go to sleep")
                    time.sleep(10*random.random())
                    device_units=[]
                    count=0
                    continue
            else:
                unit=self.pool.get()
            if len(device_units)>0:  # consistency check
                if unit[0]!=device_units[-1][0]:
                    print(f"Get {str(device_units)} and {unit} not consistent devices and throw back it")
                    self.pool.put(unit)
                    time.sleep(10*random.random())
                    continue
            count+=1
            device_units.append(unit)
            if count==self.task['cost']:
                break


        print(f"{'-'*10}  study  {self.log} no.{self.idx} get the devices {str(device_units)} and start working")
        self.device=device_units[0][0]
        try:
            exit_command=get_command_from_argsDict(self.task,self.device,self.idx)
            
            print(f"running: {exit_command}")
            subprocess.run(exit_command,shell=True)
        finally:
            for unit in device_units:
                self.pool.put(unit)
            #localtime = time.asctime( time.localtime(time.time()) )
        
        end_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"Start time: {self.start_time}\nEnd time: {end_time}\nwith {self.idx}/{self.tc} tasks")

        print(f"  {'<'*10} end  study  {self.log} no.{self.idx} of command ")



def get_command_from_argsDict(args_dict,gpu,idx):
    command='python -W ignore run_analysis.py  '
    for key in args_dict.keys():
        command+=f" --{key} {args_dict[key]} "


    command+=f" --gpu {gpu} "
    if os.name!="nt":
        command+=f"   > ./log/{args_dict['study_name']}.txt  "
    return command



def run_command_in_parallel(args_dict,gpus,worker_num):


    command='python -W ignore run_dist.py  '
    for key in args_dict.keys():
        command+=f" --{key} {args_dict[key]} "


    process_queue=[]
    for gpu in gpus:
        
        command+=f" --gpu {gpu} "
        command+=f"   > ./log/{args_dict['study_name']}.txt  "
        for _ in range(worker_num):
            
            print(f"running: {command}")
            p=Run(command)
            p.daemon=True
            p.start()
            process_queue.append(p)
            time.sleep(5)

    for p in process_queue:
        p.join()




def config_study_name(prefix,specified_args,extract_dict):
    study_name=prefix
    for k in specified_args:
        v=extract_dict[k]
        study_name+=f"_{k}_{v}"
    if study_name[0]=="_":
        study_name=study_name.replace("_","",1)
    study_storage=f"sqlite:///db/{study_name}.db"
    return study_name,study_storage
