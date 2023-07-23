
import random
import queue
import time
import subprocess
import multiprocessing
from threading import main_thread
import os
import pandas as pd

import copy








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
    command='python -W ignore run_dist.py  '
    for key in args_dict.keys():
        command+=f" --{key} {args_dict[key]} "


    command+=f" --gpu {gpu} "
    if os.name!="nt":
        command+=f"   > ./log/{args_dict['study_name']}.txt  "
    return command





def config_study_name(prefix,specified_args,extract_dict):
    study_name=prefix
    for k in specified_args:
        v=extract_dict[k]
        study_name+=f"_{k}_{v}"
    if study_name[0]=="_":
        study_name=study_name.replace("_","",1) 
    return study_name 
