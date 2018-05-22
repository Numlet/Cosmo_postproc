#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:44:16 2018

@author: jvergara
"""

import sys
sys.path.append('/users/jvergara/python_code')
#import matplotlib
#matplotlib.use('Agg')
import Jesuslib_eth as jle
import numpy as np
import matplotlib.pyplot as plt
import glob
import iris
from netCDF4 import Dataset
import time
import os
import matplotlib.animation as manimation
from pympler import muppy
all_objects = muppy.get_objects()
from pympler import summary
import pickle
import scipy


grid_2km='/store/c2sm/pr04/jvergara/grid_2km'
#grid_2km='/store/c2sm/pr04/jvergara/grid_2km_david'
grid_12km='/store/c2sm/pr04/jvergara/grid_12km'


run_2km='/store/c2sm/pr04/jvergara/BASE_RUN/output/'
run_2km='/project/pr04/davidle/results_clim/lm_f/'

output_path=run_2km#+'regrided_12km/'
output_path='/store/c2sm/pr04/jvergara/DAVID_REGRIDED/'
jle.Create_folder(output_path)


def Regrid(file,file_output_with_path_no_vcoord,file_output_with_path):
    do=0
    if not os.path.isfile(file_output_with_path):
        do=1
    elif os.path.isfile(file_output_with_path_no_vcoord):
        do=1
    if do:
        a=os.system('cdo delname,vcoord %s %s'%(file,file_output_with_path_no_vcoord))
        a=os.system('cdo remapcon,%s -setgrid,%s %s %s'%(grid_12km,grid_2km,file_output_with_path_no_vcoord,file_output_with_path))
        a=os.system('rm -f %s'%(file_output_with_path_no_vcoord))

#%%

folders=glob.glob(run_2km+'*/*/')


files_dict={}

for folder in folders:
    files=glob.glob(folder+'*nc')
    if len(files)>0:
        files_dict[folder]=np.sort([f for f in files]).tolist()

#cdo remapcon,grid_12km -setgrid,grid_2km lffd2001061312_2km_no_vcoord.nc lffd2001061312_2km_regrid_05.nc
#cdo delname,vcoord lffd2001061312_2km.nc lffd2001061312_2km_no_vcoord.nc

years_between_output=1

for folder in files_dict:
    print(folder)
    if years_between_output:
        
        folder_name=folder.split('/')[-3]
        folder_name2=folder.split('/')[-2]
        jle.Create_folder(output_path+folder_name)
        jle.Create_folder(output_path+folder_name+'/'+folder_name2)
        output=output_path+folder_name+'/'+folder_name2+'/'
    else:
        folder_name=folder.split('/')[-2]
        jle.Create_folder(output_path+folder_name)
        output=output_path+folder_name+'/'
    jle.Create_folder(output)
    
    import time
    import multiprocessing
    processes=24
    print ('Number of files', len(files_dict[folder]))
    list_of_chunks=np.array_split(files_dict[folder],len(files_dict[folder])/processes+1)
    start=time.time()
    for chunk in list_of_chunks:
        jobs=[]
        for file in chunk:

    
#    for file in files_dict[folder]:
            print(file)
            file_name=file.split('/')[-1]
            
            
            
            file_name_output=file_name[:-3]+'_regrided_12km.nc'
            file_name_output_no_vcoord=file_name[:-3]+'_no_vcoord.nc'
            
            print(output+file_name_output)
            file_output_with_path=output+file_name_output
            file_output_with_path_no_vcoord=output+file_name_output_no_vcoord
#            Regrid(file,file_output_with_path_no_vcoord,file_output_with_path)

#            args=(file,file_output_with_path_no_vcoord,file_output_with_path)
            p = multiprocessing.Process(target=Regrid, args=(file,file_output_with_path_no_vcoord,file_output_with_path))
            print (file,p)
            jobs.append(p)
            p.start()
        for job in jobs:
            job.join()
        
#        a=os.system('cdo delname,vcoord %s %s'%(file,file_output_with_path_no_vcoord))
#        a=os.system('cdo remapcon,%s -setgrid,%s %s %s'%(grid_12km,grid_2km,file_output_with_path_no_vcoord,file_output_with_path))
#        a=os.system('rm -f %s'%(file_output_with_path_no_vcoord))

