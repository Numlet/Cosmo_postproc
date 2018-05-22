#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:23:37 2018

@author: jvergara
"""

import sys
sys.path.append('/users/jvergara/python_code')
import matplotlib
matplotlib.use('Agg')
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


data_saving_folder='/store/c2sm/pr04/jvergara/DRY_DAYS/'
saving_folder='/users/jvergara/dry_days_plots/'
data_saving_folder='/store/c2sm/pr04/jvergara/DRY_DAYS_C/'

project_name='DRY_DAYS_F_REGRIDED'
plots_folder,data_saving_folder=jle.Create_project_folders(project_name)

def count_consecutive_ones(array):
    list_ones=[]
    ones=0
    for i in range(len(array)):
        if array[i]==1:
            ones=ones+1
        if array[i]==0:
            list_ones.append(ones)
            ones=0
    list_ones.append(ones)
    return np.max(list_ones)
nikolinas='/project/pr04/banni/results_PGW/PGW/lm_f/'
davids='/project/pr04/davidle/results_clim/lm_f/'
nikolinas_continue='/project/pr04/banni/results_PGW/PGWcontinue/lm_f/'

nikolinas='/project/pr04/banni/results_PGW/PGW/lm_c/'
davids='/project/pr04/davidle/results_clim/lm_c/'
nikolinas_continue='/project/pr04/banni/results_PGW/PGWcontinue/lm_c/'


nikolinas='/store/c2sm/pr04/jvergara/NIKOLINA_REGRIDED/'
davids='/store/c2sm/pr04/jvergara/DAVID_REGRIDED/'
nikolinas_continue=nikolinas


folder_in_path='1h/'

starting_date='2079010100'
end_date='2088123123'

future_dates=jle.Hourly_time_list(starting_date,end_date)
list_future_files=jle.Locate_files(nikolinas,folder_in_path)
future_address_dict=jle.Create_address_dict(future_dates,list_future_files)

list_future_files_continue=jle.Locate_files(nikolinas_continue,folder_in_path)
future_address_dict_continue=jle.Create_address_dict(future_dates,list_future_files_continue)

future_address_dict = {**future_address_dict, **future_address_dict_continue}


#davids='/project/pr04/davidle/results_clim/lm_f/'

folder_in_path='1h/'

starting_date='1999010100'
end_date='2008123123'

present_dates=jle.Hourly_time_list(starting_date,end_date)
list_present_files=jle.Locate_files(davids,folder_in_path)

#try: 
#    present_address_dict=pickle.load(open(saving_folder+'present_address_dict','rb'))
#except:
present_address_dict=jle.Create_address_dict(present_dates,list_present_files)
#    pickle.dump(present_address_dict,open(saving_folder+'present_address_dict','wb'))
sample_dataset=Dataset(present_address_dict[starting_date])
sample_dataset=jle.Load_sample_dataset_c()


all_address_dict={**future_address_dict, **present_address_dict}

summer_address_dict=jle.Filter_dict(all_address_dict,months=[5,9])
years=np.sort(list(set([date[:4] for date in summer_address_dict.keys()]))).tolist()

#%%
for year in years:
    print (year)
    t1=time.time()
    keys=[string for string in summer_address_dict if year in string[:4]]
    days=int(len(keys)/24)
    if len(keys)%24: raise NameError('Number of files is not divisible by 24')
    
    X=sample_dataset.variables['lon']
    Y=sample_dataset.variables['lat']
    
    
    precip=np.zeros((days,X.shape[0],X.shape[1]))
    iday=0
    for i in range(len(keys)):
        if not i==0:
            if i%24==0:
                iday=iday+1
#        print (iday,i,i-24*iday)
#        print (keys[i])
        dataset=Dataset(summer_address_dict[keys[i]])
        precip[iday,:,:]=precip[iday,:,:]+dataset.variables['TOT_PREC'][0,:,:]
    tm=time.time()
    print(tm-t1)
    dry_day=np.zeros(precip.shape)
    dry_day[precip<1]=1
    
    max_number_of_dry_days=np.zeros(X.shape)
    for ilon in range(max_number_of_dry_days.shape[0]):
        print (ilon,X.shape[0])
        for ilat in range(max_number_of_dry_days.shape[1]):
            array=dry_day[:,ilon,ilat]
            max_number_of_dry_days[ilon,ilat]=count_consecutive_ones(array)
    
    
    np.save(data_saving_folder+'summer_dry_days_'+str(year),max_number_of_dry_days)
    te=time.time()
    print(te-t1)
#%%
#import netCDF4 
#include = ['rotated_pole', 'rlon', 'rlat', 'srlon', 'srlat', 'lon', 'lat', 'slonu', 'slatu', 'slonv', 'slatv', 'vcoord', 'height_2m', 'height_10m', 'height_toa']
##dst=nc.Dataset()
#src=sample_dataset
#with netCDF4.Dataset("out.nc", "w") as dst:
#    # copy global attributes all at once via dictionary
#    dst.setncatts(src.__dict__)
#    # copy dimensions
#    for name, dimension in src.dimensions.items():
#        dst.createDimension(
#            name, (len(dimension) if not dimension.isunlimited() else None))
#    # copy all file data except for the excluded
#    for name, variable in src.variables.items():
#        if name in include:
#            x = dst.createVariable(name, variable.datatype, variable.dimensions)
#            dst[name][:] = src[name][:]
#            # copy variable attributes all at once via dictionary
#            dst[name].setncatts(src[name].__dict__)

#%%
#jle.Quick_plot(max_number_of_dry_days,'Dry days',metadata_dataset=sample_dataset)
#    precip[]




#%%

#%%

