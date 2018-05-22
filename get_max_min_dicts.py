#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 13:22:47 2018

@author: jvergara
"""

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




starting_date='2008010100'
end_date='2009010100'

list_dates=jle.Hourly_time_list(starting_date,end_date)
list_files=jle.Locate_files('/project/pr04/davidle/results_clim/lm_f/','1h/')
address_dict=jle.Create_address_dict(list_dates,list_files)
#%%


sample_file=address_dict['2008010100']

sample_dataset=Dataset(sample_file)

list_of_variables=sample_dataset.variables.keys()

exclude_variables=['time','time_bnds','rotated_pole','rlon','rlat', 'srlon', 'srlat', 'lon', 'lat', 'slonu', 'slatu', 'slonv', 'slatv', 'vcoord', 'height_2m', 'height_10m']

dict_max={}
dict_min={}
for variable in list_of_variables:
    if variable not in exclude_variables:
        dict_max[variable]=0
        dict_min[variable]=0
        
for date in address_dict:
    day_digit=date[7]
    if not int(day_digit)%5==0: continue
    print (date)
    
    dataset=Dataset(address_dict[date])
    for variable in list_of_variables:
        if variable in exclude_variables:
            continue
        
        max_value=dataset.variables[variable][:].max()
        min_value=dataset.variables[variable][:].min()
        if max_value>dict_max[variable]:dict_max[variable]=max_value
        if min_value<dict_min[variable]:dict_min[variable]=min_value
        

#%%

saving_folder='/store/c2sm/pr04/jvergara/SW_plots/'
with open(saving_folder+"dict_max.txt", "wb") as myFile:
    pickle.dump(dict_max, myFile)
with open(saving_folder+"dict_min.txt", "wb") as myFile:
    pickle.dump(dict_min, myFile)
