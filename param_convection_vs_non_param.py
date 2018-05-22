#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 11:03:32 2018

@author: jvergara
"""
import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append('/users/jvergara/python_code')
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



project_name='CONVECTIVE_FRACTION'
plots_folder,data_folder=jle.Create_project_folders(project_name)

model_output_folder='/scratch/snx3000/jvergara/cosmo-pompa_convective_precip/cosmo/test/climate/crClim2km_DVL/2_lm_c/output/'
model_output_folder='/store/c2sm/pr04/jvergara/CONVECTIVE_FRACTION_SIMULATION/output/'
model_output_folder='/store/c2sm/pr04/jvergara/CONVECTIVE_FRACTION_SIMULATION/output_without_conv_param/'
folder_in_path='1h/'



starting_date='2079010100'
end_date='2088123123'


starting_date='1999010100'
end_date='2008123123'

present_dates=jle.Hourly_time_list(starting_date,end_date)
list_present_files=jle.Locate_files(model_output_folder,folder_in_path)

present_address_dict=jle.Create_address_dict(present_dates,list_present_files)

sample_dataset=Dataset(present_address_dict[list(present_address_dict.keys())[14]])

f=jle.Load_sample_dataset_c()

lons = f.variables['lon'][:]
lats = f.variables['lat'][:]
#%%
plt.figure(figsize=(25,25))


rain_threshold=0.1
year='2006'
means=[]
for month in jle.months_number_str:
    print(month)
    month_address_dict=jle.Filter_dict(present_address_dict,years=int(year),months=int(month))
    keys=[key for key in month_address_dict]
    if len(month_address_dict)==0:
        continue
    rain_in_month=np.zeros((len(keys),lats.shape[0],lats.shape[1]))
    conv_rain_in_month=np.zeros((len(keys),lats.shape[0],lats.shape[1]))
    grsc_rain_in_month=np.zeros((len(keys),lats.shape[0],lats.shape[1]))
    for i in range(len(keys)):
        print(keys[i])        
        dataset=Dataset(month_address_dict[keys[i]])
        conv_rain_in_month[i,:,:]=dataset.variables['RAIN_CON'][0,]+dataset.variables['SNOW_CON'][0,]
        grsc_rain_in_month[i,:,:]=dataset.variables['RAIN_GSP'][0,]+dataset.variables['SNOW_GSP'][0,]
    rain_in_month=conv_rain_in_month+grsc_rain_in_month
    rain_in_month[rain_in_month<rain_threshold]=0#Hace falta esto? Tal vez sea mejor no applicar ningun threshold
    conv_rain_in_month[rain_in_month<rain_threshold]=0#Hace falta esto? Tal vez sea mejor no applicar ningun threshold
    grsc_rain_in_month[rain_in_month<rain_threshold]=0#Hace falta esto? Tal vez sea mejor no applicar ningun threshold
    convective_fraction=conv_rain_in_month.mean(axis=0)/rain_in_month.mean(axis=0)
    plt.subplot(4,3,int(month))

    jle.Quick_plot(rain_in_month.sum(axis=0),month+' '+str(rain_in_month.sum(axis=0).mean()),metadata_dataset=jle.Load_sample_dataset_c(),levels=np.logspace(-1,3,11),new_fig=0)
    means.append(rain_in_month.sum(axis=0).mean())
plt.text(0.9,0.5,'annual mean '+str(np.mean(means)))
plt.savefig(plots_folder+'precip_without_convection_all_months.png')



