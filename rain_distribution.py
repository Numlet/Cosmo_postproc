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
folder_in_path='1h/'

#con

starting_date='2079010100'
end_date='2088123123'


starting_date='1999010100'
end_date='2008123123'

present_dates=jle.Hourly_time_list(starting_date,end_date)
list_present_files=jle.Locate_files(model_output_folder,folder_in_path)

present_address_dict=jle.Create_address_dict(present_dates,list_present_files)

sample_dataset=Dataset(present_address_dict[next(iter(present_address_dict))])
sample_dataset=Dataset(present_address_dict[list(present_address_dict.keys())[14]])



f=jle.Load_sample_dataset_c()




lons = f.variables['lon'][:]
lats = f.variables['lat'][:]
#%%
plt.figure(figsize=(25,25))


rain_threshold=0.0
year='2006'
for month in jle.months_number_str:
#for month in ['07']:
    print(month)
    month_address_dict=jle.Filter_dict(present_address_dict,years=int(year),months=int(month))
    keys=[key for key in month_address_dict]
    if len(month_address_dict)==0:
        continue
    rain_in_month=np.zeros((len(keys),lats.shape[0],lats.shape[1]))
    conv_rain_in_month=np.zeros((len(keys),lats.shape[0],lats.shape[1]))
    grsc_rain_in_month=np.zeros((len(keys),lats.shape[0],lats.shape[1]))
    div_in_month=np.zeros((len(keys),lats.shape[0],lats.shape[1]))
    vor_in_month=np.zeros((len(keys),lats.shape[0],lats.shape[1]))
    for i in range(len(keys)):
        print(keys[i])        
        dataset=Dataset(month_address_dict[keys[i]])
#        rain_in_month[i,:,:]=dataset.variables['TOT_PREC'][0,]+dataset.variables['TOT_SNOW'][0,]
#        conv_rain_in_month[i,:,:]=dataset.variables['RAIN_CON'][0,]+dataset.variables['SNOW_CON'][0,]
#        grsc_rain_in_month[i,:,:]=dataset.variables['RAIN_GSP'][0,]+dataset.variables['SNOW_GSP'][0,]

        conv_rain_in_month[i,:,:]=dataset.variables['RAIN_CON'][0,]+dataset.variables['SNOW_CON'][0,]
        grsc_rain_in_month[i,:,:]=dataset.variables['RAIN_GSP'][0,]+dataset.variables['SNOW_GSP'][0,]
        rain_in_month[i,:,:]=conv_rain_in_month[i,:,:]+grsc_rain_in_month[i,:,:]
        U=sample_dataset.variables['U_10M'][0,]
        V=sample_dataset.variables['V_10M'][0,]

        div=(U[2:,1:-1]-U[:-2,1:-1])+(V[1:-1,2:]-V[1:-1,:-2])
        vor=(V[2:,1:-1]-V[:-2,1:-1])+(U[1:-1,2:]-U[1:-1,:-2])
        div_in_month[i,1:-1,1:-1]=div
        vor_in_month[i,1:-1,1:-1]=vor
#        rain_in_month[i,:,:]
    rain_in_month[rain_in_month<rain_threshold]=0#Hace falta esto? Tal vez sea mejor no applicar ningun threshold
    conv_rain_in_month[rain_in_month<rain_threshold]=0#Hace falta esto? Tal vez sea mejor no applicar ningun threshold
    grsc_rain_in_month[rain_in_month<rain_threshold]=0#Hace falta esto? Tal vez sea mejor no applicar ningun threshold
    convective_fraction=conv_rain_in_month.mean(axis=0)/rain_in_month.mean(axis=0)
    plt.subplot(4,3,int(month))
    jle.Quick_plot(convective_fraction,month,metadata_dataset=jle.Load_sample_dataset_c(),levels=np.linspace(0,1,11),new_fig=0)
    np.save(data_folder+'rain_in_month'+month,rain_in_month)
    np.save(data_folder+'conv_rain_in_month'+month,conv_rain_in_month)
    np.save(data_folder+'grsc_rain_in_month'+month,grsc_rain_in_month)
plt.savefig(plots_folder+'all_months.png')

#%%
month='01'

rain_in_month=np.load(data_folder+'rain_in_month'+month+'.npy')
conv_rain_in_month=np.load(data_folder+'conv_rain_in_month'+month+'.npy')
grsc_rain_in_month=np.load(data_folder+'grsc_rain_in_month'+month+'.npy')


print('fraction of rain below 0.1 threshold: %f'%(rain_in_month[rain_in_month<0.1].sum()/rain_in_month.sum()))
rain_threshold=0.1
plt.figure(figsize=(15,7))
plt.subplot(121)
plt.title(jle.month_names[int(month)-1])
#bins=np.linspace(-2,5,100).tolist()
bins=np.logspace(-1,2,100).tolist()
rain=rain_in_month[rain_in_month>rain_threshold]
rain_conv=conv_rain_in_month[rain_in_month>rain_threshold]
rain_grsc=grsc_rain_in_month[rain_in_month>rain_threshold]
rain_conv=rain_conv[rain_conv!=0]
rain_grsc=rain_grsc[rain_grsc!=0]
#plt.hist(np.log(rain),bins=bins,alpha=0.5,label='all_rain')
#plt.hist(np.log(rain_conv),bins=bins,alpha=0.5,label='conv')
#plt.hist(np.log(rain_grsc),bins=bins,alpha=0.5,label='grsc')
plt.hist(rain,bins=bins,alpha=0.5,label='all_rain',cumulative=0)
plt.hist(rain_conv,bins=bins,alpha=0.5,label='conv',cumulative=0)
plt.hist(rain_grsc,bins=bins,alpha=0.5,label='grsc',cumulative=0)
plt.xlabel(' (rain_intensity)')
plt.xscale('log')
plt.subplot(122)
plt.title('cumulative')
plt.hist(rain,bins=bins,alpha=0.5,label='all_rain',cumulative=1)
plt.hist(rain_conv,bins=bins,alpha=0.5,label='conv',cumulative=1)
plt.hist(rain_grsc,bins=bins,alpha=0.5,label='grsc',cumulative=1)
plt.legend()
#plt.axvline(np.log(0.1),c='r')
#plt.yscale('log')
plt.xlabel(' (rain_intensity)')
plt.xscale('log')
#plt.xlim
plt.savefig(plots_folder+'distribution_of_rains_12km_cumulative_'+jle.month_names[int(month)-1]+'.png')
#plt.savefig(plots_folder+'distribution_of_rains_12km.png')
#%%
plt.hist(rain,bins=bins,alpha=0.5,label='all_rain',cumulative=1)
plt.hist(rain_conv,bins=bins,alpha=0.5,label='conv',cumulative=1)
plt.hist(rain_grsc,bins=bins,alpha=0.5,label='grsc',cumulative=1)








#plt.title('fraction of rain below 0.1 threshold: %f'%(rain_in_month[rain_in_month<0.1].sum()/rain_in_month.sum()))
#plt.title
#plt.savefig(plots_folder+'all_rain_without_threshold_histogram')
