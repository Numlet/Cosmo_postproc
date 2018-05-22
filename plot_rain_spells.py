#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 09:53:04 2018

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
saving_folder='/users/jvergara/dry_days_plots/'
data_saving_folder='/store/c2sm/pr04/jvergara/DRY_DAYS_C/'
data_saving_folder='/store/c2sm/pr04/jvergara/DRY_DAYS_C/'

project_name='RAIN_DURATION'
plots_folder,data_folder=jle.Create_project_folders(project_name)


threshold='with'
threshold='no'

duration_files=glob.glob(data_folder+'*duration_'+threshold+'_threshold.npy')
f=jle.Load_sample_dataset_c()


domain_name='IB'
lonbounds = [ -10 , 3] # degrees east ? 
latbounds = [ 44 , 36 ]



lons = f.variables['lon'][:]
lats = f.variables['lat'][:]

duration_in_months=np.zeros((12,lats.shape[0],lats.shape[1]))

for month in jle.months_number_str:
    duration=np.load('/store/c2sm/pr04/jvergara/RAIN_DURATION/2008'+month+'duration_'+threshold+'_threshold.npy')
    duration[duration==0]=np.nan
    duration=np.nanmean(duration,axis=0)
    duration_in_months[int(month)-1,:,:]=duration

duration_in_months[np.isnan(duration_in_months)]=0
#jle.Quick_plot(duration_in_months[0,],'January',metadata_dataset=jle.Load_sample_dataset_c(),levels=np.logspace(0,3,10),cmap=plt.cm.RdBu)
#jle.Quick_plot(duration_in_months[6,],'July',metadata_dataset=jle.Load_sample_dataset_c(),levels=np.logspace(0,3,10),cmap=plt.cm.RdBu)



levels=np.logspace(0,3,10)
levels=np.linspace(0,50,10)
levels=[0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,50,70]
levels=[0,1,2,3,4,5,6,7,8,9,10,15,20,25,30]
plt.figure(figsize=(20,20))
plt.figtext(0.5,0.90,'duration in hours '+threshold+' threshold',fontsize=20)
plt.subplot(221)
jle.Quick_plot(duration_in_months[0,],'January',metadata_dataset=jle.Load_sample_dataset_c(),levels=levels,cmap=plt.cm.RdBu,new_fig=0)
plt.subplot(222)
jle.Quick_plot(duration_in_months[3,],'April',metadata_dataset=jle.Load_sample_dataset_c(),levels=levels,cmap=plt.cm.RdBu,new_fig=0)
plt.subplot(223)
jle.Quick_plot(duration_in_months[6,],'July',metadata_dataset=jle.Load_sample_dataset_c(),levels=levels,cmap=plt.cm.RdBu,new_fig=0)
plt.subplot(224)
jle.Quick_plot(duration_in_months[9,],'October',metadata_dataset=jle.Load_sample_dataset_c(),levels=levels,cmap=plt.cm.RdBu,new_fig=0)
plt.savefig(plots_folder+'duration_'+threshold+'_threshold.png')
#plt.colorbar()
#%%
mean_intensity_files=glob.glob(data_folder+'*mean_intensity_events_'+threshold+'_threshold.npy')

mean_intensity_in_months=np.zeros((12,lats.shape[0],lats.shape[1]))

for month in jle.months_number_str:
    mean_intensity=np.load('/store/c2sm/pr04/jvergara/RAIN_DURATION/2008'+month+'mean_intensity_events_'+threshold+'_threshold.npy')
    mean_intensity[mean_intensity==0]=np.nan
    mean_intensity=np.nanmean(mean_intensity,axis=0)
    mean_intensity_in_months[int(month)-1,:,:]=mean_intensity

levels=np.logspace(-4,0,10)
levels=np.linspace(0.001,1,10)
levels=np.linspace(0.001,2,10)
plt.figure(figsize=(20,20))
plt.figtext(0.5,0.90,'Mean intensity in hours '+threshold+' threshold',fontsize=20)

plt.subplot(221)
jle.Quick_plot(mean_intensity_in_months[0,],'January',metadata_dataset=jle.Load_sample_dataset_c(),levels=levels,cmap=plt.cm.RdBu_r,new_fig=0)
plt.subplot(222)
jle.Quick_plot(mean_intensity_in_months[3,],'April',metadata_dataset=jle.Load_sample_dataset_c(),levels=levels,cmap=plt.cm.RdBu_r,new_fig=0)
plt.subplot(223)
jle.Quick_plot(mean_intensity_in_months[6,],'July',metadata_dataset=jle.Load_sample_dataset_c(),levels=levels,cmap=plt.cm.RdBu_r,new_fig=0)
plt.subplot(224)
jle.Quick_plot(mean_intensity_in_months[9,],'October',metadata_dataset=jle.Load_sample_dataset_c(),levels=levels,cmap=plt.cm.RdBu_r,new_fig=0)

plt.savefig(plots_folder+'intensity_'+threshold+'_threshold.png')
#%%
std_files=glob.glob(data_folder+'*std_events_'+threshold+'_threshold.npy')

std_in_months=np.zeros((12,lats.shape[0],lats.shape[1]))

for month in jle.months_number_str:
    std=np.load('/store/c2sm/pr04/jvergara/RAIN_DURATION/2008'+month+'std_events_'+threshold+'_threshold.npy')
    std[std==0]=np.nan
    std=np.nanmean(std,axis=0)
    std_in_months[int(month)-1,:,:]=std

std_in_months[np.isnan(std_in_months)]=0

levels=np.linspace(0,0.5,10)
plt.figure(figsize=(18,18))
plt.figtext(0.5,0.90,'Std '+threshold+' threshold',fontsize=20)
plt.subplot(221)
jle.Quick_plot(std_in_months[0,],'January',metadata_dataset=jle.Load_sample_dataset_c(),levels=levels,cmap=plt.cm.RdBu,new_fig=0)
plt.subplot(222)
jle.Quick_plot(std_in_months[3,],'April',metadata_dataset=jle.Load_sample_dataset_c(),levels=levels,cmap=plt.cm.RdBu,new_fig=0)
plt.subplot(223)
jle.Quick_plot(std_in_months[6,],'July',metadata_dataset=jle.Load_sample_dataset_c(),levels=levels,cmap=plt.cm.RdBu,new_fig=0)
plt.subplot(224)
jle.Quick_plot(std_in_months[9,],'October',metadata_dataset=jle.Load_sample_dataset_c(),levels=levels,cmap=plt.cm.RdBu,new_fig=0)
plt.savefig(plots_folder+'std_'+threshold+'_threshold.png')

#%%
max_events_files=glob.glob(data_folder+'*max_events_'+threshold+'_threshold.npy')

max_events_in_months=np.zeros((12,lats.shape[0],lats.shape[1]))

for month in jle.months_number_str:
    max_events=np.load('/store/c2sm/pr04/jvergara/RAIN_DURATION/2008'+month+'max_events_'+threshold+'_threshold.npy')
    max_events[max_events==0]=np.nan
    max_events=np.nanmean(max_events,axis=0)
    max_events_in_months[int(month)-1,:,:]=max_events

max_events_in_months[np.isnan(max_events_in_months)]=0

levels=levels=np.linspace(0.001,1,10)

plt.figure(figsize=(18,18))
plt.figtext(0.5,0.90,'Max '+threshold+' threshold',fontsize=20)
plt.subplot(221)
jle.Quick_plot(max_events_in_months[0,],'January',metadata_dataset=jle.Load_sample_dataset_c(),levels=levels,cmap=plt.cm.RdBu,new_fig=0)
plt.subplot(222)
jle.Quick_plot(max_events_in_months[3,],'April',metadata_dataset=jle.Load_sample_dataset_c(),levels=levels,cmap=plt.cm.RdBu,new_fig=0)
plt.subplot(223)
jle.Quick_plot(max_events_in_months[6,],'July',metadata_dataset=jle.Load_sample_dataset_c(),levels=levels,cmap=plt.cm.RdBu,new_fig=0)
plt.subplot(224)
jle.Quick_plot(max_events_in_months[9,],'October',metadata_dataset=jle.Load_sample_dataset_c(),levels=levels,cmap=plt.cm.RdBu,new_fig=0)
#plt.colorbar()
plt.savefig(plots_folder+'max_'+threshold+'_threshold.png')
