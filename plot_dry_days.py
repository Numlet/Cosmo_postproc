#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 09:13:40 2018

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


data_saving_folder='/store/c2sm/pr04/jvergara/DRY_DAYS_F_REGRIDED/'
saving_folder='/users/jvergara/dry_days_plots/'

a=glob.glob(data_saving_folder+'*')

dry_dict={}
for name in a:
    dry_dict[name[-8:-4]]=np.load(name)


years=list(dry_dict.keys())
years=np.sort(years)

future_years=['2079', '2080', '2081', '2082', '2083', '2084',
       '2085', '2086', '2087', '2088']

present_years=['1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006',
       '2007', '2008']

present_dry_days=np.zeros((len(present_years),dry_dict['2008'].shape[0],dry_dict['2008'].shape[1]))
future_dry_days=np.zeros((len(future_years),dry_dict['2008'].shape[0],dry_dict['2008'].shape[1]))

for i in range(len(present_years)):
    present_dry_days[i,:,:]=dry_dict[present_years[i]]    

for i in range(len(future_years)):
    future_dry_days[i,:,:]=dry_dict[future_years[i]]


sample_dataset=jle.Load_sample_dataset()
sample_dataset=jle.Load_sample_dataset_c()

levels=np.linspace(0,150,15).tolist()
diff_levels=np.linspace(-50,50,20).tolist()
plt.figure(figsize=(15,15))
plt.subplot(221)
jle.Quick_plot(present_dry_days.mean(axis=0),'Present dry days',metadata_dataset=sample_dataset,levels=levels,new_fig=False)
plt.subplot(222)
jle.Quick_plot(future_dry_days.mean(axis=0),'Future dry days',metadata_dataset=sample_dataset,levels=levels,new_fig=False)
plt.subplot(223)
diff=future_dry_days.mean(axis=0)-present_dry_days.mean(axis=0)
jle.Quick_plot(diff,'Diff Future-Present',metadata_dataset=sample_dataset,cmap=plt.cm.RdBu_r,levels=diff_levels,new_fig=False,title='Future - Present    Mean:%1.3f'%diff.mean())

plt.subplot(224)
percentage=(future_dry_days.mean(axis=0)-present_dry_days.mean(axis=0))/present_dry_days.mean(axis=0)*100
jle.Quick_plot(percentage,'Diff Future-Present',metadata_dataset=sample_dataset,cmap=plt.cm.RdBu_r,levels=np.linspace(-110,110,15).tolist(),new_fig=False,title='Percentage increase Future - Present    Mean:%1.3f'%percentage.mean())

#plt.subplot(224)
#plt.title('Diff histogram')
#data=diff[0,].flatten()
#data=data[data!=0]
#plt.hist(data)
#plt.yscale('log')

plt.savefig(saving_folder+'Dry_days_comparison_Present_PGW.png')
#%%
levels=np.linspace(0,40,20).tolist()
diff_levels=np.linspace(-30,30,20).tolist()
plt.figure(figsize=(15,15))
plt.subplot(221)
jle.Quick_plot(np.std(present_dry_days,axis=0),'Present dry days',metadata_dataset=sample_dataset,levels=levels,new_fig=False)
plt.subplot(222)
jle.Quick_plot(np.std(future_dry_days,axis=0),'Future dry days',metadata_dataset=sample_dataset,levels=levels,new_fig=False)

diff=np.std(future_dry_days,axis=0)-np.std(present_dry_days,axis=0)
plt.subplot(223)
jle.Quick_plot(diff,'Diff Future-Present',metadata_dataset=sample_dataset,cmap=plt.cm.RdBu_r,levels=diff_levels,new_fig=False,title='Future - Present    Mean:%1.3f'%diff.mean())

plt.savefig(saving_folder+'Dry_days_std_comparison_Present_PGW.png')

