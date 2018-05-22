#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 10:53:41 2018

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
#import iris
from netCDF4 import Dataset
import time
import os
import matplotlib.animation as manimation
from pympler import muppy
all_objects = muppy.get_objects()
from pympler import summary
import pickle
import scipy





starting_date='2006010100'
end_date='2008123123'


list_dates=jle.Hourly_time_list(starting_date,end_date)
list_days=jle.Daily_time_list(starting_date,end_date)


saving_folder='/users/jvergara/quick_plots/'

run2='/store/c2sm/pr04/jvergara/Q_CRIT_3/'
run2='/store/c2sm/pr04/jvergara/Q_CRIT_1/'
run2='/store/c2sm/pr04/jvergara/CLC_DIAG_09/'
run2='/store/c2sm/pr04/jvergara/INP_COOPERS/'
run2='/store/c2sm/pr04/jvergara/COMPILED_BASE_RUN/'
run2='/store/c2sm/pr04/jvergara/MY_FORK_SECOND_TRY/'
run1='/store/c2sm/pr04/jvergara/BASE_RUN/'
run2='/store/c2sm/pr04/jvergara/MY_FORK_SECOND_TRY_WITH_PR/'
run2=jle.scratch+'PRECIP_BUG/c9856bd1/cosmo/test/climate/crClim2km_DVL/4_lm_f/'
run1='/store/c2sm/pr04/jvergara/BASE_RUN/'
run2='/scratch/snx3000/jvergara/my_fork_second_compile/cosmo/test/climate/crClim2km_DVL/4_lm_f/'
run2='/scratch/snx3000/jvergara/my_fork_second_compile/cosmo/test/climate/crClim2km_DVL/2_lm_c/'
run1='/scratch/snx3000/jvergara/cosmo-pompa_compiled/cosmo/test/climate/crClim2km_DVL/2_lm_c/'
#run1='/store/c2sm/pr04/jvergara/BASE_MY_FORK/'
#run1='/store/c2sm/pr04/jvergara/INP_HOBBS_RANGNO_MEYERS/'
#run2='/store/c2sm/pr04/jvergara/INP_COOPERS_SECOND_TRY/'

name1=run1.split('/')[-2]
name2=run2.split('/')[-2]
folder_in_path='output/1h/'
#folder_in_path='output/1h_second/'
print (name1,name2)
list_files1=jle.Locate_files(run1,folder_in_path)
address_dict1=jle.Create_address_dict(list_dates,list_files1)

list_files2=jle.Locate_files(run2,folder_in_path)
address_dict2=jle.Create_address_dict(list_dates,list_files2)

#%%



str_time='2006100112'
#
sample_file=address_dict2[str_time]
sample_dataset=Dataset(sample_file)
variable='CLCT'
levels=np.linspace(0.001,1,21).tolist()
variable='T_2M'
levels=np.linspace(265,310,21).tolist()
variable='TOT_PREC'
levels=np.linspace(0.01,20,21).tolist()
levels=[0.1,0.4,1.6,6.4,25.6]
#variable='ASOB_T'
#levels=np.linspace(0.1,900,11).tolist()

plt.figure(figsize=(18,18))


dataset1=Dataset(address_dict1[str_time])
dataset2=Dataset(address_dict2[str_time])
for key in dataset1.variables.keys():
    print(key,dataset1.variables[key].long_name)
#plt.title('TEXT')
plt.subplot(221)
#plt.text(-705,-40,'TEXT',fontsize=16)
jle.Quick_plot(dataset1,variable,levels=levels,new_fig=False,title=dataset1.variables[variable].long_name+' '+name1,shadedrelief=1)
plt.subplot(222)
#levels=np.array(levels)/4
jle.Quick_plot(dataset2,variable,levels=levels,new_fig=False,title=name2,shadedrelief=1)


dif=dataset2.variables[variable][:]-dataset1.variables[variable][:]

if dif.max()>np.abs(dif.min()):
    dif_limit=dif.max()
else:
    dif_limit=-dif.min()

dif_levels=np.linspace(-dif_limit,dif_limit,21).tolist()
plt.subplot(223)
if not (dif[0,]==0).sum()==dif[0,].size:
    jle.Quick_plot(dif[0,],variable,metadata_dataset=dataset1,cmap=plt.cm.RdBu,levels=dif_levels,new_fig=False,title=name2+' - '+name1+'    Mean:%1.3f'%dif.mean())
plt.subplot(224)
plt.title('Diff histogram')
data=dif[0,].flatten()
#data=data[data!=0]
plt.hist(data,bins=200)
plt.yscale('log')
#plt.savefig(saving_folder+'Histogram_diff_'+variable+'.png')
#scipy.stats.ttest_ind(dif)
#print()
plt.savefig(saving_folder+'Comparison_'+name1+'_'+name2+'_'+str_time+'_'+variable+'_pcolormesh.png')
#dataser1=
#%%
#data=dif[0,].flatten()
#data=data[data!=0]
#plt.hist(data,bins=200)
#plt.yscale('log')
#plt.savefig(saving_folder+'Histogram_diff_'+variable+'.png')
#jle.Quick_plot(dataset2,'TOT_PR')
levels=np.linspace(0.01,20,21).tolist()

accumulated_prec=dataset2.variables['TOT_PREC']
rate_calculated_prec=dataset2.variables['TOT_PR'][0,]*60*60


plt.figure(figsize=(18,18))
plt.subplot(222)
jle.Quick_plot(dataset1,variable,levels=levels,new_fig=False,title=name1,shadedrelief=1)
plt.subplot(223)
jle.Quick_plot(dataset2,variable,levels=levels,new_fig=False,title=name2,shadedrelief=1)
#plt.text(-705,-40,'TEXT',fontsize=16)
plt.subplot(224)
#levels=np.array(levels)/4
jle.Quick_plot(rate_calculated_prec,'TOT_PR',levels=levels,new_fig=False,title='calculated_from_rate',shadedrelief=1,metadata_dataset=dataset2)

print("Mean Precipitation")
print('BASE:',dataset1.variables['TOT_PREC'][:].mean())
print('MY_FORK:',dataset2.variables['TOT_PREC'][:].mean())
print('MY_FORK calculated from rate:',rate_calculated_prec.mean())

#%%
#plt.pcolormesh(dif[0,])
#plt.pcolormesh(sample_dataset.variables[variable][0,],norm=colors.BoundaryNorm(boundaries=[0.5,0.6,0.7,0.71], ncolors=256))
#plt.colorbar()

#%%
mean_values1=[]
mean_values2=[]
for date in address_dict1:
    print(date)
    if date=='2006100100':continue
    dataset1=Dataset(address_dict1[date])
    dataset2=Dataset(address_dict2[date])
    mean_values1.append(dataset1.variables[variable][:].mean())
    mean_values2.append(dataset2.variables[variable][:].mean())
    print( dataset1.variables[variable][:].mean()/dataset2.variables[variable][:].mean())
plt.figure(figsize=(10,10))
plt.xlabel('Hours since start')
plt.ylabel(dataset1.variables[variable].long_name+'    '+dataset1.variables[variable].units)
plt.plot(mean_values1,label=name1)
plt.plot(mean_values2,label=name2)
plt.legend()
plt.title(variable+'  '+name1+' :%1.3f   '%np.mean(mean_values1)+name2+': %1.3f'%np.mean(mean_values2))
plt.savefig(saving_folder+'Time_series_'+name1+'_'+name2+'_'+str_time+'_'+variable+'.png')
plt.show()
#%%


sample_file=address_dict1[str_time]
sample_dataset=Dataset(sample_file)
variable='ASOB_T'
levels=np.linspace(0.1,500,11).tolist()
variable='TOT_SNOW'
levels=np.linspace(0.00001,0.1,11).tolist()
variable='CLCT'
levels=np.linspace(0.001,1,21).tolist()
plt.figure(figsize=(18,18))
variable='TOT_PREC'
levels=np.linspace(0.01,3,21).tolist()


#mean1=np.
mean1=np.zeros_like(sample_dataset.variables[variable][:])
mean2=np.zeros_like(sample_dataset.variables[variable][:])
for date in address_dict1:
    if date=='2006100100':continue

    print (date)
    dataset1=Dataset(address_dict1[date])
    dataset2=Dataset(address_dict2[date])
    print (np.isnan(dataset1.variables[variable][:]/len(address_dict1)).sum())
    print (np.isnan(mean1).sum())
    mean1=mean1+dataset1.variables[variable][:]/len(address_dict1)
    mean2=mean2+dataset2.variables[variable][:]/len(address_dict1)
    jle.Count_nans(mean1)

dif=mean1-mean2
for key in dataset1.variables.keys():
    print(key,dataset1.variables[key].long_name)
#plt.title('TEXT')
plt.subplot(221)
#plt.text(-705,-40,'TEXT',fontsize=16)

data1=np.copy(mean1[0,])
data2=np.copy(mean2[0,])
jle.Quick_plot(data1,variable,levels=levels,new_fig=False,title=dataset1.variables[variable].long_name+' '+name1,metadata_dataset=sample_dataset)
plt.subplot(222)
jle.Quick_plot(data2,variable,levels=levels,new_fig=False,title=name2,metadata_dataset=sample_dataset)

jle.Count_nans(mean1)
jle.Count_nans(mean2)
print('diff calculated')
jle.Count_nans(dif)


if dif.max()>np.abs(dif.min()):
    dif_limit=dif.max()
else:
    dif_limit=-dif.min()
jle.Count_nans(dif)
print(dif.max())
print(dif.min())
dif_levels=np.linspace(-dif_limit,dif_limit,20).tolist()
plt.subplot(223)
print (dif_levels)
jle.Quick_plot(dif[0,],variable,metadata_dataset=sample_dataset,cmap=plt.cm.RdBu,levels=dif_levels,new_fig=False,title=name1+' - '+name2+'    Mean:%1.3f'%dif.mean())
print (dif_levels)

plt.subplot(224)
plt.title('Diff histogram')
data=dif[0,].flatten()
#data=data[data!=0]

plt.axvline(data.mean(),lw=3,ls='--',c='k')
plt.hist(data,bins=200,normed=True)
plt.yscale('log')
#plt.savefig(saving_folder+'Histogram_diff_'+variable+'.png')


plt.savefig(saving_folder+'Run_mean_comparison_'+name1+'_'+name2+'_'+'_'+variable+'_pcolormesh.png')








