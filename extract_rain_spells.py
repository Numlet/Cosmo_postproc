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
saving_folder='/users/jvergara/dry_days_plots/'
data_saving_folder='/store/c2sm/pr04/jvergara/DRY_DAYS_C/'
data_saving_folder='/store/c2sm/pr04/jvergara/DRY_DAYS_C/'

project_name='RAIN_DURATION'
plots_folder,data_folder=jle.Create_project_folders(project_name)
#%%
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


folder_in_path='1h/'

starting_date='2079010100'
end_date='2088123123'
#%%
future_dates=jle.Hourly_time_list(starting_date,end_date)
list_future_files=jle.Locate_files(nikolinas,folder_in_path)
future_address_dict=jle.Create_address_dict(future_dates,list_future_files)

list_future_files_continue=jle.Locate_files(nikolinas_continue,folder_in_path)
future_address_dict_continue=jle.Create_address_dict(future_dates,list_future_files_continue)

future_address_dict = {**future_address_dict, **future_address_dict_continue}


#davids='/project/pr04/davidle/results_clim/lm_f/'

folder_in_path='1h/'
#%%
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


all_address_dict={**future_address_dict, **present_address_dict}

summer_address_dict=jle.Filter_dict(all_address_dict,months=[5,9])
years=np.sort(list(set([date[:4] for date in summer_address_dict.keys()]))).tolist()

#%%

f=jle.Load_sample_dataset_c()


domain_name='IB'
lonbounds = [ -10 , 3] # degrees east ? 
latbounds = [ 44 , 36 ]



lons = f.variables['lon'][:]
lats = f.variables['lat'][:]
# longitude lower and upper index
lonli = np.argmin( np.abs( lons - lonbounds[0] ) )
lonui = np.argmin( np.abs( lons - lonbounds[1] ) )  
# latitude lower and upper index
latli = np.argmin( np.abs( lats - latbounds[0] ) )
latui = np.argmin( np.abs( lats - latbounds[1] ) ) 


lats = f.variables['lat'][:] 
lons = f.variables['lon'][:]

#valid=[lats<latbounds[0] *lats>latbounds[1]]
up=lats<latbounds[0]
down=lats>latbounds[1]
right=lons>lonbounds[0]
left=lons<lonbounds[1]

valid_domain=up*down*right*left

sub_lat=lats[valid_domain]
sub_lon=lons[valid_domain]

masked_lat=np.ones_like(lats)*np.nan
masked_lon=np.ones_like(lons)*np.nan

masked_lat[valid_domain]=lats[valid_domain]
masked_lon[valid_domain]=lons[valid_domain]


#plt.imshow(masked_lon)




#grid_lons,grid_lats=np.meshgrid(sub_lon,sub_lat)



#year='2008'
#year_address_dict=jle.Filter_dict(all_address_dict,years=int(year))

#keys=[string for string in summer_address_dict if year in string[:4]]

rain_threshold=0.01


#%%


duration_files=glob.glob(data_folder+'*duration*')

mean_intensity_files=glob.glob(data_folder+'*mean_intensity*')

std_files=glob.glob(data_folder+'*std*')

max_events_files=glob.glob(data_folder+'*max*')

headers=['duration','mean_intensity','max_intensity','std','month','lon','lat']

threshold='with_threshold'
data=np.empty([100000000,7])
idata=0
for month in jle.months_number_str:
    print (month)
    duration=np.load('/store/c2sm/pr04/jvergara/RAIN_DURATION/2008'+month+'duration_'+threshold+'.npy')
    mean_intensity=np.load('/store/c2sm/pr04/jvergara/RAIN_DURATION/2008'+month+'mean_intensity_events_'+threshold+'.npy')
    max_events=np.load('/store/c2sm/pr04/jvergara/RAIN_DURATION/2008'+month+'max_events_'+threshold+'.npy')
    std=np.load('/store/c2sm/pr04/jvergara/RAIN_DURATION/2008'+month+'std_events_'+threshold+'.npy')
    for ilon in range(lons.shape[0]):
#        print (ilon)
        for ilat in range(lats.shape[1]):
            duration_column=duration[:,ilat,ilon]
            mean_intensity_column=mean_intensity[:,ilat,ilon]
            std_column=std[:,ilat,ilon]
            max_events_column=max_events[:,ilat,ilon]
            
            events=[duration_column!=0]
            duration_column=duration_column[events]
            mean_intensity_column=mean_intensity_column[events]
            std_column=std_column[events]
            max_events_column=max_events_column[events]
            
            for i in range(len(duration_column)):
                
                array=[duration_column[i],mean_intensity_column[i],max_events_column[i],std_column[i],int(month),lons[ilon,ilat],lats[ilon,ilat]]
                data[idata,:]=array
                idata=idata+1
                if idata==len(data[:,0]):
                    raise NameError('Data is not large enough')
            
np.save(data_saving_folder+'events_data',data)

#%%
data=np.load(data_saving_folder+'events_data.npy')

data[:,1]=np.log(data[:,1])
data[:,2]=np.log(data[:,2])
data_subset=data[:np.sum(data[:,0]!=0),:]
from sklearn.preprocessing import normalize
import pandas as pd
#data_for_classifier=data_subset[:,:4]
random_subset=data_subset[np.random.choice(data_subset.shape[0], 20000, replace=False)]


norm = normalize(random_subset[:,:4],axis=0)
#norm = random_subset[:,:4]
df = pd.DataFrame(norm,columns=headers[:4])
plt.matshow(df.corr(),cmap=plt.cm.RdBu)
plt.xticks(range(len(list(df))),list(df),rotation=45)
plt.yticks(range(len(list(df))),list(df),rotation=0)
plt.colorbar()
plt.savefig(plots_folder+'correlation_matrix')
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(norm)
#spectral = SpectralClustering(n_clusters=3).fit(norm)
#%%
#plt.figure(figsize=(20,20))
values=np.arange(len(norm[0,:]))
import itertools
for ix,iy in itertools.combinations(values, 2):
    print (ix,iy)
    plt.figure()
    plt.scatter(norm[:1000000,ix],norm[:1000000,iy],c=kmeans.labels_,alpha=0.1)
#plt.colorbar(label=headers[ic])
    plt.colorbar(label='clasification')
    plt.xlabel('normalized '+ headers[ix])
    plt.ylabel('normalized '+ headers[iy])
    plt.savefig(plots_folder+'non_normalized_' +headers[ix]+ headers[iy])
#    plt.show()
#%%
   
pd.scatter_matrix(df, figsize=(6, 6))
plt.savefig(plots_folder+'scatter_matrix')
plt.show()
#ix=0
#iy=1
#ic=-1
#plt.scatter(norm[:1000000,ix],norm[:1000000,iy],c=kmeans.labels_,alpha=0.1)
##plt.colorbar(label=headers[ic])
#plt.colorbar(label='clasification')
#plt.xlabel('normalized '+ headers[ix])
#plt.ylabel('normalized '+ headers[iy])
##plt.scatter(data_subset[:10000,0],data_subset[:10000,1])

#plt.plot(data[:,0],data[:,1])
#plt.show()
#%%
#
#def normalize(x):
#    normalized = (x-np.min(x))/(np.max(x)-np.min(x))
#    return normalized
#
#norm_duration=normalize(data_subset[:,0])
#norm_intensity=normalize(data_subset[:,1])
#heatmap, xedges, yedges = np.histogram2d(norm_duration, norm_intensity, bins=(64,64))
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
##norm2 = normalize(data_subset[:,0]).ravel()
## Plot heatmap
#plt.clf()
#plt.title('Pythonspot.com heatmap example')
#plt.ylabel('y')
#plt.xlabel('x')
#plt.imshow(heatmap, extent=extent)
#plt.show()


#%%
'''
data=np.load(data_saving_folder+'events_data.npy')

plt.hist(data[:,0],bins=100)
plt.hist(data[:,3],bins=100)

heatmap, xedges, yedges = np.histogram2d(data[:,0], data[:,3], bins=5000)
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#%%
plt.hist(data[:,0],bins=np.linspace(0,30,30))
#%%
plt.hist(data[:,1],bins=np.linspace(0,30,30))
#%%
plt.clf()
plt.imshow(heatmap.T, origin='lower')
plt.xlim(0,0.1)
plt.ylim(0,0.1)
plt.show()
#%%

rain_more_than_0=rain_in_month.flatten()
rain_more_than_0=rain_more_than_0[rain_more_than_0>0]
plt.hist(rain_more_than_0,bins=1000,cumulative=True)
plt.xlim(0,0.2)
rain_threshold=0.01
print (rain_more_than_0[rain_more_than_0>rain_threshold].sum(),rain_more_than_0[rain_more_than_0<rain_threshold].sum(),rain_more_than_0[rain_more_than_0>rain_threshold].sum()/rain_more_than_0[rain_more_than_0<rain_threshold].sum())
print ((rain_more_than_0>rain_threshold).sum(),(rain_more_than_0<rain_threshold).sum(),(rain_more_than_0>rain_threshold).sum()/(rain_more_than_0<rain_threshold).sum())

#plt.xlim(0,0.05)
#plt.xscale('log')
#%%
'''







