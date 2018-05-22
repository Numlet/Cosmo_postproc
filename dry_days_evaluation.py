#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:20:44 2018

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
import iris
import scipy as sc

from datetime import date

from mpl_toolkits.basemap import Basemap

saving_folder='/users/jvergara/dry_days_plots/'
obs_folder='/store/c2sm/pr04/jvergara/DRY_DAYS_E-OBS/'
run12_folder='/store/c2sm/pr04/jvergara/DRY_DAYS_C/'
run2_folder='/store/c2sm/pr04/jvergara/DRY_DAYS_F_REGRIDED/'



present_years=['1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006',
       '2007', '2008']


run12_dict={}
run2_dict={}
obs_dict={}

files_12=glob.glob(run12_folder+'*')
files_2=glob.glob(run2_folder+'*')
files_obs=glob.glob(obs_folder+'*')
dry_dict={}

for name in files_12:
    run12_dict[name[-8:-4]]=np.load(name)
for name in files_2:
    run2_dict[name[-8:-4]]=np.load(name)
for name in files_obs:
    obs_dict[name[-8:-4]]=np.load(name)

run12_dry_days=np.zeros((len(present_years),run12_dict['2008'].shape[0],run12_dict['2008'].shape[1]))
run2_dry_days=np.zeros((len(present_years),run2_dict['2008'].shape[0],run2_dict['2008'].shape[1]))
obs_dry_days=np.zeros((len(present_years),obs_dict['2008'].shape[0],obs_dict['2008'].shape[1]))

for i in range(len(present_years)):
    run12_dry_days[i,:,:]=run12_dict[present_years[i]]    
    run2_dry_days[i,:,:]=run2_dict[present_years[i]]    
    obs_dry_days[i,:,:]=obs_dict[present_years[i]]    


mean_run12=run12_dry_days.mean(axis=0)
mean_run2=run2_dry_days.mean(axis=0)
mean_obs=obs_dry_days.mean(axis=0)


grid_lon_12=Dataset(jle.store+'sample_12km.nc').variables['lon']
grid_lat_12=Dataset(jle.store+'sample_12km.nc').variables['lat']

grid_lon_2=Dataset(jle.store+'sample_2km.nc').variables['lon']
grid_lat_2=Dataset(jle.store+'sample_2km.nc').variables['lat']
grid_lon_2=Dataset(jle.store+'sample_12km.nc').variables['lon']
grid_lat_2=Dataset(jle.store+'sample_12km.nc').variables['lat']




#model_lons,model_lats=stc.unrotated_grid(cube_DM10)
#times_range=np.argwhere((times_ceres >= tdi) & (times_ceres <=tde))
#times_range=np.logical_and([times_ceres >= tdi],[times_ceres <=tde])[0]

coord=np.zeros([len(grid_lon_2[0,].flatten()),2])
coord[:,0]=grid_lon_2[0,].flatten()
coord[:,1]=grid_lat_2[0,].flatten()
#X,Y=np.meshgrid(model_lons, model_lats)
#grid_z0 = sc.interpolate.griddata(coord, sat_SW, (X,Y), method='nearest')
grid_z1 = sc.interpolate.griddata(coord, mean_run2, (grid_lon_12,grid_lat_12), method='linear')

plt.imshow(grid_z1[:,:])
plt.colorbar()



run12=iris.load(jle.store+'sample_12km.nc')[0]
run2=iris.load(jle.store+'sample_2km.nc')[0]
run2=iris.load(jle.store+'sample_12km.nc')[0]

regrided_cube=run2.regrid(run12, iris.analysis.Linear())

plt.imshow(regrided_cube[0,:,:].data)
plt.imshow(run2[0,:,:].data)
plt.colorbar()


#rotated_air_temp = global_air_temp.regrid(rotated_psl, iris.analysis.Linear())



#%%








bm = Basemap()   # default: projection='cyl'


data_folder='/project/pr04/observations/meteoswiss/euro4m_APGD/'#swiss
data_folder='/project/pr04/observations/eobs_0.22deg_rot_v10.0/'



sample_dataset=Dataset(data_folder+'rr_0.22deg_rot_v10.0.nc')
jle.Quick_plot(sample_dataset,'rr',
               latitudes=sample_dataset.variables['Actual_latitude'][:],longitudes=sample_dataset.variables['Actual_longitude'][:])


present_years=['1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006',
       '2007', '2008']

for year in present_years:
    d0 = date(1950, 1, 1)
    d1 = date(int(year), 5, 1)
    d2 = date(int(year), 9, 30)
    start = (d1 - d0).days
    end = (d2 - d0).days
    #print(delta.days)
    
    year_daily_rain=sample_dataset.variables['rr'][start:end,:,:]#*sample_dataset.variables['rr'].scale_factor #add it or not??
    
    
    
    dry_days=np.array([year_daily_rain[:,:,:]<1])[0,:]
    max_number_of_dry_days=np.zeros((dry_days.shape[1],dry_days.shape[2]))
    for ilon in range(max_number_of_dry_days.shape[0]):
        print (ilon,max_number_of_dry_days.shape[0])
        for ilat in range(max_number_of_dry_days.shape[1]):
            land=bm.is_land(sample_dataset.variables['Actual_longitude'][ilon,ilat], sample_dataset.variables['Actual_latitude'][ilon,ilat])  #True
            if not land:
                max_number_of_dry_days[ilon,ilat]=np.nan
            else:
                array=dry_days[:,ilon,ilat]
                max_number_of_dry_days[ilon,ilat]=jle.Count_consecutive_ones(array)       
    np.save(data_saving_folder+'summer_dry_days_'+str(year),max_number_of_dry_days)
#np.save(data_saving_folder+'summer_dry_days_'+str(year),max_number_of_dry_days)
#%%
a=glob.glob(data_saving_folder+'*')

dry_dict={}
for name in a:
    dry_dict[name[-8:-4]]=np.load(name)


years=list(dry_dict.keys())
years=np.sort(years)

present_years=['1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006',
       '2007', '2008']

present_dry_days=np.zeros((len(present_years),dry_dict['2008'].shape[0],dry_dict['2008'].shape[1]))

for i in range(len(present_years)):
    present_dry_days[i,:,:]=dry_dict[present_years[i]]    




#%%
jle.Quick_plot(present_dry_days.mean(axis=0),'Number of dry days',
               latitudes=sample_dataset.variables['Actual_latitude'][:],longitudes=sample_dataset.variables['Actual_longitude'][:],
               levels=np.linspace(0,150,15).tolist())
plt.savefig(saving_folder+'E-OBS_dry_days.png')


#%%






