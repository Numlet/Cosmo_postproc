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
import scipy

from datetime import date

from mpl_toolkits.basemap import Basemap

saving_folder='/users/jvergara/dry_days_plots/'
data_saving_folder='/store/c2sm/pr04/jvergara/DRY_DAYS_E-OBS/'


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






