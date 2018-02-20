#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:03:09 2018

@author: jvergara
"""

import Jesuslib_eth as jle
import numpy as np
import matplotlib.pyplot as plt
import glob
import iris
from netCDF4 import Dataset


import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='SW downward radiation at the surface', artist='Jesus Vergara-Temprado (2018)',
                comment='SW downward radiation at the surface')

#list_dates=jle.Hourly_time_list('2008061300','2008091523')

list_dates=jle.Hourly_time_list('2008061300','2008063023')

list_files=jle.Locate_files('/project/pr04/davidle/results_clim/lm_f/','1h/')

address_dict=jle.Create_address_dict(list_dates,list_files)
levels=np.linspace(-200,0,11).tolist()
variable='ATHB_S'



fig=plt.figure(figsize=(10, 8))
writer = FFMpegWriter(fps=6, metadata=metadata)
with writer.saving(fig,'/store/c2sm/pr04/jvergara/SW_plots/Letwyler_2017_SW.mp4', 200):
    for date in address_dict:
        print dates
        dataset=Dataset(address_dict[date])
        jle.Quick_plot(dataset,variable,levels=levels,
                       title='Surface downward SW '+date,
                       saving_name='/store/c2sm/pr04/jvergara/SW_plots/SW_'+date+'.png',
                       cmap=plt.cm.viridis_r,new_fig=False)
        writer.grab_frame()


#%%
        
#writer = FFMpegWriter(fps=5, metadata=metadata)
#with writer.saving(fig,'/store/c2sm/pr04/jvergara/SW_plots/Letwyler_2017_SW.mp4', 200):
#    for i in range(2):
#        x=np.arange(1,10)
#        plt.plot(x,x**i)
#        writer.grab_frame()