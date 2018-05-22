#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:03:09 2018

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
#from pympler import tracker
#tr = tracker.SummaryTracker()
#tr.print_diff()   



starting_date='2008060100'
end_date='2008083123'

saving_folder='/store/c2sm/pr04/jvergara/SW_plots/'

list_dates=jle.Hourly_time_list(starting_date,end_date)
list_days=jle.Daily_time_list(starting_date,end_date)
#list_dates=jle.Hourly_time_list('2008060112','2008060113')
list_files=jle.Locate_files('/project/pr04/davidle/results_clim/lm_f/','1h/')
address_dict=jle.Create_address_dict(list_dates,list_files)
#%%

try:
    with open(saving_folder+"dict_max.txt", "rb") as myFile:
        dict_max = pickle.load(myFile)
    with open(saving_folder+"dict_min.txt", "rb") as myFile:
        dict_min = pickle.load(myFile)
except:
    print ('There are not max and min dicts!')


variable='ASWD_S'
variable='ATHB_T'
variable='TOT_PREC'
#variable='ASOB_S'


#levels=np.linspace(0,900,11).tolist()
try:
    levels=np.linspace(dict_min[variable],dict_max[variable],11).tolist()
except:
    print("Levels could not be define automatically")
#day='4'
#levels=
times=[]


levels=np.linspace(0.001,30,11).tolist()

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Leutwyler et al. 2017 simulation', artist='Jesus Vergara-Temprado (2018)',
                comment='Leutwyler et al. 2017 simulation')
writer = FFMpegWriter(fps=6, metadata=metadata)
from pympler import tracker
#tr.print_diff()
i=0

all_videos=[]

for day in list_days:
    print('DAY ------------------------')
    print(day)
    print('---------------------------')
    fig=plt.figure(figsize=(10, 8))
    video_file=saving_folder+'Letwyler_2017_'+variable+'_'+day+'.mp4'
    all_videos.append(video_file)
    with writer.saving(fig,video_file, 200):
#        print (address_dict.keys())
        steps=len(address_dict.keys())
        print (steps)
        dates=[day+hour_str for hour_str in jle.hours_in_day]
        for date in dates:
            t1=time.time()
             
            print (date)
            print (i,steps)
            dataset=Dataset(address_dict[date])
            jle.Quick_plot(dataset,variable,levels=levels,
                           title=date+' '+dataset.variables[variable].long_name,
    #                       saving_name='/store/c2sm/pr04/jvergara/SW_plots/'+variable+'_'+date+c'.png',
                           cmap=plt.cm.viridis,new_fig=False)
            i=i+1
            writer.grab_frame()
            fig.clear()
            t2=time.time()
            print ('time:',t2-t1)
            times.append(t2-t1)
    #        print('\n\n Difference:')
    #        tr.print_diff()   
    #        all_objects = muppy.get_objects()
    #        print('\n\n Summary:')
    #        sum1 = summary.summarize(all_objects)
    #        summary.print_(sum1) 
    #        del dataset
        
    plt.figure()
    plt.plot(times)
    plt.savefig(saving_folder+'time_profiler_'+day+'.png')
    plt.close()
    plt.close()
lines=["file '%s'"%f for f in all_videos]

full_video_name=saving_folder+'Variable_'+variable+'_from_'+starting_date+'_to_'+end_date
file = open(full_video_name+'.txt',"w") 
for l in lines: 
    file.write(l+'\n') 
file.close() 
command='ffmpeg -f concat -safe 0 -i %s -c copy %s.mp4'%(full_video_name+'.txt',full_video_name+'.mp4')
t1=time.time()
concatenate=os.system(command)
t2=time.time()
print(concatenate)
print('time to concatenate:')
print(t2-t1)
#%%

#writer = FFMpegWriter(fps=5, metadata=metadata)
#with writer.saving(fig,'/store/c2sm/pr04/jvergara/SW_plots/Letwyler_2017_SW.mp4', 200):
#    for i in range(2):
#        x=np.arange(1,10)
#        plt.plot(x,x**i)
#        writer.grab_frame()

#from pympler import muppy
#all_objects = muppy.get_objects()
#from pympler import summary
#sum1 = summary.summarize(all_objects)
#summary.print_(sum1)
#print (len(all_objects)/1024)
