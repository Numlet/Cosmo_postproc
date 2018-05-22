#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 09:26:06 2018

@author: jvergara
"""
import pandas as pd
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
from scipy import stats


nikolinas='/project/pr04/banni/results_PGW/PGW/lm_f/'
davids='/project/pr04/davidle/results_clim/lm_f/'
nikolinas_continue='/project/pr04/banni/results_PGW/PGWcontinue/lm_f/'

folder_in_path='1h/'

starting_date='2079010100'
end_date='2088123123'

future_dates=jle.Hourly_time_list(starting_date,end_date)
list_future_files=jle.Locate_files(nikolinas,folder_in_path)
future_address_dict=jle.Create_address_dict(future_dates,list_future_files)

list_future_files_continue=jle.Locate_files(nikolinas_continue,folder_in_path)
future_address_dict_continue=jle.Create_address_dict(future_dates,list_future_files_continue)

future_address_dict = {**future_address_dict, **future_address_dict_continue}


starting_date='1999010100'
end_date='2008123123'

present_dates=jle.Hourly_time_list(starting_date,end_date)
list_present_files=jle.Locate_files(davids,folder_in_path)
present_address_dict=jle.Create_address_dict(present_dates,list_present_files)




Year_2003_present_dict=jle.Filter_dict(present_address_dict,years=2003)
sample_dataset=Dataset(Year_2003_present_dict['2003041201'])
#%%
#    monthly_precip[month-1].append(value)
#    t2=time.time()
#    print (t2-t1)
    #filtered_future_dict=jle.Filter_dict(future_address_dict,months=[3,8])
#filtered_future_dict=jle.Filter_dict(future_address_dict,months=[3,8])
#%%
variable='TOT_PREC'
lat=41+27/60
lon=-(1+42/60)
a,b=jle.Get_lat_lon_indices(lat,lon,sample_dataset)
monthly_precip=np.array([[] for _ in range(12)])

dataframes={}
#dataframes['2078']=pd.DataFrame(columns=jle.all_days_and_month_names)
dataframes['2079']=pd.DataFrame(columns=jle.all_days_and_month_names)
dataframes['2080']=pd.DataFrame(columns=jle.all_days_and_month_names)
dataframes['2081']=pd.DataFrame(columns=jle.all_days_and_month_names)
dataframes['2082']=pd.DataFrame(columns=jle.all_days_and_month_names)
dataframes['2083']=pd.DataFrame(columns=jle.all_days_and_month_names)
dataframes['2084']=pd.DataFrame(columns=jle.all_days_and_month_names)
dataframes['2085']=pd.DataFrame(columns=jle.all_days_and_month_names)
dataframes['2086']=pd.DataFrame(columns=jle.all_days_and_month_names)
dataframes['2087']=pd.DataFrame(columns=jle.all_days_and_month_names)
dataframes['2088']=pd.DataFrame(columns=jle.all_days_and_month_names)

#all_precip=[]
for year in dataframes.keys():
    single_year_dict=jle.Filter_dict(future_address_dict,years=int(year))
    for date in single_year_dict:
        print (date)
        print (single_year_dict[date])
    #    t1=time.time()
        value=Dataset(single_year_dict[date]).variables[variable][0,a,b]
        year,month,_,_=jle.Get_date_values(date)
#        dataframes[str(year)] = dataframes[str(year)].append({jle.all_days_and_month_names[month]: value}, ignore_index=True)
        df2 = pd.DataFrame([[value,value]], columns=[jle.all_days_and_month_names[0],jle.all_days_and_month_names[month]])
        dataframes[str(year)]=dataframes[str(year)].append(df2)
#%%
variable='TOT_PREC'
lat=41+27/60
lon=-(1+42/60)
a,b=jle.Get_lat_lon_indices(lat,lon,sample_dataset)
monthly_precip=np.array([[] for _ in range(12)])

#dataframes={}
dataframes['1999']=pd.DataFrame(columns=jle.all_days_and_month_names)
dataframes['2000']=pd.DataFrame(columns=jle.all_days_and_month_names)
dataframes['2001']=pd.DataFrame(columns=jle.all_days_and_month_names)
dataframes['2002']=pd.DataFrame(columns=jle.all_days_and_month_names)
dataframes['2003']=pd.DataFrame(columns=jle.all_days_and_month_names)
dataframes['2004']=pd.DataFrame(columns=jle.all_days_and_month_names)
dataframes['2005']=pd.DataFrame(columns=jle.all_days_and_month_names)
dataframes['2006']=pd.DataFrame(columns=jle.all_days_and_month_names)
dataframes['2007']=pd.DataFrame(columns=jle.all_days_and_month_names)
dataframes['2008']=pd.DataFrame(columns=jle.all_days_and_month_names)

#all_precip=[]
for year in dataframes.keys():
    single_year_dict=jle.Filter_dict(present_address_dict,years=int(year))
    for date in single_year_dict:
        print (date)
        print (single_year_dict[date])
    #    t1=time.time()
        value=Dataset(single_year_dict[date]).variables[variable][0,a,b]
        year,month,_,_=jle.Get_date_values(date)
#        dataframes[str(year)] = dataframes[str(year)].append({jle.all_days_and_month_names[month]: value}, ignore_index=True)
        df2 = pd.DataFrame([[value,value]], columns=[jle.all_days_and_month_names[0],jle.all_days_and_month_names[month]])
        dataframes[str(year)]=dataframes[str(year)].append(df2)
        
#        dataframes[str(year)] = dataframes[str(year)].append(onth_names[month]: value}, ignore_index=True)
#        all_precip.append(value)
#    monthly_precip[month-1].append(value)
#%%
        
pickle.dump(dataframes,open(jle.qp_path+'dataframes_calatayud','wb'))
#%%
dataframes={}
dataframes=pickle.load(open(jle.qp_path+'dataframes_calatayud','rb'))
plt.figure(figsize=(10,8))

future_mean=[]
present_mean=[]
for year in dataframes.keys():
    color='r'
    if int(year)>2050:color='b'
    
    monthly_mean=np.array([np.nanmean(dataframes[year][jle.month_names[i]]) for i in range(12)])
    print (year)
    print (monthly_mean)
    plt.plot(monthly_mean*24,label=year,c=color)
    plt.axhline(np.nanmean(monthly_mean*24),color=color)
    if color=='b':future_mean.append(np.nanmean(monthly_mean*24))
    else:present_mean.append(np.nanmean(monthly_mean*24))
    print(np.mean(monthly_mean))
t2, p2 = stats.ttest_ind(present_mean,future_mean)
plt.legend()
plt.savefig(jle.qp_path+'precip_calatayud')
#%%
for year in dataframes.keys():
    monthly_mean=np.array([np.nanmean(dataframes[year][jle.month_names[i]]) for i in range(12)])
for month in jle.month_names:
    print (month)
#plt.plot()


#%%
#days=int(len(df)/24)
#for i in range(days):
#    days
    




