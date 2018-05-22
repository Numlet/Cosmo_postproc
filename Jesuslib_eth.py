#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 09:44:38 2018

@author: jvergara
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from collections import OrderedDict
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import colors, ticker, cm
from netCDF4 import Dataset
import netCDF4
import imp
import os

months_str=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
months_str_upper_case=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month_names=['January','February','March','April','May','June','July','August','September','October','November','December']
all_days_and_month_names=['All_days','January','February','March','April','May','June','July','August','September','October','November','December']
days_end_month=np.array([0,31,59,90,120,151,181,212,243,273,304,334,365])
days_end_month_leap=np.array([0,32,60,91,121,152,182,213,244,274,305,335,366])
hours_in_day=['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
months_number_str=['01','02','03','04','05','06','07','08','09','10','11','12']
days_in_month=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
days_in_month_leap=np.array([31,29,31,30,31,30,31,31,30,31,30,31])
qp_path='/users/jvergara/quick_plots/'
store='/store/c2sm/pr04/jvergara/'
scratch='/scratch/snx3000/jvergara/'
sastre='/scratch/snx3000/jvergara/cajon_de_sastre/'
home='/users/jvergara/'


def Create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def Create_project_folders(project_name,sc=0):
    data_folder=store+project_name
    if data_folder[-1]!='/':
        data_folder=data_folder+'/'
    plots_folder=home+project_name
    if plots_folder[-1]!='/':
        plots_folder=plots_folder+'/'
    
    if sc:
        scratch_folder=sastre+project_name
        if scratch_folder[-1]!='/':
            scratch_folder=scratch_folder+'/'

        
        Create_folder(data_folder)    
        Create_folder(plots_folder)    
        Create_folder(scratch_folder)    
        return plots_folder,data_folder,scratch_folder
    else:    
        Create_folder(data_folder)    
        Create_folder(plots_folder)    
        return plots_folder,data_folder


def Load_sample_dataset():
    sample_dataset=Dataset('/project/pr04/davidle/results_clim/lm_f/1h/2000/lffd2000051220.nc')
    return sample_dataset


def Load_sample_dataset_c():
    sample_dataset=Dataset('/project/pr04/davidle/results_clim/lm_c/1h/2000/lffd2000051220.nc')
    return sample_dataset

def Print_date(year,month,day,hour='No Hour'):
    date=str(year)
    if month>=10:
        date=date+str(month)
    else:
        date=date+'0'+str(month)
    if day>=10:
        date=date+str(day)
    else:
        date=date+'0'+str(day)
    if not hour=='No Hour':
        if hour>=10:
            date=date+str(hour)
        else:
            date=date+'0'+str(hour)
#    print (date)
    return date



def Daily_time_list(start_date='2000060100', end_date='2000060223'):
    if len(end_date)>8:end_date=end_date[:8]
    if len(start_date)>8:start_date=start_date[:8]

    s_year=int(start_date[:4])
    s_month=int(start_date[4:6])
    s_day=int(start_date[6:8])
    e_year=int(end_date[:4])
    e_month=int(end_date[4:6])
    e_day=int(end_date[6:8])
    list_of_dates=[]
    current_date=start_date
    list_of_dates.append(current_date)
    while current_date!=end_date:
        c_year=int(current_date[:4])
        c_month=int(current_date[4:6])
        c_day=int(current_date[6:8])
        leap_year=np.logical_not(c_year%4)
        if not leap_year:
            max_days_in_month=days_in_month
        else:
            max_days_in_month=days_in_month_leap
            
        c_day=c_day+1
        if c_day>max_days_in_month[c_month-1]:
            c_day=1
            c_month=c_month+1
            if c_month>12:
                c_month=1
                c_year=c_year+1
        current_date=Print_date(c_year,c_month,c_day)
        list_of_dates.append(current_date)
    return list_of_dates



def Hourly_time_list(start_date='2000060104', end_date='2000060204'):

    s_year=int(start_date[:4])
    s_month=int(start_date[4:6])
    s_day=int(start_date[6:8])
    s_hour=int(start_date[8:10])
    e_year=int(end_date[:4])
    e_month=int(end_date[4:6])
    e_day=int(end_date[6:8])
    e_hour=int(end_date[8:10])
    list_of_dates=[]
    current_date=start_date
    list_of_dates.append(current_date)
    while current_date!=end_date:
#        print (current_date)
        c_year=int(current_date[:4])
        c_month=int(current_date[4:6])
        c_day=int(current_date[6:8])
        c_hour=int(current_date[8:10])
        leap_year=np.logical_not(c_year%4)
        c_hour=c_hour+1
        if not leap_year:
            max_days_in_month=days_in_month
        else:
            max_days_in_month=days_in_month_leap
            
        if c_hour>23:
            c_hour=0
            c_day=c_day+1
            if c_day>max_days_in_month[c_month-1]:
                c_day=1
                c_month=c_month+1
                if c_month>12:
                    c_month=1
                    c_year=c_year+1
        current_date=Print_date(c_year,c_month,c_day,c_hour)
        list_of_dates.append(current_date)
    return list_of_dates



def Locate_files(path, resolution):
    list_of_files=[]
    first_layer=glob.glob(path+resolution+'*/')
    if len(first_layer)==0:
        print('Files in', path+resolution)
        nc_files=glob.glob(path+resolution+'*nc')
        for nc_file in nc_files:
            list_of_files.append(nc_file)
    else:
        print('Looking for files in:\n',first_layer)
        for folder_name in first_layer:
            nc_files=glob.glob(folder_name+'*nc')
            for nc_file in nc_files:
                list_of_files.append(nc_file)
    return list_of_files

def Create_address_dict(dates,addresses):
    address_dict=OrderedDict()
    for date in dates:
        for address in addresses:
            if date in address:
                address_dict[date]=address
                break
    return address_dict




def Get_date_values(date):
    year=int(date[:4])
    month=int(date[4:6])
    day=int(date[6:8])
    hour=int(date[8:10])
    return year,month,day,hour


def In_bounds(value,bounds):
    if isinstance(bounds,int):
        if value==bounds:
            return True
        else:
            return False
    elif bounds[0]<=bounds[1]:
        if value>=bounds[0] and value<=bounds[1]:
            return True
        else:
            return False
    elif bounds[0]>bounds[1]:
        if value>=bounds[0] or value<=bounds[1]:
            return True
        else:
            return False
        

def Filter_dict(address_dict,years=[0,99999],months=[0,999],days=[0,999],hours=[0,999]):
    keys_subset=[]
    for key in address_dict.keys():
        year, month, day, hour=Get_date_values(key)
        if all([In_bounds(year,years),In_bounds(month, months),In_bounds(day,days),In_bounds(hour, hours)]):
            keys_subset.append(key)
    filtered_dict=OrderedDict()
    for key in keys_subset:
        filtered_dict[key]=address_dict[key]
    return filtered_dict


def Filter_dates(list_dates,years=[0,99999],months=[0,999],days=[0,999],hours=[0,999]):
    '''
    NOT TESTED YET!
    '''
    dates_subset=[]
    for key in list_dates:
        year, month, day, hour=Get_date_values(key)
        if all([In_bounds(year,years),In_bounds(month, months),In_bounds(day,days),In_bounds(hour, hours)]):
            dates_subset.append(key)
    return dates_subset


def Find_second_nearest_vector_index(array, value):
    n = np.array([abs(i-value) for i in array])
    n[n.argmin()]=n.max()
    nindex=np.apply_along_axis(np.argmin,0,n)
    return nindex
def Find_nearest_vector_index(array, value):
    n = np.array([abs(i-value) for i in array])
    nindex=np.apply_along_axis(np.argmin,0,n)
    return nindex


def Get_lat_lon_indices(lat,lon,dataset):
    latitudes = dataset.variables['lat'][:]
    longitudes = dataset.variables['lon'][:]
    distance=(np.abs(latitudes-lat)**2+np.abs(longitudes-lon))    
    arguments=np.argwhere(distance == np.min(distance))
    a=arguments[0][0]
    b=arguments[0][1]
    return a,b    

def Count_consecutive_ones(array):
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

#%%
    
def Quick_plot(dataset,variable,levels=0,title=0,cmap=plt.cm.viridis,show=0,saving_name=0,new_fig=1,fig=0,
               height_level=0,cb_label='-',metadata_dataset=0,return_m=0,bluemarble=0,shadedrelief=0,
               npar=10,nmer=20,draw_mer_par=0,latitudes=0,longitudes=0):
    if isinstance(latitudes,int):
        try:
            latitudes = dataset.variables['lat'][:]
            longitudes = dataset.variables['lon'][:]
        except:
            latitudes = metadata_dataset.variables['lat'][:]
            longitudes = metadata_dataset.variables['lon'][:]

    if new_fig:
        fig=plt.figure(figsize=(10, 8))
    lon_0 = longitudes.mean()
    lat_0 = latitudes.mean()
    
#    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    m = Basemap(llcrnrlon=longitudes.min(),llcrnrlat=latitudes.min(),urcrnrlon=longitudes.max(),urcrnrlat=latitudes.max(),\
            resolution='l',projection='merc',\
            lat_0=lat_0,lon_0=lon_0)
    m.drawcoastlines()

    m.drawcountries()
    if draw_mer_par:
        m.drawparallels(np.linspace(10,90,npar),labels=[1,0,0,1])
# draw meridians
#    m.drawmeridians(np.arange(-180,180,nmer),labels=[1,1,0,1])
        m.drawmeridians(np.linspace(-180,180,nmer),labels=[1,0,0,1])
    if bluemarble:
        m.bluemarble()
    if shadedrelief:
        m.shadedrelief()
    if isinstance(levels,int):
        try:
            try:
                levels=np.linspace(dataset.variables[variable][0,].min(),dataset.variables[variable][0,].max(),10).tolist()
            except:
                levels=np.linspace(dataset.variables[variable][:].min(),dataset.variables[variable][0,].max(),10).tolist()
        except:
            levels=np.linspace(dataset.min(),dataset.max(),10).tolist()
    if isinstance(dataset,netCDF4._netCDF4.Dataset):
        if dataset.variables[variable]==3:
            data=dataset.variables[variable][0,]
        else:
            data=dataset.variables[variable][:]
            
        if data.ndim==3:
            data=data[height_level,]
    else:
        data=np.copy(dataset)
#    cs=m.contourf(longitudes,latitudes,data,levels,latlon=True,norm= colors.BoundaryNorm(levels, 256),cmap=cmap,interpolation='nearest')
    data[data>levels[-1]]=np.nan
    data[data<levels[0]]=np.nan
    
    cs=m.pcolormesh(longitudes,latitudes,data,latlon=True,norm= colors.BoundaryNorm(boundaries=levels, ncolors=256),cmap=cmap)
    cb = m.colorbar(cs,format='%.2e',ticks=levels)
    try:
        cb.set_label(dataset.variables[variable].units)
    except:
        cb.set_label(cb_label)
    if not isinstance(title, str):
        try:
            title=dataset.title+' - '+dataset.experiment_id
        except:
            title=variable
    plt.title(title)
    if show:
        plt.show()
    if isinstance(saving_name, str):
        plt.savefig(saving_name)
    if return_m:
        return m

#dataset=Dataset(address_dict['2000122110'])
#variable='ASWD_S'
#
#Quick_plot(dataset,variable)
def Count_nans(array):
    print('Nans:',np.isnan(array).sum())



#Print_date(2000,3,12,4)
#path='/project/pr04/davidle/results_clim/lm_f/'
#resolution='1h/'
#dates=Hourly_time_list('2000112500','2000122523')
#
#addresses=Locate_files(path, resolution)

#address_dict=Create_address_dict(dates, addresses)
#filtered_dict=Filter_dict(address_dict,hours=[22,8])


def Mask_coarse_map(array,n=45):
    array_masked=np.copy(array)
    array_masked[:n,:]=np.nan
    array_masked[-n:,:]=np.nan
    array_masked[:,-n:]=np.nan
    array_masked[:,:n]=np.nan
    return array_masked






