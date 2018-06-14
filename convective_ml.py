#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:19:10 2018

@author: jvergara
"""

import sys
sys.path.append('/users/jvergara/python_code')
#import matplotlib
#matplotlib.use('Agg')
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



project_name='CONVECTIVE_ML'
plots_folder,data_folder=jle.Create_project_folders(project_name)

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

model_output_folder='/scratch/snx3000/jvergara/cosmo-pompa_convective_precip/cosmo/test/climate/crClim2km_DVL/2_lm_c/output/'
model_output_folder='/store/c2sm/pr04/jvergara/CONVECTIVE_FRACTION_SIMULATION/output/'
folder_in_path='1h/'



starting_date='2079010100'
end_date='2088123123'


starting_date='1999010100'
end_date='2008123123'

present_dates=jle.Hourly_time_list(starting_date,end_date)
list_present_files=jle.Locate_files(model_output_folder,folder_in_path)

present_address_dict=jle.Create_address_dict(present_dates,list_present_files)

sample_dataset=Dataset(present_address_dict[next(iter(present_address_dict))])
sample_dataset=Dataset(present_address_dict[list(present_address_dict.keys())[14]])



f=jle.Load_sample_dataset_c()




lons = f.variables['lon'][:]
lats = f.variables['lat'][:]


print(sample_dataset.variables.keys())
 
print(sample_dataset.variables['PRR_GSP'][:].mean()*3600)
print(sample_dataset.variables['RAIN_GSP'][:].mean())
print(sample_dataset.variables['PRR_CON'][:].mean()*3600)
print(sample_dataset.variables['RAIN_CON'][:].mean())
print(sample_dataset.variables['TOT_PR'][:].mean()*3600)
print(sample_dataset.variables['TOT_PREC'][:].mean())

total=sample_dataset.variables['TOT_PREC'][:].flatten()
gsp=sample_dataset.variables['RAIN_GSP'][:].flatten()

print(sample_dataset.variables['PRR_GSP'][:].mean()+sample_dataset.variables['PRR_CON'][:].mean())
print(sample_dataset.variables['TOT_PR'][:].mean())


print(sample_dataset.variables['RAIN_GSP'][:].mean()+sample_dataset.variables['RAIN_CON'][:].mean())

print(sample_dataset.variables['TOT_PR'][:].mean()*3600)
print(sample_dataset.variables['TOT_PREC'][:].mean())

all(total==gsp)


year='2006'
rain_threshold=0.1
#%%

U=sample_dataset.variables['U_10M'][0,]
V=sample_dataset.variables['V_10M'][0,]

div=(U[2:,1:-1]-U[:-2,1:-1])+(V[1:-1,2:]-V[1:-1,:-2])
vor=(V[2:,1:-1]-V[:-2,1:-1])+(U[1:-1,2:]-U[1:-1,:-2])

jle.Quick_plot(sample_dataset,'U_10M',cmap=plt.cm.RdBu)
jle.Quick_plot(sample_dataset,'V_10M',cmap=plt.cm.RdBu)
lons_cut=lons[1:-1,1:-1]
lats_cut=lats[1:-1,1:-1]
jle.Quick_plot(div,'divergence',longitudes=lons_cut,latitudes=lats_cut,cmap=plt.cm.RdBu)
jle.Quick_plot(vor,'vorticity',longitudes=lons_cut,latitudes=lats_cut,cmap=plt.cm.RdBu,levels=np.linspace(vor.min(),-vor.min(),10))
jle.Quick_plot(sample_dataset,'TOT_PREC')

x=np.arange(0,4)



#%%
#fig, ax = plt.subplots()

#ax.streamplot(U[0,], V[0,], lons, lats, cmap='gist_earth')


#%%

for month in jle.months_number_str:
    print(month)
    month_address_dict=jle.Filter_dict(present_address_dict,years=int(year),months=int(month))
    keys=[key for key in month_address_dict]
    if len(month_address_dict)==0:
        continue
    rain_in_month=np.zeros((len(keys),lats.shape[0],lats.shape[1]))
    temp_in_month=np.zeros((len(keys),lats.shape[0],lats.shape[1]))
    rh_in_month=np.zeros((len(keys),lats.shape[0],lats.shape[1]))
    conv_rain_in_month=np.zeros((len(keys),lats.shape[0],lats.shape[1]))
    grsc_rain_in_month=np.zeros((len(keys),lats.shape[0],lats.shape[1]))
    div_in_month=np.zeros((len(keys),lats.shape[0],lats.shape[1]))
    vor_in_month=np.zeros((len(keys),lats.shape[0],lats.shape[1]))
    for i in range(len(keys)):
        print(keys[i])        
        dataset=Dataset(month_address_dict[keys[i]])
#        rain_in_month[i,:,:]=dataset.variables['TOT_PREC'][0,]+dataset.variables['TOT_SNOW'][0,]
#        conv_rain_in_month[i,:,:]=dataset.variables['RAIN_CON'][0,]+dataset.variables['SNOW_CON'][0,]
#        grsc_rain_in_month[i,:,:]=dataset.variables['RAIN_GSP'][0,]+dataset.variables['SNOW_GSP'][0,]

        temp_in_month[i,:,:]=dataset.variables['T_2M'][0,]
        rh_in_month[i,:,:]=dataset.variables['RELHUM_2M'][0,]
        conv_rain_in_month[i,:,:]=dataset.variables['RAIN_CON'][0,]+dataset.variables['SNOW_CON'][0,]
        grsc_rain_in_month[i,:,:]=dataset.variables['RAIN_GSP'][0,]+dataset.variables['SNOW_GSP'][0,]
        rain_in_month[i,:,:]=conv_rain_in_month[i,:,:]+grsc_rain_in_month[i,:,:]
        U=sample_dataset.variables['U_10M'][0,]
        V=sample_dataset.variables['V_10M'][0,]

        div=(U[2:,1:-1]-U[:-2,1:-1])+(V[1:-1,2:]-V[1:-1,:-2])
        vor=(V[2:,1:-1]-V[:-2,1:-1])+(U[1:-1,2:]-U[1:-1,:-2])
        div_in_month[i,1:-1,1:-1]=div
        vor_in_month[i,1:-1,1:-1]=vor
#        rain_in_month[i,:,:]
    rain_in_month[rain_in_month<rain_threshold]=0#Hace falta esto? Tal vez sea mejor no applicar ningun threshold
    conv_rain_in_month[rain_in_month<rain_threshold]=0#Hace falta esto? Tal vez sea mejor no applicar ningun threshold
    grsc_rain_in_month[rain_in_month<rain_threshold]=0#Hace falta esto? Tal vez sea mejor no applicar ningun threshold
#    rain_or_not=np.zeros_like(rain_in_month[0,])
    duration_events=np.zeros_like(rain_in_month)
    temp_events=np.zeros_like(rain_in_month)
    rh_events=np.zeros_like(rain_in_month)
    mean_intensity_events=np.zeros_like(rain_in_month)
    mean_conv_intensity_events=np.zeros_like(rain_in_month)
    mean_grsc_intensity_events=np.zeros_like(rain_in_month)
    std_events=np.zeros_like(rain_in_month)
    max_events=np.zeros_like(rain_in_month)
    conv_filter_events=np.zeros_like(rain_in_month)
    div_events=np.zeros_like(rain_in_month)
    vor_events=np.zeros_like(rain_in_month)
    for ilon in range(len(rain_in_month[0,:,0])):
        print (ilon)
        for ilat in range(len(rain_in_month[0,0,:])):
            rh_in_gridbox=rh_in_month[:,ilon,ilat]
            temp_in_gridbox=temp_in_month[:,ilon,ilat]
            rain_in_gridbox=rain_in_month[:,ilon,ilat]
            conv_rain_in_gridbox=conv_rain_in_month[:,ilon,ilat]
            grsc_rain_in_gridbox=grsc_rain_in_month[:,ilon,ilat]
            div_in_gridbox=div_in_month[:,ilon,ilat]
            vor_in_gridbox=vor_in_month[:,ilon,ilat]
            ihour=0
            while ihour < len(rain_in_gridbox):
                if rain_in_gridbox[ihour]==0:
                    ihour=ihour+1
                else:
                    raining=1
                    starting_hour=ihour
                    spell_conv_intensity=[]
                    spell_grsc_intensity=[]
                    spell_intensity=[]
                    spell_temp=[]
                    spell_rh=[]
                    spell_conv_filter=[]
                    spell_div=[]
                    spell_vor=[]
                    
                    while raining:
                        if ihour>=len(rain_in_gridbox):
                            break
                        if rain_in_gridbox[ihour]==0:
                            raining=0
                        else:
                            if ihour==0:
                                intensity_b=0
                            else:
                                intensity_b=[rain_in_gridbox[ihour-1]]
                            if ihour>=len(rain_in_gridbox)-1:
                                intensity_f=0
                            else:
                                intensity_f=[rain_in_gridbox[ihour+1]]
                            intensity=rain_in_gridbox[ihour]
#                            temp=temp_in_gridbox[ihour]
#                            rh=rh_in_gridbox[ihour]
                            
                            filter_value=-(1./4)*(intensity_b-2*intensity+intensity_f)
                            spell_temp.append(temp_in_gridbox[ihour])
                            spell_rh.append(rh_in_gridbox[ihour])
                            spell_intensity.append(rain_in_gridbox[ihour])
                            spell_div.append(div_in_gridbox[ihour])
                            spell_vor.append(vor_in_gridbox[ihour])
                            spell_conv_intensity.append(conv_rain_in_gridbox[ihour])
                            spell_grsc_intensity.append(grsc_rain_in_gridbox[ihour])
                            spell_conv_filter.append(filter_value)
                            ihour=ihour+1
                            
                    duration=len(spell_intensity)
                    std=np.std(spell_intensity)
                    maximum=np.max(spell_intensity)
                    mean=np.mean(spell_intensity)
                    mean_temp=np.mean(spell_temp)
                    mean_rh=np.mean(spell_rh)
                    mean_div=np.mean(spell_div)
                    mean_vor=np.mean(spell_vor)
                    mean_conv=np.mean(spell_conv_intensity)
                    mean_grsc=np.mean(spell_grsc_intensity)
                    mean_conv_filter=np.sum(np.array(spell_conv_filter)**2)/len(spell_conv_filter)
                    
                    
                    temp_events[starting_hour,ilon,ilat]=mean_temp
                    rh_events[starting_hour,ilon,ilat]=mean_rh
                    duration_events[starting_hour,ilon,ilat]=duration
                    mean_conv_intensity_events[starting_hour,ilon,ilat]=mean_conv
                    mean_grsc_intensity_events[starting_hour,ilon,ilat]=mean_grsc
                    mean_intensity_events[starting_hour,ilon,ilat]=mean
                    vor_events[starting_hour,ilon,ilat]=mean_vor
                    div_events[starting_hour,ilon,ilat]=mean_div
                    std_events[starting_hour,ilon,ilat]=std
                    max_events[starting_hour,ilon,ilat]=maximum
                    conv_filter_events[starting_hour,ilon,ilat]=mean_conv_filter
    np.save(data_folder+year+month+'rh_with_threshold',rh_events)
    np.save(data_folder+year+month+'temp_with_threshold',temp_events)
    np.save(data_folder+year+month+'duration_with_threshold',duration_events)
    np.save(data_folder+year+month+'mean_intensity_events_with_threshold',mean_intensity_events)
    np.save(data_folder+year+month+'div_events_with_threshold',div_events)
    np.save(data_folder+year+month+'vor_events_with_threshold',vor_events)
    np.save(data_folder+year+month+'mean_grsc_intensity_events_with_threshold',mean_grsc_intensity_events)
    np.save(data_folder+year+month+'mean_conv_intensity_events_with_threshold',mean_conv_intensity_events)
    np.save(data_folder+year+month+'std_events_with_threshold',std_events)
    np.save(data_folder+year+month+'max_events_with_threshold',max_events)
    np.save(data_folder+year+month+'conv_filter_events_with_threshold',conv_filter_events)


#%%


mean_intensity_files=glob.glob(data_folder+'*mean_intensity*')

headers=['duration','mean_intensity','max_intensity','std','convective_filter','div','vor','temp','rh','convective_or_not','convective_fraction','month','lon','lat']

threshold='with_threshold'
data=np.empty([100000000,len(headers)])
idata=0
for month in ['07']:
    print (month)
    duration=np.load(data_folder+'2006'+month+'duration_'+threshold+'.npy')
    temp=np.load(data_folder+'2006'+month+'temp_'+threshold+'.npy')
    rh=np.load(data_folder+'2006'+month+'rh_'+threshold+'.npy')
    mean_intensity=np.load(data_folder+'2006'+month+'mean_intensity_events_'+threshold+'.npy')
    mean_grsc_intensity=np.load(data_folder+'2006'+month+'mean_grsc_intensity_events_'+threshold+'.npy')
    mean_conv_intensity=np.load(data_folder+'2006'+month+'mean_conv_intensity_events_'+threshold+'.npy')
    conv_filter=np.load(data_folder+'2006'+month+'conv_filter_events_'+threshold+'.npy')
    mean_intensity=np.load(data_folder+'2006'+month+'mean_intensity_events_'+threshold+'.npy')
    mean_div=np.load(data_folder+'2006'+month+'div_events_'+threshold+'.npy')
    mean_vor=np.load(data_folder+'2006'+month+'vor_events_'+threshold+'.npy')
    max_events=np.load(data_folder+'2006'+month+'max_events_'+threshold+'.npy')
    std=np.load(data_folder+'2006'+month+'std_events_'+threshold+'.npy')
    convective_fraction=mean_conv_intensity/(mean_conv_intensity+mean_grsc_intensity)
    convective_or_not=np.zeros_like(convective_fraction)
    convective_or_not[convective_fraction>0.5]=1
#    convective_or_not[convective_fraction<0.01]=2
#    convective_or_not[convective_fraction<0.01 &&]=0
    for ilon in range(lons.shape[0]):
        print (ilon)
        for ilat in range(lats.shape[1]):
            duration_column=duration[:,ilat,ilon]
            rh_column=rh[:,ilat,ilon]
            temp_column=temp[:,ilat,ilon]
            mean_div_column=mean_div[:,ilat,ilon]
            mean_vor_column=mean_vor[:,ilat,ilon]
            mean_intensity_column=mean_intensity[:,ilat,ilon]
            std_column=std[:,ilat,ilon]
            max_events_column=max_events[:,ilat,ilon]
            convective_fraction_column=convective_fraction[:,ilat,ilon]
            conv_filter_column=conv_filter[:,ilat,ilon]
            convective_or_not_column=convective_or_not[:,ilat,ilon]
            
            events=[duration_column!=0]
            duration_column=duration_column[events]
            rh_column=rh_column[events]
            temp_column=temp_column[events]
            mean_intensity_column=mean_intensity_column[events]
            mean_div_column=mean_div_column[events]
            mean_vor_column=mean_vor_column[events]
            std_column=std_column[events]
            max_events_column=max_events_column[events]
            convective_fraction_column=convective_fraction_column[events]
            conv_filter_column=conv_filter_column[events]
            convective_or_not_column=convective_or_not_column[events]
            if np.sum(events)>0:
                for i in range(len(duration_column)):
                    
                    array=[duration_column[i],mean_intensity_column[i],max_events_column[i],std_column[i],conv_filter_column[i],mean_div_column[i],mean_vor_column[i],temp_column[i],rh_column[i],convective_or_not_column[i],convective_fraction_column[i],int(month),lons[ilon,ilat],lats[ilon,ilat]]
                    data[idata,:]=array
                    idata=idata+1
                    if idata==len(data[:,0]):
                        raise NameError('Data is not large enough')
data=data[:np.sum(data[:,0]!=0),:]
np.save(data_folder+'events_data',data)

#jle.Quick_plot((duration*mean_conv_intensity).mean(axis=0),'convective',metadata_dataset=jle.Load_sample_dataset_c())
#%%

#headers=['duration','mean_intensity','max_intensity','std','convective_filter','convective_or_not','convective_fraction','month','lon','lat']
headers=['duration','mean_intensity','max_intensity','std','convective_filter','div','vor','temp','rh','convective_or_not','convective_fraction','month','lon','lat']

data=np.load(data_folder+'events_data.npy')

#data[:,1]=np.log(data[:,1])
#data[:,2]=np.log(data[:,2])
data_subset=data[:np.sum(data[:,0]!=0),:]
from sklearn.preprocessing import normalize
import pandas as pd
#data_for_classifier=data_subset[:,:5]

#training_subset=np.random.choice(data_subset.shape[0], int(data_subset.shape[0]*0.8), replace=False)
from sklearn import preprocessing
import random
indexes=np.arange(0,data_subset.shape[0]).astype(int)
random.shuffle(indexes)
training_subset=indexes[:int(len(indexes)*0.01)]
evaluation_subset=indexes[int(len(indexes)*0.01):int(len(indexes)*0.02)]


#SWAPING COLUMNS
#data_subset[:,2]=data_subset[:,6]


#norm = normalize(random_subset[:,:4],axis=0)
#norm = normalize(random_subset[:,:4],axis=0)





training_data=data_subset[training_subset]
evaluation_data=data_subset[evaluation_subset]
evaluation_data.mean(axis=0)
training_data.mean(axis=0)



#norm = normalize(random_subset[:,:4],axis=0)
#norm = random_subset[:,:4]
df = pd.DataFrame(data,columns=headers)
X=training_data[:,:9]
y=training_data[:,9].astype(int)
#X=np.delete(np.delete(training_data[:,:],7,axis=1),7,axis=1)

#normalizer = preprocessing.Normalizer(norm='l1').fit(X)
#X=normalizer.transform(X)




from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(5)
clf.fit(X, y)

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-1,
                    hidden_layer_sizes=(10), random_state=1)
clf.fit(X, y)

from sklearn import svm
clf = svm.SVC()
clf.fit(X, y)
X_evaluation=evaluation_data[:,:9]
#X_evaluation=np.delete(np.delete(evaluation_data[:,:],7,axis=1),7,axis=1)

#X_evaluation=normalizer.transform(X_evaluation)


y_evaluation=evaluation_data[:,9].astype(int)

score=clf.score(X_evaluation, y_evaluation)

print(score )
y_predicted=clf.predict(X_evaluation)
correct=np.array(y_predicted==y_evaluation)
print(np.sum(correct)/len(correct))



#%%

#Predict
predicted_mean_convective_fraction=np.ones_like(convective_fraction[0,])*np.nan
for month in ['07']:
    print (month)
    duration=np.load(data_folder+'2006'+month+'duration_'+threshold+'.npy')
    temp=np.load(data_folder+'2006'+month+'temp_'+threshold+'.npy')
    rh=np.load(data_folder+'2006'+month+'rh_'+threshold+'.npy')
    mean_intensity=np.load(data_folder+'2006'+month+'mean_intensity_events_'+threshold+'.npy')
    mean_grsc_intensity=np.load(data_folder+'2006'+month+'mean_grsc_intensity_events_'+threshold+'.npy')
    mean_conv_intensity=np.load(data_folder+'2006'+month+'mean_conv_intensity_events_'+threshold+'.npy')
    conv_filter=np.load(data_folder+'2006'+month+'conv_filter_events_'+threshold+'.npy')
    mean_intensity=np.load(data_folder+'2006'+month+'mean_intensity_events_'+threshold+'.npy')
    mean_div=np.load(data_folder+'2006'+month+'div_events_'+threshold+'.npy')
    mean_vor=np.load(data_folder+'2006'+month+'vor_events_'+threshold+'.npy')
    max_events=np.load(data_folder+'2006'+month+'max_events_'+threshold+'.npy')
    std=np.load(data_folder+'2006'+month+'std_events_'+threshold+'.npy')
    convective_fraction=mean_conv_intensity/(mean_conv_intensity+mean_grsc_intensity)
    convective_or_not=np.zeros_like(convective_fraction)
    convective_or_not[convective_fraction>0.5]=1
#    convective_or_not[convective_fraction<0.01]=2
#    convective_or_not[convective_fraction<0.01 &&]=0
    for ilon in range(lons.shape[0]):
        print (ilon)
        for ilat in range(lats.shape[1]):
            duration_column=duration[:,ilat,ilon]
            temp_column=temp[:,ilat,ilon]
            rh_column=rh[:,ilat,ilon]
            mean_div_column=mean_div[:,ilat,ilon]
            mean_vor_column=mean_vor[:,ilat,ilon]
            mean_intensity_column=mean_intensity[:,ilat,ilon]
            std_column=std[:,ilat,ilon]
            max_events_column=max_events[:,ilat,ilon]
            convective_fraction_column=convective_fraction[:,ilat,ilon]
            conv_filter_column=conv_filter[:,ilat,ilon]
            convective_or_not_column=convective_or_not[:,ilat,ilon]
            
            events=[duration_column!=0]
            duration_column=duration_column[events]
            rh_column=rh_column[events]
            temp_column=temp_column[events]
            mean_intensity_column=mean_intensity_column[events]
            mean_div_column=mean_div_column[events]
            mean_vor_column=mean_vor_column[events]
            std_column=std_column[events]
            max_events_column=max_events_column[events]
            convective_fraction_column=convective_fraction_column[events]
            conv_filter_column=conv_filter_column[events]
            convective_or_not_column=convective_or_not_column[events]
            predicted=[]
            if np.sum(events)>0:
                for i in range(len(duration_column)):
                    
                    array=[[duration_column[i],mean_intensity_column[i],max_events_column[i],std_column[i],conv_filter_column[i],mean_div_column[i],mean_vor_column[i],temp_column[i],rh_column[i]]]
                    y_predicted=clf.predict(array)
                    predicted.append(y_predicted[0])
            
                predicted_mean_convective_fraction[ilat,ilon]=np.mean(predicted)
                
mean_convective_fraction=np.ones_like(convective_fraction[0,])*np.nan
convective_fraction[convective_fraction>0.5]=1
convective_fraction[convective_fraction<0.5]=0
for ilon in range(convective_fraction.shape[1]):
    for ilat in range(convective_fraction.shape[2]):
        if not np.isnan(convective_fraction[:,ilon,ilat]).sum()==len(convective_fraction[:,ilon,ilat]):
            mean_convective_fraction[ilon,ilat]=np.nanmean(convective_fraction[:,ilon,ilat])

#%%
plt.figure(figsize=(20,10))
plt.subplot(121)
jle.Quick_plot(mean_convective_fraction,'July 12km Convective_fraction',metadata_dataset=jle.Load_sample_dataset_c(),levels=np.linspace(0,1,11),new_fig=0)
plt.subplot(122)
jle.Quick_plot(predicted_mean_convective_fraction,'July 12km ML predicted Convective_fraction score: '+str(score),metadata_dataset=jle.Load_sample_dataset_c(),levels=np.linspace(0,1,11),new_fig=0)
#plt.savefig(plots_folder+'map_SVC.png')
plt.savefig(plots_folder+'map_svc_with_temp_and_rh.png')
plt.show()








#%%

'''
from sklearn import svm

X=data[:,:5]
y=data[:,6].astype(int)

clf = svm.SVC()
clf.fit(X, y)
#pd.scatter_matrix(df, figsize=(6, 6))
#plt.savefig(plots_folder+'scatter_matrix')
#plt.show()
#lons=
#def Regrid(values, lons, lats, grid_lons, grid_lats):

#%%
var='max_intensity'
var='convective_fraction'
values=df[var].values
points=[]    
from scipy.interpolate import griddata
coord=np.zeros([len(df['lon']),2])
coord[:,0]=df['lon']
coord[:,1]=df['lat']
grid_z0 = griddata(coord, values, (lons, lats), method='nearest')

jle.Quick_plot(grid_z0,var,metadata_dataset=jle.Load_sample_dataset_c())

'''


#%%
#blah = 1
#var=0
#ver=0
#blah_name = [ k for k,v in locals().items() if v is blah][0]
#for var,ver in locals().items():
#    print(var,ver)

