    import numpy as np
import matplotlib.pyplot as plt
import glob
import iris
from netCDF4 import Dataset


import sys
sys.path.append('/users/jvergara/python_code')
import Jesuslib_eth as jle


run_path='/project/pr04/davidle/results_clim/lm_f/'
folder_in_path='1h/2000/'
a=glob.glob(run_path+folder_in_path+'*20000601*')
run_path='/store/c2sm/pr04/jvergara/GREATER_ALPINE/GA_TRY_1/'
run_path='/store/c2sm/pr04/jvergara/GREATER_ALPINE/GA_TRY_2/'
run_path='/store/c2sm/pr04/jvergara/GREATER_ALPINE/GA_TRY_2.3/'
run_path='/store/c2sm/pr04/jvergara/GREATER_ALPINE/GA_TRY_2.2/'
run_path='/store/c2sm/pr04/jvergara/'
a=glob.glob(run_path+'output/1h/*nc')

nikolinas='/project/pr04/banni/results_PGW/PGW/lm_f/'
folder_in_path='1h/'
davids='/project/pr04/davidle/results_clim/lm_f/'

a=np.sort(a)
print (a)
#all_day=iris.load(a)
#dataset = Dataset(run_path+'lffd2001061312_2km_remaped_12km.nc')
dataset1 = Dataset(davids+'1h/2001/lffd2001061312.nc')
dataset12 = Dataset(davids+'1h_second/2001/lffd2001061312.nc')
dataset3 = Dataset(davids+'3h/2001/lffd2001061312.nc')
dataset6 = Dataset(davids+'6h/2001/lffd2001061312p.nc')
dataset24 = Dataset(davids+'24h/2001/lffd2001061300.nc')
name=run_path.split('/')[-2]
#dataset=Dataset(filtered_dict['2000122104'])
#jle.Quick_plot(dataset.variables['TOT_PREC'][0,],'TOT_PREC',metadata_dataset=jle.Load_sample_dataset_c())
dataset=Dataset('/store/c2sm/pr04/jvergara/lffd2001061312_2km_remaped_12km.nc')
dataset=Dataset('/store/c2sm/pr04/jvergara/lffd2001061312_2km_into_12km.nc')


dataset=Dataset('/store/c2sm/pr04/jvergara/lffd2001061312_2km_regrid_05.nc')
dataset=Dataset('/scratch/snx3000/jvergara/lffd2001061312_2km_regrid_05.nc')
dataset_original=Dataset('/store/c2sm/pr04/jvergara/lffd2001061312_2km.nc')

#plt.figure()
#plt.subplot(221)
#plt.subplot(222)
variable='ATHB_S'
levels=np.logspace(-3,1,15)
levels=np.linspace(-400,00,15)
#jle.Quick_plot(dataset,'TWATER')
jle.Quick_plot(dataset,variable,metadata_dataset=jle.Load_sample_dataset_c())
jle.Quick_plot(dataset,variable,metadata_dataset=jle.Load_sample_dataset_c(),levels=levels)
jle.Quick_plot(dataset_original,variable,levels=levels)

print(dataset.variables[variable][:].mean(),dataset_original.variables[variable][:].mean())
#jle.Quick_plot(jle.Load_sample_dataset_c(),'TOT_PREC')


#%%

X=dataset.variables['lon']
Y=dataset.variables['lat']
levels=np.arange(-14,26,1).tolist()
variable='lat'
levels=np.arange(37,55,1).tolist()
#levels=np.linspace(255,300,21).tolist()

#dataset


m=jle.Quick_plot(dataset,variable,levels=levels,title=dataset.variables[variable].long_name+' '+name,shadedrelief=1,return_m=1,npar=30,nmer=80)
#TLC to TRC
x, y = m([-3.33,17.27], [50.04,51.99])
m.plot(x, y, 'o-', markersize=5, linewidth=2,c='r') 

#TLC to BLC
x, y = m([-3.33,1.011], [50.04,38.26])
m.plot(x, y, 'o-', markersize=5, linewidth=2,c='r') 

#BRC to TRC
x, y = m([17.42,17.27], [39.81,51.99])

m.plot(x, y, 'o-', markersize=5, linewidth=2,c='r')
#BRC to BLC
x, y = m([17.42,1.011], [39.81,38.266])
m.plot(x, y, 'o-', markersize=5, linewidth=2,c='r')



x, y = m([1,17], [50,50])
m.plot(x, y, 'o-', markersize=5, linewidth=2,c='w') 

#TLC to BLC
x, y = m([1,1], [50.0,40])
m.plot(x, y, 'o-', markersize=5, linewidth=2,c='w') 

#BRC to TRC
x, y = m([17.0,17.0], [40,50])

m.plot(x, y, 'o-', markersize=5, linewidth=2,c='w') 
#BRC to BLC
x, y = m([17.0,1.0], [40,40])
m.plot(x, y, 'o-', markersize=5, linewidth=2,c='w') 




print(X[:].min(),Y[:].min())





plt.figure()
#plt.subplot(121)
plt.pcolormesh(X,Y,dataset[variable][:,:])


x, y = ([-3.33,17.27], [50.04,51.99])
plt.plot(x, y, 'o-', markersize=5, linewidth=2,c='r') 

#TLC to BLC
x, y = ([-3.33,1.011], [50.04,38.26])
plt.plot(x, y, 'o-', markersize=5, linewidth=2,c='r')

#BRC to TRC
x, y = ([17.42,17.27], [39.81,51.99])
plt.plot(x, y, 'o-', markersize=5, linewidth=2,c='r') 

#BRC to BLC
x, y = ([17.42,1.011], [39.81,38.266])
plt.plot(x, y, 'o-', markersize=5, linewidth=2,c='r') 



x, y = ([1,17], [50,50])
plt.plot(x, y, 'o-', markersize=5, linewidth=2,c='w') 



#TLC to BLC
x, y = ([1,1], [50.0,40])
plt.plot(x, y, 'o-', markersize=5, linewidth=2,c='w') 

#BRC to TRC
x, y = ([17.0,17.0], [40,50])

plt.plot(x, y, 'o-', markersize=5, linewidth=2,c='w') 
#BRC to BLC
x, y = ([17.0,1.0], [40,40])
plt.plot(x, y, 'o-', markersize=5, linewidth=2,c='w') 





plt.show()
#plt.subplot(122)
#X=dataset.variables['rlon']
#Y=dataset.variables['rlat']
#plt.pcolormesh(X,Y,dataset[variable][0,:,:])

#plt.ylim(35,56)


#%%
from mpl_toolkits.basemap import Basemap
cubes=iris.load(a[0])
latitudes = cubes[6].data
longitudes = cubes[8].data
lon_0 = longitudes.mean()
lat_0 = latitudes.mean()

fig = plt.figure(figsize=(8,8))
ax = fig.add_axes([0.1,0.1,0.8,0.8])
# create polar stereographic Basemap instance.
m = Basemap(projection='stere',lon_0=lon_0,lat_0=90.,lat_ts=lat_0,\
            llcrnrlat=latitudes.max(),urcrnrlat=latitudes.min(),\
            llcrnrlon=longitudes.max(),urcrnrlon=longitudes.min(),\
            rsphere=6371200.,resolution='l',area_thresh=10000)
# draw coastlines, state and country boundaries, edge of map.
m.drawcoastlines()
m.drawstates()
m.drawcountries()
# draw parallels.
parallels = np.arange(0.,90,10.)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
# draw meridians
meridians = np.arange(180.,360.,10.)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)

#%%


cubes=iris.load(a[0])


cf=cubes[3]

plt.imshow(cf.data[0,])

#%%


rain=[0,0,0,1,1,2,1,1,1,2,2,2.5,3,1,6,1,1,1,1,0,0,0,3,5,0,0,0,1,2,3,6,1,0]
P=np.zeros(len(rain))
for i in range(len(rain)):
    if i==0:
        continue
    if i==len(rain)-1:
        continue
    print(i)
    print (rain[i])
    print (rain[i-1]-rain[i]**2+rain[i+1])
    P[i]=-0.25*(rain[i-1]-rain[i]**2+rain[i+1])
plt.plot(rain,label='rain')
plt.plot(P,label='P')
plt.legend()



