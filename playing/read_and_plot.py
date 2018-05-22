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
run_path='/store/c2sm/pr04/jvergara/GREATER_ALPINE/GA_TRY_2.4/'
a=glob.glob(run_path+'output/1h/*nc')


a=np.sort(a)
print (a)
all_day=iris.load(a)
dataset = Dataset(a[0])
name=run_path.split('/')[-2]
dataset=Dataset(a[30])
print(dataset.variables['TOT_PR'][:].mean())
print(dataset.variables['TOT_PREC'][:].mean())



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
