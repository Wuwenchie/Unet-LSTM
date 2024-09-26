import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
import numpy as np
import xarray as xr

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

file_path = "download-1.nc"
data = nc.Dataset(file_path)
print(data.variables.keys())
ds = xr.open_dataset(file_path)
print(ds)

lat = data.variables['latitude'][:]
lon = data.variables['longitude'][:]
time = data.variables['time'][:]
t2m = data.variables['t2m'][:]
sst = data.variables['sst'][:]
print(sst)
np.savetxt(r'matrix_t2m.txt', t2m.reshape(-1, t2m.shape[-1]), fmt='%f', delimiter=',')
np.savetxt(r'matrix_sst.txt', sst.reshape(-1, sst.shape[-1]), fmt='%f', delimiter=',')

# combine = np.arange(1038240).reshape(1440,721,1)
matrix_t2m = np.loadtxt('matrix_t2m.txt', dtype=float, delimiter=',').reshape(721, 1440, 1)
matrix_sst = np.loadtxt('matrix_sst.txt', dtype=float, delimiter=',').reshape(721, 1440, 1)
combine = np.concatenate((matrix_t2m, matrix_sst*-1), axis=-1)

lon2, lat2 = np.meshgrid(lon, lat)

m = Basemap(projection='cyl',
            llcrnrlat=-90,
            urcrnrlat=90,
            llcrnrlon=-180,
            urcrnrlon=180,
            resolution='c')

cx, cy = m(lon2, lat2)

cs = m.pcolormesh(cx,cy,np.squeeze(combine[:,:,0]), cmap='jet')

# 畫海岸線
m.drawcoastlines()
# 添加 colorbar
cbar = m.colorbar(cs,"bottom", pad="10%")

cbar.set_label('Temperature (K)')
plt.title('combine Temperature  2010-03')
plt.savefig('combine Temperature  2010-03.png')
plt.show()
