import netCDF4 as nc
import numpy as np
import xarray as xr
import os
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import time


# 創建一個處理單個文件的函數
def combined_data(file_path, year, month):
    # 打開 NetCDF 文件
    data = nc.Dataset(file_path)

    # 讀取數據
    sst_data = xr.open_dataset(file_path)['sst']
    t2m_data = xr.open_dataset(file_path)['t2m']
    lon = xr.open_dataset(file_path)['longitude']
    lat = xr.open_dataset(file_path)['latitude']
    land_sea_mask = xr.open_dataset(file_path)['lsm']

    # 合併SST和T2M數據
    combined_temperature = xr.where(land_sea_mask == 0, sst_data, t2m_data)

    # 保存合併的數據到NetCDF
    combined_filename = f"combined_{year}_{month}.nc"
    combined_temperature.to_netcdf(combined_filename)
    print(f"Saved combined data to {combined_filename}")

    # 生成經緯度網格
    lon2, lat2 = np.meshgrid(lon, lat)

    # 繪製地圖
    m = Basemap(projection='cyl',
                llcrnrlat=-90,
                urcrnrlat=90,
                llcrnrlon=-180,
                urcrnrlon=180,
                resolution='c')

    cx, cy = m(lon2, lat2)

    # 畫圖
    cs = m.pcolormesh(cx, cy, np.squeeze(combined_temperature[:, :]), cmap='jet')

    # 畫海岸線
    m.drawcoastlines()

    # 添加 colorbar
    cbar = m.colorbar(cs, "bottom", pad="10%")
    cbar.set_label('Temperature (K)')
    plt.title(f'Combined Temperature {year}-{month}')

    # 保存圖像
    image_filename = f'combined_temperature_{year}_{month}.png'
    plt.savefig(image_filename)
    print(f"Saved image to {image_filename}")

    plt.close()


# 遍歷所有下載的文件
download_folder = "./"  # 假設你所有的 .nc 文件都在當前目錄下
years = range(1961, 1964 + 1)  # 例如從1950年到2021年
months = [f'{i:02d}' for i in range(1, 13)]  # '01', '02', ..., '12'

for year in years:
    for month in months:
        file_path = os.path.join(download_folder, f"downloaded_{year}_{month}.nc")
        if os.path.exists(file_path):
            try:
                print(f"Processing {file_path}")
                combined_data(file_path, year, month)
                time.sleep(1)  # 可選，讓程式每次處理完等待 1 秒

            except Exception as e:
                print(f"Error processing {file_path}: {e}")


"""
# combine.py原程式碼
import netCDF4 as nc
import numpy as np
import xarray as xr
from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

file_path = "downloaded_2011_05.nc"
data = nc.Dataset(file_path)
print(data.variables.keys())

sst_data = xr.open_dataset(file_path)['sst']
t2m_data = xr.open_dataset(file_path)['t2m']
lon = xr.open_dataset(file_path)['longitude']
lat = xr.open_dataset(file_path)['latitude']
land_sea_mask = xr.open_dataset(file_path)['lsm']  # mask的維度與sst_data和t2m_data相同

# 使用掩模將SST和T2m數據合併成一個整體數據集，根據每個網格點的海陸屬性選擇SST或T2m
# 創建一個與sst_data和t2m_data相同形狀的數組來存放合併的溫度數據
combined_temperature = xr.where(land_sea_mask == 0, sst_data, t2m_data)

# 將xarray數據保存為NetCDF格式的文件
combined_temperature.to_netcdf("combined_2011_05.nc")
file = "combined_2011_06.nc"
d = nc.Dataset(file)
print(d.variables.keys())

lon2, lat2 = np.meshgrid(lon, lat)

m = Basemap(projection='cyl',
            llcrnrlat=-90,
            urcrnrlat=90,
            llcrnrlon=-180,
            urcrnrlon=180,
            resolution='c')

cx, cy = m(lon2, lat2)

cs = m.pcolormesh(cx,cy,np.squeeze(combined_temperature[:,:]), cmap='jet')

# 畫海岸線
m.drawcoastlines()
# 添加 colorbar
cbar = m.colorbar(cs,"bottom", pad="10%")

cbar.set_label('Temperature (K)')
plt.title('combine Temperature  2011-05')
plt.savefig('combine Temperature  2011-05.png')
plt.show()
"""


