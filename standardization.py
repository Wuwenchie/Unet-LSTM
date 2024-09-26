import numpy as np
import xarray as xr
import os

def process_data(file_path):
    # 假設數據是 xarray.DataArray 格式，形狀為 (time, 720, 1440)
    data = xr.open_dataset(file_path)['__xarray_dataarray_variable__']

    # 將 720x1440 的數據平均到 180x360 的 1x1 度格網
    # 我們可以使用重採樣方法，對數據進行平均
    # time 維度保持不變，只對緯度和經度進行平均
    data_avg = data.coarsen(latitude=4, longitude=4, boundary='trim').mean()    # 大小為 (time, 180, 360)

    # 使用 xarray 內建的插值方法進行雙線性插值
    # 定義新的經緯度範圍
    lat_new = np.linspace(-64, 62, 64)
    lon_new = np.linspace(-180, 180, 128)

    # 使用 xarray 的 interp 進行雙線性插值
    data_interp = data_avg.interp(latitude=lat_new, longitude=lon_new)      # 大小為 (time, 64, 128)

    # 使用 (X - min) / (max - min) 方法將數據標準化到 [0, 1]
    data_min = data_interp.min()
    data_max = data_interp.max()

    # 標準化
    data_normalized = (data_interp - data_min) / (data_max - data_min)      # 大小為 (time, 64, 128)

    return data_normalized

# 初始化一個列表來存儲處理後的數據
processed_data_list = []

# 所有合併後的 NetCDF 文件都放在同一個文件夾下
folder_path = "./validation_data"  # 合併後數據的文件夾
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".nc")]

# 迴圈讀取和處理數據
for file_path in file_paths:
    print(f"Processing {file_path}...")
    processed_data = process_data(file_path)
    processed_data_list.append(processed_data)

# 將列表轉換為 NumPy 數組
processed_data_array = np.stack([data.values for data in processed_data_list], axis=0)  # (n_months, 64, 128)

processed_data_array = np.expand_dims(processed_data_array, axis=-1)  # (n_months, height, width, 1)

time_steps = 12
future_steps = 2

# 創建訓練樣本和標籤
n_samples = len(processed_data_array) - time_steps - future_steps + 1  # 用滑動窗口技術來創建訓練樣本

x_test = []
y_test = []

for i in range(n_samples):
    x_test.append(processed_data_array[i:i+12])  # 過去 12 個月的數據
    y_test.append(processed_data_array[i+12:i+14])  # 未來 2 個月的數據

# 將列表轉為 NumPy 數組
x_test = np.array(x_test)
y_test = np.array(y_test)

# 檢查 X_train 和 y_train 的形狀
print("x_test shape:", x_test.shape)  # 應該輸出 (樣本數, 12, 64, 128)
print("y_test shape:", y_test.shape)  # 應該輸出 (樣本數, 2, 64, 128)

print("x_test size (in MB):", x_test.nbytes / (1024 * 1024))  # 轉換為 MB
print("y_test size (in MB):", y_test.nbytes / (1024 * 1024))


# 保存處理好的數據
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)


"""
# 初始化 Git 儲存庫
git init

# 添加文件
git add X_train.npy y_train.npy X_test.npy y_test.npy

# 提交更改
git commit -m "Upload processed data"

# 將更改推送到 GitHub
git remote add origin https://github.com/your_username/your_repo.git
git push -u origin main
"""