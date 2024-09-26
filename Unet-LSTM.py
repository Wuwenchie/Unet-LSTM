import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ConvLSTM2D, UpSampling2D, Input
from tensorflow.keras.layers import MaxPooling2D, concatenate, TimeDistributed, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt


def conv_lstm_layer(filters=64, kernel_size=(3, 3), activation='tanh', padding='same',
                           kernel_reg=1e-8, bias_reg=1e-8, recurrent_reg=1e-8, return_sequences=False):

    lstm_out = ConvLSTM2D(
        filters=filters,  # LSTM層的過濾器數量，可以根據需求動態設置
        kernel_size=kernel_size,  # 卷積核大小
        activation=activation,  # 激活函數
        padding=padding,  # 使用'same'填充方式保持輸入和輸出形狀一致
        kernel_regularizer=l2(kernel_reg),  # 卷積核的L2正則化
        bias_regularizer=l2(bias_reg),  # 偏置項的L2正則化
        recurrent_regularizer=l2(recurrent_reg),  # 遞歸項的L2正則化
        return_sequences=return_sequences  # 返回整個序列或僅返回最後一個步驟的輸出
    )

    return lstm_out

input_shape = (12, 64, 128, 1)

def unet_lstm_model(input_shape):
    inputs = Input(input_shape)

    # TimeDistributed層應用Unet的編碼部分
    conv1 = conv_lstm_layer(filters=64, return_sequences=True)(inputs)
    conv1 = conv_lstm_layer(filters=64, return_sequences=True)(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)

    conv2 = conv_lstm_layer(filters=128, return_sequences=True)(pool1)
    conv2 = conv_lstm_layer(filters=128, return_sequences=True)(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)

    # 編碼到瓶頸
    # 瓶頸部分
    conv3 = conv_lstm_layer(filters=256, return_sequences=True)(pool2)
    conv3 = conv_lstm_layer(filters=256, return_sequences=True)(conv3)


    up1 = TimeDistributed(UpSampling2D((2, 2)))(conv3)
    # 若需要將通道數調整為 128，可以使用 Conv2D 或其他方式
    up1 = TimeDistributed(Conv2D(128, (1, 1), activation='relu'))(up1)
    # print("up1 ", up1.shape)

    concat1 = concatenate([up1, conv2])
    conv4 = conv_lstm_layer(filters=128, return_sequences=True)(concat1)
    conv4 = conv_lstm_layer(filters=128, return_sequences=True)(conv4)
    # print("conv4", conv4.shape)

    up2 = TimeDistributed(UpSampling2D((2, 2)))(conv4)
    up2 = TimeDistributed(Conv2D(64, (1, 1), activation='relu'))(up2)
    # print("up2", up2.shape)
    concat2 = concatenate([up2, conv1])
    conv5 = conv_lstm_layer(filters=64, return_sequences=True)(concat2)
    conv5 = conv_lstm_layer(filters=64, return_sequences=True)(conv5)

    def select_last_two_steps(x):
        return x[:, -2:, :, :, :]  # 選擇最後兩個時間步

    selected_output = Lambda(select_last_two_steps)(conv5)

    outputs = TimeDistributed(Conv2D(1, (1, 1), activation='linear'))(selected_output)  # 假設預測單一變量SST

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), loss='mean_squared_error')

    return model

model = unet_lstm_model(input_shape)
print(model.summary())


# 準備訓練數據
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

# print(X_train.shape)
# print(y_train.shape)


# 訓練模型
train_history = model.fit(X_train, y_train, batch_size=32, epochs=200, validation_split=0.2)


# 來把訓練過程畫出來
def show_train_history(train_history,train,validation):

  plt.plot(train_history.history[train])
  plt.plot(train_history.history[validation])
  plt.title('Train history')
  plt.ylabel('train')
  plt.xlabel('epoch')

  # 設置圖例在左上角
  plt.legend(['train','validation'],loc='upper left')
  plt.show()

# show_train_history(train_history,'accuracy','val_accuracy')
show_train_history(train_history,'loss','val_loss')


# 使用訓練好的模型進行預測
predictions = model.predict(X_train)


def autoregressive_prediction(model, initial_input, steps=24):
    predictions = []
    input_data = initial_input

    for step in range(steps):
        pred = model.predict(input_data)
        predictions.append(pred)
        # 更新輸入數據，將預測結果作為下一次的輸入
        input_data = np.concatenate((input_data[:, 1:], pred), axis=1)

    return np.array(predictions)


# 使用初始的12個月數據進行24個月的自迴歸預測
future_predictions = autoregressive_prediction(model, X_train)

