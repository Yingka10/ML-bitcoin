# 匯入必要模組
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras.utils as image_utils
import numpy as np
import os
import shutil
import random

# 分割資料：從 train/0 與 train/1 複製部分圖片至 valid 資料夾

def split_data_to_valid(source_dir, target_dir, split_ratio=0.2):
    for label in ['0', '1']:
        src_folder = os.path.join(source_dir, label)
        dst_folder = os.path.join(target_dir, label)
        os.makedirs(dst_folder, exist_ok=True)

        files = os.listdir(src_folder)
        random.shuffle(files)
        split_num = int(len(files) * split_ratio)
        selected_files = files[:split_num]

        for fname in selected_files:
            shutil.copy(os.path.join(src_folder, fname), os.path.join(dst_folder, fname))

# 執行分割(執行一次即可)
split_data_to_valid('KLineImages/train', 'KLineImages/valid', split_ratio=0.05)

# 載入 VGG16 作為 feature extractor
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# 建立自訂分類模型
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(1)(x)  # 二分類：漲 (1), 跌 (0)
model = Model(inputs, outputs)

# 編譯模型
model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 建立 ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.2
)

valid_datagen = ImageDataGenerator(rescale=1./255)

# 建立 generator
train_generator = train_datagen.flow_from_directory(
    'KLineImages/train',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=32
)

valid_generator = valid_datagen.flow_from_directory(
    'KLineImages/valid',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=8
)

# 訓練模型
model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=20,
    steps_per_epoch=12,
    validation_steps=4
)

# 預測函式
def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def load_and_process_image(image_path):
    image_s = image_utils.load_img(image_path, target_size=(224, 224))
    image_s_array = image_utils.img_to_array(image_s) / 255.0
    image_s_array_reshape = np.expand_dims(image_s_array, axis=0)
    return image_s_array_reshape

def predict_up_down(image_path):
    show_image(image_path)
    image = load_and_process_image(image_path)
    prediction = model.predict(image)
    if prediction[0] < 0.5:
        print("預測：下跌 (0)")
    else:
        print("預測：上漲 (1)")
# 測試預測
# predict_up_down('kline_images/2018-05-14.png')  # 替換成你的圖片路徑

# 儲存模型
model.save('kline_forecast.keras')