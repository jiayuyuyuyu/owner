from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense,Dropout,Input
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adagrad
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils.data_utils import Sequence
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import keras 
from keras.models import Sequential
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.metrics import roc_auc_score
from keras import backend as K
from sklearn.model_selection import train_test_split #將資料分開成兩部分
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris #导入IRIS数据集 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

try:    #讀權重檔案，成功會print complete 
    load_model("t3.h5")
    print("Complete")
except:
    print("Fail")

#Keras模型構建主要包括5個步驟：定義（define），編譯（compile），訓練（fit），評估（evaluate），預測（prediction）#
#數據擴增
checkpoint = ModelCheckpoint(        #保存點
	"t3.h5",
	monitor='val_loss',
	verbose=1,
	save_best_only=True,
	mode='min')
batch_size = 16

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,# ((x/255)-0.5)*2  归一化到±1之间
    rotation_range=30, #旋轉
    width_shift_range=0.2, #水平位置平移
    height_shift_range=0.2, #上下位置平移
    shear_range=0.2, #讓所有點的x坐標(或者y坐標)保持不變，而對應的y坐標(或者x坐標)則按比例發生平移
    zoom_range=0.2, #讓圖片在長或寬的方向進行RESIZE,參數大於0小於1時，放大，參數大於1時，縮小
    horizontal_flip=True, #執行水平翻轉操作，意味著不一定對所有圖片都會執行水平翻轉，每次生成均是隨機選取圖片進行翻轉
    rescale=1./255
)

# 驗證集的圖片生成器，不進行數據擴增，只進行數據預處理
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
     rotation_range=30, #旋轉
    width_shift_range=0.2, 
    height_shift_range=0.2,
    shear_range=0.2, 
    zoom_range=0.2,
    horizontal_flip=True, 
    rescale=1./255
 )

train_generator = train_datagen.flow_from_directory(directory='t4\\train',   #flow_from_directory讀取圖片，ImageDataGenerator進行增強
                                  target_size=(224,224),#Inception V3規定大小
                                  batch_size=batch_size, 
                                  class_mode='categorical',
                                )  
val_generator = val_datagen.flow_from_directory(directory='t4\\val',
                                target_size=(224,224),
                                batch_size=batch_size, 
                                 class_mode='categorical',
                              shuffle=False)   

print(len(val_generator))
print(len(train_generator))

labels = (train_generator.class_indices)


base_model = InceptionV3(weights='imagenet',include_top=False)



x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x)
x = Dropout(0.5)(x)

predictions = Dense(4,activation='softmax')(x)   
model = Model(inputs=base_model.input,outputs=predictions)
model.summary()
def setup_to_transfer_learning(model,base_model):#base_model
    for layer in base_model.layers:
        layer.trainable = False
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) #編譯模型


setup_to_transfer_learning(model,base_model)

history = model.fit_generator(generator=train_generator,    #訓練模型
                    steps_per_epoch=len(train_generator)// batch_size,#800
                    epochs=100,#2
                    validation_data=val_generator,
                    callbacks=[checkpoint],
                    validation_steps=len(val_generator) // batch_size,#12
                   )
                    
