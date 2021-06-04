from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
from io import BytesIO
import base64
from django.http import HttpResponse
import tensorflow.keras as keras
from . import data
import os
from django.views.decorators.csrf import csrf_exempt
import skimage.io as io
import skimage.transform as trans
import numpy as np
from io import BytesIO
import cv2
import os
import sys
import numpy as np
import csv
import setproctitle
import time


import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16
from PIL import Image, ImageDraw
from skimage import draw


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
config = ConfigProto()
config.gpu_options.allow_growth = True  # GPU内存增长
session = InteractiveSession(config=config)


# -----------------------------槽的模型加载及预测试---------------------
# 提前加载好，以后预测的时候就速度快了
slotModel = keras.models.load_model(
    r"D:\unet\modelFile\slot0528R\unet_model.hdf5")  # 两个模型文件的位置，最好写绝对路径吧
slotModel.load_weights(r"D:\unet\modelFile\slot0528R\unet_weight.hdf5")
img = io.imread("./slotTest.png", as_gray=True)
img = img / 255
img = trans.resize(img, (256, 256))
img = np.reshape(img, img.shape + (1,)) if (not False) else img
img = np.reshape(img, (1,) + img.shape)

result = slotModel.predict(img, verbose=1)
print("load and test slot model successfully")


# 提前加载好，以后预测的时候就速度快了
# -----------------------------槽的模型加载及预测试---------------------


# -----------------------------孔的模型加载及预测试---------------------
IMAGE_W = 224  # 图片大小
IMAGE_H = 224
OUTPUT_PARAMS = 3  # x, y, radius
holeModel = keras.models.load_model(r"E:\data\hole-vgg.hdf5")  # 圆孔的模型
# holeModel.summary()
x = []
img = Image.open("./holeTest.png").resize((IMAGE_H, IMAGE_W), Image.LINEAR)
x.append(np.reshape(np.array(img, 'f'), (IMAGE_H, IMAGE_W, 3)))
x = np.array(x, dtype=np.float32) / 255.0
# print(x)
result = holeModel.predict(x, verbose=1)
print("load and test hole model successfully")
# result = result[0]*IMAGE_H
# x, y, r = result
# print(result)
# img = cv2.imread("./holeTest.png")
# cv2.circle(img, (x, y), int(r), (0, 0, 255), 1)
# cv2.imwrite("./holeResult.png",img)
# print(result)

# -----------------------------孔的模型加载及预测试---------------------


# -----------------------------槽的测试逻辑，返回的是槽的两个横坐标---------------------

@csrf_exempt
def slot(request):
    isVertical = request.POST.get('isVertical')
    print(type(isVertical))
    stringdata = request.POST.get('pic')
    img_bytes = base64.b64decode(stringdata)
    bytes_stream = BytesIO(img_bytes)
    # io.imsave("./a.png", bytes_stream)
    img = io.imread(bytes_stream)  # base64转图片
    bytes_stream.close()
    # io.imsave("./aa.png",img)  #传过来了~
    img = img / 255
    print(type(img))
    img = trans.resize(img, (256, 256))
    img = np.reshape(img, img.shape + (1,)) if (not False) else img
    img = np.reshape(img, (1,) + img.shape)
    result = slotModel.predict(img, verbose=1)
    result = result[0, :, :, 0]
    result = np.float32(result)
    io.imsave(".\pre.png",result)  ## 也确实预测出来了
    if isVertical == "1":
        colSum = np.sum(result, axis=0)
    else:
        colSum = np.sum(result, axis=1)
    # print(colSum)
    max = np.max(colSum)
    min = np.min(colSum)
    delta = max - min
    colSum = ((colSum - min) * 1.0)/delta
    smallIndex = []
    for i in range(256):
        if colSum[i] <= 0.6:
            smallIndex.append(i)
    print(len(smallIndex))
    if len(smallIndex)<10:
        smallIndex.clear()
        for i in range(256):
            if colSum[i] <=0.9:
               smallIndex.append(i)
    indexMean = np.mean(smallIndex)
    print(indexMean, len(smallIndex))
    c1 = -1
    c2 = -1
    print(indexMean)
    if(len(smallIndex) > 1):
        c1 = int(np.mean([x for x in smallIndex if x < indexMean]))*2
        c2 = int(np.mean([x for x in smallIndex if x >= indexMean]))*2
    response = []
    dict = {}
    dict['index1'] = c1
    dict['index2'] = c2
    dict['index3']=colSum[int(c1/2)]
    dict['index4']=colSum[int(c2/2)]
    response.append(dict)
    return HttpResponse(response, content_type="application/json")
# -----------------------------槽的测试逻辑，返回的是槽的两个横坐标---------------------


# -----------------------------孔的测试逻辑，返回的是孔的圆心坐标，及半径---------------------
@csrf_exempt
def hole(request):
    x = -1
    y = -1
    r = -1
    stringdata = request.POST.get('pic')
    img_bytes = base64.b64decode(stringdata)
    bytes_stream = BytesIO(img_bytes)
    img = Image.open(bytes_stream).resize((IMAGE_H, IMAGE_W), Image.LINEAR)
    bytes_stream.close()
    x = []
    x.append(np.reshape(np.array(img, 'f'), (IMAGE_H, IMAGE_W, 3)))
    x = np.array(x, dtype=np.float32) / 255.0
    # print(x)
    result = holeModel.predict(x, verbose=1)
    result = result[0]*IMAGE_H
    response = []
    dict = {}
    dict['x'], dict['y'], dict['r'] = result
    response.append(dict)
    return HttpResponse(response, content_type="application/json")
    bytes_stream.close()
# -----------------------------孔的测试逻辑，返回的是孔的圆心坐标，及半径---------------------
