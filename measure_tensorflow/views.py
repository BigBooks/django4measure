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


# # -----------------------------孔的模型加载及预测试---------------------
# IMAGE_W = 224  # 图片大小
# IMAGE_H = 224
# OUTPUT_PARAMS = 3  # x, y, radius
# holeModel = keras.models.load_model(r"E:\data\hole-vgg.hdf5")  # 圆孔的模型
# # holeModel.summary()
# x = []
# img = Image.open("./holeTest.png").resize((IMAGE_H, IMAGE_W), Image.LINEAR)
# x.append(np.reshape(np.array(img, 'f'), (IMAGE_H, IMAGE_W, 3)))
# x = np.array(x, dtype=np.float32) / 255.0
# # print(x)
# result = holeModel.predict(x, verbose=1)
# print("load and test hole model successfully")
# # result = result[0]*IMAGE_H
# # x, y, r = result
# # print(result)
# # img = cv2.imread("./holeTest.png")
# # cv2.circle(img, (x, y), int(r), (0, 0, 255), 1)
# # cv2.imwrite("./holeResult.png",img)
# # print(result)

# # -----------------------------孔的模型加载及预测试---------------------


# -----------------------------槽的测试逻辑，返回的是槽的两个横坐标---------------------

@csrf_exempt
def slot(request):
    isVertical = request.POST.get('isVertical')
    tag = request.POST.get('tag')

    print("isvertical",isVertical)
    stringdata = request.POST.get('pic')
    img_bytes = base64.b64decode(stringdata)
    bytes_stream = BytesIO(img_bytes)
    # io.imsave("./a.png", bytes_stream)
    img = io.imread(bytes_stream)  # base64转图片
    bytes_stream.close()
    # io.imsave("./aa.png",img)  #传过来了~
    img = img / 255
    img = trans.resize(img, (256, 256))
    img = np.reshape(img, img.shape + (1,)) if (not False) else img
    img = np.reshape(img, (1,) + img.shape)
    result = slotModel.predict(img, verbose=1)
    result = result[0, :, :, 0]
    result = np.float32(result)
    io.imsave(".\pre/"+tag+".png", result)  # 也确实预测出来了
    
    
    #####################槽识别canny+hough法########################
#     if isVertical == "1":
#        colSum = np.sum(result, axis=0)
#     else:
#        colSum = np.sum(result, axis=1)
#     max = np.max(colSum)
#     min = np.min(colSum)
#     delta = max - min
#     colSum = ((colSum - min) * 1.0)/delta
#     gray = np.uint8(np.interp(result, (result.min(),result.max()), (0, 255)))
# #     gray=np.uint8(result)
# #     gray = cv2.blur(gray, (3, 3))
# #     gray = cv2.medianBlur(gray, 5)
# #     gray=cv2.bilateralFilter(gray,9,75,75)
#     grad_x = cv2.Sobel(gray,cv2.CV_16SC1,1,0)  #用sobel算子求梯度。最后两个参数就是说是求的x,还是y的梯度
#     grad_y = cv2.Sobel(gray,cv2.CV_16SC1,0,1)
#     edge = cv2.Canny(grad_x,grad_y,30,150)
# #     cv2.imshow('canny', edge)
# #     cv2.waitKey(0)
#     lines = cv2.HoughLinesP(edge, 1,np.pi/180, 100, 230, 10)
# #     lines= cv2.HoughLines(edge, 1, np.pi/180, 110)
#     print("here is lines",type(lines))
#     test=None
#     isexist=isinstance(lines,type(test))
#     print("test type",type(test))
#     print("here is type",isexist)
#     
#     if isexist==False:
#        print("lines[1]is:",lines[1])
#        print(lines)
#        slot_l=[] # 存放所有检测到的线条的长度数据，便于找出最长的两条直线
#        slot_waist=[]   #存放检测到的直线的横/纵坐标数据，用于确定最终的槽的侧边坐标
#        for i in range(len(lines)):
#            for x1, y1, x2, y2 in lines[i]:
# #                cv2.line(result, (x1, y1), (x2, y2), (0,255,0), 2)
#                if isVertical == "1":
# #                   slot_l.append(abs(y1-y2))
#                   slot_waist.append(x1)
# 
#                else:
# #                   slot_l.append(abs(x1-x2))
#                   slot_waist.append(y1)
#            print("slot_waist",slot_waist)
#        c1=int(np.min(slot_waist))
#        c2=int(np.max(slot_waist))
#        print(type(c1))
#        response = []
#        dict = {}
#        dict['index1'] = int(c1)*2
#        dict['index2'] = int(c2)*2
#        dict['index3'] = colSum[int(c1)]
#        dict['index4'] = colSum[int(c2)]
#        print(dict)
#        response.append(dict)
#     else:
#        print("there is no slot")
#        response = []
#        response.append(0)
#        #        return HttpResponse(response, content_type="application/json")
#        
#     return HttpResponse(response, content_type="application/json")

    #####################槽识别canny+houghline法########################
    
    if isVertical == "1":
        colSum = np.sum(result, axis=0)
    else:
        colSum = np.sum(result, axis=1)
#     print(colSum)
    max = np.max(colSum)
    min = np.min(colSum)
#     print("max is:",max)
#     print("min is:",min)
    delta = max - min
    print(delta)
    colSum = ((colSum - min) * 1.0)/delta
    smallIndex = []
    print(tag)
    for i in range(256):
#         print("colsum:",colSum[i])
        if colSum[i] <= 0.6:
            smallIndex.append(i)
    print("the first time samallindex is:",len(smallIndex))
    # if  delta<18 or len(smallIndex)>20:
    if  len(smallIndex)>20:
       print("there is no slot")
       response = []
       response.append(0)
       return HttpResponse(response, content_type="application/json")
    elif len(smallIndex) < 10:
        smallIndex.clear()
        for i in range(256):
            if colSum[i] <= 0.9:
                smallIndex.append(i)
    indexMean = np.mean(smallIndex)
    print(indexMean, len(smallIndex))
    c1 = -1
    c2 = -1
    print(indexMean)
    if(len(smallIndex) > 1):
        c1 = int((np.mean([x for x in smallIndex if x < indexMean]))*2.0)
        c2 = int((np.mean([x for x in smallIndex if x >= indexMean]))*2.0)
    response = []
    dict = {}
    dict['index1'] = c1
    dict['index2'] = c2
    dict['index3'] = colSum[int(c1/2)]
    dict['index4'] = colSum[int(c2/2)]
    print(dict)
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
    print(x, y, r)
    bytes_stream.close()
    return HttpResponse(response, content_type="application/json")
# -----------------------------孔的测试逻辑，返回的是孔的圆心坐标，及半径---------------------
