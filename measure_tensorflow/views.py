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

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# 提前加载好，以后预测的时候就速度快了
model = keras.models.load_model(r"D:\i3c\unet_model.hdf5")  ##两个模型文件的位置，最好写绝对路径吧
model.load_weights(r"D:\i3c\unet_weight.hdf5")
print("load successful")

config = ConfigProto()
config.gpu_options.allow_growth = True  # GPU内存增长
session = InteractiveSession(config=config)

img = io.imread("./0.png", as_gray=True)
img = img / 255
img = trans.resize(img, (256, 256))
img = np.reshape(img, img.shape + (1,)) if (not False) else img
img = np.reshape(img, (1,) + img.shape)
result = model.predict(img, verbose=1)
# 提前加载好，以后预测的时候就速度快了


@csrf_exempt
def slot(request):
    isVertical = request.POST.get('isVertical')
    # print(type(isVertical))
    stringdata = request.POST.get('pic')
    img_bytes = base64.b64decode(stringdata)
    bytes_stream = BytesIO(img_bytes)  
    # io.imsave("./a.png", bytes_stream)
    img = io.imread(bytes_stream)  ##base64转图片
    bytes_stream.close()
    # io.imsave("./aa.png",img)  #传过来了~
    img = img / 255
    print(type(img)) 
    img = trans.resize(img, (256, 256))
    img = np.reshape(img, img.shape + (1,)) if (not False) else img
    img = np.reshape(img, (1,) + img.shape)
    result = model.predict(img, verbose=1)
    result = result[0, :, :, 0]
    result = np.float32(result)
    # io.imsave("./resultaa.png",result)  ## 也确实预测出来了
    if isVertical == "1":
        colSum = np.sum(result,axis=0)
    else:
        colSum = np.sum(result,axis=1)
    # print(colSum)
    max = np.max(colSum)
    min = np.min(colSum)
    delta = max - min
    colSum = ((colSum - min) * 1.0)/delta
    smallIndex = []
    for i in range(256):
        if colSum[i] <=0.6:
            smallIndex.append(i)
    indexMean = np.mean(smallIndex)
    print(indexMean,len(smallIndex))
    c1 = -1
    c2 = -1
    print(indexMean)
    if(len(smallIndex)>1):
        c1 = int(np.mean([x for x in smallIndex if x < indexMean]))*2
        c2 = int(np.mean([x for x in smallIndex if x >= indexMean]))*2
    response = []
    dict = {}
    dict['index1'] = c1
    dict['index2'] = c2
    response.append(dict)
    return HttpResponse(response, content_type="application/json")


# def params_post(request):
#     if request.method=='GET':
#         return render(request,'post.html')
#     else:
#         myFile = request.FILES.get("pic", None)
#         destination = open(os.path.join(BASE_DIR, myFile.name), 'wb+')  # 打开特定的文件进行二进制的写操作
#         for chunk in myFile.chunks():  # 分块写入文件
#             destination.write(chunk)
#         destination.close()
#         username=request.POST.get('username','')
#         password=request.POST.get('password','')
#         return HttpResponse('username='+username+"&password="+password)
