from django.http import HttpResponse
import tensorflow.keras as keras
from . import data
import os
import time
from django.views.decorators.csrf import csrf_exempt
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import glob

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
##提前加载好，以后预测的时候就速度快了
model = keras.models.load_model(r"D:\i3c\unet_model.hdf5")
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
##提前加载好，以后预测的时候就速度快了
@csrf_exempt
def slot(request):
    # pic = request.FILES.get('pic')
    img = io.imread("./test.png", as_gray=True)
    img = img / 255
    img = trans.resize(img, (256, 256))
    img = np.reshape(img, img.shape + (1,)) if (not False) else img
    img = np.reshape(img, (1,) + img.shape)
    result = model.predict(img, verbose=1)
    result = result[0, :, :, :]
    result = result[:,:,0]
    # img = labelVisualize(result, COLOR_DICT,
    #                      result) if False else result[:, :, 0]
    print(result.shape)
    # io.imsave("./result.png",img)
    return HttpResponse(img)


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
