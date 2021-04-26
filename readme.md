1. 安装django 及unet项目的相关环境https://github.com/nothasson/unet 如提示其他包未安装，使用pip install 包名  即可
2. 安装好cuda10.1和cudnn
3.  进入views.py修改模型文件路径等
4.  输入 python ./manage.py migrate
5.  输入 python ./manage.py runserver 
6.  等待启动，如未提示错误，则启动成功，就不用去管他了（不要关闭）。
7. 