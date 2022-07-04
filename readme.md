1. conda create --name django python==3.7.12
2. conda activate django
3. conda install tensorflow-gpu==2.1.0
4. conda install h5py==2.10.0
5. pip install keras==2.3.1
6. pip install scikit-image opencv-python setproctitle
7. pip install django==3.2.7
8. git clone https://github.com/BigBooks/django4measure
9. 进入views.py修改40,41行的模型文件路径
10. 执行 python ./manage.py migrate
```bash
    # output
    1/1 [==============================] - 0s 384ms/sample
    load and test slot model successfully
    Operations to perform:
    Apply all migrations: admin, auth, contenttypes, sessions
    Running migrations:
    No migrations to apply.
```
11. 执行 python ./manage.py runserver
```bash
    # output
    July 04, 2022 - 09:42:02
    Django version 3.2.7, using settings 'measure_tensorflow.settings'
    Starting development server at http://127.0.0.1:8000/
    Quit the server with CTRL-BREAK.
```
12. 等待启动，如未提示错误，则启动成功，就不用去管他了（不要关闭）。
