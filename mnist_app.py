import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_forward
import mnist_backward
from flask import Flask
from redis import Redis, RedisError
from redis import StrictRedis
from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField,SubmitField     #导入字符串字段，密码字段，提交字段
from wtforms.validators import DataRequired,ValidationError
from wtforms.validators import Required
from flask import render_template
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash
import os
import socket
import flask_login
from werkzeug.utils import secure_filename
import datetime as d

# Connect to Redis
redis = StrictRedis(host='localhost', port=6379, db=0)


app = Flask(__name__)


# 定义加载使用模型进行预测的函数
def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:

        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        preValue = tf.argmax(y, 1)
        # 加载滑动平均模型
        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:

            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 恢复当前会话,将ckpt中的值赋值给w和b
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 执行图计算
                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1


# 图片预处理函数
def pre_pic(picName):
    # 先打开传入的原始图片
    img = Image.open(picName)
    # 使用消除锯齿的方法resize图片
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    # 变成灰度图，转换成矩阵
    im_arr = np.array(reIm.convert("L"))
    threshold = 50  # 对图像进行二值化处理，设置合理的阈值，可以过滤掉噪声，让他只有纯白色的点和纯黑色点
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    # 将图像矩阵拉成1行784列，并将值变成浮点型（像素要求的仕0-1的浮点型输入）
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)

    return img_ready


def application(num,path):
    # input函数可以从控制台接受数字
    #testNum = int(input("input the number of test images:"))
    testNum = num
    re = 999
    # 使用循环来历遍需要测试的图片才结束
    for i in range(testNum):
        # input可以实现从控制台接收字符格式,图片存储路径
        #testPic = input("the path of test picture:")
        testPic = path
        print('test1')
        # 将图片路径传入图像预处理函数中
        testPicArr = pre_pic(testPic)
        print('test2')
        # 将处理后的结果输入到预测函数最后返回预测结果
        preValue = restore_model(testPicArr)
        print("The prediction number is :", preValue)
        re = preValue
    return re


UPLOAD_FOLDER = './imgs'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
re = ''
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global re
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            re = application(1, UPLOAD_FOLDER+'/'+filename)
            redis.sadd(d.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"), d.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"), filename, str(re[0]))
            print(redis.smembers(d.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return render_template('index.html', re=re)


if __name__ == "__main__":
    # main()
    app.run()

