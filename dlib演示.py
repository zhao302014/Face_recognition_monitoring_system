# -*-coding:GBK -*-
# -*-coding:utf-8-*-
import dlib
from skimage import io

detector = dlib.get_frontal_face_detector()     # 获取一个脸部检测器，这个监测器包含了脸部检测算法
win = dlib.image_window()
img = io.imread('E:/girl.png')     # 读取带辨别的图像
# 利用脸部检测器读取待检测的图像数据，第二个参数1代表读取图片像素并放大1倍以便能够收集到更多的照片细节
# 返回结果是一组人脸区域的数据
ders = detector(img, 1)
win.set_image(img)
win.add_overlay(ders)
dlib.hit_enter_to_continue()