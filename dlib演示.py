# -*-coding:GBK -*-
# -*-coding:utf-8-*-
import dlib
from skimage import io

detector = dlib.get_frontal_face_detector()     # ��ȡһ�����������������������������������㷨
win = dlib.image_window()
img = io.imread('E:/girl.png')     # ��ȡ������ͼ��
# ���������������ȡ������ͼ�����ݣ��ڶ�������1�����ȡͼƬ���ز��Ŵ�1���Ա��ܹ��ռ����������Ƭϸ��
# ���ؽ����һ���������������
ders = detector(img, 1)
win.set_image(img)
win.add_overlay(ders)
dlib.hit_enter_to_continue()