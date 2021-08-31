# -*-coding:GBK -*-
import face_recognition
import os
import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import datetime
import threading

class Recorder:
    pass

record_dic = {}
unknown_pic = []

flag_over = 0  # 定义一个是否进行来访记录的标记
# 定时去保存对比图像信息，并且将位置人员的图像保存下来
def save_recorder(name, frame):
    global record_dic
    global flag_over
    global unknown_pic

    if flag_over == 1: return
    try:
        record = record_dic[name]
        seconds_diff = (datetime.datetime.now() - record.times[-1]).total_seconds()

        if seconds_diff < 60 * 10:
            return
        record.times.append(datetime.datetime.now())
        print('更新记录', record_dic, record.times)
    except KeyError:
        newRec = Recorder()
        newRec.times = [datetime.datetime.now()]
        record_dic[name] = newRec
        print('添加记录', record_dic, newRec.times)

    if name == '未知头像':
        s = str(record_dic[name].times[-1])
        # print(s)
        # 未知人员的图片名称
        filename = s[:10]+s[-6:] + '.jpg'
        cv2.imwrite(filename, frame)
        unknown_pic.append(filename)



# 解析已有人员的所有照片并得到照片名和人物面部编码信息
def load_img(path):
    print('正在加载已知人员的图片...')

    for dirpath, dirnames, filenames in os.walk(path):
        print(filenames)
        facelib = []

        for filename in filenames:
            filepath = os.sep.join([dirpath, filename])
            # 把对应每张图片加载进来
            face_image = face_recognition.load_image_file(filepath)
            face_encoding = face_recognition.face_encodings(face_image)[0]
            facelib.append(face_encoding)

        return facelib,filenames


facelib, facenames = load_img('facelib')
# print(facenames)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    # 通过缩小图片（缩小为1/4），提高对比效率
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]  # 将opencv的BGR格式转换为RGB格式

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    # 循环多张人脸
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(facelib, face_encoding, tolerance=0.39)
        name = '未知头像'
        if True in matches:
            # 如果摄像头里面的头像匹配了已知人物头像，则取出第一个True的位置
            first_match_index = matches.index(True)
            name = facenames[first_match_index][:-4]   # 取出文件上对应的人名
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):

        # 还原原图片大小
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), thickness=2)  # 标注人脸信息
        img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        font = ImageFont.truetype('simhei.ttf', 40)
        draw = ImageDraw.Draw(img_PIL)
        draw.text((left+6, bottom-6), name, font=font, fill=(255,255,255))
        frame = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
        save_recorder(name, frame)

    cv2.imshow('capture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
