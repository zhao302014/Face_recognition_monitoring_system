# -*-coding:GBK -*-
# -*-coding:utf-8-*-
import face_recognition
from PIL import Image
import cv2

# 通过 load_image_file 方法加载待识别图片
image = face_recognition.load_image_file('E:/jiti.jpg')

# 通过 face_locations 得到图像中所有人脸位置
face_locations = face_recognition.face_locations(image)

for face_location in face_locations:
    top, right, bottom, left = face_location   # 结报操作，得到每张人脸的四个位置信息
    print("已识别到人脸部位，限速区域为：top{}, right{}, bottom{}, left{}".format(top, right, bottom, left))
    # face_image = image[top:bottom, left:right]
    # pil_image = Image.fromarray(face_image)
    # pil_image.show()
    start = (left, top)
    end = (right, bottom)

    # 在图片上绘制矩形框
    cv2.rectangle(image, start, end, (0,0,255), thickness=2)

cv2.imshow('window', image)
cv2.waitKey()