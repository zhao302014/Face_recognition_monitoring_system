# -*-coding:GBK -*-
# -*-coding:utf-8-*-
import face_recognition

image = face_recognition.load_image_file('E:/boys.jpg')

# 不管图像中有多少个人脸信息，返回值都是一个列表
face_encodings = face_recognition.face_encodings(image)
for face_encoding in face_encodings:
    print("信息编码长度为：{}\n编码信息为：{}".format(len(face_encoding), face_encoding))