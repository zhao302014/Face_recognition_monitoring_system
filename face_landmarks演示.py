# -*-coding:GBK -*-
# -*-coding:utf-8-*-
import face_recognition
from PIL import Image, ImageDraw

image = face_recognition.load_image_file('E:/boys.jpg')

face_landmarks_list = face_recognition.face_landmarks(image)

pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)   # 生成一张PIL图像

for face_landmarks in face_landmarks_list:
    facial_features = [
        'chin',
        'left_eyebrow',
        'right_eyebrow',
        'nose_bridge',
        'nose_tip',
        'left_eye',
        'right_eye',
        'bottom_lip'
    ]
    for facial_feature in facial_features:
        # print("每个人的面部特征显示在以下为位置：{}".format(facial_feature))
        d.line(face_landmarks[facial_feature], width=5)   # 直接调用PIL中的line方法在PIL图像中绘制线条，帮助我们观察特征点

pil_image.show()