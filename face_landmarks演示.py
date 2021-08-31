# -*-coding:GBK -*-
# -*-coding:utf-8-*-
import face_recognition
from PIL import Image, ImageDraw

image = face_recognition.load_image_file('E:/boys.jpg')

face_landmarks_list = face_recognition.face_landmarks(image)

pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)   # ����һ��PILͼ��

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
        # print("ÿ���˵��沿������ʾ������Ϊλ�ã�{}".format(facial_feature))
        d.line(face_landmarks[facial_feature], width=5)   # ֱ�ӵ���PIL�е�line������PILͼ���л����������������ǹ۲�������

pil_image.show()