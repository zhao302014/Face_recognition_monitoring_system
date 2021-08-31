# -*-coding:GBK -*-
# -*-coding:utf-8-*-
import face_recognition
from PIL import Image
import cv2

# ͨ�� load_image_file �������ش�ʶ��ͼƬ
image = face_recognition.load_image_file('E:/jiti.jpg')

# ͨ�� face_locations �õ�ͼ������������λ��
face_locations = face_recognition.face_locations(image)

for face_location in face_locations:
    top, right, bottom, left = face_location   # �ᱨ�������õ�ÿ���������ĸ�λ����Ϣ
    print("��ʶ��������λ����������Ϊ��top{}, right{}, bottom{}, left{}".format(top, right, bottom, left))
    # face_image = image[top:bottom, left:right]
    # pil_image = Image.fromarray(face_image)
    # pil_image.show()
    start = (left, top)
    end = (right, bottom)

    # ��ͼƬ�ϻ��ƾ��ο�
    cv2.rectangle(image, start, end, (0,0,255), thickness=2)

cv2.imshow('window', image)
cv2.waitKey()