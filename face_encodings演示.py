# -*-coding:GBK -*-
# -*-coding:utf-8-*-
import face_recognition

image = face_recognition.load_image_file('E:/boys.jpg')

# ����ͼ�����ж��ٸ�������Ϣ������ֵ����һ���б�
face_encodings = face_recognition.face_encodings(image)
for face_encoding in face_encodings:
    print("��Ϣ���볤��Ϊ��{}\n������ϢΪ��{}".format(len(face_encoding), face_encoding))