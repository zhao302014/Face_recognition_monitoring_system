# -*-coding:GBK -*-
import face_recognition

# ����һ�ź���
image1 = face_recognition.load_image_file('./facelib/yangmi+liukaiwei.jpeg')
# ����һ�ŵ�����
image2 = face_recognition.load_image_file('./facelib/yangmi.jpg')

known_face_encodings = face_recognition.face_encodings(image1)
# face_encodings���ص����б����ͣ�����ֻ��Ҫ�õ���һ���������뼴��
compare_face_encodings = face_recognition.face_encodings(image2)[0]

# ע��ڶ���������ֻ���Ǵ𰸸��沿�������룬���ܴ��б�
matches = face_recognition.compare_faces(known_face_encodings, compare_face_encodings)
print(matches)