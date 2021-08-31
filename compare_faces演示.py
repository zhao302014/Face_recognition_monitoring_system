# -*-coding:GBK -*-
import face_recognition

# 加载一张合照
image1 = face_recognition.load_image_file('./facelib/yangmi+liukaiwei.jpeg')
# 加载一张单人照
image2 = face_recognition.load_image_file('./facelib/yangmi.jpg')

known_face_encodings = face_recognition.face_encodings(image1)
# face_encodings返回的是列表类型，我们只需要拿到第一个人脸编码即可
compare_face_encodings = face_recognition.face_encodings(image2)[0]

# 注意第二个参数，只能是答案个面部特征编码，不能传列表
matches = face_recognition.compare_faces(known_face_encodings, compare_face_encodings)
print(matches)