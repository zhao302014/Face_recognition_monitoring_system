# -*-coding:GBK -*-
# -*-coding:utf-8-*-
import face_recognition
import cv2

unknown_image = face_recognition.load_image_file('E:/yangmi_liukaiwei.jpeg')
known_image = face_recognition.load_image_file('E:/yangmi.jpg')

results = []

known_face_encoding = face_recognition.face_encodings(known_image)[0]
unknown_face_encodings = face_recognition.face_encodings(unknown_image)
face_locations = face_recognition.face_locations(unknown_image)

for i in range(len(face_locations)):    # face_locations的长度就代表有多少张脸
    top, right, bottom, left = face_locations[i]
    face_image = unknown_image[top:bottom, left:right]
    face_encoding = face_recognition.face_encodings(face_image)
    if face_encoding:
        result = {}
        matches = face_recognition.compare_faces([unknown_face_encodings[i]], known_face_encoding, tolerance=0.39)
        if True in matches:
            print('在未知图片中找到已知面孔')
            result['face_encoding'] = face_encoding
            result['is_view'] = True
            result['location'] = face_locations[i]
            result['face_id'] = i+1
            results.append(result)

            if result['is_view']:
                print('已知面孔匹配照片上的第{}张脸！'.format(result['face_id']))

view_face_locations = [i['location'] for i in results if i['is_view']]

if len(view_face_locations) > 0:
    for location in view_face_locations:
        top, right, bottom, left = location
        start = (left,top)
        end = (right, bottom)

        cv2.rectangle(unknown_image, start, end, (0,0,255), thickness=2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(unknown_image, 'yangmi', (left+6, bottom+26), font, 1.0, (255,255,255), thickness=1)

cv2.imshow('window', unknown_image)
cv2.waitKey()
