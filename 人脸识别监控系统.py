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

flag_over = 0  # ����һ���Ƿ�������ü�¼�ı��
# ��ʱȥ����Ա�ͼ����Ϣ�����ҽ�λ����Ա��ͼ�񱣴�����
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
        print('���¼�¼', record_dic, record.times)
    except KeyError:
        newRec = Recorder()
        newRec.times = [datetime.datetime.now()]
        record_dic[name] = newRec
        print('��Ӽ�¼', record_dic, newRec.times)

    if name == 'δ֪ͷ��':
        s = str(record_dic[name].times[-1])
        # print(s)
        # δ֪��Ա��ͼƬ����
        filename = s[:10]+s[-6:] + '.jpg'
        cv2.imwrite(filename, frame)
        unknown_pic.append(filename)



# ����������Ա��������Ƭ���õ���Ƭ���������沿������Ϣ
def load_img(path):
    print('���ڼ�����֪��Ա��ͼƬ...')

    for dirpath, dirnames, filenames in os.walk(path):
        print(filenames)
        facelib = []

        for filename in filenames:
            filepath = os.sep.join([dirpath, filename])
            # �Ѷ�Ӧÿ��ͼƬ���ؽ���
            face_image = face_recognition.load_image_file(filepath)
            face_encoding = face_recognition.face_encodings(face_image)[0]
            facelib.append(face_encoding)

        return facelib,filenames


facelib, facenames = load_img('facelib')
# print(facenames)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    # ͨ����СͼƬ����СΪ1/4������߶Ա�Ч��
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]  # ��opencv��BGR��ʽת��ΪRGB��ʽ

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    # ѭ����������
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(facelib, face_encoding, tolerance=0.39)
        name = 'δ֪ͷ��'
        if True in matches:
            # �������ͷ�����ͷ��ƥ������֪����ͷ����ȡ����һ��True��λ��
            first_match_index = matches.index(True)
            name = facenames[first_match_index][:-4]   # ȡ���ļ��϶�Ӧ������
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):

        # ��ԭԭͼƬ��С
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), thickness=2)  # ��ע������Ϣ
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
