# -*-coding:GBK -*-
from scipy.spatial import distance
import dlib
import cv2
from imutils import face_utils

def eye_aspect_ratio(eye):
    '''
    ����EARֵ
    :param eye: �۲�����������
    :return: EARֵ
    '''
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A+B) / (2.0*C)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# �����۾��ݺ�ȵ���ֵ
EAR_THRESH = 0.3
# ���Ǽٶ�����3֡���ϵ�EAR��ֵ��С����ֵ����ȷ���ǲ�����գ�۲���
EAR_CONSEC_FRAMES = 3

# �����������ж�Ӧ�۾����Ǽ�������������
RIGHT_EYE_START = 37-1
RIGHT_EYE_END = 42-1
LEFT_EYE_START = 43-1
LEFT_EYE_END = 48-1

frame_counter = 0  # ����֡�ļ���
blink_counter = 0  # գ�۵ļ���

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # ת��Ϊ�Ҷ�ͼ��
    rects = detector(gray, 1)     # �������

    if len(rects) > 0:
        shape = predictor(gray, rects[0])    # ���������
        points = face_utils.shape_to_np(shape)
        leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]  # ȡ������������
        rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]   # ȡ������������
        # ���������۵�EARֵ
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # ��������EAR��ƽ��ֵ
        ear = (leftEAR+rightEAR) / 2.0

        # ʵ���ж�һ�����������ִ��벢���Ǳ����
        # Ѱ�������۵�����
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        # ��������������
        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # ���EARС����ֵ����ʼ��������֡
        if ear < EAR_THRESH:
            frame_counter += 1
        else:
            if frame_counter >= EAR_CONSEC_FRAMES:
                print('գ�ۼ��ɹ��������')
                frame_counter += 1
                break
            frame_counter = 0

        cv2.putText(frame, "COUNTER: {}".format(frame_counter), (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Blinks: {}".format(blink_counter), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # cv2.putText(frame, 'blink:{}'.format(blink_counter))
    cv2.imshow('window', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('out.jpg', frame)
        break

cap.release()
cv2.destroyAllWindows()