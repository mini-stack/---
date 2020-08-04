import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np

# 人脸
face_classifier = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")
# 表情
expression_net = load_model('model/Emotion_little_vgg.h5')
# 年龄
age_net = cv2.dnn.readNetFromCaffe('model/deploy_age.prototxt', 'model/age_net.caffemodel')
# 性别
gender_net = cv2.dnn.readNetFromCaffe('model/deploy_gender.prototxt', 'model/gender_net.caffemodel')
# 五种情绪   生气、高兴、没有表情、伤心、惊讶
expression_list = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
# 年龄
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
# 性别  male:男性   femle:女性
gender_list = ['Male', 'Female']
font = cv2.FONT_HERSHEY_SIMPLEX
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # 更改画面效果
    # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    emotion = np.array([[0, 0, 0, 0, 0, 0, 0]])
    cv2.putText(frame, "the number of people:%d" % len(faces), (30, 70), font, 1.2, (0, 0, 255), 2)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        face_img = frame[y:y + h, h:h + w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        # print("Gender : " + gender)

        # Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        # print("Age Range: " + age)

        overlay_text = "%s %s" % (gender, age)
        cv2.putText(frame, overlay_text, (x, y + h + 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Predict expression
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            # 根据ROI（region of interest感兴趣区域）预测表情结果
            preds = expression_net.predict(roi)[0]
            label = expression_list[preds.argmax()]  # label就是表情结果字符串
            # 显示结果到窗口里
            cv2.putText(frame, label, (x, y - 20), font, 2, (0, 255, 255), 3)
        else:
            cv2.putText(frame, 'No Face Found', (400, 400), font, 2, (0, 255, 0), 3)

    cv2.imshow('Camera', frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    if key == ord("s"):
        cv2.imwrite("asd.png", frame)
cap.release()
cv2.destroyAllWindows()
