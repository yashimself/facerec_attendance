import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime, time

path = '.resources/prefaces'
postpath = '.resources/facescap'
images = []  # LIST CONTAINING ALL THE IMAGES
className = []  # LIST CONTAINING ALL THE CORRESPONDING CLASS Names
myList = os.listdir(path)
UnknownCount = 0
print("Total Classes Detected:", len(myList))
for x, cl in enumerate(myList):
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        # Convert from BGR to RGB. OpenCV imports a picture as BGR, but the face_recognition library reads only as RGB.
        # So, to make the picture readable by face_recognition library, convert the picture from BGR to RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def record(name):
    today = datetime.today()
    today = today.strftime("%d-%m-%Y")
    fn = today + '.csv'
    if not os.path.exists(fn):
        open(fn, 'a').close()
        with open(fn, 'w') as f:
            f.write('Name,Time')

    with open(fn, 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            timern = now.strftime("%H:%M:%S")
            cv2.imwrite((postpath + '/' + (name + "_" + timern)) + ".jpg", img)
            f.writelines(f'\n{name},{timern}')


encodeListKnown = findEncodings(images)
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # returns top, right, bottom, left of the face locations
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    # TODO
    # Work out logic for count of unknowns
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < 0.6:
            name = className[matchIndex].upper()
            record(name)
        else:
            name = 'Unknown'
        # print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        record(name)
    cv2.imshow('Face', img)
    cv2.waitKey(1)
