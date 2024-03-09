import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path = 'ImagesAttendance'

images = []
classNames = []

# reading all the directories/files inside our path
myList=os.listdir(path)

print(myList)
for cl in myList:
    currImg=cv2.imread(f'{path}/{cl}')
    # reading and storing all the images in the images array
    images.append(currImg)
    # creating separate classes for each person and adding their names to the classNames array

    classNames.append(os.path.splitext(cl)[0])

print(classNames)

# a function for determining and storing all the encodings for each image
def findEncodings(images):
    encodeList = []
    # traversing through all the stored images and coverting them to rgb and then calculating their encodings, pushing them in the array and then appending each encoding in the encodeList and then returning the list
    for img in images:
        img=imgElon = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode);
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        # print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dateTimeString=now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dateTimeString}')
# markAttendance('Elon')
# this is the list of all the known person's encodings
encodeListKnown = findEncodings(images)
print('Encoding Completed')

# initialising webcam for test image
cap = cv2.VideoCapture(0)

# while loop to get each image frame by frame
while True:
    success,img = cap.read();
    # we want to reduce the image size
    imgS = cv2.resize(img,(0,0),None, 0.25, 0.25)
    # converting the image to RGB
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    # There can be multiple people on the screen, so getting all the the face locations and then passing it to get the encodings
    facesCurrFrame = face_recognition.face_locations(imgS)
    encodesCurrFrame = face_recognition.face_encodings(imgS,facesCurrFrame)
    # matching the faces -> we will iterate through all the faces found in our curr frame and we will compare all these faces and encodings that we found before and stored

    for encodeFace, faceLoc in zip(encodesCurrFrame,facesCurrFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        # the image with lowest distance will be our best matched answer, this will be an array based on comparing with each stored face registration we have
        print(faceDis)
        # it will give the index of the best match with lowest face distance value (cause lowest face distance-> means best match)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1= faceLoc
            y1,x2,y2,x1= y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('Attendance',img)
    cv2.waitKey(1)