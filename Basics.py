import cv2
import numpy as np
import face_recognition


# Taking and converting the images to RGB

# for original image
imgElon = face_recognition.load_image_file('ImagesBasic/elon.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

# for test image
imgTest = face_recognition.load_image_file('ImagesBasic/bill.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)


# Getting face location and face encodings(128 measurements)
faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
# marking a rectangle in the face
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)


faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# we are gonna use linear SVM for checking between these two encodings

# comparing these two images
results=face_recognition.compare_faces([encodeElon],encodeTest)
print(results)

# getting faceDistance and comparing, lower the distance, accurate it will be
faceDis=face_recognition.face_distance([encodeElon],encodeTest)
# printing the comparison of face distances
print(faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)



cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)