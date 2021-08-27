import cv2
from random import randrange

#img = cv2.imread('ElonMusk.jpg')

#cv2.imshow('Elon Musk',img)

webcam=cv2.VideoCapture(0) #0->default cam

trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    successful_frame_read, frame =webcam.read()
    #successful_frame_read, frame =webcam.read('video.mp4')
    grayimg =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayimg)
    for [x,y,w,h] in face_coordinates:
         cv2.rectangle(frame,(x ,y),(x+w,  y+w), (randrange(126,256),randrange(126,256),randrange(126,256)),2)
    cv2.imshow('face detector',frame)
    key=cv2.waitKey(1) #one frame for 1millisec
    if key==81 or key==113:
        break
webcam.release()

#cv2.imshow('Elon Musk',grayimg)

#checks all the training set scales with the given image & find a match(sliding)

face_coordinates = trained_face_data.detectMultiScale(grayimg) #return a rectangle with coordinates(x,y,w,h) around the detected face

print(face_coordinates)
[x,y,w,h] = face_coordinates[0] #returns a list of list and x,y will be upper left cordinates of the rectangle

cv2.rectangle(img,(135 ,127),(135+190,  127+190), (0,255,0),2)
for [x,y,w,h] in face_coordinates:
    cv2.rectangle(img,(x ,y),(x+w,  y+w), (randrange(126,256),randrange(126,256),randrange(126,256)),2)
print(face_coordinates)

cv2.imshow('Elon Musk',img)


cv2.waitKey() #any key press will terminate the event( allows you to wait for a specific time in milliseconds until you press any button on the keyword. It accepts time in milliseconds as an argument.ex:cv2.waitKey(6000))
