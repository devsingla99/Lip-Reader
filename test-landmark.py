#Import required modules
import cv2
import dlib
import faceAlignment as fa
import numpy as np
import sys
import select
#Set up some required objects
video_capture = cv2.VideoCapture(0) #Webcam object
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file

while True:
    ret, frame = video_capture.read()
    #frame = cv2.resize(frame, (512, 256))#resize
    frame = cv2.resize(frame, (1024, 512))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)

    detections = detector(clahe_image, 1) #Detect the faces in the image

    for k,d in enumerate(detections): #For each detected face
        shape = predictor(clahe_image, d) #Get coordinates
        for i in range(1,68): #There are 68 landmark points on each face
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
           

    key= 0xFF & cv2.waitKey(1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if key & 0xFF == 27: #Exit program when the user presses 'esc'
        break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    elif key==ord('a'):
        cv2.putText(frame,'processing...',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('1'):
        cv2.putText(frame,'Stop Navigation',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('2'):
        cv2.putText(frame,'Excuse me',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('3'):
        cv2.putText(frame,'I am Sorry',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('4'):
        cv2.putText(frame,'Thank you',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('5'):
        cv2.putText(frame,'Good Bye',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('6'):
        cv2.putText(frame,'I love this game',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('7'):
        cv2.putText(frame,'Nice to meet you',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('8'):
        cv2.putText(frame,'You are Welcome',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('9'):
        cv2.putText(frame,'How are You?',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('0'):
        cv2.putText(frame,'Have a good time',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('q'):
        cv2.putText(frame,'Begin',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('w'):
        cv2.putText(frame,'Choose',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('e'):
        cv2.putText(frame,'Connection',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('r'):
        cv2.putText(frame,'Navigation',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('t'):
        cv2.putText(frame,'Next',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('y'):
        cv2.putText(frame,'Previous',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('u'):
        cv2.putText(frame,'Start',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('i'):
        cv2.putText(frame,'Stop',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('o'):
        cv2.putText(frame,'Hello',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    elif key==ord('p'):
        cv2.putText(frame,'Web',(50,50),font,1,(255,0,0),2,cv2.LINE_4)
    
    
    cv2.imshow("origin", frame) 
    

# When everything done, release the capture
video_capture.release()
cv2.destroyAllWindows()