# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:37:51 2023

@author: allen
"""
# import the opencv library 
import cv2 
  
  
# define a video capture object 
vid = cv2.VideoCapture(1) 
if(vid.isOpened()):
    print ("open")
else:
    print("fail")
          
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    
    cap.release()
cv2.destroyAllWindows()