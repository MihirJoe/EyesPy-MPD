# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:06:53 2023

@author: allen
"""
import cv2  
img = cv2.imread('dog.JPG')
edges = cv2.Canny(img,200,300,True)
cv2.imshow("Edge Detected Image", edges)  
cv2.imshow("Original Image", img)  
cv2.waitKey(0)  # waits until a key is pressed  
cv2.destroyAllWindows()  # destroys the window showing image