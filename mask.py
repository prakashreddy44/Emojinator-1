#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 23:23:12 2019

@author: pranav
"""
import numpy as np
import cv2

def mask(shape,img):
  mask = np.zeros((400, 600), dtype=np.uint8)  
  right_eye_centre=(int((shape[36,0]+shape[37,0]+shape[38,0]+shape[39,0]+shape[40,0]+shape[41,0])/6),int((shape[36,1]+shape[37,1]+shape[38,1]+shape[39,1]+shape[40,1]+shape[41,1])/6))
  right_eye_radius=(3*(shape[39,0]-shape[36,0])/4)+(shape[37,1]-shape[41,1])/4
  left_eye_centre=(int((shape[42,0]+shape[43,0]+shape[44,0]+shape[45,0]+shape[46,0]+shape[47,0])/6),int((shape[42,1]+shape[43,1]+shape[44,1]+shape[45,1]+shape[46,1]+shape[47,1])/6))
  left_eye_radius=(3*(shape[39,0]-shape[36,0])/4)+(shape[37,1]-shape[41,1])/4
  sum_x=shape[48,0]+shape[49,0]+shape[50,0]+shape[51,0]+shape[52,0]+shape[53,0]+shape[54,0]+shape[55,0]+shape[56,0]+shape[57,0]+shape[58,0]+shape[59,0]+shape[60,0]
  sum_y=shape[48,1]+shape[49,1]+shape[50,1]+shape[51,1]+shape[52,1]+shape[53,1]+shape[54,1]+shape[55,1]+shape[56,1]+shape[57,1]+shape[58,1]+shape[59,1]+shape[60,1]
  mouth_centre=(int(sum_x/13),int(sum_y/13))
  mouth_radius=((shape[54,0]-shape[48,0])*.4)+(shape[57,1]-shape[52,1])*.4
  #left_eyebrow=(17,18,19,20,21)
  #right_eyebrow = (22, 24, 25, 26, 26)
  nose = (27, 31, 33, 35)
  #left_e = [shape[point] for point in left_eyebrow]
  #right_e = [shape[point] for point in right_eyebrow]
  nose_p = [shape[point] for point in nose]
  #left_e=np.array(left_e)
  #right_e=np.array(right_e)
  nose_p=np.array(nose_p)
  
 # mask=cv2.fillPoly(mask,[left_e],255)
 # mask=cv2.fillPoly(mask,[right_e],255)
  mask=cv2.fillPoly(mask,[nose_p],255)
  mask=cv2.circle(mask,right_eye_centre,int(right_eye_radius),255,-1)
  mask=cv2.circle(mask,left_eye_centre,int(left_eye_radius),255,-1)
  mask=cv2.circle(mask,mouth_centre,int(mouth_radius),255,-1)
  return mask