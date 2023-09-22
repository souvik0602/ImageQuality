import argparse, imutils, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
if not sys.warnoptions:
	import warnings
	warnings.simplefilter("ignore")
with warnings.catch_warnings():  
	warnings.filterwarnings("ignore",category=FutureWarning)
	warnings.filterwarnings("ignore", category=DeprecationWarning)
import filetype
from pathlib import Path
from datetime import datetime
import logging
logging.disable(logging.WARNING)
import numpy as np
from Models.brisquequality import test_measure_BRISQUE
#import uuid
import cv2
import time


def qcheck(path):
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
	tempname="temp.jpg"
	cnt_pic=0
	flg=-1
	flg_lst=[]
	cnt_flg1=0
	cnt_flg2=0
	timestr = time.strftime("%Y%m%d-%H%M%S")
	try:
		files= os.listdir(path)
		files.sort(key = len)
		print("Path:  "+path)
		print("No. of Files: "+str(len(files)))
		files_with_path = [os.path.join(path, file) for file in files]
		for pic in files_with_path:
			try:
				cnt_pic+=1
				print(str(cnt_pic)+"/"+str(len(files))+" ...Processing...")
				img = cv2.imread(pic)
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				faces = face_cascade.detectMultiScale(gray, 1.3, 5) #face_detection
				if len(faces)>0:
					for (x,y,w,h) in faces:
						roi_color = img[y:y+h, x:x+w]
						cv2.imwrite(tempname,roi_color)
						
						qscore=test_measure_BRISQUE(tempname) #img_quality_score_generation
						if qscore>=20: 						#20_set_as_current_threshold_face_pics
							flg=1 							#Low_Quality
						else:
							flg=0 							#High_Quality
						
						flg_lst.append(flg)
						with open(timestr+"_f.csv", "a") as file:
							file.write(pic+","+str(qscore)+","+str(flg)+"\n")
				else:
					qscore=test_measure_BRISQUE(pic) #img_quality_score_generation
					if qscore>=6: 						#6_set_as_current_threshold_non_face_pics
						flg=1 							#Low_Quality
					else:
						flg=0 							#High_Quality
						
					flg_lst.append(flg)
					with open(timestr+"_nf.csv", "a") as file:
						file.write(pic+","+str(qscore)+","+str(flg)+"\n")

			except Exception as e1:
				print("Inside: "+e1)
				continue
	
		for x in range(len(flg_lst)):
			if flg_lst[x]==1:
				cnt_flg1+=1
			if flg_lst[x]==0:
				cnt_flg2+=1
		
		print("High Quality/Gan Type Images = "+str(cnt_flg2))
		print("Low Quality/Non-Gan Type Images = "+str(cnt_flg1))
			
	except Exception as e:
		print(e)

if __name__ == '__main__':
	path = sys.argv[1]
	qcheck(path)
	print("END")