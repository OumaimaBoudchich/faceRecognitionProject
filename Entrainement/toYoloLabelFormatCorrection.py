import os
import shutil
import numpy as np
import cv2

if not os.path.isdir('./train_final/'):
	os.mkdir('./train_final/')
	if not os.path.isdir('./train_final/labels/'):
		os.mkdir('./train_final/labels/')
	if not os.path.isdir('./train_final/images/'):
		os.mkdir('./train_final/images/')

if not os.path.isdir('./val_final/'):
	os.mkdir('./val_final/')
	if not os.path.isdir('./val_final/labels/'):
		os.mkdir('./val_final/labels/')
	if not os.path.isdir('./val_final/images/'):
		os.mkdir('./val_final/images/')


with open('./wider_face_split/wider_face_train_bbx_gt.txt', 'r') as labels:
	lines = labels.readlines()

for line in lines:
	if '/' in line:
		skip = False
		print('\nPATH :', line)
		filename = line.split('/')[-1].replace('\n', '')
		#print(filename)
		img = cv2.imread('./WIDER_train/images/' + line.replace('\n', ''), 0)
		try: 
			h, w = img.shape
			shutil.copyfile('./WIDER_train/images/' + line.replace('\n', ''), './train_final/images/' + filename)
		except:
			print('error on', filename)
			skip = True
		
	elif len(line) < 10:
		pass
		#count = int(line.replace('\n', '')) + 1
		#print(count)
	elif not skip:
		features = line.split(' ')
		newX = str(max(min(int(features[0]) / w, 1), 0))
		newY = str(max(min(int(features[1]) / w, 1), 0))
		newW = str(max(min(int(features[2]) / w, 1), 0))
		newH = str(max(min(int(features[3]) / w, 1), 0))

		print('FEATURES :', features)
		#print(newX, newY, newW, newH)
		#rect = cv2.rectangle(img, (int(features[0]), int(features[1])), (int(features[0]) + int(features[2]), int(features[1]) + int(features[3])), 0, 3)
		#cv2.imshow('rect', rect)
		#cv2.waitKey(0)
		
		with open('./train_final/labels/' + filename[:-4] + '.txt', 'a') as label:
			label.write('0' + ' ' + newX + ' ' + newY + ' ' + newW + ' ' + newH + '\n')

"""
with open('./wider_face_split/wider_face_val_bbx_gt.txt', 'r') as labels:
	lines = labels.readlines()

for line in lines:
	if '/' in line:
		skip = False
		print('\nPATH :', line)
		filename = line.split('/')[-1].replace('\n', '')
		#print(filename)
		img = cv2.imread('./WIDER_val/images/' + line.replace('\n', ''), 0)
		try: 
			h, w = img.shape
			shutil.copyfile('./WIDER_val/images/' + line.replace('\n', ''), './val_final/images/' + filename)
		except:
			skip = True
		
	elif len(line) < 10:
		pass
		#count = int(line.replace('\n', '')) + 1
		#print(count)
	elif not skip:
		features = line.split(' ')
		print('FEATURES :', features)
		newX = str(max(min(int(features[0]) / w, 1), 0))
		newY = str(max(min(int(features[1]) / w, 1), 0))
		newW = str(max(min(int(features[2]) / w, 1), 0))
		newH = str(max(min(int(features[3]) / w, 1), 0))
		#print(newX, newY, newW, newH)
		#rect = cv2.rectangle(img, (int(features[0]), int(features[1])), (int(features[0]) + int(features[2]), int(features[1]) + int(features[3])), 0, 3)
		#cv2.imshow('rect', rect)
		#cv2.waitKey(0)
		with open('./val_final/labels/' + filename[:-4] + '.txt', 'a') as label:
			label.write('0' + ' ' + newX + ' ' + newY + ' ' + newW + ' ' + newH + '\n')
"""