import os
import cv2

folder = './val_final/'
folder_images = folder + 'images/'
folder_labels = folder + 'labels/'

for filename in os.listdir(folder_images)[1000:1002]:
	print('see image :', filename)
	img = cv2.imread(folder_images + filename)
	h, w, _ = img.shape
	with open(folder_labels + filename[:-4] + '.txt', 'r') as labels:
			lines = labels.readlines()
	for line in lines:

		points = line.split(' ')
		points[1] = int(w * float(points[1].replace('\n', '')))
		points[2] = int(h * float(points[2].replace('\n', '')))
		points[3] = int(w * float(points[3].replace('\n', '')))
		points[4] = int(h * float(points[4].replace('\n', '')))
		#points[1] = 140
		#points[2] = 345
		#points[3] = 26
		#points[4] = 36
		print(points)
		img = cv2.rectangle(img, (points[1], points[2]), (points[1] + points[3], points[2] + points[4]), (255, 0, 0), 5)
		cv2.imshow(filename, img)
		cv2.waitKey(0)