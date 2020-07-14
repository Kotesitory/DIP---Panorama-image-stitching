import cv2
import numpy as np
import math
import sys
from os import listdir

def resize_image(img):
	while(img.shape[0] > 800 or img.shape[1] > 800):
		img = cv2.pyrDown(img)

	return img

def stitch_left(left_list, direction = 'right'):
	a = left_list[0]
	for b in left_list[1:]:
		H, pt1, pt2 = findHomographyFromImages(a, b)
		xh = np.linalg.inv(H)
		ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]));
		ds = ds/ds[-1]
		f1 = np.dot(xh, np.array([0,0,1]))
		f1 = f1/f1[-1]
		xh[0][-1] += abs(f1[0])
		xh[1][-1] += abs(f1[1])
		ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
		offsety = abs(int(f1[1]))
		offsetx = abs(int(f1[0]))
		dsize = (abs(int(ds[0]))+offsetx, abs(int(ds[1])) + offsety)
		dsize = (dsize[0]*2, dsize[1]*2)
		tmp = cv2.warpPerspective(a, xh, dsize)
		offset = (offsety, offsetx)
		a = stitch_two_images(b, tmp, offset=offset)

	return tmp

def stitch_right(leftimage, right_list):
	for each in right_list:
		H, pt1, pt2 = findHomographyFromImages(leftimage, each)
		txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
		txyz = txyz/txyz[-1]
		dsize = (int(txyz[0])+leftimage.shape[1], int(txyz[1])+leftimage.shape[0])
		dsize = (dsize[0]*2, dsize[1]*2)
		tmp = cv2.warpPerspective(each, H, dsize)
		leftimage = stitch_two_images(leftimage, tmp)

	return leftimage

### Secong image shape must be bigger than the first in both axis
def stitch_two_images(img1, img2, offset = (0,0)):
	if img1.shape[0] > img2.shape[0] or img1.shape[1] > img2.shape[1]:
		raise ValueError('Shape of first image is bigger that shape of the second image {} <=> {}'.format(img1.shape, img2.shape))

	ret, mask = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY)
	print(img1.shape, img2.shape)
	tmp = img2[offset[0]:img1.shape[0] + offset[0], offset[1]:img1.shape[1] + offset[1]]
	mask = cv2.bitwise_not(mask)
	print(mask.shape, tmp.shape)
	tmp = cv2.bitwise_and(tmp, mask)
	tmp = cv2.bitwise_or(img1, tmp)
	img2[offset[0]:img1.shape[0] + offset[0], offset[1]:img1.shape[1] + offset[1]] = tmp
	img2 = crop_image(img2)
	return img2

def crop_image(img):
	tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
	_, contours, hierarchy = cv2.findContours(mask, 1, 2)
	boundingBox = cv2.boundingRect(contours[0])
	img =  img[boundingBox[1]:boundingBox[1]+boundingBox[3], boundingBox[0]:boundingBox[0]+boundingBox[2]]
	return img


def findHomographyFromImages(img1, img2):
	im1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	im2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	descriptor = cv2.xfeatures2d.SIFT_create()
	keypoints1, descriptors1 = descriptor.detectAndCompute(im1Gray, None)
	keypoints2, descriptors2 = descriptor.detectAndCompute(im2Gray, None)
	bf = cv2.BFMatcher(crossCheck=False)
	matches = bf.knnMatch(descriptors1, descriptors2, k=2)
	
	# Testing quality of kp matches
	good = []
	for m, n in matches:
		if m.distance < 0.75 * n.distance:
			good.append(m)

	# Taking best 25 matches
	good = sorted(good,key = lambda x:x.distance)
	good = good[:25]

	# Extract location of good matches
	points1 = np.array([keypoints1[good_match.queryIdx].pt for good_match in good], dtype=np.float32)
	points1 = points1.reshape((-1, 1, 2))
	points2 = np.array([keypoints2[good_match.trainIdx].pt for good_match in good], dtype=np.float32)
	points2 = points2.reshape((-1, 1, 2))

	# Find homography
	H, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
	return H, points1, points2

""" 
	Alternative image stitching alorighm when the final image-size is known
	Authors:
		- Saurabh Kemekar
		- Arihant Gaur
		- Pranav Patil
		- Danish Gada
	
	Source: https://github.com/saurabhkemekar/PANORAMA
"""
def image_stiching(img1,img2):
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    flag = np.array((gray1,gray2))
    indx = np.argmax(flag,axis=0)
    ind1 = np.where(indx ==0)
    ind2 = np.where(indx==1)
    img = np.zeros_like(img1)
    img[ind1] = img1[ind1]
    img[ind2] = img2[ind2]
    return  img

"""
	Author: Roy Shilkrot
	Source: https://gist.github.com/royshil/0b21e8e7c6c1f46a16db66c384742b2b#file-cylindricalwarping-py
"""
def cylindrical_warp(img,K):
    foc_len = (K[0][0] + K[1][1])/2
    cylinder = np.zeros_like(img)
    temp = np.mgrid[0:img.shape[1],0:img.shape[0]]
    x,y = temp[0],temp[1]
    color = img[y,x]
    theta= (x- K[0][2])/foc_len # angle theta
    h = (y-K[1][2])/foc_len # height
    p = np.array([np.sin(theta),h,np.cos(theta)])
    p = p.T
    p = p.reshape(-1,3)
    image_points = K.dot(p.T).T
    points = image_points[:,:-1]/image_points[:,[-1]]
    points = points.reshape(img.shape[0],img.shape[1],-1)
    cylinder = cv2.remap(img, (points[:, :, 0]).astype(np.float32), (points[:, :, 1]).astype(np.float32), cv2.INTER_LINEAR)
    return cylinder

"""
	Images need to be provided in a left to right order in a txt file
	each new line is a direct or relative path to an image.
	Limited to about 8 images because the final imagesize and computation
	time will be huge.
"""
def read_images(txt_path):
	fp = open(txt_path, 'r')
	img_names = np.array([img_path.rstrip('\r\n') for img_path in fp.readlines()])
	if len(img_names) < 2:
		raise ValueError("Not enough images provided.")
	
	images = [resize_image(cv2.imread(img)) for img in img_names[:8]]
	return np.array(images)

def calculate_intrinsics(width, height, fov=60):
	focal_len = width/(2*math.tan(fov))
	return np.array([[focal_len ,0, width/2], [0, focal_len, height/2], [0, 0, 1]])

def planar_stitch(images):
	mid = int((len(images) + 1)/2)
	left_list = images[:mid]
	right_list = images[mid:]
	left = stitch_left(left_list)
	pano = stitch_right(left, right_list)
	return pano

def cylindrical_stitch(images):
	K = calculate_intrinsics(images[0].shape[1], images[0].shape[0])

	# Predetermined canvas size calculatet by the circumference of the cylinder
	canvas = np.zeros((images[0].shape[0]+300,int(2*math.pi*K[0][0]),images[0].shape[2]),np.uint8)
	img1 = cylindrical_warp(images[0],K)
	canvas[0:img1.shape[0],0:img1.shape[1]] = img1
	tmp = img1.copy()
	for img in images[1:]:
		img1 = tmp.copy()
		img2 = cylindrical_warp(img,K)
		H, pt1, pt2 = findHomographyFromImages(img1, img2)
		p = pt1 - pt2
		dist = np.mean(p,axis = 0)
		M = np.array([[1,0,dist[0,0]],[0,1,dist[0,1]], [0.,0.,1.]])
		tmp = cv2.warpPerspective(img2,M,(canvas.shape[1],canvas.shape[0]))
		canvas = image_stiching(canvas,tmp)
	return crop_image(canvas)

def cylindrical_planar_stitch(images):
	K = calculate_intrinsics(images[0].shape[1], images[0].shape[0])
	images = [cylindrical_warp(img, K) for img in images]
	images = np.array(images)
	return planar_stitch(images)

if __name__ == "__main__":
	if len(sys.argv) == 2:
		dest_dir = '.'
	elif len(sys.argv) == 3:
		dest_dir = sys.argv[2]
	else:
		raise ValueError("Argument missing. Path to text file containing image paths in left to right order requeired.")

	txt_path = sys.argv[1]
	images = read_images(txt_path)
	
	# Planar image stitching
	try:
		p_pano = planar_stitch(images)
		cv2.imwrite(dest_dir + "/planar_pano.jpg", p_pano)
	except:
		print("Planar Stitching failed")

	# Cylindrical image stitching
	try:
		c_pano = cylindrical_stitch(images)
		cv2.imwrite(dest_dir + "/cylindrical_pano.jpg", c_pano)
	except:
		print("Cylindrical Stitching failed")		

	# Hybrid cylindrical-planar stitching
	try:
		h_pano = cylindrical_planar_stitch(images)
		cv2.imwrite(dest_dir + "/hybrid_pano.jpg", h_pano)
	except:
		print("Hybrid Stitching failed")

	# Official OpenCV stitching
	stitcher = cv2.createStitcher()
	ret, o_pano = stitcher.stitch(images)
	if ret == cv2.STITCHER_OK:
		cv2.imwrite(dest_dir + "/official_pano.jpg", o_pano)
	else:
		print('Official Stitching failse')

	cv2.imshow("Planar Stitching", p_pano)
	cv2.imshow("Cylindrical Stitching", c_pano)
	cv2.imshow("Hybrid Stitching", h_pano)
	cv2.imshow("OpenCV Stitching", o_pano)
	cv2.waitKey(0)
	cv2.destroyAllWindows()