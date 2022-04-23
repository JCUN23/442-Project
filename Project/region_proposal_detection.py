### take an image and return region proposals
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2


def get_iou(bb1, bb2):
	pass


def selective_search(image, method="fast"):
	# initialize OpenCV's selective search implementation and set the
	# input image
	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(image)
	# check to see if we are using the *fast* but *less accurate* version
	# of selective search
	if method == "fast":
		ss.switchToSelectiveSearchFast()
	# otherwise we are using the *slower* but *more accurate* version
	else:
		ss.switchToSelectiveSearchQuality()
	# run selective search on the input image
	rects = ss.process()
	# return the region proposal bounding boxes
	return rects
def get_iou2(x1, x2):
	"""
	Calculate the Intersection over Union (IoU) of two bounding boxes.

	Parameters
	----------
	bb1 : dict
		Keys: {'x1', 'x2', 'y1', 'y2'}
		The (x1, y1) position is at the top left corner,
		the (x2, y2) position is at the bottom right corner
	bb2 : dict
		Keys: {'x1', 'x2', 'y1', 'y2'}
		The (x, y) position is at the top left corner,
		the (x2, y2) position is at the bottom right corner

	Returns
	-------
	float
		in [0, 1]
	"""

	bb1 = {
		'x1' : x1[0],
		'y1' : x1[1],
		'x2' : x1[0] +  x1[3],
		'y2' : x1[1]+  x1[2],
	}
	bb2 = {
		'x1' : x2[0],
		'y1' : x2[1],
		'x2' : x2[0] +  x2[3],
		'y2' : x2[1]+  x2[2],
	}
	assert bb1['x1'] < bb1['x2']
	assert bb1['y1'] < bb1['y2']
	assert bb2['x1'] < bb2['x2']
	assert bb2['y1'] < bb2['y2']

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1['x1'], bb2['x1'])
	y_top = max(bb1['y1'], bb2['y1'])
	x_right = min(bb1['x2'], bb2['x2'])
	y_bottom = min(bb1['y2'], bb2['y2'])

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left) * (y_bottom - y_top)

	# compute the area of both AABBs
	bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
	bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	assert iou >= 0.0
	assert iou <= 1.0
	return iou
def calc_iou(box1, box2):
	x1,y1,w1,h1 =  box1
	x2,y2,w2,h2 =  box2

	box1_area = (w1*h1)
	box2_area = (w2*h2)
	if x1 + w1 < x2 or x2 + w2 < x1:
		return 0.0
	if y1 + h1 < y2 or y2 + h2 < y1:
		return 0.0
	max_x = max(x1,x2)
	max_y = max(y1,y2)
	min_x = min(x1 + w1, x2 + w2)
	min_y = min(y1 + h1, y2 + h2)
	dist_x = min_x - max_x
	dist_y = min_y - max_y
	iou = (dist_x * dist_y) / (box1_area+box2_area)
	return iou


# find smallest bounding for each item # then return 
def get_best_boxes(pred_conf):
	THRESHOLD = 0.60
	converged = False
	highest = 0 

	#get rid of duplicates
	for key in pred_conf.keys():
		pred_conf[key]['boxes'] = [l.tolist() for l in pred_conf[key]['boxes']]
		result = [] 
		[result.append(x) for x in pred_conf[key]['boxes'] if x not in result] 
		pred_conf[key]['boxes'] = result
	while not converged: 
		for key in pred_conf.keys():
			converged = True
			boxes = pred_conf[key]['boxes']
			to_remove = []
			to_remove_probs = []
			for i, box1 in enumerate(boxes):
				for j, box2 in enumerate(boxes):
					if i == j :
						continue
					iou = get_iou2(box1,box2)
					#print(iou)
					highest = max(highest, iou)
					box_to_remove = None
					prob_to_remove = None
					if pred_conf[key]['prob'][i] > pred_conf[key]['prob'][j]:
						box_to_remove = box2
						prob_to_remove = pred_conf[key]['prob'][j]
					else: 
						box_to_remove = box1
						prob_to_remove = pred_conf[key]['prob'][i]

					if iou > THRESHOLD:
						converged = False
						if box_to_remove not in to_remove:
							to_remove.append(box_to_remove)
						if prob_to_remove not in to_remove_probs:
							to_remove_probs.append(prob_to_remove)
						continue
			for box in to_remove:
				if box in pred_conf[key]['boxes']:
					pred_conf[key]['boxes'].remove(box)
			for prob in to_remove_probs:
				if prob in pred_conf[key]['prob']:
					pred_conf[key]['prob'].remove(prob)
	print(highest)
	print(len(pred_conf[key]['boxes']))
	return pred_conf[key]['boxes']