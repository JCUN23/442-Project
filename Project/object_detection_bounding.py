import os
import json
import cv2
import torch
import torchvision
import torchvision.transforms as T
import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from region_proposal_detection import selective_search, get_best_boxes, get_boxes_iou
from PIL import Image 

SQUEEZENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
SQUEEZENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)

def preprocess(img, size=224):
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
              std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled
  
def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X


def segment(img):
    segments = []
    segment_coors = [] #Tracks the coordinate of the center of each segment
    #px = img.load()
    height = np.asarray(img).shape[0]
    width = np.asarray(img).shape[1]
    for y in range(0, height, 50):
        for x in range(0, width, 100):
            segments.append(Image.fromarray(np.asarray(img)[y:y+200, x:x+400]))
            segment_coors.append(np.array([x, y, 400, 200]))
    return segments, segment_coors

def classify(img):
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    model = torchvision.models.squeezenet1_1(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    model.eval()

    class_idx = json.load(open("imagenet_class_index.json"))
    idx2label = {k:class_idx[str(k)][1] for k in range(len(class_idx))}


    segments_ss = selective_search(img)

    pred_conf = {} # label -> boxes, prob
    good_labels = ['sports_car', 'trailer_truck', 'traffic_light', 'convertible', 'moving-van']
    for i, seg in enumerate(segments_ss):
        x,y,w,h = seg

        img_wnd = Image.fromarray(np.asarray(img)[y:y+h,x:x+w])
        X = preprocess(img_wnd)
        result = model(X)
        pred_class = torch.argmax(result).item()
        class_label = idx2label[pred_class]

        if class_label in good_labels:
            # print(idx2label[pred_class], seg)
            if class_label not in pred_conf:
                pred_conf[class_label] = {}
            if 'boxes' not in pred_conf[class_label]:
                pred_conf[class_label]['boxes'] = []
            if 'prob' not in pred_conf[class_label]:
                pred_conf[class_label]['prob'] = []

            pred_conf[class_label]['boxes'].append(seg)
            pred_conf[class_label]['prob'].append(pred_class)
    
    
    return get_boxes_iou(pred_conf)
