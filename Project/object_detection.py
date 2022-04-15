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
SQUEEZENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
SQUEEZENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)
from PIL import Image

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

def compute_saliency_maps(X, y, model):
    # Make input tensor require gradient
    X.requires_grad_()

    saliency = None
    # 1. Do the forward pass for input X to get the prediction score y pred.
    y_pred = model.forward(X)
    # 2. Take the score/probability of the correct class (indicated by y) from y pred and sum over the minibatch to get loss.
    loss = torch.sum(y_pred[torch.arange(0, y.size()[0]), y])
    # # 3. Call loss.backward() and extract the gradient on input X as X grad (gradient will be calculated
    # # automatically).
    loss.backward()
    # # 4. Finally, take the maximum between the absolute value of X grad and 1 as the final saliency map.
    saliency, _ = torch.max(torch.abs(X.grad), 1)
    # ##############################################################################
    # #               END OF YOUR CODE                                             #
    # ##############################################################################
    return saliency

def show_saliency_maps(X_tensor, y_tensor, model, idx2label):
    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.to('cpu').numpy()
    N = X_tensor.shape[0]
    plt.figure(figsize=(8,8))
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(deprocess(X_tensor[i].cpu().unsqueeze(0)))
        plt.axis('off')
        plt.title(idx2label[y_tensor[i].item()])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 8)
    plt.show()

def segment(img):
    segments = []
    px = img.load()
    height = np.asarray(img).shape[0]
    width = np.asarray(img).shape[1]
    for y in range(0, height, 50):
        for x in range(0, width, 100):
            segments.append(Image.fromarray(np.asarray(img)[y:y+200, x:x+400]))
    return segments

def classify(filename):
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    model = torchvision.models.squeezenet1_1(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    # Make sure the model is in "eval" mode
    model.eval()

    class_idx = json.load(open("imagenet_class_index.json"))
    idx2label = {k:class_idx[str(k)][1] for k in range(len(class_idx))}

    # img = np.asarray(Image.open(filename).convert('RGB'))
    # # y = torch.argmax(model(X), dim=1)
    # # show_saliency_maps(X, y, model, idx2label)

    # gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # gray = np.float32(gray)
    # dst = cv2.cornerHarris(gray,2,5,0.02)
    # #result is dilated for marking the corners, not important
    # dst = cv2.dilate(dst,None)
    # # Threshold for an optimal value, it may vary depending on the image.
    # img[dst>0.0001*dst.max()]=[0,0,255]
    # plt.figure()
    # plt.imshow(img)
    # plt.show()

    img = Image.open(filename).convert('RGB')
    segments = segment(img)
    good_labels = ['sports_car', 'trailer_truck', 'traffic_light', 'go-kart', 'convertible', 'golfcart', 'moving-van']
    for i, seg in enumerate(segments):
        X = preprocess(seg)
        pred_class = torch.argmax(model(X)).item()
        plt.figure(figsize=(6,8))
        plt.title('Predicted Class: %s' % idx2label[pred_class])
        plt.axis('off')
        if idx2label[pred_class] in good_labels:
            plt.imshow(seg)
        # plt.savefig(f'{filename.split(".")[0]}-{i}_pred.jpg')
            plt.show()
            print(idx2label[pred_class])
        
