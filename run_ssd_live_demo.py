from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys

import torch
import urllib
from PIL import Image
from torchvision import transforms
from torchsummary import summary
import numpy as np
import struct
import os

import torch.nn as nn
from vision.ssd.data_preprocessing import PredictionTransform
from vision.ssd.config import mobilenetv2_ssd_config as config


def bin_write(f, data):
    data = data.flatten()
    # print(data)
    fmt = 'f'*len(data)
    bin = struct.pack(fmt, *data)
    f.write(bin)


outputs = {}


def hook(module, input, output):
    outputs[module] = output


def print_wb_output(model, input_batch, image, predictor):

    if not os.path.exists('debug'):
        os.makedirs('debug')  
    if not os.path.exists('layers'):
        os.makedirs('layers')  

    for n, m in model.named_modules():
        m.register_forward_hook(hook)

    predictor.predict(image)
    i = input_batch.data.numpy()
    i = np.array(i, dtype=np.float32)
    i.tofile("debug/input.bin", format="f")

    f = None
    for n, m in model.named_modules():
        t = '-'.join(n.split('.'))

        if m not in outputs:
            continue
        in_output = outputs[m]
        o = in_output.data.numpy()
        o = np.array(o, dtype=np.float32)

        t = '-'.join(n.split('.'))
        o.tofile("debug/" + t + ".bin", format="f")

        if not(' of Conv2d' in str(m.type) or ' of Linear' in str(m.type) or ' of BatchNorm2d' in str(m.type)):
            continue

        if ' of Conv2d' in str(m.type) or ' of Linear' in str(m.type):
            file_name = "layers/" + t + ".bin"
            print("open file: ", file_name)
            f = open(file_name, mode='wb')

        print(n, ' ----------------------------------------------------------------')

        w = np.array([])
        b = np.array([])

        if 'weight' in m._parameters and m._parameters['weight'] is not None:
            w = m._parameters['weight'].data.numpy()
            w = np.array(w, dtype=np.float32)
            print("    weights shape:", np.shape(w))

        if 'bias' in m._parameters and m._parameters['bias'] is not None:
            b = m._parameters['bias'].data.numpy()
            b = np.array(b, dtype=np.float32)
            print("    bias shape:", np.shape(b))

        if 'BatchNorm2d' in str(m.type):
            b = m._parameters['bias'].data.numpy()
            b = np.array(b, dtype=np.float32)
            s = m._parameters['weight'].data.numpy()
            s = np.array(s, dtype=np.float32)
            rm = m.running_mean.data.numpy()
            rm = np.array(rm, dtype=np.float32)
            rv = m.running_var.data.numpy()
            rv = np.array(rv, dtype=np.float32)
            bin_write(f, b)
            bin_write(f, s)
            bin_write(f, rm)
            bin_write(f, rv)
            print("    s shape:", np.shape(s))
            print("    rm shape:", np.shape(rm))
            print("    rv shape:", np.shape(rv))

        else:
            bin_write(f, w)
            if b.size > 0:
                bin_write(f, b)

        if ' of BatchNorm2d' in str(m.type) or ' of Linear' in str(m.type):
            f.close()
            print("close file")
            f = None


if len(sys.argv) < 4:
    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> [video file]')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]

if len(sys.argv) >= 5:
    cap = cv2.VideoCapture(sys.argv[4])  # capture from file
else:
    cap = cv2.VideoCapture(0)   # capture from camera
    cap.set(3, 1920)
    cap.set(4, 1080)

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)


if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)

    # download std image
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    # open image
    input_image = cv2.imread(filename, cv2.COLOR_BGR2RGB)

    # get input tensor
    transform = PredictionTransform(
        config.image_size, config.image_mean, config.image_std)
    input_tensor = transform(input_image)
    input_batch = input_tensor.unsqueeze(0)

    orig_image = cv2.imread(filename, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(orig_image, 10, 0.4)

    print(list(net.children()))
    print_wb_output(net, input_batch, orig_image, predictor)

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        print('box', box)
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(orig_image, (box[0], box[1]),
                      (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(orig_image, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imshow('annotated', orig_image)
    cv2.waitKey(0)


elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)

summary(net, (3, 512, 512))

timer = Timer()
while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(orig_image, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
