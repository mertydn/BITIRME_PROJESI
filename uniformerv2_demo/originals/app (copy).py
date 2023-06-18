import os
import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader
from decord import cpu
from uniformerv2 import uniformerv2_b16
from kinetics_class_index import kinetics_classnames
from transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)

from huggingface_hub import hf_hub_download

class Uniformerv2(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model
    
    def forward(self, x):
        return self.backbone(x)

# Device on which to run the model
# Set to cuda to load on GPU
device = "cpu"
model_path = hf_hub_download(repo_id="Andy1621/uniformerv2", filename="k400+k710_uniformerv2_b16_8x224.pyth")
# Pick a pretrained model 
model = Uniformerv2(uniformerv2_b16(pretrained=False, t_size=8, no_lmhra=True, temporal_downsample=False))
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)

# Set to eval mode and move to desired device
model = model.to(device)
model = model.eval()

# Create an id to label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[k] = v


def get_index(num_frames, num_segments=8):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    frame_indices = get_index(num_frames, 8)

    # transform
    crop_size = 224
    scale_size = 256
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]

    transform = T.Compose([
        GroupScale(int(scale_size)),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    return torch_imgs
    

def inference(video):
    vid = load_video(video)
    
    # The model expects inputs of shape: B x C x H x W
    TC, H, W = vid.shape
    inputs = vid.reshape(1, TC//3, 3, H, W).permute(0, 2, 1, 3, 4)
    
    prediction = model(inputs)
    prediction = F.softmax(prediction, dim=1).flatten()

    return {kinetics_id_to_classname[str(i)]: float(prediction[i]) for i in range(400)}
    

def set_example_video(example: list) -> dict:
    return gr.Video.update(value=example[0])

#input_video = '/home/mert/uniformerv2_demo/hitting_baseball.mp4'
input_video = sys.argv[1]
label = inference(input_video)

sorted_values = sorted(label.items(), key=lambda x: x[1], reverse=True)
top_5 = sorted_values[:5]
# for label, value in top_5:
    # print(f"Label: {label}, Value: {value}")

with open('/home/mert/uniformerv2_demo/uniformer.txt', 'w') as f:
    for label, value in top_5:
        f.write(f"{label},{value}\n")

