import os
import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
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
import ast
import numpy as np

# Read the data from the text file
with open('/home/mert/pyskl/box2.txt', 'r') as file:
    content = file.read()

# Parse the data using ast.literal_eval()
data = ast.literal_eval(content)

# Convert the data to a numpy array
data_array = np.array(data)

bounding_boxes = data_array

class Uniformerv2(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model
        self.positional_embedding = nn.Parameter(torch.zeros(197, 1, 1))

    def forward(self, x):
        # Reshape positional_embedding if necessary
        if self.positional_embedding.shape != (197, 1, 1):
            self.positional_embedding = nn.Parameter(
                self.positional_embedding.view(197, 1, 1)
            )

        x = x + self.positional_embedding.to(x.device)
        return self.backbone(x)

# Device on which to run the model
# Set to cuda to load on GPU
device = "cpu"
model_path = hf_hub_download(repo_id="Andy1621/uniformerv2", filename="k400+k710_uniformerv2_b16_8x224.pyth")

# Pick a pretrained model 
model = Uniformerv2(uniformerv2_b16(pretrained=False, t_size=8, no_lmhra=True, temporal_downsample=False))

state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict, strict=False)

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


def load_video(video_path, bounding_boxes):
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    frame_indices = get_index(num_frames, 8)

    crop_size = 448
    scale_size = 256
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.448, 0.225]

    transform = T.Compose([
        GroupScale(int(scale_size)),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

    bounding_box_imgs = []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())

        # Crop the frame based on bounding box coordinates
        for box in bounding_boxes[frame_index]:
            x1, y1, x2, y2, _ = box
            cropped_img = img.crop((x1, y1, x2, y2))
            bounding_box_imgs.append(cropped_img)

    torch_imgs = transform(bounding_box_imgs)
    print("frames: ", len(torch_imgs))

    # Save bounding box images (for testing purposes)
    save_path = '/home/mert/frames/'
    for i, torch_img in enumerate(torch_imgs):
        save_filename = os.path.join(save_path, f"image_{i}.png")
        save_image(torch_img, save_filename)

    return torch_imgs


def inference(video, bounding_boxes):
    vid = load_video(video, bounding_boxes)
    
    # The model expects inputs of shape: B x C x H x W
    TC, H, W = vid.shape
    inputs = vid.reshape(1, TC // 3, 3, H, W).permute(0, 2, 1, 3, 4)
    
    prediction = model(inputs)
    prediction = F.softmax(prediction, dim=1).flatten()

    return {kinetics_id_to_classname[str(i)]: float(prediction[i]) for i in range(400)}


input_video = sys.argv[1]
modified_string = input_video.replace(".", "resized.")
input_video = modified_string
label = inference(input_video, bounding_boxes)

sorted_values = sorted(label.items(), key=lambda x: x[1], reverse=True)
top_5 = sorted_values[:5]

with open('/home/mert/uniformerv2_demo/uniformerwithboxes.txt', 'w') as f:
    for label, value in top_5:
        f.write(f"{label},{value}\n")

