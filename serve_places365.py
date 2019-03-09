"""
Including code adapted from https://github.com/CSAILVision/places365
"""
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

from starlette.applications import Starlette
from starlette.responses import JSONResponse
import sys
from io import BytesIO
import uvicorn
import aiohttp
import asyncio
from collections import OrderedDict
import numpy as np

# TODO: configure architecture to use, host and port
host = 'localhost'
port = 8008

# the architecture to use
arch = 'resnet18'

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# load the image transformer
centre_crop = trn.Compose([
    trn.Resize((256, 256)),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)
print(f'Loaded places365 CNN with {arch} architecture.')


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


app = Starlette()


@app.route("/classify", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data['image'].read())
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes):
    img = Image.open(BytesIO(bytes))
    input_img = V(centre_crop(img).unsqueeze(0))
    logits = model.forward(input_img)
    h_x = F.softmax(logits, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    out = OrderedDict()
    for i in range(0, 5):
        out[classes[idx[i]]] = np.round(probs[i].detach().item(), 3)
    # print(out)
    return JSONResponse(out)


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
