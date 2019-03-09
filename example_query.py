import requests
from PIL import Image
import os
import io

#load the test image
img_name = '12.jpg'
if not os.access(img_name, os.W_OK):
    img_url = 'http://places.csail.mit.edu/demo/' + img_name
    os.system('wget ' + img_url)

# open image and convert to bytes
img = Image.open(img_name)
imgBytes = io.BytesIO()
img.save(imgBytes, format='PNG')
imgBytes = imgBytes.getvalue()

# query
URL = 'http://localhost:8008/classify'
response = requests.post(URL, files={'image': imgBytes})
print(response.json())
