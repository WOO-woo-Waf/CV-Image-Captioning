from transformers import CLIPModel
import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:33210'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:33210'

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.save_pretrained("D:/Clip")