from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms # Transformations we can perform on our dataset
import torch.nn.functional as F # All functions that don't have any parameters
from torch.utils.data import DataLoader, Dataset # Gives easier dataset managment and creates mini batches
from torchvision.datasets import ImageFolder
from PIL import Image
from io import BytesIO


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu or cpu

from torchvision import models


model = models.resnet50()

# If you want to do finetuning then set requires_grad = False
# Remove these two lines if you want to train entire model,
# and only want to load the pretrain weights.

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model.to(device)

checkpoint = torch.load("./checpoint_epoch_4.pt") 
model.load_state_dict(checkpoint['model_state_dict']) # Load the model state dict) 


model.eval()


def RandomImagePrediction(contents):
    img_array = Image.open(BytesIO(contents)).convert("RGB")
    data_transforms=transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = data_transforms(img_array).unsqueeze(dim=0) # Returns a new tensor with a dimension of size one inserted at the specified position.
    load = DataLoader(img)
    
    for x in load:
        x=x.to(device)
        pred = model(x)
        _, preds = torch.max(pred, 1)
        print(f"class : {preds}")
        if preds[0] == 1: return "Dog"
        else: return "Cat"
    return "Error"

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()
    detect = RandomImagePrediction(contents)
    return JSONResponse(content={"detect": detect})
