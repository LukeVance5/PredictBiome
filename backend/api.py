from fastapi import FastAPI
import torch
from torchvision import transforms
from model.utils import load_model
from PIL import Image
app = FastAPI()


model, labels = load_model()
model.eval()

transform = transforms.Compose([transforms.Resize((size,size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                          std=[0.5, 0.5, 0.5])
                                     ])
@app.post("/predict")
async def predict(file):
  try:
    image = Image.open(file.file).convert("RGB")
  except Exception as e:
    return {"message": "Error Incompatable file uploaded"}  