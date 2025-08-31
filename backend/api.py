from fastapi import FastAPI, File, UploadFile
from model.utils import load_model,process_single_image
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch.nn.functional as F 
import json
import io
import traceback
app = FastAPI()


model, labels = load_model()
model.eval()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  try:
    contents = await file.read()
    image =  Image.open(io.BytesIO(contents)).convert("RGB")
    processed = process_single_image(image)
    probabilities = (F.softmax(model(processed).data,dim=1)[0]).view(-1).tolist()
    print(probabilities)
    top_five = top_k(probabilities, labels, 5)
    return {"success":True, "probabilities": top_five}
  except Exception as e:
    traceback.print_exc()
    return {"success": False, "message": "Error Incompatible file uploaded"}  
  
def top_k(probabilities, labels, k):
  sorting_list = []
  for i in range(0,len(probabilities)):
    sorting_list.append((labels[i],probabilities[i]))
  sorted_list = sorted(sorting_list, key= lambda x: x[1])
  sorted_list.reverse()
  top_ks = sorted_list[:5]
  json_list =  [{"label": tup[0], "probability": tup[1]} for tup in top_ks]
  return json_list
