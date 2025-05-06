from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from predict import RandomImagePrediction

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()
    detect = RandomImagePrediction(contents)
    return JSONResponse(content={"detect": detect})