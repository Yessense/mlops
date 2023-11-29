from pydantic import BaseModel
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import mlflow
import onnxruntime
import numpy as np
import onnx
import uvicorn

MODEL_NAME: str = "./onnx/model.onnx"

class OnnxRequest(BaseModel):
    data: list

@asynccontextmanager
async def lifespan(app: FastAPI):
    global session
    global input_name
    global label_name

    session = onnxruntime.InferenceSession(MODEL_NAME)

    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    yield
    pass

app = FastAPI(
    description="API Service for MLflow ONNX Models", lifespan=lifespan
)


@app.get("/healthcheck")
async def healthcheck():
    return 200

@app.post("/predict")
def predict(features: OnnxRequest):
    data = features.dict()["data"]

    prediction = session.run(
        [label_name], {input_name: np.array(data).astype(np.float32)}
    )
    return {"prediction": prediction[0].tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)