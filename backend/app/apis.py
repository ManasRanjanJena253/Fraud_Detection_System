import os.path
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pickle
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from model.autoencoder_model_architecture import AutoEncoder
import torch
import matplotlib.pyplot as plt
import numpy as np
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def greetings():
    return {
        "INFO": "Transactional Fraud Detection App",
        "CREATOR": "Manas Ranjan Jena",
        "GitHub": "https://github.com/ManasRanjanJena253"
    }


@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Endpoint to upload csv files by the user for predictions.
    :param file : The csv file containing the transaction data.
    """
    try:
        contents = await file.read()
        data = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        df = data
        if "Class" in df.columns:
            df = df.drop(["Class"], axis=1)

        with open("model/saved_models/min_max_scaler.pkl", "rb") as f:
            min_max_scaler = pickle.load(f)
            scaled_data = min_max_scaler.transform(df)

        return {"Data": scaled_data.tolist()}
    except Exception as e:
        print(f"UPLOAD ERROR: {e}")
        return {"ERROR": str(e)}


@app.post("/upload_csv/reconstruction_graph")
async def show_reconstruction_graph(data: list, idx: int = None):
    """
    Endpoint to show the plot difference between actual data and reconstructed data.
    :param idx : The index of the row that the user want to see the reconstructed graph of.
    :param data : The csv file.
    """
    try:
        model = AutoEncoder()
        model.load_state_dict(torch.load("model/saved_models/autoencoder_model.pth"))
    except Exception as e:
        return {"ERROR": str(e)}

    X_scaled = np.asarray(data)
    if len(X_scaled.shape) == 1:
        X_scaled = X_scaled.reshape(1, -1)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.inference_mode():
        reconstructed = model(X_tensor).cpu().numpy()

    if idx is None:
        idx = np.random.randint(0, len(X_scaled))

    reconstruction_error = torch.mean((X_tensor - reconstructed) ** 2, dim=1)

    plt.figure(figsize = (12, 4))
    plt.plot(X_scaled[idx], label = 'Original')
    plt.plot(reconstructed[idx], label = 'Reconstructed')
    plt.legend()
    plt.title(f"{idx} Reconstruction \n Reconstruction Error : {reconstruction_error.cpu().numpy().round(5)}")
    plot_path = f"Reconstruction_Plot_{idx}.png"
    plt.savefig(plot_path)

    if os.path.exists(plot_path):
        return FileResponse(plot_path, media_type = "image/png")
    else:
        return {"ERROR": "Unable to Display the Plot."}


@app.post("/upload_csv/predict")
async def predict(data: list, idx: int):
    """
    Make predictions on the processed data uploaded through upload_csv endpoint.
    """
    try:
        with open("model/saved_models/hybrid_classifierV1.pkl", "rb") as f:
            ml_model = pickle.load(f)
        dl_model = AutoEncoder()
        dl_model.load_state_dict(torch.load("model/saved_models/autoencoder_model.pth"))
    except Exception as e:
        return {"ERROR": str(e)}

    X_scaled = np.asarray(data[idx]).reshape(1, -1)

    X_tensor = torch.tensor(X_scaled, dtype = torch.float32)
    with torch.inference_mode():
        reconstructed = dl_model(X_tensor).cpu().numpy()

    reconstruction_error = torch.mean((X_tensor - reconstructed) ** 2, dim = 1)

    pred = ml_model.predict(X_scaled)[0]
    confidence_score = ml_model.predict_proba(X_scaled)[0][pred] * 100

    RE = reconstruction_error.cpu().numpy().round(5)
    if pred == 0:
        pred_class = "Not Fraud"
    else:
        pred_class = "Risky" if confidence_score < 70 else "Fraud"

    return {
        "Predicted Class": pred_class,
        "Pred Confidence": confidence_score,
        "Reconstruction Error": RE
    }


if __name__ == "__main__":
    uvicorn.run("apis:app", host = "127.0.0.1", port = 8000, reload = True)
