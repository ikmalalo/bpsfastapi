from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import xgboost as xgb
import os

# Inisialisasi FastAPI
app = FastAPI()

# Tentukan path model dan scaler
model_path = os.path.join(os.getcwd(), 'xgboost_model.json')
scaler_path = os.path.join(os.getcwd(), 'scaler.pkl')

# Pastikan model XGBoost ada
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")
model = xgb.Booster()
model.load_model(model_path)  # Menggunakan path yang benar untuk model

# Pastikan scaler ada
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file '{scaler_path}' not found.")
scaler = joblib.load(scaler_path)  # Menggunakan path yang benar untuk scaler

# Membuat model Pydantic untuk validasi input data
class SalesData(BaseModel):
    Produksi_kWh: float
    Kesusutan_kWh: float
    Persentase_: float
    Efficiency_: float
    Energy_Loss_kWh: float
    Customer_Growth_Rate: float
    Quarter_Q1: float
    Quarter_Q2: float
    Quarter_Q3: float
    Quarter_Q4: Optional[float] = 0  # Membuat Quarter_Q4 opsional, default 0

@app.post("/predict")
def predict_sales(data: SalesData):
    # Mengubah input data menjadi dictionary
    data_dict = data.dict()

    # Pilih hanya 9 fitur yang relevan sesuai dengan yang digunakan oleh scaler
    input_data = np.array([[data_dict['Produksi_kWh'],
                            data_dict['Kesusutan_kWh'],
                            data_dict['Persentase_'],
                            data_dict['Efficiency_'],
                            data_dict['Energy_Loss_kWh'],
                            data_dict['Customer_Growth_Rate'],
                            data_dict['Quarter_Q1'],
                            data_dict['Quarter_Q2'],
                            data_dict['Quarter_Q3']]])  # Mengabaikan Quarter_Q4
    
    # Lakukan scaling terhadap input data
    try:
        scaled_data = scaler.transform(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during scaling: {str(e)}")

    # Ubah scaled_data menjadi DMatrix
    dmatrix_data = xgb.DMatrix(scaled_data)

    # Prediksi menggunakan model
    try:
        prediction = model.predict(dmatrix_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

    # Tentukan tahun prediksi berdasarkan input (misal, jika data input tahun 2022, maka hasilnya 2023)
    tahun_prediksi = 2023  # Untuk tahun berikutnya setelah data 2022

    # Pastikan hasil prediksi dikembalikan sebagai tipe data standar Python (float)
    return {"tahun_prediksi": tahun_prediksi, "prediksi_penjualan": float(prediction[0])}  # Menambahkan tahun prediksi
