import requests
import json
import pandas as pd

def test_detect_churn():
    url = "http://127.0.0.1:8000/predict"

    try:
        df_preprocessed = pd.read_csv("preprocessing/ecommerce_customer_churn_dataset_preprocessing.csv")
        sample_row = df_preprocessed.drop(columns=['Churned']).iloc[[0]]
    except FileNotFoundError:
        print("File preprocessing tidak ditemukan. Pastikan jalur file benar.")
        return

    payload = {
        "dataframe_split": {
            "columns": sample_row.columns.tolist(),
            "data": sample_row.values.tolist()
        }
    }
    

    print("--- Mengirim Permintaan Ke Endpoint ---")
    print(f"Payload: {json.dumps(payload, indent=2)}\n")

    try:
        response = requests.post(
            url, 
            json=payload, 
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            print("Status: Berhasil (200)")
            print("Hasil Prediksi:", response.json())
        else:
            print(f"Status: Gagal ({response.status_code})")
            print("Pesan Error:", response.text)

    except requests.exceptions.ConnectionError:
        print("Error: Tidak dapat terhubung ke server. Pastikan aplikasi Flask (port 8000) sudah berjalan.")

if __name__ == "__main__":
    test_detect_churn()