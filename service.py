import pandas as pd
import io
import pickle
from enum import Enum
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel, Field, conint, condecimal

app = FastAPI()


class FuelType(str, Enum):
    diesel = "Diesel"
    petrol = "Petrol"
    lpg = "LPG"
    cng = "CNG"


class SellerType(str, Enum):
    individual = "Individual"
    dealer = "Dealer"
    trustmark_dealer = "Trustmark Dealer"


class TransmissionType(str, Enum):
    manual = "Manual"
    automatic = "Automatic"


class OwnerType(str, Enum):
    first = "First Owner"
    second = "Second Owner"
    third = "Third Owner"
    fourth_and_above = "Fourth & Above Owner"
    test_drive = "Test Drive Car"


class Item(BaseModel):
    name: str = Field(..., description="Model of the vehicle")
    year: conint(ge=1900, le=2100) = Field(..., description="Release year")
    selling_price: conint(ge=0) = Field(..., description="Selling price in currency units")
    km_driven: conint(ge=0) = Field(..., description="Kilometers driven")
    fuel: FuelType = Field(..., description="Type of fuel")
    seller_type: SellerType = Field(..., description="Type of the seller")
    transmission: TransmissionType = Field(..., description="Type of transmission")
    owner: OwnerType = Field(..., description="Ownership type")
    mileage: str = Field(..., description="Mileage")
    engine: str = Field(..., description="Engine capacity")
    max_power: str = Field(..., description="Maximum power")
    torque: str = Field(..., description="Torque specification (ignored by model)")
    seats: condecimal(ge=1, le=50) = Field(..., description="Number of seats")


columns = ['engine', 'km_driven', 'max_power', 'mileage', 'year', 'fuel_Diesel',
           'fuel_LPG', 'fuel_Petrol', 'owner_Fourth & Above Owner',
           'owner_Second Owner', 'owner_Test Drive Car', 'owner_Third Owner',
           'seller_type_Individual', 'seller_type_Trustmark Dealer',
           'transmission_Manual', 'seats_4', 'seats_5', 'seats_6', 'seats_7',
           'seats_8', 'seats_9', 'seats_10', 'seats_14']

model: Ridge
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

x_scaler: StandardScaler
with open('scaler.pkl', 'rb') as file:
    x_scaler = pickle.load(file)


def safe_float_conversion(value, column_name, replaced_value):
    try:
        if isinstance(value, str):
            return float(value.split()[0])
        return float(value)
    except (ValueError, TypeError):
        print(f"WARN: Не удалось преобразовать значение '{value}' в столбце {column_name} "
              f"к типу float - значение заменено на {replaced_value}")
        return replaced_value


def preprocess_df(df):

    df = df.copy()
    for col in ['mileage', 'engine', 'max_power']:
        df[col] = df[col].apply(lambda x: safe_float_conversion(x, col, np.nan))

    if 'torque' in df.columns:
        df.drop(columns=['torque'], inplace=True)

    df['engine'] = df['engine'].fillna(0).apply(lambda x: int(x))
    df['seats'] = df['seats'].fillna(0).apply(lambda x: int(x))
    return df


def get_scaled_features(df):
    categorical_features = ['fuel', 'owner', 'seller_type', 'transmission', 'seats']
    df = preprocess_df(df)
    df = pd.get_dummies(df, columns=categorical_features)
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    for col in df.columns:
        if col not in columns:
            df = df.drop(columns=[col])
    df = df[columns]
    return x_scaler.transform(df)


@app.post("/predict_item",
          description="This endpoint takes a single vehicle's features as "
                      "input and returns the predicted selling price.")
async def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.dict()])
    X_scaled = get_scaled_features(df)
    pred = model.predict(X_scaled)
    return round(pred[0])


@app.post("/predict_items",
          description="This endpoint processes a CSV file containing details "
                      "of multiple vehicles and returns a CSV file with "
                      "predicted selling prices for each vehicle.")
def predict_items(file: UploadFile = File(
    ..., description="The input CSV file to be processed"
)) -> StreamingResponse:

    try:
        input_df = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read the uploaded CSV file"
        )

    X_scaled = get_scaled_features(input_df)
    y_pred = model.predict(X_scaled)
    input_df['selling_price'] = np.round(y_pred)

    output_buffer = io.BytesIO()
    input_df.to_csv(output_buffer, index=False)
    output_buffer.seek(0)
    return StreamingResponse(
        output_buffer,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"}
    )
