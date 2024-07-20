import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
np.random.seed(42)
temperature = np.random.uniform(10, 30, 1000)
humidity = np.random.uniform(40, 100, 1000)
wind_speed = np.random.uniform(0, 30, 1000)
precipitation = np.random.uniform(0, 100, 1000)
X = np.vstack((temperature, humidity, wind_speed, precipitation)).T
rain = (precipitation > 50) & (humidity > 80)
thunderstorm = (precipitation > 70) & (wind_speed > 20)
X_train, X_test, y_temp_train, y_temp_test, y_rain_train, y_rain_test, y_thunderstorm_train, y_thunderstorm_test = train_test_split(
    X, temperature, rain, thunderstorm, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
temp_model = LinearRegression().fit(X_train, y_temp_train)
rain_model = LogisticRegression().fit(X_train, y_rain_train)
thunderstorm_model = LogisticRegression().fit(X_train, y_thunderstorm_train)
joblib.dump(temp_model, 'temp_model.pkl')
joblib.dump(rain_model, 'rain_model.pkl')
joblib.dump(thunderstorm_model, 'thunderstorm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
app = FastAPI()
city_data = {}
class WeatherInput(BaseModel):
    Temperature: float
    Humidity: float
    Wind_speed: float
    Precipitation: float
class WeatherPrediction(BaseModel):
    predicted_temperature: float
    rain_tomorrow: bool
    thunderstorm_tomorrow: bool
@app.post("/add_city/{city_name}")
async def add_city(city_name: str, weather: WeatherInput):
    city_data[city_name.lower()] = {
        "Temperature": weather.Temperature,
        "Humidity": weather.Humidity,
        "Wind_speed": weather.Wind_speed,
        "Precipitation": weather.Precipitation
    }
    return {"message": f"Weather data for {city_name} added/updated successfully"}
@app.get("/predict/{city_name}", response_model=WeatherPrediction)
async def predict_weather(city_name: str):
    city = city_name.lower()
    if city not in city_data:
        raise HTTPException(status_code=404, detail="City not found")
    weather = city_data[city]
    features = np.array([[weather['Temperature'], weather['Humidity'], weather['Wind_speed'], weather['Precipitation']]])
    features_scaled = scaler.transform(features)
    predicted_temp = float(temp_model.predict(features_scaled)[0])
    rain_tomorrow = bool(rain_model.predict(features_scaled)[0])
    thunderstorm_tomorrow = bool(thunderstorm_model.predict(features_scaled)[0])
    return {
        "predicted_temperature": predicted_temp,
        "rain_tomorrow": rain_tomorrow,
        "thunderstorm_tomorrow": thunderstorm_tomorrow
    }
