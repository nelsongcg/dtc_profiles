
import pickle
import numpy as np
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import sklearn

app = FastAPI(title="Predicting profiles")

templates = Jinja2Templates(directory="app/templates")

# Represents a particular wine (or datapoint)
class Profile(BaseModel):
    churches:float
    resorts:float
    beaches:float
    parks:float
    theatres:float
    museums:float
    malls:float
    zoo:float
    restaurants:float
    pubs_bars:float
    local_services:float
    burger_pizza_shops:float
    hotels_other_lodgings:float
    juice_bars:float
    art_galleries:float
    dance_clubs:float
    swimming_pools:float
    gyms:float
    bakeries:float
    beauty_spas:float
    cafes:float
    view_points:float
    monuments:float
    gardens:float
        
@app.on_event("startup")
def load_clf():
    # Load classifier from pickle file
    with open("app/profiles.pkl", "rb") as file:
        global clf
        clf = pickle.load(file)
        
@app.post("/predict")
def predict(prof: Profile):
    data_point = np.array(
        [
            [
                prof.churches,
                prof.resorts,
                prof.beaches,
                prof.parks,
                prof.theatres,
                prof.museums,
                prof.malls,
                prof.zoo,
                prof.restaurants,
                prof.pubs_bars,
                prof.local_services,
                prof.burger_pizza_shops,
                prof.hotels_other_lodgings,
                prof.juice_bars,
                prof.art_galleries,
                prof.dance_clubs,
                prof.swimming_pools,
                prof.gyms,
                prof.bakeries,
                prof.beauty_spas,
                prof.cafes,
                prof.view_points,
                prof.monuments,
                prof.gardens
            ]
        ]
    )

    pred = clf.predict_proba(data_point).tolist()
    pred = pred[0]
    print(pred)
    return {"Prediction": pred}

@app.get("/")
def dashboard(request:Request):
    return templates.TemplateResponse("dashboard.html",{
        "request": request
    })
