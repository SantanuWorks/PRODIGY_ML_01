from django.shortcuts import render
from responses import POST
import pandas as pd
import joblib
import os
from HOUSEPREDICTION.settings import BASE_DIR

def index(request):
    return render(request, 'index.html')

def predict(request):

    # Bring test features
    area = float(request.POST['area'])
    beds = int(request.POST['bed'])
    baths = int(request.POST['bath'])

    # making a list of data
    data = { "area" : [area], "bedrooms" : [beds], "bathrooms" : [baths] }

    # making data frame for model
    test_data = pd.DataFrame(data)

    # model path
    path = os.path.join(BASE_DIR, 'static', "model/hpricepredictor.sav")
    
    # load model from the file
    model = joblib.load(path)
    
    # predict the price of house using above features
    result = model.predict(test_data)

    # result value
    res = round(result[0])

    # comma separated
    res = f"{res: ,d}"

    return render(request, 'result.html', { "area" : area, "beds" : beds, "baths" : baths, "result" : res })