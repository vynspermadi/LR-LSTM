from flask import Flask, request, render_template
from datetime import timedelta
import datetime as dt
import pickle
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', price_pred=0)

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the price based on user inputs
    and render the result to the html page
    '''

    NamaKomoditas, Tanggal = [x for x in request.form.values()]

    if NamaKomoditas == 'Ayam':
        model_file = open('regresi_ayam.pkl', 'rb')
        model_in = pickle.load(model_file, encoding='bytes')

        modelLoad = load_model("lstm_ayam.h5")
        with open('df_ayam.pkl', 'rb') as data:
            df = pickle.load(data)
        
    elif NamaKomoditas == 'Bawang Merah':
        model_file = open('regresi_bamer.pkl', 'rb')
        model_in = pickle.load(model_file, encoding='bytes')
        
        modelLoad = load_model("lstm_bamer.h5")
        with open('df_bamer.pkl', 'rb') as data:
            df = pickle.load(data)

    elif NamaKomoditas == 'Bawang Putih':
        model_file = open('regresi_baput.pkl', 'rb')
        model_in = pickle.load(model_file, encoding='bytes')
        
        modelLoad = load_model("lstm_baput.h5")
        with open('df_baput.pkl', 'rb') as data:
            df = pickle.load(data)

    elif NamaKomoditas == 'Beras':
        model_file = open('regresi_beras.pkl', 'rb')
        model_in = pickle.load(model_file, encoding='bytes')
        
        modelLoad = load_model("lstm_beras.h5")
        with open('df_beras.pkl', 'rb') as data:
            df = pickle.load(data)

    elif NamaKomoditas == 'Cabai Merah':
        model_file = open('regresi_camer.pkl', 'rb')
        model_in = pickle.load(model_file, encoding='bytes')

        modelLoad = load_model("lstm_camer.h5")
        with open('df_camer.pkl', 'rb') as data:
            df = pickle.load(data)
        
    elif NamaKomoditas == 'Cabai Rawit':
        model_file = open('regresi_cawit.pkl', 'rb')
        model_in = pickle.load(model_file, encoding='bytes')
        
        modelLoad = load_model("lstm_cawit.h5")
        with open('df_cawit.pkl', 'rb') as data:
            df = pickle.load(data)

    elif NamaKomoditas == 'Gula':
        model_file = open('regresi_gula.pkl', 'rb')
        model_in = pickle.load(model_file, encoding='bytes')

        modelLoad = load_model("lstm_gula.h5")
        with open('df_gula.pkl', 'rb') as data:
            df = pickle.load(data)

    elif NamaKomoditas == 'Minyak':
        model_file = open('regresi_minyak.pkl', 'rb')
        model_in = pickle.load(model_file, encoding='bytes')

        modelLoad = load_model("lstm_minyak.h5")
        with open('df_minyak.pkl', 'rb') as data:
            df = pickle.load(data)
        
    elif NamaKomoditas == 'Daging Sapi':
        model_file = open('regresi_sapi.pkl', 'rb')
        model_in = pickle.load(model_file, encoding='bytes')
        
        modelLoad = load_model("lstm_sapi.h5")
        with open('df_sapi.pkl', 'rb') as data:
            df = pickle.load(data)

    else:
        model_file = open('regresi_telur.pkl', 'rb')
        model_in = pickle.load(model_file, encoding='bytes')

        modelLoad = load_model("lstm_telur.h5")
        with open('df_telur.pkl', 'rb') as data:
            df = pickle.load(data)

    #return LR result
    price_pred = Tanggal
    price_pred= dt.datetime.strptime(price_pred, "%Y-%m-%d")
    price_pred_ord = price_pred.toordinal()
    
    prediction = model_in.predict(np.array([[price_pred_ord]]))
    output = round(prediction[0], 2)

    #return LSTM result
    FullData = df[['Harga']].values
    sc=MinMaxScaler()
    DataScaler = sc.fit(FullData)

    d1 = timedelta(days=1)
    d14 = timedelta(days=14)
    last14date = price_pred - d14
    price_pred = price_pred -d1
    price_pred=price_pred.strftime('%Y-%m-%d')
    last14date= last14date.strftime('%Y-%m-%d')

    Last14Days= df[last14date:price_pred]
    Last14Days = Last14Days[['Harga']].values

    NumSamples=1
    TimeSteps=14
    NumFeatures=1
    Last14Days=Last14Days.reshape(NumSamples,TimeSteps,NumFeatures)
    predicted_Price = modelLoad.predict(Last14Days)
    predicted_Price = DataScaler.inverse_transform(predicted_Price)
    predicted_Price = [float(np.round(predicted_Price)) for predicted_Price in predicted_Price]
    outputLSTM = round(predicted_Price[0],2)

    return render_template('index.html', pred_harga=output, pred_harga_LSTM=outputLSTM , NamaKomoditas= NamaKomoditas, Tanggal=Tanggal)

if __name__ == '__main__':
    app.run(debug=True)
