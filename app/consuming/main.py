from statistics import linear_regression
from flask import Flask, jsonify, request, make_response, abort
import pandas as pd
import requests
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def baixandoDados(channel):
    """recuperando dados do thingspeak"""
    data = requests.get(f"https://thingspeak.com/channels/{channel}/feed")
    return data.json()

def criandoDataFrame(dados):
    """criando csv através dos dados"""
    df = pd.DataFrame.from_records(dados['feeds'])
    df.drop(columns=['created_at', 'entry_id'], inplace=True)
    return df

def treinando_modelo(df, y_target):
    X = df.drop(columns=[y_target])
    y = df[y_target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    return model

app = Flask(__name__)
@app.route("/", methods=["GET"])
def get_result():
    channel = request.args['channel']
    target = request.args['target']
    content = request.json
    data = baixandoDados(channel)
    df = criandoDataFrame(data)
    model = treinando_modelo(df, target)
    predict = model.predict(pd.json_normalize(content).drop(columns=[target]))
    return jsonify({
            "predição": predict[0],
            "coefs": str(model.coef_)
        })

app.run(host="0.0.0.0", port=5000)
    