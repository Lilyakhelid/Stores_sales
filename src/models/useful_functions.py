import json
import os
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def sarima_sarimax_forecast(series_test, series_train, results, exog_columns=None):
    """
    Effectue des prévisions avec un modèle SARIMA ou SARIMAX, calcule les métriques NRMSE, MAPE, et R^2,
    et ajoute les prévisions et les intervalles de confiance au DataFrame de test.

    Parameters:
        series_test (pd.DataFrame): Données de test contenant la variable cible et, si applicable, les variables exogènes.
        series_train (pd.DataFrame): Données d'entraînement contenant la variable cible pour le calcul des métriques.
        results: Résultats du modèle SARIMA ou SARIMAX après ajustement.
        exog_columns (list, optional): Liste des colonnes exogènes pour la prévision. Par défaut None.

    Returns:
        pd.DataFrame: DataFrame de test avec colonnes supplémentaires pour les prévisions et intervalles de confiance.
        float: NRMSE.
        float: MAPE.
        float: R^2.
    """
    forecast_steps = len(series_test)  # Nombre de pas pour la prévision

    # Extraire les données exogènes futures si des colonnes sont spécifiées
    if exog_columns:
        exog_future = series_test[exog_columns].values
    else:
        exog_future = None

    # Prévisions avec le modèle SARIMA ou SARIMAX
    forecast = results.get_forecast(steps=forecast_steps, exog=exog_future)

    # Récupérer les valeurs prédites et les intervalles de confiance
    forecast_values = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    # Ajouter les prévisions et les intervalles de confiance au DataFrame de test
    series_test = series_test.copy()  # Éviter de modifier l'original
    series_test['Forecast'] = forecast_values.values
    series_test['Lower_CI'] = confidence_intervals.iloc[:, 0].values
    series_test['Upper_CI'] = confidence_intervals.iloc[:, 1].values

    # Calcul des métriques
    nrmse = calculate_nrmse(series_test['sales'], series_test['Forecast'], series_train['sales'])
    mape = calculate_mape(series_test)
    r2 = calculate_r2(series_test['sales'], series_test['Forecast'])

    # Afficher les métriques
    print(f"NRMSE : {nrmse:.4f}")
    print(f"MAPE : {mape:.2f} %")
    print(f"R^2 : {r2:.4f}")

    return nrmse, mape, r2 , series_test

def calculate_nrmse(actual, forecast, train_sales):
    """Calcul du NRMSE."""
    return np.sqrt(mean_squared_error(actual, forecast)) / (train_sales.max() - train_sales.min())

def calculate_mape(series_test):
    """Calcul du MAPE en excluant les valeurs où 'sales' est nul ou manquant."""
    valid_test = series_test[series_test['sales'] > 0].dropna(subset=['Forecast'])
    if valid_test.empty:
        raise ValueError("Les données de test valides pour le calcul de MAPE sont vides.")
    return (np.abs((valid_test['sales'] - valid_test['Forecast']) / valid_test['sales']).mean()) * 100

def calculate_r2(actual, forecast):
    """Calcul du R^2."""
    return r2_score(actual, forecast)






# --sauvegardes--


def save_model_results(file_path, model_name, params, nrmse, mape, r2):
    try:
        with open(file_path, "r") as file:
            results = json.load(file)
    except FileNotFoundError:
        results = []
    
    results.append({
        "Model": model_name,
        "Parameters": params,
        "NRMSE": nrmse,
        "MAPE": mape,
        "R²": r2
    })
    
    with open(file_path, "w") as file:
        json.dump(results, file, indent=4)


def calculate_metrics(y_true, y_pred):
    nrmse = mean_squared_error(y_true, y_pred, squared=False) / (y_true.max() - y_true.min())
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return nrmse, mape, r2

