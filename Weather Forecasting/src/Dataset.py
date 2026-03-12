import kagglehub
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
def downloadDataset():
    print("--- Pobieranie zbioru danych... ---")
    folder_path = kagglehub.dataset_download("zeeshier/weather-forecast-dataset")
    
    # Szukamy pliku CSV w tym folderze
    files = os.listdir(folder_path)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    if not csv_files:
        raise Exception("Błąd: Nie znaleziono pliku CSV w pobranym folderze.")
    
    full_csv_path = os.path.join(folder_path, csv_files[0])
    print(f"Sukces! Plik znajduje się w: {full_csv_path}")
    return full_csv_path

def prepareData(path): #returns x_train, x_test, y_train, y_test
    data = pd.read_csv(path)
    X = data[['Temperature','Humidity','Wind_Speed','Cloud_Cover','Pressure']].values
    y = data['Rain'].values

    X = standardize_data(X) # Ściska ogromne liczby do bezpiecznego przedziału
    x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=40,test_size=0.2)
    return x_train, x_test, y_train, y_test
def remodelData(Y): #instead of rain no rain -> 1 0
    bin_Y = [1 if s == 'rain' else 0 for s in Y]
    return bin_Y
def printData(X, Y):
    print(f"Dataset shape is {X.shape[0]} and {Y.shape[0]} ")
    print(X[:5])
    print(Y[:5])
def standardize_data(X): # https://www.geeksforgeeks.org/machine-learning/data-pre-processing-wit-sklearn-using-standard-and-minmax-scaler/
    mu = np.mean(X, axis=0)  
    sigma = np.std(X, axis=0)
    
    Z = (X - mu) / sigma 
    return Z