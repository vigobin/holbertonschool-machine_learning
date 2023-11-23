#!/usr/bin/env python3
"""When to Invest"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(data):
    """Preprocess data"""
    # Select only the 'Timestamp' and 'Weighted_Price' columns
    data = data[['Timestamp', 'Weighted_Price']].copy()

    # Convert Unix time to datetime
    data.loc[:, 'Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')

    # Set datetime as index
    data.set_index('Timestamp', inplace=True)

    # Drop any NaN values
    data.dropna(inplace=True)

    # Resample data to hourly intervals
    data = data.resample('H').mean()

    # Rescale the data
    scaler = MinMaxScaler()
    data.loc[:, :] = scaler.fit_transform(data)

    return data

def main():
    """Call preprocess function"""
    # Load the data
    coinbase_data = pd.read_csv(COINBASE)
    bitstamp_data = pd.read_csv(BITSTAMP)

    # Preprocess the data
    coinbase_data = preprocess_data(coinbase_data)
    bitstamp_data = preprocess_data(bitstamp_data)

    # Save the preprocessed data
    coinbase_data.to_csv('coinbase_preprocessed.csv', index=True)
    bitstamp_data.to_csv('bitstamp_preprocessed.csv', index=True)

if __name__ == "__main__":
    main()
