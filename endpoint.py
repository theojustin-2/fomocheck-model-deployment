from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import ast

app = Flask(__name__)

# Load and preprocess the dataset ONCE at startup
df_full = pd.read_csv("meme_tokens_enriched_v1.csv")
df_full.columns = df_full.columns.str.strip().str.replace(r'\u200b', '', regex=True)

# Parse DEX volume string into dictionaries
df_full['DEX Volume (USD)'] = df_full['DEX Volume (USD)'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})

# Extract specific DEX volumes
for key in ['m5', 'h1', 'h24']:
    df_full[f'DEX Volume {key}'] = df_full['DEX Volume (USD)'].apply(lambda x: float(x.get(key, 0)))

# Preprocessing function
def preprocess(df):
    df['Volume_to_MarketCap'] = df['Total Volume (USD)'] / df['Market Cap (USD)']
    df['Volume_to_MarketCap'] = df['Volume_to_MarketCap'].clip(upper=10)

    df['Missing_Launch'] = df['Launch Date'].isna()
    df['Is_Meme'] = df['Categories'].str.contains("Meme|Pump.fun", na=False).astype(int)
    df['Low_Volume'] = df['Total Volume (USD)'] < 1e5

    df['Vol_H1_H24_Ratio'] = df['DEX Volume h1'] / (df['DEX Volume h24'] + 1e-6)
    df['Vol_M5_H1_Ratio'] = df['DEX Volume m5'] / (df['DEX Volume h1'] + 1e-6)

    return df

# Preprocess the full dataset
df_full = preprocess(df_full)

# Define feature columns
features = [
    'Current Price (USD)', 'Price Change 24h (%)',
    'Total Volume (USD)', 'Volume_to_MarketCap',
    'Is_Meme', 'Missing_Launch', 'Low_Volume',
    'Vol_H1_H24_Ratio', 'Vol_M5_H1_Ratio'
]

# Prepare full feature set
X_full = df_full[features].fillna(0)

# Scale the features
scaler = StandardScaler()
X_scaled_full = scaler.fit_transform(X_full)

# Train the Isolation Forest model ONCE
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X_scaled_full)

# Define API endpoint
@app.route('/score', methods=['POST'])
def score():
    try:
        coin_name = request.json.get('name')
        if not coin_name:
            return jsonify({'error': 'Missing "name" in request'}), 400

        # Find the coin
        df = df_full[df_full['Name'].str.lower() == coin_name.lower()]
        if df.empty:
            return jsonify({'error': f'Coin "{coin_name}" not found'}), 404

        # Prepare input features for the single coin
        X = df[features].fillna(0)
        X_scaled = scaler.transform(X)  # use transform, NOT fit_transform

        # Predict pump score
        pump_score = model.decision_function(X_scaled)[0] * -1  # optional: multiply by -1 if you want bigger = more likely to pump

        result = {
            'name': df['Name'].values[0],
            'symbol': df['Symbol'].values[0],
            'pump_probability': pump_score
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
