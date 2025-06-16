import pandas as pd
from sklearn.preprocessing import LabelEncoder


def add_time_features(df):
    # Quarter
    df['quarter'] = pd.to_datetime(df[['year', 'month']].assign(day=1)).dt.quarter

    # Season
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })

    return df


def encode_categoricals(df):
    label_encoders = {}

    for col in ['carrier', 'airport', 'carrier_name', 'airport_name', 'season']:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_label'] = le.fit_transform(df[col])
            label_encoders[col] = le

    return df, label_encoders


def add_interaction_features(df):
    df['carrier_airport'] = df['carrier'] + "_" + df['airport']
    le = LabelEncoder()
    df['carrier_airport_label'] = le.fit_transform(df['carrier_airport'])
    return df


def add_aggregated_features(df):
    # Carrier-level average delay
    carrier_avg_delay = df.groupby('carrier')['carrier_delay'].mean().rename('avg_carrier_delay')
    df = df.merge(carrier_avg_delay, on='carrier', how='left')

    # Airport-level total flights
    airport_total_flights = df.groupby('airport')['arr_flights'].sum().rename('total_flights_per_airport')
    df = df.merge(airport_total_flights, on='airport', how='left')

    return df
