import pandas as pd

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    delay_columns = [
        'arr_flights', 'arr_del15', 'carrier_ct', 'weather_ct', 'nas_ct',
        'security_ct', 'late_aircraft_ct', 'arr_cancelled', 'arr_diverted',
        'arr_delay', 'carrier_delay', 'weather_delay', 'nas_delay',
        'security_delay', 'late_aircraft_delay'
    ]
    df.dropna(subset=delay_columns, inplace=True)

    airport_counts = df['airport'].value_counts()
    valid_airports = airport_counts[airport_counts > 100].index
    df = df[df['airport'].isin(valid_airports)]

    print(f"Filtered data shape: {df.shape}")
    return df
