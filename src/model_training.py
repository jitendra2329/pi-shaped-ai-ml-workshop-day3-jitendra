from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.feature_selection import select_top_k_features

def train_model(df):
    all_features = [
        'year', 'month', 'quarter', 'season_label',
        'carrier_label', 'airport_label', 'carrier_airport_label',
        'arr_flights', 'arr_del15', 'carrier_ct', 'weather_ct', 'nas_ct',
        'security_ct', 'late_aircraft_ct', 'arr_cancelled', 'arr_diverted',
        'arr_delay', 'carrier_delay', 'weather_delay', 'nas_delay',
        'security_delay', 'late_aircraft_delay',
        'avg_carrier_delay', 'total_flights_per_airport'
    ]

    target_col = 'airport_label'

    # Ensure the feature exists in df
    feature_cols = [col for col in all_features if col in df.columns]
    X = df[feature_cols]
    y = df[target_col]

    # Feature Selection step
    X_selected, selected_feature_names, selector = select_top_k_features(X, y, k=15)

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model
