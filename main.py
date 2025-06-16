from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import (
    add_time_features,
    encode_categoricals,
    add_interaction_features,
    add_aggregated_features
)
from src.model_training import train_model


def main():
    df = load_and_clean_data("data/Airline_Delay_Cause.csv")

    # Feature Engineering Steps
    df = add_time_features(df)
    df, label_encoders = encode_categoricals(df)
    df = add_interaction_features(df)
    df = add_aggregated_features(df)

    model = train_model(df)


if __name__ == "__main__":
    main()
