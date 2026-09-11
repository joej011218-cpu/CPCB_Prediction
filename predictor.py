import os
from database import read_dataframe
from prediction_storage import save_predictions_to_postgres
import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_PATH = "data/cpcb_vadodara.db"
MODEL_BASE = "models/Vadodara"
OUTPUT_FILE = "vadodara_live_predictions.csv"

AQI_LAGS = [
    1, 2, 3, 6, 12, 24, 48, 72, 168
]

POLLUTANT_LAGS = [
    1, 2, 3, 6, 12, 24, 48, 72, 168
]

ROLLING_WINDOWS = [
    3, 6, 12, 24
]

TREND_WINDOWS = [
    3, 6, 12
]

CHANGE_LAGS = [
    1, 3, 6, 12, 24
]


# ============================================================
# IMPORTANT:
# NO HAS BEEN COMPLETELY REMOVED
#
# These are the ONLY pollutants used by the trained
# 224-feature XGBoost model.
# ============================================================

POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
    "nh3",
    "so2",
    "co",
    "o3"
]


# ============================================================
# CPCB AQI POLLUTANTS
#
# NO is NOT part of CPCB composite AQI.
# ============================================================

AQI_POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
    "nh3",
    "so2",
    "co",
    "o3"
]


# ============================================================
# CPCB BREAKPOINTS
# ============================================================

PM25_BREAKPOINTS = [
    (0, 30, 0, 50),
    (31, 60, 51, 100),
    (61, 90, 101, 200),
    (91, 120, 201, 300),
    (121, 250, 301, 400),
    (251, np.inf, 401, 500)
]


PM10_BREAKPOINTS = [
    (0, 50, 0, 50),
    (51, 100, 51, 100),
    (101, 250, 101, 200),
    (251, 350, 201, 300),
    (351, 430, 301, 400),
    (431, np.inf, 401, 500)
]


NO2_BREAKPOINTS = [
    (0, 40, 0, 50),
    (41, 80, 51, 100),
    (81, 180, 101, 200),
    (181, 280, 201, 300),
    (281, 400, 301, 400),
    (401, np.inf, 401, 500)
]


NH3_BREAKPOINTS = [
    (0, 200, 0, 50),
    (201, 400, 51, 100),
    (401, 800, 101, 200),
    (801, 1200, 201, 300),
    (1201, 1800, 301, 400),
    (1801, np.inf, 401, 500)
]


SO2_BREAKPOINTS = [
    (0, 40, 0, 50),
    (41, 80, 51, 100),
    (81, 380, 101, 200),
    (381, 800, 201, 300),
    (801, 1600, 301, 400),
    (1601, np.inf, 401, 500)
]


CO_BREAKPOINTS = [
    (0, 1.0, 0, 50),
    (1.1, 2.0, 51, 100),
    (2.1, 10.0, 101, 200),
    (10.1, 17.0, 201, 300),
    (17.1, 34.0, 301, 400),
    (34.1, np.inf, 401, 500)
]


O3_BREAKPOINTS = [
    (0, 50, 0, 50),
    (51, 100, 51, 100),
    (101, 168, 101, 200),
    (169, 208, 201, 300),
    (209, 748, 301, 400),
    (749, np.inf, 401, 500)
]


# ============================================================
# SUBINDEX
# ============================================================

def calculate_subindex(value, breakpoints):

    if pd.isna(value):
        return np.nan

    value = float(value)

    if value < 0:
        return np.nan

    for c_low, c_high, i_low, i_high in breakpoints:

        if value <= c_high:

            if np.isinf(c_high):
                return float(i_low)

            return (
                ((i_high - i_low) / (c_high - c_low))
                * (value - c_low)
                + i_low
            )

    return 500.0


# ============================================================
# CPCB AQI
# ============================================================

def calculate_cpcb_aqi(row):

    subindices = []

    # --------------------------------------------------------
    # PM2.5
    # --------------------------------------------------------

    value = calculate_subindex(
        row["pm25_24h"],
        PM25_BREAKPOINTS
    )

    if not pd.isna(value):
        subindices.append(value)

    # --------------------------------------------------------
    # PM10
    # --------------------------------------------------------

    value = calculate_subindex(
        row["pm10_24h"],
        PM10_BREAKPOINTS
    )

    if not pd.isna(value):
        subindices.append(value)

    # --------------------------------------------------------
    # NO2
    # --------------------------------------------------------

    value = calculate_subindex(
        row["no2_24h"],
        NO2_BREAKPOINTS
    )

    if not pd.isna(value):
        subindices.append(value)

    # --------------------------------------------------------
    # NH3
    # --------------------------------------------------------

    value = calculate_subindex(
        row["nh3_24h"],
        NH3_BREAKPOINTS
    )

    if not pd.isna(value):
        subindices.append(value)

    # --------------------------------------------------------
    # SO2
    # --------------------------------------------------------

    value = calculate_subindex(
        row["so2_24h"],
        SO2_BREAKPOINTS
    )

    if not pd.isna(value):
        subindices.append(value)

    # --------------------------------------------------------
    # CO
    # --------------------------------------------------------

    value = calculate_subindex(
        row["co_8h"],
        CO_BREAKPOINTS
    )

    if not pd.isna(value):
        subindices.append(value)

    # --------------------------------------------------------
    # O3
    # --------------------------------------------------------

    value = calculate_subindex(
        row["o3_8h"],
        O3_BREAKPOINTS
    )

    if not pd.isna(value):
        subindices.append(value)

    # --------------------------------------------------------
    # COMPOSITE AQI
    # --------------------------------------------------------

    if len(subindices) == 0:
        return np.nan

    return max(subindices)


# ============================================================
# LOAD DATABASE
# ============================================================

def load_database():

    print()
    print("=" * 80)
    print("LOADING VADODARA DATABASE")
    print("=" * 80)

    query = """
    SELECT
        timestamp,
        pm25,
        pm10,
        no2,
        nh3,
        so2,
        co,
        o3
        FROM cpcb_hourly
        ORDER BY timestamp
    """

    df = read_dataframe(
        query
    )


    if df.empty:
        raise ValueError(
            "Database contains no data."
        )

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp"]
    )

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    for col in POLLUTANTS:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # REMOVE NEGATIVE VALUES
    # --------------------------------------------------------

    for col in POLLUTANTS:

        df.loc[
            df[col] < 0,
            col
        ] = np.nan

    # --------------------------------------------------------
    # SORT / DUPLICATES
    # --------------------------------------------------------

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "First timestamp:",
        df["timestamp"].iloc[0]
    )

    print(
        "Last timestamp:",
        df["timestamp"].iloc[-1]
    )

    print()
    print("Missing values:")

    print(
        df[POLLUTANTS]
        .isna()
        .sum()
    )

    return df


# ============================================================
# IMPUTE MISSING VALUES
#
# IMPORTANT:
# This is for the live prediction pipeline.
#
# Historical database values are reconstructed using the
# same general feature-preparation logic used during training.
# ============================================================

def impute_data(df):

    data = df.copy()

    for col in POLLUTANTS:

        data[col] = (
            data[col]
            .interpolate(
                method="linear",
                limit_direction="both"
            )
        )

        data[col] = (
            data[col]
            .fillna(
                data[col].median()
            )
        )

    return data


# ============================================================
# CREATE CPCB AQI
# ============================================================

def create_aqi(df):

    data = df.copy()

    print()
    print("=" * 80)
    print("CALCULATING CPCB ROLLING CONCENTRATIONS")
    print("=" * 80)

    # --------------------------------------------------------
    # 24-HOUR POLLUTANTS
    # --------------------------------------------------------

    for col in [
        "pm25",
        "pm10",
        "no2",
        "nh3",
        "so2"
    ]:

        data[f"{col}_24h"] = (
            data[col]
            .rolling(
                window=24,
                min_periods=24
            )
            .mean()
        )

    # --------------------------------------------------------
    # 8-HOUR POLLUTANTS
    # --------------------------------------------------------

    for col in [
        "co",
        "o3"
    ]:

        data[f"{col}_8h"] = (
            data[col]
            .rolling(
                window=8,
                min_periods=8
            )
            .mean()
        )

    # --------------------------------------------------------
    # AQI
    # --------------------------------------------------------

    print(
        "Calculating CPCB AQI..."
    )

    data["aqi"] = data.apply(
        calculate_cpcb_aqi,
        axis=1
    )

    data["aqi"] = pd.to_numeric(
        data["aqi"],
        errors="coerce"
    )

    return data


# ============================================================
# CREATE FEATURES
#
# MUST MATCH TRAINING CODE
#
# NO IS NOT USED ANYWHERE.
# ============================================================

def create_features(df):

    data = (
        df
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    feature_columns = []

    # ========================================================
    # AQI LAGS
    # ========================================================

    print(
        "Creating historical AQI lags..."
    )

    for lag in AQI_LAGS:

        name = f"aqi_lag_{lag}"

        data[name] = (
            data["aqi"]
            .shift(lag)
        )

        feature_columns.append(
            name
        )

    # ========================================================
    # POLLUTANT LAGS
    # ========================================================

    print(
        "Creating pollutant lags..."
    )

    for pollutant in POLLUTANTS:

        for lag in POLLUTANT_LAGS:

            name = (
                f"{pollutant}_lag_{lag}"
            )

            data[name] = (
                data[pollutant]
                .shift(lag)
            )

            feature_columns.append(
                name
            )

    # ========================================================
    # AQI ROLLING STATISTICS
    # ========================================================

    print(
        "Creating AQI rolling statistics..."
    )

    for window in ROLLING_WINDOWS:

        prefix = (
            f"aqi_roll_{window}h"
        )

        data[f"{prefix}_mean"] = (
            data["aqi"]
            .rolling(
                window=window,
                min_periods=window
            )
            .mean()
        )

        data[f"{prefix}_max"] = (
            data["aqi"]
            .rolling(
                window=window,
                min_periods=window
            )
            .max()
        )

        data[f"{prefix}_min"] = (
            data["aqi"]
            .rolling(
                window=window,
                min_periods=window
            )
            .min()
        )

        data[f"{prefix}_std"] = (
            data["aqi"]
            .rolling(
                window=window,
                min_periods=window
            )
            .std()
        )

        feature_columns.extend([
            f"{prefix}_mean",
            f"{prefix}_max",
            f"{prefix}_min",
            f"{prefix}_std"
        ])

    # ========================================================
    # POLLUTANT ROLLING STATISTICS
    # ========================================================

    print(
        "Creating pollutant rolling statistics..."
    )

    for pollutant in POLLUTANTS:

        for window in ROLLING_WINDOWS:

            prefix = (
                f"{pollutant}_roll_{window}h"
            )

            data[f"{prefix}_mean"] = (
                data[pollutant]
                .rolling(
                    window=window,
                    min_periods=window
                )
                .mean()
            )

            data[f"{prefix}_std"] = (
                data[pollutant]
                .rolling(
                    window=window,
                    min_periods=window
                )
                .std()
            )

            feature_columns.extend([
                f"{prefix}_mean",
                f"{prefix}_std"
            ])

    # ========================================================
    # POLLUTANT CHANGE FEATURES
    # ========================================================

    print(
        "Creating pollutant change features..."
    )

    for pollutant in POLLUTANTS:

        for lag in CHANGE_LAGS:

            name = (
                f"{pollutant}_change_{lag}h"
            )

            data[name] = (
                data[pollutant]
                -
                data[pollutant].shift(lag)
            )

            feature_columns.append(
                name
            )

    # ========================================================
    # AQI CHANGE FEATURES
    # ========================================================

    print(
        "Creating AQI change features..."
    )

    for lag in CHANGE_LAGS:

        name = (
            f"aqi_change_{lag}h"
        )

        data[name] = (
            data["aqi"]
            -
            data["aqi"].shift(lag)
        )

        feature_columns.append(
            name
        )

    # ========================================================
    # POLLUTANT TREND FEATURES
    # ========================================================

    print(
        "Creating pollutant trend features..."
    )

    for pollutant in POLLUTANTS:

        for window in TREND_WINDOWS:

            name = (
                f"{pollutant}_trend_{window}h"
            )

            data[name] = (
                data[pollutant].diff(window)
                /
                window
            )

            feature_columns.append(
                name
            )

    # ========================================================
    # AQI TREND FEATURES
    # ========================================================

    print(
        "Creating AQI trend features..."
    )

    for window in TREND_WINDOWS:

        name = (
            f"aqi_trend_{window}h"
        )

        data[name] = (
            data["aqi"].diff(window)
            /
            window
        )

        feature_columns.append(
            name
        )

    # ========================================================
    # POLLUTANT ACCELERATION
    # ========================================================

    print(
        "Creating pollutant acceleration features..."
    )

    for pollutant in POLLUTANTS:

        change_1h = (
            data[pollutant].diff(1)
        )

        change_3h = (
            data[pollutant].diff(3)
        )

        name = (
            f"{pollutant}_acceleration"
        )

        data[name] = (
            change_1h
            -
            change_3h / 3
        )

        feature_columns.append(
            name
        )

    # ========================================================
    # AQI ACCELERATION
    # ========================================================

    print(
        "Creating AQI acceleration..."
    )

    aqi_change_1h = (
        data["aqi"].diff(1)
    )

    aqi_change_3h = (
        data["aqi"].diff(3)
    )

    data["aqi_acceleration"] = (
        aqi_change_1h
        -
        aqi_change_3h / 3
    )

    feature_columns.append(
        "aqi_acceleration"
    )

    # ========================================================
    # PM2.5 / PM10 RATIO
    # ========================================================

    print(
        "Creating pollutant ratio features..."
    )

    data["pm25_pm10_ratio"] = (
        data["pm25"]
        /
        data["pm10"].replace(
            0,
            np.nan
        )
    )

    feature_columns.append(
        "pm25_pm10_ratio"
    )

    # ========================================================
    # PM2.5 DOMINANCE
    # ========================================================

    data["pm25_dominance"] = (
        data["pm25"]
        /
        (
            data["pm25"]
            +
            data["pm10"]
            +
            1e-6
        )
    )

    feature_columns.append(
        "pm25_dominance"
    )

    # ========================================================
    # CYCLICAL TIME FEATURES
    # ========================================================

    print(
        "Creating cyclical time features..."
    )

    hour = (
        data["timestamp"].dt.hour
    )

    day_of_week = (
        data["timestamp"].dt.dayofweek
    )

    day_of_year = (
        data["timestamp"].dt.dayofyear
    )

    # --------------------------------------------------------
    # Hour
    # --------------------------------------------------------

    data["sin_hour"] = (
        np.sin(
            2
            * np.pi
            * hour
            / 24
        )
    )

    data["cos_hour"] = (
        np.cos(
            2
            * np.pi
            * hour
            / 24
        )
    )

    # --------------------------------------------------------
    # Day of week
    # --------------------------------------------------------

    data["sin_day"] = (
        np.sin(
            2
            * np.pi
            * day_of_week
            / 7
        )
    )

    data["cos_day"] = (
        np.cos(
            2
            * np.pi
            * day_of_week
            / 7
        )
    )

    # --------------------------------------------------------
    # Day of year
    # --------------------------------------------------------

    data["sin_year"] = (
        np.sin(
            2
            * np.pi
            * day_of_year
            / 365.25
        )
    )

    data["cos_year"] = (
        np.cos(
            2
            * np.pi
            * day_of_year
            / 365.25
        )
    )

    feature_columns.extend([
        "sin_hour",
        "cos_hour",
        "sin_day",
        "cos_day",
        "sin_year",
        "cos_year"
    ])

    # ========================================================
    # FORCE NUMERIC
    # ========================================================

    for col in feature_columns:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # ========================================================
    # REMOVE INFINITY
    # ========================================================

    data[feature_columns] = (
        data[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    # ========================================================
    # FEATURE COUNT CHECK
    # ========================================================

    print()
    print("=" * 80)
    print("GENERATED FEATURE SUMMARY")
    print("=" * 80)

    print(
        "Generated features:",
        len(feature_columns)
    )

    return (
        data,
        feature_columns
    )


# ============================================================
# LOAD ALL MODELS
# ============================================================

def load_models():

    models = {}

    print()
    print("=" * 80)
    print("LOADING XGBOOST DELTA-ONLY MODELS")
    print("=" * 80)

    for horizon in range(1, 25):

        folder = os.path.join(
            MODEL_BASE,
            f"horizon_{horizon:02d}h"
        )

        model_path = os.path.join(
            folder,
            "xgboost_delta_model.joblib"
        )

        feature_path = os.path.join(
            folder,
            "feature_columns.joblib"
        )

        config_path = os.path.join(
            folder,
            "model_config.joblib"
        )

        # ----------------------------------------------------
        # CHECK FILES
        # ----------------------------------------------------

        if not os.path.exists(
            model_path
        ):

            raise FileNotFoundError(
                f"Model not found:\n{model_path}"
            )

        if not os.path.exists(
            feature_path
        ):

            raise FileNotFoundError(
                f"Feature list not found:\n{feature_path}"
            )

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        model = joblib.load(
            model_path
        )

        feature_columns = joblib.load(
            feature_path
        )

        config = None

        if os.path.exists(
            config_path
        ):

            config = joblib.load(
                config_path
            )

        models[horizon] = {
            "model": model,
            "features": feature_columns,
            "config": config
        }

        print(
            f"Horizon {horizon:02d}h: "
            f"{len(feature_columns)} features loaded"
        )

        # ----------------------------------------------------
        # EXPECTED FEATURE COUNT
        # ----------------------------------------------------

        if len(feature_columns) != 224:

            raise ValueError(
                f"Horizon {horizon:02d}h model has "
                f"{len(feature_columns)} features. "
                f"Expected 224."
            )

    return models


# ============================================================
# VALIDATE FEATURES
# ============================================================

def validate_model_features(
    available_features,
    models
):

    print()
    print("=" * 80)
    print("VALIDATING MODEL FEATURES")
    print("=" * 80)

    available_set = set(
        available_features
    )

    for horizon in range(1, 25):

        model_features = models[horizon][
            "features"
        ]

        model_set = set(
            model_features
        )

        missing = (
            model_set
            -
            available_set
        )

        extra = (
            available_set
            -
            model_set
        )

        if missing:

            raise ValueError(
                f"\nHorizon {horizon:02d}h "
                f"is missing features:\n"
                f"{sorted(missing)}"
            )

        if len(model_features) != len(
            available_features
        ):

            print(
                f"Warning: horizon {horizon:02d}h "
                f"feature count differs from generated "
                f"features."
            )

        print(
            f"Horizon {horizon:02d}h: "
            f"feature validation OK"
        )

    print()
    print(
        "All model feature lists validated."
    )


# ============================================================
# PREDICT
# ============================================================

def make_predictions():

    # ========================================================
    # LOAD DATABASE
    # ========================================================

    df = load_database()

    # ========================================================
    # IMPUTE
    # ========================================================

    print()
    print(
        "Imputing missing pollutant values..."
    )

    df = impute_data(
        df
    )

    # ========================================================
    # CREATE CPCB AQI
    # ========================================================

    df = create_aqi(
        df
    )

    # ========================================================
    # CREATE FEATURES
    # ========================================================

    data, available_features = (
        create_features(
            df
        )
    )

    # ========================================================
    # LOAD MODELS
    # ========================================================

    models = load_models()

    # ========================================================
    # VALIDATE FEATURES
    # ========================================================

    validate_model_features(
        available_features,
        models
    )

    # ========================================================
    # LAST VALID AQI ROW
    # ========================================================

    valid_rows = data.dropna(
        subset=["aqi"]
    )

    if valid_rows.empty:

        raise ValueError(
            "No valid AQI rows available."
        )

    latest = (
        valid_rows.iloc[-1]
    )

    latest_timestamp = (
        latest["timestamp"]
    )

    current_aqi = float(
        latest["aqi"]
    )

    # ========================================================
    # FORECAST ORIGIN
    # ========================================================

    print()
    print("=" * 80)
    print("LATEST INPUT POLLUTANTS")
    print("=" * 80)

    print("Timestamp:", latest_timestamp)
    print("PM2.5:", latest["pm25"])
    print("PM10 :", latest["pm10"])
    print("CO   :", latest["co"])
    print("NH3  :", latest["nh3"])
    print("NO2  :", latest["no2"])
    print("SO2  :", latest["so2"])
    print("O3   :", latest["o3"])

    print()
    print("=" * 80)
    print("FORECAST ORIGIN")
    print("=" * 80)

    print(
        "Timestamp:",
        latest_timestamp
    )

    print(
        "Current CPCB AQI:",
        round(
            current_aqi,
            3
        )
    )

    print(
        "Generated features:",
        len(available_features)
    )

    # ========================================================
    # CHECK CURRENT ROW
    # ========================================================

    if latest[
        available_features
    ].isna().any():

        nan_features = (
            latest[
                available_features
            ]
            .index[
                latest[
                    available_features
                ].isna()
            ]
            .tolist()
        )

        print()
        print("=" * 80)
        print("ERROR: CURRENT ROW CONTAINS NaN FEATURES")
        print("=" * 80)

        for feature in nan_features:
            print(
                feature
            )

        raise ValueError(
            "Current forecast row contains NaN features."
        )

    # ========================================================
    # PREDICT ALL 24 HORIZONS
    # ========================================================

    results = []

    print()
    print("=" * 80)
    print("GENERATING 1–24 HOUR DELTA FORECAST")
    print("=" * 80)

    for horizon in range(
        1,
        25
    ):

        model = models[horizon][
            "model"
        ]

        feature_columns = models[horizon][
            "features"
        ]

        # ----------------------------------------------------
        # CHECK FEATURES
        # ----------------------------------------------------

        missing_features = [
            col
            for col in feature_columns
            if col not in data.columns
        ]

        if missing_features:

            raise ValueError(
                f"Missing features for "
                f"{horizon}h:\n"
                f"{missing_features}"
            )

        # ----------------------------------------------------
        # MODEL INPUT
        # ----------------------------------------------------

        X = (
            latest[
                feature_columns
            ]
            .to_frame()
            .T
        )

        X = X.astype(
            np.float32
        )

        # ----------------------------------------------------
        # NaN CHECK
        # ----------------------------------------------------

        if X.isna().any().any():

            nan_features = (
                X.columns[
                    X.isna().any()
                ]
                .tolist()
            )

            raise ValueError(
                f"NaN feature found for "
                f"{horizon}h model:\n"
                f"{nan_features}"
            )

        # ----------------------------------------------------
        # PREDICT DELTA
        # ----------------------------------------------------

        predicted_delta = float(
            model.predict(
                X
            )[0]
        )

        # ----------------------------------------------------
        # DELTA → AQI
        #
        # Future AQI =
        # Current AQI + Predicted Delta
        # ----------------------------------------------------

        predicted_aqi = (
            current_aqi
            +
            predicted_delta
        )

        # ----------------------------------------------------
        # CPCB AQI RANGE
        # ----------------------------------------------------

        predicted_aqi = float(
            np.clip(
                predicted_aqi,
                0,
                500
            )
        )

        # ----------------------------------------------------
        # FORECAST TIMESTAMP
        # ----------------------------------------------------

        forecast_timestamp = (
            latest_timestamp
            +
            pd.Timedelta(
                hours=horizon
            )
        )

        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        results.append({

            "forecast_origin":
                latest_timestamp,

            "forecast_timestamp":
                forecast_timestamp,

            "horizon_hours":
                horizon,

            "current_aqi":
                current_aqi,

            "predicted_delta":
                predicted_delta,

            "predicted_aqi":
                predicted_aqi

        })

        print(
            f"{horizon:02d}h | "
            f"Delta = {predicted_delta:8.3f} | "
            f"Predicted AQI = {predicted_aqi:8.3f}"
        )

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # SAVE
    # ========================================================

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

        # ========================================================
    # SAVE TO POSTGRESQL
    # ========================================================

    save_predictions_to_postgres(
        results_df
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("=" * 80)
    print("VADODARA 24-HOUR AQI FORECAST")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE CONFIRMATION
    # ========================================================

    print()
    print("=" * 80)
    print("RESULT SAVED")
    print("=" * 80)

    print(
        os.path.abspath(
            OUTPUT_FILE
        )
    )

    return results_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    make_predictions()