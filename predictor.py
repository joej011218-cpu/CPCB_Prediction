import os
import gc

from database import read_dataframe
from prediction_storage import save_predictions_to_postgres

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_PATH = "data/cpcb_vadodara.db"

# ------------------------------------------------------------
# RANDOM FOREST MODEL LOCATION
#
# LOCAL DEFAULT:
#   models/Vadodara/Random Forest
#
# GITHUB ACTIONS / CLOUD:
#   RF_MODEL_BASE=RF_Deployment
#
# This allows the same predictor.py to work locally and online.
# ------------------------------------------------------------

MODEL_BASE = os.getenv(
    "RF_MODEL_BASE",
    "models/Vadodara/Random Forest"
)

# ------------------------------------------------------------
# RANDOM FOREST MODEL FILENAME
#
# LOCAL DEFAULT:
#   random_forest_delta_model_compressed.joblib
#
# CLOUD:
#   random_forest_delta_model.joblib
# ------------------------------------------------------------

MODEL_FILENAME = os.getenv(
    "RF_MODEL_FILENAME",
    "random_forest_delta_model_compressed.joblib"
)

OUTPUT_FILE = "vadodara_live_predictions.csv"


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

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
# POLLUTANTS
#
# NO IS EXCLUDED.
#
# These seven pollutants match the 224-feature training setup.
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
# CALCULATE SUB-INDEX
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
# CALCULATE CPCB AQI
# ============================================================

def calculate_cpcb_aqi(row):

    subindices = []

    pollutant_values = [
        ("pm25_24h", PM25_BREAKPOINTS),
        ("pm10_24h", PM10_BREAKPOINTS),
        ("no2_24h", NO2_BREAKPOINTS),
        ("nh3_24h", NH3_BREAKPOINTS),
        ("so2_24h", SO2_BREAKPOINTS),
        ("co_8h", CO_BREAKPOINTS),
        ("o3_8h", O3_BREAKPOINTS),
    ]

    for column, breakpoints in pollutant_values:

        value = calculate_subindex(
            row[column],
            breakpoints
        )

        if not pd.isna(value):
            subindices.append(value)

    if not subindices:
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

    df = read_dataframe(query)

    if df.empty:
        raise ValueError(
            "Database contains no data."
        )

    # Timestamp conversion
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp"]
    )

    # Numeric conversion
    for col in POLLUTANTS:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Remove negative pollutant values
    for col in POLLUTANTS:

        df.loc[
            df[col] < 0,
            col
        ] = np.nan

    # Sort and remove duplicate timestamps
    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    print("Rows:", len(df))
    print("First timestamp:", df["timestamp"].iloc[0])
    print("Last timestamp:", df["timestamp"].iloc[-1])

    print()
    print("Missing values:")

    print(
        df[POLLUTANTS]
        .isna()
        .sum()
    )

    return df


# ============================================================
# IMPUTE MISSING DATA
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

    print("Calculating CPCB AQI...")

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
# CREATE 224 FEATURES
#
# MUST MATCH TRAINING FEATURE ENGINEERING.
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

    print("Creating historical AQI lags...")

    for lag in AQI_LAGS:

        name = f"aqi_lag_{lag}"

        data[name] = (
            data["aqi"]
            .shift(lag)
        )

        feature_columns.append(name)

    # ========================================================
    # POLLUTANT LAGS
    # ========================================================

    print("Creating pollutant lags...")

    for pollutant in POLLUTANTS:

        for lag in POLLUTANT_LAGS:

            name = (
                f"{pollutant}_lag_{lag}"
            )

            data[name] = (
                data[pollutant]
                .shift(lag)
            )

            feature_columns.append(name)

    # ========================================================
    # AQI ROLLING STATISTICS
    # ========================================================

    print("Creating AQI rolling statistics...")

    for window in ROLLING_WINDOWS:

        prefix = f"aqi_roll_{window}h"

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

    print("Creating pollutant rolling statistics...")

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

    print("Creating pollutant change features...")

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

            feature_columns.append(name)

    # ========================================================
    # AQI CHANGE FEATURES
    # ========================================================

    print("Creating AQI change features...")

    for lag in CHANGE_LAGS:

        name = (
            f"aqi_change_{lag}h"
        )

        data[name] = (
            data["aqi"]
            -
            data["aqi"].shift(lag)
        )

        feature_columns.append(name)

    # ========================================================
    # POLLUTANT TREND FEATURES
    # ========================================================

    print("Creating pollutant trend features...")

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

            feature_columns.append(name)

    # ========================================================
    # AQI TREND FEATURES
    # ========================================================

    print("Creating AQI trend features...")

    for window in TREND_WINDOWS:

        name = (
            f"aqi_trend_{window}h"
        )

        data[name] = (
            data["aqi"].diff(window)
            /
            window
        )

        feature_columns.append(name)

    # ========================================================
    # POLLUTANT ACCELERATION
    # ========================================================

    print("Creating pollutant acceleration features...")

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

        feature_columns.append(name)

    # ========================================================
    # AQI ACCELERATION
    # ========================================================

    print("Creating AQI acceleration...")

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

    print("Creating pollutant ratio features...")

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

    print("Creating cyclical time features...")

    hour = (
        data["timestamp"].dt.hour
    )

    day_of_week = (
        data["timestamp"].dt.dayofweek
    )

    day_of_year = (
        data["timestamp"].dt.dayofyear
    )

    data["sin_hour"] = (
        np.sin(
            2 * np.pi * hour / 24
        )
    )

    data["cos_hour"] = (
        np.cos(
            2 * np.pi * hour / 24
        )
    )

    data["sin_day"] = (
        np.sin(
            2 * np.pi * day_of_week / 7
        )
    )

    data["cos_day"] = (
        np.cos(
            2 * np.pi * day_of_week / 7
        )
    )

    data["sin_year"] = (
        np.sin(
            2 * np.pi * day_of_year / 365.25
        )
    )

    data["cos_year"] = (
        np.cos(
            2 * np.pi * day_of_year / 365.25
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

    if len(feature_columns) != 224:

        raise ValueError(
            f"Generated {len(feature_columns)} features. "
            f"Expected 224."
        )

    return data, feature_columns


# ============================================================
# GET RANDOM FOREST MODEL FILES
#
# IMPORTANT:
# Only ONE RF model is loaded at a time.
#
# This prevents all 24 large RF models from being kept
# in memory simultaneously.
# ============================================================

def get_model_files(horizon):

    folder = os.path.join(
        MODEL_BASE,
        f"horizon_{horizon:02d}h"
    )

    model_path = os.path.join(
        folder,
        MODEL_FILENAME
    )

    feature_path = os.path.join(
        folder,
        "feature_columns.joblib"
    )

    config_path = os.path.join(
        folder,
        "model_config.joblib"
    )

    if not os.path.isdir(folder):

        raise FileNotFoundError(
            f"Random Forest model folder not found:\n"
            f"{folder}"
        )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Random Forest model not found:\n"
            f"{model_path}"
        )

    if not os.path.exists(feature_path):

        raise FileNotFoundError(
            f"Feature list not found:\n"
            f"{feature_path}"
        )

    feature_columns = joblib.load(
        feature_path
    )

    if len(feature_columns) != 224:

        raise ValueError(
            f"Horizon {horizon:02d}h has "
            f"{len(feature_columns)} model features. "
            f"Expected 224."
        )

    config = None

    if os.path.exists(config_path):

        config = joblib.load(
            config_path
        )

    return (
        model_path,
        feature_columns,
        config
    )


# ============================================================
# VALIDATE MODEL FEATURES
# ============================================================

def validate_model_features(
    horizon,
    available_features,
    model_features
):

    available_set = set(
        available_features
    )

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
            f"Horizon {horizon:02d}h requires "
            f"features that were not generated:\n"
            f"{sorted(missing)}"
        )

    if extra:

        raise ValueError(
            f"Horizon {horizon:02d}h generated "
            f"unexpected features:\n"
            f"{sorted(extra)}"
        )

    if len(model_features) != 224:

        raise ValueError(
            f"Horizon {horizon:02d}h expected "
            f"224 model features but found "
            f"{len(model_features)}."
        )


# ============================================================
# DISPLAY DEPLOYMENT CONFIGURATION
# ============================================================

def display_model_configuration():

    print()
    print("=" * 80)
    print("RANDOM FOREST DEPLOYMENT CONFIGURATION")
    print("=" * 80)

    print(
        "Model base:",
        os.path.abspath(MODEL_BASE)
    )

    print(
        "Model filename:",
        MODEL_FILENAME
    )

    print(
        "Model type:",
        "Random Forest Delta-only"
    )

    print(
        "Forecast horizons:",
        "1-24 hours"
    )

    print(
        "Expected features:",
        224
    )


# ============================================================
# MAKE PREDICTIONS
# ============================================================

def make_predictions():

    # --------------------------------------------------------
    # DISPLAY MODEL CONFIGURATION
    # --------------------------------------------------------

    display_model_configuration()

    # --------------------------------------------------------
    # LOAD DATABASE
    # --------------------------------------------------------

    df = load_database()

    # --------------------------------------------------------
    # IMPUTE MISSING DATA
    # --------------------------------------------------------

    print()
    print("Imputing missing pollutant values...")

    df = impute_data(
        df
    )

    # --------------------------------------------------------
    # CALCULATE AQI
    # --------------------------------------------------------

    df = create_aqi(
        df
    )

    # --------------------------------------------------------
    # CREATE FEATURES
    # --------------------------------------------------------

    data, available_features = (
        create_features(
            df
        )
    )

    # --------------------------------------------------------
    # GET LAST VALID AQI ROW
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DISPLAY CURRENT INPUT
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LATEST INPUT POLLUTANTS")
    print("=" * 80)

    print(
        "Timestamp:",
        latest_timestamp
    )

    print(
        "PM2.5:",
        latest["pm25"]
    )

    print(
        "PM10 :",
        latest["pm10"]
    )

    print(
        "CO   :",
        latest["co"]
    )

    print(
        "NH3  :",
        latest["nh3"]
    )

    print(
        "NO2  :",
        latest["no2"]
    )

    print(
        "SO2  :",
        latest["so2"]
    )

    print(
        "O3   :",
        latest["o3"]
    )

    # --------------------------------------------------------
    # FORECAST ORIGIN
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CURRENT FEATURE NaN CHECK
    # --------------------------------------------------------

    current_features = (
        latest[
            available_features
        ]
    )

    if current_features.isna().any():

        nan_features = (
            current_features
            .index[
                current_features.isna()
            ]
            .tolist()
        )

        print()
        print("=" * 80)
        print("ERROR: CURRENT ROW CONTAINS NaN FEATURES")
        print("=" * 80)

        for feature in nan_features:
            print(feature)

        raise ValueError(
            "Current forecast row contains NaN features."
        )

    # --------------------------------------------------------
    # GENERATE 24 FORECASTS
    # --------------------------------------------------------

    results = []

    print()
    print("=" * 80)
    print(
        "GENERATING 1-24 HOUR RANDOM FOREST DELTA FORECAST"
    )
    print("=" * 80)

    for horizon in range(
        1,
        25
    ):

        print()
        print(
            f"[{horizon:02d}/24] "
            f"Preparing Random Forest horizon..."
        )

        # ----------------------------------------------------
        # GET MODEL INFORMATION
        # ----------------------------------------------------

        (
            model_path,
            feature_columns,
            config
        ) = get_model_files(
            horizon
        )

        # ----------------------------------------------------
        # VALIDATE FEATURES
        # ----------------------------------------------------

        validate_model_features(
            horizon,
            available_features,
            feature_columns
        )

        print(
            f"[{horizon:02d}/24] "
            f"Feature validation OK "
            f"({len(feature_columns)} features)"
        )

        # ----------------------------------------------------
        # CREATE INPUT IN EXACT TRAINING FEATURE ORDER
        # ----------------------------------------------------

        X = (
            latest[
                feature_columns
            ]
            .to_frame()
            .T
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # INPUT VALIDATION
        # ----------------------------------------------------

        if X.isna().any().any():

            nan_features = (
                X.columns[
                    X.isna().any()
                ]
                .tolist()
            )

            raise ValueError(
                f"NaN features found for "
                f"horizon {horizon:02d}h:\n"
                f"{nan_features}"
            )

        if np.isinf(
            X.to_numpy()
        ).any():

            raise ValueError(
                f"Infinite feature value found for "
                f"horizon {horizon:02d}h."
            )

        # ----------------------------------------------------
        # LOAD ONLY CURRENT RANDOM FOREST
        # ----------------------------------------------------

        print(
            f"[{horizon:02d}/24] "
            f"Loading Random Forest..."
        )

        model = joblib.load(
            model_path
        )

        model_name = (
            type(model).__name__
        )

        print(
            f"[{horizon:02d}/24] "
            f"Loaded: {model_name}"
        )

        # ----------------------------------------------------
        # VERIFY MODEL TYPE
        # ----------------------------------------------------

        if model_name != "RandomForestRegressor":

            print(
                f"WARNING: Horizon {horizon:02d}h "
                f"loaded model type is {model_name}"
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
        # RELEASE LARGE MODEL IMMEDIATELY
        # ----------------------------------------------------

        del model

        gc.collect()

        # ----------------------------------------------------
        # DELTA -> FINAL AQI
        # ----------------------------------------------------

        predicted_aqi = (
            current_aqi
            +
            predicted_delta
        )

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

    # --------------------------------------------------------
    # CREATE RESULTS DATAFRAME
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # VERIFY 24 FORECASTS
    # --------------------------------------------------------

    if len(results_df) != 24:

        raise ValueError(
            f"Expected 24 forecast rows, "
            f"but generated {len(results_df)}."
        )

    # --------------------------------------------------------
    # SAVE LOCAL CSV
    # --------------------------------------------------------

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        "Local prediction CSV saved successfully."
    )

    # --------------------------------------------------------
    # SAVE TO POSTGRESQL
    # --------------------------------------------------------

    try:

        save_predictions_to_postgres(
            results_df
        )

        print(
            "PostgreSQL prediction save step completed."
        )

    except Exception as exc:

        print()
        print(
            "WARNING: PostgreSQL prediction save failed."
        )

        print(
            "Reason:",
            exc
        )

        print(
            "The Random Forest forecasts were still "
            "generated and saved locally."
        )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "VADODARA RANDOM FOREST 24-HOUR AQI FORECAST"
    )
    print("=" * 80)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # FINAL CONFIRMATION
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RESULT SAVED")
    print("=" * 80)

    print(
        "CSV:",
        os.path.abspath(
            OUTPUT_FILE
        )
    )

    print(
        "Model base:",
        os.path.abspath(
            MODEL_BASE
        )
    )

    print(
        "Model filename:",
        MODEL_FILENAME
    )

    print()
    print(
        "24 Random Forest forecasts generated successfully."
    )

    return results_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    make_predictions()