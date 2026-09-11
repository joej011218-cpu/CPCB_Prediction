import os

import numpy as np
import pandas as pd
from flask import Flask, jsonify

from database import (
    database_available,
    read_dataframe,
    using_postgres,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cpcb_vadodara.db"
)

PREDICTION_FILE = os.path.join(
    BASE_DIR,
    "vadodara_live_predictions.csv"
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# CORS
# ============================================================

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    return response


# ============================================================
# POLLUTANTS
# ============================================================

POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
    "nh3",
    "so2",
    "co",
    "o3",
]


# ============================================================
# CPCB AQI BREAKPOINTS
# ============================================================

PM25_BREAKPOINTS = [
    (0, 30, 0, 50),
    (31, 60, 51, 100),
    (61, 90, 101, 200),
    (91, 120, 201, 300),
    (121, 250, 301, 400),
    (251, np.inf, 401, 500),
]

PM10_BREAKPOINTS = [
    (0, 50, 0, 50),
    (51, 100, 51, 100),
    (101, 250, 101, 200),
    (251, 350, 201, 300),
    (351, 430, 301, 400),
    (431, np.inf, 401, 500),
]

NO2_BREAKPOINTS = [
    (0, 40, 0, 50),
    (41, 80, 51, 100),
    (81, 180, 101, 200),
    (181, 280, 201, 300),
    (281, 400, 301, 400),
    (401, np.inf, 401, 500),
]

NH3_BREAKPOINTS = [
    (0, 200, 0, 50),
    (201, 400, 51, 100),
    (401, 800, 101, 200),
    (801, 1200, 201, 300),
    (1201, 1800, 301, 400),
    (1801, np.inf, 401, 500),
]

SO2_BREAKPOINTS = [
    (0, 40, 0, 50),
    (41, 80, 51, 100),
    (81, 380, 101, 200),
    (381, 800, 201, 300),
    (801, 1600, 301, 400),
    (1601, np.inf, 401, 500),
]

CO_BREAKPOINTS = [
    (0, 1.0, 0, 50),
    (1.1, 2.0, 51, 100),
    (2.1, 10.0, 101, 200),
    (10.1, 17.0, 201, 300),
    (17.1, 34.0, 301, 400),
    (34.1, np.inf, 401, 500),
]

O3_BREAKPOINTS = [
    (0, 50, 0, 50),
    (51, 100, 51, 100),
    (101, 168, 101, 200),
    (169, 208, 201, 300),
    (209, 748, 301, 400),
    (749, np.inf, 401, 500),
]


# ============================================================
# SUBINDEX
# ============================================================

def calculate_subindex(value, breakpoints):
    if pd.isna(value):
        return np.nan

    try:
        value = float(value)
    except Exception:
        return np.nan

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

    value = calculate_subindex(
        row.get("pm25_24h"),
        PM25_BREAKPOINTS,
    )
    if not pd.isna(value):
        subindices.append(value)

    value = calculate_subindex(
        row.get("pm10_24h"),
        PM10_BREAKPOINTS,
    )
    if not pd.isna(value):
        subindices.append(value)

    value = calculate_subindex(
        row.get("no2_24h"),
        NO2_BREAKPOINTS,
    )
    if not pd.isna(value):
        subindices.append(value)

    value = calculate_subindex(
        row.get("nh3_24h"),
        NH3_BREAKPOINTS,
    )
    if not pd.isna(value):
        subindices.append(value)

    value = calculate_subindex(
        row.get("so2_24h"),
        SO2_BREAKPOINTS,
    )
    if not pd.isna(value):
        subindices.append(value)

    value = calculate_subindex(
        row.get("co_8h"),
        CO_BREAKPOINTS,
    )
    if not pd.isna(value):
        subindices.append(value)

    value = calculate_subindex(
        row.get("o3_8h"),
        O3_BREAKPOINTS,
    )
    if not pd.isna(value):
        subindices.append(value)

    if not subindices:
        return np.nan

    return max(subindices)


# ============================================================
# LOAD CPCB DATABASE
# ============================================================

def load_cpcb_database():
    if not database_available():
        return None

    try:
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

    except Exception as e:
        print("Database error:", str(e))
        return None

    if df.empty:
        return None

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["timestamp"]
    )

    for col in POLLUTANTS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    for col in POLLUTANTS:
        df.loc[
            df[col] < 0,
            col,
        ] = np.nan

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# IMPUTE DATA
# ============================================================

def impute_data(df):
    data = df.copy()

    for col in POLLUTANTS:
        data[col] = (
            data[col]
            .interpolate(
                method="linear",
                limit_direction="both",
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
# CREATE CPCB ROLLING AQI
# ============================================================

def create_cpcb_aqi_dataframe(df):
    data = df.copy()

    for col in [
        "pm25",
        "pm10",
        "no2",
        "nh3",
        "so2",
    ]:
        data[f"{col}_24h"] = (
            data[col]
            .rolling(
                window=24,
                min_periods=24,
            )
            .mean()
        )

    for col in [
        "co",
        "o3",
    ]:
        data[f"{col}_8h"] = (
            data[col]
            .rolling(
                window=8,
                min_periods=8,
            )
            .mean()
        )

    data["aqi"] = data.apply(
        calculate_cpcb_aqi,
        axis=1,
    )

    data["aqi"] = pd.to_numeric(
        data["aqi"],
        errors="coerce",
    )

    return data


# ============================================================
# GET CURRENT DATA
# ============================================================

def get_current_data():
    df = load_cpcb_database()

    if df is None:
        return None

    df = impute_data(df)
    df = create_cpcb_aqi_dataframe(df)

    if df.empty:
        return None

    latest = df.iloc[-1]
    timestamp = latest["timestamp"]

    return {
        "timestamp":
            timestamp.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if not pd.isna(timestamp)
            else None,

        "aqi":
            None
            if pd.isna(latest["aqi"])
            else float(latest["aqi"]),

        "pm25":
            None
            if pd.isna(latest["pm25"])
            else float(latest["pm25"]),

        "pm10":
            None
            if pd.isna(latest["pm10"])
            else float(latest["pm10"]),

        "no2":
            None
            if pd.isna(latest["no2"])
            else float(latest["no2"]),

        "nh3":
            None
            if pd.isna(latest["nh3"])
            else float(latest["nh3"]),

        "so2":
            None
            if pd.isna(latest["so2"])
            else float(latest["so2"]),

        "co":
            None
            if pd.isna(latest["co"])
            else float(latest["co"]),

        "o3":
            None
            if pd.isna(latest["o3"])
            else float(latest["o3"]),
    }


# ============================================================
# LAST 24 HOURS
# ============================================================

def get_last_24_hours():
    df = load_cpcb_database()

    if df is None:
        return None

    df = impute_data(df)
    df = create_cpcb_aqi_dataframe(df)

    if df.empty:
        return None

    df = (
        df
        .sort_values("timestamp")
        .tail(24)
    )

    records = []

    for _, row in df.iterrows():
        timestamp = row["timestamp"]

        record = {
            "timestamp":
                timestamp.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                if not pd.isna(timestamp)
                else None,

            "aqi":
                None
                if pd.isna(row["aqi"])
                else float(row["aqi"]),

            "pm25":
                None
                if pd.isna(row["pm25"])
                else float(row["pm25"]),

            "pm10":
                None
                if pd.isna(row["pm10"])
                else float(row["pm10"]),

            "no2":
                None
                if pd.isna(row["no2"])
                else float(row["no2"]),

            "nh3":
                None
                if pd.isna(row["nh3"])
                else float(row["nh3"]),

            "so2":
                None
                if pd.isna(row["so2"])
                else float(row["so2"]),

            "co":
                None
                if pd.isna(row["co"])
                else float(row["co"]),

            "o3":
                None
                if pd.isna(row["o3"])
                else float(row["o3"]),
        }

        records.append(record)

    return records


# ============================================================
# LOAD PREDICTIONS
#
# Cloud:
#   PostgreSQL -> latest forecast origin
#
# Local:
#   CSV fallback
# ============================================================

def get_predictions():

    # --------------------------------------------------------
    # POSTGRESQL
    # --------------------------------------------------------

    if using_postgres():

        try:
            query = """
                SELECT
                    forecast_origin,
                    forecast_timestamp,
                    horizon_hours,
                    current_aqi,
                    predicted_delta,
                    predicted_aqi
                FROM aqi_predictions
                WHERE forecast_origin = (
                    SELECT MAX(forecast_origin)
                    FROM aqi_predictions
                )
                ORDER BY horizon_hours
            """

            df = read_dataframe(query)

            if not df.empty:
                return df

            print(
                "PostgreSQL prediction table is empty."
            )

        except Exception as e:
            print(
                "PostgreSQL prediction error:",
                str(e),
            )

    # --------------------------------------------------------
    # LOCAL CSV FALLBACK
    # --------------------------------------------------------

    if not os.path.exists(
        PREDICTION_FILE
    ):
        return None

    try:
        df = pd.read_csv(
            PREDICTION_FILE
        )

    except Exception as e:
        print(
            "Prediction file error:",
            str(e),
        )
        return None

    if df.empty:
        return None

    return df


# ============================================================
# DATAFRAME -> JSON RECORDS
# ============================================================

def dataframe_to_records(df):
    records = []

    for _, row in df.iterrows():
        record = {}

        for column in df.columns:
            value = row[column]

            if pd.isna(value):
                record[column] = None
                continue

            if isinstance(
                value,
                (pd.Timestamp, np.datetime64),
            ):
                value = pd.to_datetime(
                    value
                ).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            elif hasattr(
                value,
                "item",
            ):
                value = value.item()

            record[column] = value

        records.append(record)

    return records


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"],
)
def health():

    prediction_source = (
        "postgresql"
        if using_postgres()
        else "csv"
    )

    return jsonify({
        "status":
            "ok",

        "database_available":
            database_available(),

        "database_type":
            "postgresql"
            if using_postgres()
            else "sqlite",

        "prediction_source":
            prediction_source,

        "prediction_file_exists":
            os.path.exists(
                PREDICTION_FILE
            ),
    })


# ============================================================
# CURRENT CPCB READING
# ============================================================

@app.route(
    "/api/current",
    methods=["GET"],
)
def current():
    current_data = get_current_data()

    if current_data is None:
        return jsonify({
            "success":
                False,

            "message":
                "No CPCB data available.",
        }), 404

    return jsonify({
        "success":
            True,

        "data":
            current_data,
    })


# ============================================================
# LATEST CPCB READING
# ============================================================

@app.route(
    "/api/latest",
    methods=["GET"],
)
def latest():
    current_data = get_current_data()

    if current_data is None:
        return jsonify({
            "success":
                False,

            "message":
                "No CPCB data available.",
        }), 404

    return jsonify({
        "success":
            True,

        "data":
            current_data,
    })


# ============================================================
# LAST 24 HOURS
# ============================================================

@app.route(
    "/api/history",
    methods=["GET"],
)
def history():
    records = get_last_24_hours()

    if records is None:
        return jsonify({
            "success":
                False,

            "message":
                "No historical CPCB data available.",
        }), 404

    return jsonify({
        "success":
            True,

        "count":
            len(records),

        "history":
            records,
    })


# ============================================================
# 24-HOUR FORECAST
# ============================================================

@app.route(
    "/api/predictions",
    methods=["GET"],
)
def predictions():
    df = get_predictions()

    if df is None:
        return jsonify({
            "success":
                False,

            "message":
                "No prediction data available.",
        }), 404

    records = dataframe_to_records(df)

    if not records:
        return jsonify({
            "success":
                False,

            "message":
                "Prediction data is empty.",
        }), 404

    return jsonify({
        "success":
            True,

        "count":
            len(records),

        "forecast_origin":
            records[0].get(
                "forecast_origin"
            ),

        "current_aqi":
            records[0].get(
                "current_aqi"
            ),

        "predictions":
            records,
    })


# ============================================================
# COMBINED CURRENT + HISTORY + FORECAST
# ============================================================

@app.route(
    "/api/forecast",
    methods=["GET"],
)
def forecast():
    current_data = get_current_data()
    prediction_df = get_predictions()
    history_data = get_last_24_hours()

    if prediction_df is None:
        return jsonify({
            "success":
                False,

            "message":
                "Prediction data not available.",
        }), 404

    predictions = dataframe_to_records(
        prediction_df
    )

    if not predictions:
        return jsonify({
            "success":
                False,

            "message":
                "Prediction data is empty.",
        }), 404

    current_aqi = None

    if current_data is not None:
        current_aqi = current_data.get(
            "aqi"
        )

    return jsonify({
        "success":
            True,

        "current":
            current_data,

        "history":
            history_data,

        "current_aqi":
            current_aqi,

        "forecast_origin":
            predictions[0].get(
                "forecast_origin"
            ),

        "forecast":
            predictions,
    })


# ============================================================
# ROOT
# ============================================================

@app.route(
    "/",
    methods=["GET"],
)
def index():
    return jsonify({
        "application":
            "CPCB Vadodara AQI Prediction API",

        "status":
            "running",

        "database":
            "postgresql"
            if using_postgres()
            else "sqlite",

        "endpoints": [
            "/api/health",
            "/api/current",
            "/api/latest",
            "/api/history",
            "/api/predictions",
            "/api/forecast",
        ],
    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("CPCB VADODARA AQI API")
    print("=" * 70)
    print()

    print(
        "Database:",
        "PostgreSQL"
        if using_postgres()
        else DATABASE_PATH,
    )

    print()

    print(
        "Prediction source:",
        "PostgreSQL"
        if using_postgres()
        else PREDICTION_FILE,
    )

    print()

    print("API:")
    print("http://localhost:5000")

    print()
    print("=" * 70)
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )
