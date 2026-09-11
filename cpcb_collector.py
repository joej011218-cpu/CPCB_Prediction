import os
import time
import subprocess
import sys

import requests
import pandas as pd

from dotenv import load_dotenv

from database import (
    database_available,
    get_connection,
    read_dataframe,
    sql_placeholder,
    using_postgres,
)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ENV_PATH = os.path.join(
    BASE_DIR,
    ".env"
)

load_dotenv(
    ENV_PATH
)


DATABASE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cpcb_vadodara.db"
)

PREDICTOR_PATH = os.path.join(
    BASE_DIR,
    "predictor.py"
)

API_URL = (
    "https://api.data.gov.in/resource/"
    "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
)

API_KEY = os.getenv(
    "CPCB_API_KEY",
    ""
)
CHECK_INTERVAL = 60
API_TIMEOUT = 45
MAX_RETRIES = 3
RETRY_DELAY = 10


def get_api_params():
    return {
        "api-key": API_KEY,
        "format": "json",
        "offset": 0,
        "limit": 100,
        "filters[state]": "Gujarat",
        "filters[city]": "Vadodara",
    }


POLLUTANTS = [
    "pm25",
    "pm10",
    "no",
    "no2",
    "nh3",
    "so2",
    "co",
    "o3",
]


POLLUTANT_NAME_MAP = {
    "PM25": "pm25",
    "PM2.5": "pm25",
    "PM2_5": "pm25",
    "PM2-5": "pm25",
    "PM10": "pm10",
    "NO": "no",
    "NO2": "no2",
    "NH3": "nh3",
    "SO2": "so2",
    "CO": "co",
    "O3": "o3",
    "OZONE": "o3",
}


def initialize_database():
    print()
    print("=" * 70)
    print("CHECKING DATABASE")
    print("=" * 70)

    if not using_postgres():
        os.makedirs(
            os.path.dirname(DATABASE_PATH),
            exist_ok=True,
        )

    conn = get_connection()

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cpcb_hourly (
                timestamp TEXT PRIMARY KEY,
                pm25 REAL,
                pm10 REAL,
                no REAL,
                no2 REAL,
                nh3 REAL,
                so2 REAL,
                co REAL,
                o3 REAL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    if using_postgres():
        print("Database ready: PostgreSQL")
    else:
        print("Database ready:")
        print(DATABASE_PATH)


def fetch_cpcb_data():
    print()
    print("=" * 70)
    print("FETCHING CPCB VADODARA DATA")
    print("=" * 70)

    if not API_KEY:
        print()
        print("ERROR: CPCB_API_KEY environment variable is not configured.")
        return None

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/151.0 Safari/537.36"
            ),
            "Accept": "application/json",
        }
    )

    api_params = get_api_params()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print()
            print(f"API attempt {attempt}/{MAX_RETRIES}")

            response = session.get(
                API_URL,
                params=api_params,
                timeout=API_TIMEOUT,
            )

            print("HTTP Status:", response.status_code)

            if response.status_code == 200:
                try:
                    result = response.json()
                except ValueError:
                    print("ERROR: CPCB returned invalid JSON.")
                    return None

                records = result.get("records", [])

                print()
                print("=" * 70)
                print("RAW CPCB CO RECORD")
                print("=" * 70)

                for record in records:
                    if str(record.get("pollutant_id", "")).upper() == "CO":
                        print(record)

                if not records:
                    print("CPCB API returned zero records.")
                    return None

                print("Records received:", len(records))
                return records

            if response.status_code in (500, 502, 503, 504):
                print()
                print("CPCB API HTTP error:")
                print(f"{response.status_code} Server Error")

                if attempt < MAX_RETRIES:
                    print(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                    continue

                print()
                print("=" * 70)
                print("CPCB API FAILED")
                print("=" * 70)
                print(f"All {MAX_RETRIES} API attempts failed.")
                return None

            print()
            print("CPCB API HTTP error:")
            print(response.text[:500])
            return None

        except requests.exceptions.Timeout:
            print()
            print("CPCB API request timed out.")

            if attempt < MAX_RETRIES:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
                continue

            print()
            print("=" * 70)
            print("CPCB API FAILED")
            print("=" * 70)
            return None

        except requests.exceptions.ConnectionError as e:
            print()
            print("CPCB connection error:")
            print(e)

            if attempt < MAX_RETRIES:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
                continue

            return None

        except requests.exceptions.RequestException as e:
            print()
            print("CPCB request failed:")
            print(e)
            return None

        except Exception as e:
            print()
            print("Unexpected API error:")
            print(e)
            return None

    return None


def normalize_pollutant_name(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()
    return POLLUTANT_NAME_MAP.get(value)


def find_value_column(df):
    possible_columns = [
        "avg_value",
        "pollutant_avg",
        "average",
        "avg",
        "value",
    ]

    for column in possible_columns:
        if column in df.columns:
            return column

    return None


def to_database_value(value):
    if pd.isna(value):
        return None
    return float(value)


def convert_records_to_hourly(records):
    if not records:
        return None

    df = pd.DataFrame(records)

    if df.empty:
        return None

    print()
    print("CPCB columns received:")
    print(df.columns.tolist())

    timestamp_column = None
    for column in ["last_update", "timestamp", "datetime", "date", "time"]:
        if column in df.columns:
            timestamp_column = column
            break

    if timestamp_column is None:
        print()
        print("ERROR: No CPCB timestamp column found.")
        return None

    pollutant_column = None
    for column in ["pollutant_id", "pollutant", "pollutant_name"]:
        if column in df.columns:
            pollutant_column = column
            break

    if pollutant_column is None:
        print()
        print("ERROR: No pollutant_id column found.")
        return None

    value_column = find_value_column(df)

    if value_column is None:
        print()
        print("ERROR: No pollutant value column found.")
        print("Available columns:")
        print(df.columns.tolist())
        return None

    print()
    print("Timestamp column:", timestamp_column)
    print("Pollutant column:", pollutant_column)
    print("Value column:", value_column)

    data = df.copy()

    data["timestamp"] = pd.to_datetime(
        data[timestamp_column],
        errors="coerce",
        dayfirst=True,
    )

    data = data.dropna(subset=["timestamp"])

    data["pollutant"] = data[pollutant_column].apply(
        normalize_pollutant_name
    )

    data["value"] = pd.to_numeric(
        data[value_column],
        errors="coerce",
    ).astype(float)

    co_mask = data["pollutant"] == "co"
    data.loc[co_mask, "value"] = data.loc[co_mask, "value"] / 1000.0

    if data.empty:
        print()
        print("ERROR: No valid pollutant records after cleaning.")
        return None

    print()
    print("Pollutants received:")
    print(
        sorted(
            data["pollutant"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    hourly = (
        data
        .pivot_table(
            index="timestamp",
            columns="pollutant",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )

    hourly.columns.name = None

    for pollutant in POLLUTANTS:
        if pollutant not in hourly.columns:
            hourly[pollutant] = None

    hourly = hourly[
        [
            "timestamp",
            "pm25",
            "pm10",
            "no",
            "no2",
            "nh3",
            "so2",
            "co",
            "o3",
        ]
    ].copy()

    hourly = hourly.sort_values("timestamp").reset_index(drop=True)
    return hourly


def get_latest_database_timestamp():
    if not database_available():
        return None

    try:
        df = read_dataframe(
            """
            SELECT MAX(timestamp) AS timestamp
            FROM cpcb_hourly
            """
        )
    except Exception:
        return None

    if df.empty:
        return None

    value = df.iloc[0]["timestamp"]

    if pd.isna(value):
        return None

    timestamp = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(timestamp):
        return None

    return timestamp


def save_hourly_data(df):
    if df is None or df.empty:
        return 0

    conn = get_connection()
    affected = 0

    placeholder = sql_placeholder()
    placeholders = ", ".join([placeholder] * 9)

    query = f"""
        INSERT INTO cpcb_hourly
        (
            timestamp,
            pm25,
            pm10,
            no,
            no2,
            nh3,
            so2,
            co,
            o3
        )
        VALUES
        ({placeholders})

        ON CONFLICT(timestamp)
        DO UPDATE SET
            pm25 = excluded.pm25,
            pm10 = excluded.pm10,
            no = excluded.no,
            no2 = excluded.no2,
            nh3 = excluded.nh3,
            so2 = excluded.so2,
            co = excluded.co,
            o3 = excluded.o3
    """

    try:
        cursor = conn.cursor()

        for _, row in df.iterrows():
            timestamp = row["timestamp"]

            if pd.isna(timestamp):
                continue

            timestamp_string = pd.Timestamp(timestamp).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            values = (
                timestamp_string,
                to_database_value(row["pm25"]),
                to_database_value(row["pm10"]),
                to_database_value(row["no"]),
                to_database_value(row["no2"]),
                to_database_value(row["nh3"]),
                to_database_value(row["so2"]),
                to_database_value(row["co"]),
                to_database_value(row["o3"]),
            )

            cursor.execute(query, values)
            affected += 1

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()

    return affected


def collect_new_data():
    previous_latest_timestamp = get_latest_database_timestamp()

    records = fetch_cpcb_data()

    if records is None:
        print()
        print("No data received from CPCB.")
        return False

    hourly = convert_records_to_hourly(records)

    if hourly is None:
        print()
        print("CPCB data could not be converted.")
        return False

    if hourly.empty:
        print()
        print("No hourly records available.")
        return False

    latest = hourly.iloc[-1]
    latest_api_timestamp = pd.Timestamp(latest["timestamp"])

    print()
    print("=" * 70)
    print("LATEST CPCB HOURLY RECORD")
    print("=" * 70)
    print("Timestamp:", latest_api_timestamp)

    for pollutant in POLLUTANTS:
        print(f"{pollutant.upper():5s}:", latest[pollutant])

    affected = save_hourly_data(hourly)

    print()
    print("Hourly records processed:", affected)

    if previous_latest_timestamp is None:
        print()
        print("Database previously had no CPCB timestamp.")
        return True

    previous_latest_timestamp = pd.Timestamp(previous_latest_timestamp)

    if latest_api_timestamp > previous_latest_timestamp:
        print()
        print("New CPCB timestamp detected:")
        print(previous_latest_timestamp, "->", latest_api_timestamp)
        return True

    print()
    print("Latest CPCB timestamp is unchanged.")
    return False


def regenerate_prediction():
    print()
    print("=" * 70)
    print("REGENERATING VADODARA PREDICTION")
    print("=" * 70)

    if not os.path.exists(PREDICTOR_PATH):
        print()
        print("ERROR: predictor.py not found:")
        print(PREDICTOR_PATH)
        return False

    try:
        result = subprocess.run(
            [sys.executable, PREDICTOR_PATH],
            cwd=BASE_DIR,
            check=False,
        )

        if result.returncode == 0:
            print()
            print("Prediction regenerated successfully.")
            return True

        print()
        print("Prediction failed.")
        print("Exit code:", result.returncode)
        return False

    except Exception as e:
        print()
        print("Prediction regeneration error:")
        print(e)
        return False


def main():
    print()
    print("=" * 70)
    print("VADODARA CPCB LIVE COLLECTOR")
    print("=" * 70)
    print()

    if using_postgres():
        print("Database: PostgreSQL")
    else:
        print("Database:", DATABASE_PATH)

    print("Predictor:", PREDICTOR_PATH)
    print("Check interval:", CHECK_INTERVAL, "seconds")

    initialize_database()

    while True:
        try:
            print()
            print("=" * 70)
            print("CPCB COLLECTION CYCLE")
            print("=" * 70)

            new_data = collect_new_data()

            if new_data:
                print()
                print("New CPCB hour available.")

                prediction_success = regenerate_prediction()

                if prediction_success:
                    print()
                    print("Live prediction updated.")
                else:
                    print()
                    print("Prediction could not be regenerated.")
            else:
                print()
                print("No new CPCB hour available.")
                print("Existing prediction retained.")

            print()
            print("=" * 70)
            print(f"Next CPCB check in {CHECK_INTERVAL} seconds.")
            print("=" * 70)

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print()
            print("=" * 70)
            print("CPCB COLLECTOR STOPPED")
            print("=" * 70)
            break

        except Exception as e:
            print()
            print("=" * 70)
            print("UNEXPECTED COLLECTOR ERROR")
            print("=" * 70)
            print(e)
            print()
            print("Existing prediction retained.")
            print(f"Retrying in {CHECK_INTERVAL} seconds.")
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
