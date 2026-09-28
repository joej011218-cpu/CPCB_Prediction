import os
import sys
import time
import subprocess
from enum import Enum

import pandas as pd
import requests
from dotenv import load_dotenv

from database import (
    database_available,
    get_connection,
    read_dataframe,
    sql_placeholder,
    using_postgres,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ENV_PATH = os.path.join(
    BASE_DIR,
    ".env",
)

load_dotenv(ENV_PATH)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cpcb_vadodara.db",
)

PREDICTOR_PATH = os.path.join(
    BASE_DIR,
    "predictor.py",
)


# ============================================================
# CPCB API
# ============================================================

API_URL = (
    "https://api.data.gov.in/resource/"
    "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
)

API_KEY = os.getenv(
    "CPCB_API_KEY",
    "",
)

# Local continuous mode only.
CHECK_INTERVAL = 60

# A shorter timeout is better for GitHub Actions.
# The API sometimes becomes slow/unavailable.
API_TIMEOUT = 30

MAX_RETRIES = 3

# Increasing delay between retries:
# attempt 1 -> 15 sec
# attempt 2 -> 30 sec
RETRY_DELAYS = [
    15,
    30,
]

API_LIMIT = 100


def get_api_params():
    return {
        "api-key": API_KEY,
        "format": "json",
        "offset": 0,
        "limit": API_LIMIT,
        "filters[state]": "Gujarat",
        "filters[city]": "Vadodara",
    }


# ============================================================
# POLLUTANTS
# ============================================================

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


# ============================================================
# COLLECTION RESULT
# ============================================================

class CollectionStatus(Enum):
    NEW_DATA = "new_data"
    NO_NEW_DATA = "no_new_data"
    API_UNAVAILABLE = "api_unavailable"


class CPCBAPIUnavailable(Exception):
    """Raised when CPCB could not be contacted successfully."""


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

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


# ============================================================
# RETRY HELPER
# ============================================================

def wait_before_retry(attempt):

    index = min(
        attempt - 1,
        len(RETRY_DELAYS) - 1,
    )

    delay = RETRY_DELAYS[index]

    print(
        f"Retrying in {delay} seconds..."
    )

    time.sleep(delay)


# ============================================================
# CPCB API FETCH
# ============================================================

def fetch_cpcb_data():

    print()
    print("=" * 70)
    print("FETCHING CPCB VADODARA DATA")
    print("=" * 70)

    if not API_KEY:

        raise CPCBAPIUnavailable(
            "CPCB_API_KEY environment variable is not configured."
        )

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 "
                "(X11; Linux x86_64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/151.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Connection": "close",
        }
    )

    api_params = get_api_params()

    last_error = None

    try:

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):

            print()
            print(
                f"API attempt "
                f"{attempt}/{MAX_RETRIES}"
            )

            try:

                response = session.get(
                    API_URL,
                    params=api_params,
                    timeout=API_TIMEOUT,
                )

                print(
                    "HTTP Status:",
                    response.status_code,
                )

                # =================================================
                # HTTP 200
                # =================================================

                if response.status_code == 200:

                    try:
                        result = response.json()

                    except ValueError as e:

                        last_error = (
                            "CPCB returned HTTP 200 "
                            "but the response was not valid JSON."
                        )

                        print()
                        print(
                            "WARNING:",
                            last_error,
                        )

                        if attempt < MAX_RETRIES:
                            wait_before_retry(attempt)
                            continue

                        break

                    records = result.get(
                        "records",
                        [],
                    )

                    if records is None:
                        records = []

                    print()
                    print(
                        "API response total:",
                        result.get("total"),
                    )

                    print(
                        "API response count:",
                        result.get("count"),
                    )

                    print(
                        "Records received:",
                        len(records),
                    )

                    # =================================================
                    # HTTP 200 BUT EMPTY RESPONSE
                    # =================================================

                    if not records:

                        last_error = (
                            "CPCB API returned HTTP 200 "
                            "but zero records."
                        )

                        print()
                        print(
                            "WARNING:",
                            last_error,
                        )

                        if attempt < MAX_RETRIES:
                            wait_before_retry(attempt)
                            continue

                        break

                    # =================================================
                    # PRINT ONE CO RECORD
                    # =================================================

                    print()
                    print("=" * 70)
                    print("RAW CPCB CO RECORD")
                    print("=" * 70)

                    co_record_found = False

                    for record in records:

                        pollutant_id = str(
                            record.get(
                                "pollutant_id",
                                "",
                            )
                        ).strip().upper()

                        if pollutant_id == "CO":

                            print(record)

                            co_record_found = True
                            break

                    if not co_record_found:
                        print(
                            "No CO record found in this response."
                        )

                    # =================================================
                    # SHOW API TIMESTAMP RANGE
                    # =================================================

                    timestamps = []

                    for record in records:

                        timestamp_value = (
                            record.get("last_update")
                            or record.get("timestamp")
                            or record.get("datetime")
                        )

                        if timestamp_value:

                            parsed = pd.to_datetime(
                                timestamp_value,
                                errors="coerce",
                                dayfirst=True,
                            )

                            if not pd.isna(parsed):
                                timestamps.append(parsed)

                    if timestamps:

                        print()
                        print(
                            "Oldest timestamp in API response:",
                            min(timestamps),
                        )

                        print(
                            "Newest timestamp in API response:",
                            max(timestamps),
                        )

                    print()
                    print(
                        "CPCB API fetch successful."
                    )

                    return records

                # =================================================
                # TEMPORARY HTTP ERRORS
                # =================================================

                if response.status_code in (
                    408,
                    425,
                    429,
                    500,
                    502,
                    503,
                    504,
                ):

                    last_error = (
                        "Temporary CPCB API HTTP error: "
                        f"{response.status_code}"
                    )

                    print()
                    print(last_error)

                    if attempt < MAX_RETRIES:
                        wait_before_retry(attempt)
                        continue

                    break

                # =================================================
                # PERMANENT / OTHER HTTP ERROR
                # =================================================

                last_error = (
                    "CPCB API HTTP error "
                    f"{response.status_code}: "
                    f"{response.text[:300]}"
                )

                print()
                print(last_error)

                # Do not retry authentication/client errors.
                if 400 <= response.status_code < 500:
                    break

                if attempt < MAX_RETRIES:
                    wait_before_retry(attempt)
                    continue

                break

            # =====================================================
            # TIMEOUT
            # =====================================================

            except requests.exceptions.Timeout:

                last_error = (
                    f"CPCB API request timed out after "
                    f"{API_TIMEOUT} seconds."
                )

                print()
                print(last_error)

                if attempt < MAX_RETRIES:
                    wait_before_retry(attempt)
                    continue

                break

            # =====================================================
            # CONNECTION ERROR
            # =====================================================

            except requests.exceptions.ConnectionError as e:

                last_error = (
                    "CPCB connection error: "
                    f"{e}"
                )

                print()
                print(last_error)

                if attempt < MAX_RETRIES:
                    wait_before_retry(attempt)
                    continue

                break

            # =====================================================
            # OTHER REQUEST ERROR
            # =====================================================

            except requests.exceptions.RequestException as e:

                last_error = (
                    "CPCB request failed: "
                    f"{e}"
                )

                print()
                print(last_error)

                if attempt < MAX_RETRIES:
                    wait_before_retry(attempt)
                    continue

                break

    finally:

        session.close()

    # ============================================================
    # IMPORTANT:
    # Reaching here means the API never produced usable data.
    # This is NOT the same as "no new CPCB data".
    # ============================================================

    print()
    print("=" * 70)
    print("CPCB API UNAVAILABLE")
    print("=" * 70)

    if last_error:
        print(last_error)

    raise CPCBAPIUnavailable(
        last_error
        or "CPCB API did not return usable data."
    )


# ============================================================
# DATA CLEANING
# ============================================================

def normalize_pollutant_name(value):

    if pd.isna(value):
        return None

    value = (
        str(value)
        .strip()
        .upper()
    )

    return POLLUTANT_NAME_MAP.get(
        value
    )


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


# ============================================================
# CPCB RECORDS -> HOURLY TABLE
# ============================================================

def convert_records_to_hourly(records):

    if not records:
        return None

    df = pd.DataFrame(records)

    if df.empty:
        return None

    print()
    print("CPCB columns received:")
    print(df.columns.tolist())

    # ========================================================
    # TIMESTAMP COLUMN
    # ========================================================

    timestamp_column = None

    for column in [
        "last_update",
        "timestamp",
        "datetime",
        "date",
        "time",
    ]:

        if column in df.columns:

            timestamp_column = column
            break

    if timestamp_column is None:

        print()
        print(
            "ERROR: No CPCB timestamp column found."
        )

        return None

    # ========================================================
    # POLLUTANT COLUMN
    # ========================================================

    pollutant_column = None

    for column in [
        "pollutant_id",
        "pollutant",
        "pollutant_name",
    ]:

        if column in df.columns:

            pollutant_column = column
            break

    if pollutant_column is None:

        print()
        print(
            "ERROR: No pollutant_id column found."
        )

        return None

    # ========================================================
    # VALUE COLUMN
    # ========================================================

    value_column = find_value_column(df)

    if value_column is None:

        print()
        print(
            "ERROR: No pollutant value column found."
        )

        print(
            "Available columns:"
        )

        print(
            df.columns.tolist()
        )

        return None

    print()
    print(
        "Timestamp column:",
        timestamp_column,
    )

    print(
        "Pollutant column:",
        pollutant_column,
    )

    print(
        "Value column:",
        value_column,
    )

    # ========================================================
    # CLEAN DATA
    # ========================================================

    data = df.copy()

    data["timestamp"] = pd.to_datetime(
        data[timestamp_column],
        errors="coerce",
        dayfirst=True,
    )

    data = data.dropna(
        subset=["timestamp"]
    )

    data["pollutant"] = (
        data[pollutant_column]
        .apply(
            normalize_pollutant_name
        )
    )

    data["value"] = pd.to_numeric(
        data[value_column],
        errors="coerce",
    ).astype(float)

    data = data.dropna(
        subset=["pollutant"]
    )

    # ========================================================
    # CPCB CO CONVERSION
    #
    # CPCB API:
    #     µg/m³
    #
    # Database/model:
    #     mg/m³
    #
    # Therefore:
    #     divide by 1000
    # ========================================================

    co_mask = (
        data["pollutant"] == "co"
    )

    data.loc[
        co_mask,
        "value",
    ] = (
        data.loc[
            co_mask,
            "value",
        ]
        / 1000.0
    )

    if data.empty:

        print()
        print(
            "ERROR: No valid pollutant "
            "records after cleaning."
        )

        return None

    print()
    print(
        "Pollutants received:"
    )

    print(
        sorted(
            data[
                "pollutant"
            ]
            .dropna()
            .unique()
            .tolist()
        )
    )

    # ========================================================
    # PIVOT POLLUTANTS
    # ========================================================

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

    # ========================================================
    # ENSURE ALL COLUMNS EXIST
    # ========================================================

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

    hourly = (
        hourly
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return hourly


# ============================================================
# GET LATEST DATABASE TIMESTAMP
# ============================================================

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

    except Exception as e:

        print()
        print(
            "Could not read latest database timestamp:"
        )
        print(e)

        return None

    if df.empty:
        return None

    value = (
        df.iloc[0]["timestamp"]
    )

    if pd.isna(value):
        return None

    timestamp = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(timestamp):
        return None

    return timestamp


# ============================================================
# SAVE CPCB DATA
# ============================================================

def save_hourly_data(df):

    if (
        df is None
        or df.empty
    ):
        return 0

    conn = get_connection()

    affected = 0

    placeholder = (
        sql_placeholder()
    )

    placeholders = ", ".join(
        [placeholder] * 9
    )

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

            timestamp_string = (
                pd.Timestamp(timestamp)
                .strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

            values = (
                timestamp_string,

                to_database_value(
                    row["pm25"]
                ),

                to_database_value(
                    row["pm10"]
                ),

                to_database_value(
                    row["no"]
                ),

                to_database_value(
                    row["no2"]
                ),

                to_database_value(
                    row["nh3"]
                ),

                to_database_value(
                    row["so2"]
                ),

                to_database_value(
                    row["co"]
                ),

                to_database_value(
                    row["o3"]
                ),
            )

            cursor.execute(
                query,
                values,
            )

            affected += 1

        conn.commit()

    except Exception:

        conn.rollback()
        raise

    finally:

        conn.close()

    return affected


# ============================================================
# COLLECT NEW CPCB DATA
# ============================================================

def collect_new_data():

    previous_latest_timestamp = (
        get_latest_database_timestamp()
    )

    print()
    print("=" * 70)
    print(
        "DATABASE STATUS BEFORE CPCB FETCH"
    )
    print("=" * 70)

    print(
        "Latest database timestamp:",
        previous_latest_timestamp,
    )

    # ========================================================
    # FETCH API
    #
    # CPCBAPIUnavailable is deliberately NOT converted to
    # False here.
    # ========================================================

    records = fetch_cpcb_data()

    hourly = convert_records_to_hourly(
        records
    )

    if hourly is None or hourly.empty:

        raise CPCBAPIUnavailable(
            "CPCB returned records, but they could "
            "not be converted into valid hourly data."
        )

    # ========================================================
    # FIND TRUE NEWEST API TIMESTAMP
    # ========================================================

    latest_api_timestamp = pd.Timestamp(
        hourly["timestamp"].max()
    )

    latest_rows = hourly[
        hourly["timestamp"]
        == latest_api_timestamp
    ]

    if latest_rows.empty:

        raise CPCBAPIUnavailable(
            "Latest CPCB row could not be selected."
        )

    latest = (
        latest_rows.iloc[-1]
    )

    print()
    print("=" * 70)
    print(
        "LATEST CPCB HOURLY RECORD"
    )
    print("=" * 70)

    print(
        "Timestamp:",
        latest_api_timestamp,
    )

    for pollutant in POLLUTANTS:

        print(
            f"{pollutant.upper():5s}:",
            latest[pollutant],
        )

    # ========================================================
    # SAVE ALL RETURNED DATA
    # ========================================================

    affected = save_hourly_data(
        hourly
    )

    print()
    print(
        "Hourly records processed:",
        affected,
    )

    # ========================================================
    # CHECK DATABASE AFTER SAVE
    # ========================================================

    database_latest_after_save = (
        get_latest_database_timestamp()
    )

    print(
        "Database latest after save:",
        database_latest_after_save,
    )

    # ========================================================
    # EMPTY DATABASE
    # ========================================================

    if previous_latest_timestamp is None:

        print()
        print(
            "Database previously had no CPCB timestamp."
        )

        return CollectionStatus.NEW_DATA

    previous_latest_timestamp = pd.Timestamp(
        previous_latest_timestamp
    )

    # ========================================================
    # NEW CPCB HOUR
    # ========================================================

    if (
        latest_api_timestamp
        > previous_latest_timestamp
    ):

        print()
        print(
            "New CPCB timestamp detected:"
        )

        print(
            previous_latest_timestamp,
            "->",
            latest_api_timestamp,
        )

        return CollectionStatus.NEW_DATA

    # ========================================================
    # SAME / OLDER CPCB HOUR
    # ========================================================

    print()
    print(
        "Latest CPCB timestamp is unchanged."
    )

    print(
        "Database timestamp:",
        previous_latest_timestamp,
    )

    print(
        "API timestamp:",
        latest_api_timestamp,
    )

    return CollectionStatus.NO_NEW_DATA


# ============================================================
# RUN PREDICTOR
# ============================================================

def regenerate_prediction():

    print()
    print("=" * 70)
    print(
        "REGENERATING VADODARA PREDICTION"
    )
    print("=" * 70)

    if not os.path.exists(
        PREDICTOR_PATH
    ):

        print()
        print(
            "ERROR: predictor.py not found:"
        )

        print(
            PREDICTOR_PATH
        )

        return False

    try:

        result = subprocess.run(
            [
                sys.executable,
                PREDICTOR_PATH,
            ],
            cwd=BASE_DIR,
            check=False,
        )

        if result.returncode == 0:

            print()
            print(
                "Prediction regenerated successfully."
            )

            return True

        print()
        print(
            "Prediction failed."
        )

        print(
            "Exit code:",
            result.returncode,
        )

        return False

    except Exception as e:

        print()
        print(
            "Prediction regeneration error:"
        )

        print(e)

        return False


# ============================================================
# ONE COLLECTION CYCLE
# ============================================================

def run_collection_cycle():

    print()
    print("=" * 70)
    print(
        "CPCB COLLECTION CYCLE"
    )
    print("=" * 70)

    try:

        status = collect_new_data()

    except CPCBAPIUnavailable as e:

        print()
        print("=" * 70)
        print(
            "CPCB API TEMPORARILY UNAVAILABLE"
        )
        print("=" * 70)

        print(e)

        print()
        print(
            "Database was not treated as up-to-date."
        )

        print(
            "Existing prediction retained."
        )

        return CollectionStatus.API_UNAVAILABLE

    if status == CollectionStatus.NEW_DATA:

        print()
        print(
            "New CPCB hour available."
        )

        prediction_success = (
            regenerate_prediction()
        )

        if prediction_success:

            print()
            print(
                "Live prediction updated."
            )

        else:

            print()
            print(
                "Prediction could not be regenerated."
            )

        return CollectionStatus.NEW_DATA

    print()
    print(
        "No new CPCB hour available."
    )

    print(
        "Existing prediction retained."
    )

    return CollectionStatus.NO_NEW_DATA


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "VADODARA CPCB LIVE COLLECTOR"
    )
    print("=" * 70)
    print()

    if using_postgres():

        print(
            "Database: PostgreSQL"
        )

    else:

        print(
            "Database:",
            DATABASE_PATH,
        )

    print(
        "Predictor:",
        PREDICTOR_PATH,
    )

    # ========================================================
    # COMMAND-LINE MODES
    # ========================================================

    run_once = (
        "--once" in sys.argv
    )

    collect_only = (
        "--collect-only" in sys.argv
    )

    if collect_only:

        print(
            "Mode: CLOUD COLLECTION ONLY"
        )

    elif run_once:

        print(
            "Mode: ONE-TIME CLOUD UPDATE"
        )

    else:

        print(
            "Mode: CONTINUOUS LOCAL COLLECTOR"
        )

        print(
            "Check interval:",
            CHECK_INTERVAL,
            "seconds",
        )

    # ========================================================
    # INITIALIZE DATABASE
    # ========================================================

    initialize_database()

    # ========================================================
    # CLOUD COLLECTION-ONLY MODE
    #
    # Used by GitHub Actions before RF model restoration.
    #
    # EXIT CODES:
    #
    #   0  = API succeeded, but no newer CPCB timestamp
    #
    #   10 = API succeeded, new CPCB timestamp found
    #        and saved
    #
    #   20 = CPCB API unavailable / timeout / empty
    #        response after all retries
    #
    #   1  = unexpected collector/database error
    #
    # predictor.py is NOT executed in this mode.
    # ========================================================

    if collect_only:

        try:

            status = collect_new_data()

            print()
            print("=" * 70)

            if status == CollectionStatus.NEW_DATA:

                print(
                    "NEW CPCB DATA AVAILABLE"
                )

                print("=" * 70)

                sys.exit(10)

            print(
                "NO NEW CPCB DATA"
            )

            print("=" * 70)

            sys.exit(0)

        except CPCBAPIUnavailable as e:

            print()
            print("=" * 70)
            print(
                "CPCB API UNAVAILABLE"
            )
            print("=" * 70)

            print(e)

            print()
            print(
                "This run does NOT mean that "
                "there is no new CPCB data."
            )

            print(
                "The API could not be checked successfully."
            )

            print("=" * 70)

            sys.exit(20)

        except Exception as e:

            print()
            print("=" * 70)
            print(
                "CPCB UPDATE ERROR"
            )
            print("=" * 70)

            print(e)

            sys.exit(1)

    # ========================================================
    # NORMAL ONE-TIME CLOUD MODE
    #
    # collect CPCB
    # -> detect new hour
    # -> run predictor when new data exists
    # ========================================================

    if run_once:

        try:

            status = run_collection_cycle()

            print()
            print("=" * 70)

            if (
                status
                == CollectionStatus.API_UNAVAILABLE
            ):

                print(
                    "ONE-TIME CPCB UPDATE FINISHED "
                    "WITH API UNAVAILABLE"
                )

            elif (
                status
                == CollectionStatus.NEW_DATA
            ):

                print(
                    "ONE-TIME CPCB UPDATE FINISHED "
                    "WITH NEW DATA"
                )

            else:

                print(
                    "ONE-TIME CPCB UPDATE FINISHED "
                    "WITH NO NEW DATA"
                )

            print("=" * 70)

        except Exception as e:

            print()
            print("=" * 70)
            print(
                "CPCB UPDATE ERROR"
            )
            print("=" * 70)

            print(e)

            sys.exit(1)

        return

    # ========================================================
    # CONTINUOUS LOCAL MODE
    # ========================================================

    while True:

        try:

            run_collection_cycle()

            print()
            print("=" * 70)

            print(
                f"Next CPCB check in "
                f"{CHECK_INTERVAL} seconds."
            )

            print("=" * 70)

            time.sleep(
                CHECK_INTERVAL
            )

        except KeyboardInterrupt:

            print()
            print("=" * 70)
            print(
                "CPCB COLLECTOR STOPPED"
            )
            print("=" * 70)

            break

        except Exception as e:

            print()
            print("=" * 70)
            print(
                "UNEXPECTED COLLECTOR ERROR"
            )
            print("=" * 70)

            print(e)

            print()
            print(
                "Existing prediction retained."
            )

            print(
                f"Retrying in "
                f"{CHECK_INTERVAL} seconds."
            )

            time.sleep(
                CHECK_INTERVAL
            )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()