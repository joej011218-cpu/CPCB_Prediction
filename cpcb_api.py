import os
import time
import requests
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

CPCB_API_URL = (
    "https://api.data.gov.in/resource/"
    "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
)

# Recommended:
# macOS/Linux:
#
# export CPCB_API_KEY="YOUR_API_KEY"
#
# If you already have the key hardcoded in your old file,
# you can put it in the string below.

CPCB_API_KEY = os.getenv(
    "CPCB_API_KEY",
    "YOUR_API_KEY"
)

API_LIMIT = 100

STATE = "Gujarat"
CITY = "Vadodara"

# Number of API attempts before giving up
MAX_API_RETRIES = 3

# Seconds between retries
RETRY_DELAY_SECONDS = 10

# Request timeout
REQUEST_TIMEOUT_SECONDS = 90


# ============================================================
# FETCH CPCB DATA
# ============================================================

def fetch_vadodara_data():

    print()
    print("=" * 70)
    print("FETCHING CPCB VADODARA DATA")
    print("=" * 70)

    params = {
        "api-key": CPCB_API_KEY,
        "format": "json",
        "limit": API_LIMIT,
        "filters[state]": STATE,
        "filters[city]": CITY
    }

    # --------------------------------------------------------
    # CHECK API KEY
    # --------------------------------------------------------

    if (
        CPCB_API_KEY is None
        or CPCB_API_KEY.strip() == ""
        or CPCB_API_KEY == "YOUR_API_KEY"
    ):

        print()
        print("ERROR: CPCB API key is not configured.")

        print()
        print(
            "Set your API key using:"
        )

        print(
            'export CPCB_API_KEY="YOUR_API_KEY"'
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # RETRY LOOP
    # --------------------------------------------------------

    for attempt in range(
        1,
        MAX_API_RETRIES + 1
    ):

        print()
        print(
            f"API attempt "
            f"{attempt}/{MAX_API_RETRIES}"
        )

        try:

            response = requests.get(

                CPCB_API_URL,

                params=params,

                timeout=REQUEST_TIMEOUT_SECONDS
            )

            print(
                "HTTP Status:",
                response.status_code
            )

            response.raise_for_status()

            # ------------------------------------------------
            # PARSE JSON
            # ------------------------------------------------

            try:

                data = response.json()

            except ValueError as e:

                print()
                print(
                    "CPCB returned invalid JSON."
                )

                print(e)

                if attempt < MAX_API_RETRIES:

                    print(
                        f"Retrying in "
                        f"{RETRY_DELAY_SECONDS} seconds..."
                    )

                    time.sleep(
                        RETRY_DELAY_SECONDS
                    )

                    continue

                return pd.DataFrame()

            # ------------------------------------------------
            # GET RECORDS
            # ------------------------------------------------

            records = data.get(
                "records",
                []
            )

            if not records:

                print()
                print(
                    "CPCB API returned no records."
                )

                return pd.DataFrame()

            df = pd.DataFrame(
                records
            )

            print()
            print(
                "Records received:",
                len(df)
            )

            print()
            print(
                "CPCB columns:"
            )

            print(
                df.columns.tolist()
            )

            return df

        # ----------------------------------------------------
        # TIMEOUT
        # ----------------------------------------------------

        except requests.exceptions.Timeout:

            print()
            print(
                "CPCB API request timed out."
            )

            print(
                f"Timeout: "
                f"{REQUEST_TIMEOUT_SECONDS} seconds"
            )

            if attempt < MAX_API_RETRIES:

                print(
                    f"Retrying in "
                    f"{RETRY_DELAY_SECONDS} seconds..."
                )

                time.sleep(
                    RETRY_DELAY_SECONDS
                )

        # ----------------------------------------------------
        # CONNECTION ERROR
        # ----------------------------------------------------

        except requests.exceptions.ConnectionError as e:

            print()
            print(
                "CPCB API connection error:"
            )

            print(e)

            if attempt < MAX_API_RETRIES:

                print(
                    f"Retrying in "
                    f"{RETRY_DELAY_SECONDS} seconds..."
                )

                time.sleep(
                    RETRY_DELAY_SECONDS
                )

        # ----------------------------------------------------
        # HTTP ERROR
        # ----------------------------------------------------

        except requests.exceptions.HTTPError as e:

            print()
            print(
                "CPCB API HTTP error:"
            )

            print(e)

            # 4xx errors usually should not be blindly retried
            if response.status_code < 500:

                return pd.DataFrame()

            if attempt < MAX_API_RETRIES:

                print(
                    f"Retrying in "
                    f"{RETRY_DELAY_SECONDS} seconds..."
                )

                time.sleep(
                    RETRY_DELAY_SECONDS
                )

        # ----------------------------------------------------
        # OTHER REQUEST ERROR
        # ----------------------------------------------------

        except requests.exceptions.RequestException as e:

            print()
            print(
                "CPCB API request failed:"
            )

            print(e)

            if attempt < MAX_API_RETRIES:

                print(
                    f"Retrying in "
                    f"{RETRY_DELAY_SECONDS} seconds..."
                )

                time.sleep(
                    RETRY_DELAY_SECONDS
                )

        # ----------------------------------------------------
        # UNEXPECTED ERROR
        # ----------------------------------------------------

        except Exception as e:

            print()
            print(
                "Unexpected CPCB API error:"
            )

            print(
                type(e).__name__,
                ":",
                e
            )

            if attempt < MAX_API_RETRIES:

                print(
                    f"Retrying in "
                    f"{RETRY_DELAY_SECONDS} seconds..."
                )

                time.sleep(
                    RETRY_DELAY_SECONDS
                )

    # ========================================================
    # ALL ATTEMPTS FAILED
    # ========================================================

    print()
    print("=" * 70)
    print("CPCB API FAILED")
    print("=" * 70)

    print(
        f"All {MAX_API_RETRIES} API attempts failed."
    )

    return pd.DataFrame()


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

def normalize_columns(df):

    if df.empty:

        return df

    data = df.copy()

    # --------------------------------------------------------
    # LOWERCASE + STRIP
    # --------------------------------------------------------

    data.columns = [
        str(column)
        .strip()
        .lower()
        for column in data.columns
    ]

    # --------------------------------------------------------
    # CPCB COLUMN MAPPING
    # --------------------------------------------------------

    rename_map = {

        # Timestamp
        "date": "timestamp",
        "datetime": "timestamp",
        "last_update": "timestamp",
        "last updated": "timestamp",
        "timestamp": "timestamp",

        # PM2.5
        "pm2.5": "pm25",
        "pm2_5": "pm25",
        "pm25": "pm25",

        # PM10
        "pm10": "pm10",

        # Nitric oxide
        "no": "no",

        # Nitrogen dioxide
        "no2": "no2",

        # Ammonia
        "nh3": "nh3",

        # Sulphur dioxide
        "so2": "so2",

        # Carbon monoxide
        "co": "co",

        # Ozone
        "o3": "o3"
    }

    data = data.rename(
        columns=rename_map
    )

    return data


# ============================================================
# PREPARE VADODARA DATA
# ============================================================

def prepare_vadodara_data(df):

    if df.empty:

        return df

    data = normalize_columns(
        df
    )

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [

        "timestamp",

        "pm25",
        "pm10",
        "no",
        "no2",
        "nh3",
        "so2",
        "co",
        "o3"
    ]

    # --------------------------------------------------------
    # CREATE MISSING COLUMNS
    # --------------------------------------------------------

    for column in required_columns:

        if column not in data.columns:

            data[column] = np.nan

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    data["timestamp"] = pd.to_datetime(

        data["timestamp"],

        errors="coerce"
    )

    # --------------------------------------------------------
    # NUMERIC POLLUTANTS
    # --------------------------------------------------------

    pollutant_columns = [

        "pm25",
        "pm10",
        "no",
        "no2",
        "nh3",
        "so2",
        "co",
        "o3"
    ]

    for column in pollutant_columns:

        data[column] = pd.to_numeric(

            data[column],

            errors="coerce"
        )

    # --------------------------------------------------------
    # SELECT REQUIRED COLUMNS
    # --------------------------------------------------------

    data = data[
        required_columns
    ]

    # --------------------------------------------------------
    # REMOVE INVALID TIMESTAMPS
    # --------------------------------------------------------

    data = data.dropna(
        subset=["timestamp"]
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    data = (
        data
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    return data


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    raw_data = fetch_vadodara_data()

    prepared_data = prepare_vadodara_data(
        raw_data
    )

    print()
    print("=" * 70)
    print("PREPARED CPCB VADODARA DATA")
    print("=" * 70)

    if prepared_data.empty:

        print(
            "No usable records."
        )

    else:

        print(
            prepared_data.to_string(
                index=False
            )
        )