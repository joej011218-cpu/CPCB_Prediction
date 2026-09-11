import os
import shutil
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "data/cpcb_vadodara.db"

EXCEL_PATH = "RealTimeReport_Vadodara.xlsx"

TABLE_NAME = "cpcb_hourly"

BACKUP_FOLDER = "data/backups"


# ============================================================
# HELPER
# ============================================================

def safe_float(value):

    if pd.isna(value):

        return None

    try:

        value = float(value)

        if np.isnan(value):

            return None

        return value

    except Exception:

        return None


# ============================================================
# START
# ============================================================

print("=" * 70)
print("IMPORT CPCB EXCEL DATA INTO SQLITE")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(DB_PATH):

    raise FileNotFoundError(
        f"Database not found: {DB_PATH}"
    )


if not os.path.exists(EXCEL_PATH):

    raise FileNotFoundError(
        f"Excel file not found: {EXCEL_PATH}"
    )


# ============================================================
# LOAD DATABASE TIMESTAMPS
# ============================================================

print()
print("Loading database timestamps...")

conn = sqlite3.connect(DB_PATH)

db_df = pd.read_sql_query(
    f"""
    SELECT timestamp
    FROM {TABLE_NAME}
    """,
    conn
)

db_df["timestamp"] = pd.to_datetime(
    db_df["timestamp"],
    errors="coerce"
)

db_df = db_df.dropna(
    subset=["timestamp"]
)

existing_timestamps = set(
    db_df["timestamp"]
)


print(
    "Existing database timestamps:",
    len(existing_timestamps)
)


# ============================================================
# LOAD EXCEL
# ============================================================

print()
print("Loading CPCB Excel file...")

excel_df = pd.read_excel(
    EXCEL_PATH
)

# Clean column names

excel_df.columns = [

    str(column)
    .strip()
    .lower()

    for column in excel_df.columns
]


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [

    "timestamp",

    "pm25",
    "pm10",

    "no2",
    "nh3",
    "so2",

    "co",
    "o3"
]


missing_columns = [

    column

    for column in required_columns

    if column not in excel_df.columns
]


if missing_columns:

    raise ValueError(

        "Excel file is missing required columns: "
        f"{missing_columns}"

    )


# ============================================================
# PARSE TIMESTAMPS
# ============================================================

excel_df["timestamp"] = pd.to_datetime(

    excel_df["timestamp"],

    format="%d-%m-%Y %H:%M",

    errors="coerce"
)


excel_df = excel_df.dropna(

    subset=["timestamp"]

)


# ============================================================
# REMOVE DUPLICATE EXCEL TIMESTAMPS
# ============================================================

excel_df = (

    excel_df

    .sort_values("timestamp")

    .drop_duplicates(

        subset=["timestamp"],

        keep="last"

    )

    .reset_index(drop=True)

)


print(
    "Excel unique timestamps:",
    len(excel_df)
)


# ============================================================
# FIND ONLY NEW TIMESTAMPS
# ============================================================

rows_to_insert = excel_df.loc[

    ~excel_df["timestamp"].isin(

        existing_timestamps

    )

].copy()


print()
print(
    "Rows available for insertion:",
    len(rows_to_insert)
)


# ============================================================
# STOP IF NOTHING TO INSERT
# ============================================================

if rows_to_insert.empty:

    print()

    print(
        "No new timestamps found."
    )

    print(
        "Database was not modified."
    )

    conn.close()

    raise SystemExit


# ============================================================
# CREATE BACKUP
# ============================================================

print()
print("=" * 70)
print("CREATING DATABASE BACKUP")
print("=" * 70)


os.makedirs(

    BACKUP_FOLDER,

    exist_ok=True

)


backup_timestamp = datetime.now().strftime(

    "%Y%m%d_%H%M%S"

)


backup_path = os.path.join(

    BACKUP_FOLDER,

    f"cpcb_vadodara_before_excel_import_"
    f"{backup_timestamp}.db"

)


# Close before copying

conn.close()


shutil.copy2(

    DB_PATH,

    backup_path

)


print(
    "Backup created:"
)

print(
    backup_path
)


# ============================================================
# REOPEN DATABASE
# ============================================================

conn = sqlite3.connect(

    DB_PATH

)


cursor = conn.cursor()


# ============================================================
# INSERT QUERY
# ============================================================

insert_query = f"""

INSERT INTO {TABLE_NAME}

(

    timestamp,

    pm25,
    pm10,

    co,
    no2,
    nh3,
    so2,
    o3,

    station,

    latitude,
    longitude,

    collected_at,

    no

)

VALUES

(

    ?,

    ?,
    ?,

    ?,
    ?,
    ?,
    ?,
    ?,

    ?,

    ?,
    ?,

    ?,

    ?

)

"""


# ============================================================
# INSERT RECORDS
# ============================================================

print()
print("=" * 70)
print("INSERTING MISSING CPCB RECORDS")
print("=" * 70)


inserted = 0


for _, row in rows_to_insert.iterrows():

    timestamp = row["timestamp"]

    timestamp_string = timestamp.strftime(

        "%Y-%m-%d %H:%M:%S"

    )


    values = (

        # timestamp

        timestamp_string,


        # pollutants

        safe_float(row["pm25"]),

        safe_float(row["pm10"]),

        safe_float(row["co"]),

        safe_float(row["no2"]),

        safe_float(row["nh3"]),

        safe_float(row["so2"]),

        safe_float(row["o3"]),


        # station

        None,


        # latitude

        None,


        # longitude

        None,


        # collected_at

        datetime.now().strftime(

            "%Y-%m-%d %H:%M:%S"

        ),


        # no

        None

    )


    try:

        cursor.execute(

            insert_query,

            values

        )

        inserted += 1


    except sqlite3.IntegrityError as e:

        print()

        print(

            "Skipped timestamp due to database constraint:"

        )

        print(

            timestamp_string

        )

        print(

            e

        )


# ============================================================
# COMMIT
# ============================================================

conn.commit()


print()
print(
    "Records inserted:",
    inserted
)


# ============================================================
# VERIFY
# ============================================================

print()
print("=" * 70)
print("VERIFYING DATABASE")
print("=" * 70)


verify_df = pd.read_sql_query(

    f"""

    SELECT timestamp

    FROM {TABLE_NAME}

    """,

    conn

)


verify_df["timestamp"] = pd.to_datetime(

    verify_df["timestamp"],

    errors="coerce"

)


verify_df = verify_df.dropna(

    subset=["timestamp"]

)


unique_timestamps = pd.DatetimeIndex(

    verify_df["timestamp"]
    .drop_duplicates()

)


start = unique_timestamps.min()

end = unique_timestamps.max()


expected = pd.date_range(

    start=start,

    end=end,

    freq="h"

)


remaining_missing = expected.difference(

    unique_timestamps

)


print()

print(
    "Total database rows:",
    len(verify_df)
)

print(
    "Unique timestamps:",
    len(unique_timestamps)
)

print(
    "Remaining missing timestamps:",
    len(remaining_missing)
)


# ============================================================
# SHOW REMAINING GAPS
# ============================================================

print()
print("=" * 70)
print("REMAINING GAPS")
print("=" * 70)


if len(remaining_missing) == 0:

    print()

    print(
        "NO REMAINING GAPS."
    )


else:

    for ts in remaining_missing:

        print(

            ts.strftime(

                "%Y-%m-%d %H:%M:%S"

            )

        )


# ============================================================
# CLOSE
# ============================================================

conn.close()


print()
print("=" * 70)
print("IMPORT COMPLETE")
print("=" * 70)

print()

print(
    "Backup:"
)

print(
    backup_path
)

print()

print(
    "Original database updated:"
)

print(
    DB_PATH
)
