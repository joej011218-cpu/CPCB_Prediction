import sqlite3
import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

EXCEL_PATH = (
    "/Users/khaingthinzartun/AQI-Test/datasets/"
    "RealTimeReport_Vadodara.xlsx"
)

DATABASE_PATH = "data/cpcb_vadodara.db"


# ============================================================
# CREATE DATABASE DIRECTORY
# ============================================================

os.makedirs(
    os.path.dirname(DATABASE_PATH),
    exist_ok=True
)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

print()
print("=" * 70)
print("INITIALIZING CPCB DATABASE")
print("=" * 70)

print()
print("Historical dataset:")
print(EXCEL_PATH)

print()
print("Database:")
print(DATABASE_PATH)


df = pd.read_excel(EXCEL_PATH)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)


print()
print("Columns found:")
print(df.columns.tolist())


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "timestamp",
    "pm25",
    "pm10",
    "co",
    "no2",
    "nh3",
    "so2",
    "o3"
]


missing = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing:

    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)


df = df.dropna(
    subset=["timestamp"]
)


# ============================================================
# SORT
# ============================================================

df = (
    df
    .sort_values("timestamp")
    .drop_duplicates(
        subset=["timestamp"],
        keep="first"
    )
    .reset_index(drop=True)
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

pollutants = [
    "pm25",
    "pm10",
    "co",
    "no2",
    "nh3",
    "so2",
    "o3"
]


for col in pollutants:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# REMOVE NEGATIVE VALUES
# ============================================================

for col in pollutants:

    df.loc[
        df[col] < 0,
        col
    ] = None


# ============================================================
# CREATE SQLITE DATABASE
# ============================================================

conn = sqlite3.connect(
    DATABASE_PATH
)

cursor = conn.cursor()


cursor.execute("""
    CREATE TABLE IF NOT EXISTS cpcb_hourly (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        timestamp TEXT UNIQUE,

        pm25 REAL,
        pm10 REAL,
        co REAL,
        no2 REAL,
        nh3 REAL,
        so2 REAL,
        o3 REAL,

        station TEXT,
        latitude REAL,
        longitude REAL,

        collected_at TEXT
    )
""")


conn.commit()


# ============================================================
# INSERT HISTORICAL DATA
# ============================================================

inserted = 0
skipped = 0


for _, row in df.iterrows():

    timestamp = row["timestamp"].strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    try:

        cursor.execute("""
            INSERT OR IGNORE INTO cpcb_hourly (

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

                collected_at

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (

            timestamp,

            row["pm25"],
            row["pm10"],
            row["co"],
            row["no2"],
            row["nh3"],
            row["so2"],
            row["o3"],

            "Historical Vadodara Dataset",

            None,
            None,

            timestamp
        ))


        if cursor.rowcount == 1:

            inserted += 1

        else:

            skipped += 1


    except Exception as e:

        print(
            "Error inserting:",
            timestamp,
            e
        )


# ============================================================
# COMMIT
# ============================================================

conn.commit()


# ============================================================
# DATABASE SUMMARY
# ============================================================

cursor.execute(
    "SELECT COUNT(*) FROM cpcb_hourly"
)

total_rows = cursor.fetchone()[0]


cursor.execute(
    "SELECT MIN(timestamp), MAX(timestamp) "
    "FROM cpcb_hourly"
)

min_timestamp, max_timestamp = cursor.fetchone()


conn.close()


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("DATABASE INITIALIZATION COMPLETE")
print("=" * 70)

print()
print("Historical rows found:", len(df))

print("Rows inserted:", inserted)

print("Rows skipped:", skipped)

print("Total database rows:", total_rows)

print()
print("First timestamp:")
print(min_timestamp)

print()
print("Last timestamp:")
print(max_timestamp)

print()
print("Database:")
print(DATABASE_PATH)

print()
print("=" * 70)