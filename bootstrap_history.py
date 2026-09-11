import sqlite3
import pandas as pd


# ============================================================
# PATHS
# ============================================================

EXCEL_PATH = (
    "/Users/khaingthinzartun/"
    "AQI-Test/datasets/"
    "RealTimeReport_Vadodara.xlsx"
)

DATABASE_PATH = "data/cpcb_vadodara.db"


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("BOOTSTRAPPING VADODARA HISTORICAL DATA")
print("=" * 70)


# ============================================================
# LOAD EXCEL
# ============================================================

print()
print("Loading Excel file:")
print(EXCEL_PATH)

df = pd.read_excel(EXCEL_PATH)

print()
print("Original rows:", len(df))


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.lower()
)

print()
print("Columns:")
print(df.columns.tolist())


# ============================================================
# REQUIRED COLUMNS
# ============================================================

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

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

df = df.dropna(
    subset=["timestamp"]
)


# ============================================================
# SORT AND REMOVE DUPLICATES
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

print()
print(
    "Rows after timestamp cleaning:",
    len(df)
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

pollutants = [
    "pm25",
    "pm10",
    "no",
    "no2",
    "nh3",
    "so2",
    "co",
    "o3"
]

for col in pollutants:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# CONNECT TO DATABASE
# ============================================================

print()
print("Opening database:")
print(DATABASE_PATH)

conn = sqlite3.connect(
    DATABASE_PATH
)

cursor = conn.cursor()


# ============================================================
# CHECK DATABASE COLUMNS
# ============================================================

columns = cursor.execute(
    "PRAGMA table_info(cpcb_hourly)"
).fetchall()

database_columns = [
    column[1]
    for column in columns
]

print()
print("Database columns:")
print(database_columns)


if "no" not in database_columns:

    conn.close()

    raise RuntimeError(
        "The database does not contain the 'no' column. "
        "Run the ALTER TABLE command first."
    )


# ============================================================
# INSERT HISTORICAL DATA
# ============================================================

inserted = 0
skipped = 0

print()
print("Importing historical records...")


for _, row in df.iterrows():

    timestamp = row["timestamp"].strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    try:

        cursor.execute(
            """
            INSERT INTO cpcb_hourly (
                timestamp,
                pm25,
                pm10,
                no,
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
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?
            )
            """,
            (
                timestamp,

                row["pm25"],
                row["pm10"],
                row["no"],
                row["co"],
                row["no2"],
                row["nh3"],
                row["so2"],
                row["o3"],

                "Historical Vadodara Dataset",

                None,
                None,

                "historical_bootstrap"
            )
        )

        inserted += 1

    except sqlite3.IntegrityError:

        skipped += 1


# ============================================================
# COMMIT
# ============================================================

conn.commit()


# ============================================================
# DATABASE SUMMARY
# ============================================================

total_rows = cursor.execute(
    """
    SELECT COUNT(*)
    FROM cpcb_hourly
    """
).fetchone()[0]


first_timestamp = cursor.execute(
    """
    SELECT MIN(timestamp)
    FROM cpcb_hourly
    """
).fetchone()[0]


last_timestamp = cursor.execute(
    """
    SELECT MAX(timestamp)
    FROM cpcb_hourly
    """
).fetchone()[0]


# ============================================================
# CLOSE DATABASE
# ============================================================

conn.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("BOOTSTRAP COMPLETE")
print("=" * 70)

print()
print("Historical Excel rows:", len(df))

print(
    "New rows inserted:",
    inserted
)

print(
    "Duplicate rows skipped:",
    skipped
)

print(
    "Total database rows:",
    total_rows
)

print(
    "First database timestamp:",
    first_timestamp
)

print(
    "Last database timestamp:",
    last_timestamp
)

print()
print("=" * 70)
print("DATABASE READY FOR PREDICTOR")
print("=" * 70)