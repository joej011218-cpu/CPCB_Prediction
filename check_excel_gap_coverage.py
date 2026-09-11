import sqlite3
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "data/cpcb_vadodara.db"
EXCEL_PATH = "RealTimeReport_Vadodara.xlsx"

TABLE_NAME = "cpcb_hourly"


# ============================================================
# LOAD DATABASE TIMESTAMPS
# ============================================================

print("=" * 70)
print("CHECKING CPCB EXCEL GAP COVERAGE")
print("=" * 70)

conn = sqlite3.connect(DB_PATH)

db_df = pd.read_sql_query(
    f"""
    SELECT timestamp
    FROM {TABLE_NAME}
    """,
    conn
)

conn.close()

db_df["timestamp"] = pd.to_datetime(
    db_df["timestamp"],
    errors="coerce"
)

db_df = db_df.dropna(
    subset=["timestamp"]
)

db_timestamps = pd.DatetimeIndex(
    db_df["timestamp"].drop_duplicates()
)


# ============================================================
# CREATE EXPECTED HOURLY RANGE
# ============================================================

start = db_timestamps.min()
end = db_timestamps.max()

expected_timestamps = pd.date_range(
    start=start,
    end=end,
    freq="h"
)

missing_timestamps = expected_timestamps.difference(
    db_timestamps
)

print()
print("Database timestamps:", len(db_timestamps))
print("Expected timestamps:", len(expected_timestamps))
print("Missing timestamps:", len(missing_timestamps))


# ============================================================
# LOAD EXCEL
# ============================================================

excel_df = pd.read_excel(
    EXCEL_PATH
)

# Remove whitespace from column names
excel_df.columns = [
    str(column).strip().lower()
    for column in excel_df.columns
]

print()
print("Excel rows:", len(excel_df))
print("Excel columns:")
print(excel_df.columns.tolist())


# ============================================================
# PARSE EXCEL TIMESTAMP
# ============================================================

excel_df["timestamp"] = pd.to_datetime(
    excel_df["timestamp"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

excel_df = excel_df.dropna(
    subset=["timestamp"]
)

excel_timestamps = pd.DatetimeIndex(
    excel_df["timestamp"].drop_duplicates()
)


# ============================================================
# COMPARE
# ============================================================

recoverable = missing_timestamps.intersection(
    excel_timestamps
)

still_missing = missing_timestamps.difference(
    excel_timestamps
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("COVERAGE RESULTS")
print("=" * 70)

print()
print("Missing DB timestamps:", len(missing_timestamps))
print("Recoverable from Excel:", len(recoverable))
print("Still unavailable:", len(still_missing))


# ============================================================
# RECOVERABLE GAPS
# ============================================================

print()
print("=" * 70)
print("RECOVERABLE TIMESTAMPS")
print("=" * 70)

if len(recoverable) == 0:

    print("None")

else:

    for ts in recoverable:

        print(
            ts.strftime("%Y-%m-%d %H:%M:%S")
        )


# ============================================================
# STILL MISSING
# ============================================================

print()
print("=" * 70)
print("STILL MISSING FROM EXCEL")
print("=" * 70)

if len(still_missing) == 0:

    print("None")

else:

    for ts in still_missing:

        print(
            ts.strftime("%Y-%m-%d %H:%M:%S")
        )


# ============================================================
# SAVE RESULTS
# ============================================================

pd.DataFrame({
    "timestamp": recoverable
}).to_csv(
    "recoverable_from_excel.csv",
    index=False
)

pd.DataFrame({
    "timestamp": still_missing
}).to_csv(
    "still_missing_after_excel.csv",
    index=False
)


print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print("recoverable_from_excel.csv")
print("still_missing_after_excel.csv")


print()
print("=" * 70)
print("CHECK COMPLETE — DATABASE NOT MODIFIED")
print("=" * 70)
