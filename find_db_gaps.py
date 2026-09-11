import sqlite3
import pandas as pd

DB_PATH = "data/cpcb_vadodara.db"

# ============================================================
# LOAD DATABASE
# ============================================================

conn = sqlite3.connect(DB_PATH)

df = pd.read_sql_query(
    """
    SELECT
        id,
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
    FROM cpcb_hourly
    ORDER BY timestamp
    """,
    conn
)

conn.close()

print("=" * 70)
print("CPCB VADODARA DATABASE GAP ANALYSIS")
print("=" * 70)

print()
print("Database:", DB_PATH)
print("Rows:", len(df))

# ============================================================
# TIMESTAMP PREPARATION
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df = df.dropna(
    subset=["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

# ============================================================
# BASIC RANGE
# ============================================================

start = df["timestamp"].min()
end = df["timestamp"].max()

print()
print("First timestamp:", start)
print("Last timestamp: ", end)

# ============================================================
# EXPECTED HOURLY TIMESTAMPS
# ============================================================

expected = pd.date_range(
    start=start,
    end=end,
    freq="1h"
)

actual = pd.DatetimeIndex(
    df["timestamp"].drop_duplicates()
)

missing = expected.difference(
    actual
)

duplicates = (
    df["timestamp"]
    .duplicated(keep=False)
)

duplicate_timestamps = (
    df.loc[
        duplicates,
        "timestamp"
    ]
    .drop_duplicates()
    .sort_values()
)

# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print()
print("Expected hourly records:", len(expected))
print("Actual unique timestamps:", len(actual))
print("Missing timestamps:", len(missing))
print("Duplicate timestamps:", len(duplicate_timestamps))

# ============================================================
# SHOW MISSING TIMESTAMPS
# ============================================================

print()
print("=" * 70)
print("MISSING TIMESTAMPS")
print("=" * 70)

if len(missing) == 0:

    print()
    print("NO MISSING HOURLY TIMESTAMPS.")

else:

    for ts in missing:

        print(
            ts.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

# ============================================================
# GROUP MISSING HOURS INTO GAPS
# ============================================================

print()
print("=" * 70)
print("MISSING TIME RANGES")
print("=" * 70)

if len(missing) > 0:

    missing_series = pd.Series(
        missing
    )

    gap_groups = (
        missing_series
        .diff()
        .ne(pd.Timedelta(hours=1))
        .cumsum()
    )

    gaps = (
        missing_series
        .groupby(gap_groups)
        .agg(
            start="min",
            end="max",
            hours="count"
        )
        .reset_index(drop=True)
    )

    print()

    for _, row in gaps.iterrows():

        print(
            f"{row['start']} "
            f"→ "
            f"{row['end']} "
            f"({row['hours']} hour(s))"
        )

    print()
    print("Number of separate gaps:", len(gaps))

# ============================================================
# SAVE GAP LIST
# ============================================================

if len(missing) > 0:

    gap_df = pd.DataFrame({
        "timestamp": missing
    })

    gap_df.to_csv(
        "cpcb_missing_timestamps.csv",
        index=False
    )

    print()
    print(
        "Missing timestamp list saved to:"
    )

    print(
        "cpcb_missing_timestamps.csv"
    )

# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("GAP ANALYSIS COMPLETE")
print("=" * 70)
