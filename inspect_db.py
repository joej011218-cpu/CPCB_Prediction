import sqlite3
import os

DB_PATH = "data/cpcb_vadodara.db"

print("=" * 70)
print("DATABASE INSPECTION")
print("=" * 70)

if not os.path.exists(DB_PATH):
    print(f"ERROR: {DB_PATH} not found.")
    print("Current folder:", os.getcwd())
    raise SystemExit

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# ------------------------------------------------------------
# TABLES
# ------------------------------------------------------------

cur.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type='table'
    ORDER BY name
""")

tables = cur.fetchall()

print("\nTables:")
for table in tables:
    print("  -", table[0])

# ------------------------------------------------------------
# INSPECT EACH TABLE
# ------------------------------------------------------------

for (table_name,) in tables:

    print()
    print("=" * 70)
    print("TABLE:", table_name)
    print("=" * 70)

    cur.execute(f'PRAGMA table_info("{table_name}")')

    columns = cur.fetchall()

    for column in columns:

        cid, name, data_type, notnull, default, pk = column

        print(
            f"{name:25s} "
            f"{data_type:12s} "
            f"NOT NULL={notnull} "
            f"PK={pk} "
            f"DEFAULT={default}"
        )

    # --------------------------------------------------------
    # ROW COUNT
    # --------------------------------------------------------

    cur.execute(
        f'SELECT COUNT(*) FROM "{table_name}"'
    )

    count = cur.fetchone()[0]

    print("\nRows:", count)

    # --------------------------------------------------------
    # FIRST/LAST TIMESTAMP IF POSSIBLE
    # --------------------------------------------------------

    column_names = [
        column[1]
        for column in columns
    ]

    timestamp_column = None

    for candidate in [
        "timestamp",
        "datetime",
        "date",
        "time"
    ]:

        if candidate in column_names:

            timestamp_column = candidate
            break

    if timestamp_column:

        cur.execute(
            f'''
            SELECT
                MIN("{timestamp_column}"),
                MAX("{timestamp_column}")
            FROM "{table_name}"
            '''
        )

        result = cur.fetchone()

        print(
            "Time range:",
            result[0],
            "→",
            result[1]
        )

conn.close()

print()
print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
