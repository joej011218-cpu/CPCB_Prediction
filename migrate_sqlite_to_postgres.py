import os
import sqlite3

import psycopg2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SQLITE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cpcb_vadodara.db"
)

DATABASE_URL = os.getenv("DATABASE_URL")

TABLE_NAME = "cpcb_hourly"

COLUMNS = [
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


def main():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not configured. "
            "Run: export DATABASE_URL='YOUR_RENDER_EXTERNAL_DATABASE_URL'"
        )

    if not os.path.exists(SQLITE_PATH):
        raise FileNotFoundError(
            f"SQLite database not found: {SQLITE_PATH}"
        )

    print("=" * 70)
    print("MIGRATING CPCB SQLITE -> RENDER POSTGRESQL")
    print("=" * 70)
    print(f"SQLite source: {SQLITE_PATH}")
    print()

    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_conn.row_factory = sqlite3.Row

    postgres_conn = psycopg2.connect(DATABASE_URL)

    try:
        sqlite_cur = sqlite_conn.cursor()
        pg_cur = postgres_conn.cursor()

        pg_cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cpcb_hourly (
                timestamp TEXT PRIMARY KEY,
                pm25 DOUBLE PRECISION,
                pm10 DOUBLE PRECISION,
                no DOUBLE PRECISION,
                no2 DOUBLE PRECISION,
                nh3 DOUBLE PRECISION,
                so2 DOUBLE PRECISION,
                co DOUBLE PRECISION,
                o3 DOUBLE PRECISION
            )
            """
        )
        postgres_conn.commit()

        sqlite_cur.execute(
            """
            SELECT
                timestamp,
                pm25,
                pm10,
                no,
                no2,
                nh3,
                so2,
                co,
                o3
            FROM cpcb_hourly
            ORDER BY timestamp
            """
        )

        rows = sqlite_cur.fetchall()

        print(f"SQLite rows found: {len(rows)}")

        upsert_sql = """
            INSERT INTO cpcb_hourly (
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
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (timestamp) DO UPDATE SET
                pm25 = EXCLUDED.pm25,
                pm10 = EXCLUDED.pm10,
                no = EXCLUDED.no,
                no2 = EXCLUDED.no2,
                nh3 = EXCLUDED.nh3,
                so2 = EXCLUDED.so2,
                co = EXCLUDED.co,
                o3 = EXCLUDED.o3
        """

        migrated = 0

        for row in rows:
            values = tuple(row[column] for column in COLUMNS)
            pg_cur.execute(upsert_sql, values)
            migrated += 1

            if migrated % 500 == 0:
                postgres_conn.commit()
                print(f"Migrated: {migrated}/{len(rows)}")

        postgres_conn.commit()

        pg_cur.execute(
            "SELECT COUNT(*) FROM cpcb_hourly"
        )
        postgres_count = pg_cur.fetchone()[0]

        pg_cur.execute(
            """
            SELECT MIN(timestamp), MAX(timestamp)
            FROM cpcb_hourly
            """
        )
        first_timestamp, last_timestamp = pg_cur.fetchone()

        print()
        print("=" * 70)
        print("MIGRATION COMPLETE")
        print("=" * 70)
        print(f"Rows processed : {migrated}")
        print(f"Postgres rows  : {postgres_count}")
        print(f"First timestamp: {first_timestamp}")
        print(f"Last timestamp : {last_timestamp}")

        if postgres_count == len(rows):
            print()
            print("VALIDATION: PASS")
        else:
            print()
            print(
                "VALIDATION: CHECK REQUIRED "
                f"(SQLite={len(rows)}, Postgres={postgres_count})"
            )

    finally:
        sqlite_conn.close()
        postgres_conn.close()


if __name__ == "__main__":
    main()
