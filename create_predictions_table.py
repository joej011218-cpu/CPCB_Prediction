import os
import psycopg2


DATABASE_URL = os.getenv("DATABASE_URL")


def main():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not configured in this Terminal."
        )

    conn = psycopg2.connect(DATABASE_URL)

    try:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS aqi_predictions (
                forecast_origin TEXT NOT NULL,
                forecast_timestamp TEXT NOT NULL,
                horizon_hours INTEGER NOT NULL,
                current_aqi DOUBLE PRECISION,
                predicted_delta DOUBLE PRECISION,
                predicted_aqi DOUBLE PRECISION,
                PRIMARY KEY (forecast_origin, horizon_hours)
            )
            """
        )

        conn.commit()

        cur.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'aqi_predictions'
            ORDER BY ordinal_position
            """
        )

        columns = cur.fetchall()

        cur.execute(
            "SELECT COUNT(*) FROM aqi_predictions"
        )
        count = cur.fetchone()[0]

        print("=" * 70)
        print("AQI PREDICTIONS TABLE READY")
        print("=" * 70)

        for name, data_type in columns:
            print(f"{name:22s} {data_type}")

        print()
        print(f"Current rows: {count}")

        expected = {
            "forecast_origin",
            "forecast_timestamp",
            "horizon_hours",
            "current_aqi",
            "predicted_delta",
            "predicted_aqi",
        }

        actual = {name for name, _ in columns}

        if expected == actual:
            print("VALIDATION: PASS")
        else:
            print("VALIDATION: CHECK REQUIRED")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
