from database import get_connection, using_postgres


def save_predictions_to_postgres(results_df):
    """
    Save a 1-24 hour forecast DataFrame into Render PostgreSQL.

    Expected columns:
      forecast_origin
      forecast_timestamp
      horizon_hours
      current_aqi
      predicted_delta
      predicted_aqi

    Local SQLite mode is left unchanged; the predictor can still write CSV.
    """
    if not using_postgres():
        print()
        print("=" * 80)
        print("POSTGRESQL FORECAST SAVE")
        print("=" * 80)
        print("DATABASE_URL is not set.")
        print("Skipping PostgreSQL forecast save.")
        return

    conn = get_connection()

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

        upsert_sql = """
            INSERT INTO aqi_predictions (
                forecast_origin,
                forecast_timestamp,
                horizon_hours,
                current_aqi,
                predicted_delta,
                predicted_aqi
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (forecast_origin, horizon_hours)
            DO UPDATE SET
                forecast_timestamp = EXCLUDED.forecast_timestamp,
                current_aqi = EXCLUDED.current_aqi,
                predicted_delta = EXCLUDED.predicted_delta,
                predicted_aqi = EXCLUDED.predicted_aqi
        """

        for row in results_df.itertuples(index=False):
            cur.execute(
                upsert_sql,
                (
                    str(row.forecast_origin),
                    str(row.forecast_timestamp),
                    int(row.horizon_hours),
                    float(row.current_aqi),
                    float(row.predicted_delta),
                    float(row.predicted_aqi),
                )
            )

        conn.commit()

        latest_origin = str(
            results_df["forecast_origin"].iloc[0]
        )

        cur.execute(
            """
            SELECT COUNT(*)
            FROM aqi_predictions
            WHERE forecast_origin = %s
            """,
            (latest_origin,)
        )

        saved_count = cur.fetchone()[0]

        print()
        print("=" * 80)
        print("POSTGRESQL FORECAST SAVED")
        print("=" * 80)
        print("Forecast origin:", latest_origin)
        print("Rows stored:", saved_count)

        if saved_count == len(results_df):
            print("VALIDATION: PASS")
        else:
            print(
                "VALIDATION: CHECK REQUIRED "
                f"(expected {len(results_df)}, found {saved_count})"
            )

    finally:
        conn.close()
