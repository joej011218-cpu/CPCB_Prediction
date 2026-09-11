import os
import sqlite3

import pandas as pd
import psycopg2


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SQLITE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "cpcb_vadodara.db"
)

DATABASE_URL = os.getenv(
    "DATABASE_URL"
)


def using_postgres():
    return bool(DATABASE_URL)


def database_available():

    if using_postgres():
        return True

    return os.path.exists(
        SQLITE_PATH
    )


def get_connection():

    if using_postgres():

        return psycopg2.connect(
            DATABASE_URL
        )

    return sqlite3.connect(
        SQLITE_PATH
    )


def sql_placeholder():

    if using_postgres():
        return "%s"

    return "?"


def read_dataframe(query, params=None):

    conn = get_connection()

    try:

        if using_postgres():

            cursor = conn.cursor()

            cursor.execute(
                query,
                params
            )

            rows = cursor.fetchall()

            columns = [
                description[0]
                for description
                in cursor.description
            ]

            return pd.DataFrame(
                rows,
                columns=columns
            )

        return pd.read_sql_query(
            query,
            conn,
            params=params
        )

    finally:

        conn.close()
