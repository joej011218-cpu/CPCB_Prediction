import os
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

EXCEL_PATH = "RealTimeReport_Vadodara.xlsx"

# ============================================================
# CHECK FILE
# ============================================================

print("=" * 70)
print("CPCB EXCEL FILE INSPECTION")
print("=" * 70)

if not os.path.exists(EXCEL_PATH):

    print()
    print("ERROR: File not found:")
    print(EXCEL_PATH)

    print()
    print("Current folder:")
    print(os.getcwd())

    raise SystemExit

# ============================================================
# SHOW SHEETS
# ============================================================

excel = pd.ExcelFile(EXCEL_PATH)

print()
print("Excel sheets:")
print(excel.sheet_names)

# ============================================================
# INSPECT EACH SHEET
# ============================================================

for sheet_name in excel.sheet_names:

    print()
    print("=" * 70)
    print("SHEET:", sheet_name)
    print("=" * 70)

    df = pd.read_excel(
        EXCEL_PATH,
        sheet_name=sheet_name
    )

    print()
    print("Rows:", len(df))
    print("Columns:")

    for i, column in enumerate(df.columns):

        print(f"{i}: {repr(column)}")

    print()
    print("First 10 rows:")

    print(
        df.head(10).to_string(
            index=False
        )
    )

print()
print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
