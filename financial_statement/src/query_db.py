import sqlite3
from pathlib import Path
import pandas as pd


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "samin_financial_1403.db"


def run_sample_queries() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"دیتابیس پیدا نشد: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    try:
        objects_df = pd.read_sql_query(
            """
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table', 'view')
            ORDER BY type, name;
            """,
            conn,
        )
        print("=== لیست جداول و ویوها در SQLite ===")
        print(objects_df.to_string(index=False))

        income_df = pd.read_sql_query(
            """
            SELECT line_item, period, value_num
            FROM statement_items
            WHERE sheet_name = 'صورت سود و زیان'
              AND line_item IN (
                  'درآمدهای عملیاتی',
                  'بهای تمام شده درآمدهای عملیاتی',
                  'سود (زیان) ناخالص',
                  'سود (زیان) عملیاتی',
                  'سود (زیان) خالص'
              )
            ORDER BY row_order, period;
            """,
            conn,
        )
        print("\n=== اقلام کلیدی صورت سود و زیان ===")
        print(income_df.pivot(index="line_item", columns="period", values="value_num").to_string())
    finally:
        conn.close()


if __name__ == "__main__":
    run_sample_queries()
