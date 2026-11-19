import sqlite3
from pathlib import Path
import pandas as pd


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "samin_financial_1403.db"


def validate_financial_db(db_path: Path = DB_PATH) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"دیتابیس پیدا نشد: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT sheet_name, line_item, period, value_num
            FROM statement_items
            WHERE sheet_name = 'صورت سود و زیان' AND value_num IS NOT NULL;
            """,
            conn,
        )

        piv = df.pivot(index="line_item", columns="period", values="value_num")
        print("=== کنترل جمع‌های کنترلی صورت سود و زیان ===")

        for period in piv.columns:
            rev = piv.loc["درآمدهای عملیاتی", period] if "درآمدهای عملیاتی" in piv.index else 0.0
            cogs = piv.loc["بهای تمام شده درآمدهای عملیاتی", period] if "بهای تمام شده درآمدهای عملیاتی" in piv.index else 0.0
            gp_reported = piv.loc["سود (زیان) ناخالص", period] if "سود (زیان) ناخالص" in piv.index else 0.0
            gp_calc = rev + cogs  # چون بهای تمام‌شده با علامت منفی ثبت شده است

            op_exp = piv.loc["هزینه های فروش، اداری و عمومی", period] if "هزینه های فروش، اداری و عمومی" in piv.index else 0.0
            op_inc_reported = piv.loc["سود (زیان) عملیاتی", period] if "سود (زیان) عملیاتی" in piv.index else 0.0
            op_inc_calc = gp_reported + op_exp

            print(
                f"دوره {period}: "
                f"سود ناخالص محاسباتی={gp_calc:,.1f} | گزارش‌شده={gp_reported:,.1f} | "
                f"سود عملیاتی محاسباتی={op_inc_calc:,.1f} | گزارش‌شده={op_inc_reported:,.1f}"
            )

        null_stats = pd.read_sql_query(
            """
            SELECT
                sheet_name,
                COUNT(*) AS total_rows,
                SUM(CASE WHEN value_num IS NULL THEN 1 ELSE 0 END) AS non_numeric_or_null
            FROM statement_items
            GROUP BY sheet_name
            ORDER BY sheet_name;
            """,
            conn,
        )
        print("\n=== وضعیت مقادیر عددی در هر شیت ===")
        print(null_stats.to_string(index=False))
    finally:
        conn.close()


if __name__ == "__main__":
    validate_financial_db()
