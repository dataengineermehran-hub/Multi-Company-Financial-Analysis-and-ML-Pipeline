import re
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "companies_financial_1403.db"


def norm_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u200c", " ").replace("\u200f", " ").replace("\u00a0", " ")
    s = s.replace("ي", "ی").replace("ك", "ک")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pick_value(sub_df: pd.DataFrame, candidates: list[str], prefer_nonzero: bool = True) -> float | None:
    norm_candidates = [norm_text(c) for c in candidates]
    matches = sub_df[sub_df["line_norm"].isin(norm_candidates) & sub_df["value_num"].notna()]
    if matches.empty:
        return None
    for nc in norm_candidates:
        m = matches[matches["line_norm"] == nc]["value_num"].astype(float).tolist()
        if m:
            if prefer_nonzero:
                non_zero = [v for v in m if v != 0.0]
                if non_zero:
                    return non_zero[0]
            return m[0]
    return float(matches["value_num"].iloc[0])


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    res = float(numerator / denominator)
    return None if (np.isnan(res) or np.isinf(res)) else res


def build_features(db_path: Path = DB_PATH) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"دیتابیس پیدا نشد: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        items_df = pd.read_sql_query(
            """
            SELECT company_name, sheet_code, sheet_name, line_item, period, value_num
            FROM statement_items
            WHERE value_num IS NOT NULL;
            """,
            conn,
        )
        if items_df.empty:
            raise ValueError("جدول statement_items خالی است.")

        items_df["line_norm"] = items_df["line_item"].map(norm_text)

        is_df = items_df[items_df["sheet_code"] == "income_statement"]
        bs_df = items_df[items_df["sheet_code"] == "balance_sheet"]
        cf_df = items_df[items_df["sheet_code"] == "cash_flow"]

        rows: list[dict] = []

        for company, comp_is in is_df.groupby("company_name"):
            comp_bs = bs_df[bs_df["company_name"] == company]
            comp_cf = cf_df[cf_df["company_name"] == company]

            periods = sorted(comp_is["period"].dropna().unique().tolist())
            for p in periods:
                inc = comp_is[comp_is["period"] == p]
                bal = comp_bs[comp_bs["period"] == p]
                cf = comp_cf[comp_cf["period"] == p]

                revenue = pick_value(inc, ["درآمدهای عملیاتی"])
                cogs_raw = pick_value(inc, ["بهای تمام شده درآمدهای عملیاتی"])
                cogs_abs = abs(cogs_raw) if cogs_raw is not None else None
                gross_profit = pick_value(inc, ["سود (زیان) ناخالص"])
                opex_raw = pick_value(inc, ["هزینه های فروش، اداری و عمومی"])
                opex_abs = abs(opex_raw) if opex_raw is not None else None
                operating_profit = pick_value(inc, ["سود (زیان) عملیاتی"])
                fin_raw = pick_value(inc, ["هزینه‌های مالی", "هزینه های مالی"])
                fin_cost_abs = abs(fin_raw) if fin_raw is not None else None
                net_profit = pick_value(inc, ["سود (زیان) خالص"])

                total_assets = pick_value(
                    bal,
                    ["جمع حقوق مالکانه و بدهی‌ها", "جمع بدهی‌ها و حقوق مالکانه", "جمع دارایی‌ها"],
                )
                non_current_assets = pick_value(bal, ["جمع دارایی‌های غیرجاری", "جمع دارايي‌هاي غیرجاري"])
                current_liabilities = pick_value(bal, ["جمع بدهی‌های جاری"])
                equity = pick_value(bal, ["جمع حقوق مالکانه"])
                cash = pick_value(bal, ["موجودی نقد"])
                receivables = pick_value(bal, ["دریافتنی‌های تجاری و سایر دریافتنی‌ها"])
                ocf = pick_value(
                    cf,
                    [
                        "جریان خالص ورود (خروج) نقد حاصل از فعالیت‌های عملیاتی",
                        "جريان ‌خالص ‌ورود‌ (خروج) ‌نقد حاصل از فعاليت‌هاي ‌عملياتي",
                    ],
                )

                gross_margin = safe_div(gross_profit, revenue)
                operating_margin = safe_div(operating_profit, revenue)
                net_margin = safe_div(net_profit, revenue)
                cogs_to_revenue = safe_div(cogs_abs, revenue)
                opex_to_revenue = safe_div(opex_abs, revenue)
                fin_cost_to_revenue = safe_div(fin_cost_abs, revenue)
                roa = safe_div(net_profit, total_assets)
                roe = safe_div(net_profit, equity)
                debt_to_assets = safe_div(current_liabilities, total_assets)
                cash_to_curr_liab = safe_div(cash, current_liabilities)
                ocf_to_net_profit = safe_div(ocf, net_profit)

                is_profitable = 1 if (net_profit is not None and net_profit > 0) else 0

                rows.append(
                    {
                        "company_name": company,
                        "period": p,
                        "revenue": revenue,
                        "cogs_abs": cogs_abs,
                        "gross_profit": gross_profit,
                        "opex_abs": opex_abs,
                        "operating_profit": operating_profit,
                        "fin_cost_abs": fin_cost_abs,
                        "net_profit": net_profit,
                        "total_assets": total_assets,
                        "non_current_assets": non_current_assets,
                        "current_liabilities": current_liabilities,
                        "equity": equity,
                        "cash": cash,
                        "receivables": receivables,
                        "operating_cash_flow": ocf,
                        "gross_margin": gross_margin,
                        "operating_margin": operating_margin,
                        "net_margin": net_margin,
                        "cogs_to_revenue": cogs_to_revenue,
                        "opex_to_revenue": opex_to_revenue,
                        "fin_cost_to_revenue": fin_cost_to_revenue,
                        "roa": roa,
                        "roe": roe,
                        "debt_to_assets": debt_to_assets,
                        "cash_to_curr_liab": cash_to_curr_liab,
                        "ocf_to_net_profit": ocf_to_net_profit,
                        "is_profitable": is_profitable,
                    }
                )

        if not rows:
            raise ValueError("هیچ ردیف ویژگی معتبری ساخته نشد.")

        feat_df = pd.DataFrame(rows).sort_values(["company_name", "period"]).reset_index(drop=True)
        feat_df["revenue_yoy_growth"] = feat_df.groupby("company_name")["revenue"].pct_change()
        feat_df["net_profit_yoy_change"] = feat_df.groupby("company_name")["net_profit"].diff()

        feat_df.to_sql("ml_financial_features", conn, if_exists="replace", index=False)
        conn.commit()

        print("=== جدول ویژگی‌های مالی ساخته شد (ml_financial_features) ===")
        cols_preview = [
            "company_name",
            "period",
            "revenue",
            "net_profit",
            "gross_margin",
            "net_margin",
            "roa",
            "roe",
            "is_profitable",
        ]
        print(feat_df[cols_preview].to_string(index=False))
    finally:
        conn.close()


if __name__ == "__main__":
    build_features()
