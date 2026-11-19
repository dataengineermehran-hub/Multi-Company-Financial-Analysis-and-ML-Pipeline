from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import re
import sqlite3
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_DIR = Path(r"C:\Users\mehran\PycharmProjects\pythonProject\financial_statement")
RAW_DIR = PROJECT_DIR / "data" / "raw"
DB_PATH = PROJECT_DIR / "data" / "processed" / "companies_financial_1403.db"

SHEET_NAME_MAP: dict[str, str] = {
    "عمومی - نسبتها": "general_ratios",
    "صورت وضعیت مالی": "balance_sheet",
    "صورت سود و زیان": "income_statement",
    "صورت سود و زیان جامع": "comprehensive_income",
    "صورت تغییرات در حقوق مالکانه": "equity_changes",
    "صورت جریان های نقدی": "cash_flow",
    "سایر اطلاعات": "other_info",
    "معاملات اشخاص وابسته": "related_party_tx",
    "درآمدهای عملیاتی": "operating_revenue",
    "بهای تمام شده درآمدهای عملیاتی": "cogs_details",
    "هزینه های فروش، اداری و عمومی": "opex_details",
    "ساير درآمدها": "other_income",
    "سایر هزینه ها": "other_expenses",
    "هزینه های مالی": "finance_costs",
    "ساير درآمدها و هزينه هاى غيرعمل": "non_op_income_expense",
    "نقد حاصل از عملیات": "operating_cash_details",
    "فروش بتفکیک داخلی و خارجی": "sales_domestic_foreign",
    "خرید مواد و کالا بتفکیک داخلی و": "purchases_domestic_foreign",
}

PERSIAN_TO_LATIN_MAP: dict[str, str] = {
    "ا": "a", "آ": "a", "ب": "b", "پ": "p", "ت": "t", "ث": "s",
    "ج": "j", "چ": "ch", "ح": "h", "خ": "kh", "د": "d", "ذ": "z",
    "ر": "r", "ز": "z", "ژ": "zh", "س": "s", "ش": "sh", "ص": "s",
    "ض": "z", "ط": "t", "ظ": "z", "ع": "a", "غ": "gh", "ف": "f",
    "ق": "gh", "ک": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n",
    "و": "v", "ه": "h", "ی": "y", "ئ": "y",
}

FEATURE_COLS: list[str] = [
    "revenue",
    "cogs_abs",
    "gross_profit",
    "opex_abs",
    "operating_profit",
    "fin_cost_abs",
    "gross_margin",
    "operating_margin",
    "cogs_to_revenue",
    "opex_to_revenue",
    "fin_cost_to_revenue",
    "roa",
    "roe",
    "debt_to_assets",
    "cash_to_curr_liab",
    "ocf_to_net_profit",
    "revenue_yoy_growth",
]


def _norm_text(val: object) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).replace("\u200c", " ").replace("\u200f", " ").replace("\u00a0", " ")
    s = s.replace("ي", "ی").replace("ك", "ک")
    return re.sub(r"\s+", " ", s).strip()


def _clean_period(col_name: object) -> str | None:
    s = re.sub(r"\.\d+$", "", _norm_text(col_name))
    return s if re.match(r"^\d{4}/\d{2}/\d{2}$", s) else None


def _infer_company_name(file_path: Path) -> str:
    stem = re.sub(r"^Copy\s+of\s+", "", file_path.stem, flags=re.IGNORECASE)
    stem = re.sub(r"\s*140\d.*$|\s*-\s*نهائی.*$|\s*\(\d+\)\s*$", "", stem)
    return _norm_text(stem) or _norm_text(file_path.stem)


def _get_english_table_name(sheet_name: str) -> str:
    n_sheet = _norm_text(sheet_name)
    if n_sheet in SHEET_NAME_MAP:
        slug = SHEET_NAME_MAP[n_sheet]
    else:
        out = []
        for ch in n_sheet.lower():
            if ch.isascii() and ch.isalnum():
                out.append(ch)
            elif ch in PERSIAN_TO_LATIN_MAP:
                out.append(PERSIAN_TO_LATIN_MAP[ch])
            else:
                out.append("_")
        slug = re.sub(r"_+", "_", "".join(out)).strip("_") or "sheet"
    return f"tbl_{slug[:55]}"


def _parse_number(val: object) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = _norm_text(val).replace(",", "").replace("٬", "")
    if s in {"", "-", "—", "…", "…..", "………", "nan", "None"}:
        return None
    for i, ch in enumerate("۰۱۲۳۴۵۶۷۸۹"):
        s = s.replace(ch, str(i))
    for i, ch in enumerate("٠١٢٣٤٥٦٧٨٩"):
        s = s.replace(ch, str(i))
    neg_paren = bool(re.match(r"^\(.*\)$", s))
    s_clean = re.sub(r"[^\d.\-+]", "", s)
    try:
        num = float(s_clean)
        return -abs(num) if neg_paren else num
    except ValueError:
        return None


def _pick_value(sub_df: pd.DataFrame, candidates: list[str], prefer_nonzero: bool = True) -> float | None:
    norm_candidates = [_norm_text(c) for c in candidates]
    matches = sub_df[sub_df["line_norm"].isin(norm_candidates) & sub_df["value_num"].notna()]
    if matches.empty:
        return None
    for nc in norm_candidates:
        vals = matches[matches["line_norm"] == nc]["value_num"].astype(float).tolist()
        if vals:
            if prefer_nonzero:
                non_zero = [v for v in vals if v != 0.0]
                if non_zero:
                    return non_zero[0]
            return vals[0]
    return float(matches["value_num"].iloc[0])


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    res = float(numerator / denominator)
    return None if (np.isnan(res) or np.isinf(res)) else res


def task_ingest_excel_to_tables(**context) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    excel_files = sorted([p for p in RAW_DIR.glob("*.xlsx") if not p.name.startswith("~$")])
    if not excel_files:
        raise FileNotFoundError(f"هیچ فایل اکسل (.xlsx) در {RAW_DIR} پیدا نشد.")

    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    try:
        all_items: list[dict] = []
        raw_sheet_frames: dict[str, list[pd.DataFrame]] = {}

        for fpath in excel_files:
            company = _infer_company_name(fpath)
            xls = pd.ExcelFile(fpath, engine="openpyxl")
            for sname in xls.sheet_names:
                n_sname = _norm_text(sname)
                tname = _get_english_table_name(n_sname)
                sheet_code = tname.removeprefix("tbl_")

                df = pd.read_excel(fpath, sheet_name=sname, engine="openpyxl")
                if df.empty or len(df.columns) < 2:
                    continue

                df_raw_copy = df.copy()
                df_raw_copy.insert(0, "company_name", company)
                df_raw_copy.insert(1, "source_file", fpath.name)
                raw_sheet_frames.setdefault(tname, []).append(df_raw_copy)

                first_col = df.columns[0]
                period_cols = {col: p for col in df.columns[1:] if (p := _clean_period(col)) is not None}
                if not period_cols:
                    continue

                for _, row in df.iterrows():
                    line_item = _norm_text(row[first_col])
                    if not line_item:
                        continue
                    for col, period_str in period_cols.items():
                        raw_val = row[col]
                        all_items.append(
                            {
                                "company_name": company,
                                "source_file": fpath.name,
                                "table_name": tname,
                                "sheet_code": sheet_code,
                                "sheet_name": n_sname,
                                "line_item": line_item,
                                "period": period_str,
                                "value_num": _parse_number(raw_val),
                                "value_raw": None if pd.isna(raw_val) else str(raw_val),
                            }
                        )

        if not all_items:
            raise ValueError("هیچ قلم مالی معتبری از فایل‌های اکسل استخراج نشد.")

        items_df = pd.DataFrame(all_items)
        items_df.to_sql("statement_items", conn, if_exists="replace", index=False)

        for tname, frames in raw_sheet_frames.items():
            merged_raw = pd.concat(frames, ignore_index=True)
            merged_raw.to_sql(tname, conn, if_exists="replace", index=False)

        conn.commit()
        context["ti"].xcom_push(key="companies", value=sorted(items_df["company_name"].unique().tolist()))
        context["ti"].xcom_push(key="raw_tables", value=sorted(raw_sheet_frames.keys()))
        print(f"Ingested {len(items_df)} rows across {len(raw_sheet_frames)} English tables into {DB_PATH.name}.")
    finally:
        conn.close()


def task_validate_and_build_features(**context) -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"دیتابیس پیدا نشد: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    try:
        all_df = pd.read_sql_query(
            "SELECT company_name, table_name, sheet_code, sheet_name, line_item, period, value_num FROM statement_items;",
            conn,
        )
        if all_df.empty:
            raise ValueError("جدول statement_items خالی است.")

        val_summary = (
            all_df.groupby(["company_name", "table_name", "sheet_code"], as_index=False)
            .agg(
                total_rows=("value_num", "size"),
                valid_numeric_rows=("value_num", lambda s: int(s.notna().sum())),
                null_or_non_numeric_rows=("value_num", lambda s: int(s.isna().sum())),
            )
        )
        val_summary.to_sql("ml_validation_summary", conn, if_exists="replace", index=False)

        num_df = all_df[all_df["value_num"].notna()].copy()
        num_df["line_norm"] = num_df["line_item"].map(_norm_text)

        is_df = num_df[num_df["sheet_code"] == "income_statement"]
        bs_df = num_df[num_df["sheet_code"] == "balance_sheet"]
        cf_df = num_df[num_df["sheet_code"] == "cash_flow"]

        rows: list[dict] = []
        for company, comp_is in is_df.groupby("company_name"):
            comp_bs = bs_df[bs_df["company_name"] == company]
            comp_cf = cf_df[cf_df["company_name"] == company]

            for p in sorted(comp_is["period"].dropna().unique().tolist()):
                inc = comp_is[comp_is["period"] == p]
                bal = comp_bs[comp_bs["period"] == p]
                cf = comp_cf[comp_cf["period"] == p]

                revenue = _pick_value(inc, ["درآمدهای عملیاتی"])
                cogs_raw = _pick_value(inc, ["بهای تمام شده درآمدهای عملیاتی"])
                cogs_abs = abs(cogs_raw) if cogs_raw is not None else None
                gross_profit = _pick_value(inc, ["سود (زیان) ناخالص"])
                opex_raw = _pick_value(inc, ["هزینه های فروش، اداری و عمومی"])
                opex_abs = abs(opex_raw) if opex_raw is not None else None
                operating_profit = _pick_value(inc, ["سود (زیان) عملیاتی"])
                fin_raw = _pick_value(inc, ["هزینه‌های مالی", "هزینه های مالی"])
                fin_cost_abs = abs(fin_raw) if fin_raw is not None else None
                net_profit = _pick_value(inc, ["سود (زیان) خالص"])

                total_assets = _pick_value(
                    bal,
                    ["جمع حقوق مالکانه و بدهی‌ها", "جمع بدهی‌ها و حقوق مالکانه", "جمع دارایی‌ها"],
                )
                non_current_assets = _pick_value(bal, ["جمع دارایی‌های غیرجاری", "جمع دارايي‌هاي غیرجاري"])
                current_liabilities = _pick_value(bal, ["جمع بدهی‌های جاری"])
                equity = _pick_value(bal, ["جمع حقوق مالکانه"])
                cash = _pick_value(bal, ["موجودی نقد"])
                receivables = _pick_value(bal, ["دریافتنی‌های تجاری و سایر دریافتنی‌ها"])
                ocf = _pick_value(
                    cf,
                    [
                        "جریان خالص ورود (خروج) نقد حاصل از فعالیت‌های عملیاتی",
                        "جريان ‌خالص ‌ورود‌ (خروج) ‌نقد حاصل از فعاليت‌هاي ‌عملياتي",
                    ],
                )

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
                        "gross_margin": _safe_div(gross_profit, revenue),
                        "operating_margin": _safe_div(operating_profit, revenue),
                        "net_margin": _safe_div(net_profit, revenue),
                        "cogs_to_revenue": _safe_div(cogs_abs, revenue),
                        "opex_to_revenue": _safe_div(opex_abs, revenue),
                        "fin_cost_to_revenue": _safe_div(fin_cost_abs, revenue),
                        "roa": _safe_div(net_profit, total_assets),
                        "roe": _safe_div(net_profit, equity),
                        "debt_to_assets": _safe_div(current_liabilities, total_assets),
                        "cash_to_curr_liab": _safe_div(cash, current_liabilities),
                        "ocf_to_net_profit": _safe_div(ocf, net_profit),
                        "is_profitable": 1 if (net_profit is not None and net_profit > 0) else 0,
                    }
                )

        if not rows:
            raise ValueError("هیچ ردیف ویژگی معتبری ساخته نشد.")

        feat_df = pd.DataFrame(rows).sort_values(["company_name", "period"]).reset_index(drop=True)
        feat_df["revenue_yoy_growth"] = feat_df.groupby("company_name")["revenue"].pct_change()
        feat_df["net_profit_yoy_change"] = feat_df.groupby("company_name")["net_profit"].diff()

        feat_df.to_sql("ml_financial_features", conn, if_exists="replace", index=False)
        conn.commit()
        context["ti"].xcom_push(key="feature_rows", value=len(feat_df))
        print(f"Built {len(feat_df)} feature rows in ml_financial_features and saved ml_validation_summary.")
    finally:
        conn.close()


def task_train_and_predict_models(**context) -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"دیتابیس پیدا نشد: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM ml_financial_features;", conn)
        valid_df = df[df["revenue"].notna() & df["net_margin"].notna()].reset_index(drop=True)
        if len(valid_df) < 2:
            raise ValueError("تعداد نمونه‌های معتبر برای آموزش مدل کافی نیست.")

        X_raw = valid_df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
        active_features = [c for c in FEATURE_COLS if X_raw[c].notna().any()]
        X = X_raw[active_features]
        y_cls = valid_df["is_profitable"].astype(int)
        y_reg = valid_df["net_margin"].astype(float)

        clf_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ).fit(X, y_cls)

        reg_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=42)),
            ]
        ).fit(X, y_reg)

        dummy_reg = DummyRegressor(strategy="mean").fit(X, y_reg)

        cls_preds = clf_pipe.predict(X)
        cls_probs = clf_pipe.predict_proba(X)[:, 1]
        reg_preds = reg_pipe.predict(X)
        base_preds = dummy_reg.predict(X)

        acc = float(accuracy_score(y_cls, cls_preds))
        prec = float(precision_score(y_cls, cls_preds, zero_division=0))
        rec = float(recall_score(y_cls, cls_preds, zero_division=0))
        f1 = float(f1_score(y_cls, cls_preds, zero_division=0))
        mae = float(mean_absolute_error(y_reg, reg_preds))
        rmse = float(root_mean_squared_error(y_reg, reg_preds))
        base_mae = float(mean_absolute_error(y_reg, base_preds))
        base_rmse = float(root_mean_squared_error(y_reg, base_preds))

        pred_df = pd.DataFrame(
            {
                "company_name": valid_df["company_name"],
                "period": valid_df["period"],
                "actual_is_profitable": y_cls,
                "pred_is_profitable": cls_preds,
                "prob_profitable": np.round(cls_probs, 4),
                "actual_net_margin": np.round(y_reg, 6),
                "pred_net_margin": np.round(reg_preds, 6),
                "reg_abs_error": np.round(np.abs(y_reg - reg_preds), 6),
            }
        )
        pred_df.to_sql("ml_predictions", conn, if_exists="replace", index=False)

        coefs = clf_pipe.named_steps["model"].coef_[0]
        ridge_coefs = reg_pipe.named_steps["model"].coef_
        weights_df = (
            pd.DataFrame(
                {
                    "feature": active_features,
                    "logit_coef": np.round(coefs, 6),
                    "abs_logit_coef": np.round(np.abs(coefs), 6),
                    "ridge_coef": np.round(ridge_coefs, 6),
                }
            )
            .sort_values("abs_logit_coef", ascending=False)
            .reset_index(drop=True)
        )
        weights_df.to_sql("ml_feature_weights", conn, if_exists="replace", index=False)

        metrics_df = pd.DataFrame(
            [
                {
                    "trained_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
                    "n_samples": int(len(valid_df)),
                    "n_companies": int(valid_df["company_name"].nunique()),
                    "n_active_features": int(len(active_features)),
                    "accuracy": round(acc, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1_score": round(f1, 4),
                    "mae": round(mae, 6),
                    "rmse": round(rmse, 6),
                    "baseline_mae": round(base_mae, 6),
                    "baseline_rmse": round(base_rmse, 6),
                }
            ]
        )
        metrics_df.to_sql("ml_model_metrics", conn, if_exists="replace", index=False)
        conn.commit()
        context["ti"].xcom_push(key="metrics", value=metrics_df.iloc[0].to_dict())
        print("Models trained and saved to ml_predictions, ml_feature_weights, ml_model_metrics.")
    finally:
        conn.close()


def task_audit_output_tables(**context) -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"دیتابیس پیدا نشد: {DB_PATH}")

    expected_tables = [
        "statement_items",
        "tbl_income_statement",
        "tbl_balance_sheet",
        "tbl_cash_flow",
        "ml_validation_summary",
        "ml_financial_features",
        "ml_predictions",
        "ml_feature_weights",
        "ml_model_metrics",
    ]

    conn = sqlite3.connect(DB_PATH)
    try:
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }
        missing = [t for t in expected_tables if t not in existing]
        if missing:
            raise RuntimeError(f"جداول زیر در دیتابیس یافت نشدند: {missing}")

        counts = {}
        for t in expected_tables:
            cnt = int(conn.execute(f"SELECT COUNT(*) FROM [{t}];").fetchone()[0])
            if cnt <= 0:
                raise RuntimeError(f"جدول {t} خالی است (0 ردیف).")
            counts[t] = cnt

        print("Audit passed successfully. Table row counts:", counts)
    finally:
        conn.close()


default_args = {
    "owner": "financial-ml-platform",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="companies_financial_end_to_end_tables_dag",
    default_args=default_args,
    description="پایپلاین کامل استخراج صورت‌های مالی به جداول انگلیسی SQLite، اعتبارسنجی، مهندسی ویژگی و آموزش مدل",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["airflow", "finance", "sqlite", "ml", "tables"],
) as dag:

    t1_ingest = PythonOperator(
        task_id="ingest_excel_to_tables",
        python_callable=task_ingest_excel_to_tables,
    )

    t2_features = PythonOperator(
        task_id="validate_and_build_features",
        python_callable=task_validate_and_build_features,
    )

    t3_train = PythonOperator(
        task_id="train_and_predict_models",
        python_callable=task_train_and_predict_models,
    )

    t4_audit = PythonOperator(
        task_id="audit_output_tables",
        python_callable=task_audit_output_tables,
    )

    t1_ingest >> t2_features >> t3_train >> t4_audit