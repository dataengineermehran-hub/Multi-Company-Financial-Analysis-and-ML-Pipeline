import os
import sqlite3
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
RAW_DIR = BASE_DIR / "data" / "raw"
DB_PATH = BASE_DIR / "data" / "processed" / "companies_financial_1403.db"

FEATURE_COLS = [
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

st.set_page_config(
    page_title="داشبورد تحلیل مالی و یادگیری ماشین چندشرکتی",
    layout="wide",
)


@st.cache_data
def load_table(table_name: str, db_mtime: float) -> pd.DataFrame:
    _ = db_mtime
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(f"SELECT * FROM [{table_name}];", conn)
    finally:
        conn.close()


def list_raw_sheet_tables(db_mtime: float) -> list[str]:
    _ = db_mtime
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name LIKE 'tbl_%'
            ORDER BY name;
            """
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def run_full_pipeline() -> tuple[bool, str]:
    logs = []
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    for script_name in ["ingest_to_sqlite.py", "build_features.py", "train_model.py"]:
        script_path = SRC_DIR / script_name
        if not script_path.exists():
            return False, f"فایل {script_name} پیدا نشد."
        res = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        logs.append(f"--- {script_name} (exit={res.returncode}) ---")
        if res.stdout:
            logs.append(res.stdout.strip())
        if res.stderr:
            logs.append(res.stderr.strip())
        if res.returncode != 0:
            return False, "\n".join(logs)
    st.cache_data.clear()
    return True, "\n".join(logs)


def compute_model_diagnostics(feat_df: pd.DataFrame) -> dict:
    valid_df = feat_df[feat_df["revenue"].notna() & feat_df["net_margin"].notna()].reset_index(drop=True)
    n_samples = len(valid_df)
    if n_samples < 2:
        return {"ok": False, "n_samples": n_samples}

    X_raw = valid_df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    active_features = [c for c in FEATURE_COLS if X_raw[c].notna().any()]
    dropped_features = [c for c in FEATURE_COLS if c not in active_features]
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

    cls_pred = clf_pipe.predict(X)
    reg_pred = reg_pipe.predict(X)
    base_reg_pred = dummy_reg.predict(X)

    coefs = clf_pipe.named_steps["model"].coef_[0]
    weights_df = (
        pd.DataFrame({"feature": active_features, "coef": coefs, "abs_coef": np.abs(coefs)})
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )

    class_counts = y_cls.value_counts()
    min_class_count = int(class_counts.min()) if len(class_counts) > 1 else 0
    cv_info = None
    if n_samples >= 4 and min_class_count >= 2:
        n_splits = min(4, min_class_count)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf_pipe, X, y_cls, cv=cv, scoring="accuracy")
        cv_info = f"{n_splits}-Fold CV: {scores.mean():.2%} (+/- {scores.std():.2%})"

    return {
        "ok": True,
        "n_samples": n_samples,
        "n_companies": valid_df["company_name"].nunique() if "company_name" in valid_df.columns else 1,
        "active_features": active_features,
        "dropped_features": dropped_features,
        "accuracy": accuracy_score(y_cls, cls_pred),
        "precision": precision_score(y_cls, cls_pred, zero_division=0),
        "recall": recall_score(y_cls, cls_pred, zero_division=0),
        "f1": f1_score(y_cls, cls_pred, zero_division=0),
        "mae": mean_absolute_error(y_reg, reg_pred),
        "base_mae": mean_absolute_error(y_reg, base_reg_pred),
        "rmse": root_mean_squared_error(y_reg, reg_pred),
        "base_rmse": root_mean_squared_error(y_reg, base_reg_pred),
        "weights_df": weights_df,
        "cv_info": cv_info,
    }


def main() -> None:
    st.title("داشبورد تحلیل مالی و خروجی یادگیری ماشین چندشرکتی")
    st.caption("به‌روزرسانی خودکار با افزودن فایل‌های اکسل جدید در پوشه data/raw و بازآموزی مدل")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with st.sidebar:
        st.header("مدیریت فایل‌ها و پایپلاین")
        uploaded_files = st.file_uploader(
            "افزودن فایل اکسل جدید (.xlsx):",
            type=["xlsx"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            for uf in uploaded_files:
                dest = RAW_DIR / uf.name
                dest.write_bytes(uf.getbuffer())
            st.success(f"{len(uploaded_files)} فایل در data/raw ذخیره شد.")

        if st.button("به‌روزرسانی کامل و بازآموزی مدل", use_container_width=True):
            with st.spinner("در حال خواندن فایل‌ها، ساخت ویژگی‌ها و آموزش مجدد مدل..."):
                ok, output_log = run_full_pipeline()
            if ok:
                st.success("پایپلاین کامل با موفقیت اجرا شد.")
            else:
                st.error("خطا در اجرای پایپلاین.")
            with st.expander("مشاهده لاگ اجرا", expanded=not ok):
                st.code(output_log, language="text")

    if not DB_PATH.exists():
        st.warning("هنوز دیتابیس ساخته نشده است. فایل‌های اکسل را در data/raw قرار دهید و دکمه «به‌روزرسانی کامل و بازآموزی مدل» را بزنید.")
        return

    db_mtime = DB_PATH.stat().st_mtime
    pred_df = load_table("ml_predictions", db_mtime)
    feat_df = load_table("ml_financial_features", db_mtime)
    items_df = load_table("statement_items", db_mtime)
    raw_tables = list_raw_sheet_tables(db_mtime)

    companies = ["همه شرکت‌ها"]
    if "company_name" in pred_df.columns and not pred_df.empty:
        companies += sorted(pred_df["company_name"].dropna().unique().tolist())

    selected_company = st.selectbox("انتخاب شرکت برای بررسی جزئیات:", companies)

    filtered_pred = pred_df.copy()
    filtered_feat = feat_df.copy()
    filtered_items = items_df.copy()

    if selected_company != "همه شرکت‌ها":
        if "company_name" in filtered_pred.columns:
            filtered_pred = filtered_pred[filtered_pred["company_name"] == selected_company]
        if "company_name" in filtered_feat.columns:
            filtered_feat = filtered_feat[filtered_feat["company_name"] == selected_company]
        if "company_name" in filtered_items.columns:
            filtered_items = filtered_items[filtered_items["company_name"] == selected_company]

    # ۱) خروجی مدل یادگیری ماشین
    st.subheader("۱. خروجی مدل یادگیری ماشین (جدول ml_predictions)")
    if not filtered_pred.empty:
        row = filtered_pred.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        pred_cls_label = "سودده (1)" if int(row["pred_is_profitable"]) == 1 else "زیان‌ده (0)"
        actual_cls_label = "سودده" if int(row["actual_is_profitable"]) == 1 else "زیان‌ده"
        c1.metric("پیش‌بینی وضعیت سوددهی", pred_cls_label, delta=f"واقعی: {actual_cls_label}")
        c2.metric("احتمال سوددهی", f"{float(row['prob_profitable']) * 100:.2f}%")
        c3.metric("حاشیه سود واقعی", f"{float(row['actual_net_margin']) * 100:.2f}%")
        c4.metric(
            "حاشیه سود پیش‌بینی‌شده",
            f"{float(row['pred_net_margin']) * 100:.2f}%",
            delta=f"{(float(row['pred_net_margin']) - float(row['actual_net_margin'])) * 100:+.2f}%",
        )

        st.dataframe(filtered_pred, use_container_width=True, hide_index=True)

        plot_pred = filtered_pred.copy()
        if "company_name" in plot_pred.columns and selected_company == "همه شرکت‌ها":
            plot_pred["label"] = plot_pred["company_name"] + " (" + plot_pred["period"] + ")"
        else:
            plot_pred["label"] = plot_pred["period"]

        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            st.markdown("**مقایسه حاشیه سود خالص واقعی و پیش‌بینی‌شده**")
            st.line_chart(plot_pred.set_index("label")[["actual_net_margin", "pred_net_margin"]])
        with ch_col2:
            st.markdown("**احتمال پیش‌بینی‌شده سوددهی (prob_profitable)**")
            st.bar_chart(plot_pred.set_index("label")[["prob_profitable"]])
    else:
        st.warning("خروجی پیش‌بینی برای انتخاب فعلی وجود ندارد.")

    st.divider()

    # ۲) ارزیابی و تشخیص سلامت مدل روی کل مجموعه داده
    st.subheader("۲. ارزیابی عملکرد و تشخیص سلامت مدل (روی کل دیتاست)")
    if not feat_df.empty:
        diag = compute_model_diagnostics(feat_df)
        if diag["ok"]:
            st.caption(
                f"تعداد نمونه‌های معتبر: {diag['n_samples']} در {diag['n_companies']} شرکت | "
                f"ویژگی‌های فعال: {len(diag['active_features'])} از {len(FEATURE_COLS)}"
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy (طبقه‌بندی)", f"{diag['accuracy']:.2%}")
            m2.metric("F1-Score (طبقه‌بندی)", f"{diag['f1']:.2%}")
            m3.metric(
                "MAE (رگرسیون)",
                f"{diag['mae']:.6f}",
                delta=f"Baseline: {diag['base_mae']:.6f}",
                delta_color="inverse",
            )
            m4.metric(
                "RMSE (رگرسیون)",
                f"{diag['rmse']:.6f}",
                delta=f"Baseline: {diag['base_rmse']:.6f}",
                delta_color="inverse",
            )

            if diag["cv_info"]:
                st.success(f"اعتبارسنجی متقابل -> {diag['cv_info']}")

            st.markdown("**وزن ویژگی‌ها در مدل طبقه‌بندی**")
            st.dataframe(diag["weights_df"], use_container_width=True, hide_index=True)

    st.divider()

    # ۳) جدول ویژگی‌ها و نسبت‌های مالی
    st.subheader("۳. ویژگی‌ها و نسبت‌های مالی (جدول ml_financial_features)")
    if not filtered_feat.empty:
        st.dataframe(filtered_feat, use_container_width=True, hide_index=True)

    st.divider()

    # ۴) کاوشگر اقلام صورت‌های مالی
    st.subheader("۴. کاوشگر اقلام صورت‌های مالی")
    tab1, tab2 = st.tabs(["اقلام استاندارد (statement_items)", "جدول خام هر شیت (tbl_*)"])

    with tab1:
        if not filtered_items.empty:
            sheets = ["همه شیت‌ها"] + sorted(filtered_items["sheet_name"].dropna().unique().tolist())
            col_f1, col_f2 = st.columns([1, 2])
            chosen_sheet = col_f1.selectbox("فیلتر شیت:", sheets)
            search_text = col_f2.text_input("جستجو در نام سرفصل (line_item):", "")

            sub_items = filtered_items.copy()
            if chosen_sheet != "همه شیت‌ها":
                sub_items = sub_items[sub_items["sheet_name"] == chosen_sheet]
            if search_text.strip():
                sub_items = sub_items[
                    sub_items["line_item"].str.contains(search_text.strip(), case=False, na=False)
                ]
            st.dataframe(sub_items, use_container_width=True, hide_index=True)

    with tab2:
        if raw_tables:
            chosen_tbl = st.selectbox("انتخاب جدول خام شیت:", raw_tables)
            tbl_df = load_table(chosen_tbl, db_mtime)
            st.dataframe(tbl_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
