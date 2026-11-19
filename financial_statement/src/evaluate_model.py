import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
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


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "samin_financial_1403.db"

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


def evaluate_models(db_path: Path = DB_PATH) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"دیتابیس پیدا نشد: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM ml_financial_features;", conn)
        valid_df = df[df["revenue"].notna() & df["net_margin"].notna()].reset_index(drop=True)

        n_samples = len(valid_df)
        print(f"تعداد نمونه‌های معتبر: {n_samples}")

        if n_samples < 2:
            print("نمونه کافی برای ارزیابی وجود ندارد.")
            return

        X_raw = valid_df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
        active_features = [c for c in FEATURE_COLS if X_raw[c].notna().any()]
        X = X_raw[active_features]
        y_cls = valid_df["is_profitable"].astype(int)
        y_reg = valid_df["net_margin"].astype(float)

        # ۱) مدل‌های پایه (Baseline) برای مقایسه
        dummy_cls = DummyClassifier(strategy="most_frequent").fit(X, y_cls)
        dummy_reg = DummyRegressor(strategy="mean").fit(X, y_reg)

        # ۲) مدل‌های اصلی
        clf_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(random_state=42)),
            ]
        ).fit(X, y_cls)

        reg_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=42)),
            ]
        ).fit(X, y_reg)

        cls_pred = clf_pipe.predict(X)
        reg_pred = reg_pipe.predict(X)
        base_reg_pred = dummy_reg.predict(X)

        print("\n=== ارزیابی طبقه‌بندی (is_profitable) ===")
        print(f"Accuracy (مدل): {accuracy_score(y_cls, cls_pred):.2%}")
        print(f"Precision:        {precision_score(y_cls, cls_pred, zero_division=0):.2%}")
        print(f"Recall:           {recall_score(y_cls, cls_pred, zero_division=0):.2%}")
        print(f"F1-Score:         {f1_score(y_cls, cls_pred, zero_division=0):.2%}")

        print("\n=== ارزیابی رگرسیون (net_margin) در مقایسه با میانگین (Baseline) ===")
        print(f"MAE  (مدل): {mean_absolute_error(y_reg, reg_pred):.6f} | Baseline: {mean_absolute_error(y_reg, base_reg_pred):.6f}")
        print(f"RMSE (مدل): {root_mean_squared_error(y_reg, reg_pred):.6f} | Baseline: {root_mean_squared_error(y_reg, base_reg_pred):.6f}")

        # ۳) اعتبارسنجی متقابل اگر حداقل هر کلاس ۲ نمونه داشته باشد
        class_counts = y_cls.value_counts()
        min_class_count = int(class_counts.min()) if len(class_counts) > 1 else 0
        if n_samples >= 6 and min_class_count >= 2:
            n_splits = min(5, min_class_count)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(clf_pipe, X, y_cls, cv=cv, scoring="accuracy")
            print(f"\n=== اعتبارسنجی متقابل ({n_splits}-Fold CV) ===")
            print(f"میانگین دقت CV: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
        else:
            print("\n[توجه] تعداد نمونه‌ها هنوز برای اعتبارسنجی متقابل چندبخشی کم است؛ با افزودن فایل‌های مالی بیشتر، CV خودکار فعال می‌شود.")
    finally:
        conn.close()


if __name__ == "__main__":
    evaluate_models()
