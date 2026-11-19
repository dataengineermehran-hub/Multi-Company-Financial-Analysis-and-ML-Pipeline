import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "companies_financial_1403.db"

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


def train_and_evaluate(db_path: Path = DB_PATH) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"دیتابیس پیدا نشد: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM ml_financial_features;", conn)
        valid_df = df[df["revenue"].notna() & df["net_margin"].notna()].reset_index(drop=True)

        if len(valid_df) < 2:
            raise ValueError("تعداد ردیف‌های معتبر برای آموزش مدل کافی نیست.")

        X_raw = valid_df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
        active_features = [col for col in FEATURE_COLS if X_raw[col].notna().any()]
        dropped_features = [col for col in FEATURE_COLS if col not in active_features]

        if dropped_features:
            print(f"ویژگی‌های بدون مقدار معتبر که کنار گذاشته شدند: {dropped_features}")

        X = X_raw[active_features]
        y_cls = valid_df["is_profitable"].astype(int)
        y_reg = valid_df["net_margin"].astype(float)

        clf_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        clf_pipe.fit(X, y_cls)
        cls_preds = clf_pipe.predict(X)
        cls_probs = clf_pipe.predict_proba(X)[:, 1]
        acc = accuracy_score(y_cls, cls_preds)
        f1 = f1_score(y_cls, cls_preds, zero_division=0)

        reg_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=42)),
            ]
        )
        reg_pipe.fit(X, y_reg)
        reg_preds = reg_pipe.predict(X)

        base_reg = DummyRegressor(strategy="mean").fit(X, y_reg)
        base_preds = base_reg.predict(X)

        mae = mean_absolute_error(y_reg, reg_preds)
        rmse = root_mean_squared_error(y_reg, reg_preds)
        base_mae = mean_absolute_error(y_reg, base_preds)
        base_rmse = root_mean_squared_error(y_reg, base_preds)

        print("\n=== نتایج آموزش چندشرکتی مدل یادگیری ماشین ===")
        print(f"تعداد نمونه‌های معتبر: {len(valid_df)} | تعداد شرکت‌ها: {valid_df['company_name'].nunique()} | ویژگی‌های فعال: {len(active_features)}")
        print(f"دقت طبقه‌بندی (Accuracy): {acc:.2%} | F1-Score: {f1:.2%}")
        print(f"MAE مدل: {mae:.6f} (Baseline: {base_mae:.6f})")
        print(f"RMSE مدل: {rmse:.6f} (Baseline: {base_rmse:.6f})")

        class_counts = y_cls.value_counts()
        min_class_count = int(class_counts.min()) if len(class_counts) > 1 else 0
        if len(valid_df) >= 4 and min_class_count >= 2:
            n_splits = min(4, min_class_count)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(clf_pipe, X, y_cls, cv=cv, scoring="accuracy")
            print(f"اعتبارسنجی متقابل ({n_splits}-Fold CV Accuracy): {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

        coefs = clf_pipe.named_steps["model"].coef_[0]
        imp_df = (
            pd.DataFrame({"feature": active_features, "coef": coefs, "abs_coef": np.abs(coefs)})
            .sort_values("abs_coef", ascending=False)
            .head(5)
        )
        print("\n=== ۵ ویژگی با بیشترین وزن در مدل طبقه‌بندی ===")
        print(imp_df[["feature", "coef"]].to_string(index=False))

        pred_df = pd.DataFrame(
            {
                "company_name": valid_df["company_name"],
                "period": valid_df["period"],
                "actual_is_profitable": y_cls,
                "pred_is_profitable": cls_preds,
                "prob_profitable": np.round(cls_probs, 4),
                "actual_net_margin": np.round(y_reg, 6),
                "pred_net_margin": np.round(reg_preds, 6),
            }
        )
        pred_df.to_sql("ml_predictions", conn, if_exists="replace", index=False)
        conn.commit()

        print("\n=== خروجی پیش‌بینی ذخیره شد در ml_predictions ===")
        print(pred_df.to_string(index=False))
    finally:
        conn.close()


if __name__ == "__main__":
    train_and_evaluate()
