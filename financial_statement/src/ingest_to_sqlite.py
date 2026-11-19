import re
import sqlite3
import sys
from pathlib import Path
import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
DB_PATH = BASE_DIR / "data" / "processed" / "companies_financial_1403.db"

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


def norm_text(val: object) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).replace("\u200c", " ").replace("\u200f", " ").replace("\u00a0", " ")
    s = s.replace("ي", "ی").replace("ك", "ک")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_period(col_name: object) -> str | None:
    s = norm_text(col_name)
    s = re.sub(r"\.\d+$", "", s)
    if re.match(r"^\d{4}/\d{2}/\d{2}$", s):
        return s
    return None


def infer_company_name(file_path: Path) -> str:
    stem = file_path.stem
    stem = re.sub(r"^Copy\s+of\s+", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\s*140\d.*$", "", stem)
    stem = re.sub(r"\s*-\s*نهائی.*$", "", stem)
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)
    return norm_text(stem) or norm_text(file_path.stem)


def to_english_slug(text: str) -> str:
    cleaned = norm_text(text).lower()
    out = []
    for ch in cleaned:
        if ch.isascii() and ch.isalnum():
            out.append(ch)
        elif ch in PERSIAN_TO_LATIN_MAP:
            out.append(PERSIAN_TO_LATIN_MAP[ch])
        else:
            out.append("_")
    slug = re.sub(r"_+", "_", "".join(out)).strip("_")
    return slug or "sheet"


def get_english_table_name(sheet_name: str) -> str:
    n_sheet = norm_text(sheet_name)
    if n_sheet in SHEET_NAME_MAP:
        slug = SHEET_NAME_MAP[n_sheet]
    else:
        slug = to_english_slug(n_sheet)
    return f"tbl_{slug[:55]}"


def parse_number(val: object) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = norm_text(val).replace(",", "").replace("٬", "")
    if s in {"", "-", "—", "…", "…..", "………", "nan", "None"}:
        return None
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    for i, ch in enumerate(persian_digits):
        s = s.replace(ch, str(i))
    for i, ch in enumerate(arabic_digits):
        s = s.replace(ch, str(i))
    neg_paren = bool(re.match(r"^\(.*\)$", s))
    s_num = re.sub(r"[^\d.\-+]", "", s)
    try:
        num = float(s_num)
        return -abs(num) if neg_paren else num
    except ValueError:
        return None


def ingest_excel_files(raw_dir: Path = RAW_DIR, db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    excel_files = sorted(
        [p for p in raw_dir.glob("*.xlsx") if not p.name.startswith("~$")]
    )
    if not excel_files:
        raise FileNotFoundError(f"هیچ فایل اکسل (.xlsx) در مسیر {raw_dir} پیدا نشد.")

    # بازسازی تمیز دیتابیس با نام جدید
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        all_items: list[dict] = []
        raw_sheet_frames: dict[str, list[pd.DataFrame]] = {}

        for fpath in excel_files:
            company = infer_company_name(fpath)
            xls = pd.ExcelFile(fpath, engine="openpyxl")
            print(f"در حال پردازش فایل: {fpath.name} | شرکت: {company} | شیت‌ها: {len(xls.sheet_names)}")

            for sname in xls.sheet_names:
                n_sname = norm_text(sname)
                tname = get_english_table_name(n_sname)
                sheet_code = tname.removeprefix("tbl_")

                df = pd.read_excel(fpath, sheet_name=sname, engine="openpyxl")
                if df.empty or len(df.columns) < 2:
                    continue

                df_raw_copy = df.copy()
                df_raw_copy.insert(0, "company_name", company)
                df_raw_copy.insert(1, "source_file", fpath.name)
                raw_sheet_frames.setdefault(tname, []).append(df_raw_copy)

                first_col = df.columns[0]
                period_cols = {col: p for col in df.columns[1:] if (p := clean_period(col)) is not None}
                if not period_cols:
                    continue

                for _, row in df.iterrows():
                    line_item = norm_text(row[first_col])
                    if not line_item:
                        continue
                    for col, period_str in period_cols.items():
                        raw_val = row[col]
                        num_val = parse_number(raw_val)
                        all_items.append(
                            {
                                "company_name": company,
                                "source_file": fpath.name,
                                "table_name": tname,
                                "sheet_code": sheet_code,
                                "sheet_name": n_sname,
                                "line_item": line_item,
                                "period": period_str,
                                "value_num": num_val,
                                "value_raw": None if pd.isna(raw_val) else str(raw_val),
                            }
                        )

        items_df = pd.DataFrame(all_items)
        items_df.to_sql("statement_items", conn, if_exists="replace", index=False)

        for tname, frames in raw_sheet_frames.items():
            merged_raw = pd.concat(frames, ignore_index=True)
            merged_raw.to_sql(tname, conn, if_exists="replace", index=False)

        conn.commit()
        print(f"\n=== دیتابیس جدید ساخته شد: {db_path.name} ===")
        print("جداول صورت‌های مالی ایجادشده:", sorted(raw_sheet_frames.keys()))
    finally:
        conn.close()


if __name__ == "__main__":
    ingest_excel_files()
