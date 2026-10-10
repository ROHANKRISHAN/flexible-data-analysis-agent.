from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional

def load_dataset(file_obj_or_path):
    """Load CSV without requiring specific column names."""
    if hasattr(file_obj_or_path, "read"):
        try:
            file_obj_or_path.seek(0)
        except Exception:
            pass
        df = pd.read_csv(file_obj_or_path)
    else:
        df = pd.read_csv(file_obj_or_path)
    df.columns = [str(c).strip() for c in df.columns]
    # Avoid silently accepting completely empty or duplicate header names.
    if df.empty:
        raise ValueError("The uploaded dataset has no rows.")
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("The dataset has duplicate column names. Rename duplicate headers and upload again.")
    return df

def profile_dataset(df: pd.DataFrame) -> dict:
    """Infer likely roles from data types and column-name hints, but do not require names."""
    roles = {}
    date_cols, numeric_cols, categorical_cols, id_cols = [], [], [], []
    for col in df.columns:
        s = df[col]
        name = str(col).lower().replace("_", " ").replace("-", " ")
        nonnull = s.dropna()
        parsed_dates = pd.to_datetime(nonnull, errors="coerce") if len(nonnull) else pd.Series(dtype="datetime64[ns]")
        date_ratio = float(parsed_dates.notna().mean()) if len(nonnull) else 0
        name_suggests_date = any(w in name for w in ["date", "time", "day", "month", "year", "timestamp"])
        if (date_ratio >= 0.85 and (name_suggests_date or not pd.api.types.is_numeric_dtype(s))) and len(nonnull):
            date_cols.append(col)
            roles[col] = "date"
            continue
        converted = pd.to_numeric(s, errors="coerce")
        numeric_ratio = float(converted.notna().mean()) if len(nonnull) else 0
        name_suggests_id = any(w in name for w in ["id", "key", "code", "reference", "ref", "number"])
        if pd.api.types.is_numeric_dtype(s) or (numeric_ratio >= 0.9 and len(nonnull)):
            if name_suggests_id and s.nunique(dropna=True) > max(20, int(len(s)*0.7)):
                id_cols.append(col); roles[col] = "identifier"
            else:
                numeric_cols.append(col); roles[col] = "numeric"
        elif s.nunique(dropna=True) <= max(50, int(max(1,len(s))*0.25)):
            categorical_cols.append(col); roles[col] = "categorical"
        elif name_suggests_id:
            id_cols.append(col); roles[col] = "identifier"
        else:
            categorical_cols.append(col); roles[col] = "categorical"
    return {
        "rows": int(len(df)), "columns": int(len(df.columns)),
        "column_names": list(df.columns), "roles": roles,
        "date_columns": date_cols, "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols, "identifier_columns": id_cols,
        "missing_by_column": {str(c): int(df[c].isna().sum()) for c in df.columns},
        "duplicate_rows": int(df.duplicated().sum()),
    }

def clean_dataset(df: pd.DataFrame, drop_exact_duplicates=False):
    """Conservative cleaning: trim headers, parse likely dates and numeric-looking columns."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if drop_exact_duplicates:
        out = out.drop_duplicates().copy()
    profile = profile_dataset(out)
    for col in profile["date_columns"]:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    for col in profile["numeric_columns"]:
        # Convert numeric-like strings while preserving already numeric values.
        converted = pd.to_numeric(out[col], errors="coerce")
        if converted.notna().sum() > 0:
            out[col] = converted
    return out, profile_dataset(out)

def choose_default_metric(df):
    p = profile_dataset(df)
    # Prefer numeric columns with common business-measure words; otherwise first numeric column.
    preferred = ["sales","revenue","amount","value","profit","cost","price","quantity","units","count","total","margin","rate","score"]
    for word in preferred:
        for col in p["numeric_columns"]:
            if word in col.lower():
                return col
    return p["numeric_columns"][0] if p["numeric_columns"] else None

def summarize_numeric(df, metric: Optional[str] = None):
    profile = profile_dataset(df)
    metrics = [metric] if metric in profile["numeric_columns"] else profile["numeric_columns"]
    if not metrics:
        return {"message":"No numeric columns were detected. Choose a date or categorical column, or upload a dataset with measurable values."}
    result = {}
    for col in metrics:
        s = pd.to_numeric(df[col], errors="coerce")
        result[col] = {
            "sum": round(float(s.sum()), 4) if s.notna().any() else None,
            "mean": round(float(s.mean()), 4) if s.notna().any() else None,
            "median": round(float(s.median()), 4) if s.notna().any() else None,
            "min": round(float(s.min()), 4) if s.notna().any() else None,
            "max": round(float(s.max()), 4) if s.notna().any() else None,
            "missing": int(s.isna().sum()),
        }
    return result

def grouped_analysis(df, dimension, metric, aggregation="sum", top_n=20):
    if dimension not in df.columns:
        raise ValueError(f"Unknown dimension: {dimension}")
    if metric not in df.columns:
        raise ValueError(f"Unknown metric: {metric}")
    if aggregation not in {"sum","mean","median","min","max","count"}:
        raise ValueError("Unsupported aggregation.")
    work = df[[dimension, metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    if aggregation == "count":
        result = work.groupby(dimension, dropna=False)[metric].count().reset_index(name="count")
        sort_col = "count"
    else:
        result = work.groupby(dimension, dropna=False)[metric].agg(aggregation).reset_index(name=f"{aggregation}_{metric}")
        sort_col = f"{aggregation}_{metric}"
    return result.sort_values(sort_col, ascending=False).head(int(top_n)).reset_index(drop=True)

def time_analysis(df, date_col, metric, aggregation="sum"):
    if date_col not in df.columns or metric not in df.columns:
        raise ValueError("Select existing date and metric columns.")
    if aggregation not in {"sum","mean","count"}:
        raise ValueError("Time aggregation must be sum, mean, or count.")
    work = df[[date_col,metric]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[date_col])
    work["Period"] = work[date_col].dt.to_period("M").astype(str)
    if aggregation == "count":
        return work.groupby("Period")[metric].count().reset_index(name="count").reset_index(drop=True)
    return work.groupby("Period")[metric].agg(aggregation).reset_index(name=f"{aggregation}_{metric}").reset_index(drop=True)

def question_to_plan(df, question):
    """Schema-aware offline planner: uses actual column names and simple language heuristics."""
    p = profile_dataset(df)
    q = question.lower()
    cols = list(df.columns)
    metric = None
    # Match names and tokens against actual schema.
    for col in sorted(cols, key=lambda x: len(str(x)), reverse=True):
        if str(col).lower() in q:
            if col in p["numeric_columns"]:
                metric = col
                break
    if metric is None:
        metric = choose_default_metric(df)
    dimension = None
    for col in sorted(cols, key=lambda x: len(str(x)), reverse=True):
        if col in p["categorical_columns"] and str(col).lower() in q:
            dimension = col
            break
    if dimension is None:
        for col in p["categorical_columns"]:
            values = [str(v).lower() for v in df[col].dropna().astype(str).unique()[:100]]
            if any(v and v in q for v in values):
                dimension = col
                break
    date_col = next((c for c in p["date_columns"] if str(c).lower() in q), None)
    if date_col is None and any(w in q for w in ["monthly","month","trend","over time","time trend"]):
        date_col = p["date_columns"][0] if p["date_columns"] else None
    if date_col and metric and any(w in q for w in ["monthly","month","trend","over time","time trend"]):
        return {"kind":"time","date":date_col,"metric":metric,"aggregation":"mean" if "average" in q else "sum"}
    if dimension and metric:
        aggregation = "mean" if any(w in q for w in ["average","mean"]) else "sum"
        return {"kind":"group","dimension":dimension,"metric":metric,"aggregation":aggregation}
    if metric:
        return {"kind":"summary","metric":metric}
    return {"kind":"profile"}

def answer_question(df, question):
    plan = question_to_plan(df, question)
    if plan["kind"] == "time":
        data = time_analysis(df, plan["date"], plan["metric"], plan["aggregation"])
        return {"summary":f"Time analysis of '{plan['metric']}' using date column '{plan['date']}'.","data":data,"plan":plan,"mode":"offline"}
    if plan["kind"] == "group":
        data = grouped_analysis(df, plan["dimension"], plan["metric"], plan["aggregation"])
        return {"summary":f"Grouped '{plan['metric']}' by '{plan['dimension']}' using {plan['aggregation']}.","data":data,"plan":plan,"mode":"offline"}
    if plan["kind"] == "summary":
        return {"summary":f"Summary statistics for '{plan['metric']}'.","data":summarize_numeric(df,plan["metric"]),"plan":plan,"mode":"offline"}
    return {"summary":"I could not infer a metric from the question. Review the detected columns or select fields in Explore Data.","data":profile_dataset(df),"plan":plan,"mode":"offline"}
