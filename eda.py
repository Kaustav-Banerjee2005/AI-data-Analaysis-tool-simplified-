import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


@st.cache_data
def get_heatmap_figure(df):

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    # Remove ID-like columns
    numeric_cols = [
        col for col in numeric_cols
        if not (
            "id" in col.lower()
            or "sr" in col.lower()
            or "serial" in col.lower()
        )
    ]

    if len(numeric_cols) < 2:
        return None, "Heatmap requires at least two valid numeric columns."

    corr_matrix = df[numeric_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )

    return fig, None


def detect_outliers_iqr(df):

    results = {}

    num_cols = df.select_dtypes(
        include="number"
    ).columns

    # Remove ID-like columns
    num_cols = [
        col for col in num_cols
        if not (
            "id" in col.lower()
            or "sr" in col.lower()
            or "serial" in col.lower()
        )
    ]

    for col in num_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[
            (df[col] < lower) |
            (df[col] > upper)
        ]

        results[col] = {
            "count": len(outliers),
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_values": outliers[col].tolist()
        }

    return results


def missing_value_summary(df):

    missing = df.isnull().sum()

    summary = pd.DataFrame({
        "Column": missing.index,
        "Missing Count": missing.values,
        "Missing %": np.round(
            (missing.values / len(df)) * 100,
            2
        )
    })

    return summary.sort_values(
        "Missing %",
        ascending=False
    )


def unique_value_summary(df):

    summary = pd.DataFrame({
        "Column": df.columns,
        "Unique Values": [
            df[col].nunique()
            for col in df.columns
        ]
    })

    return summary.sort_values(
        "Unique Values",
        ascending=False
    )


def data_quality_report(df):

    report = []

    for col in df.columns:

        report.append({
            "Column": col,
            "Data Type": str(df[col].dtype),
            "Missing Values": df[col].isnull().sum(),
            "Duplicates": df[col].duplicated().sum(),
            "Unique Values": df[col].nunique()
        })

    return pd.DataFrame(report)


def distribution_summary(df):

    numeric_cols = df.select_dtypes(
        include="number"
    )

    # Remove ID-like columns
    numeric_cols = numeric_cols.drop(
        columns=[
            col for col in numeric_cols.columns
            if (
                "id" in col.lower()
                or "sr" in col.lower()
                or "serial" in col.lower()
            )
        ],
        errors="ignore"
    )

    result = []

    for col in numeric_cols.columns:

        result.append({
            "Column": col,
            "Skewness": round(
                df[col].skew(),
                2
            ),
            "Kurtosis": round(
                df[col].kurtosis(),
                2
            )
        })

    return pd.DataFrame(result)


def categorical_summary(df):

    cat_cols = df.select_dtypes(
        include=["object"]
    ).columns

    summary = {}

    for col in cat_cols:

        temp = (
            df[col]
            .value_counts(dropna=False)
            .reset_index()
        )

        temp.columns = [
            col,
            "Count"
        ]

        summary[col] = temp

    return summary

