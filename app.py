import streamlit as st
import pandas as pd
import numpy as np
from ml import show_ai_prediction_center

from loader import load_data
from stats import clean_data, basic_stats, dataset_info

from eda import (
    get_heatmap_figure,
    detect_outliers_iqr,
    missing_value_summary,
    unique_value_summary,
    data_quality_report,
    distribution_summary,
    categorical_summary
)

from deep_dive import deep_dive_analysis

st.title("Data Viewer App")

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None

if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None

if 'uploaded_file_key' not in st.session_state:
    st.session_state.uploaded_file_key = None

if 'show_heatmap' not in st.session_state:
    st.session_state.show_heatmap = False

if 'show_outliers' not in st.session_state:
    st.session_state.show_outliers = False

if 'show_deep_dive' not in st.session_state:
    st.session_state.show_deep_dive = False

page = st.sidebar.selectbox(
    "Select a page",
    [
        "Basic Cleaning",
        "Charts and Visualizations",
        "EDA",
        "Deep Dive Analysis",
        "AI Prediction Center"
    ]
)

# File uploader
data = st.sidebar.file_uploader(
    "Upload your data",
    type=["csv", "xlsx", "xls"]
)

if data is not None:

    current_file_key = (
        data.name,
        data.size
    )

    if st.session_state.uploaded_file_key != current_file_key:

        st.session_state.df = load_data(data)
        st.session_state.df_cleaned = None
        st.session_state.uploaded_file_key = current_file_key


# ---------------------------------------------------
# BASIC CLEANING
# ---------------------------------------------------

if page == "Basic Cleaning":

    if st.session_state.df is not None:

        st.dataframe(
            st.session_state.df.head(10)
        )

        st.write("Dataset Information:")
        dataset_info(st.session_state.df)

        if st.button(
            "Clean data",
            key="clean_data_btn"
        ):
            st.session_state.df_cleaned = clean_data(
                st.session_state.df
            )

    if st.session_state.df_cleaned is not None:

        st.write("Cleaned Data:")

        st.dataframe(
            st.session_state.df_cleaned
        )

        st.write("Basic Statistics:")

        st.dataframe(
            basic_stats(
                st.session_state.df_cleaned
            )
        )


# ---------------------------------------------------
# CHARTS
# ---------------------------------------------------

elif page == "Charts and Visualizations":

    if st.session_state.df is not None:

        from plots import create_visualization

        create_visualization(
            st.session_state.df
        )

    else:

        st.write(
            "Please upload data first on the Basic Analysis page"
        )


# ---------------------------------------------------
# EDA
# ---------------------------------------------------

elif page == "EDA":

    st.title("Exploratory Data Analysis (EDA)")

    st.write(
        "This page helps you understand data quality, distributions, relationships and outliers in your dataset."
    )

    if st.session_state.df is not None:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
         "Data Quality",
        "Missing Values",
        "Unique Values",
        "Distribution",
        "Categorical",
        "Heatmap",
        "Outliers"
        ])

        # Data Quality Report
        with tab1:

            st.subheader("Data Quality Report")

            st.caption(
                 "Shows data type, missing values, duplicates and unique values."
                )

            quality_df = data_quality_report(
            st.session_state.df
                )

            st.dataframe(
                quality_df,
                use_container_width=True
                )
            
        # Missing Values Analysis
        with tab2:

            st.subheader("Missing Values Analysis")

            st.caption(
                "Identifies missing records and their percentage."
                 )

            missing_df = missing_value_summary(
                st.session_state.df
                 )

            st.dataframe(
                 missing_df,
                 use_container_width=True
                 )

        # Unique Values Analysis
        with tab3:

            st.subheader("Unique Values Analysis")

            st.caption(
                 "Shows distinct values in each column."
                 )

            unique_df = unique_value_summary(
                 st.session_state.df
                 )

            st.dataframe(
                 unique_df,
                 use_container_width=True
                 )

        # Distribution Summary
        with tab4:

            st.subheader("Distribution Summary")

            st.caption(
                "Shows skewness and kurtosis of numerical variables."
                 )

            dist_df = distribution_summary(
                st.session_state.df
                 )

            if len(dist_df) > 0:

                 st.dataframe(
                    dist_df,
                    use_container_width=True
                     )

            else:

                st.info(
                    "No valid numeric columns available."
                     )

        # Categorical Summary
        with tab5:

             st.subheader("Categorical Data Summary")

             st.caption(
                 "Shows frequency distribution of categorical columns."
                 )

             cat_summary = categorical_summary(
                st.session_state.df
                 )

             if len(cat_summary) > 0:

                for col, table in cat_summary.items():

                    st.write(f"### {col}")

                    st.dataframe(
                        table,
                        use_container_width=True
                     )

                    st.bar_chart(
                         table.set_index(col)
                     )

             else:

                st.info(
                    "No categorical columns found."
                     )

        # Heatmap
        with tab6:

            st.subheader("Correlation Heatmap")

            st.caption(
                 "Visualizes relationships between numerical variables."
                 )
            
            fig, error = get_heatmap_figure(
                 st.session_state.df
            )

            if error:

                 st.error(error)

            else:

                 st.plotly_chart(
                    fig,
                    use_container_width=True
             )

        # Outlier Detection
        with tab7:

            st.subheader(
                "Outlier Detection (IQR Method)"
            )

            st.caption(
                 "Detects unusual values using the Interquartile Range method."
                 )

            
            outliers = detect_outliers_iqr(
                st.session_state.df
            )

            outlier_summary = []

            for col, info in outliers.items():

                outlier_summary.append({
                    "Column": col,
                    "Outlier Count": info["count"],
                    "Lower Bound": round(
                        info["lower_bound"],
                        2
                    ),
                    "Upper Bound": round(
                        info["upper_bound"],
                        2
                    )
                })

            summary_df = pd.DataFrame(
                outlier_summary
            )

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )
        
                  
# ---------------------------------------------------
# DEEP DIVE ANALYSIS
# ---------------------------------------------------

elif page == "Deep Dive Analysis":

    st.title("Deep Dive Analysis")

    st.write(
        "In this page you can select a particular column from the dataset and perform an in-depth analysis of that column."
    )

    if st.session_state.df is not None:

        deep_dive_analysis(
            st.session_state.df
        )

    else:

        st.write(
            "Please upload data first on the Basic Analysis page"
        )

elif page == "AI Prediction Center":

    if st.session_state.df_cleaned is None:

        st.warning(
            "Please run Basic Cleaning first."
        )

    else:

        show_ai_prediction_center(st.session_state.df_cleaned)