import streamlit as st
import pandas as pd
import numpy as np
import traceback

from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier
)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)

from sklearn.metrics import (
    r2_score,
    accuracy_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def detect_problem_type(target):
    if target.dtype == "object":
        return "classification"

    if target.nunique() <= 15:
        return "classification"

    return "regression"


def remove_outliers_iqr(df):
    df_clean = df.copy()

    numeric_cols = df_clean.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df_clean = df_clean[
            (df_clean[col] >= lower) & (df_clean[col] <= upper)
        ]

    return df_clean


def train_best_model(df, target_column):
    # Separate features and target
    X = df.drop(columns=[target_column])
    

    # Remove ID-like columns
    id_cols = []

    for col in X.columns:
        col_lower = col.lower()

        if (
            "id" in col_lower
            or "txn" in col_lower
            or "transaction" in col_lower
            or "invoice" in col_lower
            or "order" in col_lower
        ):
            id_cols.append(col)

    X = X.drop(columns=id_cols, errors="ignore")
    y = df[target_column]

    problem_type = detect_problem_type(y)
    
    # Validate target column
    if problem_type == "regression" and not pd.api.types.is_numeric_dtype(y):
        try:
            y = pd.to_numeric(y, errors='coerce')
            if y.isna().any():
                raise ValueError("Target column contains non-numeric values that cannot be converted.")
        except:
            raise ValueError("Target column must be numeric for regression problems.")
    
    # Encode target column for classification
    if problem_type == "classification" and y.dtype == "object":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)

    # Fix Pandas StringDtype issue
    for col in X.columns:
        if "string" in str(X[col].dtype).lower():
            X[col] = X[col].astype(object)

    # Convert all string dtypes to object
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype(str)

    # Detect all non-numeric columns
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (   
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols
            )
        ],
        remainder="passthrough"
    )   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=42
    )

    if problem_type == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42),
            "Random Forest": RandomForestRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=5, n_estimators=50, random_state=42)
        }
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, C=0.1, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42),
            "Random Forest": RandomForestClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, n_estimators=50, random_state=42)
        }

    best_model = None
    best_model_name = None
    best_score = -999

    for name, model in models.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        if problem_type == "regression":
            score = r2_score(y_test, preds)
        else:
            score = accuracy_score(y_test, preds)

        if score > best_score:
            best_score = score
            best_model = pipe
            best_model_name = name

    # Extract feature importance/coefficients
    feature_importance = {}
    try:
        model_step = best_model.named_steps['model']
        prep_step = best_model.named_steps['prep']
        feature_names = prep_step.get_feature_names_out()
        
        if hasattr(model_step, 'feature_importances_'):
            importances = model_step.feature_importances_
        elif hasattr(model_step, 'coef_'):
            coefs = model_step.coef_
            importances = np.abs(coefs[0]) if len(coefs.shape) > 1 else np.abs(coefs)
        else:
            importances = np.ones(len(feature_names))

        for i, feat_name in enumerate(feature_names):
            if feat_name.startswith('cat__'):
                orig_col = feat_name[5:].rsplit('_', 1)[0]
            elif feat_name.startswith('remainder__'):
                orig_col = feat_name[11:]
            else:
                orig_col = feat_name
            
            if not orig_col or orig_col == 'remainder':
                continue
                
            if orig_col not in feature_importance:
                feature_importance[orig_col] = 0
            feature_importance[orig_col] += abs(importances[i])
            
    except Exception as e:
        pass
    
    # Fallback
    if not feature_importance:
        for col in X.columns:
            feature_importance[col] = 1.0
    
    return (
        best_model,
        best_model_name,
        best_score,
        X,  # Return the base features dataframe for building full input profiles later
        problem_type,
        feature_importance
    )


def show_ai_prediction_center(df):
    st.title("🔮 AI Prediction Center")
    st.write("Explore different scenarios and predict possible outcomes.")

    # Busy-state to show loader while ML/prediction runs
    if "ai_busy" not in st.session_state:
        st.session_state["ai_busy"] = False

    # Auto detect Dates/Timestamps to show helpful banner
    date_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        elif df[col].dtype == "object":
            try:
                pd.to_datetime(df[col])
                date_cols.append(col)
            except:
                pass

    if date_cols:
        st.info(f"📅 Date/Time columns detected: {', '.join(date_cols)}. Make future predictions by entering dates ahead of today!")

    # Outliers setup
    option = st.radio(
        "Choose Data Preparation Mode",
        ["Keep All Records", "Remove Unusual Records"]
    )

    working_df = df.copy()
    if option == "Remove Unusual Records":
        before = len(working_df)
        working_df = remove_outliers_iqr(working_df)
        st.info(f"{before - len(working_df)} unusual records removed.")
    else:
        st.info("All records will be used.")

    st.metric("Records Available", len(working_df))
    st.warning("⚠️ Target column will NOT be used for training. Only other features will be used to make predictions.")

    target_column = st.selectbox(
        "What would you like to predict?",
        working_df.columns
    )

    # ORIGINAL BUTTON NAME 1
    prepare_disabled = st.session_state["ai_busy"]
    if st.button("Prepare AI Prediction Engine", disabled=prepare_disabled):
        st.session_state["ai_busy"] = True
        try:
            with st.spinner("Training AI Prediction Engine (this may take a while)..."):
                (
                    model,
                    model_name,
                    score,
                    training_df,
                    problem_type,
                    feature_importance
                ) = train_best_model(working_df, target_column)

            # Store references in state
            st.session_state["model"] = model
            st.session_state["training_df"] = training_df
            st.session_state["target"] = target_column
            st.session_state["score"] = score
            st.session_state["model_name"] = model_name
            st.session_state["problem_type"] = problem_type

            # Sort features
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            st.session_state["sorted_features"] = sorted_features

            # Filter top 5 features
            top_5 = [f[0] for f in sorted_features if f[0] in training_df.columns][:5]
            if not top_5:
                top_5 = list(training_df.columns)[:5]
            st.session_state["top_features"] = top_5

            st.success("AI Prediction Engine Ready")
        except Exception as e:
            st.error(str(e))
            st.code(traceback.format_exc())
        finally:
            st.session_state["ai_busy"] = False

    if st.session_state["ai_busy"]:
        st.info("AI is running... please wait.")


    # Interactive Simulator UI
    if "model" in st.session_state:
        st.divider()
        
        training_df = st.session_state["training_df"]
        top_features = st.session_state["top_features"]
        sorted_features = st.session_state["sorted_features"]

        st.subheader("🎛 Scenario Builder")
        st.info(
            """
            These are the 5 most important
            factors identified by the AI.

            Change them to explore
            future outcomes.
            """
        )

        # Show feature importance list and chart
        if "sorted_features" in st.session_state and st.session_state["sorted_features"]:
            st.subheader("📊 Top Factors Affecting Prediction")
            chart_data = [item for item in sorted_features if item[0] in training_df.columns][:5]
            
            for rank, (feat, importance) in enumerate(chart_data, start=1):
                st.write(f"{rank}. {feat}")
                
            importance_df = pd.DataFrame(chart_data, columns=["Feature", "Importance"])
            st.bar_chart(importance_df.set_index("Feature"))

        user_input_values = {}
        
        # Build widgets for Top 5 features
        for col in top_features:
            column_data = training_df[col]
            
            is_date_col = False
            if pd.api.types.is_datetime64_any_dtype(column_data):
                is_date_col = True
            elif column_data.dtype == "object":
                try:
                    pd.to_datetime(column_data)
                    is_date_col = True
                except:
                    pass

            if is_date_col:
                parsed_dates = pd.to_datetime(column_data)
                max_date = parsed_dates.max()
                default_date = max_date + pd.Timedelta(days=30)
                
                col1, col2 = st.columns(2)
                with col1:
                    selected_date = st.date_input(f"{col} (Date)", value=default_date.date())
                with col2:
                    selected_time = st.time_input(f"{col} (Time - Optional)", value=pd.Timestamp.now().time())
                
                # Combine into datetime timestamp object
                user_input_values[col] = pd.Timestamp.combine(selected_date, selected_time)

            elif pd.api.types.is_numeric_dtype(column_data):
                min_val = float(column_data.min())
                max_val = float(column_data.max())
                mean_val = float(column_data.mean())

                if (max_val - min_val) < 1000 and min_val != max_val:
                    user_input_values[col] = st.slider(col, min_val, max_val, mean_val)
                else:
                    user_input_values[col] = st.number_input(col, value=mean_val)
            else:
                available_options = column_data.dropna().unique().tolist()
                user_input_values[col] = st.selectbox(col, available_options)

        # ORIGINAL BUTTON NAME 2
        generate_disabled = st.session_state["ai_busy"]
        if st.button("Generate Prediction", disabled=generate_disabled):
            st.session_state["ai_busy"] = True
            try:
                with st.spinner("Generating prediction..."):
                    # Build full placeholder dictionary using training context defaults
                    full_row_data = {}
                    for col in training_df.columns:
                        if pd.api.types.is_numeric_dtype(training_df[col]):
                            full_row_data[col] = training_df[col].median()
                        else:
                            mode_series = training_df[col].mode()
                            full_row_data[col] = mode_series.iloc[0] if not mode_series.empty else np.nan

                    # Overwrite the top 5 values with custom inputs from the UI fields
                    for col, value in user_input_values.items():
                        full_row_data[col] = value

                    # Put into a structured dataframe matching original training columns
                    input_df = pd.DataFrame([full_row_data], columns=training_df.columns)

                    # Type checking validation alignment for Date instances
                    for col in input_df.columns:
                        if pd.api.types.is_datetime64_any_dtype(training_df[col]):
                            input_df[col] = pd.to_datetime(input_df[col])

                    # Run full array prediction safely
                    prediction = st.session_state["model"].predict(input_df)[0]

                st.markdown("## 🎯 AI Prediction Result")

                # Clean output parsing matching formatting rules
                if isinstance(prediction, (int, float, np.number)):
                    if "price" in st.session_state["target"].lower() or "₹" in st.session_state["target"]:
                        st.success(f"Predicted {st.session_state['target']} = ₹{prediction:,.2f}")
                    else:
                        st.success(f"Predicted {st.session_state['target']} = {prediction:,.2f}")
                else:
                    st.success(f"Predicted {st.session_state['target']} = {prediction}")
            finally:
                st.session_state["ai_busy"] = False


        # Advanced statistics section hidden away nicely
        with st.expander("Advanced Details"):
            st.write("Model Used:", st.session_state["model_name"])
            st.write("Validation Score:", f"{st.session_state['score']:.4f}")
            st.write("Problem Type:", st.session_state["problem_type"])
            st.write("Rows Used:", len(working_df))
            
            complete_importance_df = pd.DataFrame(sorted_features, columns=["Feature", "Impact Weight"])
            st.write("Top Influential Features Table")
            st.dataframe(complete_importance_df)