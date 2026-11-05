
import os
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

# Custom design
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
h1 {
    color: #183153;
}
h2, h3 {
    color: #24527a;
}
[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Retail Sales Forecasting System")
st.write("Machine Learning Based Sales Prediction Dashboard")
st.divider()

# Dataset location
BASE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
FILE_PATH = os.path.join(BASE_FOLDER, "sales_predictions.csv")
MODEL_PATH = os.path.join(BASE_FOLDER, "sales_model.pkl")

if os.path.exists(FILE_PATH):

    data = pd.read_csv(FILE_PATH)

    required_columns = ["Actual_Sales", "Predicted_Sales"]

    if not all(col in data.columns for col in required_columns):
        st.error("The prediction file does not contain the required columns.")
        st.stop()

    data = data.dropna(subset=required_columns)

    st.success("Sales prediction data loaded successfully!")

    # Sidebar
    st.sidebar.header("Dashboard Filters")

    max_rows = min(500, len(data))
    number_of_rows = st.sidebar.slider(
        "Records to display",
        min_value=10 if max_rows >= 10 else 1,
        max_value=max_rows,
        value=min(50, max_rows)
    )

    chart_data = data.head(number_of_rows)

    # KPI cards
    st.subheader("Sales Prediction Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        f"{len(data):,}"
    )

    col2.metric(
        "Average Actual Sales",
        f"{data['Actual_Sales'].mean():,.2f}"
    )

    col3.metric(
        "Average Predicted Sales",
        f"{data['Predicted_Sales'].mean():,.2f}"
    )

    st.divider()

    # Charts
    st.subheader("Actual vs Predicted Sales")

    tab1, tab2 = st.tabs(["Sales Comparison", "Prediction Accuracy"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            range(len(chart_data)),
            chart_data["Actual_Sales"],
            label="Actual Sales",
            marker="o",
            markersize=3
        )

        ax.plot(
            range(len(chart_data)),
            chart_data["Predicted_Sales"],
            label="Predicted Sales",
            marker="x",
            markersize=3
        )

        ax.set_xlabel("Record Number")
        ax.set_ylabel("Weekly Sales")
        ax.set_title("Actual vs Predicted Sales")
        ax.legend()
        ax.grid(alpha=0.25)

        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(
            data["Actual_Sales"],
            data["Predicted_Sales"],
            alpha=0.4
        )

        minimum = min(
            data["Actual_Sales"].min(),
            data["Predicted_Sales"].min()
        )

        maximum = max(
            data["Actual_Sales"].max(),
            data["Predicted_Sales"].max()
        )

        ax.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--",
            label="Perfect Prediction"
        )

        ax.set_xlabel("Actual Sales")
        ax.set_ylabel("Predicted Sales")
        ax.set_title("Prediction Accuracy")
        ax.legend()
        ax.grid(alpha=0.25)

        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # Results table
    st.subheader("Sales Prediction Details")
    st.dataframe(
        chart_data,
        use_container_width=True,
        hide_index=True
    )

    # Download button
    csv_data = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download All Predictions",
        data=csv_data,
        file_name="sales_predictions.csv",
        mime="text/csv"
    )

    st.caption(
        "Retail Sales Forecasting | Powered by Python, "
        "Pandas, Scikit-learn and Streamlit"
    )

else:
    st.error("Prediction file not found.")
    st.write("Please run your backend file first:")
   # Interactive Sales Prediction
st.divider()
st.header("🔮 Predict Sales for a New Record")

MODEL_PATH = os.path.join(BASE_FOLDER, "sales_model.pkl")

if os.path.exists(MODEL_PATH):
    try:
        saved_data = joblib.load(MODEL_PATH)
        model = saved_data["model"]
        feature_columns = saved_data["feature_columns"]
        default_values = saved_data["default_values"]

        st.write("Enter the details below to estimate weekly sales.")

        with st.form("sales_prediction_form"):
            input_values = {}

            cols = st.columns(2)

            for i, feature in enumerate(feature_columns):
                default = default_values.get(feature, 0)

                with cols[i % 2]:
                    if feature == "year":
                        input_values[feature] = st.number_input(
                            "Year", value=int(default), step=1
                        )
                    elif feature == "month":
                        input_values[feature] = st.number_input(
                            "Month", min_value=1, max_value=12,
                            value=max(1, min(12, int(default))), step=1
                        )
                    elif feature == "day":
                        input_values[feature] = st.number_input(
                            "Day", min_value=1, max_value=31,
                            value=max(1, min(31, int(default))), step=1
                        )
                    elif feature == "day_of_week":
                        input_values[feature] = st.number_input(
                            "Day of week (0=Monday, 6=Sunday)",
                            min_value=0, max_value=6,
                            value=max(0, min(6, int(default))), step=1
                        )
                    else:
                        input_values[feature] = st.number_input(
                            feature.replace("_", " ").title(),
                            value=float(default),
                            format="%.2f"
                        )

            submitted = st.form_submit_button("Predict Sales")

        if submitted:
            input_df = pd.DataFrame(
                [input_values],
                columns=feature_columns
            )
            prediction = model.predict(input_df)[0]

            st.success(f"Estimated Weekly Sales: {prediction:,.2f}")

    except Exception as e:
        st.error(f"Could not load or use the model: {e}")
else:
    st.warning(f"Model file not found: {MODEL_PATH}")