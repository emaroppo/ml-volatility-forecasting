import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime
from data.db.DBManager import DBManager
from data.processing.HARDailyVolatilityPipeline import HARDailyVolatilityPipeline
from data.processing.UnivariateDailyVolatilityPipeline import (
    UnivariateDailyVolatilityPipeline,
)


def dataset_data():
    st.title("Processed Dataset Visualization")

    # Initialize session state for dataframe and sequence index if they don't exist
    if "dataset_df" not in st.session_state:
        st.session_state.dataset_df = pd.DataFrame()
    if "current_sequence" not in st.session_state:
        st.session_state.current_sequence = 0

    db_manager = DBManager(db_path="data/db/tickers.db")
    tickers = db_manager.get_all_tickers()
    selected_ticker = st.selectbox("Select a Ticker", tickers)

    # set date as log_returns index

    start_date = st.date_input("Start Date", datetime(2020, 1, 1), key="dataset_start")
    end_date = st.date_input("End Date", datetime.now(), key="dataset_end")
    start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
    end_timestamp = int(datetime.combine(end_date, datetime.min.time()).timestamp())

    if st.button("Load Processed Data"):
        st.session_state.pressed_load_data = True
        stock_data = db_manager.get_daily_stock_data(
            selected_ticker,
            start=start_timestamp,
            end=end_timestamp,
        )
        stock_data = stock_data.sort_values(by="DATE")
        # compute log returns
        stock_data["log_return"] = (
            stock_data["ADJ_CLOSE"] / stock_data["ADJ_CLOSE"].shift(1)
        ).apply(lambda x: np.log(x))
        stock_data = stock_data.dropna(subset=["log_return"])
        log_returns = stock_data["log_return"]
        # convert DATE to timestamp
        log_returns.index = stock_data["DATE"]

        har_pipeline = UnivariateDailyVolatilityPipeline()

        processed_data = har_pipeline.process_ticker(log_returns=log_returns)
        X, y = har_pipeline.create_sequences(processed_data, seq_length=22)

        # Combine X and y into a single DataFrame for easier visualization
        combined_data = np.hstack((X.reshape(X.shape[0], X.shape[1]), y.reshape(-1, 1)))
        feature_cols = [f"day_{i+1}" for i in range(X.shape[1])] + ["TARGET"]
        combined_df = pd.DataFrame(combined_data, columns=feature_cols)
        combined_df["DATE"] = stock_data["DATE"].iloc[-len(combined_df) :].values

        st.session_state.dataset_df = combined_df
        # Reset sequence index
        st.session_state.current_sequence = 0

    # Only show visualization options if data is loaded
    if not st.session_state.dataset_df.empty:
        st.write(
            f"Displaying processed data for {selected_ticker} from {start_date} to {end_date}"
        )

        # Calculate total number of sequences (assuming each row is a sequence)
        total_sequences = len(st.session_state.dataset_df)

        if total_sequences > 0:
            # Create navigation controls
            col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])

            with col1:
                if st.button("⏮️ First"):
                    st.session_state.current_sequence = 0

            with col2:
                if st.button("⬅️ Prev") and st.session_state.current_sequence > 0:
                    st.session_state.current_sequence -= 1

            with col3:
                # Display current sequence number and total
                st.write(
                    f"Sequence {st.session_state.current_sequence + 1} of {total_sequences}"
                )

                # Add a slider for quickly jumping to sequences
                new_sequence = (
                    st.slider(
                        "Jump to sequence",
                        min_value=1,
                        max_value=total_sequences,
                        value=st.session_state.current_sequence + 1,
                    )
                    - 1
                )
                if new_sequence != st.session_state.current_sequence:
                    st.session_state.current_sequence = new_sequence

            with col4:
                if (
                    st.button("➡️ Next")
                    and st.session_state.current_sequence < total_sequences - 1
                ):
                    st.session_state.current_sequence += 1

            with col5:
                if st.button("⏭️ Last"):
                    st.session_state.current_sequence = total_sequences - 1

            # Get the current sequence
            current_idx = st.session_state.current_sequence

            # Display the date for the current sequence
            current_date = st.session_state.dataset_df.iloc[current_idx]["DATE"]
            st.write(f"Date: {current_date}")

            # Get all features (excluding DATE and TARGET)
            feature_cols = [
                col
                for col in st.session_state.dataset_df.columns
                if col not in ["DATE", "TARGET"]
            ]

            # Create a line chart showing the feature values as a time series
            # First, we need to transform the data to a suitable format
            feature_values = st.session_state.dataset_df.iloc[current_idx][
                feature_cols
            ].to_dict()

            # Create an index for x-axis (1 to 22 for each position in sequence)
            sequence_positions = list(range(1, len(feature_values) + 1))

            # Create a DataFrame for plotting
            line_df = pd.DataFrame(
                {"Position": sequence_positions, "Value": list(feature_values.values())}
            )

            # Create the line chart
            line_fig = px.line(
                line_df,
                x="Position",
                y="Value",
                title=f"Sequence {current_idx + 1} Feature Values as Time Series",
                markers=True,
                line_shape="linear",
            )

            # Customize the line chart
            line_fig.update_layout(
                xaxis_title="Position in Sequence",
                yaxis_title="Feature Value",
                xaxis=dict(tickmode="linear", dtick=1),  # Show all position numbers
                height=400,
                hovermode="x unified",
            )

            # Add the target value as a horizontal line
            target_value = st.session_state.dataset_df.iloc[current_idx]["TARGET"]
            line_fig.add_hline(
                y=target_value,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Target: {target_value:.6f}",
                annotation_position="top right",
            )

            st.plotly_chart(line_fig, use_container_width=True)

            # Show summary stats for this sequence
            st.write("### Sequence Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Target Value", f"{target_value:.6f}")
            with col2:
                st.metric("Mean", f"{np.mean(list(feature_values.values())):.6f}")
            with col3:
                st.metric("Max", f"{np.max(list(feature_values.values())):.6f}")

            # Show data for the current sequence
            st.write("### Current Sequence Data")

            # Format the current row data as a transposed DataFrame for better viewing
            row_data = (
                st.session_state.dataset_df.iloc[current_idx].drop("DATE").to_dict()
            )
            row_df = pd.DataFrame(
                {"Feature": list(row_data.keys()), "Value": list(row_data.values())}
            )

            # Custom function to format the values
            def format_value(val):
                if isinstance(val, (int, float)):
                    return f"{val:.6f}"
                return val

            # Apply formatting
            row_df["Value"] = row_df["Value"].apply(format_value)

            # Display the formatted data
            st.dataframe(row_df, use_container_width=True)

        # Show the full dataset in a collapsible section
        with st.expander("View Full Dataset"):
            st.dataframe(st.session_state.dataset_df)
