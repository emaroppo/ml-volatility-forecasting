import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
from typing import List, Optional, Tuple
from datetime import datetime
from data.db.DBManager import DBManager
import numpy as np
from data.processing.HARDailyVolatilityPipeline import HARDailyVolatilityPipeline
from data.processing.UnivariateDailyVolatilityPipeline import (
    UnivariateDailyVolatilityPipeline,
)


def highlight_special_rows(df):
    # Create a DataFrame with the same shape as the input DataFrame
    styled = pd.DataFrame("", index=df.index, columns=df.columns)

    # Highlight rows with splits
    if "SPLIT" in df.columns:
        split_mask = df["SPLIT"] > 1
        styled.loc[split_mask, :] = "background-color: #ffcccc; color:red"  # Light red

    # Highlight rows with dividends (and override if there's both)
    if "DIVIDEND" in df.columns:
        dividend_mask = df["DIVIDEND"] > 0
        styled.loc[dividend_mask, :] = (
            "background-color: #ccffcc; color: black"  # Light green
        )

    # If a row has both split and dividend, use a different color
    if "SPLIT" in df.columns and "DIVIDEND" in df.columns:
        both_mask = (df["SPLIT"] > 1) & (df["DIVIDEND"] > 0)
        styled.loc[both_mask, :] = (
            "background-color: #ffffcc; color:black"  # Light yellow
        )

    return styled


def stock_data():
    st.title("Stock Data Visualization")

    # Initialize session state for dataframe and selected columns if they don't exist
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()
    if "selected_columns" not in st.session_state:
        st.session_state.selected_columns = []

    db_manager = DBManager(db_path="data/db/tickers.db")
    tickers = db_manager.get_all_tickers()
    selected_ticker = st.selectbox("Select a Ticker", tickers)

    start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.date_input("End Date", datetime.now())
    start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
    end_timestamp = int(datetime.combine(end_date, datetime.min.time()).timestamp())

    if st.button("Load Data"):

        st.session_state.pressed_load_data = True
        df = db_manager.get_daily_stock_data(
            selected_ticker,
            start=start_timestamp,
            end=end_timestamp,
        )
        # convert DATE to date only
        df["DATE"] = df["DATE"].dt.date
        # drop ticker column if it exists
        if "TICKER" in df.columns:
            df = df.drop(columns=["TICKER"])

        # add log returns column
        df["LOG_RETURN"] = (df["ADJ_CLOSE"] / df["ADJ_CLOSE"].shift(1)).apply(
            lambda x: np.log(x) if x > 0 else 0
        )

        st.session_state.df = df
        # Reset selected columns when loading new data
        st.session_state.selected_columns = ["CLOSE", "ADJ_CLOSE"]

    # Only show visualization options if data is loaded
    if not st.session_state.df.empty:
        st.write(
            f"Displaying data for {selected_ticker} from {start_date} to {end_date}"
        )

        # Get columns from the stored dataframe
        columns = st.session_state.df.columns.tolist()

        # Use the session state directly for multiselect
        selected_cols = st.multiselect(
            "Select columns to plot",
            columns,
            default=st.session_state.selected_columns,
        )

        # Update session state with current selection
        st.session_state.selected_columns = selected_cols

        # Plot the chart if columns are selected
        if selected_cols:
            fig = px.line(
                st.session_state.df,
                x="DATE",
                y=selected_cols,
                title=f"{selected_ticker} Stock Data",
            )

            # Add markers for splits and dividends
            if (
                "SPLIT" in st.session_state.df.columns
                and "DIVIDEND" in st.session_state.df.columns
            ):
                # Add markers for splits
                split_days = st.session_state.df[st.session_state.df["SPLIT"] > 1]
                if not split_days.empty:
                    for col in selected_cols:
                        fig.add_scatter(
                            x=split_days["DATE"],
                            y=split_days[col],
                            mode="markers",
                            marker=dict(symbol="star", size=12, color="red"),
                            name=f"Split ({col})",
                            showlegend=col
                            == selected_cols[0],  # Only show once in legend
                        )

                # Add markers for dividends
                dividend_days = st.session_state.df[st.session_state.df["DIVIDEND"] > 0]
                if not dividend_days.empty:
                    for col in selected_cols:
                        fig.add_scatter(
                            x=dividend_days["DATE"],
                            y=dividend_days[col],
                            mode="markers",
                            marker=dict(symbol="diamond", size=8, color="green"),
                            name=f"Dividend ({col})",
                            showlegend=col
                            == selected_cols[0],  # Only show once in legend
                        )

            st.plotly_chart(fig)

        # Show data table with highlighted rows for splits and dividends
        # Display the styled dataframe
        st.write("### Values")

        st.dataframe(st.session_state.df.style.apply(highlight_special_rows, axis=None))
        st.write("🟢 Dividend 🔴 Split 🟡 Both")

    elif "pressed_load_data" in st.session_state and st.session_state.pressed_load_data:
        st.write("No data available for the selected ticker and date range.")


def dataset_data():
    st.title("Processed Dataset Visualization")

    # Initialize session state for dataframe and sequence index if they don't exist
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()
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

        processed_data, validation_data = har_pipeline.process_ticker(
            log_returns=log_returns
        )
        processed_data["inputs"] = processed_data["inputs"].reshape(
            processed_data["inputs"].shape[0], -1
        )  # (N, 22, 1) -> (N, 22)
        validation_data["inputs"] = validation_data["inputs"].reshape(
            validation_data["inputs"].shape[0], -1
        )  # (M, 22, 1) -> (M, 22)
        # combine training and validation data for visualization
        print(processed_data["inputs"].shape, processed_data["targets"].shape)
        processed_data = {
            "inputs": np.concatenate(
                (processed_data["inputs"], validation_data["inputs"])
            ),
            "targets": np.concatenate(
                (processed_data["targets"], validation_data["targets"])
            ),
        }

        inputs_df = pd.DataFrame(processed_data["inputs"])
        targets_df = pd.DataFrame(processed_data["targets"], columns=["TARGET"])

        combined_df = pd.concat([inputs_df, targets_df], axis=1)
        combined_df["DATE"] = stock_data["DATE"].iloc[-len(combined_df) :].values

        st.session_state.df = combined_df
        # Reset sequence index
        st.session_state.current_sequence = 0

    # Only show visualization options if data is loaded
    if not st.session_state.df.empty:
        st.write(
            f"Displaying processed data for {selected_ticker} from {start_date} to {end_date}"
        )

        # Calculate total number of sequences (assuming each row is a sequence)
        total_sequences = len(st.session_state.df)

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
            current_date = st.session_state.df.iloc[current_idx]["DATE"]
            st.write(f"Date: {current_date}")

            # Get all features (excluding DATE and TARGET)
            feature_cols = [
                col
                for col in st.session_state.df.columns
                if col not in ["DATE", "TARGET"]
            ]

            # Create a line chart showing the feature values as a time series
            # First, we need to transform the data to a suitable format
            feature_values = st.session_state.df.iloc[current_idx][
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
            target_value = st.session_state.df.iloc[current_idx]["TARGET"]
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
            row_data = st.session_state.df.iloc[current_idx].drop("DATE").to_dict()
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
            st.dataframe(st.session_state.df)


def main():
    # navigation component
    st.sidebar.title("Navigation")
    options = ["Stock Data", "Processed Data", "Models Performance"]
    choice = st.sidebar.selectbox("Go to", options)
    if choice == "Stock Data":
        stock_data()
    elif choice == "Processed Data":
        dataset_data()


main()
