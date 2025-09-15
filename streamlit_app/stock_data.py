import numpy as np
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px

from data.db.DBManager import DBManager


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
    if "stock_df" not in st.session_state:
        st.session_state.stock_df = pd.DataFrame()
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

        st.session_state.stock_df = df
        # Reset selected columns when loading new data
        st.session_state.selected_columns = ["CLOSE", "ADJ_CLOSE"]

    # Only show visualization options if data is loaded
    if not st.session_state.stock_df.empty:
        st.write(
            f"Displaying data for {selected_ticker} from {start_date} to {end_date}"
        )

        # Get columns from the stored dataframe
        columns = st.session_state.stock_df.columns.tolist()

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
                st.session_state.stock_df,
                x="DATE",
                y=selected_cols,
                title=f"{selected_ticker} Stock Data",
            )

            # Add markers for splits and dividends
            if (
                "SPLIT" in st.session_state.stock_df.columns
                and "DIVIDEND" in st.session_state.stock_df.columns
            ):
                # Add markers for splits
                split_days = st.session_state.stock_df[
                    st.session_state.stock_df["SPLIT"] > 1
                ]
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
                dividend_days = st.session_state.stock_df[
                    st.session_state.stock_df["DIVIDEND"] > 0
                ]
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

        st.dataframe(
            st.session_state.stock_df.style.apply(highlight_special_rows, axis=None)
        )
        st.write("🟢 Dividend 🔴 Split 🟡 Both")

    elif "pressed_load_data" in st.session_state and st.session_state.pressed_load_data:
        st.write("No data available for the selected ticker and date range.")
