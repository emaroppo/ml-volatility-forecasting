import streamlit as st
from streamlit_app.stock_data import stock_data
from streamlit_app.dataset_data import dataset_data


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
