# main.py
import streamlit as st
import plotly.express as px
from src.portfolio import Portfolio
import pandas as pd


st.title("ETF Portfolio Analyzer")
    
# Initialize portfolio
portfolio = Portfolio()

tab1, tab2, tab3 = st.tabs(["Portfolio Management", "Metrics", "Data Explorer"])

with tab1:
    # Portfolio Management Section
    st.header("Portfolio Management")
    
    # Add new ETF
    col1, col2 = st.columns(2)
    with col1:
        new_etf = st.text_input("ETF Ticker", "").upper()
    with col2:
        allocation = st.number_input("Allocation (%)", min_value=0.0, max_value=100.0, value=0.0)
    
    if st.button("Add ETF"):
        if new_etf and allocation:
            if portfolio.add_holding(new_etf, allocation):
                st.success(f"Added {new_etf} to portfolio")
            else:
                st.error(f"Error adding {new_etf}")
    
    # Display current portfolio
    if portfolio.holdings:
        st.subheader("Current Portfolio")
        st.dataframe(portfolio.get_holdings_summary())

        # Delete ETF button
        etf_to_delete = st.selectbox("Select ETF to remove", list(portfolio.holdings.keys()))
        if st.button("Remove ETF"):
            portfolio.remove_holding(etf_to_delete)
            st.success(f"Removed {etf_to_delete} from portfolio")
            st.rerun()
        
        # Calculate and display metrics
        metrics, returns = portfolio.calculate_metrics()

with tab2:
    # Display current portfolio
    if portfolio.holdings:
        st.header("Current Portfolio")
        st.dataframe(portfolio.get_holdings_summary())

    if metrics:
        st.subheader("Portfolio Metrics")
        metrics_df = pd.DataFrame([metrics]).T
        metrics_df.columns = ['Value']
        st.dataframe(metrics_df.style.format("{:.2%}"))
            
    # Plot portfolio returns
    st.subheader("Portfolio Performance")
    cumulative_returns = (1 + returns).cumprod()
    fig = px.line(cumulative_returns, title="Cumulative Portfolio Returns")
    st.plotly_chart(fig)

with tab3:
    if portfolio.holdings:
        # Data Exploration Section
        st.header("ETF Data Explorer")
        selected_etf = st.selectbox("Select ETF to explore", list(portfolio.holdings.keys()) if portfolio.holdings else [])

        if selected_etf:
            holding = portfolio.holdings[selected_etf]
            if holding.historical_data is not None:
                st.subheader(f"{selected_etf} Price History")
                fig = px.line(holding.historical_data, y='Close', title=f"{selected_etf} Price History")
                st.plotly_chart(fig)
            
                st.subheader("Raw Data")
                st.dataframe(holding.historical_data)