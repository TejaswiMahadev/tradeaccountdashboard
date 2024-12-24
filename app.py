import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast

# Step 1: Data Loading
def load_and_clean_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    data.dropna(subset=['Trade_History'], inplace=True)
    data['Trade_History'] = data['Trade_History'].apply(ast.literal_eval)
    return data

# Step 2: Parse Trade_History into a structured format
def parse_trade_history(data):
    # Explode the Trade_History column
    trade_history_expanded = data.explode('Trade_History', ignore_index=True)
    
    # Parse the nested dictionaries in the Trade_History column
    trade_details = pd.json_normalize(trade_history_expanded['Trade_History'])
    
    # Add Port_IDs to the parsed details
    trade_details['Port_IDs'] = trade_history_expanded['Port_IDs'].reset_index(drop=True)
    
    # Classify trades by side and position
    trade_details['Position_Type'] = trade_details.apply(
        lambda x: f"{x['side'].lower()}_{x['positionSide'].lower()}", axis=1
    )
    return trade_details

# Step 3: Calculate Metrics
def calculate_metrics(trade_details):
    grouped = trade_details.groupby('Port_IDs')
    metrics = pd.DataFrame()

    # ROI
    metrics['ROI'] = grouped['realizedProfit'].sum() / grouped['quantity'].sum()

    # PnL
    metrics['PnL'] = grouped['realizedProfit'].sum()

    # Sharpe Ratio
    metrics['Sharpe_Ratio'] = grouped['realizedProfit'].mean() / grouped['realizedProfit'].std()

    # MDD
    def max_drawdown(profits):
        cumulative = profits.cumsum()
        drawdown = cumulative - cumulative.cummax()
        return drawdown.min()

    metrics['MDD'] = grouped['realizedProfit'].apply(max_drawdown)

    # Win Rate, Win Positions, Total Positions
    metrics['Win_Rate'] = grouped.apply(lambda x: (x['realizedProfit'] > 0).mean())
    metrics['Win_Positions'] = grouped.apply(lambda x: (x['realizedProfit'] > 0).sum())
    metrics['Total_Positions'] = grouped.size()

    return metrics

# Step 4: Ranking Algorithm
def rank_accounts(metrics):
    weights = {'ROI': 0.3, 'PnL': 0.3, 'Sharpe_Ratio': 0.2, 'Win_Rate': 0.1, 'Total_Positions': 0.1}
    metrics['Score'] = (
        metrics['ROI'] * weights['ROI'] +
        metrics['PnL'] * weights['PnL'] +
        metrics['Sharpe_Ratio'] * weights['Sharpe_Ratio'] +
        metrics['Win_Rate'] * weights['Win_Rate'] +
        metrics['Total_Positions'] * weights['Total_Positions']
    )
    return metrics.sort_values(by='Score', ascending=False)

# Step 5: Visualization
def visualize_results(metrics):
    st.subheader("Top 20 Accounts by Score")
    top_20 = metrics.head(20)

    # Barplot of scores
    st.bar_chart(top_20['Score'])

    # Distribution of ROI
    st.subheader("Distribution of ROI")
    st.pyplot(sns.histplot(metrics['ROI'], kde=True).figure)

    # Scatterplot of Sharpe Ratio vs ROI
    st.subheader("Sharpe Ratio vs ROI")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Sharpe_Ratio', y='ROI', data=metrics, ax=ax)
    st.pyplot(fig)

def plot_time_series(trade_details):
    st.subheader("Time-Series of Profits")
    trade_details['timestamp'] = pd.to_datetime(trade_details['timestamp'])
    profit_time_series = trade_details.groupby(['Port_IDs', 'timestamp'])['realizedProfit'].sum().reset_index()

    for port_id in profit_time_series['Port_IDs'].unique():
        account_data = profit_time_series[profit_time_series['Port_IDs'] == port_id]
        plt.plot(account_data['timestamp'], account_data['realizedProfit'], label=f'Port_ID {port_id}')
    
    plt.xlabel("Time")
    plt.ylabel("Profit")
    plt.legend()
    st.pyplot(plt.figure())

# Streamlit App
def main():
    st.title("Trade Account Ranking Dashboard")

    # File Upload
    uploaded_file = st.file_uploader("Upload your trade data CSV", type=["csv"])
    if uploaded_file is not None:
        with st.spinner("Processing data..."):
            # Step 1: Load and clean data
            data = load_and_clean_data(uploaded_file)

            # Step 2: Parse Trade_History
            trade_details = parse_trade_history(data)

            # Step 3: Calculate metrics
            metrics = calculate_metrics(trade_details)

            # Step 4: Rank accounts
            ranked_metrics = rank_accounts(metrics)

            # Step 5: Display results
            st.success("Data processed successfully!")
            visualize_results(ranked_metrics)

            # Show data
            st.subheader("Ranked Metrics")
            st.dataframe(ranked_metrics)

            # Download Button
            csv_data = ranked_metrics.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Ranked Metrics as CSV",
                data=csv_data,
                file_name="ranked_metrics.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()

