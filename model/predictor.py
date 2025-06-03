import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io, base64
from datetime import timedelta

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Amount'] > 0]  # remove invalid rows
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_expenses = df[df['Category'].isin([
        "Groceries", "Rent", "Transport", "Utilities", "Entertainment", 
        "Healthcare", "Shopping", "Education"
    ])]
    grouped = monthly_expenses.groupby('Month')['Amount'].sum().reset_index()
    grouped['Month'] = grouped['Month'].astype(str)
    grouped['Month_Num'] = range(len(grouped))
    return grouped

def predict_expenses(grouped):
    X = grouped[['Month_Num']]
    y = grouped['Amount']
    model = LinearRegression()
    model.fit(X, y)

    # Predict next 3 months
    future = np.array([len(grouped), len(grouped)+1, len(grouped)+2]).reshape(-1, 1)
    future_preds = model.predict(future)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(grouped['Month_Num'], y, label="Past Expenses", marker='o')
    plt.plot(future, future_preds, label="Forecast", linestyle='--', marker='x')
    plt.xlabel("Month Index")
    plt.ylabel("Total Spent")
    plt.title("Spending Forecast")
    plt.legend()
    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_url = base64.b64encode(buf.read()).decode('utf-8')

    forecast = {
        f'Month+{i+1}': round(f, 2) for i, f in enumerate(future_preds)
    }
    return forecast, plot_url

def predict_by_category(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    df = df[df['Amount'] > 0]

    results = {}
    categories = df['Category'].unique()

    for cat in categories:
        cat_df = df[df['Category'] == cat]
        monthly = cat_df.groupby('Month')['Amount'].sum().reset_index()
        monthly['Month_Num'] = range(len(monthly))
        if len(monthly) < 3:
            continue  # not enough data
        X = monthly[['Month_Num']]
        y = monthly['Amount']
        model = LinearRegression().fit(X, y)
        future = np.array([len(X), len(X)+1, len(X)+2]).reshape(-1, 1)
        preds = model.predict(future)
        results[cat] = [round(p, 2) for p in preds]
    return results

def detect_recurring_expenses(df):
    df['Date'] = pd.to_datetime(df['Date'])
    recurring = df.groupby(['Category', 'Amount']).size().reset_index(name='Count')
    recurring = recurring[recurring['Count'] >= 3]  # Consider 3+ occurrences as recurring
    return recurring[['Category', 'Amount', 'Count']].sort_values(by='Count', ascending=False)

def calculate_spending_streak(df, budget_per_day=500):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Amount'] > 0]
    daily_totals = df.groupby(df['Date'].dt.date)['Amount'].sum()
    streak = 0
    max_streak = 0

    for amt in daily_totals:
        if amt <= budget_per_day:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "current_streak": streak,
        "max_streak": max_streak
    }
