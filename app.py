from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io, base64

app = Flask(__name__)

SPENDING_LIMIT = 1000  # example monthly budget limit in INR

def format_inr(amount):
    return f"₹{round(amount, 2):,.2f}"

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Amount'] > 0]
    df['Month'] = df['Date'].dt.to_period('M')
    monthly = df.groupby('Month')['Amount'].sum().reset_index()
    monthly['Month'] = monthly['Month'].astype(str)
    monthly['Month_Num'] = range(len(monthly))
    return monthly

def predict_expenses(monthly):
    X = monthly[['Month_Num']]
    y = monthly['Amount']
    model = LinearRegression()
    model.fit(X, y)

    future = np.array([len(monthly), len(monthly)+1, len(monthly)+2]).reshape(-1, 1)
    preds = model.predict(future)

    plt.figure(figsize=(8, 4))
    plt.plot(monthly['Month_Num'], y, label="Past Expenses", marker='o')
    plt.plot(future, preds, label="Forecast", linestyle='--', marker='x')
    plt.xlabel("Month Index")
    plt.ylabel("Total Spent (₹)")
    plt.title("Spending Forecast")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.read()).decode()

    forecast = {f'Month+{i+1}': format_inr(val) for i, val in enumerate(preds)}
    return forecast, plot_url

def predict_by_category(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    df = df[df['Amount'] > 0]

    results = {}
    for cat in df['Category'].unique():
        cat_df = df[df['Category'] == cat]
        monthly = cat_df.groupby('Month')['Amount'].sum().reset_index()
        monthly['Month_Num'] = range(len(monthly))
        if len(monthly) < 3:
            continue
        X = monthly[['Month_Num']]
        y = monthly['Amount']
        model = LinearRegression().fit(X, y)
        future = np.array([len(X), len(X)+1, len(X)+2]).reshape(-1, 1)
        preds = model.predict(future)
        results[cat] = [format_inr(p) for p in preds]
    return results

def calculate_spending_streak(df, limit=SPENDING_LIMIT):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_sum = df.groupby('Month')['Amount'].sum().reset_index()

    streak = 0
    max_streak = 0
    for amount in monthly_sum['Amount']:
        if amount <= limit:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak

def detect_recurring_expenses(df):
    if 'Description' not in df.columns:
        return []  # avoid crashing if no Description

    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Amount'] > 0]
    df['Month'] = df['Date'].dt.to_period('M')

    grouped = df.groupby(['Description', 'Month'])['Amount'].sum().reset_index()
    counts = grouped.groupby('Description')['Month'].nunique()
    recurring_desc = counts[counts >= 3].index.tolist()
    recurring_df = grouped[grouped['Description'].isin(recurring_desc)]

    recurring_df['Amount'] = recurring_df['Amount'].apply(format_inr)
    recurring_list = recurring_df.sort_values(['Description', 'Month']).to_dict(orient='records')
    return recurring_list

def check_budget_alerts(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_sum = df.groupby('Month')['Amount'].sum()
    alerts = []
    for month, total in monthly_sum.items():
        if total > SPENDING_LIMIT:
            alerts.append({'month': str(month), 'amount': format_inr(total)})
    return alerts

def calculate_category_spending(df):
    category_spending = df.groupby('Category')['Amount'].sum().reset_index()
    spending_dict = dict(zip(category_spending['Category'], category_spending['Amount']))
    return {k: format_inr(v) for k, v in spending_dict.items()}

@app.route('/', methods=['GET', 'POST'])
def index():
    forecast = None
    plot_url = None
    cat_forecast = None
    streak = None
    alert = None
    category_spending = None
    recurring = None
    budget_alerts = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            alert = "No file uploaded."
            return render_template('index.html', alert=alert)

        df = pd.read_csv(file)
        if 'Category' not in df.columns or 'Amount' not in df.columns or 'Date' not in df.columns:
            alert = "CSV must contain Date, Amount, and Category columns."
            return render_template('index.html', alert=alert)

        monthly = preprocess_data(df)
        forecast, plot_url = predict_expenses(monthly)
        cat_forecast = predict_by_category(df)
        streak = calculate_spending_streak(df)
        category_spending = calculate_category_spending(df)
        recurring = detect_recurring_expenses(df)
        budget_alerts = check_budget_alerts(df)

    return render_template('index.html',
                           forecast=forecast,
                           plot_url=plot_url,
                           cat_forecast=cat_forecast,
                           streak=streak,
                           alert=alert,
                           category_spending=category_spending,
                           recurring=recurring,
                           budget_alerts=budget_alerts)

if __name__ == '__main__':
    app.run(debug=True)
