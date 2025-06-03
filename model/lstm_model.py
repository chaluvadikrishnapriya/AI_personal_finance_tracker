from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def lstm_forecast(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.reshape(-1, 1))

    # Prepare sequence data
    X, y = [], []
    for i in range(len(scaled) - 3):
        X.append(scaled[i:i+3])
        y.append(scaled[i+3])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(3, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, verbose=0)

    future_input = scaled[-3:].reshape(1, 3, 1)
    preds = []
    for _ in range(3):
        pred = model.predict(future_input)[0][0]
        preds.append(pred)
        future_input = np.append(future_input[:, 1:, :], [[[pred]]], axis=1)

    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
