###  Import libraries

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pulp
import matplotlib.pyplot as plt

###  synthetic_data

def generate_synthetic_microgrid(days=30, seed=42):
    np.random.seed(seed)
    hours = days * 24
    t = np.arange(hours)

    # Load profile (daily sinusoidal + noise)
    load = 50 + 20*np.sin(2*np.pi*t/24) + np.random.normal(0, 3, hours)

    # PV generation (daytime only)
    pv = np.maximum(0, 30*np.sin(2*np.pi*(t-6)/24))

    # Time-of-use pricing
    price = np.where((t % 24 >= 17) & (t % 24 <= 21), 0.30, 0.15)

    return load, pv, price


###  forecasting_dataset-lstm_model

class TimeSeriesDataset:
    def __init__(self, series, window_size=24):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.series_scaled = self.scaler.fit_transform(series.reshape(-1,1))

    def create_sequences(self):
        X, y = [], []
        for i in range(len(self.series_scaled) - self.window_size):
            X.append(self.series_scaled[i:i+self.window_size])
            y.append(self.series_scaled[i+self.window_size])
        return np.array(X), np.array(y)

    def inverse_transform(self, values):
        return self.scaler.inverse_transform(values)


###  forecasting-lstm model

def build_lstm_model(window_size):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(window_size, 1)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mse',
        metrics=['mae']
    )

    return model

###  forecasting-train_forecast

def train_forecaster(series, window=24, epochs=10):

    dataset = TimeSeriesDataset(series, window)
    X, y = dataset.create_sequences()

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_lstm_model(window)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        verbose=0
    )

    return model, dataset

### optimization-battery_milp


class BatteryOptimizer:

    def __init__(self,
                 load_forecast,
                 pv_forecast,
                 price,
                 capacity=100,
                 max_power=50,
                 eta_c=0.95,
                 eta_d=0.95):

        self.T = len(load_forecast)
        self.load = load_forecast
        self.pv = pv_forecast
        self.price = price
        self.capacity = capacity
        self.max_power = max_power
        self.eta_c = eta_c
        self.eta_d = eta_d

    def solve(self):

        prob = pulp.LpProblem("Battery_Scheduling", pulp.LpMinimize)

        P_ch = pulp.LpVariable.dicts("P_ch", range(self.T), 0, self.max_power)
        P_dis = pulp.LpVariable.dicts("P_dis", range(self.T), 0, self.max_power)
        SOC = pulp.LpVariable.dicts("SOC", range(self.T), 0, self.capacity)
        P_grid = pulp.LpVariable.dicts("P_grid", range(self.T), 0)
        u = pulp.LpVariable.dicts("u", range(self.T), cat="Binary")

        prob += pulp.lpSum(self.price[t] * P_grid[t] for t in range(self.T))

        for t in range(self.T):

            prob += self.load[t] == \
                    self.pv[t] + P_dis[t] - P_ch[t] + P_grid[t]

            prob += P_ch[t] <= u[t] * self.max_power
            prob += P_dis[t] <= (1-u[t]) * self.max_power

            if t == 0:
                prob += SOC[t] == \
                        self.eta_c * P_ch[t] - \
                        (1/self.eta_d) * P_dis[t]
            else:
                prob += SOC[t] == SOC[t-1] + \
                        self.eta_c * P_ch[t] - \
                        (1/self.eta_d) * P_dis[t]

        prob.solve()

        return {
            "P_ch": [P_ch[t].value() for t in range(self.T)],
            "P_dis": [P_dis[t].value() for t in range(self.T)],
            "SOC": [SOC[t].value() for t in range(self.T)],
            "P_grid": [P_grid[t].value() for t in range(self.T)]
        }

### ems_controller

def run_pipeline(load_series, pv_series, price):

    model_load, ds_load = train_forecaster(load_series)
    model_pv, ds_pv = train_forecaster(pv_series)

    # For demo: use last 24h as forecast horizon
    load_forecast = load_series[-24:]
    pv_forecast = pv_series[-24:]

    optimizer = BatteryOptimizer(
        load_forecast,
        pv_forecast,
        price[-24:]
    )

    results = optimizer.solve()

    return results

### Main

def main():

    load, pv, price = generate_synthetic_microgrid(days=30)

    results = run_pipeline(load, pv, price)

    print("Optimization completed.")
    print("Grid import (first 5 hours):", results["P_grid"][:5])

    plt.plot(results["SOC"])
    plt.title("Battery State of Charge")
    plt.xlabel("Hour")
    plt.ylabel("SOC")
    plt.show()

if __name__ == "__main__":
    main()


