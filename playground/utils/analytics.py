import math
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import statistics as stat
from sklearn import linear_model
from copy import deepcopy
import matplotlib.pyplot as plt
from random import randint
import seaborn as sns


class Methods:
    def __init__(self):
        self._simple_methods = {
            "ema": self.ema,
            "simple_lr": self.simple_lr
        }
    
    @property
    def simple_methods(self):
        return self._simple_methods
    
    @property
    def _simple_methods_str(self):
        strlist = [f"'{x}'" for x in self._simple_methods.keys()]
        return " | ".join(strlist)

    def _advanced_lr_list(self, values, window = None, 
                          *feature_lists, feature_pred_method: str = "ema", snapshot: tuple = None):
        if feature_pred_method not in self._simple_methods:
            raise ValueError(f"feature_pred_method must be one of the following: {self._simple_methods_str}")
        
        if snapshot:
            values = values[snapshot[0]:snapshot[1]]

        preds = []
        for i in range(len(values)):
            if window:
                if i < window:
                    continue

            feature_preds = []
            features = []
            for feature in feature_lists:
                if snapshot:
                    feature = feature[snapshot[0]: snapshot[1]]
                pred_method = self._simple_methods[feature_pred_method]
                feature_preds.append(pred_method(feature[:i], window))
                if window:
                    features.append(feature[i-window:i])

            predicted_price = self.advanced_lr(values[i - window:i], feature_preds, *features, window=window)
            preds.append(predicted_price)
        
        return preds

    def advanced_lr(self, values, next_conditions, *feature_lists, snapshot: tuple = None, window = None, get_list = False):    
        if window:
            if len(values) < window:
                raise ValueError("Value list is too short for the given window.")
        
        if get_list:
            return self._advanced_lr_list(values, window, *feature_lists, snapshot=snapshot)
        
        if snapshot:
            raise ValueError("Snapshot only available when get_list is TRUE")
        
        if window:
            values = values[-window:]

        X = np.array(list(zip(*feature_lists)))
        y = np.array(values)

        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
        
        coefficients = np.linalg.pinv(X.T @ X) @ X.T @ y
        
        next_conditions = np.hstack((np.ones(1), next_conditions))
        
        predicted_price = next_conditions @ coefficients
        return predicted_price

    def _simple_lr_list(self, values, window):
        preds = []
        for i in range(len(values)):
            if i < window:
                continue
            
            preds.append(self.simple_lr(values[:i], window))

        return preds

    def simple_lr(self, values, window = None, get_list = False):
        if len(values) < window:
            raise ValueError("Value list is too short for the given window.")

        if get_list:
            return self._simple_lr_list(values, window)
        
        if window is not None:
            values = values[-window:]

        time_indices = list(range(len(values)))
        mean_x = stat.mean(time_indices)
        mean_y = stat.mean(values)

        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(time_indices, values))
        denominator = sum((x - mean_x) ** 2 for x in time_indices)
        slope = numerator / denominator
        
        intercept = mean_y - slope * mean_x
        
        next_time_index = len(values)
        predicted_next_price = slope * next_time_index + intercept
        
        return predicted_next_price

    def _calc_ema(self, values, window):
        if len(values) < window:
            raise ValueError("Value list is too short for the given window.")
        
        k = 2 / (window + 1)
        ema = sum(values[:window]) / window
        ema_values = [ema]
        for price in values[window:]:
            ema = (price * k) + (ema * (1 - k))
            ema_values.append(ema)

        return ema_values

    def ema(self, values, window, get_list=False):
        if get_list:
            return self._calc_ema(values, window)
        
        k = 2 / (window + 1)
        ema = sum(values[:window]) / window
        for val in values[window:]:
            ema = (val * k) + (ema * (1 - k))

        return ema


class Basics:
    def simple_plot(self, **kwargs) -> None:
        plt.figure(figsize=(16, 9))
        for arg in kwargs:
            plt.plot(kwargs[arg], label=arg)
        plt.legend(loc='upper center')
        plt.show()
        
    def pad(self, pred, actual, offset: int) -> tuple[list, list]:
        if isinstance(actual, pd.Series):
            actual = list(actual)
        
        if isinstance(pred, pd.Series):
            pred = list(pred)
        
        actual = actual + [actual[-1]] * 2
        pred = [pred[0]] * (offset + 1) + pred
        return pred, actual

    def basic_test(self, name: str, function, **kwargs):
        print(name.title())
        print("-" * len(name))
        for metric in kwargs:
            res = function(kwargs[metric])
            print(f"{metric}: {res}")
        print("-" * len(name))


from parsing import parse_asset_data

class Report:
    def __init__(self, assets, depth, requested_methods):
        self._df = parse_asset_data(print, depth, assets)
        self._assets = assets
        self._methods = [k for k, v in requested_methods.items() if k[v]]

        self._all_methods = {
            "df_snapshot": self.df_snapshot,
            "simple_plot": self.simple_plot,
            "price_distro": self.price_distributions,
            "simple_sma": self.simple_sma_graph,
            "asset_vol": self.asset_volatility,
            "price_corr_map": self.price_correlation_map,
            "price_corr_viz": self.price_correlation_graphs,
            "sma_corr_map": self.sma_correlation_map#,
            #"monthly_returns": self.monthly_returns
        }

        sns.set_style("whitegrid")
    
    @property
    def df_snapshot(self):
        return self._df.head(5)
    
    @property
    def all_methods(self):
        return self._all_methods.keys()
    
    def simple_plot(self, streamlit=False):
        fig = plt.figure(figsize=(10, 6))
        for column in self._df.columns:
            plt.plot(self._df.index, self._df[column], label=column)
        
        plt.title("Time Series of Asset Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def price_distributions(self, streamlit=False):
        fig = plt.figure(figsize=(12, 6))
        self._df.hist(bins=20, figsize=(12, 8), color='skyblue', edgecolor='black', linewidth=1.5)
        plt.suptitle('Histograms of Asset Prices', y=1.02)
        plt.tight_layout()
        plt.show()

    def simple_sma_graph(self):
        fig = plt.figure(figsize=(12, 6))
        rolling_mean = self._df.rolling(window=30).mean() 
        for column in self._df.columns:
            plt.plot(self._df.index, self._df[column], label=column)
            plt.plot(self._df.index, rolling_mean[column], label=column + ' (30-Day Rolling Mean)', linestyle='--')
        plt.title('Time Series of Asset Prices and 30-Day Rolling Mean')
        plt.xlabel('Index')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def asset_volatility(self):
        fig = plt.figure(figsize=(12, 6))
        self._df.boxplot(column=self._df.columns.tolist())
        plt.title('Distribution of Asset Prices (Boxplots)')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
    
    def price_correlation_map(self):
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(self._df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap of Asset Prices')
        plt.show()
    
    def price_correlation_graphs(self):
        fig = plt.figure(figsize=(12, 8))
        sns.pairplot(self._df)
        plt.suptitle('Pairplot of Asset Prices', y=1.02)
        plt.show()
    
    def sma_correlation_map(self):
        rolling_mean_30 = self._df.rolling(window=30).mean()

        corr_matrix = rolling_mean_30.corr()

        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap of 30-Day Moving Averages of Asset Prices')
        plt.show()
    
    def monthly_returns(self):
        self._df.index = pd.to_datetime(self._df.index)
        monthly_returns = self._df.resample('M').ffill().pct_change()
        correlation_matrix = monthly_returns.corr()

        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap of Monthly Returns')
        plt.show()
    
    def generate_report(self):
        for func in self.all_methods.values():
            func()
            
    def __str__(self):
        return f"Report on {', '.join(self._assets)} | Methods: {', '.join(self._methods)}"
