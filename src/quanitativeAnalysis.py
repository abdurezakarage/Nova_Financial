import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import talib
import numpy as np
from pynance import metrics

class QuanitativeAnalysis:
    def __init__(self, df, stock_name):
        self.df = df
        self.stock_name = stock_name
    def change_to_datetime(self):
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.set_index('date', inplace=True)
    def reset_index(self):
        self.df.reset_index(inplace=True)
    def plot_stock_price(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.df['close'], label='Close Price')
        plt.title(f'{self.stock_name} Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
    #load data from csv file
    def load_data(self, file_path):
     
        self.df = pd.read_csv(file_path)
        # Ensure required columns are present
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in the data.")
    #calculate technical indicators
    def calculate_technical_indicators(self):
        # Calculate Simple Moving Averages
        self.df['SMA_20'] = talib.SMA(self.df['Close'], timeperiod=20)
        self.df['SMA_50'] = talib.SMA(self.df['Close'], timeperiod=50)
        self.df['SMA_200'] = talib.SMA(self.df['Close'], timeperiod=200)
        
        # Calculate RSI
        self.df['RSI'] = talib.RSI(self.df['Close'], timeperiod=14)
        
        # Calculate MACD
        macd, macd_signal, macd_hist = talib.MACD(
            self.df['Close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        self.df['MACD'] = macd
        self.df['MACD_Signal'] = macd_signal
        self.df['MACD_Hist'] = macd_hist
        
        return self.df
    #plot technical indicators
    def plot_technical_indicators(self):
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
        # Plot price and moving averages
        ax1.plot(self.df.index, self.df['Close'], label='Close Price')
        ax1.plot(self.df.index, self.df['SMA_20'], label='20-day SMA')
        ax1.plot(self.df.index, self.df['SMA_50'], label='50-day SMA')
        ax1.plot(self.df.index, self.df['SMA_200'], label='200-day SMA')
        ax1.set_title(f'{self.stock_name} Price and Moving Averages')
        ax1.set_ylabel('Price')
        ax1.legend()
        
        # Plot RSI
        ax2.plot(self.df.index, self.df['RSI'], label='RSI')
        ax2.axhline(y=70, color='r', linestyle='--')
        ax2.axhline(y=30, color='g', linestyle='--')
        ax2.set_ylabel('RSI')
        ax2.legend()
        
        # Plot MACD
        ax3.plot(self.df.index, self.df['MACD'], label='MACD')
        ax3.plot(self.df.index, self.df['MACD_Signal'], label='Signal Line')
        ax3.bar(self.df.index, self.df['MACD_Hist'], label='MACD Histogram')
        ax3.set_ylabel('MACD')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()

    def calculate_financial_metrics(self):
        # Calculate daily returns
        self.df['Daily_Return'] = self.df['Close'].pct_change()
        
        # Calculate volatility (annualized)
        self.df['Volatility'] = self.df['Daily_Return'].rolling(window=252).std() * np.sqrt(252)
        
        # Calculate cumulative returns
        self.df['Cumulative_Return'] = (1 + self.df['Daily_Return']).cumprod() - 1
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 0.02)
        risk_free_rate = 0.02
        excess_returns = self.df['Daily_Return'] - risk_free_rate/252
        self.df['Sharpe_Ratio'] = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Calculate Maximum Drawdown
        rolling_max = self.df['Close'].expanding().max()
        self.df['Drawdown'] = self.df['Close'] / rolling_max - 1
        self.df['Max_Drawdown'] = self.df['Drawdown'].expanding().min()
        
        return self.df
    
    def plot_financial_metrics(self):
        """
        Plot financial metrics including returns, volatility, and drawdown.
        """
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot cumulative returns
        ax1.plot(self.df.index, self.df['Cumulative_Return'] * 100, label='Cumulative Returns (%)')
        ax1.set_title(f'{self.stock_name} Cumulative Returns')
        ax1.set_ylabel('Returns (%)')
        ax1.legend()
        
        # Plot volatility
        ax2.plot(self.df.index, self.df['Volatility'] * 100, label='Volatility (%)')
        ax2.set_title('Annualized Volatility')
        ax2.set_ylabel('Volatility (%)')
        ax2.legend()
        
        # Plot drawdown
        ax3.fill_between(self.df.index, self.df['Max_Drawdown'] * 100, 0, color='red', alpha=0.3)
        ax3.set_title('Maximum Drawdown')
        ax3.set_ylabel('Drawdown (%)')
        
        plt.tight_layout()
        plt.show()
        
    def print_metrics_summary(self):
        """
        Print a summary of key financial metrics.
        """
        print(f"\nFinancial Metrics Summary for {self.stock_name}:")
        print("-" * 50)
        print(f"Total Return: {self.df['Cumulative_Return'].iloc[-1]*100:.2f}%")
        print(f"Annualized Volatility: {self.df['Volatility'].iloc[-1]*100:.2f}%")
        print(f"Sharpe Ratio: {self.df['Sharpe_Ratio'].iloc[-1]:.2f}")
        print(f"Maximum Drawdown: {self.df['Max_Drawdown'].min()*100:.2f}%")
        print(f"Average Daily Return: {self.df['Daily_Return'].mean()*100:.2f}%")
        print(f"Return/Risk Ratio: {(self.df['Daily_Return'].mean() / self.df['Daily_Return'].std()):.2f}")

    def plot_price_volume_analysis(self):
        """
        Create a comprehensive price and volume analysis plot.
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot candlestick-like price movement
        ax1.plot(self.df.index, self.df['Close'], label='Close Price', color='blue')
        ax1.fill_between(self.df.index, self.df['High'], self.df['Low'], alpha=0.2, color='gray')
        
        # Add moving averages
        ax1.plot(self.df.index, self.df['SMA_20'], label='20-day SMA', color='orange')
        ax1.plot(self.df.index, self.df['SMA_50'], label='50-day SMA', color='red')
        
        ax1.set_title(f'{self.stock_name} Price and Volume Analysis')
        ax1.set_ylabel('Price')
        ax1.legend()
        
        # Plot volume
        ax2.bar(self.df.index, self.df['Volume'], color='green', alpha=0.5, label='Volume')
        ax2.set_ylabel('Volume')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_indicator_relationships(self):
        """
        Create plots showing relationships between different technical indicators.
        """
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # RSI vs Price
        ax1.scatter(self.df['RSI'], self.df['Close'], alpha=0.5)
        ax1.set_xlabel('RSI')
        ax1.set_ylabel('Price')
        ax1.set_title('RSI vs Price')
        
        # MACD vs Price
        ax2.scatter(self.df['MACD'], self.df['Close'], alpha=0.5)
        ax2.set_xlabel('MACD')
        ax2.set_ylabel('Price')
        ax2.set_title('MACD vs Price')
        
        # Volume vs Price Change
        ax3.scatter(self.df['Volume'], self.df['Daily_Return'], alpha=0.5)
        ax3.set_xlabel('Volume')
        ax3.set_ylabel('Daily Return')
        ax3.set_title('Volume vs Daily Return')
        
        # Volatility vs Price
        ax4.scatter(self.df['Volatility'], self.df['Close'], alpha=0.5)
        ax4.set_xlabel('Volatility')
        ax4.set_ylabel('Price')
        ax4.set_title('Volatility vs Price')
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self):
        """
        Create a correlation heatmap of key metrics and indicators.
        """
        # Select relevant columns for correlation
        correlation_columns = [
            'Close', 'Volume', 'RSI', 'MACD', 
            'Daily_Return', 'Volatility', 'SMA_20', 'SMA_50'
        ]
        
        # Calculate correlation matrix
        corr_matrix = self.df[correlation_columns].corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        
        # Add labels
        plt.xticks(range(len(correlation_columns)), correlation_columns, rotation=45)
        plt.yticks(range(len(correlation_columns)), correlation_columns)
        
        # Add correlation values
        for i in range(len(correlation_columns)):
            for j in range(len(correlation_columns)):
                plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                        ha='center', va='center')
        
        plt.title(f'{self.stock_name} Correlation Heatmap')
        plt.tight_layout()
        plt.show()
    
    def plot_trading_signals(self):
        """
        Create a plot showing potential trading signals based on technical indicators.
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price and moving averages
        ax1.plot(self.df.index, self.df['Close'], label='Close Price', color='blue')
        ax1.plot(self.df.index, self.df['SMA_20'], label='20-day SMA', color='orange')
        ax1.plot(self.df.index, self.df['SMA_50'], label='50-day SMA', color='red')
        
        # Generate trading signals
        # Buy signal: RSI < 30 and MACD crosses above signal line
        buy_signal = (self.df['RSI'] < 30) & (self.df['MACD'] > self.df['MACD_Signal'])
        # Sell signal: RSI > 70 and MACD crosses below signal line
        sell_signal = (self.df['RSI'] > 70) & (self.df['MACD'] < self.df['MACD_Signal'])
        
        # Plot buy and sell signals
        ax1.scatter(self.df.index[buy_signal], self.df['Close'][buy_signal], 
                   marker='^', color='green', s=100, label='Buy Signal')
        ax1.scatter(self.df.index[sell_signal], self.df['Close'][sell_signal], 
                   marker='v', color='red', s=100, label='Sell Signal')
        
        ax1.set_title(f'{self.stock_name} Trading Signals')
        ax1.set_ylabel('Price')
        ax1.legend()
        
        # Plot RSI with overbought/oversold levels
        ax2.plot(self.df.index, self.df['RSI'], label='RSI', color='purple')
        ax2.axhline(y=70, color='r', linestyle='--', label='Overbought')
        ax2.axhline(y=30, color='g', linestyle='--', label='Oversold')
        ax2.set_ylabel('RSI')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()








