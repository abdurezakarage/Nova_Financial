import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
from dateutil import parser

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class CorrelationAnalyzer:
    def __init__(self, stock_data, news_data, ticker):
        self.ticker = ticker
        self.stock_data = stock_data.copy()  # Make a copy to avoid modifying original
        self.news_data = news_data.copy()    # Make a copy to avoid modifying original
        self.daily_sentiment = None
        self.merged_data = None
        self.correlation = None
        
        # Prepare data immediately
        self.prepare_stock_data()
        self.prepare_news_data()
        self.calculate_daily_sentiment()

    def prepare_stock_data(self):
        # Convert Date column to datetime if it's not already
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])
        self.stock_data.set_index('Date', inplace=True)
        
        # Calculate daily returns
        self.stock_data['Daily_Return'] = self.stock_data['Close'].pct_change() * 100
        return self.stock_data

    def prepare_news_data(self):
        # Convert date to datetime and remove time component
        self.news_data['date'] = pd.to_datetime(self.news_data['date']).dt.date
        self.news_data['date'] = pd.to_datetime(self.news_data['date'])
        return self.news_data

    @staticmethod
    def analyze_sentiment(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def calculate_daily_sentiment(self):
        if self.news_data is None:
            raise ValueError("News data not provided")
            
        news_df = self.news_data
        if news_df.empty:
            self.daily_sentiment = pd.DataFrame()
            return self.daily_sentiment
            
        news_df['sentiment'] = news_df['headline'].apply(self.analyze_sentiment)
        daily_sentiment = news_df.groupby('date')['sentiment'].mean().reset_index()
        daily_sentiment.set_index('date', inplace=True)
        self.daily_sentiment = daily_sentiment
        return daily_sentiment

    def analyze_correlation(self):
        if self.stock_data is None:
            raise ValueError("Stock data not provided")
        if self.daily_sentiment is None:
            self.calculate_daily_sentiment()
            
        merged_data = pd.merge(self.stock_data['Daily_Return'],
                             self.daily_sentiment['sentiment'],
                             left_index=True,
                             right_index=True,
                             how='inner')
        correlation = merged_data.corr()
        self.merged_data = merged_data
        self.correlation = correlation
        return merged_data, correlation

    def plot_correlation(self, save_path=None):
        if self.merged_data is None:
            merged_data, correlation = self.analyze_correlation()
            
        plt.figure(figsize=(12, 6))
        plt.scatter(self.merged_data['sentiment'], self.merged_data['Daily_Return'], alpha=0.5)
        plt.title(f'Correlation between News Sentiment and Stock Returns for {self.ticker}')
        plt.xlabel('News Sentiment Score')
        plt.ylabel('Daily Return (%)')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def run_full_analysis(self, plot=False, save_path=None):
        self.prepare_stock_data()
        self.prepare_news_data()
        self.calculate_daily_sentiment()
        merged_data, correlation = self.analyze_correlation()
        if plot:
            self.plot_correlation(save_path=save_path)
        return merged_data, correlation
