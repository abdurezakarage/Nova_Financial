import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from textblob import TextBlob
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from collections import Counter
import re
from scipy import stats

 # Download required NLTK data
nltk.data.find('corpora/stopwords')
nltk.data.find('corpora/wordnet')
nltk.data.find('tokenizers/punkt')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')


class Raw_Analyst_Ratings:
    def __init__(self,df,stock_name ):
        self.df=df
        self.stock_name=stock_name
       
    
    #change to datetime
    def change_to_datetime(self):   
        self.df['date'] = pd.to_datetime(self.df['date'])
    
    # set date as index
    def set_date_as_index(self):
        self.df.set_index('date', inplace=True)
    
    #reset index
    def reset_index(self):
        self.df.reset_index(inplace=True)

    def analyze_text_length(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        self.df['headline_length'] = self.df['headline'].str.len()
        sns.histplot(data=self.df, x='headline_length', bins=30, ax=ax)
        ax.set_title(f'Distribution of Headline Lengths for {self.stock_name}')
        ax.set_xlabel('Headline Length (characters)')
        ax.set_ylabel('Count')
        
        stats = self.df['headline_length'].describe()
        return fig, stats

    # publisher activity
    def analyze_publisher_activity(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        publisher_counts = self.df['publisher'].value_counts().head(10)
        sns.barplot(x=publisher_counts.values, y=publisher_counts.index, ax=ax)
        ax.set_title('Top 10 Most Active Publishers')
        ax.set_xlabel('Number of Articles')
        ax.set_ylabel('Publisher')
        return fig, publisher_counts

    def analyze_publisher_domains(self):
        """Analyze publisher domains if they are email addresses"""
        # Check if publishers are email addresses
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        email_publishers = self.df['publisher'].str.match(email_pattern)
        
        if email_publishers.any():
            # Extract domains from email addresses
            self.df['domain'] = self.df['publisher'].str.extract(r'@([^@]+)$')
            domain_counts = self.df['domain'].value_counts()
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            domain_counts.head(10).plot(kind='bar', ax=ax)
            ax.set_title('Top 10 Publisher Domains')
            ax.set_xlabel('Domain')
            ax.set_ylabel('Number of Articles')
            plt.xticks(rotation=45)
            
            return fig, domain_counts
        else:
            return None, None

    def analyze_publisher_content(self):
        """Analyze the type of news reported by different publishers"""
        # Get top publishers
        top_publishers = self.df['publisher'].value_counts().head(5).index
        
        # Create a figure for content analysis
        fig, axes = plt.subplots(len(top_publishers), 1, figsize=(12, 4*len(top_publishers)))
        if len(top_publishers) == 1:
            axes = [axes]
        
        publisher_stats = {}
        
        for idx, publisher in enumerate(top_publishers):
            publisher_news = self.df[self.df['publisher'] == publisher]
            
            # Analyze headline lengths
            headline_lengths = publisher_news['headline'].str.len()
            
            # Plot distribution
            sns.histplot(data=headline_lengths, ax=axes[idx], bins=20)
            axes[idx].set_title(f'Headline Length Distribution - {publisher}')
            axes[idx].set_xlabel('Headline Length')
            axes[idx].set_ylabel('Count')
            
            # Calculate statistics
            publisher_stats[publisher] = {
                'article_count': len(publisher_news),
                'avg_headline_length': headline_lengths.mean(),
                'std_headline_length': headline_lengths.std(),
                'date_range': (publisher_news['date'].min(), publisher_news['date'].max())
            }
        
        plt.tight_layout()
        
        # Print detailed statistics
        print("\n=== Publisher Content Analysis ===")
        for publisher, stats in publisher_stats.items():
            print(f"\nPublisher: {publisher}")
            print(f"Total Articles: {stats['article_count']}")
            print(f"Average Headline Length: {stats['avg_headline_length']:.2f} characters")
            print(f"Date Range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        
        return fig, publisher_stats

    #publication trend
    def analyze_publication_trends(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        daily_counts = self.df.groupby(self.df['date'].dt.date).size()
        daily_counts.plot(kind='line', ax=ax)
        ax.set_title('News Frequency Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Articles')
        plt.xticks(rotation=45)
        return fig, daily_counts

    #day of week
    def analyze_day_of_week(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        day_counts = self.df['day_of_week'].value_counts()
        sns.barplot(x=day_counts.index, y=day_counts.values, ax=ax)
        ax.set_title('News Frequency by Day of Week')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Number of Articles')
        plt.xticks(rotation=45)
        return fig, day_counts
    #preprocess text
    def preprocess_text(self, text):
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        return tokens
    #analyze topics
    def analyze_topics(self):
        # Combine all headlines
        all_headlines = ' '.join(self.df['headline'].astype(str))
        
        # Preprocess text
        tokens = self.preprocess_text(all_headlines)
        
        # Create frequency distribution
        fdist = FreqDist(tokens)
        
        # Create figures
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Plot most common words
        most_common = fdist.most_common(20)
        words, counts = zip(*most_common)
        sns.barplot(x=list(counts), y=list(words), ax=ax1)
        ax1.set_title('Top 20 Most Common Words')
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Words')
        
        # Plot word frequency distribution
        fdist.plot(30, cumulative=False, ax=ax2)
        ax2.set_title('Word Frequency Distribution')
        ax2.set_xlabel('Words')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Find common phrases (bigrams)
        bigrams = list(nltk.bigrams(tokens))
        bigram_freq = Counter(bigrams)
        common_phrases = bigram_freq.most_common(10)
        
        # Print statistics
        print("\n=== Text Analysis Results ===")
        print("\nTop 20 Most Common Words:")
        for word, count in most_common:
            print(f"{word}: {count}")
        
        print("\nTop 10 Common Phrases:")
        for (word1, word2), count in common_phrases:
            print(f"{word1} {word2}: {count}")
        
        return fig, (most_common, common_phrases)
    #analyze publication times
    def analyze_publication_times(self):
        # Extract hour from datetime
        self.df['hour'] = self.df['date'].dt.hour
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot hourly distribution
        hourly_counts = self.df['hour'].value_counts().sort_index()
        sns.barplot(x=hourly_counts.index, y=hourly_counts.values, ax=ax1)
        ax1.set_title('Publication Frequency by Hour')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Number of Articles')
        ax1.set_xticks(range(0, 24))
        
        # Plot heatmap of day vs hour
        pivot_table = pd.crosstab(self.df['day_of_week'], self.df['hour'])
        sns.heatmap(pivot_table, cmap='YlOrRd', ax=ax2)
        ax2.set_title('Publication Heatmap: Day vs Hour')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Day of Week')
        
        plt.tight_layout()
        
        # Calculate statistics
        peak_hour = hourly_counts.idxmax()
        peak_count = hourly_counts.max()
        avg_articles_per_hour = hourly_counts.mean()
        
        print("\n=== Publication Time Analysis ===")
        print(f"\nPeak Publication Hour: {peak_hour}:00 ({peak_count} articles)")
        print(f"Average Articles per Hour: {avg_articles_per_hour:.2f}")
        print("\nTop 5 Most Active Hours:")
        for hour, count in hourly_counts.nlargest(5).items():
            print(f"{hour}:00 - {count} articles")
        
        return fig, hourly_counts
    #analyze publication spikes
    def analyze_publication_spikes(self, threshold_std=2):
        # Calculate daily article counts
        daily_counts = self.df.groupby(self.df['date'].dt.date).size()
        
        # Calculate statistics
        mean_count = daily_counts.mean()
        std_count = daily_counts.std()
        threshold = mean_count + (threshold_std * std_count)
        
        # Identify spikes
        spikes = daily_counts[daily_counts > threshold]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Plot daily counts
        daily_counts.plot(kind='line', ax=ax, label='Daily Articles')
        
        # Plot threshold line
        ax.axhline(y=threshold, color='r', linestyle='--', label=f'Spike Threshold ({threshold_std}σ)')
        
        # Highlight spikes
        for date, count in spikes.items():
            ax.scatter(date, count, color='red', s=100, zorder=5)
            ax.annotate(f'{count} articles', 
                       (date, count),
                       xytext=(10, 10),
                       textcoords='offset points')
        
        ax.set_title('Publication Frequency with Spikes Highlighted')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.legend()
        
        # Print spike analysis
        print("\n=== Publication Spike Analysis ===")
        print(f"\nMean daily articles: {mean_count:.2f}")
        print(f"Standard deviation: {std_count:.2f}")
        print(f"Spike threshold: {threshold:.2f} articles")
        print(f"\nFound {len(spikes)} significant spikes:")
        
        for date, count in spikes.items():
            print(f"\nDate: {date}")
            print(f"Articles: {count}")
            print("Headlines from this day:")
            day_articles = self.df[self.df['date'].dt.date == date]
            for _, row in day_articles.iterrows():
                print(f"- {row['headline']}")
        
        return fig, spikes

    #perform eda
    def perform_eda(self):
        # Create a figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Text Length Analysis
        plt.subplot(2, 2, 1)
        _, text_stats = self.analyze_text_length()
        
        # 2. Publisher Activity Analysis
        plt.subplot(2, 2, 2)
        _, publisher_counts = self.analyze_publisher_activity()
        
        # 3. Publication Date Analysis
        plt.subplot(2, 2, 3)
        _, daily_counts = self.analyze_publication_trends()
        
        # 4. Day of Week Analysis
        plt.subplot(2, 2, 4)
        _, day_counts = self.analyze_day_of_week()
        
        plt.tight_layout()
        plt.show()
        
        # Perform publisher domain analysis
        domain_fig, domain_stats = self.analyze_publisher_domains()
        if domain_fig is not None:
            domain_fig.show()
        
        # Perform publisher content analysis
        content_fig, content_stats = self.analyze_publisher_content()
        content_fig.show()
        
        # Perform topic analysis
        topic_fig, topic_stats = self.analyze_topics()
        topic_fig.show()
        
        # Perform publication time analysis
        time_fig, hourly_stats = self.analyze_publication_times()
        time_fig.show()
        
        # Perform spike analysis
        spike_fig, spike_stats = self.analyze_publication_spikes()
        spike_fig.show()
        
        # Print summary statistics
        print("\n=== Summary Statistics ===")
        print("\nHeadline Length Statistics:")
        print(text_stats)
        
        print("\nTop 5 Most Active Publishers:")
        print(publisher_counts.head())
        
        print("\nPublication Date Range:")
        print(f"From: {self.df['date'].min()}")
        print(f"To: {self.df['date'].max()}")
        
        print("\nArticles per Day of Week:")
        print(day_counts) 