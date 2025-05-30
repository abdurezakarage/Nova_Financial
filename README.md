# Nova Financial Solutions

A comprehensive financial analysis and trading platform that combines technical analysis, sentiment analysis, and machine learning to provide insights into market trends and trading opportunities.

## Features

- Technical Analysis: Advanced charting and technical indicators using TA-Lib
- Sentiment Analysis: Natural Language Processing (NLP) for market sentiment analysis
- Data Visualization: Interactive charts and graphs using Matplotlib and Seaborn
- Market Data: Real-time and historical market data integration using yfinance
- Jupyter Notebooks: Interactive analysis and research environment

## Project Structure

```
Nova_Financial/
├── src/               # Source code
├── data/             # Data storage
├── notebooks/        # Jupyter notebooks for analysis
├── NovaStock/        # Core trading and analysis modules
└── requirements.txt  # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Nova_Financial.git
cd Nova_Financial
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- nltk (3.8.1): Natural Language Processing
- pandas (2.1.4): Data manipulation and analysis
- matplotlib (3.8.2): Data visualization
- jupyter (1.0.0): Interactive notebooks
- numpy (1.26.2): Numerical computing
- yfinance (0.2.36): Yahoo Finance API integration
- pynance (0.1.1): Financial data analysis
- talib (0.4.24): Technical analysis library
- textblob (0.17.1): Text processing and sentiment analysis
- seaborn (0.13.1): Statistical data visualization

## Usage

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to the `notebooks` directory to access analysis notebooks

3. For programmatic usage, import the required modules from the `src` directory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
