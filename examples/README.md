# FinRL Trading Examples

This folder contains comprehensive examples demonstrating the complete workflow from data acquisition to live trading execution.

## 📓 FinRL_Full_Workflow.ipynb

**Complete Interactive Tutorial** - Recommended starting point for learning the platform.

This Jupyter notebook demonstrates the entire quantitative trading workflow:

### What's Covered

- ✅ **Data Acquisition**: Fetch S&P 500 components, fundamental data, and historical prices
- ✅ **ML Strategy**: Implement machine learning-based stock selection strategies
- ✅ **Backtesting**: Professional backtesting with benchmark comparison (VOO, QQQ)
- ✅ **Live Trading**: Execute trades via Alpaca Paper Trading API

### Features

- **Multi-source data support**: Automatic selection of best available data source (FMP > WRDS > Yahoo)
- **Random Forest model**: Feature-based stock scoring and selection
- **Professional metrics**: Comprehensive risk and performance analysis
- **Risk management**: Position limits and portfolio constraints
- **Safe testing**: Paper trading for safe strategy validation

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd FinRL-Trading
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
# Required for trading
APCA_API_KEY=your_alpaca_key
APCA_API_SECRET=your_alpaca_secret
APCA_BASE_URL=https://paper-api.alpaca.markets

# Optional for better data quality
FMP_API_KEY=your_fmp_key
WRDS_USERNAME=your_wrds_username
WRDS_PASSWORD=your_wrds_password
```

**Get API Keys:**
- Alpaca (required): https://alpaca.markets/
- FMP (optional): https://financialmodelingprep.com/
- WRDS (optional): https://wrds.wharton.upenn.edu/

### 3. Run the Tutorial

```bash
jupyter notebook examples/FinRL_Full_Workflow.ipynb
```

## 📊 Performance Metrics

The tutorial demonstrates calculation of:

- **Returns**: Total return, annualized return
- **Risk-adjusted**: Sharpe ratio, Sortino ratio
- **Risk**: Maximum drawdown, volatility
- **Tail risk**: VaR, CVaR
- **Benchmark**: Alpha, beta, information ratio

## 🔄 Data Source Priority

The platform automatically selects the best available data source:

1. **FMP** ⭐⭐⭐⭐⭐ - High-quality paid data (recommended)
2. **WRDS** ⭐⭐⭐⭐☆ - Academic database (comprehensive)
3. **Yahoo Finance** ⭐⭐⭐☆☆ - Free data (always available)

## 🛠️ Troubleshooting

### Common Issues

**Data fetching fails:**
- Check internet connection
- Verify API keys in `.env` file
- Check API rate limits

**Alpaca connection issues:**
- Verify API credentials
- Ensure using Paper Trading URL
- Check account status at alpaca.markets

**Unexpected backtest results:**
- Verify data quality and date ranges
- Check weight calculations
- Review transaction cost settings

## 📚 Learning Path

1. **Start here**: Run `FinRL_Full_Workflow.ipynb` cell by cell
2. **Experiment**: Modify strategy parameters and rebalancing frequency
3. **Customize**: Develop your own strategies using the framework
4. **Deploy**: Test strategies with paper trading before going live

## 📝 License

This project is licensed under the MIT License.
