# Backtesting Engine - Implementation Plan

## 🎯 Goal
Create a backtesting system that simulates trading ACB stock using our trained AI models (DEFAULT_ANN or LSTM). The system will replay history: for each day, predict the next 3 days' prices, decide whether to buy/sell based on Strategy 1, and track our portfolio performance.

---

## 📚 Financial Concepts Explained (For Backend Engineers)

### What is Backtesting?
**Simple analogy**: It's like a "time machine replay" of the stock market. We pretend to trade in the past using our model's predictions and see if we would have made money.

**Example**:
- Today is Feb 24, 2026
- We backtest from Jan 1, 2024 to Dec 31, 2024
- We "replay" each day: Jan 1 → Jan 2 → ... → Dec 31
- For each day, we:
  1. Ask our model: "What will prices be in 3 days?"
  2. Decide: Buy, Sell, or Hold?
  3. Track how much money we have

### Key Trading Concepts

**1. Portfolio = Cash + Stock Holdings**
```python
# Example Portfolio State
cash = 50_000_000 VND          # Money in bank
shares = 2000                  # Number of ACB shares owned
current_price = 25_000 VND     # ACB stock price today

total_value = cash + (shares * current_price)
            = 50M + (2000 * 25000)
            = 100_000_000 VND
```

**2. Transaction Cost = Fee for Buying/Selling (0.15%)**
```python
# Example: Buy stocks worth 10M VND
buy_amount = 10_000_000
cost = buy_amount * 0.0015     # 0.15% = 15,000 VND
total_paid = 10_015_000 VND    # You pay extra
```

**3. T+3 Settlement Rule (Vietnamese Stock Market)**
- When you BUY stock on Monday, you must HOLD it until Thursday (3 days later)
- **Cannot sell before 3 days** - this is a real market rule
```python
purchase_date = "2024-01-01"   # Monday
can_sell_from = "2024-01-04"   # Thursday (3 days later)
```

**4. UP/DOWN Signal**
- Our model predicts future **prices** (not directions)
- We compare predicted price vs current price:
```python
current_close = 25_000 VND      # Today's closing price
predicted_t1 = 25_500 VND       # Model predicts tomorrow

if predicted_t1 > current_close:
    signal = "UP"               # Stock expected to rise
else:
    signal = "DOWN"             # Stock expected to fall
```

---

## 🎮 Strategy 1 (Conservative - Focus for Initial Implementation)

### Decision Rules

**BUY Signal (All 3 must agree)**
- All 3 predictions (t+1, t+2, t+3) predict price will go UP
- We have available cash
- We are NOT currently holding any stock

```python
# Example:
current_price = 25_000
pred_t1 = 25_200  # UP (+200)
pred_t2 = 25_400  # UP (+400)
pred_t3 = 25_600  # UP (+600)

# All 3 are UP → BUY fully with all available cash
```

**SELL Signal (Majority says down)**
- At least 2 out of 3 predictions say DOWN
- We currently own stock
- Stock has been held for ≥3 days (T+3 rule)

```python
# Example:
current_price = 25_000
pred_t1 = 24_800  # DOWN (-200)
pred_t2 = 24_700  # DOWN (-300)
pred_t3 = 25_100  # UP (+100)

# 2 out of 3 are DOWN → SELL all shares (if owned ≥3 days)
```

**HOLD (Do Nothing)**
- Signals are mixed (not all UP for buy, not majority DOWN for sell)
- OR we want to sell but haven't held for 3 days yet
- OR we want to buy but have no cash

---

## 🏗️ System Architecture

```
backtest_engine/
├── __init__.py                 # Package initialization
├── data_loader.py             # Load and prepare historical data
├── predictor.py               # Generate model predictions
├── portfolio.py               # Track cash, shares, transactions
├── strategy.py                # Strategy 1 buy/sell logic
├── backtest.py                # Main simulation engine
└── metrics.py                 # Calculate performance stats

run_backtest.py                 # CLI to run backtests
```

---

## 📋 Detailed Implementation Steps

### Step 1: Data Loader Module (`data_loader.py`)

**Purpose**: Load historical ACB stock data and prepare it for backtesting.

**Key Functions**:

```python
def load_acb_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load ACB stock data from CSV for the specified date range.
    
    Args:
        start_date: "YYYY-MM-DD" format (e.g., "2024-01-01")
        end_date: "YYYY-MM-DD" format
    
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
        
    Example:
        df = load_acb_data("2024-01-01", "2024-12-31")
        # Returns ~250 rows (trading days in 2024)
    """
    # 1. Read ../data/VN30_Dataset_2015_2026.csv
    # 2. Filter by Ticker == 'ACB'
    # 3. Filter by date range
    # 4. Sort by date ascending
    # 5. Return DataFrame
```

```python
def prepare_features_for_date(df: pd.DataFrame, target_date: str, 
                               window_size: int = 30) -> np.ndarray:
    """
    Prepare the 30-day feature window for making predictions.
    
    This replicates the data_preprocess.py logic:
    - Get 30 days of historical data before target_date
    - Calculate 24 technical indicators (SMA, EMA, RSI, MACD, etc.)
    - Apply MinMaxScaler transformation
    
    Args:
        df: Historical OHLC data
        target_date: Date we want to predict FROM
        window_size: Number of historical days needed (default 30)
    
    Returns:
        Scaled numpy array shape (1, 30, 24) for LSTM
        or (1, 720) for DEFAULT_ANN (flattened)
        
    Example:
        features = prepare_features_for_date(df, "2024-06-15")
        # Returns 30 days before Jun 15, with 24 features each day
    """
```

**What You Need to Know**:
- ACB data is in `../data/VN30_Dataset_2015_2026.csv`
- Available dates: 2014-06-30 to 2026-01-07
- Must calculate same 24 features as training (see `data_preprocess.py` lines 50-70)
- Must use the SAME scaler that was used during training (saved in processed data)

---

### Step 2: Predictor Module (`predictor.py`)

**Purpose**: Load trained models and generate price predictions for next 3 days.

**Key Functions**:

```python
class ModelPredictor:
    """Wrapper for loading models and making predictions."""
    
    def __init__(self, model_name: str):
        """
        Load a trained model.
        
        Args:
            model_name: "DEFAULT_ANN" or "LSTM"
        """
        # Load from ../models/{model_name}_best_model.keras
        # Load scaler from processed data
    
    def predict_next_3_days(self, features: np.ndarray, 
                            current_close: float) -> dict:
        """
        Predict prices for t+1, t+2, t+3.
        
        Args:
            features: Prepared 30-day features array
            current_close: Today's closing price (for reference)
        
        Returns:
            {
                't+1': 25200.5,  # Predicted price for tomorrow
                't+2': 25400.2,  # Predicted price for day after
                't+3': 25600.8,  # Predicted price for 3 days out
                'signals': {
                    't+1': 'UP',    # UP if pred > current_close
                    't+2': 'UP',
                    't+3': 'UP'
                }
            }
            
        Example:
            predictor = ModelPredictor("LSTM")
            result = predictor.predict_next_3_days(features, 25000)
            # Returns predictions and UP/DOWN signals
        """
        # 1. Run model.predict(features)
        # 2. Inverse transform scaled predictions to actual prices
        # 3. Compare each prediction to current_close
        # 4. Return dict with prices and signals
```

**Important Notes**:
- Model outputs are SCALED (between 0 and 1)
- Must inverse transform using scaler to get actual prices
- Model architecture:
  - DEFAULT_ANN expects input shape: (1, 720) - flattened
  - LSTM expects input shape: (1, 30, 24) - 3D
- Output shape is always: (1, 3) = [t+1_price, t+2_price, t+3_price]

---

### Step 3: Portfolio Module (`portfolio.py`)

**Purpose**: Track cash, stock holdings, and enforce trading rules.

**Key Class**:

```python
class Portfolio:
    """
    Manages portfolio state: cash, shares, transaction history.
    
    Attributes:
        cash (float): Available money in VND
        shares (int): Number of ACB shares owned
        purchase_date (str): Date when we bought current shares
        transaction_cost (float): Fee rate (0.0015 = 0.15%)
        transactions (list): History of all buy/sell actions
    """
    
    def __init__(self, initial_cash: float, transaction_cost: float = 0.0015):
        """
        Initialize portfolio.
        
        Args:
            initial_cash: Starting money (e.g., 100_000_000 VND)
            transaction_cost: Trading fee rate (default 0.15%)
        """
        self.cash = initial_cash
        self.shares = 0
        self.purchase_date = None
        self.transaction_cost = transaction_cost
        self.transactions = []
    
    def can_buy(self, price: float) -> tuple[bool, int]:
        """
        Check if we can buy and how many shares.
        
        Args:
            price: Current stock price
            
        Returns:
            (can_buy: bool, max_shares: int)
            
        Example:
            can, shares = portfolio.can_buy(25000)
            # (True, 3999) if we have 100M cash
            # We can afford 100M / 25000 = 4000 shares
            # But need to keep cash for transaction cost
        """
        if self.shares > 0:
            return False, 0  # Already holding stock
        
        # Calculate max shares we can afford
        # Cash needed = price * shares * (1 + transaction_cost)
        max_shares = int(self.cash / (price * (1 + self.transaction_cost)))
        return max_shares > 0, max_shares
    
    def buy(self, date: str, price: float, shares: int) -> bool:
        """
        Execute a buy order.
        
        Args:
            date: Purchase date
            price: Share price
            shares: Number of shares to buy
            
        Returns:
            True if successful, False if insufficient funds
            
        Side effects:
            - Reduces cash by (price * shares * (1 + transaction_cost))
            - Increases shares
            - Records purchase_date
            - Logs transaction
        """
        cost = price * shares * (1 + self.transaction_cost)
        if cost > self.cash:
            return False
        
        self.cash -= cost
        self.shares += shares
        self.purchase_date = date
        
        self.transactions.append({
            'date': date,
            'action': 'BUY',
            'price': price,
            'shares': shares,
            'cost': cost,
            'cash_after': self.cash
        })
        return True
    
    def can_sell(self, current_date: str) -> bool:
        """
        Check if we can sell (owns shares + T+3 rule).
        
        Args:
            current_date: Today's date
            
        Returns:
            True if we own shares AND held ≥3 days
            
        Example:
            portfolio.purchase_date = "2024-01-01"
            portfolio.can_sell("2024-01-03")  # False (only 2 days)
            portfolio.can_sell("2024-01-04")  # True (3 days)
        """
        if self.shares == 0:
            return False
        
        # Calculate days held
        days_held = (pd.to_datetime(current_date) - 
                     pd.to_datetime(self.purchase_date)).days
        return days_held >= 3
    
    def sell(self, date: str, price: float) -> bool:
        """
        Execute a sell order (all shares).
        
        Args:
            date: Sell date
            price: Share price
            
        Returns:
            True if successful
            
        Side effects:
            - Increases cash by (price * shares * (1 - transaction_cost))
            - Sets shares to 0
            - Clears purchase_date
            - Logs transaction
        """
        if self.shares == 0:
            return False
        
        proceeds = price * self.shares * (1 - self.transaction_cost)
        
        self.transactions.append({
            'date': date,
            'action': 'SELL',
            'price': price,
            'shares': self.shares,
            'proceeds': proceeds,
            'cash_after': self.cash + proceeds
        })
        
        self.cash += proceeds
        self.shares = 0
        self.purchase_date = None
        return True
    
    def get_total_value(self, current_price: float) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            current_price: Current stock price
            
        Returns:
            Total value = cash + (shares * current_price)
        """
        return self.cash + (self.shares * current_price)
```

---

### Step 4: Strategy Module (`strategy.py`)

**Purpose**: Implement Strategy 1 decision logic.

```python
class Strategy1:
    """
    Conservative Strategy:
    - BUY: All 3 predictions must be UP
    - SELL: At least 2 out of 3 predictions are DOWN
    """
    
    def should_buy(self, signals: dict, portfolio: Portfolio) -> bool:
        """
        Determine if we should buy.
        
        Args:
            signals: {'t+1': 'UP', 't+2': 'UP', 't+3': 'DOWN'}
            portfolio: Current portfolio state
            
        Returns:
            True if all 3 signals are UP AND we don't own stock
            
        Example:
            signals = {'t+1': 'UP', 't+2': 'UP', 't+3': 'UP'}
            strategy.should_buy(signals, portfolio)  # True
            
            signals = {'t+1': 'UP', 't+2': 'DOWN', 't+3': 'UP'}
            strategy.should_buy(signals, portfolio)  # False (mixed)
        """
        # Check all 3 are UP
        all_up = (signals['t+1'] == 'UP' and 
                  signals['t+2'] == 'UP' and 
                  signals['t+3'] == 'UP')
        
        # Check not already holding
        not_holding = portfolio.shares == 0
        
        return all_up and not_holding
    
    def should_sell(self, signals: dict, portfolio: Portfolio, 
                    current_date: str) -> bool:
        """
        Determine if we should sell.
        
        Args:
            signals: Prediction signals
            portfolio: Current portfolio state
            current_date: Today's date (for T+3 check)
            
        Returns:
            True if ≥2 signals are DOWN AND we own stock AND T+3 passed
            
        Example:
            signals = {'t+1': 'DOWN', 't+2': 'DOWN', 't+3': 'UP'}
            # 2 DOWN → should_sell returns True (if T+3 OK)
        """
        # Count DOWN signals
        down_count = sum(1 for sig in signals.values() if sig == 'DOWN')
        
        # Majority DOWN = at least 2 out of 3
        majority_down = down_count >= 2
        
        # Check T+3 rule
        can_sell = portfolio.can_sell(current_date)
        
        return majority_down and can_sell
```

---

### Step 5: Backtest Engine (`backtest.py`)

**Purpose**: The main simulation loop that ties everything together.

```python
def run_backtest(model_name: str, 
                 start_date: str, 
                 end_date: str,
                 initial_cash: float = 100_000_000,
                 transaction_cost: float = 0.0015) -> dict:
    """
    Run a backtest simulation.
    
    Args:
        model_name: "DEFAULT_ANN" or "LSTM"
        start_date: Start of backtest period "YYYY-MM-DD"
        end_date: End of backtest period "YYYY-MM-DD"
        initial_cash: Starting capital in VND
        transaction_cost: Trading fee rate
    
    Returns:
        {
            'model': model_name,
            'start_date': start_date,
            'end_date': end_date,
            'initial_cash': initial_cash,
            'final_cash': 105_234_567,
            'final_shares': 1000,
            'final_value': 130_234_567,
            'total_return_pct': 30.23,
            'num_trades': 15,
            'daily_values': [...],  # Portfolio value each day
            'transactions': [...],   # All buy/sell records
        }
    
    Algorithm:
        1. Initialize components
        2. For each trading day in [start_date, end_date]:
            a. Get current day's OHLC data
            b. Prepare 30-day feature window
            c. Generate predictions for t+1, t+2, t+3
            d. Check strategy: should buy or sell?
            e. Execute trade if applicable
            f. Record portfolio value
        3. Return final results
    """
    
    # 1. Initialize
    data_loader = DataLoader()
    predictor = ModelPredictor(model_name)
    portfolio = Portfolio(initial_cash, transaction_cost)
    strategy = Strategy1()
    
    # 2. Load historical data
    df = data_loader.load_acb_data(start_date, end_date)
    
    # 3. Track daily results
    daily_log = []
    
    # 4. Main simulation loop
    for i, row in df.iterrows():
        current_date = row['date']
        current_close = row['close']
        
        # Prepare features (30 days before current_date)
        features = data_loader.prepare_features_for_date(df, current_date)
        
        # Get predictions
        predictions = predictor.predict_next_3_days(features, current_close)
        
        # Make decision
        action = 'HOLD'
        
        if strategy.should_buy(predictions['signals'], portfolio):
            can_buy, max_shares = portfolio.can_buy(current_close)
            if can_buy:
                portfolio.buy(current_date, current_close, max_shares)
                action = f'BUY {max_shares} shares'
        
        elif strategy.should_sell(predictions['signals'], portfolio, current_date):
            portfolio.sell(current_date, current_close)
            action = f'SELL {portfolio.shares} shares'
        
        # Record daily state
        daily_log.append({
            'date': current_date,
            'close': current_close,
            'pred_t1': predictions['t+1'],
            'pred_t2': predictions['t+2'],
            'pred_t3': predictions['t+3'],
            'signal_t1': predictions['signals']['t+1'],
            'signal_t2': predictions['signals']['t+2'],
            'signal_t3': predictions['signals']['t+3'],
            'action': action,
            'cash': portfolio.cash,
            'shares': portfolio.shares,
            'portfolio_value': portfolio.get_total_value(current_close)
        })
    
    # 5. Calculate final metrics
    final_value = portfolio.get_total_value(df.iloc[-1]['close'])
    total_return = ((final_value - initial_cash) / initial_cash) * 100
    
    return {
        'model': model_name,
        'start_date': start_date,
        'end_date': end_date,
        'initial_cash': initial_cash,
        'final_cash': portfolio.cash,
        'final_shares': portfolio.shares,
        'final_value': final_value,
        'total_return_pct': total_return,
        'num_trades': len(portfolio.transactions),
        'daily_log': daily_log,
        'transactions': portfolio.transactions
    }
```

---

### Step 6: Metrics Module (`metrics.py`)

**Purpose**: Calculate and display performance statistics.

```python
def calculate_metrics(backtest_result: dict) -> dict:
    """
    Calculate performance metrics from backtest results.
    
    Args:
        backtest_result: Output from run_backtest()
    
    Returns:
        {
            'total_return_pct': 30.5,
            'num_trades': 15,
            'win_rate': 60.0,           # % of profitable trades
            'max_drawdown_pct': -12.3,  # Worst peak-to-trough decline
            'avg_holding_days': 8.2,
            'final_value': 130_500_000
        }
    """
```

```python
def print_summary(backtest_result: dict, metrics: dict):
    """
    Print a nice summary table.
    
    Example output:
    
    ============================================
    BACKTEST RESULTS: LSTM Model
    ============================================
    Period:          2024-01-01 to 2024-12-31
    Initial Capital: 100,000,000 VND
    Final Value:     130,234,567 VND
    Total Return:    +30.23%
    
    Trading Statistics:
    - Total Trades:  15
    - Win Rate:      60.0%
    - Avg Hold Days: 8.2
    - Max Drawdown:  -12.3%
    
    Final Holdings:
    - Cash:          45,234,567 VND
    - Shares:        3,400 shares @ 25,000 VND
    - Stock Value:   85,000,000 VND
    ============================================
    """
```

---

### Step 7: CLI Runner (`run_backtest.py`)

**Purpose**: User-friendly command-line interface.

```python
"""
Backtesting CLI

Usage:
    python run_backtest.py --model LSTM --start 2024-01-01 --end 2024-12-31
    
Arguments:
    --model: Model name (DEFAULT_ANN or LSTM)
    --start: Start date YYYY-MM-DD
    --end: End date YYYY-MM-DD
    --cash: Initial cash (default: 100000000)
    --transaction-cost: Fee rate (default: 0.0015)
"""

if __name__ == "__main__":
    # 1. Parse command line arguments
    # 2. Validate dates and model name
    # 3. Run backtest
    # 4. Display results
    # 5. Save to results/backtest_YYYYMMDD_HHMMSS.csv
```

---

## 🔍 Testing & Validation

### Manual Test Case

Run a simple backtest and verify each step:

```bash
python run_backtest.py --model DEFAULT_ANN --start 2024-01-01 --end 2024-01-31 --cash 100000000
```

**Expected Behavior**:
1. Load ~20 trading days from January 2024
2. For each day:
   - Make predictions for t+1, t+2, t+3
   - Show UP/DOWN signals
   - Execute buy/sell based on Strategy 1 rules
3. Print summary showing:
   - Final cash and shares
   - Total return %
   - List of all transactions

**Validation Checklist**:
- [ ] T+3 rule enforced: No sell within 3 days of purchase
- [ ] Transaction costs applied: 0.15% on both buy and sell
- [ ] Can't buy when already holding stock
- [ ] Can't sell when not holding stock
- [ ] All 3 UP required for buy signal
- [ ] At least 2 DOWN required for sell signal
- [ ] Portfolio value = cash + (shares × price)
- [ ] Cannot spend more cash than available

---

## 📁 File Structure

```
finance-forecasting-model-evaluation/
├── backtest_engine/
│   ├── __init__.py
│   ├── data_loader.py          # Load ACB data, prepare features
│   ├── predictor.py            # Model predictions
│   ├── portfolio.py            # Portfolio tracking
│   ├── strategy.py             # Strategy 1 logic
│   ├── backtest.py             # Main simulation
│   ├── metrics.py              # Performance calculations
│   └── PLAN.md                 # This file
├── run_backtest.py             # CLI entry point
├── data/
│   └── VN30_Dataset_2015_2026.csv
├── models/
│   ├── DEFAULT_ANN_best_model.keras
│   └── LSTM_best_model.keras
└── results/
    └── backtest_*.csv          # Output files
```

---

## 🚀 Future Enhancements (After Strategy 1 Works)

Once the basic engine is working, we can add:

1. **Strategy 2 (Aggressive)**: Buy on majority UP (2/3), sell on all DOWN
2. **Strategy 3 (Confidence-Weighted)**: Position sizing based on signal strength
3. **Multiple Stocks**: Extend beyond ACB to other VN30 stocks
4. **Advanced Metrics**: Sharpe ratio, risk-free rate comparison
5. **Slippage Simulation**: Add 0.1% price impact on executions
6. **Visualizations**: Equity curves, drawdown charts, prediction accuracy plots

---

## 💡 Key Implementation Tips

1. **Start Simple**: Get Strategy 1 working first before adding complexity
2. **Log Everything**: Print each day's decisions to debug logic
3. **Test with Short Periods**: Use 1 month of data first, then expand
4. **Verify Math**: Check transaction cost calculations manually
5. **Handle Edge Cases**: 
   - Not enough data for 30-day window at start
   - Market holidays (missing dates)
   - Division by zero in metrics
6. **Reuse Existing Code**: Copy feature engineering logic from `data_preprocess.py`

---

## 📞 Questions to Resolve During Implementation

- [ ] Where exactly is the scaler saved in processed_data? Need to load it
- [ ] Should we skip days when model has insufficient history (first 30 days)?
- [ ] What if backtest date range is outside available data?
- [ ] Should transaction log be saved as CSV, JSON, or both?
- [ ] Display progress bar for long backtests?

---

## ✅ Definition of Done

Strategy 1 implementation is complete when:

1. ✅ Can run: `python run_backtest.py --model LSTM --start 2024-01-01 --end 2024-12-31`
2. ✅ Prints clear summary with final portfolio value and return %
3. ✅ T+3 rule is enforced (verified in transaction log)
4. ✅ Transaction costs are correctly applied (0.15%)
5. ✅ Strategy logic works: All UP → buy, majority DOWN → sell
6. ✅ Results saved to CSV file
7. ✅ Code is readable with comments explaining finance concepts
8. ✅ No crashes on valid date ranges (2015-2026)

---

**Ready to implement? Start with `data_loader.py` and work through steps 1-7!** 🚀
