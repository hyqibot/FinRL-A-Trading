# 权重分配方法使用指南

## 概述

ML策略模块现在支持多种权重分配方法，用于为选中的股票分配投资组合权重。

## 支持的权重分配方法

### 1. 等权重 (Equal Weight) - 默认方法

为所有选中的股票分配相等的权重。

**优点：**
- 简单易懂
- 不需要额外的市场数据
- 计算速度快

**使用方法：**
```python
result = strategy.generate_weights(
    data_dict,
    prediction_mode='single',
    weight_method='equal'
)
```

### 2. 最小方差 (Min Variance)

使用历史收益率数据构建最小方差投资组合，目标是最小化组合的波动性。

**优点：**
- 考虑股票间的相关性
- 降低组合整体风险
- 更优的风险调整收益
- **自动使用基本面数据中的季度价格（adj_close_q），无需额外提供价格数据**

**缺点：**
- 计算时间较长
- 对数据质量要求高

**使用方法：**
```python
# 方法1：直接使用基本面数据中的 adj_close_q（推荐）
data_dict = {
    'fundamentals': fundamentals_df  # 必须包含 adj_close_q 列
}

result = strategy.generate_weights(
    data_dict,
    prediction_mode='single',
    weight_method='min_variance',
    lookback_periods=8  # 用于计算协方差矩阵的回溯季度数（默认8，即2年）
)

# 方法2：使用日度价格数据（可选，如果需要更精细的协方差估计）
data_dict = {
    'fundamentals': fundamentals_df,
    'prices': prices_df  # 包含 ['date', 'tic', 'close']
}

result = strategy.generate_weights(
    data_dict,
    prediction_mode='single',
    weight_method='min_variance',
    lookback_periods=252  # 用于计算协方差矩阵的回溯天数
)
```

## 价格数据格式要求

使用 `min_variance` 方法时支持两种数据格式：

### 格式1：基本面数据（推荐，自动使用）
如果基本面数据包含以下列，系统会自动使用：
- `datadate`: 季度日期
- `gvkey` 或 `tic`: 股票标识符
- `adj_close_q`: 季度调整收盘价

示例：
```python
fundamentals_df = pd.DataFrame({
    'datadate': ['2024-01-31', '2024-04-30', ...],
    'gvkey': ['001055', '001055', ...],
    'adj_close_q': [72.56, 75.06, ...],
    # ... 其他基本面指标
})
```

### 格式2：日度价格数据（可选）
如果需要使用日度数据进行更精细的协方差估计：
- `date`: 日期
- `tic` 或 `gvkey`: 股票标识符
- `close` 或 `adj_close`: 收盘价

示例：
```python
prices_df = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-02', ...],
    'tic': ['AAPL', 'AAPL', ...],
    'close': [150.5, 152.3, ...]
})
```

## 完整示例

### 示例1：单次预测 + 等权重
```python
from src.strategies.ml_strategy import MLStockSelectionStrategy
from src.strategies.base_strategy import StrategyConfig

config = StrategyConfig(
    name="ML Stock Selection",
    description="Machine learning based stock selection"
)

strategy = MLStockSelectionStrategy(config)

data_dict = {
    'fundamentals': fundamentals_df  # 包含 y_return 列
}

result = strategy.generate_weights(
    data_dict,
    test_quarters=4,
    top_quantile=0.75,
    prediction_mode='single',
    weight_method='equal'
)

print(result.weights)
```

### 示例2：滚动预测 + 最小方差（使用基本面数据）
```python
# 只需要基本面数据，会自动使用 adj_close_q 列
data_dict = {
    'fundamentals': fundamentals_df  # 包含 adj_close_q 列
}

result = strategy.generate_weights(
    data_dict,
    test_quarters=4,
    top_quantile=0.75,
    prediction_mode='rolling',
    weight_method='min_variance',
    lookback_periods=8  # 回溯8个季度
)

print(result.weights)
```

### 示例3：行业中立策略 + 最小方差（使用基本面数据）
```python
from src.strategies.ml_strategy import SectorNeutralMLStrategy

sector_config = StrategyConfig(
    name="Sector Neutral ML",
    description="Sector-neutral ML strategy"
)

sector_strategy = SectorNeutralMLStrategy(sector_config)

# 只需要基本面数据，会自动使用 adj_close_q 列
data_dict = {
    'fundamentals': fundamentals_df  # 包含 sector/gsector 和 adj_close_q 列
}

result = sector_strategy.generate_weights(
    data_dict,
    test_quarters=4,
    top_quantile=0.75,
    prediction_mode='rolling',
    weight_method='min_variance',
    lookback_periods=8  # 回溯8个季度
)

print(result.weights)
```

## 参数说明

### 通用参数
- `prediction_mode`: 预测模式
  - `'single'`: 单次预测（使用最后一个日期）
  - `'rolling'`: 滚动预测（所有可用日期）
  
- `weight_method`: 权重分配方法
  - `'equal'`: 等权重（默认）
  - `'min_variance'`: 最小方差
  
- `test_quarters`: 验证窗口季度数（默认4）
- `train_quarters`: 训练窗口季度数（默认16，仅用于rolling模式）
- `top_quantile`: 选股分位数阈值（默认0.75，即选择预测收益率前25%的股票）

### 最小方差方法专用参数
- `lookback_periods`: 计算协方差矩阵的回溯期数
  - 使用基本面数据（adj_close_q）时：默认8（8个季度，约2年）
  - 使用日度价格数据时：默认252（252个交易日，约1年）

## 注意事项

1. **数据自动识别**：
   - 系统会自动检测基本面数据中的 `adj_close_q` 列并优先使用
   - 如果基本面数据中没有价格信息，会尝试使用额外提供的 `prices` 数据
   - 无需手动选择数据源

2. **数据要求**：
   - 使用季度数据（adj_close_q）时，至少需要3个季度
   - 使用日度数据时，至少需要3个交易日
   - 推荐使用至少8个季度或252个交易日以获得稳定的协方差估计

3. **计算性能**：最小方差方法需要优化求解，计算时间较等权重方法长

4. **数据质量**：最小方差方法对数据质量敏感，缺失值会导致部分股票被排除

5. **自动降级**：如果价格数据不足或优化失败，系统会自动降级为等权重方法并记录警告

6. **风控限制**：所有权重分配方法都会应用策略配置中的风控限制（如单只股票最大权重）

## 扩展

如果需要添加新的权重分配方法，可以在 `MLStockSelectionStrategy` 类中扩展 `allocate_weights` 方法：

```python
def allocate_weights(self, selected_stocks, method='equal', **kwargs):
    if method == 'your_new_method':
        # 实现你的权重分配逻辑
        weights_df = self._compute_your_method_weights(selected_stocks, **kwargs)
    elif method == 'equal':
        weights_df = self._compute_equal_weights(selected_stocks['gvkey'].tolist())
    # ... 其他方法
    
    return result
```

