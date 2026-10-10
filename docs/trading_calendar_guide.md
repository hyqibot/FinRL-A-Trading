# 交易日历与增量更新指南

## 概述

本项目现在使用**真实的NYSE交易日历**来判断交易日和非交易日，确保增量数据更新的准确性。

## 主要特性

### 1. 真实交易日历

- ? 使用 `pandas-market-calendars` 或 `exchange-calendars` 获取NYSE交易日历
- ? 自动排除周末（周六、周日）
- ? 自动排除美国股市节假日（New Year, Martin Luther King Jr. Day, Presidents Day, Good Friday, Memorial Day, Independence Day, Labor Day, Thanksgiving, Christmas等）
- ? 如果库未安装，自动fallback到基础日历（只排除周末）

### 2. 智能增量更新

系统会检查数据库中已有的数据，只请求缺失的部分：

#### 场景1：完全缺失
```python
数据库: 无数据
请求: 2024-01-01 到 2024-01-31
结果: 向API请求整个月的数据
```

#### 场景2：部分缺失（端点）
```python
数据库: 2024-01-10 到 2024-01-20
请求: 2024-01-01 到 2024-01-31
结果: 
  - 请求 2024-01-01 到 2024-01-09
  - 请求 2024-01-21 到 2024-01-31
  - 不请求已有的 01-10 到 01-20
```

#### 场景3：中间有gap
```python
数据库: 2024-01-02~05, 2024-01-11~15
请求: 2024-01-01 到 2024-01-20
结果:
  - 请求 2024-01-01 (端点缺失)
  - 请求 2024-01-08~10 (中间gap，交易日)
  - 请求 2024-01-16~20 (端点缺失)
  - 自动排除 01-06, 01-07 (周末)
```

## 安装

### 方式1：推荐（pandas-market-calendars）
```bash
pip install pandas-market-calendars
```

### 方式2：现代替代（exchange-calendars）
```bash
pip install exchange-calendars
```

### 方式3：从requirements.txt安装
```bash
pip install -r requirements.txt
```

## 使用示例

### 基础用法

```python
from src.data.trading_calendar import (
    get_trading_days, 
    is_trading_day,
    get_missing_trading_days
)

# 获取某个时间范围的所有交易日
trading_days = get_trading_days('2024-01-01', '2024-01-31')
print(f"2024年1月有 {len(trading_days)} 个交易日")

# 检查某一天是否为交易日
if is_trading_day('2024-01-01'):
    print("2024-01-01 是交易日")
else:
    print("2024-01-01 是非交易日")  # New Year's Day

# 查找缺失的交易日
existing = ['2024-01-02', '2024-01-03', '2024-01-08']
missing = get_missing_trading_days(
    existing, 
    '2024-01-02', 
    '2024-01-10'
)
print(f"缺失的交易日: {missing}")
# 输出: ['2024-01-04', '2024-01-05', '2024-01-09', '2024-01-10']
# 注意: 01-06, 01-07 (周末) 不在缺失列表中
```

### 数据获取示例

```python
from src.data.data_fetcher import get_data_manager

# 创建数据管理器
manager = get_data_manager(preferred_source='FMP')

# 第一次请求：获取2024年1月数据
# 系统会从API获取所有交易日数据并保存到数据库
price_data = manager.get_price_data(
    tickers=['AAPL', 'MSFT'],
    start_date='2024-01-01',
    end_date='2024-01-31'
)

# 第二次请求：扩展到2月
# 系统检测到1月数据已存在，只会从API请求2月的数据
price_data = manager.get_price_data(
    tickers=['AAPL', 'MSFT'],
    start_date='2024-01-01',
    end_date='2024-02-29'
)
```

### 直接使用DataStore

```python
from src.data.data_store import get_data_store

store = get_data_store()

# 检查缺失的数据范围
missing_ranges = store.get_missing_price_dates(
    ticker='AAPL',
    start_date='2024-01-01',
    end_date='2024-12-31'
)

print(f"需要获取的数据范围: {missing_ranges}")
# 输出示例: [('2024-01-01', '2024-01-05'), ('2024-03-10', '2024-03-15')]
```

## 交易日历功能详解

### 支持的交易所

默认使用NYSE（纽约证券交易所）日历，也可以指定其他交易所：

```python
# NYSE (默认)
trading_days = get_trading_days('2024-01-01', '2024-12-31', exchange='NYSE')

# NASDAQ
trading_days = get_trading_days('2024-01-01', '2024-12-31', exchange='NASDAQ')

# 其他交易所（需要相应的日历库支持）
# 'LSE' (伦敦), 'TSE' (东京), 'HKEX' (香港) 等
```

### 美国股市节假日列表（2024年）

| 日期 | 节假日 |
|------|--------|
| 2024-01-01 | New Year's Day |
| 2024-01-15 | Martin Luther King Jr. Day |
| 2024-02-19 | Presidents Day |
| 2024-03-29 | Good Friday |
| 2024-05-27 | Memorial Day |
| 2024-07-04 | Independence Day |
| 2024-09-02 | Labor Day |
| 2024-11-28 | Thanksgiving |
| 2024-12-25 | Christmas |

**注意**：这些日期会被自动排除，不会出现在缺失数据检测中。

## 配置

### 数据库位置

通过环境变量 `DATA_BASE_DIR` 设置：

```bash
# .env 文件
DATA_BASE_DIR=/path/to/your/database
```

或在代码中：

```python
import os
os.environ['DATA_BASE_DIR'] = '/path/to/your/database'

from src.data.data_store import get_data_store
store = get_data_store()
```

## 测试

运行测试脚本验证功能：

```bash
python test_incremental_update.py
```

测试内容包括：
1. ? 交易日历功能测试
2. ? 周末自动排除
3. ? 节假日自动排除
4. ? 中间gap检测
5. ? 增量更新逻辑
6. ? 数据填充验证

## 性能优化

### 缓存机制

交易日历数据会被缓存，避免重复计算：

```python
# 第一次调用：从日历库获取
trading_days1 = get_trading_days('2024-01-01', '2024-12-31')

# 后续调用：使用缓存（快速）
trading_days2 = get_trading_days('2024-01-01', '2024-12-31')
```

### 批量检查

使用 `get_trading_days_set` 进行快速成员测试：

```python
# 获取交易日集合（用于快速查找）
trading_days_set = get_trading_days_set('2024-01-01', '2024-12-31')

# O(1) 时间复杂度检查
if '2024-01-15' in trading_days_set:
    print("是交易日")
```

## 故障排查

### 问题1：看到警告 "Using basic business day calendar"

**原因**：`pandas-market-calendars` 和 `exchange-calendars` 都没有安装

**影响**：只能排除周末，无法排除节假日

**解决方案**：
```bash
pip install pandas-market-calendars
# 或
pip install exchange-calendars
```

### 问题2：缺失数据检测不准确

**检查清单**：
1. 确认交易日历库已安装
2. 检查数据库中的数据格式（日期应为 YYYY-MM-DD 字符串）
3. 运行测试脚本验证功能
4. 查看日志输出

### 问题3：特定节假日未被排除

**可能原因**：
- 使用了基础日历（未安装交易日历库）
- 节假日在不同年份日期不同
- 使用了非NYSE交易所日历

**解决方案**：
```python
# 安装库并指定正确的交易所
pip install pandas-market-calendars

# 使用时明确指定交易所
trading_days = get_trading_days('2024-01-01', '2024-12-31', exchange='NYSE')
```

## 最佳实践

1. **始终安装交易日历库**
   ```bash
   pip install pandas-market-calendars
   ```

2. **使用环境变量配置数据库位置**
   ```bash
   # .env
   DATA_BASE_DIR=/data/trading_db
   ```

3. **定期清理过期缓存**
   ```python
   store = get_data_store()
   store.cleanup_expired_cache()
   ```

4. **监控数据完整性**
   ```python
   # 检查数据统计
   stats = store.get_storage_stats()
   print(f"数据库大小: {stats['total_size_mb']:.2f} MB")
   print(f"价格记录数: {stats['price_records']}")
   ```

5. **使用日志记录**
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   
   # 查看增量更新日志
   manager = get_data_manager()
   data = manager.get_price_data(['AAPL'], '2024-01-01', '2024-12-31')
   ```

## 相关文件

- `src/data/trading_calendar.py` - 交易日历核心模块
- `src/data/data_store.py` - 数据存储与缺失检测
- `src/data/data_fetcher.py` - 数据获取与增量更新
- `test_incremental_update.py` - 功能测试脚本
- `requirements.txt` - 依赖库配置

## 更新日志

### v2.0.0 (当前版本)
- ? 新增真实交易日历支持
- ? 智能增量更新（检测首尾和中间gap）
- ? 自动排除周末和节假日
- ? 统一配置管理（`DATA_BASE_DIR`环境变量）
- ? 完善的测试覆盖

### v1.0.0
- 基于硬编码阈值的gap检测
- CSV文件存储
- 只检测首尾端点

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License

