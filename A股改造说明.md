# FinRL-Trading 美股通道 → A股改造说明

> 目标：把 `D:\FinRL-Trading` 的交易通道从美股改成 A 股，**策略代码一行不改**，数据采用通达信本地行情并叠加 akshare 多源补全。不偷工减料。

---

## 一、总体思路

原项目策略层返回 `StrategyResult(weights: DataFrame)`，美股专用部分集中在**数据获取**（yfinance/FMP）、**日历**（NYSE / pandas_market_calendars）、**标的 id**（gvkey/Compustat）、**下单通道**（Alpaca）。

改造不动策略，只做三件事，用**三个新文件**把 A 股数据“喂进”策略、把权重“接出来”、再独立下单：

```
A股日线+基本面 ──[ashare_data_adapter.py]──▶ {fundamentals, prices}
                                               │
                                     策略 generate_weights()   ← 代码未改
                                               │
                              result.weights ──[run_ashare_weights.py]──▶ a_share_weights.csv
                                               │
                           a_share_weights.csv ──[ashare_executor.py]──▶ easytrader 下单（默认 dry-run）
```

---

## 二、新增文件（都是新建，未改动任何原有策略/框架代码）

### 1. `src/data/ashare_data_adapter.py` —— A股数据适配器
把 A 股原始数据转成 FinRL 策略要的两张表，并自算估值与标签。

- **历史价（通达信本地日线）**：直接读 `<tdx_dir>\vipdoc\{sh,sz,bj}\lday\*.day`（`tdx_dir` 可配）（32 字节定长记录），内联通达信 `.day` 读取逻辑（避免外部编译依赖 / DLL 加载失败）。产出长表 `['date','tic','close','open','high','low','volume']`。
  - 代码→市场前缀规则：688/605/603/601/600/689… → sh；301/300/003/002/001/000 → sz；430/830…/920 → bj。
- **基本面（akshare 多源融合）**：
  - **主源**：新浪 `ak.stock_financial_analysis_indicator`（实测 86 个字段 × ~34 个季频报告期，2018→今，最全）。
  - **兜底/交叉补全**：东财 `stock_financial_analysis_indicator_em`（本机 akshare 1.12.98 无此接口，自动跳过）、同花顺 `stock_financial_abstract_ths`，按 `datadate` 对齐**补缺失列**。
  - `SINA_RENAME` 把约 65 个中文字段映射成干净英文（EPS/BPS/每股经营现金流/各类利润率/ROE/周转率/流动速动现金比率/资产负债率/产权比率/增长率…）。
  - 所有数值列统一 `_to_num` 强转（处理 `,`、`--`、空值），保证可作特征、可写 parquet。
- **估值自算（通达信季末价 × 财报）**：
  - `adj_close_q` = 报告期季末的通达信价；`EPS_ttm` 用新浪 YTD 累计值“累计差分还原单季 → 滚动 4 季”求 TTM。
  - `pe = 价/EPS_ttm`、`pb = 价/BPS`、`peg = pe/净利润增长率`、`ps = 价/(EPS_ttm/销售净利率)`。
  - **股息率 = 最近一次真实每股现金分红 / 价 × 100**：用东财 `stock_fhps_detail_em` 的“现金分红比例（每 10 股派 X 元）÷ 10”得每股派息，`_fetch_dps` 只保留**真实有派息**的报告期，取 datadate 当年及以前最近一次派息。
- **标签**：`y_return = ln(下一报告期 adj_close_q / 本期 adj_close_q)`，按 `gvkey` 分组。
- **行业**：`get_industry_map` 用 akshare 行业板块成分做 code→行业（`--no-industry` 可跳过加速；base ML 不需要）。
- **顶层 `build_dataset(...)`**：组装 `{'fundamentals','prices'}`，特征 NaN 默认中位数填充（与 ML 的 `dropna(X)` 兼容），支持 `cache_path` 落 parquet 缓存（二次运行秒级读缓存）。
- **契约要点**：ML 策略把 `fundamentals` 里**除 id/日期/标签外的所有数值列**都当特征，`gvkey` 直接用 6 位代码即可，无需硬凑美股 55 列。

### 2. `run_ashare_weights.py`（仓库根）—— 跑策略导出权重
- 构建数据集 → 调 `strategy.generate_weights(...)` → 落盘 `a_share_weights.csv`。
- 支持 `--strategy ml|equal|minvar`、`--universe 文件|--codes`、`--weight-method min_variance|equal`、`--prediction-mode single|rolling`、`--cache`、`--no-industry`、`--min-weight`。
- **导出清洗**：统一 tic 为 6 位、**去重**（同一标的留权重最大的一行）、**滤掉接近 0 的噪声仓**（min_variance 会给出 ~1e-19 数值零，`--min-weight` 默认 0.01%）、**重新归一**、按权重排序。
- **不改策略代码的两处绕法**：
  - 精简版 `BaseStrategy` 没建 `self.logger` 而 `ml_strategy` 到处用它 → 在 run 脚本里给策略对象**注入** `logger`。
  - ML 策略类名是 `MLStockSelectionStrategy`（不是 Selector）。

### 3. `ashare_executor.py`（仓库根）—— A股下单执行器（替代 Alpaca）
- 读 `a_share_weights.csv` + 账户现金 + 当前持仓 → 生成**整手（100 股）**订单 → **默认 dry-run 预览**，不加 `--live` 绝不真实下单。
- 实时价：`easyquotation(qq)` 优先，akshare 现货 `stock_zh_a_spot_em` 兜底（优先 easyquotation、akshare 兜底）。
- A 股规则：100 股整手向下取整、**T+1 可卖校验**（只卖 enable_amount）、不在目标里的持仓清仓、实盘先卖后买。
- 券商对接 `easytrader`：`--client ths|yh_client|ht_client`。dry-run 无需登录，用 `ASHARE_DRYRUN_CASH`（默认 100 万）+ 空持仓模拟。

---

## 三、踩坑与修复

| 问题 | 处理 |
|---|---|
| `stock_financial_analysis_indicator_em` 在本机 akshare 1.12.98 不存在 | 改以新浪 `stock_financial_analysis_indicator`（86 字段）为主，EM 有则补 |
| `import stock_codes` 触发编译依赖 DLL 加载失败 | 把 TDX 纯 Python 读取逻辑内联进 adapter |
| akshare 抓取偶发 `Connection aborted 10053` 限速 | 加 `_retry`（指数退避）+ 逐只 0.35~0.4s sleep |
| `ModuleNotFoundError: pandas_market_calendars`（策略→data_fetcher 间接 import 美股日历）| `pip install pandas_market_calendars` |
| 精简 `BaseStrategy` 无 `self.logger` | run 脚本注入 logger（不改策略） |
| parquet 写缓存报 object 列转换失败 | EM/THS 列统一数值强转 + 写前兜底强转 |
| 导出 CSV 有重复行 + 一堆 ~1e-19 噪声权重 | run 脚本加去重 + 滤微仓 + 重新归一 |
| LightGBM/XGBoost 卡慢源/清华源 403 | 换**阿里云源** `--only-binary :all:`（Python 3.10）装好 |
| 股息率偏小 ~0.0002%（新浪“股息发放率”多为空）| 改用东财真实现金分红算；`_fetch_dps` 只取有派息报告期、取最近一次派息 |

---

## 四、怎么跑

```powershell
# 1) 跑策略出权重（沪深300，ML + 最小方差，用缓存）
cd D:\FinRL-Trading
python run_ashare_weights.py --strategy ml --universe hs300.txt --no-industry `
  --cache .\data\cache_hs300 --start-year 2018 --weight-method min_variance `
  --out a_share_weights.csv

# 2) 下单预览（默认 dry-run，不真实下单）
$env:ASHARE_DRYRUN_CASH='1000000'
python ashare_executor.py --weights a_share_weights.csv

# 3) 确认无误后实盘（需券商客户端 + 配置）
python ashare_executor.py --weights a_share_weights.csv --client ths --config account.json --live
```

通达信根目录：`user_config.json` 的 `TDX_CONFIG.tdx_dir`，或环境变量 `TDX_DIR`，或 A 股页「通达信目录」/ CLI `--tdx-dir`；文档默认值 `D:\dgzq`。解析见 `src/config/ashare_tdx.py`。

---

## 五、当前结果（沪深300 全量跑通）

- 数据：300 只成分股全量抓取，`fundamentals` 9937 行 × 115 特征，2018→2026，已落盘缓存 `data\cache_hs300.*.parquet`（二次运行秒级）。
- 模型：sklearn 随机森林/梯度提升 + LightGBM + XGBoost 集成。
- 组合：ML 选股 + 最小方差定权，**55 只持仓（权重和 1.0）**，偏银行/电力/低波动板块。
- dry-run：100 万模拟资金 → 55 只目标生成 44 笔整手买单（11 只权重太小不够一手被自然跳过）。
- 股息率校验：茅台约 3.6–4.4%，全市场中位数 1.6%。

## 六、后续可做
- 放量到全市场（去掉沪深300限制，逐只抓完落缓存即可）。
- 极少数科创板（688 部分）分红接口偶发返回空，已用中位数兜底；如需更准可再加一路分红源。
- 行业中性策略 `SectorNeutralMLStrategy` 需要 `gsector`，届时去掉 `--no-industry` 即可带行业。

---

## 六、A股页策略与财联社情绪（本轮接入）

### 6.1 README 三种用例预设（界面单选）

| 界面标签 | 对应 README | 默认策略族 |
|---|---|---|
| 用例1·组合配置范式 | Use Case 1 — Portfolio Allocation Paradigms | equal_weight（可改 min_variance / mean_variance / drl） |
| 用例2·滚动选股+DRL | Use Case 2 — Rolling Stock Selection + DRL | ml_then_drl（rolling ML → DRL 定权） |
| 用例3·自适应多资产轮动 | Use Case 3 — Adaptive Multi-Asset Rotation | adaptive_rotation（A股代理指数 + 行业四组） |

### 6.2 策略族与参数

- **equal_weight**：原 `EqualWeightStrategy`
- **min_variance / mean_variance**：经典组合（全池价格，scipy）
- **ml**：`MLStockSelectionStrategy`，参数：prediction_mode / weight_method / train_quarters / test_quarters / top_quantile / confirm_mode / lookback_periods
- **sector_neutral_ml**：`SectorNeutralMLStrategy`，自动 `with_industry=True`，缓存前缀加 `_ind`
- **ml_bucket**：A股无美股 GICS 原桶数据；按行业中性路径落地
- **drl**：`ashare_drl_runner`（优先 stable-baselines3，否则 torch REINFORCE）；算法 A2C/PPO/DDPG
- **ml_then_drl**：用例2
- **adaptive_rotation**：写出 `AdaptiveRotationConf_ashare.yaml` + 日线 CSV（`^GSPC/^VIX/QQQ` 为 A股代理），调用原引擎
- **custom**：加载用户 `.py` 中 `BaseStrategy` 子类（示例：`examples/ashare_custom_strategy_example.py`）
- **market_cap**：通达信 gbbq 股本（万股）× 收盘价 → 总市值/流通市值定权；UI 可选口径

### 6.3 财联社新闻情绪（对齐原项目：按股标注）

用法与原项目 fetch_news(ticker, start, end, analyze_sentiment=True) 一致：**按股票代码**拉新闻，给每条新闻标 positive/neutral/negative；**不写入基本面、不微调权重**。新闻源用财联社电报替代 FMP（NEWStodo 仅有 roll/nodeapi，无按股官方接口；实现为拉电报后按请求股的证券全称/代码过滤）。

配置键（user_config.json）：

`json
"CLS_CONFIG": {
  "page_size": 50,
  "max_pages": 5,
  "openai_model": "gpt-4o-mini",
  "cache_dir": "data/cls_news"
}
`

- API：fetch_cls_news(ticker, start_date, end_date, analyze_sentiment=...)；批量 fetch_cls_news_many
- 标注：有 OPENAI_API_KEY 用 GPT（对齐 _annotate_sentiment）；否则中文词典兜底
- 界面：股票代码列表 + 日期区间 +「拉取新闻并分析情绪」；结果表含 ticker/time/title/sentiment
- **不走 FMP**；已移除权重倾斜 / news_sentiment_score 特征注入


### 6.4 新增依赖（用户自行安装）

```powershell
python -m pip install "PyYAML>=6.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
# 可选 DRL SB3 路径：
python -m pip install "stable-baselines3>=2.0.0" "gymnasium>=0.29.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 6.5 市值加权（通达信 gbbq）

- 股本文件：<tdx_dir>/T0002/hq_cache/gbbq（	dx_dir 可配置）
- 实现：src/data/tdx_gbbq.py；事件缓存 data/cache_gbbq/events_m<mtime>.parquet
- 单位：gbbq 股本类记录为**万股**；市值(元) = close × 万股 × 10000
- 依赖：pytdx。安装：
`powershell
python -m pip install "pytdx>=1.72" -i https://pypi.tuna.tsinghua.edu.cn/simple
`


## 主界面子页 A 股化（app.py）

已将侧边栏除「A股实盘」外的页面全部改为 A 股口径（中文标签），实现集中在 `src/web/ashare_pages.py`；`src/web/app.py` 只负责导航与调用。「A股实盘」仍走 `ashare_dashboard.show_ashare_live`，本轮未改其财联社逻辑。

| 原英文页 | 现中文页 | 改动要点 |
|---------|---------|---------|
| Overview | 总览 | 展示沪深300/全部A股股票池、通达信目录、本地缓存体积、最近权重；示意净值用缓存行情×权重（人民币） |
| Data Management | 数据管理 | 管理 hs300.txt / all_a.txt 与 cache_hs300、cache_all_a；可重建数据集（通达信+akshare）；停用 WRDS/S&P500/FMP 拉取 |
| Strategy Backtesting | 策略回测 | 策略族对接 `ashare_strategy_runner`（等权/最小方差/均值方差/市值加权/ML/行业中性/分桶/DRL/自适应/自定义）；简化单期权重持有回测，结果写入 `data/backtest_last_*` |
| Portfolio Analysis | 绩效分析 | 读回测净值/权重；收益、回撤、持仓、相对缓存等权基准；无结果时仍可看最近权重文件 |
| Settings | 设置 | 通达信目录、券商路径、CLS_CONFIG、OPENAI_API_KEY（不回显密钥）；不再以 Alpaca 为主配置 |

侧边栏快捷状态改为 A 股股票池数量与缓存/通达信状态。美股模块（alpaca_manager 等）保持软导入失败。

剩余美股痕迹：`app.py` 仍可选导入 `alpaca_manager` / `trade_executor`（软失败，供旧代码兼容）；数据管理页用说明文字标明 WRDS/FMP 已停用。实盘下单入口只在「A股实盘」。


## FinRL滚动DRL（全面复制 US 路径）

- 新模块：`src/strategies/ashare_finrl_env.py`（StockPortfolioEnv + DRLAgent）、`src/strategies/ashare_finrl_drl.py`（滚动编排）
- 策略族：`finrl_rolling_drl`（用例4）；短训仍为 `drl` / `ashare_drl_runner`
- 产物目录：`artifacts/ashare_drl/`（models / results / progress.json / latest_weights.csv）
- 美股脚本 `rl_model.py`、`fundamental_portfolio_drl.py` 保持可运行，未删改主逻辑
- 与 US 差异：再平衡日期由 A 股交易日历季/月末生成（不再依赖 `all_stocks_info.pickle` / `sp500_*.csv`）；技术指标默认本地实现（不强制 stockstats/finrl）；TD3/SAC 已默认可选启用

安装见 README「FinRL 滚动 DRL」一节。
