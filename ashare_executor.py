# -*- coding: utf-8 -*-
"""
A股下单执行器（独立于策略，替代原美股 Alpaca 通道）
==================================================
读取 a_share_weights.csv -> 读取账户现金与当前持仓 -> 计算买卖委托（买入按100股整手；卖出按T+1可卖数量全额，不整手取整）
-> 默认 dry-run 预览，确认后再实盘。

实盘用 easytrader 对接券商客户端:
  client_type: 'ths'(同花顺) / 'yh_client'(银河) / 'ht_client'(华泰) ...
实时价: 优先 easyquotation(qq)，akshare 现货兜底（券商对接说明中的取价方式）。

安全默认 dry_run=True —— 不传 --live 绝不真实下单。
"""
import os
import sys
import math
import argparse
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ashare_executor")

LOT = 100  # A股一手=100股


def _market_prefix(code):
    code = str(code).zfill(6)
    if code.startswith("920"):
        return "bj"
    if code.startswith(("5", "6", "9")):
        return "sh"
    if code.startswith(("0", "3")):
        return "sz"
    return "sh"


class PriceSource:
    """实时价：easyquotation(qq) 优先，akshare 现货兜底。"""
    def __init__(self):
        self._eq = None
        self._spot = None
        try:
            import easyquotation
            self._eq = easyquotation.use("qq")
        except Exception as e:
            logger.info(f"easyquotation 不可用，改用 akshare 兜底: {e}")

    def _load_spot(self):
        if self._spot is None:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            df["代码"] = df["代码"].astype(str).str.zfill(6)
            self._spot = df.set_index("代码")["最新价"].to_dict()
        return self._spot

    def get(self, code):
        code = str(code).zfill(6)
        if self._eq is not None:
            try:
                q = self._eq.real([code], prefix=False)
                rec = q.get(code) or q.get(_market_prefix(code) + code)
                if rec and rec.get("now"):
                    return float(rec["now"])
            except Exception as e:
                logger.debug(f"easyquotation 取价失败 {code}: {e}")
        try:
            return float(self._load_spot().get(code))
        except Exception as e:
            logger.warning(f"取价失败 {code}: {e}")
            return None


class AShareExecutor:
    def __init__(self, client_type="ths", dry_run=True, config_path=None, cash_override=None):
        self.dry_run = dry_run
        self.client_type = client_type
        self.price = PriceSource()
        self.user = None
        self.cash_override = cash_override
        if not dry_run and config_path and client_type in ("ths", "yh_client", "ht_client", "xq"):
            import easytrader
            self.user = easytrader.use(client_type)
            self.user.prepare(config_path)
            logger.info(f"easytrader 已连接 {client_type}")

    def connect_broker(self, spec):
        """按 easytrader 约定连接券商。spec: kind=xq|miniqmt|ths 及对应路径。"""
        import easytrader
        kind = (spec or {}).get("kind", "ths")
        if kind == "xq":
            self.user = easytrader.use("xq")
            self.user.prepare(spec["prepare_file"])
        elif kind == "miniqmt":
            self.user = easytrader.use("miniqmt")
            self.user.connect(
                miniqmt_path=spec["miniqmt_path"],
                stock_account=spec["stock_account"],
                trader_callback=None,
            )
        else:
            self.user = easytrader.use("universal_client")
            self.user.connect(spec["ths_path"])
            if hasattr(self.user, "enable_type_keys_for_editor"):
                self.user.enable_type_keys_for_editor()
        self.dry_run = False
        self.client_type = kind
        logger.info(f"券商已连接: {kind}")

    # ---- 账户 ----
    def get_account_cash(self):
        if self.cash_override is not None and (self.dry_run or self.user is None):
            return float(self.cash_override)
        if self.dry_run or self.user is None:
            return float(os.environ.get("ASHARE_DRYRUN_CASH", "1000000"))
        bal = self.user.balance
        rec = bal[0] if isinstance(bal, list) else bal
        for k in ("可用金额", "可用资金", "enable_balance", "available"):
            if k in rec:
                return float(rec[k])
        return 0.0

    def _pick(self, rec, names, default=0):
        for n in names:
            if n in rec and rec[n] not in (None, ""):
                return rec[n]
        return default

    def get_current_positions(self):
        """返回 {code: {'amount':持仓, 'sellable':可卖(T+1)}}。字段兼容同花顺/雪球/miniQMT。"""
        if self.dry_run or self.user is None:
            return {}
        pos = {}
        raw = self.user.position or []
        for p in raw:
            if not isinstance(p, dict):
                continue
            code = str(self._pick(p, ["证券代码", "股票代码", "stock_code", "code", "symbol"], "")).strip()
            code = code[-6:].zfill(6) if code else ""
            if not code.isdigit() or code == "000000":
                continue
            amount = float(self._pick(p, ["股票余额", "当前持仓", "持仓数量", "current_amount", "qty"], 0) or 0)
            sellable = float(self._pick(p, ["可用余额", "可卖数量", "可用数量", "enable_amount", "enable_qty"], amount) or 0)
            pos[code] = {"amount": amount, "sellable": sellable}
        return pos

    # ---- 目标与下单计算 ----
    def load_target_weights(self, path):
        df = pd.read_csv(path)
        if "tic" in df.columns:
            df["symbol"] = df["tic"].astype(str).str.zfill(6)
        elif "gvkey" in df.columns:
            df["symbol"] = df["gvkey"].astype(str).str.zfill(6)
        else:
            raise ValueError("权重文件需含 tic 或 gvkey 列")
        df = df[df["weight"].fillna(0) > 0].copy()
        df["weight"] = df["weight"] / df["weight"].sum()
        return df[["symbol", "weight"]]

    def calculate_orders(self, targets, total_value=None):
        """由目标权重 × 可投资总额计算买卖委托；买入按100股整手，卖出按T+1可卖数量全额出（不整手取整）。"""
        cash = self.get_account_cash()
        positions = self.get_current_positions()
        # 可投资总额 = 现金 + 现有持仓市值
        pos_value = 0.0
        prices = {}
        for code, p in positions.items():
            px = self.price.get(code)
            prices[code] = px
            if px:
                pos_value += px * p["amount"]
        total_value = total_value or (cash + pos_value)

        orders = []
        target_codes = set(targets["symbol"])
        # 1) 目标股票：买入整手取整；卖出=min(需减仓,可卖)，不取整为100整数倍
        for _, row in targets.iterrows():
            code = row["symbol"]
            px = prices.get(code) or self.price.get(code)
            prices[code] = px
            if not px or px <= 0:
                logger.warning(f"无价跳过 {code}")
                continue
            target_val = total_value * row["weight"]
            target_shares = int(target_val // (px * LOT)) * LOT
            held = positions.get(code, {}).get("amount", 0)
            sellable = positions.get(code, {}).get("sellable", 0)
            delta = target_shares - held
            if delta >= LOT:
                orders.append({"symbol": code, "side": "buy", "shares": int(delta), "price": px})
            elif delta < 0:
                # 卖出：按实际可卖数量，不取整为100的整数倍
                sell = int(min(-delta, sellable))
                if sell > 0:
                    orders.append({"symbol": code, "side": "sell", "shares": sell, "price": px})
        # 2) 不在目标内的持仓 -> 按可卖数量全额清仓（不整手取整）
        for code, p in positions.items():
            if code in target_codes:
                continue
            # 清仓：可卖数量全额卖出（允许零股，不做 //LOT*LOT）
            sellable = int(p["sellable"])
            if sellable > 0:
                px = prices.get(code) or self.price.get(code)
                orders.append({"symbol": code, "side": "sell", "shares": sellable, "price": px})
        return orders, total_value

    # ---- 执行 ----
    def execute(self, weights_path, dry_run=None):
        dry_run = self.dry_run if dry_run is None else dry_run
        targets = self.load_target_weights(weights_path)
        orders, total_value = self.calculate_orders(targets)

        print(f"\n可投资总额 ≈ {total_value:,.0f} 元  | 目标 {len(targets)} 只 | 拟下单 {len(orders)} 笔")
        print("-" * 64)
        for o in orders:
            amt = (o["shares"] * o["price"]) if o["price"] else 0
            print(f"  {o['side'].upper():4s} {o['symbol']}  {o['shares']:>7d} 股 @ {o['price']}  ≈ {amt:,.0f} 元")
        print("-" * 64)

        if dry_run:
            print(">> DRY-RUN 预览，未真实下单。确认无误后加 --live 实盘。")
            return orders

        # 实盘：先卖后买
        for o in sorted(orders, key=lambda x: 0 if x["side"] == "sell" else 1):
            try:
                if o["side"] == "buy":
                    self.user.buy(o["symbol"], price=o["price"], amount=o["shares"])
                else:
                    self.user.sell(o["symbol"], price=o["price"], amount=o["shares"])
                logger.info(f"已提交 {o['side']} {o['symbol']} {o['shares']}")
            except Exception as e:
                logger.error(f"下单失败 {o}: {e}")
        return orders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="a_share_weights.csv")
    ap.add_argument("--client", default="ths", help="ths/yh_client/ht_client ...")
    ap.add_argument("--config", default=None, help="easytrader 账户配置json")
    ap.add_argument("--live", action="store_true", help="实盘下单（默认 dry-run）")
    args = ap.parse_args()

    ex = AShareExecutor(client_type=args.client, dry_run=not args.live, config_path=args.config)
    ex.execute(args.weights, dry_run=not args.live)


if __name__ == "__main__":
    main()
