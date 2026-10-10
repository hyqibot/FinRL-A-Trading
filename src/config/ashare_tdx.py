# -*- coding: utf-8 -*-
"""
A-share TDX (Tongdaxin) root directory config.

user_config.json shape:
  {
    "TDX_CONFIG": {
      "tdx_dir": "D:\\dgzq"
    }
  }

Resolution order for get_tdx_dir(override=...):
  1) explicit override argument (UI / CLI for this run)
  2) env var TDX_DIR
  3) project-root user_config.json -> TDX_CONFIG.tdx_dir
  4) documented default DEFAULT_TDX_DIR ("D:\\dgzq")

Keys / env (document for operators):
  - JSON key: TDX_CONFIG.tdx_dir   (file: <repo>/user_config.json)
  - Env:      TDX_DIR
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("ashare_tdx")

# Documented default only.
# Prefer env / user_config.json / UI override over relying on this.
DEFAULT_TDX_DIR = r"D:\dgzq"
USER_CONFIG_NAME = "user_config.json"
TDX_CONFIG_KEY = "TDX_CONFIG"
TDX_DIR_KEY = "tdx_dir"
ENV_TDX_DIR = "TDX_DIR"

# Dataset build window (fundamentals / prices start year)
DATA_CONFIG_KEY = "DATA_CONFIG"
START_YEAR_KEY = "start_year"
DEFAULT_START_YEAR = 2018
ENV_START_YEAR = "ASHARE_START_YEAR"
MIN_START_YEAR = 2005
MAX_START_YEAR = 2026


def project_root() -> Path:
    # src/config/ashare_tdx.py -> repo root
    return Path(__file__).resolve().parents[2]


def user_config_path() -> Path:
    return project_root() / USER_CONFIG_NAME


def _read_user_config() -> Dict[str, Any]:
    path = user_config_path()
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        # Allow // comments in JSON; strip naive line comments for tolerance
        lines = []
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("//"):
                continue
            if "//" in line:
                before = line.split("//", 1)[0]
                if "'" not in before and '"' not in before:
                    line = before.rstrip()
            lines.append(line)
        cleaned = "\n".join(lines)
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("failed to read %s: %s", path, e)
        return {}


def _tdx_dir_from_user_config() -> Optional[str]:
    data = _read_user_config()
    block = data.get(TDX_CONFIG_KEY)
    if isinstance(block, dict):
        val = block.get(TDX_DIR_KEY)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def get_tdx_dir(override: Optional[str] = None) -> str:
    """Resolve Tongdaxin install root (contains vipdoc/). See module docstring for order."""
    if override is not None and str(override).strip():
        return str(override).strip()
    env = os.environ.get(ENV_TDX_DIR)
    if env is not None and str(env).strip():
        return str(env).strip()
    from_file = _tdx_dir_from_user_config()
    if from_file:
        return from_file
    return DEFAULT_TDX_DIR


def save_tdx_dir(tdx_dir: str) -> Path:
    """Persist TDX_CONFIG.tdx_dir into project-root user_config.json (merge, keep other keys)."""
    path = user_config_path()
    data = _read_user_config()
    block = data.get(TDX_CONFIG_KEY)
    if not isinstance(block, dict):
        block = {}
    block[TDX_DIR_KEY] = str(tdx_dir).strip()
    data[TDX_CONFIG_KEY] = block
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    logger.info("saved %s.%s=%s -> %s", TDX_CONFIG_KEY, TDX_DIR_KEY, tdx_dir, path)
    return path




def _clamp_start_year(year: int) -> int:
    try:
        y = int(year)
    except Exception:
        return DEFAULT_START_YEAR
    if y < MIN_START_YEAR:
        return MIN_START_YEAR
    if y > MAX_START_YEAR:
        return MAX_START_YEAR
    return y


def get_start_year(override=None) -> int:
    """Resolve fundamentals/dataset start year.

    Order: explicit override -> env ASHARE_START_YEAR -> user_config DATA_CONFIG.start_year -> 2018.
    """
    if override is not None and str(override).strip() != "":
        return _clamp_start_year(override)
    env = os.environ.get(ENV_START_YEAR)
    if env is not None and str(env).strip() != "":
        return _clamp_start_year(env)
    data = _read_user_config()
    block = data.get(DATA_CONFIG_KEY)
    if isinstance(block, dict) and block.get(START_YEAR_KEY) is not None:
        return _clamp_start_year(block.get(START_YEAR_KEY))
    return DEFAULT_START_YEAR


def save_start_year(start_year: int) -> Path:
    """Persist DATA_CONFIG.start_year into user_config.json (merge)."""
    path = user_config_path()
    data = _read_user_config()
    block = data.get(DATA_CONFIG_KEY)
    if not isinstance(block, dict):
        block = {}
    y = _clamp_start_year(start_year)
    block[START_YEAR_KEY] = y
    data[DATA_CONFIG_KEY] = block
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    logger.info("saved %s.%s=%s -> %s", DATA_CONFIG_KEY, START_YEAR_KEY, y, path)
    return path

def validate_tdx_dir(tdx_dir: Optional[str] = None, *, require_vipdoc: bool = True) -> str:
    """
    Resolve and validate. Raises FileNotFoundError with a clear setup hint if missing.
    Does not invent alternate disk paths.
    """
    resolved = get_tdx_dir(tdx_dir)
    root = Path(resolved)
    if not root.is_dir():
        raise FileNotFoundError(
            "通达信目录不存在: {p}\n"
            "请在 A 股页填写「通达信目录」，或设置环境变量 TDX_DIR，"
            "或在项目根目录 user_config.json 写入 "
            '{{"TDX_CONFIG": {{"tdx_dir": "你的通达信安装目录"}}}} '
            "(键名 TDX_CONFIG.tdx_dir；文档默认值为 {d})".format(p=resolved, d=DEFAULT_TDX_DIR)
        )
    if require_vipdoc:
        vipdoc = root / "vipdoc"
        if not vipdoc.is_dir():
            raise FileNotFoundError(
                "通达信目录下未找到 vipdoc: {v}\n"
                "请确认「通达信目录」指向通达信安装根目录（其下应有 vipdoc\\sh|sz|bj\\lday），"
                "当前解析值为: {p}".format(v=str(vipdoc), p=resolved)
            )
    return str(root)


def resolve_gbbq_path(tdx_dir: Optional[str] = None) -> Path:
    """Return <tdx_dir>/T0002/hq_cache/gbbq; raise FileNotFoundError if missing."""
    root = Path(validate_tdx_dir(tdx_dir, require_vipdoc=False))
    path = root / "T0002" / "hq_cache" / "gbbq"
    if not path.is_file():
        raise FileNotFoundError(
            "gbbq not found: {p} (tdx root={r})".format(p=str(path), r=str(root))
        )
    return path
