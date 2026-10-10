"""
Configuration Settings Module
===========================

Centralized configuration management using Pydantic settings.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import validator
from pydantic_settings import BaseSettings
from pydantic.types import SecretStr
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    url: str = "sqlite:///finrl_trading.db"
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30

    class Config:
        env_prefix = "DB_"


class AlpacaSettings(BaseSettings):
    """Alpaca API configuration settings."""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: str = "https://paper-api.alpaca.markets/v2"
    use_paper_trading: bool = True

    class Config:
        env_prefix = "APCA_"


class WRDSSettings(BaseSettings):
    """WRDS database configuration settings."""
    username: Optional[str] = None
    password: Optional[SecretStr] = None
    hostname: str = "wrds.wharton.upenn.edu"
    port: int = 9737
    database: str = "wrds"

    class Config:
        env_prefix = "WRDS_"


class FMPSettings(BaseSettings):
    """Financial Modeling Prep API configuration settings."""
    api_key: Optional[SecretStr] = None

    class Config:
        env_prefix = "FMP_"


class OpenAISettings(BaseSettings):
    """OpenAI GPT configuration settings."""
    api_key: Optional[SecretStr] = None
    model: str = "gpt-4o-mini"
    request_timeout: int = 30

    class Config:
        env_prefix = "OPENAI_"


class DataSettings(BaseSettings):
    """
    Data management configuration settings.
    
    Attributes:
        base_dir: Main directory for all data storage (including SQLite database).
                  Set via DATA_BASE_DIR environment variable.
        cache_dir: Directory for cached data files
        processed_dir: Directory for processed data files
        raw_dir: Directory for raw data files
        cache_ttl_hours: Time-to-live for cache entries in hours
        max_cache_size_mb: Maximum cache size in megabytes
    """
    base_dir: str = "./data"
    cache_dir: str = "./data/cache"
    processed_dir: str = "./data/processed"
    raw_dir: str = "./data/raw"
    cache_ttl_hours: int = 24
    max_cache_size_mb: int = 1000

    class Config:
        env_prefix = "DATA_"
    
    def get_database_path(self) -> Path:
        """Get the path to the SQLite database file."""
        return Path(self.base_dir) / "finrl_trading.db"


class StrategySettings(BaseSettings):
    """Strategy configuration settings."""
    default_rebalance_freq: str = "Q"
    max_weight_per_stock: float = 0.1
    max_sector_weight: float = 0.3
    max_turnover: float = 0.5
    risk_free_rate: float = 0.02
    benchmark_tickers: List[str] = ["SPY", "QQQ"]

    class Config:
        env_prefix = "STRATEGY_"


class TradingSettings(BaseSettings):
    """Trading configuration settings."""
    max_order_value: float = 100000.0
    max_portfolio_turnover: float = 0.5
    min_order_size: float = 100.0
    risk_checks_enabled: bool = True
    execution_timeout: int = 300
    log_orders: bool = True
    order_log_path: str = "./logs/orders"

    class Config:
        env_prefix = "TRADING_"


class WebSettings(BaseSettings):
    """Web interface configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8501
    debug: bool = False
    theme: str = "light"

    class Config:
        env_prefix = "WEB_"


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = "./logs/finrl_trading.log"
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5

    class Config:
        env_prefix = "LOG_"


class FinRLSettings(BaseSettings):
    """Main FinRL configuration settings."""
    app_name: str = "FinRL Trading"
    version: str = "2.0.0"
    environment: str = "development"

    # Sub-settings
    database: DatabaseSettings = DatabaseSettings()
    alpaca: AlpacaSettings = AlpacaSettings()
    wrds: WRDSSettings = WRDSSettings()
    fmp: FMPSettings = FMPSettings()
    openai: OpenAISettings = OpenAISettings()
    data: DataSettings = DataSettings()
    strategy: StrategySettings = StrategySettings()
    trading: TradingSettings = TradingSettings()
    web: WebSettings = WebSettings()
    logging: LoggingSettings = LoggingSettings()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # 忽略未定义的环境变量

    @validator('environment')
    def validate_environment(cls, v):
        if v not in ['development', 'testing', 'staging', 'production']:
            raise ValueError('Environment must be one of: development, testing, staging, production')
        return v

    def is_development(self) -> bool:
        return self.environment == 'development'

    def is_production(self) -> bool:
        return self.environment == 'production'

    def get_data_dir(self) -> Path:
        """Get data directory path."""
        return Path(self.data.base_dir)

    def get_cache_dir(self) -> Path:
        """Get cache directory path."""
        return Path(self.data.cache_dir)

    def get_processed_dir(self) -> Path:
        """Get processed data directory path."""
        return Path(self.data.processed_dir)

    def get_log_dir(self) -> Path:
        """Get log directory path."""
        if self.logging.file_path:
            return Path(self.logging.file_path).parent
        return Path("./logs")
    
    def get_database_path(self) -> Path:
        """Get the path to the SQLite database file."""
        return self.data.get_database_path()


# Global settings instance
_settings: Optional[FinRLSettings] = None


def get_config() -> FinRLSettings:
    """Get global configuration instance."""
    global _settings
    if _settings is None:
        _settings = FinRLSettings()
    return _settings


def reload_config() -> FinRLSettings:
    """Reload configuration from environment."""
    global _settings
    _settings = FinRLSettings()
    return _settings


def create_env_file(template_path: Optional[str] = None) -> str:
    """
    Create a .env template file with all available configuration options.

    Args:
        template_path: Path to save the template file

    Returns:
        Path to the created template file
    """
    if template_path is None:
        template_path = ".env.template"

    template_content = """# FinRL Trading Configuration Template
# Copy this file to .env and fill in your values

# Application Settings
APP_NAME=FinRL Trading
VERSION=2.0.0
ENVIRONMENT=development

# Database Settings
DB_URL=sqlite:///finrl_trading.db
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
DB_POOL_TIMEOUT=30

# Alpaca API Settings
APCA_API_KEY=your_alpaca_api_key_here
APCA_API_SECRET=your_alpaca_secret_here
APCA_BASE_URL=https://paper-api.alpaca.markets
APCA_USE_PAPER_TRADING=true

# WRDS Database Settings
WRDS_USERNAME=your_wrds_username
WRDS_PASSWORD=your_wrds_password
WRDS_HOSTNAME=wrds.wharton.upenn.edu
WRDS_PORT=9737
WRDS_DATABASE=wrds

# Financial Modeling Prep API Settings
FMP_API_KEY=your_fmp_api_key_here

# OpenAI GPT Settings
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_REQUEST_TIMEOUT=30

# Data Management Settings
# DATA_BASE_DIR: Directory where the SQLite database and all data files are stored
# This is the main data directory used by DataStore for persistent storage
DATA_BASE_DIR=./data
DATA_CACHE_DIR=./data/cache
DATA_PROCESSED_DIR=./data/processed
DATA_RAW_DIR=./data/raw
DATA_CACHE_TTL_HOURS=24
DATA_MAX_CACHE_SIZE_MB=1000

# Strategy Settings
STRATEGY_DEFAULT_REBALANCE_FREQ=Q
STRATEGY_MAX_WEIGHT_PER_STOCK=0.1
STRATEGY_MAX_SECTOR_WEIGHT=0.3
STRATEGY_MAX_TURNOVER=0.5
STRATEGY_RISK_FREE_RATE=0.02
STRATEGY_BENCHMARK_TICKERS=SPY,QQQ

# Trading Settings
TRADING_MAX_ORDER_VALUE=100000.0
TRADING_MAX_PORTFOLIO_TURNOVER=0.5
TRADING_MIN_ORDER_SIZE=100.0
TRADING_RISK_CHECKS_ENABLED=true
TRADING_EXECUTION_TIMEOUT=300
TRADING_LOG_ORDERS=true
TRADING_ORDER_LOG_PATH=./logs/orders

# Web Interface Settings
WEB_HOST=0.0.0.0
WEB_PORT=8501
WEB_DEBUG=false
WEB_THEME=light

# Logging Settings
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOG_FILE_PATH=./logs/finrl_trading.log
LOG_MAX_FILE_SIZE=10485760
LOG_BACKUP_COUNT=5
"""

    with open(template_path, 'w') as f:
        f.write(template_content)

    return template_path


def validate_config() -> List[str]:
    """
    Validate current configuration and return list of issues.

    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    config = get_config()

    # Check Alpaca configuration
    if not config.alpaca.api_key or not config.alpaca.api_secret:
        issues.append("Alpaca API credentials not configured")

    # Check data directories
    data_dir = config.get_data_dir()
    if not data_dir.exists():
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create data directory: {e}")

    # Check log directory
    log_dir = config.get_log_dir()
    if config.logging.file_path and not log_dir.exists():
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create log directory: {e}")

    return issues


if __name__ == "__main__":
    # Example usage
    config = get_config()
    print(f"Application: {config.app_name} v{config.version}")
    print(f"Environment: {config.environment}")
    print(f"Data directory: {config.get_data_dir()}")

    # Validate configuration
    issues = validate_config()
    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration is valid")

    # Create environment template
    template_path = create_env_file()
    print(f"Environment template created: {template_path}")
