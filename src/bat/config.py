import os
from dataclasses import dataclass

import torch
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    backend: str
    label: str
    icon: str
    shared_memory: bool = False

    @property
    def is_accelerated(self) -> bool:
        return self.backend != "cpu"

    @property
    def uses_torch_cuda_api(self) -> bool:
        return self.backend in {"cuda", "rocm"}


class Config:
    # API Settings
    API_KEY = os.getenv('BINANCE_API_KEY')
    API_SECRET = os.getenv('BINANCE_API_SECRET')
    VERSION = '1.0.1'

    # Trading Settings
    SYMBOL = 'BTCUSDT'
    INTERVAL = '15m'
    RISK_PROFILE = os.getenv('BAT_RISK_PROFILE', 'STANDARD')  # CONSERVATIVE, STANDARD, AGGRESSIVE

    # Data Settings
    SEQ_LENGTH = 60    # 視窗長度
    FEATURE_COLS = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "RSI",
        "EMA_20",
        "RET_1",
        "VOL_20",
        "VOL_Z",
        "RSI_1H",
        "EMA_20_1H",
    ]

    # Model Settings
    HIDDEN_SIZE = 256
    NUM_LAYERS = 3
    DROPOUT = 0.2
    RETURN_THRESHOLD = float(os.getenv("BAT_RETURN_THRESHOLD", "0.001"))
    RETURN_HORIZON = int(os.getenv("BAT_RETURN_HORIZON", "3"))

    # Training Settings
    EPOCHS = 50
    BATCH_SIZE = 64
    LR = 0.001
    SIMULATION_STEPS = int(os.getenv("BAT_SIM_STEPS", "120"))

    # 交易模式設定
    TRADING_MODE = os.getenv('TRADING_MODE', 'TESTNET').upper()
    USER_NAME = os.getenv('BAT_USER_NAME') or os.getenv('USER') or "User"

    # 根據模式選擇正確的 Key
    if TRADING_MODE == 'REAL':
        API_KEY = os.getenv('BINANCE_API_KEY')
        API_SECRET = os.getenv('BINANCE_API_SECRET')
        IS_TESTNET = False
        print("⚠️ 警告：目前處於 [正式實盤] 模式，將使用真實資金！")
    else:
        API_KEY = os.getenv('TESTNET_API_KEY')
        API_SECRET = os.getenv('TESTNET_API_SECRET')
        IS_TESTNET = True
        print("✅ 目前處於 [測試網] 模式，資金為虛擬資產。")

    @classmethod
    def validate_api_config(cls, is_testnet=None):
        use_testnet = cls.IS_TESTNET if is_testnet is None else is_testnet
        mode = "TESTNET" if use_testnet else "REAL"
        api_key = os.getenv("TESTNET_API_KEY") if use_testnet else os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("TESTNET_API_SECRET") if use_testnet else os.getenv("BINANCE_API_SECRET")
        missing = []
        if not api_key:
            missing.append(f"{mode} API_KEY")
        if not api_secret:
            missing.append(f"{mode} API_SECRET")

        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                f"缺少必要環境變數: {missing_str}。請檢查 .env 與 TRADING_MODE 設定。"
            )

    @classmethod
    def get_testnet_credentials(cls):
        return os.getenv('TESTNET_API_KEY'), os.getenv('TESTNET_API_SECRET')

    @classmethod
    def get_credentials(cls, is_testnet=None):
        use_testnet = cls.IS_TESTNET if is_testnet is None else is_testnet
        if use_testnet:
            return os.getenv('TESTNET_API_KEY'), os.getenv('TESTNET_API_SECRET')
        return os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET')


    @property
    def DEVICE(self):
        return self.DEVICE_INFO.device

    @property
    def DEVICE_INFO(self):
        if not hasattr(self, '_device_info'):
            self._device_info = self._get_optimal_device_info()
        return self._device_info

    @property
    def DEVICE_BACKEND(self):
        return self.DEVICE_INFO.backend

    @property
    def DEVICE_NAME(self):
        return self.DEVICE_INFO.label

    def _get_optimal_device(self):
        return self._get_optimal_device_info().device

    def _get_optimal_device_info(self):
        if torch.cuda.is_available():
            hip_version = getattr(getattr(torch, "version", None), "hip", None)
            if hip_version:
                info = DeviceInfo(torch.device("cuda"), "rocm", "AMD ROCm", "🚀")
            else:
                info = DeviceInfo(torch.device("cuda"), "cuda", "NVIDIA CUDA", "🚀")
            print(f"{info.icon} 使用裝置: {info.label}")
            return info

        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is not None and mps_backend.is_available() and mps_backend.is_built():
            info = DeviceInfo(torch.device("mps"), "mps", "Apple Silicon MPS", "🍎", shared_memory=True)
            print(f"{info.icon} 使用裝置: {info.label}")
            return info

        xpu_backend = getattr(torch, "xpu", None)
        if xpu_backend is not None and xpu_backend.is_available():
            info = DeviceInfo(torch.device("xpu"), "xpu", "Intel XPU", "⚡")
            print(f"{info.icon} 使用裝置: {info.label}")
            return info

        info = DeviceInfo(torch.device("cpu"), "cpu", "CPU", "🖥️")
        print(f"{info.icon} 使用裝置: {info.label}")
        return info

    DEFAULT_ASSETS = ["USDT", "BTC", "BNB", "ETH"]

conf = Config()
