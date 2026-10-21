import json
import os
import hashlib
import secrets
import copy
import time
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

DEFAULT_CONFIG = {
    "server": {"host": "0.0.0.0", "port": 8800},
    "admin": {"password_hash": "", "salt": "", "jwt_secret": ""},
    "tabbit": {
        "base_url": "https://web.tabbit.com",
        "client_id": "2dd8eb4c1ed9c344d173",
    },
    "tokens": [],
    "proxy": {"api_key": "", "system_prompt": ""},
    "claude": {"default_model": "best", "system_prompt": ""},
    "openai": {"default_model": "best"},
    "agent": {
        "cleaner": {"enabled": False},
        "context": {"enabled": False, "max_turns": 20, "threshold_ratio": 0.8},
        "tools": {"enabled": False},
        "router": {"enabled": False},
        "token_pool": {"enabled": False, "cooldown_seconds": 300, "max_consecutive_errors": 3, "encryption_key": ""},
    },
    "logging": {"max_entries": 500},
}

ENV_VAR_MAP = {
    "TABBIT_SERVER_HOST": ("server", "host"),
    "TABBIT_SERVER_PORT": ("server", "port"),
    "TABBIT_BASE_URL": ("tabbit", "base_url"),
    "TABBIT_CLIENT_ID": ("tabbit", "client_id"),
    "TABBIT_API_KEY": ("proxy", "api_key"),
    "TABBIT_SYSTEM_PROMPT": ("proxy", "system_prompt"),
    "TABBIT_CLAUDE_DEFAULT_MODEL": ("claude", "default_model"),
    "TABBIT_CLAUDE_SYSTEM_PROMPT": ("claude", "system_prompt"),
    "TABBIT_OPENAI_DEFAULT_MODEL": ("openai", "default_model"),
}


def _apply_env_overrides(config: dict) -> dict:
    for env_var, keys in ENV_VAR_MAP.items():
        value = os.environ.get(env_var)
        if value is not None:
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            if keys[-1] == "port":
                try:
                    d[keys[-1]] = int(value)
                except ValueError:
                    pass
            else:
                d[keys[-1]] = value

    # 支持通过环境变量直接注入 Token (单个或逗号分隔的多个)
    tabbit_token = os.environ.get("TABBIT_TOKEN")
    if tabbit_token:
        tokens_list = config.setdefault("tokens", [])
        existing_values = {t.get("value") for t in tokens_list if isinstance(t, dict)}
        for idx, tk in enumerate(tabbit_token.split(",")):
            tk = tk.strip()
            if tk and tk not in existing_values:
                token_id = f"env-{idx + 1}"
                tokens_list.append({
                    "id": token_id,
                    "name": f"ENV Token {idx + 1}" if idx > 0 else "Default Token",
                    "value": tk,
                    "enabled": True,
                    "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "last_used_at": None,
                    "total_requests": 0,
                    "error_count": 0,
                    "status": "active",
                })
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256((password + salt).encode()).hexdigest()
    return hashed, salt


class ConfigManager:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else CONFIG_PATH
        self.config = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            config = _deep_merge(copy.deepcopy(DEFAULT_CONFIG), saved)
            config = _apply_env_overrides(config)
            self._save(config)
            return config

        config = copy.deepcopy(DEFAULT_CONFIG)
        config["admin"]["jwt_secret"] = secrets.token_hex(32)
        pw_hash, salt = hash_password("admin")
        config["admin"]["password_hash"] = pw_hash
        config["admin"]["salt"] = salt
        config = _apply_env_overrides(config)
        self._save(config)
        return config

    def _save(self, config: dict | None = None):
        if config is None:
            config = self.config
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def save(self):
        self._save()

    def get(self, *keys, default=None):
        val = self.config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val

    def set_val(self, *keys_and_value):
        """set_val('server', 'port', 8800) — 最后一个参数是值"""
        keys = keys_and_value[:-1]
        value = keys_and_value[-1]
        d = self.config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        self.save()
