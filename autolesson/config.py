from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Optional


class ConfigError(Exception):
    pass


DEFAULTS: dict[str, Any] = {
    "window": {
        "focus": True,
        "title_keywords": ["课程学习", "智慧树", "zhihuishu", "stuStudy"],
        "settle_seconds": 3.0,
    },
    "layout": {
        "auto_calibrate": True,
        "panel_left": 0,
        "panel_top": 0,
        "panel_left_ratio": 0.80,
        "panel_top_ratio": 0.30,
        "panel_right_margin": 0,
    },
    "panel": {
        "scroll_step": -5,
        "scroll_wait": 0.8,
        "max_scrolls": 40,
        "top_scroll_clicks": 18,
        "row_title_gap": 2.5,
        "min_title_chars": 2,
        "ocr_upscale": 0.0,
        "right_icon_width": 62,
    },
    "ocr": {
        # 限制 ONNX Runtime 线程，避免单次 OCR 占满所有 CPU 核心
        "intra_threads": 1,
        "inter_threads": 1,
        # OpenCV 内部线程数；截图/形态学/连通域都是小任务，单线程足够
        "cv_threads": 1,
        # 检测网络最小边长；736 会把小图大幅上采样，416 已足够读清界面文字
        "det_limit_side": 416,
        # 网课界面文字均为水平方向，关闭方向分类器可省一部分 CPU
        "use_cls": False,
    },
    "player": {
        "poll_seconds": 4.0,
        "controls_wiggle": True,
        "pause_grace_seconds": 12.0,
        "stall_timeout_seconds": 45.0,
        "max_recoveries": 3,
        "no_bar_timeout_seconds": 60.0,
        # 排除屏幕底部任务栏的高度比例，避免任务栏被误判成进度条
        "bottom_margin_ratio": 0.05,
        "completion_ratio": 0.995,
        "completion_grace_seconds": 6.0,
        "network_wait_seconds": 120.0,
        "expected_total_tolerance": 180,
        "breadcrumb_check_seconds": 90.0,
    },
    "practice": {
        # 自动练习：始终截图调用多模态模型
        "answer_mode": "image",
        "max_tests": 0,
        "question_wait": 0.8,
        "open_wait": 2.0,
        "page_timeout": 30.0,
        "default_questions": 10,
    },
    "quiz": {
        "enabled": True,
        "answer_mode": "auto",  # auto | text | image
        "max_scrolls": 8,
        "scroll_step": -4,
        "scroll_wait": 0.7,
        "close_wait": 2.0,
        "image_fallback_text": True,
    },
    "llm": {
        # 关闭或不填 API Key 时进入无 LLM 模式：只允许 run 视频模式，
        # 弹题随机选择；auto/practice 会被禁用。
        "enabled": True,
        # DeepSeek V4.1 Flash（API 模型名 deepseek-flash）原生支持文字 + 图片
        "base_url": "${LLM_BASE_URL:-https://api.deepseek.com}",
        "api_key": "",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model": "${LLM_MODEL:-deepseek-flash}",
        # 留空表示与 model 相同，适合原生多模态模型
        "vision_model": "${LLM_VISION_MODEL:-}",
        "temperature": 0.0,
        "max_tokens": 400,
        "timeout": 90,
        "max_retries": 2,
        # 大模型返回空内容/无法解析时，最多再重试几次
        "empty_retries": 2,
        "empty_retry_wait": 1.0,
        # ⚠️ 风险兜底：重试仍失败时随机选一个选项（可能答错）。
        # 视频弹题默认开启：LLM 连续失败后随机选一个可见选项，不中断观看。
        # 练习模式不使用随机兜底；练习失败只跳过，不选择。
        "fallback_random_enabled": True,
        "cache": True,
        # DeepSeek 原生 JSON 输出；接口不支持时会自动去掉后重试
        "json_mode": True,
        # 弹题只求快速答案，默认关闭 thinking；可设为 true 开启
        "thinking": False,
        "image_max_width": 1100,
        "max_images": 8,
        "image_jpeg_quality": 82,
    },
    "overlay": {
        "card_threshold": 225,
        "card_close_kernel": 15,
        "max_cards": 3,
        # 启动时收起 AI 助教弹窗/悬浮球，并把竖直标签拖到上方
        "ai_assistant_cleanup": True,
    },
    "notify": {
        "desktop": True,
        "sound": True,
        "webhook_url": "${AL_NOTIFY_WEBHOOK:-}",
        "webhook_type": "generic",  # generic | serverchan | telegram
        "telegram_bot_token": "${AL_TELEGRAM_BOT_TOKEN:-}",
        "telegram_chat_id": "${AL_TELEGRAM_CHAT_ID:-}",
        "dedup_seconds": 60,
    },
    "runtime": {
        "state_file": "runtime/state.json",
        "log_dir": "runtime/logs",
        "shot_dir": "runtime/shots",
        "dry_run": False,
        "max_lessons": 0,
        "max_minutes": 0,
        # 同一视频连续失败几次后停止；2 表示失败后立即重试一次
        "max_lesson_attempts": 2,
        # 全局停止热键（Windows 原生 RegisterHotKey）：F1~F12 / ESC / PAUSE
        "stop_hotkey": "F9",
        # 全局暂停/继续热键；与停止热键或额外停止键冲突时会自动禁用暂停键
        "pause_hotkey": "F8",
        # 额外停止键；ESC 用低级键盘钩子监听且不拦截浏览器按键
        "extra_stop_hotkeys": ["ESC"],
        "monitor": 1,
    },
    "safety": {
        "fail_safe_corner": True,
        "min_action_interval": 0.05,
    },
}


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default if default is not None else "")

    return _ENV_RE.sub(repl, value)


def _expand_tree(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _expand_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_tree(v) for v in node]
    if isinstance(node, str):
        return _expand_env(node)
    return node


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Config:
    """支持属性访问的嵌套配置。``cfg.player.poll_seconds``。"""

    def __init__(self, data: dict[str, Any], base_dir: Path, source_path: Optional[Path] = None):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "base_dir", Path(base_dir))
        object.__setattr__(self, "source_path", source_path)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        data = object.__getattribute__(self, "_data")
        if name not in data:
            raise AttributeError(f"缺少配置项: {name}")
        value = data[name]
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value, self.base_dir, self.source_path)
            data[name] = value
        return value

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            try:
                node = getattr(node, part)
            except AttributeError:
                return default
        return node

    def to_dict(self) -> dict[str, Any]:
        def convert(node: Any) -> Any:
            if isinstance(node, Config):
                return {k: convert(v) for k, v in node._data.items()}
            if isinstance(node, dict):
                return {k: convert(v) for k, v in node.items()}
            if isinstance(node, list):
                return [convert(v) for v in node]
            return node

        return convert(self)

    def path(self, value: str) -> Path:
        p = Path(os.path.expanduser(str(value)))
        if not p.is_absolute():
            p = self.base_dir / p
        return p

    def llm_ready(self) -> bool:
        """有可用 API 且未显式关闭时，才允许列表/练习和 LLM 答题。"""
        if not bool(getattr(self.llm, "enabled", True)):
            return False
        return bool(self.resolved_api_key())

    def resolved_api_key(self) -> str:
        key = str(self.llm.api_key or "").strip()
        if key:
            return key
        env_names = [str(self.llm.api_key_env or "").strip(), "LLM_API_KEY", "OPENAI_API_KEY"]
        for env_name in env_names:
            if env_name:
                value = os.environ.get(env_name, "").strip()
                if value:
                    return value
        return ""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - 正常安装不会发生
        raise ConfigError("缺少 PyYAML，请先 pip install -r requirements.txt") from exc
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except Exception as exc:
        raise ConfigError(f"读取配置文件失败 {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件必须是 YAML 字典: {path}")
    return data


def load_config(path: Optional[str | Path] = None, overrides: Optional[dict[str, Any]] = None) -> Config:
    """加载并合并默认配置。相对路径的运行时文件以配置文件所在目录为基准。"""

    source_path: Optional[Path] = None
    data = copy.deepcopy(DEFAULTS)
    if path:
        source_path = Path(path).expanduser().resolve()
        if source_path.exists():
            data = _deep_merge(data, _load_yaml(source_path))
        elif str(path) not in ("config.yaml", "./config.yaml"):
            raise ConfigError(f"配置文件不存在: {source_path}")
    if overrides:
        data = _deep_merge(data, overrides)
    data = _expand_tree(data)

    base_dir = source_path.parent if source_path else Path.cwd()
    cfg = Config(data, base_dir, source_path)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.quiz.answer_mode not in ("auto", "text", "image"):
        raise ConfigError("quiz.answer_mode 只能是 auto/text/image")
    if cfg.notify.webhook_type not in ("generic", "serverchan", "telegram"):
        raise ConfigError("notify.webhook_type 只能是 generic/serverchan/telegram")
    for key in ("poll_seconds", "pause_grace_seconds", "stall_timeout_seconds"):
        if float(cfg.get(f"player.{key}")) <= 0:
            raise ConfigError(f"player.{key} 必须大于 0")
    if int(cfg.runtime.monitor) < 1:
        raise ConfigError("runtime.monitor 从 1 开始")
    if not str(cfg.llm.base_url or "").strip():
        raise ConfigError("llm.base_url 不能为空")
    if not str(cfg.llm.model or "").strip():
        raise ConfigError("llm.model 不能为空")
