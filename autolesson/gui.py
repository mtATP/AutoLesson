"""AutoLesson 图形控制台。

只使用 Python 标准库 tkinter，不额外安装 GUI 依赖。
真正运行时以子进程调用 `python -m autolesson <mode>`，因此：
- OCR/鼠标/截图崩溃不会带崩 GUI；
- 日志可以实时显示；
- 停止按钮通过模拟 F9 让子进程安全停止，超时再强制结束。
"""

from __future__ import annotations

import copy
import ctypes
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Optional

import yaml

from .config import DEFAULTS
from .hotkey import parse_hotkey

MODES = {
    "run": ("视频模式", "只观看未完成视频，跳过平时测试"),
    "auto": ("列表模式", "按目录顺序完成视频和平时测试"),
    "practice": ("测试模式", "只处理平时测试，只暂存不提交"),
}

SECTION_TITLES = {
    "window": "窗口",
    "layout": "布局",
    "panel": "目录面板",
    "ocr": "OCR",
    "player": "播放器",
    "practice": "平时测试",
    "quiz": "弹题",
    "llm": "大模型",
    "overlay": "弹窗识别",
    "notify": "通知",
    "runtime": "运行",
    "safety": "安全",
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class ScrollFrame(ttk.Frame):
    """带纵向滚动条的设置页。"""

    def __init__(self, master: tk.Misc):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind(
            "<Configure>",
            lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.inner.bind("<MouseWheel>", self._wheel)

    def _wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")


class App(tk.Tk):
    def __init__(self, config_path: str):
        super().__init__()
        self.title("AutoLesson 控制台")
        self.geometry("1120x780")
        self.minsize(900, 620)
        self.project_root = Path(__file__).resolve().parent.parent
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = self.project_root / self.config_path
        self.data = self._load_raw()
        self.vars: dict[str, tuple[str, tk.Variable]] = {}
        self.proc: Optional[subprocess.Popen] = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self._stop_sent = False
        self._bar: Optional[tk.Toplevel] = None
        self._bar_status: Optional[tk.Label] = None
        self._bar_log: Optional[tk.Label] = None
        self._bar_hide_job: Optional[str] = None
        self._build_style()
        self._build_header()
        self._build_settings()
        self._build_log()
        self.after(120, self._drain_log)
        # 置顶小条默认打开；等主窗口映射后再显示，避免闪一下
        self.after(250, self._show_default_bar)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ 配置
    def _load_raw(self) -> dict:
        data = copy.deepcopy(DEFAULTS)
        if self.config_path.exists():
            try:
                raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
                if isinstance(raw, dict):
                    data = _deep_merge(data, raw)
            except Exception as exc:  # noqa: BLE001
                messagebox.showwarning("读取配置失败", f"{self.config_path}\n{exc}")
        return data

    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("Hint.TLabel", foreground="#666")
        style.configure("Run.TButton", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Danger.TButton", foreground="#a00000")

    def _build_header(self) -> None:
        top = ttk.Frame(self, padding=(12, 10, 12, 6))
        top.pack(fill="x")
        ttk.Label(top, text="AutoLesson", style="Title.TLabel").pack(side="left")
        ttk.Label(
            top,
            text="纯视觉网课助手",
            style="Hint.TLabel",
        ).pack(side="left", padx=(10, 24))

        self.mode_var = tk.StringVar(value="run")
        self.mode_box = ttk.Combobox(
            top,
            values=[f"{key}  {value[0]}" for key, value in MODES.items()],
            state="readonly",
            width=24,
        )
        self.mode_box.set(self._mode_label("run"))
        self.mode_box.pack(side="left")
        self.mode_box.bind("<<ComboboxSelected>>", lambda _e: self._update_mode_hint())

        self.clear_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="启动前清空记录", variable=self.clear_var).pack(
            side="left", padx=(14, 6)
        )
        self.bar_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top, text="置顶小条", variable=self.bar_var, command=self._toggle_bar
        ).pack(side="left")

        self.stop_btn = ttk.Button(
            top, text="停止 (F9/ESC)", command=self.stop, style="Danger.TButton"
        )
        self.stop_btn.pack(side="right")
        self.start_btn = ttk.Button(
            top, text="开始", command=self.start, style="Run.TButton"
        )
        self.start_btn.pack(side="right", padx=8)
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(top, textvariable=self.status_var, foreground="#1a6b2a").pack(
            side="right", padx=12
        )

        self.mode_hint = ttk.Label(self, text="", style="Hint.TLabel", padding=(14, 0))
        self.mode_hint.pack(fill="x")
        self._refresh_modes()

    def _llm_ready(self) -> bool:
        llm = self.data.get("llm", {}) if isinstance(self.data, dict) else {}
        if not isinstance(llm, dict) or not bool(llm.get("enabled", True)):
            return False
        if str(llm.get("api_key") or "").strip():
            return True
        names = [str(llm.get("api_key_env") or "").strip(), "LLM_API_KEY", "OPENAI_API_KEY"]
        return any(name and os.environ.get(name, "").strip() for name in names)

    def _refresh_modes(self) -> None:
        available = ["run"] if not self._llm_ready() else ["run", "auto", "practice"]
        self.mode_box.configure(values=[self._mode_label(k) for k in available])
        current = self._selected_mode()
        if current not in available:
            self.mode_box.set(self._mode_label("run"))
        self._update_mode_hint()

    def _mode_label(self, key: str) -> str:
        return f"{key}  {MODES[key][0]}"

    def _selected_mode(self) -> str:
        return self.mode_box.get().split()[0]

    def _update_mode_hint(self) -> None:
        key = self._selected_mode()
        text = MODES[key][1]
        if not self._llm_ready():
            text = "无 LLM：只允许视频模式，弹题随机选择；列表/测试模式已禁用"
        self.mode_hint.configure(text=text)

    def _build_settings(self) -> None:
        box = ttk.Frame(self, padding=(12, 4, 12, 4))
        box.pack(fill="both", expand=True)
        notebook = ttk.Notebook(box)
        notebook.pack(fill="both", expand=True)
        for section, values in self.data.items():
            if not isinstance(values, dict):
                continue
            page = ScrollFrame(notebook)
            notebook.add(page, text=SECTION_TITLES.get(section, section))
            for key, value in values.items():
                self._add_field(page.inner, section, key, value)
        bottom = ttk.Frame(self, padding=(12, 4))
        bottom.pack(fill="x")
        ttk.Button(bottom, text="保存到 config.yaml", command=self.save_config).pack(
            side="left"
        )
        ttk.Button(bottom, text="重新载入", command=self.reload_config).pack(
            side="left", padx=8
        )
        ttk.Label(
            bottom,
            text="提示：修改后在下次启动生效；含 ${ENV} 的项由启动时环境变量展开。",
            style="Hint.TLabel",
        ).pack(side="left", padx=12)

    def _add_field(self, parent: ttk.Frame, section: str, key: str, value: Any) -> None:
        row = ttk.Frame(parent, padding=(4, 4))
        row.pack(fill="x", padx=6, pady=2)
        label = f"{key}"
        ttk.Label(row, text=label, width=30).pack(side="left", anchor="w")
        path = f"{section}.{key}"
        if isinstance(value, bool):
            var = tk.BooleanVar(value=value)
            ttk.Checkbutton(row, variable=var).pack(side="left")
            self.vars[path] = ("bool", var)
        elif isinstance(value, (list, dict)):
            var = tk.StringVar(value=json.dumps(value, ensure_ascii=False))
            ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)
            self.vars[path] = ("json", var)
        elif isinstance(value, int):
            var = tk.StringVar(value=str(value))
            ttk.Entry(row, textvariable=var, width=18).pack(side="left")
            self.vars[path] = ("int", var)
        elif isinstance(value, float):
            var = tk.StringVar(value=str(value))
            ttk.Entry(row, textvariable=var, width=18).pack(side="left")
            self.vars[path] = ("float", var)
        else:
            var = tk.StringVar(value="" if value is None else str(value))
            entry = ttk.Entry(row, textvariable=var)
            if key in ("api_key", "telegram_bot_token"):
                entry.configure(show="*")
            entry.pack(side="left", fill="x", expand=True)
            self.vars[path] = ("str", var)

    def _collect_config(self) -> dict:
        data = copy.deepcopy(self.data)
        for path, (kind, var) in self.vars.items():
            section, key = path.split(".", 1)
            raw = var.get()
            if kind == "bool":
                value: Any = bool(raw)
            elif kind == "int":
                value = int(str(raw).strip())
            elif kind == "float":
                value = float(str(raw).strip())
            elif kind == "json":
                value = json.loads(str(raw).strip() or "null")
            else:
                value = str(raw)
            data.setdefault(section, {})[key] = value
        return data

    def save_config(self) -> None:
        try:
            data = self._collect_config()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("配置格式错误", str(exc))
            return
        if not messagebox.askyesno(
            "保存配置",
            "将重写 config.yaml（原有注释会丢失，但会先生成 config.yaml.bak）。继续？",
        ):
            return
        try:
            if self.config_path.exists():
                backup = self.config_path.with_suffix(self.config_path.suffix + ".bak")
                backup.write_bytes(self.config_path.read_bytes())
            self.config_path.write_text(
                yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )
            self.data = data
            self._refresh_modes()
            messagebox.showinfo("已保存", f"配置已写入\n{self.config_path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("保存失败", str(exc))

    def reload_config(self) -> None:
        if self.proc is not None:
            messagebox.showwarning("正在运行", "请先停止当前任务再重新载入配置。")
            return
        self.data = self._load_raw()
        for child in self.winfo_children():
            child.destroy()
        self.vars.clear()
        self._build_header()
        self._build_settings()
        self._build_log()
        self._refresh_modes()
        self.status_var.set("已重新载入配置")

    # ------------------------------------------------------------------ 日志
    def _build_log(self) -> None:
        frame = ttk.LabelFrame(self, text="运行日志", padding=(8, 4))
        frame.pack(fill="both", expand=False, padx=12, pady=(2, 12))
        self.log_text = scrolledtext.ScrolledText(
            frame, height=12, wrap="word", state="disabled",
            font=("Consolas", 9), background="#111", foreground="#d8d8d8",
        )
        self.log_text.pack(fill="both", expand=True)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        line = (text or "").strip()
        if line and self._bar_log is not None:
            if len(line) > 110:
                line = "…" + line[-109:]
            self._bar_log.configure(text=line)

    def _drain_log(self) -> None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(line)
        self.after(120, self._drain_log)

    # ------------------------------------------------------------------ 运行
    def _tmp_config(self) -> Path:
        data = self._collect_config()
        path = self.project_root / ".gui_config.yaml"
        path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        return path

    def start(self) -> None:
        if self.proc is not None:
            messagebox.showinfo("正在运行", "当前任务尚未结束。")
            return
        try:
            config = self._tmp_config()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("配置错误", str(exc))
            return
        mode = self._selected_mode()
        if mode != "run" and not self._llm_ready():
            messagebox.showwarning(
                "无 LLM 模式",
                "未配置 API Key 或 llm.enabled=false：只允许视频模式。\n"
                "列表模式和测试模式已禁用。",
            )
            return
        cmd = [sys.executable, "-m", "autolesson", mode, "--config", str(config)]
        if self.clear_var.get():
            cmd.append("--clear-record")
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=flags,
            )
        except Exception as exc:  # noqa: BLE001
            self.proc = None
            messagebox.showerror("启动失败", str(exc))
            return
        self._stop_sent = False
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set(f"运行中：{MODES[mode][0]}")
        self._set_bar_status(f"运行中：{MODES[mode][0]}")
        self._cancel_bar_autohide()
        self._append_log(f"$ {' '.join(cmd)}\n")
        threading.Thread(target=self._read_output, daemon=True).start()
        self.after(3000, self._cleanup_tmp_config)

    def _read_output(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                self.log_queue.put(line)
        except Exception as exc:  # noqa: BLE001
            self.log_queue.put(f"[读取日志失败] {exc}\n")
        code = proc.wait()
        self.log_queue.put(f"\n[进程结束] 退出码 {code}\n")
        self.after(0, lambda: self._on_process_end(code))

    def _cleanup_tmp_config(self) -> None:
        try:
            tmp = self.project_root / ".gui_config.yaml"
            if tmp.exists():
                tmp.unlink()
        except Exception:  # noqa: BLE001
            pass

    def _on_process_end(self, code: int) -> None:
        self.proc = None
        self._cleanup_tmp_config()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set(f"已结束（退出码 {code}）")
        self._set_bar_status(f"已结束（{code}）")
        # 任务结束后小条自动收起，避免一直挡在屏幕顶部；10 秒内重新开始会取消。
        if self._bar is not None:
            self._schedule_bar_autohide(10000)

    def stop(self) -> None:
        if self.proc is None:
            return
        if self._stop_sent:
            return
        self._stop_sent = True
        self.status_var.set("正在停止…")
        self._set_bar_status("正在停止…")
        # 子进程注册了全局停止热键；模拟一次，让它走安全停止流程。
        runtime = self.data.get("runtime", {}) if isinstance(self.data, dict) else {}
        vk = parse_hotkey(str(runtime.get("stop_hotkey", "F9"))) or 0x78
        try:
            user32 = ctypes.windll.user32
            user32.keybd_event(vk, 0, 0, 0)
            user32.keybd_event(vk, 0, 2, 0)
        except Exception:  # noqa: BLE001
            pass
        self.after(7000, self._force_kill_if_needed)

    def _force_kill_if_needed(self) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            self._append_log("\n[停止超时] 强制结束子进程\n")
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
            self.after(3000, self._kill_if_needed)

    def _kill_if_needed(self) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ 置顶小条
    def _show_default_bar(self) -> None:
        if self.bar_var.get() and self._bar is None:
            self._show_bar()

    def _cancel_bar_autohide(self) -> None:
        if self._bar_hide_job is not None:
            try:
                self.after_cancel(self._bar_hide_job)
            except Exception:  # noqa: BLE001
                pass
            self._bar_hide_job = None

    def _schedule_bar_autohide(self, delay_ms: int) -> None:
        self._cancel_bar_autohide()
        self._bar_hide_job = self.after(delay_ms, self._hide_bar)

    def _hide_bar(self) -> None:
        self._cancel_bar_autohide()
        if self._bar is not None:
            try:
                self._bar.destroy()
            except Exception:  # noqa: BLE001
                pass
        self._bar = None
        self._bar_status = None
        self._bar_log = None
        self.bar_var.set(False)

    def _toggle_bar(self) -> None:
        if self.bar_var.get():
            self._show_bar()
        else:
            self._hide_bar()

    def _show_bar(self) -> None:
        self._cancel_bar_autohide()
        if self._bar is not None:
            self._bar.deiconify()
            self._bar.lift()
            return
        bar = tk.Toplevel(self)
        bar.overrideredirect(True)
        bar.attributes("-topmost", True)
        bar.configure(bg="#1f2937")
        width = 720
        x = max(0, (self.winfo_screenwidth() - width) // 2)
        bar.geometry(f"{width}x66+{x}+8")
        top = tk.Frame(bar, bg="#1f2937")
        top.pack(fill="x", padx=(12, 8), pady=(5, 0))
        tk.Label(
            top, text="AutoLesson", bg="#1f2937", fg="#ffffff",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(side="left")
        self._bar_status = tk.Label(
            top, text=self.status_var.get(), bg="#1f2937", fg="#9ae6b4",
            font=("Microsoft YaHei UI", 9),
        )
        self._bar_status.pack(side="left", padx=(10, 0))
        tk.Button(
            top, text="×", command=self._hide_bar, bg="#1f2937", fg="#cbd5e1",
            activebackground="#7f1d1d", activeforeground="white",
            relief="flat", padx=7, font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(side="right", padx=(8, 0))
        tk.Button(
            top, text="停止", command=self.stop, bg="#7f1d1d", fg="white",
            relief="flat", padx=12,
        ).pack(side="right", padx=(6, 0))
        tk.Button(
            top, text="开始", command=self.start, bg="#166534", fg="white",
            relief="flat", padx=12,
        ).pack(side="right", padx=(6, 0))
        self._bar_log = tk.Label(
            bar, text="等待日志…", bg="#111827", fg="#cbd5e1",
            anchor="w", font=("Consolas", 8), padx=8,
        )
        self._bar_log.pack(fill="x", padx=8, pady=(4, 5))
        self._bar = bar
        # 未运行时打开小条：2 分钟无操作自动收起；开始任务会取消定时器。
        if self.proc is None:
            self._schedule_bar_autohide(120000)

    def _set_bar_status(self, text: str) -> None:
        if self._bar_status is not None:
            self._bar_status.configure(text=text)

    # ------------------------------------------------------------------ 关闭
    def _on_close(self) -> None:
        if self.proc is not None:
            if not messagebox.askyesno("正在运行", "任务仍在运行，停止并退出？"):
                return
            self.stop()
            deadline = time.monotonic() + 4.0
            while self.proc is not None and time.monotonic() < deadline:
                self.update()
                time.sleep(0.1)
            if self.proc is not None:
                self._kill_if_needed()
        try:
            tmp = self.project_root / ".gui_config.yaml"
            if tmp.exists():
                tmp.unlink()
        except Exception:  # noqa: BLE001
            pass
        self.destroy()


def main(config_path: str = "config.yaml") -> int:
    app = App(config_path)
    app.mainloop()
    return 0
