"""天空日記觀察站桌面程式主程式。"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import signal
import socket
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser

import numpy as np  # noqa: F401
from PIL import Image

import skycam_tool

APP_VERSION = "1.0.0"
SHARE_URL = ""

TAIWAN_COUNTIES = (
    "臺北市",
    "新北市",
    "桃園市",
    "臺中市",
    "臺南市",
    "高雄市",
    "基隆市",
    "新竹市",
    "嘉義市",
    "新竹縣",
    "苗栗縣",
    "彰化縣",
    "南投縣",
    "雲林縣",
    "嘉義縣",
    "屏東縣",
    "宜蘭縣",
    "花蓮縣",
    "臺東縣",
    "澎湖縣",
    "金門縣",
    "連江縣",
)

DATE_RE = re.compile(r"^\d{8}$")
PHOTO_RE = re.compile(r"^\d{6}(_d)?\.jpg$")
ANALYSIS_RE = re.compile(
    r"^\d{8}(?:\.csv|_sun_trail\.png|_sky_strip\.png|\.mp4|_summary\.json)$"
)
CLASS_CODE_RE = re.compile(r"^[A-Za-z0-9]{0,12}$")
TIME_COLON_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})$")
TIME_COMPACT_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})")

RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
MAX_BODY_BYTES = 1024 * 1024


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    """允許重複使用位址並以背景執行緒處理連線的本機 HTTP 伺服器。"""

    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """停用 urllib 自動轉址，讓程式自行處理 301/302/303 轉址。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """攔截轉址並回傳 None 以觸發 HTTPError。"""
        return None


def now_str() -> str:
    """回傳目前本機時間字串 YYYY-MM-DD HH:MM:SS。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_ui_html_path() -> Path:
    """取得 ui.html 檔案路徑，支援直接執行與 PyInstaller 打包。"""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / "ui.html"


def get_share_url() -> str:
    """取得並驗證有效的分享接收網址，無效或未設定時回傳空字串。"""
    raw = os.environ.get("SKYCAM_SHARE_URL", "").strip() or SHARE_URL.strip()
    if not raw:
        return ""
    if raw.startswith("https://script.google.com/macros/s/") and raw.endswith("/exec"):
        return raw
    if raw.startswith("http://127.0.0.1:"):
        return raw
    return ""


def normalize_camera_url(raw: str) -> str:
    """將使用者輸入的相機位址正規化為 http://主機[:埠]。"""
    text = (raw or "").strip()
    if not text:
        raise ValueError("請輸入相機網址或 IP 位址")
    if "://" not in text:
        text = "http://" + text
    parsed = urllib.parse.urlsplit(text)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("相機網址格式不正確")
    host = parsed.hostname
    if not host:
        raise ValueError("相機網址格式不正確")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("相機通訊埠格式不正確") from exc
    if port is not None:
        return f"http://{host}:{port}"
    return f"http://{host}"


def get_local_ipv4() -> str:
    """透過 UDP socket 讀取電腦目前對外連線使用的區域網路 IPv4。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    finally:
        sock.close()


def http_get_json(url: str, timeout: float = 5.0) -> dict:
    """對指定網址發送 HTTP GET 並解析回傳的 JSON 物件。"""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("回應不是 JSON 物件")
    return data


def post_json_follow_redirect(url: str, payload: dict, timeout: float = 30.0) -> dict:
    """以 text/plain 送出 JSON POST，並手動以 GET 跟隨 301/302/303 轉址。"""
    opener = urllib.request.build_opener(_NoRedirectHandler)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    current_url = url
    req = urllib.request.Request(
        current_url,
        data=body,
        headers={"Content-Type": "text/plain;charset=utf-8"},
        method="POST",
    )
    for _ in range(6):
        try:
            with opener.open(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("接收端回應格式不正確")
            return data
        except urllib.error.HTTPError as err:
            code = err.code
            location = err.headers.get("Location") if err.headers else None
            err.close()
            if code in (301, 302, 303) and location:
                current_url = urllib.parse.urljoin(current_url, location)
                req = urllib.request.Request(current_url, method="GET")
                continue
            raise
    raise RuntimeError("轉址次數過多")


def is_sun_found(val) -> bool:
    """判斷 sun_found 欄位是否代表有找到太陽。"""
    if isinstance(val, bool):
        return val
    try:
        return int(val) == 1
    except (TypeError, ValueError):
        return False


def parse_row_time(row: dict) -> tuple[str, int] | None:
    """從分析資料列解析 HH:MM:SS 時間字串與午夜起算秒數。"""
    raw_time = str(row.get("time", "")).strip()
    match = TIME_COLON_RE.match(raw_time)
    if not match:
        match = TIME_COMPACT_RE.match(raw_time)
    if not match:
        match = TIME_COMPACT_RE.match(str(row.get("name", "")).strip())
    if not match:
        return None
    hour, minute, second = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None
    return f"{hour:02d}:{minute:02d}:{second:02d}", hour * 3600 + minute * 60 + second


def compute_summary(date: str, rows: list[dict]) -> dict:
    """依 analyze_day 的結果計算單日夕陽摘要物件。"""
    normal_items: list[tuple[int, str, str, dict]] = []
    dark_count = 0
    dark_sun_items: list[tuple[int, str, str, dict]] = []

    for row in rows:
        kind = str(row.get("kind", "")).strip()
        name = str(row.get("name", "")).strip()
        parsed = parse_row_time(row)
        time_str, seconds = parsed if parsed is not None else ("", -1)
        if kind == "normal":
            normal_items.append((seconds, name, time_str, row))
        elif kind == "dark":
            dark_count += 1
            if is_sun_found(row.get("sun_found", 0)):
                dark_sun_items.append((seconds, name, time_str, row))

    dark_sun_items.sort(key=lambda item: (item[0], item[1]))

    if dark_sun_items:
        first_sun_time = dark_sun_items[0][2]
        last_sun_time = dark_sun_items[-1][2]
        last_row = dark_sun_items[-1][3]
        last_sun_x: float | None = float(last_row["sun_x"])
        last_sun_y: float | None = float(last_row["sun_y"])
    else:
        first_sun_time = ""
        last_sun_time = ""
        last_sun_x = None
        last_sun_y = None

    target_1630 = 16 * 3600 + 30 * 60
    near_1630 = [
        item for item in dark_sun_items if item[0] >= 0 and abs(item[0] - target_1630) <= 300
    ]
    if near_1630:
        best_1630 = min(near_1630, key=lambda item: (abs(item[0] - target_1630), item[0], item[1]))
        sun_x_1630: float | None = float(best_1630[3]["sun_x"])
    else:
        sun_x_1630 = None

    target_1700 = 17 * 3600
    valid_normals = [item for item in normal_items if item[0] >= 0]
    if valid_normals:
        best_1700 = min(
            valid_normals, key=lambda item: (abs(item[0] - target_1700), item[0], item[1])
        )
        r_row = best_1700[3]
        sky_avg_rgb_1700: list[float] | None = [
            float(r_row["sky_r"]),
            float(r_row["sky_g"]),
            float(r_row["sky_b"]),
        ]
    else:
        sky_avg_rgb_1700 = None

    return {
        "date": date,
        "photos": len(normal_items),
        "dark_photos": dark_count,
        "sun_found": len(dark_sun_items),
        "first_sun_time": first_sun_time,
        "last_sun_time": last_sun_time,
        "last_sun_x": last_sun_x,
        "last_sun_y": last_sun_y,
        "sun_x_1630": sun_x_1630,
        "sky_avg_rgb_1700": sky_avg_rgb_1700,
    }


def validate_share_settings(body: dict) -> dict:
    """驗證分享設定欄位並回傳正規化後的設定字典。"""
    if not isinstance(body, dict):
        raise ValueError("分享設定格式不正確")

    enabled = body.get("enabled")
    consent = body.get("consent")
    auto = body.get("auto", True)

    if not isinstance(enabled, bool):
        raise ValueError("分享開關欄位格式不正確")
    if not isinstance(consent, bool):
        raise ValueError("同意分享欄位格式不正確")
    if not isinstance(auto, bool):
        raise ValueError("自動分享欄位格式不正確")
    if enabled and not consent:
        raise ValueError("開啟分享前，必須先勾選同意分享")

    raw_nick = body.get("nickname")
    if not isinstance(raw_nick, str):
        raise ValueError("請填寫暱稱")
    nickname = raw_nick.strip()
    if not (1 <= len(nickname) <= 12):
        raise ValueError("暱稱長度必須是 1 到 12 個字")
    if "@" in nickname:
        raise ValueError("暱稱不可以包含 @ 符號")
    if sum(ch.isdigit() for ch in nickname) > 4:
        raise ValueError("暱稱裡的數字不可以超過 4 個，請不要填電話或學號")

    raw_class = body.get("class_code", "")
    if raw_class is None:
        raw_class = ""
    if not isinstance(raw_class, str):
        raise ValueError("班級代碼格式不正確")
    class_code = raw_class.strip()
    if not CLASS_CODE_RE.fullmatch(class_code):
        raise ValueError("班級代碼只能填 0 到 12 個英文字母或數字")

    raw_county = body.get("county")
    if not isinstance(raw_county, str) or raw_county.strip() not in TAIWAN_COUNTIES:
        raise ValueError("請從清單中選擇正確的縣市")
    county = raw_county.strip()

    return {
        "enabled": enabled,
        "consent": consent,
        "nickname": nickname,
        "class_code": class_code,
        "county": county,
        "auto": auto,
    }


def validate_camera_config(body: dict) -> dict[str, int]:
    """驗證相機設定參數範圍並回傳送往相機的查詢參數。"""
    if not isinstance(body, dict):
        raise ValueError("請提供至少一項相機設定")
    if not body:
        return {}                     # 空的請求＝只讀取相機目前的設定

    allowed = {"flip", "dark_aec", "powersave", "txpower"}
    unknown = set(body.keys()) - allowed
    if unknown:
        raise ValueError("包含不支援的相機設定項目")

    params: dict[str, int] = {}
    for key in ("flip", "powersave"):
        if key in body:
            val = body[key]
            if isinstance(val, bool):
                val = int(val)
            if not isinstance(val, int) or val not in (0, 1):
                raise ValueError(f"{key} 只能是 0 或 1")
            params[key] = val

    if "dark_aec" in body:
        val = body["dark_aec"]
        if isinstance(val, bool) or not isinstance(val, int) or not (1 <= val <= 1200):
            raise ValueError("暗曝光亮度必須是 1 到 1200 之間的整數")
        params["dark_aec"] = val

    if "txpower" in body:
        val = body["txpower"]
        if isinstance(val, bool) or not isinstance(val, int) or not (2 <= val <= 20):
            raise ValueError("Wi-Fi 發射功率必須是 2 到 20 之間的整數")
        params["txpower"] = val

    return params


def encode_sun_trail_base64(image_path: Path) -> str:
    """將太陽軌跡圖縮至寬度不超過 800 像素並壓縮成 JPEG 的 base64 字串。"""
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        if width > 800:
            new_height = max(1, round(height * 800 / width))
            rgb = rgb.resize((800, new_height), RESAMPLE_LANCZOS)

        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=70, optimize=True)
        raw = buf.getvalue()
        if len(raw) > 200 * 1024:
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=50, optimize=True)
            raw = buf.getvalue()

    return base64.b64encode(raw).decode("ascii")


def atomic_write_text(path: Path, text: str) -> None:
    """以暫存檔替換的方式安全寫入文字檔。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


class SkyCamApp:
    """封裝天空日記觀察站的資料存取、同步、分析與分享邏輯。"""

    def __init__(self, data_dir: Path) -> None:
        """初始化資料目錄結構、執行緒鎖與設定檔。"""
        self.data_dir = data_dir.expanduser().resolve()
        self.photos_dir = self.data_dir / "photos"
        self.analysis_dir = self.data_dir / "analysis"
        self.thumbs_dir = self.data_dir / "thumbs"
        self.settings_path = self.data_dir / "settings.json"
        self.share_log_path = self.data_dir / "share_log.jsonl"

        self.photos_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)

        self._settings_lock = threading.Lock()
        self._sync_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._fw_lock = threading.Lock()

        self._camera_fw = ""
        self._sync_state: dict = {
            "running": False,
            "last": None,
            "error": None,
            "finished_at": None,
        }
        self.stop_event = threading.Event()
        self._ensure_settings()

    def _default_settings(self) -> dict:
        """建立包含新裝置識別碼的預設設定字典。"""
        return {
            "camera": "",
            "device_id": uuid.uuid4().hex,
            "share": {
                "enabled": False,
                "consent": False,
                "nickname": "",
                "class_code": "",
                "county": "",
                "auto": True,
            },
        }

    def _ensure_settings(self) -> None:
        """確保 settings.json 存在且包含完整必要鍵值。"""
        with self._settings_lock:
            defaults = self._default_settings()
            if not self.settings_path.is_file():
                atomic_write_text(
                    self.settings_path,
                    json.dumps(defaults, ensure_ascii=False, indent=2),
                )
                return
            try:
                loaded = json.loads(self.settings_path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    loaded = {}
            except (OSError, json.JSONDecodeError):
                loaded = {}

            changed = False
            if "camera" not in loaded or not isinstance(loaded["camera"], str):
                loaded["camera"] = defaults["camera"]
                changed = True
            if not loaded.get("device_id") or not isinstance(loaded["device_id"], str):
                loaded["device_id"] = defaults["device_id"]
                changed = True
            share = loaded.get("share")
            if not isinstance(share, dict):
                share = {}
                loaded["share"] = share
                changed = True
            for key, default_val in defaults["share"].items():
                if key not in share:
                    share[key] = default_val
                    changed = True

            if changed:
                atomic_write_text(
                    self.settings_path,
                    json.dumps(loaded, ensure_ascii=False, indent=2),
                )

    def get_settings(self) -> dict:
        """安全讀取目前的 settings.json 內容。"""
        with self._settings_lock:
            try:
                data = json.loads(self.settings_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (OSError, json.JSONDecodeError):
                pass
        self._ensure_settings()
        with self._settings_lock:
            return json.loads(self.settings_path.read_text(encoding="utf-8"))

    def save_settings(self, settings: dict) -> None:
        """安全寫入 settings.json 檔案。"""
        with self._settings_lock:
            atomic_write_text(
                self.settings_path,
                json.dumps(settings, ensure_ascii=False, indent=2),
            )

    def set_camera_fw(self, fw: str) -> None:
        """記錄最近一次從相機狀態取得的韌體版本。"""
        with self._fw_lock:
            self._camera_fw = str(fw or "")

    def get_camera_fw(self) -> str:
        """讀取最近一次記錄的相機韌體版本。"""
        with self._fw_lock:
            return self._camera_fw

    def list_days(self) -> list[str]:
        """列出所有已有照片的日期清單，由新到舊排序。"""
        days: list[str] = []
        if not self.photos_dir.is_dir():
            return days
        for entry in self.photos_dir.iterdir():
            if entry.is_dir() and DATE_RE.fullmatch(entry.name):
                has_photo = any(
                    f.is_file() and PHOTO_RE.fullmatch(f.name) for f in entry.iterdir()
                )
                if has_photo:
                    days.append(entry.name)
        days.sort(reverse=True)
        return days

    def get_state(self) -> dict:
        """組裝首頁初始化所需的系統狀態資料。"""
        return {
            "ok": True,
            "version": APP_VERSION,
            "data_dir": str(self.data_dir),
            "settings": self.get_settings(),
            "share_url_set": bool(get_share_url()),
            "days": self.list_days(),
        }

    def connect_camera(self, raw_address: str) -> tuple[str, dict]:
        """正規化相機網址、測試 /status 連線並寫入設定。"""
        camera_url = normalize_camera_url(raw_address)
        status = http_get_json(f"{camera_url}/status", timeout=5.0)
        if "fw" in status:
            self.set_camera_fw(str(status.get("fw", "")))
        settings = self.get_settings()
        settings["camera"] = camera_url
        self.save_settings(settings)
        return camera_url, status

    def fetch_camera_status(self) -> dict:
        """代理查詢目前已設定相機的 /status 資訊。"""
        camera_url = self.get_settings().get("camera", "").strip()
        if not camera_url:
            raise ValueError("尚未設定相機網址，請先連線相機")
        status = http_get_json(f"{camera_url}/status", timeout=5.0)
        if "fw" in status:
            self.set_camera_fw(str(status.get("fw", "")))
        return status

    def scan_cameras(self) -> list[dict]:
        """掃描 mDNS 與同區網 /24 內所有回應 fw 與 sensor_pid 的相機。"""
        targets: list[str] = ["http://skycam.local"]
        try:
            local_ip = get_local_ipv4()
            parts = local_ip.split(".")
            if len(parts) == 4:
                prefix = ".".join(parts[:3])
                targets.extend(f"http://{prefix}.{i}" for i in range(1, 255))
        except OSError:
            pass

        def probe(address: str) -> dict | None:
            """探測單一位址是否為天空日記相機。"""
            try:
                data = http_get_json(f"{address}/status", timeout=0.6)
            except Exception:
                return None
            if "fw" in data and "sensor_pid" in data:
                return {"address": address, "fw": str(data.get("fw", ""))}
            return None

        found: list[dict] = []
        first = probe(targets[0])
        if first is not None:
            found.append(first)

        lan_targets = targets[1:]
        if lan_targets:
            with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
                for result in executor.map(probe, lan_targets):
                    if result is not None:
                        found.append(result)
        return found

    def update_camera_config(self, body: dict) -> dict:
        """驗證相機設定參數並代理送到相機 /config。"""
        params = validate_camera_config(body)
        camera_url = self.get_settings().get("camera", "").strip()
        if not camera_url:
            raise ValueError("尚未設定相機網址，請先連線相機")
        query = urllib.parse.urlencode(params)
        return http_get_json(f"{camera_url}/config?{query}", timeout=5.0)

    def get_sync_status(self) -> dict:
        """取得目前同步工作的狀態快照。"""
        with self._sync_lock:
            return dict(self._sync_state)

    def start_sync(self) -> bool:
        """啟動背景同步工作；若已有同步進行中則回傳 False。"""
        camera_url = self.get_settings().get("camera", "").strip()
        if not camera_url:
            raise ValueError("尚未設定相機網址，請先連線相機")
        with self._sync_lock:
            if self._sync_state["running"]:
                return False
            self._sync_state["running"] = True
            self._sync_state["error"] = None

        thread = threading.Thread(
            target=self._run_sync_worker,
            args=(camera_url,),
            daemon=True,
        )
        thread.start()
        return True

    def _run_sync_worker(self, camera_url: str) -> None:
        """在背景執行 skycam_tool.pull 並更新同步狀態。"""
        try:
            result = skycam_tool.pull(camera_url, str(self.photos_dir))
            with self._sync_lock:
                self._sync_state["last"] = result
                self._sync_state["error"] = None
                self._sync_state["finished_at"] = now_str()
                self._sync_state["running"] = False
        except Exception as exc:
            with self._sync_lock:
                self._sync_state["error"] = f"同步失敗：{exc}"
                self._sync_state["finished_at"] = now_str()
                self._sync_state["running"] = False

    def _safe_photo_path(self, date: str, filename: str) -> Path:
        """驗證日期與照片檔名並回傳受限於照片資料夾內的絕對路徑。"""
        if not DATE_RE.fullmatch(date) or not PHOTO_RE.fullmatch(filename):
            raise ValueError("無效的日期或照片檔名")
        target = (self.photos_dir / date / filename).resolve()
        if not target.is_relative_to(self.photos_dir):
            raise ValueError("不允許的路徑存取")
        return target

    def _safe_analysis_path(self, filename: str) -> Path:
        """驗證分析檔名並回傳受限於分析資料夾內的絕對路徑。"""
        if not ANALYSIS_RE.fullmatch(filename):
            raise ValueError("無效的分析檔名")
        target = (self.analysis_dir / filename).resolve()
        if not target.is_relative_to(self.analysis_dir):
            raise ValueError("不允許的路徑存取")
        return target

    def get_day_info(self, date: str) -> dict:
        """回傳指定日期的照片清單、分析檔案存在狀態與摘要。"""
        if not DATE_RE.fullmatch(date):
            raise ValueError("日期格式必須是 8 位數字")
        day_dir = (self.photos_dir / date).resolve()
        if not day_dir.is_relative_to(self.photos_dir) or not day_dir.is_dir():
            raise FileNotFoundError("找不到這一天的照片資料夾")

        photos: list[dict] = []
        for entry in sorted(day_dir.iterdir(), key=lambda p: p.name):
            if entry.is_file() and PHOTO_RE.fullmatch(entry.name):
                name = entry.name
                photos.append(
                    {
                        "name": name,
                        "time": f"{name[0:2]}:{name[2:4]}:{name[4:6]}",
                        "kind": "dark" if name.endswith("_d.jpg") else "normal",
                    }
                )

        analysis = {
            "csv": (self.analysis_dir / f"{date}.csv").is_file(),
            "sun_trail": (self.analysis_dir / f"{date}_sun_trail.png").is_file(),
            "sky_strip": (self.analysis_dir / f"{date}_sky_strip.png").is_file(),
            "timelapse": (self.analysis_dir / f"{date}.mp4").is_file(),
        }

        summary = self._load_summary(date)
        return {
            "ok": True,
            "date": date,
            "photos": photos,
            "analysis": analysis,
            "summary": summary,
        }

    def _load_summary(self, date: str) -> dict | None:
        """讀取指定日期的摘要 JSON，若不存在或損毀則回傳 None。"""
        summary_path = self.analysis_dir / f"{date}_summary.json"
        if not summary_path.is_file():
            return None
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def analyze_day_full(self, date: str) -> dict:
        """執行單日照片分析、產生圖表與縮時影片並儲存摘要 JSON。"""
        if not DATE_RE.fullmatch(date):
            raise ValueError("日期格式必須是 8 位數字")
        day_dir = (self.photos_dir / date).resolve()
        if not day_dir.is_relative_to(self.photos_dir) or not day_dir.is_dir():
            raise FileNotFoundError("找不到這一天的照片資料夾")

        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.analysis_dir / f"{date}.csv"
        sun_trail_path = self.analysis_dir / f"{date}_sun_trail.png"
        sky_strip_path = self.analysis_dir / f"{date}_sky_strip.png"
        timelapse_path = self.analysis_dir / f"{date}.mp4"
        summary_path = self.analysis_dir / f"{date}_summary.json"

        rows = skycam_tool.analyze_day(str(day_dir))
        skycam_tool.write_csv(rows, str(csv_path))
        skycam_tool.make_sun_trail(str(day_dir), str(sun_trail_path))
        skycam_tool.make_sky_strip(str(day_dir), str(sky_strip_path))
        try:
            skycam_tool.make_timelapse(str(day_dir), str(timelapse_path))
        except Exception:
            pass

        summary = compute_summary(date, rows)
        atomic_write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    def get_compare(self) -> list[dict]:
        """讀取所有已完成分析的日期摘要，由舊到新排序。"""
        summaries: list[dict] = []
        if not self.analysis_dir.is_dir():
            return summaries
        for entry in sorted(self.analysis_dir.iterdir(), key=lambda p: p.name):
            if not entry.is_file() or not entry.name.endswith("_summary.json"):
                continue
            date_part = entry.name[: -len("_summary.json")]
            if not DATE_RE.fullmatch(date_part):
                continue
            summary = self._load_summary(date_part)
            if summary is not None:
                summaries.append(summary)
        summaries.sort(key=lambda item: str(item.get("date", "")))
        return summaries

    def get_photo_path(self, date: str, filename: str) -> Path:
        """取得原始照片路徑，若不存在則拋出 FileNotFoundError。"""
        path = self._safe_photo_path(date, filename)
        if not path.is_file():
            raise FileNotFoundError("找不到這張照片")
        return path

    def get_or_make_thumb(self, date: str, filename: str) -> Path:
        """取得寬度 320 像素的縮圖路徑，尚未產生時自動建立快取。"""
        orig_path = self.get_photo_path(date, filename)
        thumb_path = (self.thumbs_dir / date / filename).resolve()
        if not thumb_path.is_relative_to(self.thumbs_dir):
            raise ValueError("不允許的縮圖路徑")

        need_build = True
        if thumb_path.is_file():
            try:
                if thumb_path.stat().st_mtime >= orig_path.stat().st_mtime:
                    need_build = False
            except OSError:
                need_build = True

        if need_build:
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(orig_path) as img:
                rgb = img.convert("RGB")
                width, height = rgb.size
                if width > 320:
                    new_height = max(1, round(height * 320 / width))
                    rgb = rgb.resize((320, new_height), RESAMPLE_LANCZOS)
                tmp_path = thumb_path.with_name(f".{thumb_path.name}.{uuid.uuid4().hex}.tmp")
                rgb.save(tmp_path, format="JPEG", quality=80, optimize=True)
            os.replace(tmp_path, thumb_path)
        return thumb_path

    def get_analysis_file(self, filename: str) -> tuple[Path, str]:
        """取得分析資料夾中的檔案路徑與對應的 Content-Type。"""
        path = self._safe_analysis_path(filename)
        if not path.is_file():
            raise FileNotFoundError("找不到指定的分析檔案")
        suffix = path.suffix.lower()
        content_types = {
            ".png": "image/png",
            ".csv": "text/csv; charset=utf-8",
            ".mp4": "video/mp4",
            ".json": "application/json; charset=utf-8",
        }
        return path, content_types.get(suffix, "application/octet-stream")

    def save_share_settings(self, body: dict) -> dict:
        """驗證並儲存分享設定到 settings.json。"""
        validated = validate_share_settings(body)
        settings = self.get_settings()
        settings["share"] = validated
        self.save_settings(settings)
        return validated

    def append_share_log(self, date: str, ok: bool, message: str) -> None:
        """在 share_log.jsonl 新增一筆分享紀錄。"""
        entry = {
            "at": now_str(),
            "date": date,
            "ok": bool(ok),
            "message": str(message),
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._log_lock:
            self.share_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.share_log_path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    def read_share_log(self, limit: int = 30) -> list[dict]:
        """讀取 share_log.jsonl 最近 limit 筆紀錄並由新到舊回傳。"""
        with self._log_lock:
            if not self.share_log_path.is_file():
                return []
            lines = self.share_log_path.read_text(encoding="utf-8", errors="replace").splitlines()

        entries: list[dict] = []
        for line in lines:
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
                if isinstance(item, dict):
                    entries.append(item)
            except json.JSONDecodeError:
                continue
        entries.reverse()
        return entries[:limit]

    def has_shared_successfully(self, date: str) -> bool:
        """檢查指定日期是否已經有成功分享的紀錄。"""
        with self._log_lock:
            if not self.share_log_path.is_file():
                return False
            lines = self.share_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines:
            try:
                item = json.loads(line)
                if isinstance(item, dict) and item.get("date") == date and item.get("ok") is True:
                    return True
            except json.JSONDecodeError:
                continue
        return False

    def share_day(self, date: str) -> tuple[bool, str, int]:
        """將指定日期的夕陽摘要與太陽軌跡縮圖送給老師並寫入紀錄。"""
        if not DATE_RE.fullmatch(date):
            return False, "日期格式必須是 8 位數字", 400

        share_url = get_share_url()
        if not share_url:
            return False, "老師還沒有設定接收網址", 400

        settings = self.get_settings()
        share = settings.get("share", {})
        if not (share.get("enabled") is True and share.get("consent") is True):
            return False, "請先在「分享給老師」頁面勾選同意分享並儲存設定", 400

        summary = self._load_summary(date)
        sun_trail_path = self.analysis_dir / f"{date}_sun_trail.png"
        if summary is None or not sun_trail_path.is_file():
            return False, "這一天還沒有完成夕陽分析，請先分析後再分享", 400

        try:
            image_b64 = encode_sun_trail_base64(sun_trail_path)
        except Exception as exc:
            msg = f"壓縮太陽軌跡圖失敗：{exc}"
            self.append_share_log(date, False, msg)
            return False, msg, 500

        payload = {
            "v": 1,
            "app": "skycam-desktop",
            "app_version": APP_VERSION,
            "device_id": str(settings.get("device_id", "")),
            "nickname": str(share.get("nickname", "")),
            "class_code": str(share.get("class_code", "")),
            "county": str(share.get("county", "")),
            "date": date,
            "camera_fw": self.get_camera_fw(),
            "summary": summary,
            "image": image_b64,
        }

        try:
            resp = post_json_follow_redirect(share_url, payload, timeout=30.0)
        except Exception as exc:
            msg = f"傳送失敗，請確認網路連線：{exc}"
            self.append_share_log(date, False, msg)
            return False, msg, 502

        if resp.get("ok") is True:
            msg = str(resp.get("message") or "分享成功")
            self.append_share_log(date, True, msg)
            return True, msg, 200

        msg = str(resp.get("error") or resp.get("message") or "接收端回應分享失敗")
        self.append_share_log(date, False, msg)
        return False, msg, 502

    def start_auto_share_loop(self) -> None:
        """啟動每 10 分鐘檢查一次的傍晚自動分享背景執行緒。"""
        thread = threading.Thread(target=self._auto_share_worker, daemon=True)
        thread.start()

    def _auto_share_worker(self) -> None:
        """在背景定時檢查是否符合傍晚自動分享條件並執行同步、分析與分享。"""
        while not self.stop_event.wait(600.0):
            try:
                self._check_and_auto_share()
            except Exception:
                continue

    def _check_and_auto_share(self) -> None:
        """檢查今天是否已過 18:30 且具備自動分享條件，符合就同步、分析並分享。"""
        now = datetime.now()
        if (now.hour, now.minute) < (18, 30):
            return
        if not get_share_url():
            return
        settings = self.get_settings()
        share = settings.get("share", {})
        if not (
            share.get("enabled") is True
            and share.get("consent") is True
            and share.get("auto") is True
        ):
            return

        today = now.strftime("%Y%m%d")
        if self.has_shared_successfully(today):
            return

        day_dir = self.photos_dir / today
        summary_path = self.analysis_dir / f"{today}_summary.json"
        if not day_dir.is_dir() or not summary_path.is_file():
            return

        camera_url = settings.get("camera", "").strip()
        if camera_url:
            can_sync = False
            with self._sync_lock:
                if not self._sync_state["running"]:
                    self._sync_state["running"] = True
                    self._sync_state["error"] = None
                    can_sync = True
            if can_sync:
                self._run_sync_worker(camera_url)

        self.analyze_day_full(today)
        self.share_day(today)


def make_handler(app: SkyCamApp):
    """建立綁定指定 SkyCamApp 實例的 HTTP 請求處理類別。"""

    class SkyCamHandler(BaseHTTPRequestHandler):
        """處理天空日記觀察站的前端頁面與 API 請求。"""

        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args) -> None:
            """靜音預設的存取紀錄輸出，保持終端機乾淨。"""
            return

        def _send_bytes(self, status: int, content_type: str, data: bytes) -> None:
            """送出指定狀態碼、MIME 類型與位元組內容的回應。"""
            self.close_connection = True
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(data)
            except OSError:
                pass

        def _send_json(self, status: int, obj: dict) -> None:
            """將字典序列化為 UTF-8 JSON 並送出回應。"""
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self._send_bytes(status, "application/json; charset=utf-8", raw)

        def _send_file(self, path: Path, content_type: str) -> None:
            """送出本地靜態檔案，並支援影片播放所需的 Range 請求。"""
            self.close_connection = True
            size = path.stat().st_size
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                spec = range_header[6:].split(",")[0].strip()
                if "-" in spec:
                    start_s, end_s = spec.split("-", 1)
                    try:
                        start = int(start_s) if start_s else 0
                        end = int(end_s) if end_s else size - 1
                    except ValueError:
                        self._send_json(416, {"ok": False, "error": "無效的區段請求"})
                        return
                    if start < 0 or start >= size or end < start:
                        self._send_json(416, {"ok": False, "error": "請求區段超出範圍"})
                        return
                    end = min(end, size - 1)
                    length = end - start + 1
                    self.send_response(206)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.send_header("Content-Length", str(length))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    try:
                        with path.open("rb") as fh:
                            fh.seek(start)
                            remaining = length
                            while remaining > 0:
                                chunk = fh.read(min(65536, remaining))
                                if not chunk:
                                    break
                                self.wfile.write(chunk)
                                remaining -= len(chunk)
                    except OSError:
                        pass
                    return

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(size))
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                with path.open("rb") as fh:
                    while True:
                        chunk = fh.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except OSError:
                pass

        def _read_json_body(self) -> dict:
            """讀取並解析 POST 請求的 JSON 本文。"""
            length_str = self.headers.get("Content-Length", "0")
            try:
                length = int(length_str)
            except ValueError as exc:
                raise ValueError("無效的 Content-Length") from exc
            if length < 0 or length > MAX_BODY_BYTES:
                raise ValueError("請求內容過大")
            if length == 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8", errors="replace").strip()
            if not raw:
                return {}
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("請求資料不是有效的 JSON 格式") from exc
            if not isinstance(data, dict):
                raise ValueError("請求資料必須是 JSON 物件")
            return data

        def _has_unsafe_path(self, raw_path: str) -> bool:
            """檢查原始與解碼後的網址路徑是否含路徑穿越字元。"""
            decoded = urllib.parse.unquote(raw_path)
            if ".." in raw_path or ".." in decoded:
                return True
            if "\\" in decoded or "\x00" in decoded:
                return True
            return False

        def do_GET(self) -> None:
            """分派處理所有 GET 頁面、API 與媒體檔案請求。"""
            raw_path = urllib.parse.urlsplit(self.path).path
            if self._has_unsafe_path(raw_path):
                self._send_json(400, {"ok": False, "error": "不允許的路徑"})
                return

            if raw_path == "/":
                ui_path = get_ui_html_path()
                if not ui_path.is_file():
                    self._send_json(500, {"ok": False, "error": "找不到介面檔案 ui.html"})
                    return
                self._send_bytes(200, "text/html; charset=utf-8", ui_path.read_bytes())
                return

            if raw_path == "/api/state":
                self._send_json(200, app.get_state())
                return

            if raw_path == "/api/camera/status":
                try:
                    status = app.fetch_camera_status()
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except Exception:
                    self._send_json(
                        502,
                        {"ok": False, "error": "連不到相機，請確認電腦和相機連的是同一個 Wi-Fi"},
                    )
                    return
                self._send_json(200, status)
                return

            if raw_path == "/api/sync":
                self._send_json(200, app.get_sync_status())
                return

            if raw_path == "/api/compare":
                self._send_json(200, {"ok": True, "days": app.get_compare()})
                return

            if raw_path == "/api/share/log":
                self._send_json(200, {"ok": True, "logs": app.read_share_log(30)})
                return

            if raw_path.startswith("/api/day/"):
                rest = raw_path[len("/api/day/") :]
                if "/" in rest:
                    self._send_json(404, {"ok": False, "error": "找不到指定路徑"})
                    return
                date = urllib.parse.unquote(rest)
                try:
                    info = app.get_day_info(date)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except FileNotFoundError as exc:
                    self._send_json(404, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, info)
                return

            if raw_path.startswith("/photo/"):
                parts = raw_path[len("/photo/") :].split("/")
                if len(parts) != 2:
                    self._send_json(400, {"ok": False, "error": "無效的照片路徑"})
                    return
                date = urllib.parse.unquote(parts[0])
                filename = urllib.parse.unquote(parts[1])
                try:
                    path = app.get_photo_path(date, filename)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except FileNotFoundError as exc:
                    self._send_json(404, {"ok": False, "error": str(exc)})
                    return
                self._send_file(path, "image/jpeg")
                return

            if raw_path.startswith("/thumb/"):
                parts = raw_path[len("/thumb/") :].split("/")
                if len(parts) != 2:
                    self._send_json(400, {"ok": False, "error": "無效的縮圖路徑"})
                    return
                date = urllib.parse.unquote(parts[0])
                filename = urllib.parse.unquote(parts[1])
                try:
                    path = app.get_or_make_thumb(date, filename)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except FileNotFoundError as exc:
                    self._send_json(404, {"ok": False, "error": str(exc)})
                    return
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": f"產生縮圖失敗：{exc}"})
                    return
                self._send_file(path, "image/jpeg")
                return

            if raw_path.startswith("/analysis/"):
                rest = raw_path[len("/analysis/") :]
                filename = urllib.parse.unquote(rest)
                if "/" in rest or "/" in filename:
                    self._send_json(400, {"ok": False, "error": "無效的分析檔名"})
                    return
                try:
                    path, content_type = app.get_analysis_file(filename)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except FileNotFoundError as exc:
                    self._send_json(404, {"ok": False, "error": str(exc)})
                    return
                self._send_file(path, content_type)
                return

            self._send_json(404, {"ok": False, "error": "找不到指定的頁面或 API"})

        def do_POST(self) -> None:
            """分派處理所有 POST API 請求。"""
            raw_path = urllib.parse.urlsplit(self.path).path
            if self._has_unsafe_path(raw_path):
                self._send_json(400, {"ok": False, "error": "不允許的路徑"})
                return

            try:
                body = self._read_json_body()
            except ValueError as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
                return

            if raw_path == "/api/quit":
                self._send_json(200, {"ok": True, "message": "天空日記觀察站已結束，可以關閉這個視窗"})
                app.stop_event.set()
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return

            if raw_path == "/api/camera":
                address = body.get("address")
                if not isinstance(address, str) or not address.strip():
                    self._send_json(400, {"ok": False, "error": "請輸入相機網址或 IP 位址"})
                    return
                try:
                    camera_url, status = app.connect_camera(address)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except Exception:
                    self._send_json(
                        502,
                        {"ok": False, "error": "連不到相機，請確認電腦和相機連的是同一個 Wi-Fi"},
                    )
                    return
                self._send_json(200, {"ok": True, "camera": camera_url, "status": status})
                return

            if raw_path == "/api/camera/scan":
                found = app.scan_cameras()
                self._send_json(200, {"ok": True, "found": found})
                return

            if raw_path == "/api/camera/config":
                try:
                    result = app.update_camera_config(body)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except Exception:
                    self._send_json(
                        502,
                        {"ok": False, "error": "連不到相機，無法套用設定"},
                    )
                    return
                self._send_json(200, result)
                return

            if raw_path == "/api/sync":
                try:
                    started = app.start_sync()
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                if not started:
                    self._send_json(409, {"ok": False, "error": "目前正在同步照片，請稍候再試"})
                    return
                self._send_json(200, {"ok": True, "started": True})
                return

            if raw_path == "/api/share/settings":
                try:
                    share = app.save_share_settings(body)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, "share": share})
                return

            if raw_path.startswith("/api/day/") and raw_path.endswith("/analyze"):
                middle = raw_path[len("/api/day/") : -len("/analyze")]
                if "/" in middle:
                    self._send_json(404, {"ok": False, "error": "找不到指定路徑"})
                    return
                date = urllib.parse.unquote(middle)
                try:
                    summary = app.analyze_day_full(date)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except FileNotFoundError as exc:
                    self._send_json(404, {"ok": False, "error": str(exc)})
                    return
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": f"分析照片時發生錯誤：{exc}"})
                    return
                self._send_json(200, {"ok": True, "summary": summary})
                return

            if raw_path.startswith("/api/share/"):
                rest = raw_path[len("/api/share/") :]
                if "/" in rest:
                    self._send_json(404, {"ok": False, "error": "找不到指定路徑"})
                    return
                date = urllib.parse.unquote(rest)
                ok, message, status = app.share_day(date)
                if ok:
                    self._send_json(status, {"ok": True, "date": date, "message": message})
                else:
                    self._send_json(status, {"ok": False, "error": message})
                return

            self._send_json(404, {"ok": False, "error": "找不到指定的 API"})

    return SkyCamHandler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令列參數 --port、--data 與 --no-browser。"""
    default_data = Path.home() / "Documents" / "天空日記"
    parser = argparse.ArgumentParser(description="天空日記觀察站桌面程式")
    parser.add_argument("--port", type=int, default=0, help="本機伺服器埠號（預設 0 自動選擇）")
    parser.add_argument("--data", type=str, default=str(default_data), help="照片與資料儲存資料夾")
    parser.add_argument("--no-browser", action="store_true", help="啟動時不自動開啟瀏覽器")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """啟動天空日記觀察站本機網頁伺服器與自動分享排程。"""
    args = parse_args(argv)
    # Windows 沒有主控台視窗的安裝版裡 sys.stdout／sys.stderr 是 None，網頁伺服器寫紀錄會出錯、請求卡住；
    # 改寫到資料夾裡的 skydiary.log（也方便學生把紀錄傳給老師看）。
    if sys.stdout is None or sys.stderr is None:
        try:
            Path(args.data).mkdir(parents=True, exist_ok=True)
            log_file = open(Path(args.data) / "skydiary.log", "a", encoding="utf-8", buffering=1)
        except OSError:
            log_file = open(os.devnull, "w")
        if sys.stdout is None:
            sys.stdout = log_file
        if sys.stderr is None:
            sys.stderr = log_file
    bundled_ffmpeg = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "ffmpeg_bin"
    if bundled_ffmpeg.is_dir():          # 安裝版內附 ffmpeg（做縮時影片用），不必另外安裝
        os.environ["PATH"] = str(bundled_ffmpeg) + os.pathsep + os.environ.get("PATH", "")
    app = SkyCamApp(Path(args.data))
    handler_cls = make_handler(app)
    server = ReusableThreadingHTTPServer(("127.0.0.1", args.port), handler_cls)
    actual_port = server.server_address[1]
    url = f"http://127.0.0.1:{actual_port}/"

    def handle_stop_signal(signum, frame) -> None:
        """收到終止訊號時於背景安全停止伺服器並釋放埠號。"""
        app.stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    for sig_name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, handle_stop_signal)
            except (ValueError, OSError):
                pass

    app.start_auto_share_loop()
    print(f"天空日記觀察站已啟動：{url}", flush=True)

    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
