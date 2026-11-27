"""天空日記觀察站桌面程式主程式。"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser

import numpy as np
from PIL import Image

import models
import skycam_tool

APP_VERSION = "1.2.0"
SHARE_URL = ""
MAX_LIVE_CAMERAS = 6

TZ_TAIPEI = timezone(timedelta(hours=8))

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

VALID_TOPICS = ("sunset", "plant", "moon", "cloud", "shadow")

DATE_RE = re.compile(r"^\d{8}$")
PHOTO_RE = re.compile(r"^\d{6}(_d)?\.jpg$")
ANALYSIS_RE = re.compile(
    r"^\d{8}(?:\.csv|_sun_trail\.png|_sky_strip\.png|\.mp4|_summary\.json)$"
)
CLASS_CODE_RE = re.compile(r"^[A-Za-z0-9]{0,12}$")
TIME_COLON_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})$")
TIME_HHMM_RE = re.compile(r"^(\d{2}):(\d{2})(?::(\d{2}))?$")
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


def http_get_bytes(url: str, timeout: float = 15.0) -> bytes:
    """對指定網址發送 HTTP GET 並讀取原始位元組資料。"""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if not data:
        raise ValueError("相機回傳空的圖片資料")
    return data


def validate_live_camera_address(raw) -> str:
    """驗證即時畫面相機網址：必須以 http:// 或 https:// 開頭，並去掉結尾斜線。"""
    if not isinstance(raw, str):
        raise ValueError("相機網址必須以 http:// 或 https:// 開頭")
    text = raw.strip()
    if not (text.startswith("http://") or text.startswith("https://")):
        raise ValueError("相機網址必須以 http:// 或 https:// 開頭")
    text = text.rstrip("/")
    if any(ch.isspace() for ch in text):
        raise ValueError("相機網址格式不正確")
    parsed = urllib.parse.urlsplit(text)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("相機網址格式不正確")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("相機通訊埠格式不正確") from exc
    return text


def validate_camera_name(raw, default_name: str | None = None) -> str:
    """驗證相機名字：去空白後 1～20 字；若提供 default_name，空白時套用預設名。"""
    if raw is None:
        if default_name is not None:
            return default_name
        raise ValueError("相機名字長度必須是 1 到 20 個字")
    if not isinstance(raw, str):
        raise ValueError("相機名字格式不正確")
    cleaned = raw.strip()
    if not cleaned:
        if default_name is not None:
            return default_name
        raise ValueError("相機名字長度必須是 1 到 20 個字")
    if len(cleaned) > 20:
        raise ValueError("相機名字長度必須是 1 到 20 個字")
    return cleaned


def infer_camera_model(status: dict) -> str:
    """依相機 /status 判斷型號。"""
    raw_model = status.get("model")
    if isinstance(raw_model, str) and raw_model.strip():
        return raw_model.strip()
    fw = str(status.get("fw") or "").strip()
    if fw.startswith("amb-"):
        return "AMB82-MINI"
    return "XIAO ESP32-S3"


def model_min_interval_s(model: str | None) -> int:
    """依相機型號決定最小即時拍照間隔（秒）。"""
    return 2 if model == "AMB82-MINI" else 10


class _LiveCameraState:
    """保存單台即時相機的拍照鎖、型號記憶與最近一張 JPEG 快取。"""

    def __init__(self) -> None:
        self.fetch_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.model: str | None = None
        self.last_jpeg: bytes | None = None
        self.last_ts: float = 0.0


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


def parse_date_yyyymmdd(date_str: str) -> date:
    """驗證 8 位數字日期字串並回傳 date 物件。"""
    if not isinstance(date_str, str) or not DATE_RE.fullmatch(date_str):
        raise ValueError("日期格式必須是 8 位數字")
    try:
        return datetime.strptime(date_str, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError("日期不正確，請確認年月日") from exc


def parse_photo_datetime(date_str: str, filename: str) -> datetime:
    """將資料夾日期與照片檔名轉成臺灣時間（UTC+8）datetime。"""
    d = parse_date_yyyymmdd(date_str)
    if not isinstance(filename, str) or not PHOTO_RE.fullmatch(filename):
        raise ValueError("照片檔名格式不正確")
    hh = int(filename[0:2])
    mm = int(filename[2:4])
    ss = int(filename[4:6])
    if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
        raise ValueError("照片檔名中的時間不正確")
    return datetime(d.year, d.month, d.day, hh, mm, ss, tzinfo=TZ_TAIPEI)


def parse_hhmm_seconds(val) -> int | None:
    """將 HH:MM、HH:MM:SS 或 datetime 轉為午夜起算秒數。"""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.hour * 3600 + val.minute * 60 + val.second
    text = str(val).strip()
    match = TIME_HHMM_RE.fullmatch(text)
    if not match:
        return None
    hh = int(match.group(1))
    mm = int(match.group(2))
    ss = int(match.group(3) or "0")
    if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
        return None
    return hh * 3600 + mm * 60 + ss


def _is_finite_number(val) -> bool:
    """判斷是否為有限的整數或浮點數（排除布林值）。"""
    return not isinstance(val, bool) and isinstance(val, (int, float)) and math.isfinite(val)


def _parse_point(val, label: str) -> tuple[float, float]:
    """驗證二維座標點 [x, y]。"""
    if not isinstance(val, (list, tuple)) or len(val) != 2:
        raise ValueError(f"{label} 必須是兩個數字的座標 [x, y]")
    if not (_is_finite_number(val[0]) and _is_finite_number(val[1])):
        raise ValueError(f"{label} 的座標必須是有效數字")
    return float(val[0]), float(val[1])


def _parse_ruler(val) -> tuple[tuple[float, float], tuple[float, float]]:
    """驗證尺的兩端座標 [[x1, y1], [x2, y2]]。"""
    if not isinstance(val, (list, tuple)) or len(val) != 2:
        raise ValueError("尺的座標必須包含兩個端點")
    p1 = _parse_point(val[0], "尺的第一端")
    p2 = _parse_point(val[1], "尺的第二端")
    return p1, p2


def find_moon_in_gray(gray: np.ndarray) -> tuple[bool, float | None, float | None, int]:
    """在灰階圖中找月亮亮區：先用太陽偵測，找不到再以最亮 70%、面積 ≥ 20 尋找。"""
    if gray.size == 0:
        return False, None, None, 0
    peak = int(gray.max())
    found, sx, sy = skycam_tool._find_sun_in_gray(gray)
    if found:
        return True, float(sx), float(sy), peak
    if peak < 10:
        return False, None, None, peak

    thresh = peak * 0.7
    mask = gray >= thresh
    total = int(mask.sum())
    if total < 20 or total > gray.size * 0.5:
        return False, None, None, peak

    row_has = mask.any(axis=1)
    active_rows = np.flatnonzero(row_has)
    parent: list[int] = []
    areas: list[int] = []
    sum_xs: list[float] = []
    sum_ys: list[float] = []

    def find_root(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        ra = find_root(a)
        rb = find_root(b)
        if ra != rb:
            if areas[ra] < areas[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            areas[ra] += areas[rb]
            sum_xs[ra] += sum_xs[rb]
            sum_ys[ra] += sum_ys[rb]

    prev_row = -2
    prev_runs: list[tuple[int, int, int]] = []
    width = gray.shape[1]
    padded = np.empty(width + 2, dtype=bool)
    padded[0] = False
    padded[-1] = False

    for r_np in active_rows:
        r = int(r_np)
        padded[1:-1] = mask[r]
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        starts = changes[0::2]
        ends = changes[1::2]
        curr_runs: list[tuple[int, int, int]] = []
        for s_np, e_np in zip(starts, ends):
            s = int(s_np)
            e = int(e_np)
            length = e - s
            nid = len(parent)
            parent.append(nid)
            areas.append(length)
            sum_xs.append(0.5 * (s + e - 1) * length)
            sum_ys.append(float(r * length))
            curr_runs.append((s, e, nid))

        if r == prev_row + 1 and prev_runs:
            i = 0
            j = 0
            n_prev = len(prev_runs)
            n_curr = len(curr_runs)
            while i < n_prev and j < n_curr:
                s1, e1, id1 = prev_runs[i]
                s2, e2, id2 = curr_runs[j]
                if s1 <= e2 and s2 <= e1:
                    union(id1, id2)
                if e1 <= e2:
                    i += 1
                else:
                    j += 1

        prev_row = r
        prev_runs = curr_runs

    best_root = -1
    best_area = 0
    for idx in range(len(parent)):
        if parent[idx] == idx and areas[idx] > best_area:
            best_area = areas[idx]
            best_root = idx

    if best_root >= 0 and best_area >= 20:
        cx = sum_xs[best_root] / best_area
        cy = sum_ys[best_root] / best_area
        return True, cx, cy, peak
    return False, None, None, peak


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


def compute_summary(date_str: str, rows: list[dict]) -> dict:
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
        "date": date_str,
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
        return {}

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
        self.plant_dir = self.data_dir / "plant"
        self.moon_dir = self.data_dir / "moon"
        self.cloud_dir = self.data_dir / "cloud"
        self.shadow_dir = self.data_dir / "shadow"
        self.settings_path = self.data_dir / "settings.json"
        self.share_log_path = self.data_dir / "share_log.jsonl"

        self.photos_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        self.plant_dir.mkdir(parents=True, exist_ok=True)
        self.moon_dir.mkdir(parents=True, exist_ok=True)
        self.cloud_dir.mkdir(parents=True, exist_ok=True)
        self.shadow_dir.mkdir(parents=True, exist_ok=True)

        self._settings_lock = threading.Lock()
        self._sync_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._fw_lock = threading.Lock()
        self._plant_lock = threading.Lock()
        self._moon_lock = threading.Lock()
        self._cloud_lock = threading.Lock()
        self._shadow_lock = threading.Lock()
        self._cameras_lock = threading.Lock()
        self._live_map_lock = threading.Lock()

        self._live_states: dict[str, _LiveCameraState] = {}
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
            "cameras": [],
            "device_id": uuid.uuid4().hex,
            "location": "臺北市",
            "topic": "sunset",
            "plant": {"roi": None},
            "cloud": {"horizon_y": None, "threshold": 0.75},
            "share": {
                "enabled": False,
                "consent": False,
                "nickname": "",
                "class_code": "",
                "county": "",
                "auto": True,
            },
        }

    def _fill_missing_settings(self, loaded: dict) -> tuple[dict, bool]:
        """補齊缺少的設定鍵並回傳（補齊後字典, 是否有變更）。"""
        defaults = self._default_settings()
        changed = False
        if "camera" not in loaded or not isinstance(loaded["camera"], str):
            loaded["camera"] = defaults["camera"]
            changed = True
        if "cameras" not in loaded or not isinstance(loaded["cameras"], list):
            cam_addr = loaded["camera"].strip().rstrip("/")
            if cam_addr:
                loaded["cameras"] = [{"address": cam_addr, "name": "相機 1"}]
            else:
                loaded["cameras"] = []
            changed = True
        else:
            cleaned_cams: list[dict] = []
            cams_changed = False
            for item in loaded["cameras"]:
                if len(cleaned_cams) >= MAX_LIVE_CAMERAS:
                    cams_changed = True
                    break
                if (
                    isinstance(item, dict)
                    and isinstance(item.get("address"), str)
                    and item["address"].strip()
                ):
                    addr = item["address"].strip().rstrip("/")
                    raw_name = item.get("name")
                    if isinstance(raw_name, str) and raw_name.strip():
                        name = raw_name.strip()[:20]
                    else:
                        name = f"相機 {len(cleaned_cams) + 1}"
                    if addr != item.get("address") or name != item.get("name"):
                        cams_changed = True
                    cleaned_cams.append({"address": addr, "name": name})
                else:
                    cams_changed = True
            if cams_changed:
                loaded["cameras"] = cleaned_cams
                changed = True
        if not loaded.get("device_id") or not isinstance(loaded["device_id"], str):
            loaded["device_id"] = defaults["device_id"]
            changed = True
        if (
            "location" not in loaded
            or not isinstance(loaded["location"], str)
            or loaded["location"] not in models.COUNTY_COORDS
        ):
            loaded["location"] = defaults["location"]
            changed = True
        if (
            "topic" not in loaded
            or not isinstance(loaded["topic"], str)
            or loaded["topic"] not in VALID_TOPICS
        ):
            loaded["topic"] = defaults["topic"]
            changed = True

        plant = loaded.get("plant")
        if not isinstance(plant, dict):
            plant = {}
            loaded["plant"] = plant
            changed = True
        if "roi" not in plant:
            plant["roi"] = None
            changed = True

        cloud = loaded.get("cloud")
        if not isinstance(cloud, dict):
            cloud = {}
            loaded["cloud"] = cloud
            changed = True
        if "horizon_y" not in cloud:
            cloud["horizon_y"] = None
            changed = True
        if "threshold" not in cloud or not _is_finite_number(cloud["threshold"]):
            cloud["threshold"] = 0.75
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

        return loaded, changed

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

            loaded, changed = self._fill_missing_settings(loaded)
            if changed:
                atomic_write_text(
                    self.settings_path,
                    json.dumps(loaded, ensure_ascii=False, indent=2),
                )

    def get_settings(self) -> dict:
        """安全讀取目前的 settings.json 內容，缺少新鍵時自動補預設值。"""
        with self._settings_lock:
            try:
                data = json.loads(self.settings_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data, changed = self._fill_missing_settings(data)
                    if changed:
                        atomic_write_text(
                            self.settings_path,
                            json.dumps(data, ensure_ascii=False, indent=2),
                        )
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

    def get_location_coords(self) -> tuple[str, float, float]:
        """取得目前設定縣市與其經緯度。"""
        settings = self.get_settings()
        loc = settings.get("location", "臺北市")
        if loc not in models.COUNTY_COORDS:
            loc = "臺北市"
        lat, lon = models.COUNTY_COORDS[loc]
        return loc, float(lat), float(lon)

    def set_location(self, county: str) -> str:
        """驗證並儲存所在縣市。"""
        if not isinstance(county, str) or county.strip() not in models.COUNTY_COORDS:
            raise ValueError("請從清單中選擇正確的縣市")
        cleaned = county.strip()
        settings = self.get_settings()
        settings["location"] = cleaned
        self.save_settings(settings)
        return cleaned

    def set_topic(self, topic: str, apply_camera: bool = False) -> dict:
        """儲存觀察主題，並在需要時代理更新相機的拍照時間表。"""
        if not isinstance(topic, str) or topic not in VALID_TOPICS:
            raise ValueError("觀察主題必須是 sunset、plant、moon、cloud 或 shadow")
        if not isinstance(apply_camera, bool):
            raise ValueError("apply_camera 必須是布林值")

        settings = self.get_settings()
        settings["topic"] = topic
        self.save_settings(settings)

        result: dict = {
            "ok": True,
            "topic": topic,
            "camera_schedule": None,
        }
        if not apply_camera:
            return result

        camera_url = settings.get("camera", "").strip()
        if not camera_url:
            result["warning"] = "尚未設定相機網址，已先儲存觀察主題"
            return result

        try:
            sched = http_get_json(f"{camera_url}/schedule?preset={topic}", timeout=5.0)
            result["camera_schedule"] = sched
        except Exception:
            result["warning"] = "連不到相機，已儲存觀察主題，但尚未更改相機的拍照時間表"
        return result

    def fetch_camera_schedule(self) -> dict:
        """代理查詢目前已設定相機的 /schedule。"""
        camera_url = self.get_settings().get("camera", "").strip()
        if not camera_url:
            raise ValueError("尚未設定相機網址，請先連線相機")
        return http_get_json(f"{camera_url}/schedule", timeout=5.0)

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

    # -------------------- 多台即時畫面相機 --------------------

    @staticmethod
    def _format_cameras(cameras: list[dict]) -> list[dict]:
        """將設定中的相機清單轉成含 index 的回應格式。"""
        return [
            {
                "index": idx,
                "address": str(cam.get("address", "")),
                "name": str(cam.get("name", f"相機 {idx + 1}")),
            }
            for idx, cam in enumerate(cameras)
        ]

    def _get_live_state(self, address: str) -> _LiveCameraState:
        """依相機網址取得或建立對應的即時狀態物件。"""
        key = address.rstrip("/")
        with self._live_map_lock:
            st = self._live_states.get(key)
            if st is None:
                st = _LiveCameraState()
                self._live_states[key] = st
            return st

    def _drop_live_state(self, address: str) -> None:
        """移除已刪除相機的即時快取狀態。"""
        key = address.rstrip("/")
        with self._live_map_lock:
            self._live_states.pop(key, None)

    def list_cameras(self) -> list[dict]:
        """回傳目前的即時畫面相機清單。"""
        settings = self.get_settings()
        cameras = settings.get("cameras", [])
        return self._format_cameras(cameras)

    def add_camera(self, body: dict) -> list[dict]:
        """新增一台即時畫面相機到清單並存檔。"""
        if not isinstance(body, dict):
            raise ValueError("相機資料格式不正確")
        address = validate_live_camera_address(body.get("address"))
        with self._cameras_lock:
            settings = self.get_settings()
            cameras = list(settings.get("cameras", []))
            if any(str(c.get("address", "")).rstrip("/") == address for c in cameras):
                raise ValueError("這台相機已經在清單裡")
            if len(cameras) >= MAX_LIVE_CAMERAS:
                raise ValueError(f"最多只能加入 {MAX_LIVE_CAMERAS} 台相機")
            default_name = f"相機 {len(cameras) + 1}"
            name = validate_camera_name(body.get("name"), default_name=default_name)
            cameras.append({"address": address, "name": name})
            settings["cameras"] = cameras
            self.save_settings(settings)
            return self._format_cameras(cameras)

    def remove_camera(self, body: dict) -> list[dict]:
        """依 index 刪除一台相機並存檔。"""
        if not isinstance(body, dict):
            raise ValueError("刪除參數格式不正確")
        index = body.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("相機編號不正確")
        with self._cameras_lock:
            settings = self.get_settings()
            cameras = list(settings.get("cameras", []))
            if not (0 <= index < len(cameras)):
                raise ValueError("相機編號不正確")
            removed = cameras.pop(index)
            settings["cameras"] = cameras
            self.save_settings(settings)
        self._drop_live_state(str(removed.get("address", "")))
        return self._format_cameras(cameras)

    def rename_camera(self, body: dict) -> list[dict]:
        """依 index 修改相機名字並存檔。"""
        if not isinstance(body, dict):
            raise ValueError("改名參數格式不正確")
        index = body.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("相機編號不正確")
        name = validate_camera_name(body.get("name"), default_name=None)
        with self._cameras_lock:
            settings = self.get_settings()
            cameras = list(settings.get("cameras", []))
            if not (0 <= index < len(cameras)):
                raise ValueError("相機編號不正確")
            cameras[index] = {
                "address": str(cameras[index].get("address", "")),
                "name": name,
            }
            settings["cameras"] = cameras
            self.save_settings(settings)
            return self._format_cameras(cameras)

    def add_scanned_cameras(self) -> tuple[int, list[dict]]:
        """掃描區網相機，把清單裡沒有的加入（總數最多 6 台）。"""
        found = self.scan_cameras()
        with self._cameras_lock:
            settings = self.get_settings()
            cameras = list(settings.get("cameras", []))
            existing = {str(c.get("address", "")).rstrip("/") for c in cameras}
            added = 0
            for item in found:
                if len(cameras) >= MAX_LIVE_CAMERAS:
                    break
                if not isinstance(item, dict):
                    continue
                try:
                    addr = validate_live_camera_address(item.get("address"))
                except ValueError:
                    continue
                if addr in existing:
                    continue
                cameras.append({"address": addr, "name": f"相機 {len(cameras) + 1}"})
                existing.add(addr)
                added += 1
            if added > 0:
                settings["cameras"] = cameras
                self.save_settings(settings)
            return added, self._format_cameras(cameras)

    def get_live_status(self) -> dict:
        """同時查詢清單中所有相機的 /status（每台逾時 3 秒）。"""
        cameras = self.list_cameras()
        if not cameras:
            return {"ok": True, "cameras": []}

        def query_one(cam: dict) -> dict:
            address = cam["address"].rstrip("/")
            st = self._get_live_state(address)
            try:
                data = http_get_json(f"{address}/status", timeout=3.0)
            except Exception:
                with st.data_lock:
                    known_model = st.model
                    st.last_jpeg = None
                    st.last_ts = 0.0
                return {
                    "index": cam["index"],
                    "name": cam["name"],
                    "address": cam["address"],
                    "online": False,
                    "fw": None,
                    "model": None,
                    "rssi": None,
                    "chip_temp_c": None,
                    "shots_today": None,
                    "last_file": None,
                    "min_interval_s": model_min_interval_s(known_model),
                }

            model = infer_camera_model(data)
            min_int = model_min_interval_s(model)
            with st.data_lock:
                st.model = model
            rssi = data.get("rssi") if "rssi" in data else data.get("wifi_rssi")
            return {
                "index": cam["index"],
                "name": cam["name"],
                "address": cam["address"],
                "online": True,
                "fw": data.get("fw"),
                "model": model,
                "rssi": rssi,
                "chip_temp_c": data.get("chip_temp_c"),
                "shots_today": data.get("shots_today"),
                "last_file": data.get("last_file"),
                "min_interval_s": min_int,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cameras)) as executor:
            results = list(executor.map(query_one, cameras))
        return {"ok": True, "cameras": results}

    def get_live_jpeg(self, index: int) -> tuple[bytes, str]:
        """代理抓取指定相機的即時 JPEG（不含 save=1，附最小間隔節流與每台鎖）。"""
        cameras = self.list_cameras()
        if not (0 <= index < len(cameras)):
            raise FileNotFoundError("找不到這台相機")
        address = cameras[index]["address"].rstrip("/")
        st = self._get_live_state(address)

        with st.fetch_lock:
            with st.data_lock:
                known_model = st.model
                cached_jpeg = st.last_jpeg
                cached_ts = st.last_ts

            if known_model is None:
                try:
                    status = http_get_json(f"{address}/status", timeout=3.0)
                    known_model = infer_camera_model(status)
                    with st.data_lock:
                        st.model = known_model
                except Exception:
                    known_model = None

            min_int = model_min_interval_s(known_model)
            if cached_jpeg is not None and (time.monotonic() - cached_ts) < min_int:
                return cached_jpeg, "hit"

            try:
                jpeg_bytes = http_get_bytes(f"{address}/capture", timeout=15.0)
            except Exception as exc:
                with st.data_lock:
                    st.last_jpeg = None
                    st.last_ts = 0.0
                raise ConnectionError("相機沒有回應") from exc

            with st.data_lock:
                st.last_jpeg = jpeg_bytes
                st.last_ts = time.monotonic()
            return jpeg_bytes, "miss"

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

    def _safe_photo_path(self, date_str: str, filename: str) -> Path:
        """驗證日期與照片檔名並回傳受限於照片資料夾內的絕對路徑。"""
        if not DATE_RE.fullmatch(date_str) or not PHOTO_RE.fullmatch(filename):
            raise ValueError("無效的日期或照片檔名")
        target = (self.photos_dir / date_str / filename).resolve()
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

    def get_day_info(self, date_str: str) -> dict:
        """回傳指定日期的照片清單、分析檔案存在狀態與摘要。"""
        if not DATE_RE.fullmatch(date_str):
            raise ValueError("日期格式必須是 8 位數字")
        day_dir = (self.photos_dir / date_str).resolve()
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
            "csv": (self.analysis_dir / f"{date_str}.csv").is_file(),
            "sun_trail": (self.analysis_dir / f"{date_str}_sun_trail.png").is_file(),
            "sky_strip": (self.analysis_dir / f"{date_str}_sky_strip.png").is_file(),
            "timelapse": (self.analysis_dir / f"{date_str}.mp4").is_file(),
        }

        summary = self._load_summary(date_str)
        return {
            "ok": True,
            "date": date_str,
            "photos": photos,
            "analysis": analysis,
            "summary": summary,
        }

    def _load_summary(self, date_str: str) -> dict | None:
        """讀取指定日期的摘要 JSON，若不存在或損毀則回傳 None。"""
        summary_path = self.analysis_dir / f"{date_str}_summary.json"
        if not summary_path.is_file():
            return None
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def analyze_day_full(self, date_str: str) -> dict:
        """執行單日照片分析、產生圖表與縮時影片並儲存摘要 JSON。"""
        if not DATE_RE.fullmatch(date_str):
            raise ValueError("日期格式必須是 8 位數字")
        day_dir = (self.photos_dir / date_str).resolve()
        if not day_dir.is_relative_to(self.photos_dir) or not day_dir.is_dir():
            raise FileNotFoundError("找不到這一天的照片資料夾")

        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.analysis_dir / f"{date_str}.csv"
        sun_trail_path = self.analysis_dir / f"{date_str}_sun_trail.png"
        sky_strip_path = self.analysis_dir / f"{date_str}_sky_strip.png"
        timelapse_path = self.analysis_dir / f"{date_str}.mp4"
        summary_path = self.analysis_dir / f"{date_str}_summary.json"

        rows = skycam_tool.analyze_day(str(day_dir))
        skycam_tool.write_csv(rows, str(csv_path))
        skycam_tool.make_sun_trail(str(day_dir), str(sun_trail_path))
        skycam_tool.make_sky_strip(str(day_dir), str(sky_strip_path))
        try:
            skycam_tool.make_timelapse(str(day_dir), str(timelapse_path))
        except Exception:
            pass

        summary = compute_summary(date_str, rows)
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

    def get_compare_with_models(self) -> dict:
        """回傳多天比較摘要與對應日期的模型日落預測。"""
        days = self.get_compare()
        _loc, lat, lon = self.get_location_coords()
        models_map: dict[str, dict] = {}
        for item in days:
            d_str = str(item.get("date", ""))
            try:
                d_obj = parse_date_yyyymmdd(d_str)
            except ValueError:
                continue
            ev = models.sun_events(d_obj, lat, lon, tz_hours=8)
            models_map[d_str] = {
                "sunset": ev.get("sunset"),
                "sunset_az": ev.get("sunset_az"),
            }
        return {"ok": True, "days": days, "models": models_map}

    def get_sunset_model(self, date_str: str) -> dict:
        """計算單日模型日出日落資料並與觀測最後看到太陽時間比較。"""
        d_obj = parse_date_yyyymmdd(date_str)
        location, lat, lon = self.get_location_coords()
        ev = models.sun_events(d_obj, lat, lon, tz_hours=8)

        summary = self._load_summary(date_str)
        raw_obs = summary.get("last_sun_time") if summary else None
        observed_last_sun_time = str(raw_obs).strip() if raw_obs else None
        if observed_last_sun_time == "":
            observed_last_sun_time = None

        obs_sec = parse_hhmm_seconds(observed_last_sun_time)
        model_sec = parse_hhmm_seconds(ev.get("sunset"))
        if obs_sec is not None and model_sec is not None:
            diff_minutes: float | None = round((obs_sec - model_sec) / 60.0, 1)
        else:
            diff_minutes = None

        return {
            "ok": True,
            "date": date_str,
            "location": location,
            "model": ev,
            "observed_last_sun_time": observed_last_sun_time,
            "diff_minutes": diff_minutes,
        }

    # -------------------- 植物生長 --------------------

    def set_plant_roi(self, body: dict) -> dict:
        """驗證並儲存植物觀察的框選範圍（1600×1200 原圖像素座標）。"""
        if not isinstance(body, dict):
            raise ValueError("框選範圍格式不正確")
        coords: dict[str, int] = {}
        for key in ("x0", "y0", "x1", "y1"):
            if key not in body:
                raise ValueError("請提供完整的 x0、y0、x1、y1 座標")
            val = body[key]
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(f"{key} 必須是整數")
            coords[key] = val

        x0, y0, x1, y1 = coords["x0"], coords["y0"], coords["x1"], coords["y1"]
        if not (0 <= x0 < x1 <= 1600 and 0 <= y0 < y1 <= 1200):
            raise ValueError("框選範圍必須在 1600×1200 以內，且 x0 < x1、y0 < y1")

        roi = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
        settings = self.get_settings()
        settings["plant"]["roi"] = roi
        self.save_settings(settings)
        return roi

    def _load_json_dict(self, path: Path) -> dict:
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _load_json_list(self, path: Path) -> list:
        if not path.is_file():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def get_plant_gcc(self, from_date: str | None, to_date: str | None) -> dict:
        """計算日期區間內每張一般照片的綠度（GCC）並快取結果。"""
        all_days = sorted(self.list_days())
        if from_date is not None and from_date != "":
            parse_date_yyyymmdd(from_date)
        else:
            from_date = all_days[0] if all_days else None
        if to_date is not None and to_date != "":
            parse_date_yyyymmdd(to_date)
        else:
            to_date = all_days[-1] if all_days else None

        if from_date and to_date and from_date > to_date:
            raise ValueError("開始日期不能晚於結束日期")

        settings = self.get_settings()
        roi = settings.get("plant", {}).get("roi")
        if isinstance(roi, dict) and all(k in roi for k in ("x0", "y0", "x1", "y1")):
            roi_key = f"{roi['x0']},{roi['y0']},{roi['x1']},{roi['y1']}"
        else:
            roi = None
            roi_key = "full"

        target_days = [
            d for d in all_days if (from_date is None or d >= from_date) and (to_date is None or d <= to_date)
        ]

        cache_path = self.plant_dir / "gcc_cache.json"
        with self._plant_lock:
            cache = self._load_json_dict(cache_path)

        tasks: list[tuple[str, str, Path, str]] = []
        cached_points: dict[tuple[str, str], float] = {}

        for d in target_days:
            day_dir = self.photos_dir / d
            if not day_dir.is_dir():
                continue
            for entry in sorted(day_dir.iterdir(), key=lambda p: p.name):
                if not entry.is_file() or not PHOTO_RE.fullmatch(entry.name):
                    continue
                if entry.name.endswith("_d.jpg"):
                    continue
                try:
                    st = entry.stat()
                except OSError:
                    continue
                ckey = f"{d}/{entry.name}|{roi_key}|{st.st_mtime_ns}|{st.st_size}"
                if ckey in cache and _is_finite_number(cache[ckey]):
                    cached_points[(d, entry.name)] = float(cache[ckey])
                else:
                    tasks.append((d, entry.name, entry, ckey))

        def compute_one(item: tuple[str, str, Path, str]) -> tuple[str, str, str, float] | None:
            d_s, fname, path, ckey = item
            try:
                with Image.open(path) as img:
                    rgb = np.asarray(img.convert("RGB"))
                if roi is not None:
                    h, w = rgb.shape[:2]
                    x0 = max(0, min(w, int(roi["x0"])))
                    x1 = max(0, min(w, int(roi["x1"])))
                    y0 = max(0, min(h, int(roi["y0"])))
                    y1 = max(0, min(h, int(roi["y1"])))
                    if x0 < x1 and y0 < y1:
                        mask = np.zeros((h, w), dtype=bool)
                        mask[y0:y1, x0:x1] = True
                        gcc_val = models.green_chromatic_coordinate(rgb, mask=mask)
                    else:
                        gcc_val = models.green_chromatic_coordinate(rgb, mask=None)
                else:
                    gcc_val = models.green_chromatic_coordinate(rgb, mask=None)
                return d_s, fname, ckey, round(float(gcc_val), 6)
            except Exception:
                return None

        new_entries: dict[str, float] = {}
        if tasks:
            if len(tasks) > 4:
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(compute_one, tasks))
            else:
                results = [compute_one(t) for t in tasks]
            for res in results:
                if res is not None:
                    d_s, fname, ckey, val = res
                    cached_points[(d_s, fname)] = val
                    new_entries[ckey] = val

        if new_entries:
            with self._plant_lock:
                latest_cache = self._load_json_dict(cache_path)
                latest_cache.update(new_entries)
                atomic_write_text(cache_path, json.dumps(latest_cache, ensure_ascii=False))

        points: list[dict] = []
        for d_s, fname in sorted(cached_points.keys()):
            points.append(
                {
                    "date": d_s,
                    "time": f"{fname[0:2]}:{fname[2:4]}:{fname[4:6]}",
                    "gcc": cached_points[(d_s, fname)],
                }
            )
        return {"ok": True, "roi": roi, "points": points}

    def save_plant_mark(self, body: dict) -> dict:
        """根據尺與植物上下端像素位置計算植物高度並儲存標記。"""
        if not isinstance(body, dict):
            raise ValueError("標記資料格式不正確")
        date_str = body.get("date")
        name = body.get("name")
        parse_photo_datetime(date_str, name)

        r1, r2 = _parse_ruler(body.get("ruler"))
        ruler_cm = body.get("ruler_cm")
        if not _is_finite_number(ruler_cm) or float(ruler_cm) <= 0:
            raise ValueError("尺的長度必須是大於 0 的數字")

        base = _parse_point(body.get("base"), "植物底部")
        top = _parse_point(body.get("top"), "植物頂端")

        ruler_px = math.hypot(r2[0] - r1[0], r2[1] - r1[1])
        if ruler_px <= 0:
            raise ValueError("尺的兩端座標不能重合")

        px_per_cm = ruler_px / float(ruler_cm)
        plant_px = math.hypot(top[0] - base[0], top[1] - base[1])
        height_cm = round(plant_px / px_per_cm, 2)
        px_per_cm_r = round(px_per_cm, 4)

        entry = {
            "date": date_str,
            "name": name,
            "ruler": [[r1[0], r1[1]], [r2[0], r2[1]]],
            "ruler_cm": float(ruler_cm),
            "base": [base[0], base[1]],
            "top": [top[0], top[1]],
            "px_per_cm": px_per_cm_r,
            "height_cm": height_cm,
        }

        marks_path = self.plant_dir / "marks.json"
        with self._plant_lock:
            marks = self._load_json_list(marks_path)
            marks = [
                m
                for m in marks
                if not (isinstance(m, dict) and m.get("date") == date_str and m.get("name") == name)
            ]
            marks.append(entry)
            marks.sort(key=lambda m: (str(m.get("date", "")), str(m.get("name", ""))))
            atomic_write_text(marks_path, json.dumps(marks, ensure_ascii=False, indent=2))

        return {"ok": True, "height_cm": height_cm, "px_per_cm": px_per_cm_r}

    def get_plant_growth(self) -> dict:
        """回傳植物高度紀錄並在達 4 個不同日期時做邏輯斯生長曲線擬合。"""
        marks_path = self.plant_dir / "marks.json"
        with self._plant_lock:
            raw_marks = self._load_json_list(marks_path)

        valid_marks: list[tuple[datetime, dict]] = []
        for m in raw_marks:
            if not isinstance(m, dict):
                continue
            try:
                dt = parse_photo_datetime(str(m.get("date", "")), str(m.get("name", "")))
                h = float(m["height_cm"])
            except (ValueError, KeyError, TypeError):
                continue
            valid_marks.append((dt, {"date": str(m["date"]), "name": str(m["name"]), "height_cm": h}))

        valid_marks.sort(key=lambda item: (item[0], item[1]["name"]))
        points = [item[1] for item in valid_marks]
        distinct_dates = len({p["date"] for p in points})

        fit: dict | None = None
        if distinct_dates < 4:
            fit_note = "資料少於 4 天，還不能畫生長曲線"
        else:
            dt0 = valid_marks[0][0]
            ts = [(dt - dt0).total_seconds() / 86400.0 for dt, _ in valid_marks]
            ys = [p["height_cm"] for p in points]
            try:
                raw_fit = models.logistic_fit(ts, ys)
                fit = {
                    "K": round(float(raw_fit["K"]), 4),
                    "r": round(float(raw_fit["r"]), 4),
                    "t0": round(float(raw_fit["t0"]), 4),
                    "r2": round(float(raw_fit["r2"]), 4),
                }
                fit_note = (
                    f"最多長到約 {fit['K']:.1f} 公分、長最快的是第 {fit['t0']:.1f} 天"
                    f"（吻合度 R² = {fit['r2']:.2f}）"
                )
            except Exception as exc:
                fit = None
                fit_note = f"目前資料還無法擬合生長曲線（{exc}）"

        return {"ok": True, "points": points, "fit": fit, "fit_note": fit_note}

    # -------------------- 月亮 --------------------

    def get_moon_day(self, date_str: str) -> dict:
        """分析指定日期夜間暗曝光照片中的月亮位置並附上模型預測。"""
        d_obj = parse_date_yyyymmdd(date_str)
        day_dir = (self.photos_dir / date_str).resolve()
        if not day_dir.is_relative_to(self.photos_dir) or not day_dir.is_dir():
            raise FileNotFoundError("找不到這一天的照片資料夾")

        location, lat, lon = self.get_location_coords()

        cache_path = self.moon_dir / "moon_cache.json"
        with self._moon_lock:
            cache = self._load_json_dict(cache_path)

        candidates: list[tuple[str, datetime, Path, str]] = []
        for entry in sorted(day_dir.iterdir(), key=lambda p: p.name):
            if not entry.is_file() or not PHOTO_RE.fullmatch(entry.name):
                continue
            if not entry.name.endswith("_d.jpg"):
                continue
            try:
                dt = parse_photo_datetime(date_str, entry.name)
            except ValueError:
                continue
            if not (dt.hour >= 18 or dt.hour <= 6):
                continue
            try:
                st = entry.stat()
            except OSError:
                continue
            ckey = f"{date_str}/{entry.name}|{st.st_mtime_ns}|{st.st_size}"
            candidates.append((entry.name, dt, entry, ckey))

        uncached = [c for c in candidates if c[3] not in cache]

        def detect_one(item: tuple[str, datetime, Path, str]) -> tuple[str, dict]:
            _name, _dt, path, ckey = item
            try:
                with Image.open(path) as img:
                    gray = np.asarray(img.convert("L"))
                found, cx, cy, peak = find_moon_in_gray(gray)
            except Exception:
                found, cx, cy, peak = False, None, None, 0
            return ckey, {
                "found": 1 if found else 0,
                "x": round(float(cx), 1) if found and cx is not None else None,
                "y": round(float(cy), 1) if found and cy is not None else None,
                "peak": int(peak),
            }

        new_entries: dict[str, dict] = {}
        if uncached:
            if len(uncached) > 4:
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(detect_one, uncached))
            else:
                results = [detect_one(u) for u in uncached]
            for ckey, det in results:
                cache[ckey] = det
                new_entries[ckey] = det

        if new_entries:
            with self._moon_lock:
                latest = self._load_json_dict(cache_path)
                latest.update(new_entries)
                atomic_write_text(cache_path, json.dumps(latest, ensure_ascii=False))

        photos_out: list[dict] = []
        found_count = 0
        for name, dt, _path, ckey in candidates:
            det = cache.get(ckey, {"found": 0, "x": None, "y": None, "peak": 0})
            alt, az = models.moon_position(dt, lat, lon)
            illum = models.moon_illumination(dt)
            is_found = 1 if det.get("found") else 0
            if is_found:
                found_count += 1
            photos_out.append(
                {
                    "name": name,
                    "time": f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}",
                    "found": is_found,
                    "x": det.get("x"),
                    "y": det.get("y"),
                    "peak": int(det.get("peak", 0)),
                    "model_alt": round(float(alt), 2),
                    "model_az": round(float(az), 2),
                    "fraction": round(float(illum["fraction"]), 3),
                    "phase_name": str(illum["name"]),
                }
            )

        events = models.moon_events(d_obj, lat, lon, tz_hours=8)
        dt_21 = datetime(d_obj.year, d_obj.month, d_obj.day, 21, 0, 0, tzinfo=TZ_TAIPEI)
        illum_21 = models.moon_illumination(dt_21)

        return {
            "ok": True,
            "date": date_str,
            "location": location,
            "photos": photos_out,
            "events": events,
            "summary": {
                "photos": len(photos_out),
                "found": found_count,
                "phase_name": str(illum_21["name"]),
                "fraction": round(float(illum_21["fraction"]), 3),
            },
            "waxing": bool(illum_21.get("waxing", True)),
        }

    def get_moon_calendar(self, from_str: str | None, days_str: str | None) -> dict:
        """產生未來 N 天月相與傍晚西方可見日期的月曆資料。"""
        if from_str is None or from_str == "":
            start_day = datetime.now(TZ_TAIPEI).date()
        else:
            start_day = parse_date_yyyymmdd(from_str)

        if days_str is None or days_str == "":
            n_days = 30
        else:
            try:
                n_days = int(days_str)
            except ValueError as exc:
                raise ValueError("days 必須是 1 到 45 之間的整數") from exc
            if not (1 <= n_days <= 45):
                raise ValueError("days 必須是 1 到 45 之間的整數")

        location, lat, lon = self.get_location_coords()
        entries: list[dict] = []
        for offset in range(n_days):
            d_obj = start_day + timedelta(days=offset)
            d_str = d_obj.strftime("%Y%m%d")
            dt_21 = datetime(d_obj.year, d_obj.month, d_obj.day, 21, 0, 0, tzinfo=TZ_TAIPEI)
            illum = models.moon_illumination(dt_21)
            ev = models.moon_events(d_obj, lat, lon, tz_hours=8)
            dt_1830 = datetime(d_obj.year, d_obj.month, d_obj.day, 18, 30, 0, tzinfo=TZ_TAIPEI)
            alt_1830, az_1830 = models.moon_position(dt_1830, lat, lon)
            evening_west = bool(alt_1830 > 5.0 and 200.0 <= az_1830 <= 340.0)
            entries.append(
                {
                    "date": d_str,
                    "phase_name": str(illum["name"]),
                    "fraction": round(float(illum["fraction"]), 3),
                    "waxing": bool(illum.get("waxing", True)),
                    "moonrise": ev.get("moonrise"),
                    "moonset": ev.get("moonset"),
                    "evening_west": evening_west,
                }
            )
        return {"ok": True, "location": location, "days": entries}

    # -------------------- 雲量 --------------------

    def save_cloud_settings(self, body: dict) -> dict:
        """驗證並儲存雲量計算用的地平線位置與亮度門檻。"""
        if not isinstance(body, dict):
            raise ValueError("雲量設定格式不正確")
        if "horizon_y" not in body and "threshold" not in body:
            raise ValueError("請提供 horizon_y 或 threshold 設定")
        unknown = set(body.keys()) - {"horizon_y", "threshold"}
        if unknown:
            raise ValueError("包含不支援的雲量設定項目")

        settings = self.get_settings()
        cloud_cfg = dict(settings.get("cloud", {"horizon_y": None, "threshold": 0.75}))

        if "horizon_y" in body:
            hy = body["horizon_y"]
            if hy is None:
                cloud_cfg["horizon_y"] = None
            elif isinstance(hy, bool) or not isinstance(hy, int) or not (0 <= hy <= 1199):
                raise ValueError("地平線高度 horizon_y 必須是 0 到 1199 的整數，或 null")
            else:
                cloud_cfg["horizon_y"] = hy

        if "threshold" in body:
            th = body["threshold"]
            if not _is_finite_number(th) or not (0.4 <= float(th) <= 1.2):
                raise ValueError("雲量門檻值 threshold 必須在 0.4 到 1.2 之間")
            cloud_cfg["threshold"] = round(float(th), 4)

        settings["cloud"] = cloud_cfg
        self.save_settings(settings)
        return cloud_cfg

    def _fetch_openmeteo_cloud(self, date_str: str, location: str, lat: float, lon: float) -> tuple[list[dict], str]:
        """取得 Open-Meteo 模型預報雲量並快取；離線或失敗時回傳空清單且不拋錯。"""
        if os.environ.get("SKYCAM_OFFLINE", "").strip() == "1":
            return [], "離線模式，沒有對照資料"

        cache_path = self.cloud_dir / f"openmeteo_{date_str}.json"
        with self._cloud_lock:
            cached = self._load_json_dict(cache_path)
        if cached.get("location") == location and isinstance(cached.get("reference"), list):
            ref = cached["reference"]
            if ref:
                return ref, "這是天氣模型算的，不是實際觀測（資料來源：Open-Meteo）"
            return [], "天氣模型沒有這一天的預報資料"

        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&hourly=cloud_cover"
            f"&timezone=Asia%2FTaipei&past_days=7&forecast_days=1"
        )
        try:
            data = http_get_json(url, timeout=8.0)
            hourly = data.get("hourly", {}) if isinstance(data, dict) else {}
            times = hourly.get("time", []) if isinstance(hourly, dict) else []
            covers = hourly.get("cloud_cover", []) if isinstance(hourly, dict) else []
            prefix = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}T"
            ref: list[dict] = []
            for t_val, c_val in zip(times, covers):
                if isinstance(t_val, str) and t_val.startswith(prefix) and _is_finite_number(c_val):
                    ref.append({"hour": t_val[11:16], "cloud_cover": float(c_val)})
            with self._cloud_lock:
                atomic_write_text(
                    cache_path,
                    json.dumps({"location": location, "reference": ref}, ensure_ascii=False, indent=2),
                )
            if ref:
                return ref, "這是天氣模型算的，不是實際觀測（資料來源：Open-Meteo）"
            return [], "天氣模型沒有這一天的預報資料（僅提供最近 7 天）"
        except Exception:
            return [], "目前無法連線取得天氣模型預報雲量，僅顯示相機觀測結果"

    def get_cloud_day(self, date_str: str) -> dict:
        """計算指定日期每張一般照片的雲量，並附帶 Open-Meteo 天氣模型預報對照。"""
        parse_date_yyyymmdd(date_str)
        day_dir = (self.photos_dir / date_str).resolve()
        if not day_dir.is_relative_to(self.photos_dir) or not day_dir.is_dir():
            raise FileNotFoundError("找不到這一天的照片資料夾")

        settings = self.get_settings()
        cloud_cfg = settings.get("cloud", {})
        horizon_y = cloud_cfg.get("horizon_y", None)
        threshold = float(cloud_cfg.get("threshold", 0.75))
        location, lat, lon = self.get_location_coords()

        cache_path = self.cloud_dir / "cloud_cache.json"
        with self._cloud_lock:
            cache = self._load_json_dict(cache_path)

        tasks: list[tuple[str, Path, str]] = []
        cached_vals: dict[str, float] = {}
        for entry in sorted(day_dir.iterdir(), key=lambda p: p.name):
            if not entry.is_file() or not PHOTO_RE.fullmatch(entry.name):
                continue
            if entry.name.endswith("_d.jpg"):
                continue
            try:
                st = entry.stat()
            except OSError:
                continue
            ckey = f"{date_str}/{entry.name}|{horizon_y}|{threshold:.4f}|{st.st_mtime_ns}|{st.st_size}"
            if ckey in cache and _is_finite_number(cache[ckey]):
                cached_vals[entry.name] = float(cache[ckey])
            else:
                tasks.append((entry.name, entry, ckey))

        def compute_one(item: tuple[str, Path, str]) -> tuple[str, str, float] | None:
            fname, path, ckey = item
            try:
                with Image.open(path) as img:
                    rgb = np.asarray(img.convert("RGB"))
                if horizon_y is not None:
                    h, w = rgb.shape[:2]
                    hy = max(0, min(h, int(horizon_y)))
                    if hy > 0:
                        mask = np.zeros((h, w), dtype=bool)
                        mask[:hy, :] = True
                        cf = models.cloud_fraction(rgb, mask=mask, threshold=threshold)
                    else:
                        cf = 0.0
                else:
                    cf = models.cloud_fraction(rgb, mask=None, threshold=threshold)
                return fname, ckey, round(float(cf), 4)
            except Exception:
                return None

        new_entries: dict[str, float] = {}
        if tasks:
            if len(tasks) > 4:
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(compute_one, tasks))
            else:
                results = [compute_one(t) for t in tasks]
            for res in results:
                if res is not None:
                    fname, ckey, val = res
                    cached_vals[fname] = val
                    new_entries[ckey] = val

        if new_entries:
            with self._cloud_lock:
                latest = self._load_json_dict(cache_path)
                latest.update(new_entries)
                atomic_write_text(cache_path, json.dumps(latest, ensure_ascii=False))

        points = [
            {
                "time": f"{fname[0:2]}:{fname[2:4]}:{fname[4:6]}",
                "cloud_fraction": cached_vals[fname],
            }
            for fname in sorted(cached_vals.keys())
        ]

        reference, reference_note = self._fetch_openmeteo_cloud(date_str, location, lat, lon)

        return {
            "ok": True,
            "date": date_str,
            "location": location,
            "horizon_y": horizon_y,
            "threshold": threshold,
            "points": points,
            "reference": reference,
            "reference_note": reference_note,
        }

    # -------------------- 立竿影子 --------------------

    def save_shadow_mark(self, body: dict) -> dict:
        """根據尺與立竿影子端點計算影長與實測太陽高度角，並和模型對照。"""
        if not isinstance(body, dict):
            raise ValueError("影子標記格式不正確")
        date_str = body.get("date")
        name = body.get("name")
        when = parse_photo_datetime(date_str, name)

        r1, r2 = _parse_ruler(body.get("ruler"))
        ruler_cm = body.get("ruler_cm")
        if not _is_finite_number(ruler_cm) or float(ruler_cm) <= 0:
            raise ValueError("尺的長度必須是大於 0 的數字")

        stick_cm = body.get("stick_cm")
        if not _is_finite_number(stick_cm) or float(stick_cm) <= 0:
            raise ValueError("棒子高度必須是大於 0 的數字")

        base = _parse_point(body.get("base"), "棒子底部")
        tip = _parse_point(body.get("tip"), "影子尖端")

        ruler_px = math.hypot(r2[0] - r1[0], r2[1] - r1[1])
        if ruler_px <= 0:
            raise ValueError("尺的兩端座標不能重合")

        px_per_cm = ruler_px / float(ruler_cm)
        shadow_px = math.hypot(tip[0] - base[0], tip[1] - base[1])
        shadow_cm = shadow_px / px_per_cm

        try:
            alt_measured = float(models.shadow_to_sun_altitude(float(stick_cm), shadow_cm))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"無法用目前的影長計算太陽高度角：{exc}") from exc

        location, lat, lon = self.get_location_coords()
        alt_model, _az_model = models.sun_position(when, lat, lon)
        alt_model_f = float(alt_model)
        diff_f = alt_measured - alt_model_f

        if alt_model_f <= 0:
            expected_val = None
        else:
            raw_exp = models.expected_shadow_length(float(stick_cm), alt_model_f)
            expected_val = round(float(raw_exp), 2) if raw_exp is not None else None

        shadow_cm_r = round(float(shadow_cm), 2)
        alt_measured_r = round(alt_measured, 2)
        alt_model_r = round(alt_model_f, 2)
        diff_r = round(diff_f, 2)

        entry = {
            "date": date_str,
            "name": name,
            "time": f"{when.hour:02d}:{when.minute:02d}:{when.second:02d}",
            "location": location,
            "ruler": [[r1[0], r1[1]], [r2[0], r2[1]]],
            "ruler_cm": float(ruler_cm),
            "base": [base[0], base[1]],
            "tip": [tip[0], tip[1]],
            "stick_cm": float(stick_cm),
            "shadow_cm": shadow_cm_r,
            "altitude_measured": alt_measured_r,
            "altitude_model": alt_model_r,
            "diff": diff_r,
            "expected_shadow_cm": expected_val,
        }

        marks_path = self.shadow_dir / "marks.json"
        with self._shadow_lock:
            marks = self._load_json_list(marks_path)
            marks = [
                m
                for m in marks
                if not (isinstance(m, dict) and m.get("date") == date_str and m.get("name") == name)
            ]
            marks.append(entry)
            marks.sort(key=lambda m: (str(m.get("date", "")), str(m.get("name", ""))))
            atomic_write_text(marks_path, json.dumps(marks, ensure_ascii=False, indent=2))

        return {
            "ok": True,
            "shadow_cm": shadow_cm_r,
            "altitude_measured": alt_measured_r,
            "altitude_model": alt_model_r,
            "diff": diff_r,
            "expected_shadow_cm": expected_val,
        }

    def list_shadow_marks(self) -> dict:
        """回傳所有立竿影子標記與實測、模型高度角統計比較。"""
        marks_path = self.shadow_dir / "marks.json"
        with self._shadow_lock:
            raw_marks = self._load_json_list(marks_path)

        marks = [m for m in raw_marks if isinstance(m, dict)]
        marks.sort(key=lambda m: (str(m.get("date", "")), str(m.get("name", ""))))

        obs = [float(m["altitude_measured"]) for m in marks if _is_finite_number(m.get("altitude_measured")) and _is_finite_number(m.get("altitude_model"))]
        pred = [float(m["altitude_model"]) for m in marks if _is_finite_number(m.get("altitude_measured")) and _is_finite_number(m.get("altitude_model"))]

        try:
            compare = models.compare_table(obs, pred)
        except Exception:
            compare = {"n": 0, "mean_diff": None, "mean_abs_diff": None, "max_abs_diff": None}

        return {"ok": True, "marks": marks, "compare": compare}

    def get_photo_path(self, date_str: str, filename: str) -> Path:
        """取得原始照片路徑，若不存在則拋出 FileNotFoundError。"""
        path = self._safe_photo_path(date_str, filename)
        if not path.is_file():
            raise FileNotFoundError("找不到這張照片")
        return path

    def get_or_make_thumb(self, date_str: str, filename: str) -> Path:
        """取得寬度 320 像素的縮圖路徑，尚未產生時自動建立快取。"""
        orig_path = self.get_photo_path(date_str, filename)
        thumb_path = (self.thumbs_dir / date_str / filename).resolve()
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

    def append_share_log(self, date_str: str, ok: bool, message: str) -> None:
        """在 share_log.jsonl 新增一筆分享紀錄。"""
        entry = {
            "at": now_str(),
            "date": date_str,
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

    def has_shared_successfully(self, date_str: str) -> bool:
        """檢查指定日期是否已經有成功分享的紀錄。"""
        with self._log_lock:
            if not self.share_log_path.is_file():
                return False
            lines = self.share_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines:
            try:
                item = json.loads(line)
                if isinstance(item, dict) and item.get("date") == date_str and item.get("ok") is True:
                    return True
            except json.JSONDecodeError:
                continue
        return False

    def share_day(self, date_str: str) -> tuple[bool, str, int]:
        """將指定日期的夕陽摘要與太陽軌跡縮圖送給老師並寫入紀錄。"""
        if not DATE_RE.fullmatch(date_str):
            return False, "日期格式必須是 8 位數字", 400

        share_url = get_share_url()
        if not share_url:
            return False, "老師還沒有設定接收網址", 400

        settings = self.get_settings()
        share = settings.get("share", {})
        if not (share.get("enabled") is True and share.get("consent") is True):
            return False, "請先在「分享給老師」頁面勾選同意分享並儲存設定", 400

        summary = self._load_summary(date_str)
        sun_trail_path = self.analysis_dir / f"{date_str}_sun_trail.png"
        if summary is None or not sun_trail_path.is_file():
            return False, "這一天還沒有完成夕陽分析，請先分析後再分享", 400

        try:
            image_b64 = encode_sun_trail_base64(sun_trail_path)
        except Exception as exc:
            msg = f"壓縮太陽軌跡圖失敗：{exc}"
            self.append_share_log(date_str, False, msg)
            return False, msg, 500

        payload = {
            "v": 1,
            "app": "skycam-desktop",
            "app_version": APP_VERSION,
            "device_id": str(settings.get("device_id", "")),
            "nickname": str(share.get("nickname", "")),
            "class_code": str(share.get("class_code", "")),
            "county": str(share.get("county", "")),
            "date": date_str,
            "camera_fw": self.get_camera_fw(),
            "summary": summary,
            "image": image_b64,
        }

        try:
            resp = post_json_follow_redirect(share_url, payload, timeout=30.0)
        except Exception as exc:
            msg = f"傳送失敗，請確認網路連線：{exc}"
            self.append_share_log(date_str, False, msg)
            return False, msg, 502

        if resp.get("ok") is True:
            msg = str(resp.get("message") or "分享成功")
            self.append_share_log(date_str, True, msg)
            return True, msg, 200

        msg = str(resp.get("error") or resp.get("message") or "接收端回應分享失敗")
        self.append_share_log(date_str, False, msg)
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

        def _send_bytes(
            self,
            status: int,
            content_type: str,
            data: bytes,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            """送出指定狀態碼、MIME 類型與位元組內容的回應。"""
            self.close_connection = True
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if extra_headers:
                for hk, hv in extra_headers.items():
                    self.send_header(hk, hv)
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
            parsed = urllib.parse.urlsplit(self.path)
            raw_path = parsed.path
            query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
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

            if raw_path == "/api/cameras":
                self._send_json(200, {"ok": True, "cameras": app.list_cameras()})
                return

            if raw_path == "/api/live/status":
                self._send_json(200, app.get_live_status())
                return

            if raw_path.startswith("/api/live/") and raw_path.endswith(".jpg"):
                idx_str = raw_path[len("/api/live/") : -len(".jpg")]
                if not idx_str.isdigit():
                    self._send_json(404, {"ok": False, "error": "找不到這台相機"})
                    return
                index = int(idx_str)
                try:
                    jpeg_bytes, cache_status = app.get_live_jpeg(index)
                except FileNotFoundError as exc:
                    self._send_json(404, {"ok": False, "error": str(exc)})
                    return
                except ConnectionError:
                    self._send_json(502, {"ok": False, "error": "相機沒有回應"})
                    return
                except Exception:
                    self._send_json(502, {"ok": False, "error": "相機沒有回應"})
                    return
                self._send_bytes(
                    200,
                    "image/jpeg",
                    jpeg_bytes,
                    extra_headers={"X-Live-Cache": cache_status},
                )
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

            if raw_path == "/api/camera/schedule":
                try:
                    sched = app.fetch_camera_schedule()
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except Exception:
                    self._send_json(
                        502,
                        {"ok": False, "error": "連不到相機，無法讀取拍照時間表"},
                    )
                    return
                self._send_json(200, sched)
                return

            if raw_path == "/api/sync":
                self._send_json(200, app.get_sync_status())
                return

            if raw_path == "/api/compare":
                self._send_json(200, app.get_compare_with_models())
                return

            if raw_path == "/api/share/log":
                self._send_json(200, {"ok": True, "logs": app.read_share_log(30)})
                return

            if raw_path.startswith("/api/sunset/model/"):
                rest = raw_path[len("/api/sunset/model/") :]
                if "/" in rest:
                    self._send_json(404, {"ok": False, "error": "找不到指定路徑"})
                    return
                date_str = urllib.parse.unquote(rest)
                try:
                    res = app.get_sunset_model(date_str)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, res)
                return

            if raw_path == "/api/plant/gcc":
                from_vals = query.get("from")
                to_vals = query.get("to")
                from_date = from_vals[0] if from_vals else None
                to_date = to_vals[0] if to_vals else None
                try:
                    res = app.get_plant_gcc(from_date, to_date)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, res)
                return

            if raw_path == "/api/plant/growth":
                self._send_json(200, app.get_plant_growth())
                return

            if raw_path.startswith("/api/moon/day/"):
                rest = raw_path[len("/api/moon/day/") :]
                if "/" in rest:
                    self._send_json(404, {"ok": False, "error": "找不到指定路徑"})
                    return
                date_str = urllib.parse.unquote(rest)
                try:
                    res = app.get_moon_day(date_str)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except FileNotFoundError as exc:
                    self._send_json(404, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, res)
                return

            if raw_path == "/api/moon/calendar":
                from_vals = query.get("from")
                days_vals = query.get("days")
                from_str = from_vals[0] if from_vals else None
                days_str = days_vals[0] if days_vals else None
                try:
                    res = app.get_moon_calendar(from_str, days_str)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, res)
                return

            if raw_path.startswith("/api/cloud/day/"):
                rest = raw_path[len("/api/cloud/day/") :]
                if "/" in rest:
                    self._send_json(404, {"ok": False, "error": "找不到指定路徑"})
                    return
                date_str = urllib.parse.unquote(rest)
                try:
                    res = app.get_cloud_day(date_str)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                except FileNotFoundError as exc:
                    self._send_json(404, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, res)
                return

            if raw_path == "/api/shadow/list":
                self._send_json(200, app.list_shadow_marks())
                return

            if raw_path.startswith("/api/day/"):
                rest = raw_path[len("/api/day/") :]
                if "/" in rest:
                    self._send_json(404, {"ok": False, "error": "找不到指定路徑"})
                    return
                date_str = urllib.parse.unquote(rest)
                try:
                    info = app.get_day_info(date_str)
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
                date_str = urllib.parse.unquote(parts[0])
                filename = urllib.parse.unquote(parts[1])
                try:
                    path = app.get_photo_path(date_str, filename)
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
                date_str = urllib.parse.unquote(parts[0])
                filename = urllib.parse.unquote(parts[1])
                try:
                    path = app.get_or_make_thumb(date_str, filename)
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

            if raw_path == "/api/location":
                county = body.get("county")
                try:
                    loc = app.set_location(county)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, "location": loc})
                return

            if raw_path == "/api/topic":
                topic = body.get("topic")
                apply_camera = body.get("apply_camera", False)
                try:
                    res = app.set_topic(topic, apply_camera=apply_camera)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, res)
                return

            if raw_path == "/api/plant/roi":
                try:
                    roi = app.set_plant_roi(body)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, "roi": roi})
                return

            if raw_path == "/api/plant/mark":
                try:
                    res = app.save_plant_mark(body)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, res)
                return

            if raw_path == "/api/cloud/settings":
                try:
                    cfg = app.save_cloud_settings(body)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, "cloud": cfg})
                return

            if raw_path == "/api/shadow/mark":
                try:
                    res = app.save_shadow_mark(body)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, res)
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

            if raw_path == "/api/cameras":
                try:
                    cameras = app.add_camera(body)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, "cameras": cameras})
                return

            if raw_path == "/api/cameras/remove":
                try:
                    cameras = app.remove_camera(body)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, "cameras": cameras})
                return

            if raw_path == "/api/cameras/rename":
                try:
                    cameras = app.rename_camera(body)
                except ValueError as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, "cameras": cameras})
                return

            if raw_path == "/api/cameras/add_scanned":
                added, cameras = app.add_scanned_cameras()
                self._send_json(200, {"ok": True, "added": added, "cameras": cameras})
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
                date_str = urllib.parse.unquote(middle)
                try:
                    summary = app.analyze_day_full(date_str)
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
                date_str = urllib.parse.unquote(rest)
                ok, message, status = app.share_day(date_str)
                if ok:
                    self._send_json(status, {"ok": True, "date": date_str, "message": message})
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
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass
    bundled_ffmpeg = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "ffmpeg_bin"
    if bundled_ffmpeg.is_dir():
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
