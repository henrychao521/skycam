"""天空日記相機電腦端工具：下載照片、分析太陽位置與天空顏色、製作軌跡圖、色條圖與縮時影片。"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import os
import re
import shutil
import time
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

# 符合相機命名規則的正規表示式：HHMMSS.jpg（一般）或 HHMMSS_d.jpg（暗曝光）
_PHOTO_NAME_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})(_d)?\.jpg$", re.IGNORECASE)
_DAY_DIR_RE = re.compile(r"^\d{8}$")

# CSV 輸出欄位順序
_CSV_COLUMNS = [
    "name",
    "time",
    "kind",
    "sun_found",
    "sun_x",
    "sun_y",
    "sky_r",
    "sky_g",
    "sky_b",
    "brightness",
]

# Pillow 版本相容的雙線性縮放設定
_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def _fetch_json(url: str, timeout: float) -> object:
    """向相機網址發送 GET 請求並解析回傳的 JSON 資料。"""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_one(url: str, dest: Path, expected_size: int, timeout: float,
                  chunk: int = 16384, max_fail: int = 12) -> bool:
    """分段下載單張照片：每次只要 16 KB（HTTP Range），斷掉就從 .part 已有的地方接著下載。
    網路不穩時只是變慢，不會整張重來；連續失敗 max_fail 次才放棄（.part 保留，下次再接著抓）。"""
    part_path = dest.with_name(f"{dest.name}.part")
    have = part_path.stat().st_size if part_path.exists() else 0
    if have > expected_size:
        part_path.unlink(); have = 0
    fails = 0
    while have < expected_size:
        end = min(have + chunk, expected_size) - 1
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                if resp.status == 200:                 # 舊版韌體不支援 Range：整張一次給
                    if len(data) == expected_size:
                        part_path.write_bytes(data); have = expected_size; break
                    raise OSError(f"整張下載只收到 {len(data)} 位元組")
            if len(data) != end - have + 1:
                raise OSError(f"這一段只收到 {len(data)}／{end - have + 1} 位元組")
            with open(part_path, "ab") as f:
                f.write(data)
            have += len(data); fails = 0
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
            fails += 1
            if fails >= max_fail:
                print(f"下載暫停：{dest.name} 已收 {have}／{expected_size} 位元組，連續失敗 {fails} 次（{exc}）；下次執行會接著抓")
                return False
            time.sleep(min(2 * fails, 10))
    if part_path.stat().st_size == expected_size:
        os.replace(part_path, dest)
        return True
    part_path.unlink(missing_ok=True)
    return False

def _parse_photo_filename(name: str) -> tuple[str, str] | None:
    """解析照片檔名，回傳 (時間字串 HH:MM:SS, 類型 normal/dark)，不符規則回傳 None。"""
    match = _PHOTO_NAME_RE.match(name)
    if not match:
        return None
    hh, mm, ss = int(match.group(1)), int(match.group(2)), int(match.group(3))
    if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
        return None
    kind = "dark" if match.group(4) else "normal"
    return f"{hh:02d}:{mm:02d}:{ss:02d}", kind


def _list_day_photos(day_dir: str) -> list[tuple[Path, str, str, str]]:
    """列出日期資料夾內所有符合命名規則的照片並依檔名排序。"""
    folder = Path(day_dir)
    if not folder.is_dir():
        return []
    items: list[tuple[Path, str, str, str]] = []
    for entry in sorted(folder.iterdir(), key=lambda p: p.name):
        if not entry.is_file():
            continue
        parsed = _parse_photo_filename(entry.name)
        if parsed is None:
            continue
        time_str, kind = parsed
        items.append((entry, entry.name, time_str, kind))
    return items


def _find_sun_in_gray(gray: np.ndarray) -> tuple[bool, float, float]:
    """從灰階陣列找出面積最大的亮區（亮度 >= 200），面積 >= 100 才回傳太陽質心。"""
    mask = gray >= 200
    # 總亮像素不到 100 個時，不可能出現面積 >= 100 的相連區域，直接結束
    if int(mask.sum()) < 100:
        return False, 0.0, 0.0

    # 左右補一欄 False，用向量化方式一次找出每一列所有連續亮線段的起點與終點
    padded = np.pad(mask, ((0, 0), (1, 1)), constant_values=False)
    sy, sx = np.nonzero((~padded[:, :-1]) & padded[:, 1:])
    _, ex = np.nonzero(padded[:, :-1] & (~padded[:, 1:]))

    n_runs = len(sy)
    if n_runs == 0:
        return False, 0.0, 0.0

    sy_list = sy.tolist()
    sx_list = sx.tolist()
    ex_list = ex.tolist()

    # 使用並查集（Union-Find）把上下相鄰且水平重疊的線段合併成相連亮區
    parent = list(range(n_runs))
    area = [ex_list[i] - sx_list[i] for i in range(n_runs)]

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if area[ra] < area[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        area[ra] += area[rb]

    # 依列掃描，用雙指標比對相鄰兩列的線段是否 4 連通重疊
    prev_start = 0
    prev_end = 0
    prev_y = -2
    idx = 0
    while idx < n_runs:
        curr_y = sy_list[idx]
        curr_start = idx
        while idx < n_runs and sy_list[idx] == curr_y:
            idx += 1
        curr_end = idx

        if curr_y == prev_y + 1:
            p = prev_start
            c = curr_start
            while p < prev_end and c < curr_end:
                if sx_list[p] < ex_list[c] and sx_list[c] < ex_list[p]:
                    union(p, c)
                if ex_list[p] < ex_list[c]:
                    p += 1
                elif ex_list[c] < ex_list[p]:
                    c += 1
                else:
                    p += 1
                    c += 1

        prev_start = curr_start
        prev_end = curr_end
        prev_y = curr_y

    # 找出面積最大的相連區域
    best_root = -1
    best_area = 0
    for i in range(n_runs):
        if parent[i] == i and area[i] > best_area:
            best_area = area[i]
            best_root = i

    if best_area < 100 or best_root < 0:
        return False, 0.0, 0.0

    # 只加總最大相連區域內的線段座標，避免小雜訊干擾質心
    total_x = 0.0
    total_y = 0.0
    total_area = 0
    for i in range(n_runs):
        if find(i) == best_root:
            length = ex_list[i] - sx_list[i]
            total_area += length
            total_x += length * (sx_list[i] + ex_list[i] - 1) * 0.5
            total_y += length * sy_list[i]

    return True, total_x / total_area, total_y / total_area


def pull(
    base_url: str,
    out_dir: str,
    days: list[str] | None = None,
    timeout: float = 8.0,
) -> dict:
    """透過 Wi-Fi 從天空日記相機下載缺少或大小不符的照片到指定資料夾。"""
    base = base_url.rstrip("/")
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if days is None:
        raw_days = _fetch_json(f"{base}/days", timeout=timeout)
        target_days = [str(d) for d in raw_days]
    else:
        target_days = [str(d) for d in days]

    downloaded = 0
    skipped = 0
    failed: list[str] = []

    for day in target_days:
        day_dir = out_root / day
        day_dir.mkdir(parents=True, exist_ok=True)
        day_q = urllib.parse.quote(day, safe="")
        # 照片多時（一天兩三百張），相機列清單可能要幾十秒，而且剛好在拍照時會更慢：放寬到 60 秒並重試 3 次
        photo_list = None
        for attempt in range(3):
            try:
                photo_list = _fetch_json(f"{base}/list?day={day_q}", timeout=max(timeout, 60))
                break
            except (OSError, ValueError) as exc:
                print(f"取得 {day} 的照片清單失敗（第 {attempt + 1} 次）：{exc}", file=sys.stderr)
                time.sleep(5)
        if photo_list is None:
            failed.append(f"{day}/（清單）")
            continue

        for item in photo_list:
            name = str(item["name"])
            expected_size = int(item["size"])
            dest = day_dir / name

            # 本機已有同名照片且檔案大小相同就跳過
            if dest.is_file() and dest.stat().st_size == expected_size:
                skipped += 1
                continue

            name_q = urllib.parse.quote(name, safe="")
            photo_url = f"{base}/photo?day={day_q}&name={name_q}"
            if _download_one(photo_url, dest, expected_size, timeout):
                downloaded += 1
            else:
                failed.append(f"{day}/{name}")

    return {
        "days": target_days,
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }


def find_sun(image_path: str) -> tuple[bool, float, float]:
    """在暗曝光照片中尋找太陽，回傳 (是否找到, 質心x, 質心y)。"""
    with Image.open(image_path) as img:
        img.load()
        gray = np.asarray(img.convert("L"))
    return _find_sun_in_gray(gray)


def analyze_day(day_dir: str) -> list[dict]:
    """分析單日資料夾內的所有正常與暗曝光照片，回傳每張照片的分析結果清單。"""
    photos = _list_day_photos(day_dir)
    rows: list[dict] = []

    for path, name, time_str, kind in photos:
        try:
            with Image.open(path) as img:
                img.load()
                rgb = np.asarray(img.convert("RGB"))
                gray = np.asarray(img.convert("L"))
        except (OSError, ValueError) as exc:
            print(f"略過打不開的照片：{path}（{exc}）")
            continue

        top_rows = max(1, int(rgb.shape[0] * 0.4))
        sky_mean = rgb[:top_rows, :, :].mean(axis=(0, 1))
        brightness = float(gray.mean())

        if kind == "dark":
            found, cx, cy = _find_sun_in_gray(gray)
            sun_found = 1 if found else 0
            sun_x = round(float(cx), 1) if found else 0.0
            sun_y = round(float(cy), 1) if found else 0.0
        else:
            sun_found = 0
            sun_x = 0.0
            sun_y = 0.0

        rows.append(
            {
                "name": name,
                "time": time_str,
                "kind": kind,
                "sun_found": sun_found,
                "sun_x": sun_x,
                "sun_y": sun_y,
                "sky_r": round(float(sky_mean[0]), 1),
                "sky_g": round(float(sky_mean[1]), 1),
                "sky_b": round(float(sky_mean[2]), 1),
                "brightness": round(brightness, 1),
            }
        )

    return rows


def write_csv(rows: list[dict], csv_path: str) -> None:
    """將分析結果寫成含 UTF-8 BOM 的 CSV 檔案，方便用 Excel 直接開啟。"""
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_sun_trail(day_dir: str, out_png: str) -> int:
    """把整天的暗曝光照片疊成太陽軌跡圖並與日落前背景合成，回傳用到的暗曝光張數。"""
    photos = _list_day_photos(day_dir)
    dark_count = 0
    dark_max: np.ndarray | None = None
    last_sun_idx: int | None = None

    for idx, (path, _, _, kind) in enumerate(photos):
        if kind != "dark":
            continue
        try:
            with Image.open(path) as img:
                img.load()
                rgb = np.asarray(img.convert("RGB"))
                gray = np.asarray(img.convert("L"))
        except (OSError, ValueError) as exc:
            print(f"略過打不開的暗曝光照片：{path}（{exc}）")
            continue

        dark_count += 1
        if dark_max is None:
            dark_max = rgb.copy()
        else:
            np.maximum(dark_max, rgb, out=dark_max)

        found, _, _ = _find_sun_in_gray(gray)
        if found:
            last_sun_idx = idx

    if dark_count == 0 or dark_max is None:
        return 0

    # 往回找最後一次拍到太陽的時間之前，最近的一張可用一般照片當作背景
    result = dark_max
    if last_sun_idx is not None:
        for idx in range(last_sun_idx - 1, -1, -1):
            path, _, _, kind = photos[idx]
            if kind != "normal":
                continue
            try:
                with Image.open(path) as img:
                    img.load()
                    bg_rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
                dimmed_bg = np.clip(bg_rgb * 0.45, 0, 255).astype(np.uint8)
                result = np.maximum(dark_max, dimmed_bg)
                break
            except (OSError, ValueError) as exc:
                print(f"略過打不開的背景照片：{path}（{exc}）")

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, mode="RGB").save(out_path, format="PNG")
    return dark_count


def make_sky_strip(
    day_dir: str,
    out_png: str,
    col_width: int = 4,
    height: int = 600,
) -> int:
    """把每張一般照片的每列平均顏色縮成直條並依時間拼成天空色條圖，回傳張數。"""
    photos = _list_day_photos(day_dir)
    strips: list[Image.Image] = []

    for path, _, _, kind in photos:
        if kind != "normal":
            continue
        try:
            with Image.open(path) as img:
                img.load()
                rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
        except (OSError, ValueError) as exc:
            print(f"略過打不開的一般照片：{path}（{exc}）")
            continue

        row_means = rgb.mean(axis=1)
        col_arr = np.clip(np.rint(row_means), 0, 255).astype(np.uint8)
        col_img = Image.fromarray(col_arr.reshape(rgb.shape[0], 1, 3), mode="RGB")
        strips.append(col_img.resize((col_width, height), resample=_BILINEAR))

    if not strips:
        return 0

    canvas = Image.new("RGB", (len(strips) * col_width, height))
    for i, strip in enumerate(strips):
        canvas.paste(strip, (i * col_width, 0))

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")
    return len(strips)


def make_timelapse(day_dir: str, out_mp4: str, fps: int = 12) -> bool:
    """使用系統的 ffmpeg 把一天的一般照片依序串成 1280 寬的 MP4 縮時影片。"""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        print("找不到系統的 ffmpeg，無法製作縮時影片，請先安裝 ffmpeg。")
        return False

    photos = _list_day_photos(day_dir)
    valid_paths: list[Path] = []
    for path, _, _, kind in photos:
        if kind != "normal":
            continue
        try:
            with Image.open(path) as img:
                img.load()
            valid_paths.append(path.resolve())
        except (OSError, ValueError) as exc:
            print(f"略過打不開的一般照片：{path}（{exc}）")

    if not valid_paths:
        print(f"資料夾沒有可用的一般照片，略過縮時影片：{day_dir}")
        return False

    out_path = Path(out_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_duration = 1.0 / max(1, fps)
    concat_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            encoding="utf-8",
            delete=False,
        ) as tmp:
            concat_path = tmp.name
            for p in valid_paths:
                escaped = str(p).replace("'", "'\\''")
                tmp.write(f"file '{escaped}'\n")
                tmp.write(f"duration {frame_duration:.6f}\n")
            # 依 ffmpeg concat 慣例在最後補上一行最後張照片，確保最後一格顯示完整時間
            last_escaped = str(valid_paths[-1]).replace("'", "'\\''")
            tmp.write(f"file '{last_escaped}'\n")

        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_path,
            "-vf",
            "scale=1280:-2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            err_tail = (proc.stderr or "").strip().splitlines()[-3:]
            print(f"ffmpeg 產生縮時影片失敗：{out_path}（{' / '.join(err_tail)}）")
            return False
        return True
    finally:
        if concat_path and os.path.exists(concat_path):
            os.unlink(concat_path)


def _discover_days(photos_dir: Path) -> list[str]:
    """找出照片根目錄底下所有名稱為 YYYYMMDD 的日期資料夾並排序。"""
    if not photos_dir.is_dir():
        return []
    return sorted(
        entry.name
        for entry in photos_dir.iterdir()
        if entry.is_dir() and _DAY_DIR_RE.match(entry.name)
    )


def _run_cli(argv: list[str] | None = None) -> int:
    """解析命令列參數並執行 pull 或 analyze 子命令。"""
    parser = argparse.ArgumentParser(
        description="天空日記相機電腦端工具：下載照片、分析太陽與天空、產出圖表與縮時影片"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pull_parser = subparsers.add_parser("pull", help="從相機下載照片到電腦")
    pull_parser.add_argument("base_url", help="相機網址，例如 http://192.168.1.50")
    pull_parser.add_argument("--out", default="photos", help="照片儲存資料夾（預設 photos）")
    pull_parser.add_argument(
        "--day",
        action="extend",
        nargs="+",
        default=None,
        help="只下載指定日期 YYYYMMDD，可填多個",
    )

    analyze_parser = subparsers.add_parser("analyze", help="分析照片並產生圖表與縮時影片")
    analyze_parser.add_argument(
        "--photos", default="photos", help="照片來源資料夾（預設 photos）"
    )
    analyze_parser.add_argument(
        "--out", default="analysis", help="分析結果輸出資料夾（預設 analysis）"
    )
    analyze_parser.add_argument(
        "--day",
        action="extend",
        nargs="+",
        default=None,
        help="只分析指定日期 YYYYMMDD，可填多個",
    )

    args = parser.parse_args(argv)

    if args.command == "pull":
        summary = pull(args.base_url, args.out, days=args.day)
        print(
            f"下載完成：共 {len(summary['days'])} 天，"
            f"新下載 {summary['downloaded']} 張、"
            f"略過 {summary['skipped']} 張、"
            f"失敗 {len(summary['failed'])} 張"
        )
        return 0 if not summary["failed"] else 1

    photos_root = Path(args.photos)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    target_days = args.day if args.day is not None else _discover_days(photos_root)

    for day in target_days:
        day_dir = photos_root / day
        if not day_dir.is_dir():
            print(f"{day}：找不到資料夾 {day_dir}，已跳過")
            continue

        rows = analyze_day(str(day_dir))
        write_csv(rows, str(out_root / f"{day}.csv"))
        make_sun_trail(str(day_dir), str(out_root / f"{day}_sun_trail.png"))
        make_sky_strip(str(day_dir), str(out_root / f"{day}_sky_strip.png"))
        make_timelapse(str(day_dir), str(out_root / f"{day}.mp4"))

        normal_count = sum(1 for r in rows if r["kind"] == "normal")
        dark_count = sum(1 for r in rows if r["kind"] == "dark")
        sun_times = [
            r["time"] for r in rows if r["kind"] == "dark" and r["sun_found"] == 1
        ]
        if sun_times:
            sun_note = f"最後看到太陽 {sun_times[-1]}"
        else:
            sun_note = "這天沒有拍到太陽"
        print(f"{day}：一般 {normal_count} 張、暗曝光 {dark_count} 張，{sun_note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_run_cli())
