"""天空日記觀察站：太陽、月亮、影長、天空與植物影像及曲線擬合之科學對照模型。

公式出處與參考文獻：
1. 太陽位置與日出日落：NOAA Solar Calculator 與 Jean Meeus《Astronomical Algorithms》
   （第 2 版）第 25 章（太陽視黃經、赤經、赤緯與均時差）。
2. 格林威治恆星時與章動修正：Meeus《Astronomical Algorithms》第 12、22 章。
3. 標準大氣折射修正：Sæmundsson (1986) 公式（Meeus 第 16 章，標準狀態 10°C、1010 hPa）。
4. 月球位置與地平視差：Meeus《Astronomical Algorithms》第 47 章（ELP-2000/82 週期項簡化版）
   與第 40 章（WGS84 地球扁率站心視差修正）。
5. 月相、相角與亮面比例：Meeus《Astronomical Algorithms》第 48 章。
6. 月出月落時刻：以站心上緣接觸地平線條件（真高度 h = −0.5667° − 0.2725 × 地平視差）
   進行逐 10 分鐘掃描與二分法逼近。
7. 立竿影長換算：平面直角三角幾何關係。
8. 植物綠度指標（GCC）：PhenoCam 植物物候監測標準指標。
9. 雲量估算：天空影像紅藍比（Red-to-Blue Ratio）門檻分類法。
10. 直線與 S 形（Logistic）曲線擬合：普通最小平方法（OLS）與變數投影網格搜尋搭配
    Levenberg-Marquardt 非線性最小平方法。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import math

import numpy as np

# 臺灣 22 縣市政府所在地附近之（緯度, 經度），單位為度（北緯、東經為正）
COUNTY_COORDS: dict[str, tuple[float, float]] = {
    "臺北市": (25.0375, 121.5637),
    "新北市": (25.0120, 121.4650),
    "桃園市": (24.9937, 121.3010),
    "臺中市": (24.1618, 120.6469),
    "臺南市": (22.9927, 120.1851),
    "高雄市": (22.6203, 120.3120),
    "基隆市": (25.1310, 121.7442),
    "新竹市": (24.8066, 120.9686),
    "嘉義市": (23.4813, 120.4537),
    "新竹縣": (24.8270, 121.0128),
    "苗栗縣": (24.5650, 120.8214),
    "彰化縣": (24.0758, 120.5443),
    "南投縣": (23.9020, 120.6904),
    "雲林縣": (23.6990, 120.5264),
    "嘉義縣": (23.4588, 120.2932),
    "屏東縣": (22.6727, 120.4880),
    "宜蘭縣": (24.7303, 121.7631),
    "花蓮縣": (23.9871, 121.6012),
    "臺東縣": (22.7550, 121.1500),
    "澎湖縣": (23.5711, 119.5793),
    "金門縣": (24.4370, 118.3186),
    "連江縣": (26.1505, 119.9499),
}

_J2000_UTC = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_AU_KM = 149597870.7
_EARTH_RADIUS_KM = 6378.137
_EARTH_FLATTENING = 1.0 / 298.257223563

# Meeus 第 47 章表 47.A：月球黃經（10^-6 度）與地心距（0.001 公里）主要週期項
# 每筆格式：(D 倍數, M 倍數, M' 倍數, F 倍數, 黃經係數, 距離係數)
_MOON_LR_TERMS: tuple[tuple[int, int, int, int, int, int], ...] = (
    (0, 0, 1, 0, 6288774, -20905355),
    (2, 0, -1, 0, 1274027, -3699111),
    (2, 0, 0, 0, 658314, -2955968),
    (0, 0, 2, 0, 213618, -569925),
    (0, 1, 0, 0, -185116, 48888),
    (0, 0, 0, 2, -114332, -3149),
    (2, 0, -2, 0, 58793, 246158),
    (2, -1, -1, 0, 57066, -152138),
    (2, 0, 1, 0, 53322, -170733),
    (2, -1, 0, 0, 45758, -204586),
    (0, 1, -1, 0, -40923, -129620),
    (1, 0, 0, 0, -34720, 108743),
    (0, 1, 1, 0, -30383, 104755),
    (2, 0, 0, -2, 15327, 10321),
    (0, 0, 1, 2, -12528, 0),
    (0, 0, 1, -2, 10980, 79661),
    (4, 0, -1, 0, 10675, -34782),
    (0, 0, 3, 0, 10034, -23210),
    (4, 0, -2, 0, 8548, -21636),
    (2, 1, -1, 0, -7888, 24208),
    (2, 1, 0, 0, -6766, 30824),
    (1, 0, -1, 0, -5163, -8379),
    (1, 1, 0, 0, 4987, -16675),
    (2, -1, 1, 0, 4036, -12831),
    (2, 0, 2, 0, 3994, -10445),
    (4, 0, 0, 0, 3861, -11650),
    (2, 0, -3, 0, 3665, 14403),
    (0, 1, -2, 0, -2689, -7003),
    (2, 0, -1, 2, -2602, 0),
    (2, -1, -2, 0, 2390, 10056),
    (1, 0, 1, 0, -2348, 6322),
    (2, -2, 0, 0, 2236, -9884),
    (0, 1, 2, 0, -2120, 5751),
    (0, 2, 0, 0, -2069, 0),
    (2, -2, -1, 0, 2048, -4950),
    (2, 0, 1, -2, -1773, 4130),
    (2, 0, 0, 2, -1595, 0),
    (4, -1, -1, 0, 1215, -3958),
    (0, 0, 2, 2, -1110, 0),
    (3, 0, -1, 0, -892, 3258),
)

# Meeus 第 47 章表 47.B：月球黃緯（10^-6 度）主要週期項
# 每筆格式：(D 倍數, M 倍數, M' 倍數, F 倍數, 黃緯係數)
_MOON_B_TERMS: tuple[tuple[int, int, int, int, int], ...] = (
    (0, 0, 0, 1, 5128122),
    (0, 0, 1, 1, 280602),
    (0, 0, 1, -1, 277693),
    (2, 0, 0, -1, 173237),
    (2, 0, -1, 1, 55413),
    (2, 0, -1, -1, 46271),
    (2, 0, 0, 1, 32573),
    (0, 0, 2, 1, 17198),
    (2, 0, 1, -1, 9266),
    (0, 0, 2, -1, 8822),
    (2, -1, 0, -1, 8216),
    (2, 0, -2, -1, 4324),
    (2, 0, 1, 1, 4200),
    (2, 1, 0, -1, -3359),
    (2, -1, -1, 1, 2463),
    (2, -1, 0, 1, 2211),
    (2, -1, -1, -1, 2065),
    (0, 1, -1, -1, -1870),
    (4, 0, -1, -1, 1828),
    (0, 1, 0, 1, -1794),
    (0, 0, 0, 3, -1749),
    (0, 1, -1, 1, -1565),
    (1, 0, 0, 1, -1491),
    (0, 1, 1, 1, -1475),
    (0, 1, 1, -1, -1410),
    (0, 1, 0, -1, -1344),
    (1, 0, 0, -1, -1335),
    (0, 0, 3, 1, 1107),
    (4, 0, 0, -1, 1021),
    (4, 0, -1, 1, 833),
)

_MOON_PHASE_NAMES: tuple[str, ...] = (
    "新月",
    "眉月（蛾眉月）",
    "上弦月",
    "盈凸月",
    "滿月",
    "虧凸月",
    "下弦月",
    "殘月",
)


def _to_utc(when: datetime) -> datetime:
    """把有時區資訊的日期時間物件統一轉成 UTC。"""
    if when.tzinfo is None:
        raise ValueError("時刻參數必須包含時區資訊（timezone-aware datetime）。")
    return when.astimezone(timezone.utc)


def _delta_t_seconds(year: float) -> float:
    """估算地球時（TT）與世界時（UT）的時間差 ΔT（秒）。"""
    if 2017.0 <= year <= 2035.0:
        return 69.2
    t = year - 2000.0
    return 62.92 + 0.32217 * t + 0.005589 * t * t


def _datetime_to_jd(when: datetime) -> tuple[float, float]:
    """把時刻轉成儒略日（世界時 JD_UT 與地球時 JD_TT）。"""
    utc = _to_utc(when)
    jd_ut = 2451545.0 + (utc - _J2000_UTC).total_seconds() / 86400.0
    decimal_year = utc.year + (utc.timetuple().tm_yday - 1) / 365.25
    jd_tt = jd_ut + _delta_t_seconds(decimal_year) / 86400.0
    return jd_ut, jd_tt


def _jd_ut_to_tt(jd_ut: float) -> float:
    """由世界時儒略日估算對應的地球時儒略日。"""
    year = 2000.0 + (jd_ut - 2451545.0) / 365.25
    return jd_ut + _delta_t_seconds(year) / 86400.0


def _nutation_and_obliquity(t_tt: float) -> tuple[float, float, float]:
    """計算黃經章動、交角章動與真黃赤交角（單位皆為度）。"""
    omega = math.radians(125.04452 - 1934.136261 * t_tt)
    l_sun = math.radians(280.4665 + 36000.7698 * t_tt)
    l_moon = math.radians(218.3165 + 481267.8813 * t_tt)

    dpsi_arcsec = (
        -17.20 * math.sin(omega)
        - 1.32 * math.sin(2.0 * l_sun)
        - 0.23 * math.sin(2.0 * l_moon)
        + 0.21 * math.sin(2.0 * omega)
    )
    deps_arcsec = (
        9.20 * math.cos(omega)
        + 0.57 * math.cos(2.0 * l_sun)
        + 0.10 * math.cos(2.0 * l_moon)
        - 0.09 * math.cos(2.0 * omega)
    )
    eps0_deg = (
        23.0
        + (
            26.0
            + (
                21.448
                - t_tt * (46.8150 + t_tt * (0.00059 - t_tt * 0.001813))
            )
            / 60.0
        )
        / 60.0
    )
    dpsi_deg = dpsi_arcsec / 3600.0
    deps_deg = deps_arcsec / 3600.0
    return dpsi_deg, deps_deg, eps0_deg + deps_deg


def _gast_deg(jd_ut: float, t_tt: float) -> float:
    """計算格林威治視恆星時 GAST（度，0～360）。"""
    t_ut = (jd_ut - 2451545.0) / 36525.0
    gmst = (
        280.46061837
        + 360.98564736629 * (jd_ut - 2451545.0)
        + 0.000387933 * t_ut * t_ut
        - (t_ut * t_ut * t_ut) / 38710000.0
    )
    dpsi_deg, _, eps_deg = _nutation_and_obliquity(t_tt)
    return (gmst + dpsi_deg * math.cos(math.radians(eps_deg))) % 360.0


def _apply_refraction(h_true_deg: float) -> float:
    """以 Sæmundsson 公式（10°C、1010 hPa）為真高度角加上大氣折射修正。"""
    if h_true_deg <= -2.5:
        return h_true_deg
    arg_deg = h_true_deg + 10.3 / (h_true_deg + 5.11)
    r_arcmin = 1.02 / math.tan(math.radians(arg_deg))
    r_deg = max(0.0, r_arcmin / 60.0)
    h_app = h_true_deg + r_deg
    if h_app > -1.0:
        return h_app
    return h_true_deg


def _equatorial_to_horizontal(
    ha_deg: float, dec_deg: float, lat_deg: float
) -> tuple[float, float]:
    """將時角與赤緯轉換為地平座標的真高度角與方位角（正北 0° 順時針）。"""
    ha = math.radians(ha_deg)
    dec = math.radians(dec_deg)
    lat = math.radians(lat_deg)

    sin_alt = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha)
    sin_alt = max(-1.0, min(1.0, sin_alt))
    alt_deg = math.degrees(math.asin(sin_alt))

    az_y = -math.cos(dec) * math.sin(ha)
    az_x = math.cos(lat) * math.sin(dec) - math.sin(lat) * math.cos(dec) * math.cos(ha)
    az_deg = math.degrees(math.atan2(az_y, az_x)) % 360.0
    return alt_deg, az_deg


def _sun_apparent(t_tt: float) -> tuple[float, float, float, float]:
    """計算太陽視黃經（度）、地日距離（AU）、視赤經（度）與視赤緯（度）。"""
    l0 = (280.46646 + t_tt * (36000.76983 + 0.0003032 * t_tt)) % 360.0
    m = 357.52911 + t_tt * (35999.05029 - 0.0001537 * t_tt)
    m_rad = math.radians(m)
    e = 0.016708634 - t_tt * (0.000042037 + 0.0000001267 * t_tt)

    c = (
        math.sin(m_rad) * (1.914602 - t_tt * (0.004817 + 0.000014 * t_tt))
        + math.sin(2.0 * m_rad) * (0.019993 - 0.000101 * t_tt)
        + math.sin(3.0 * m_rad) * 0.000289
    )
    true_lon = l0 + c
    true_anom_rad = math.radians(m + c)
    r_au = (1.000001018 * (1.0 - e * e)) / (1.0 + e * math.cos(true_anom_rad))

    dpsi_deg, _, eps_deg = _nutation_and_obliquity(t_tt)
    app_lon = (true_lon + dpsi_deg - 0.0056916 / r_au) % 360.0

    lam_r = math.radians(app_lon)
    eps_r = math.radians(eps_deg)
    ra_deg = math.degrees(math.atan2(math.cos(eps_r) * math.sin(lam_r), math.cos(lam_r))) % 360.0
    dec_deg = math.degrees(math.asin(max(-1.0, min(1.0, math.sin(eps_r) * math.sin(lam_r)))))
    return app_lon, r_au, ra_deg, dec_deg


def _sun_true_alt_az(jd_ut: float, jd_tt: float, lat: float, lon: float) -> tuple[float, float, float]:
    """計算太陽不含折射的真高度角、方位角與地方時角（皆為度）。"""
    t_tt = (jd_tt - 2451545.0) / 36525.0
    _, _, ra_deg, dec_deg = _sun_apparent(t_tt)
    gast = _gast_deg(jd_ut, t_tt)
    ha_deg = ((gast + lon - ra_deg + 180.0) % 360.0) - 180.0
    alt_true, az_deg = _equatorial_to_horizontal(ha_deg, dec_deg, lat)
    return alt_true, az_deg, ha_deg


def _moon_geocentric(t_tt: float) -> tuple[float, float, float, float, float]:
    """以 Meeus 第 47 章公式計算月球視黃經、黃緯、地心距（公里）、視赤經與視赤緯。"""
    t2 = t_tt * t_tt
    t3 = t2 * t_tt
    t4 = t2 * t2

    lp = (218.3164477 + 481267.88123421 * t_tt - 0.0015786 * t2 + t3 / 538841.0 - t4 / 65194000.0) % 360.0
    d = (297.8501921 + 445267.1114034 * t_tt - 0.0018819 * t2 + t3 / 545868.0 - t4 / 113065000.0) % 360.0
    m = (357.5291092 + 35999.0502909 * t_tt - 0.0001536 * t2 + t3 / 24490000.0) % 360.0
    mp = (134.9633964 + 477198.8675055 * t_tt + 0.0087414 * t2 + t3 / 69699.0 - t4 / 14712000.0) % 360.0
    f = (93.2720950 + 483202.0175233 * t_tt - 0.0036539 * t2 - t3 / 3526000.0 + t4 / 863310000.0) % 360.0

    a1 = math.radians(119.75 + 131.849 * t_tt)
    a2 = math.radians(53.09 + 479264.290 * t_tt)
    a3 = math.radians(313.45 + 481266.484 * t_tt)
    e_factor = 1.0 - 0.002516 * t_tt - 0.0000074 * t2
    e2_factor = e_factor * e_factor

    d_r = math.radians(d)
    m_r = math.radians(m)
    mp_r = math.radians(mp)
    f_r = math.radians(f)
    lp_r = math.radians(lp)

    sum_l = 0.0
    sum_r = 0.0
    for cd, cm, cmp_, cf, cl, cr in _MOON_LR_TERMS:
        arg = cd * d_r + cm * m_r + cmp_ * mp_r + cf * f_r
        scale = e_factor if abs(cm) == 1 else (e2_factor if abs(cm) == 2 else 1.0)
        if cl:
            sum_l += cl * scale * math.sin(arg)
        if cr:
            sum_r += cr * scale * math.cos(arg)

    sum_l += (
        3958.0 * math.sin(a1)
        + 1962.0 * math.sin(lp_r - f_r)
        + 318.0 * math.sin(a2)
    )

    sum_b = 0.0
    for cd, cm, cmp_, cf, cb in _MOON_B_TERMS:
        arg = cd * d_r + cm * m_r + cmp_ * mp_r + cf * f_r
        scale = e_factor if abs(cm) == 1 else (e2_factor if abs(cm) == 2 else 1.0)
        sum_b += cb * scale * math.sin(arg)

    sum_b += (
        -2235.0 * math.sin(lp_r)
        + 382.0 * math.sin(a3)
        + 175.0 * math.sin(a1 - f_r)
        + 175.0 * math.sin(a1 + f_r)
        + 127.0 * math.sin(lp_r - mp_r)
        - 115.0 * math.sin(lp_r + mp_r)
    )

    dpsi_deg, _, eps_deg = _nutation_and_obliquity(t_tt)
    app_lon = (lp + sum_l * 1e-6 + dpsi_deg) % 360.0
    lat_ecl = sum_b * 1e-6
    dist_km = 385000.56 + sum_r * 1e-3

    lam_r = math.radians(app_lon)
    bet_r = math.radians(lat_ecl)
    eps_r = math.radians(eps_deg)

    ra_deg = math.degrees(
        math.atan2(
            math.sin(lam_r) * math.cos(bet_r) * math.cos(eps_r) - math.sin(bet_r) * math.sin(eps_r),
            math.cos(lam_r) * math.cos(bet_r),
        )
    ) % 360.0
    dec_deg = math.degrees(
        math.asin(
            max(
                -1.0,
                min(
                    1.0,
                    math.sin(bet_r) * math.cos(eps_r)
                    + math.cos(bet_r) * math.sin(eps_r) * math.sin(lam_r),
                ),
            )
        )
    )
    return app_lon, lat_ecl, dist_km, ra_deg, dec_deg


def _moon_topocentric_true(
    jd_ut: float, jd_tt: float, lat: float, lon: float
) -> tuple[float, float, float]:
    """計算月球站心真高度角（不含折射）、方位角與赤道地平視差（皆為度）。"""
    t_tt = (jd_tt - 2451545.0) / 36525.0
    _, _, dist_km, ra_deg, dec_deg = _moon_geocentric(t_tt)
    gast = _gast_deg(jd_ut, t_tt)

    ha_rad = math.radians(gast + lon - ra_deg)
    dec_rad = math.radians(dec_deg)
    lat_rad = math.radians(lat)

    sin_pi = _EARTH_RADIUS_KM / dist_km
    hp_deg = math.degrees(math.asin(min(1.0, sin_pi)))

    # WGS84 地球扁率修正後的觀測者地心座標（單位：地球赤道半徑）
    b_over_a = 1.0 - _EARTH_FLATTENING
    u = math.atan(b_over_a * math.tan(lat_rad))
    rho_sin_phi = b_over_a * math.sin(u)
    rho_cos_phi = math.cos(u)

    # 在地方子午圈赤道直角座標系中扣掉觀測者位移，求站心時角與赤緯
    x_topo = math.cos(dec_rad) * math.cos(ha_rad) - rho_cos_phi * sin_pi
    y_topo = math.cos(dec_rad) * math.sin(ha_rad)
    z_topo = math.sin(dec_rad) - rho_sin_phi * sin_pi

    ha_topo_deg = math.degrees(math.atan2(y_topo, x_topo))
    dec_topo_deg = math.degrees(math.atan2(z_topo, math.hypot(x_topo, y_topo)))
    alt_true, az_deg = _equatorial_to_horizontal(ha_topo_deg, dec_topo_deg, lat)
    return alt_true, az_deg, hp_deg


def _moon_sun_elongation_from_jd(jd_ut: float) -> float:
    """計算指定世界時儒略日的月球相對於太陽的黃經差（0～360 度）。"""
    jd_tt = _jd_ut_to_tt(jd_ut)
    t_tt = (jd_tt - 2451545.0) / 36525.0
    sun_lon, _, _, _ = _sun_apparent(t_tt)
    moon_lon, _, _, _, _ = _moon_geocentric(t_tt)
    return (moon_lon - sun_lon) % 360.0


def _format_minutes(minutes: float) -> str:
    """把當日 00:00 起的分鐘數四捨五入為 HH:MM 字串。"""
    total = int(math.floor(minutes + 0.5))
    total = max(0, min(1439, total))
    return f"{total // 60:02d}:{total % 60:02d}"


def sun_position(when: datetime, lat: float, lon: float) -> tuple[float, float]:
    """計算指定時刻與地點看到的太陽高度角（含大氣折射）與方位角。"""
    jd_ut, jd_tt = _datetime_to_jd(when)
    alt_true, az_deg, _ = _sun_true_alt_az(jd_ut, jd_tt, lat, lon)
    return _apply_refraction(alt_true), az_deg


def sun_events(day: date, lat: float, lon: float, tz_hours: float = 8) -> dict:
    """計算指定日期與地點的日出、日落、太陽正午時刻，以及方位角與正午高度角。"""
    tz = timezone(timedelta(hours=tz_hours))
    local_midnight = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=tz)
    jd_ut0, jd_tt0 = _datetime_to_jd(local_midnight)

    # 先迭代求出太陽中天（地方時角 H = 0°）的準確分鐘數
    mn_noon = 720.0 - 4.0 * lon + tz_hours * 60.0
    for _ in range(5):
        frac_day = mn_noon / 1440.0
        _, _, ha_deg = _sun_true_alt_az(jd_ut0 + frac_day, jd_tt0 + frac_day, lat, lon)
        mn_noon -= ha_deg / 0.25

    noon_dt = local_midnight + timedelta(minutes=mn_noon)
    noon_alt, _ = sun_position(noon_dt, lat, lon)

    # 日出日落定義：太陽上緣接觸地平線（天頂距 90.833°，對應真高度 −0.833°）
    h0 = -0.833
    t_tt_noon = (jd_tt0 + mn_noon / 1440.0 - 2451545.0) / 36525.0
    _, _, _, dec_noon = _sun_apparent(t_tt_noon)

    lat_r = math.radians(lat)
    dec_r = math.radians(dec_noon)
    cos_ha = (math.sin(math.radians(h0)) - math.sin(lat_r) * math.sin(dec_r)) / (
        math.cos(lat_r) * math.cos(dec_r)
    )

    def _alt_diff(mn: float) -> float:
        frac = mn / 1440.0
        alt_true, _, _ = _sun_true_alt_az(jd_ut0 + frac, jd_tt0 + frac, lat, lon)
        return alt_true - h0

    sunrise_str: str | None = None
    sunset_str: str | None = None
    sunrise_az: float | None = None
    sunset_az: float | None = None

    if -1.0 <= cos_ha <= 1.0:
        ha_est = math.degrees(math.acos(cos_ha))
        rise_est = mn_noon - 4.0 * ha_est
        set_est = mn_noon + 4.0 * ha_est

        # 二分法精修日出時刻
        lo, hi = rise_est - 20.0, rise_est + 20.0
        if _alt_diff(lo) * _alt_diff(hi) > 0.0:
            lo, hi = mn_noon - 720.0, mn_noon
        if _alt_diff(lo) <= 0.0 <= _alt_diff(hi):
            for _ in range(22):
                mid = 0.5 * (lo + hi)
                if _alt_diff(mid) < 0.0:
                    lo = mid
                else:
                    hi = mid
            mn_rise = 0.5 * (lo + hi)
            _, az_r, _ = _sun_true_alt_az(jd_ut0 + mn_rise / 1440.0, jd_tt0 + mn_rise / 1440.0, lat, lon)
            sunrise_str = _format_minutes(mn_rise)
            sunrise_az = round(az_r, 1)

        # 二分法精修日落時刻
        lo, hi = set_est - 20.0, set_est + 20.0
        if _alt_diff(lo) * _alt_diff(hi) > 0.0:
            lo, hi = mn_noon, mn_noon + 720.0
        if _alt_diff(lo) >= 0.0 >= _alt_diff(hi):
            for _ in range(22):
                mid = 0.5 * (lo + hi)
                if _alt_diff(mid) > 0.0:
                    lo = mid
                else:
                    hi = mid
            mn_set = 0.5 * (lo + hi)
            _, az_s, _ = _sun_true_alt_az(jd_ut0 + mn_set / 1440.0, jd_tt0 + mn_set / 1440.0, lat, lon)
            sunset_str = _format_minutes(mn_set)
            sunset_az = round(az_s, 1)

    return {
        "sunrise": sunrise_str,
        "sunset": sunset_str,
        "solar_noon": _format_minutes(mn_noon),
        "sunrise_az": sunrise_az,
        "sunset_az": sunset_az,
        "noon_altitude": round(noon_alt, 1),
    }


def moon_position(when: datetime, lat: float, lon: float) -> tuple[float, float]:
    """計算指定時刻與地點看到的月亮站心高度角（含視差與折射）與方位角。"""
    jd_ut, jd_tt = _datetime_to_jd(when)
    alt_true, az_deg, _ = _moon_topocentric_true(jd_ut, jd_tt, lat, lon)
    return _apply_refraction(alt_true), az_deg


def moon_illumination(when: datetime) -> dict:
    """計算指定時刻的月亮亮面比例、相角、距角、月齡、盈虧狀態與中文月相名稱。"""
    jd_ut, jd_tt = _datetime_to_jd(when)
    t_tt = (jd_tt - 2451545.0) / 36525.0

    sun_lon, sun_dist_au, sun_ra, sun_dec = _sun_apparent(t_tt)
    moon_lon, _, moon_dist_km, moon_ra, moon_dec = _moon_geocentric(t_tt)

    # 黃經差（0～360°：0 新月、90 上弦、180 滿月、270 下弦）
    elongation = (moon_lon - sun_lon) % 360.0

    # Meeus 第 48 章：以地心赤道座標計算日月張角 ψ 與月心相角 i
    sun_dec_r = math.radians(sun_dec)
    moon_dec_r = math.radians(moon_dec)
    dra_r = math.radians(sun_ra - moon_ra)
    cos_psi = (
        math.sin(sun_dec_r) * math.sin(moon_dec_r)
        + math.cos(sun_dec_r) * math.cos(moon_dec_r) * math.cos(dra_r)
    )
    cos_psi = max(-1.0, min(1.0, cos_psi))
    psi_r = math.acos(cos_psi)

    sun_dist_km = sun_dist_au * _AU_KM
    phase_angle_r = math.atan2(
        sun_dist_km * math.sin(psi_r),
        moon_dist_km - sun_dist_km * math.cos(psi_r),
    )
    phase_angle_deg = math.degrees(phase_angle_r)
    fraction = 0.5 * (1.0 + math.cos(phase_angle_r))

    # 往回以 1 天步長尋找最近一次黃經差跨過 0° 的新月時刻，再用二分法精修
    if elongation < 1e-6:
        age_days = 0.0
    else:
        t_hi = jd_ut
        e_hi = elongation
        t_lo = jd_ut - 1.0
        for _ in range(35):
            e_lo = _moon_sun_elongation_from_jd(t_lo)
            if e_lo > 270.0 and e_hi <= 90.0:
                break
            t_hi = t_lo
            e_hi = e_lo
            t_lo -= 1.0

        for _ in range(22):
            t_mid = 0.5 * (t_lo + t_hi)
            if _moon_sun_elongation_from_jd(t_mid) > 180.0:
                t_lo = t_mid
            else:
                t_hi = t_mid
        jd_new_moon = 0.5 * (t_lo + t_hi)
        age_days = max(0.0, jd_ut - jd_new_moon)

    phase_idx = int(((elongation + 22.5) % 360.0) // 45.0)
    waxing = 0.0 <= elongation < 180.0

    return {
        "fraction": round(fraction, 3),
        "phase_angle": round(phase_angle_deg, 2),
        "elongation": round(elongation, 2) % 360.0,
        "age_days": round(age_days, 1),
        "waxing": waxing,
        "name": _MOON_PHASE_NAMES[phase_idx],
    }


def moon_events(day: date, lat: float, lon: float, tz_hours: float = 8) -> dict:
    """找出指定當地日期 00:00～24:00 之間的月出、月落時刻與對應方位角。"""
    tz = timezone(timedelta(hours=tz_hours))
    local_midnight = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=tz)
    jd_ut0, jd_tt0 = _datetime_to_jd(local_midnight)

    def _horizon_margin(mn: float) -> tuple[float, float]:
        frac = mn / 1440.0
        alt_true, az_deg, hp_deg = _moon_topocentric_true(
            jd_ut0 + frac, jd_tt0 + frac, lat, lon
        )
        # 月亮上緣接觸地平線條件：站心真高度 = −0.5667° − 0.2725 × 地平視差
        h0 = -0.5667 - 0.2725 * hp_deg
        return alt_true - h0, az_deg

    moonrise_str: str | None = None
    moonset_str: str | None = None
    moonrise_az: float | None = None
    moonset_az: float | None = None

    step_min = 10.0
    prev_mn = 0.0
    prev_val, prev_az = _horizon_margin(0.0)

    for step in range(1, 145):
        curr_mn = step * step_min
        curr_val, _ = _horizon_margin(curr_mn)

        # 處理剛好在 00:00 整點跨過地平線的邊界情況
        if step == 1 and prev_val == 0.0:
            if curr_val > 0.0 and moonrise_str is None:
                moonrise_str = "00:00"
                moonrise_az = round(prev_az, 1)
            elif curr_val < 0.0 and moonset_str is None:
                moonset_str = "00:00"
                moonset_az = round(prev_az, 1)

        # 由負轉正（或到達 0）：月出
        if moonrise_str is None and prev_val < 0.0 <= curr_val:
            lo, hi = prev_mn, curr_mn
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                if _horizon_margin(mid)[0] < 0.0:
                    lo = mid
                else:
                    hi = mid
            t_rise = 0.5 * (lo + hi)
            _, az_r = _horizon_margin(t_rise)
            moonrise_str = _format_minutes(t_rise)
            moonrise_az = round(az_r, 1)

        # 由正（含 0）轉負：月落
        if moonset_str is None and prev_val >= 0.0 > curr_val:
            lo, hi = prev_mn, curr_mn
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                if _horizon_margin(mid)[0] >= 0.0:
                    lo = mid
                else:
                    hi = mid
            t_set = 0.5 * (lo + hi)
            _, az_s = _horizon_margin(t_set)
            moonset_str = _format_minutes(t_set)
            moonset_az = round(az_s, 1)

        prev_mn = curr_mn
        prev_val = curr_val

    return {
        "moonrise": moonrise_str,
        "moonset": moonset_str,
        "moonrise_az": moonrise_az,
        "moonset_az": moonset_az,
    }


def shadow_to_sun_altitude(stick_height: float, shadow_length: float) -> float:
    """根據立竿高度與影長反推當時的太陽高度角（度）。"""
    if shadow_length <= 0.0:
        raise ValueError("影長必須大於 0。")
    if stick_height < 0.0:
        raise ValueError("竿高不可小於 0。")
    return math.degrees(math.atan(stick_height / shadow_length))


def expected_shadow_length(stick_height: float, sun_altitude_deg: float) -> float | None:
    """根據立竿高度與太陽高度角推算理論影長；太陽在地平線以下時回傳 None。"""
    if sun_altitude_deg <= 0.0:
        return None
    if sun_altitude_deg >= 90.0:
        return 0.0
    return stick_height / math.tan(math.radians(sun_altitude_deg))


def green_chromatic_coordinate(
    rgb: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """計算影像指定區域的綠度指標 GCC（綠色平均值除以紅綠藍平均值總和）。"""
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("rgb 必須是形狀為 (H, W, 3) 的陣列。")

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.ndim == 3 and m.shape[2] == 1:
            m = m[:, :, 0]
        if m.shape != arr.shape[:2]:
            raise ValueError("mask 的形狀必須與 rgb 的前兩維相同。")
        r = r[m]
        g = g[m]
        b = b[m]
    else:
        r = r.ravel()
        g = g.ravel()
        b = b.ravel()

    valid = ~(np.isnan(r) | np.isnan(g) | np.isnan(b))
    r = r[valid]
    g = g[valid]
    b = b[valid]

    if r.size == 0:
        return float("nan")

    mean_r = float(np.mean(r))
    mean_g = float(np.mean(g))
    mean_b = float(np.mean(b))
    total = mean_r + mean_g + mean_b
    if total <= 0.0:
        return float("nan")
    return float(mean_g / total)


def cloud_fraction(
    rgb: np.ndarray, mask: np.ndarray | None = None, threshold: float = 0.75
) -> float:
    """以紅藍比門檻估算天空影像中的雲量比例（0～1）。"""
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("rgb 必須是形狀為 (H, W, 3) 的陣列。")

    r = arr[:, :, 0]
    b = arr[:, :, 2]

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.ndim == 3 and m.shape[2] == 1:
            m = m[:, :, 0]
        if m.shape != arr.shape[:2]:
            raise ValueError("mask 的形狀必須與 rgb 的前兩維相同。")
        r = r[m]
        b = b[m]
    else:
        r = r.ravel()
        b = b.ravel()

    if r.size == 0:
        return float("nan")

    valid = ~((b == 0.0) & (r == 0.0)) & ~(np.isnan(r) | np.isnan(b))
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return float("nan")

    r_v = r[valid]
    b_v = b[valid]
    ratio = np.divide(
        r_v,
        b_v,
        out=np.full_like(r_v, np.inf),
        where=(b_v != 0.0),
    )
    is_cloud = np.where(b_v == 0.0, True, ratio > threshold)
    return float(np.sum(is_cloud) / n_valid)


def logistic_fit(t: list[float], y: list[float]) -> dict:
    """對時間與生長量資料進行 S 形 Logistic 生長曲線的最小平方擬合。"""
    ta = np.asarray(t, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if ta.ndim != 1 or ya.ndim != 1 or ta.size != ya.size:
        raise ValueError("t 與 y 必須是長度相同的一維數列。")
    if ta.size < 4:
        raise ValueError("S 形曲線擬合至少需要 4 筆資料。")

    t_min = float(np.min(ta))
    t_max = float(np.max(ta))
    t_span = t_max - t_min
    if t_span <= 0.0:
        raise ValueError("t 的數值不可全部相同。")

    y_scale = float(np.max(np.abs(ya)))
    if y_scale == 0.0:
        return {
            "K": 0.0,
            "r": 1.0,
            "t0": float(np.mean(ta)),
            "r2": 1.0,
            "predict": [0.0] * int(ta.size),
        }

    # 正規化時間與數值到 O(1) 尺度，確保搜尋與最佳化數值穩定
    tn = (ta - t_min) / t_span
    yn = ya / y_scale

    def _grid_best(r_grid: np.ndarray, t0_grid: np.ndarray) -> tuple[float, float, float, float]:
        best_sse = float("inf")
        best_k = 1.0
        best_r = float(r_grid[0])
        best_t0 = float(t0_grid[0])
        dt_mat = tn[None, :] - t0_grid[:, None]
        for rv in r_grid:
            z = np.clip(-rv * dt_mat, -500.0, 500.0)
            g = 1.0 / (1.0 + np.exp(z))
            gg = np.sum(g * g, axis=1)
            gy = g @ yn
            valid_g = gg > 1e-15
            if not np.any(valid_g):
                continue
            k_vec = np.zeros_like(gg)
            k_vec[valid_g] = gy[valid_g] / gg[valid_g]
            res = yn[None, :] - k_vec[:, None] * g
            sse_vec = np.sum(res * res, axis=1)
            sse_vec[~valid_g] = np.inf
            idx = int(np.argmin(sse_vec))
            if sse_vec[idx] < best_sse:
                best_sse = float(sse_vec[idx])
                best_k = float(k_vec[idx])
                best_r = float(rv)
                best_t0 = float(t0_grid[idx])
        return best_k, best_r, best_t0, best_sse

    # 第一階段：全區粗網格搜尋（依趨勢涵蓋正斜率，若遞減亦涵蓋負斜率）
    r_pos = np.geomspace(0.2, 80.0, 60)
    r_grid = np.concatenate((r_pos, -r_pos)) if float(np.cov(tn, yn)[0, 1]) < 0.0 else r_pos
    t0_grid = np.linspace(-0.3, 1.3, 65)
    kn, rn, t0n, sse = _grid_best(r_grid, t0_grid)

    # 第二階段：在最佳點附近做兩輪縮放細化
    r_half = 0.35 * abs(rn)
    t0_half = 0.08
    for _ in range(2):
        r_sub = np.linspace(rn - r_half, rn + r_half, 21)
        t0_sub = np.linspace(t0n - t0_half, t0n + t0_half, 21)
        kn, rn, t0n, sse = _grid_best(r_sub, t0_sub)
        r_half *= 0.25
        t0_half *= 0.25

    # 第三階段：Levenberg-Marquardt 非線性最小平方精修 (K, r, t0)
    lam = 1e-3
    for _ in range(60):
        dt = tn - t0n
        g = 1.0 / (1.0 + np.exp(np.clip(-rn * dt, -500.0, 500.0)))
        pred = kn * g
        err = yn - pred
        dg = g * (1.0 - g)

        jac = np.empty((tn.size, 3), dtype=np.float64)
        jac[:, 0] = g
        jac[:, 1] = kn * dg * dt
        jac[:, 2] = -kn * dg * rn

        jtj = jac.T @ jac
        jte = jac.T @ err
        diag = np.diag(np.diag(jtj) + 1e-12)
        try:
            step = np.linalg.solve(jtj + lam * diag, jte)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        kn_new = kn + float(step[0])
        rn_new = rn + float(step[1])
        t0n_new = t0n + float(step[2])
        g_new = 1.0 / (1.0 + np.exp(np.clip(-rn_new * (tn - t0n_new), -500.0, 500.0)))
        err_new = yn - kn_new * g_new
        sse_new = float(np.dot(err_new, err_new))

        if sse_new < sse:
            rel_imp = (sse - sse_new) / max(sse, 1e-15)
            kn, rn, t0n, sse = kn_new, rn_new, t0n_new, sse_new
            lam = max(lam / 3.0, 1e-12)
            if rel_imp < 1e-13 and float(np.linalg.norm(step)) < 1e-10:
                break
        else:
            lam = min(lam * 5.0, 1e12)
            if lam >= 1e12:
                break

    k_val = kn * y_scale
    r_val = rn / t_span
    t0_val = t_min + t0n * t_span

    y_pred = k_val / (1.0 + np.exp(np.clip(-r_val * (ta - t0_val), -500.0, 500.0)))
    ss_res = float(np.sum((ya - y_pred) ** 2))
    ss_tot = float(np.sum((ya - np.mean(ya)) ** 2))
    if ss_tot > 0.0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = 1.0 if ss_res == 0.0 else 0.0

    return {
        "K": float(k_val),
        "r": float(r_val),
        "t0": float(t0_val),
        "r2": float(r2),
        "predict": [float(v) for v in y_pred],
    }


def linear_fit(x: list[float], y: list[float]) -> dict:
    """對兩組數列進行直線 y = a x + b 的最小平方擬合並計算決定係數。"""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.ndim != 1 or ya.ndim != 1 or xa.size != ya.size:
        raise ValueError("x 與 y 必須是長度相同的一維數列。")
    if xa.size < 2:
        raise ValueError("直線擬合至少需要 2 筆資料。")

    x_mean = float(np.mean(xa))
    y_mean = float(np.mean(ya))
    dx = xa - x_mean
    dy = ya - y_mean
    sxx = float(np.dot(dx, dx))
    if sxx == 0.0:
        raise ValueError("x 的數值不可全部相同。")

    slope = float(np.dot(dx, dy) / sxx)
    intercept = float(y_mean - slope * x_mean)

    y_pred = slope * xa + intercept
    ss_res = float(np.sum((ya - y_pred) ** 2))
    ss_tot = float(np.dot(dy, dy))
    if ss_tot > 0.0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = 1.0 if ss_res == 0.0 else 0.0

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": float(r2),
    }


def compare_table(
    observed: list[float | None], predicted: list[float | None]
) -> dict:
    """比較觀測值與模型預測值的差異，統計有效筆數、平均差、平均絕對差與最大絕對差。"""
    if len(observed) != len(predicted):
        raise ValueError("觀測值與模型值的長度必須相同。")

    diffs_list: list[float] = []
    for obs, pred in zip(observed, predicted):
        if obs is None or pred is None:
            continue
        o_val = float(obs)
        p_val = float(pred)
        if math.isnan(o_val) or math.isnan(p_val):
            continue
        diffs_list.append(o_val - p_val)

    n = len(diffs_list)
    if n == 0:
        return {
            "n": 0,
            "mean_diff": None,
            "mean_abs_diff": None,
            "max_abs_diff": None,
        }

    diffs = np.asarray(diffs_list, dtype=np.float64)
    abs_diffs = np.abs(diffs)
    return {
        "n": n,
        "mean_diff": float(np.mean(diffs)),
        "mean_abs_diff": float(np.mean(abs_diffs)),
        "max_abs_diff": float(np.max(abs_diffs)),
    }
