/**
 * 天空日記相機：夕陽摘要接收端（Google Apps Script，綁在一份試算表上）  v1
 *
 * 學生的「天空日記觀察站」在學生與家長同意後，每天傍晚送一份夕陽摘要：
 * 一張太陽軌跡圖縮圖（JPEG，≤ 200 KB）＋幾個數字＋暱稱、班級代碼、縣市。不收姓名、原始照片、電腦資訊。
 *
 * - doPost：嚴格驗證後，圖存進雲端硬碟資料夾「天空日記_收件」／<縣市>_<暱稱>_<裝置代碼前 6 碼>／日期.jpg，
 *   數字寫進「summaries」工作表（一台相機一天一列；同一天重送就覆蓋那一列）。
 * - doGet：回一行字確認運作中。
 * - 選單「天空日記 → 重新整理統計」：產生「各縣市」彙總。
 *
 * 部署步驟見同資料夾 README.md。測試請用本機假物件（test_receiver.js），不要對正式網址送測試資料。
 */

var CONFIG = {
  TOKEN: '',                    // 共用口令；留空＝不檢查（寫在公開程式裡，只能擋掃描，不是保密）
  FOLDER_NAME: '天空日記_收件',  // 雲端硬碟裡放圖片的資料夾（不存在會自動建立）
  MAX_BODY: 320000,             // 單次最多幾個字元（圖 200 KB 的 base64 約 273,000 字元）
  MAX_IMAGE_BYTES: 200 * 1024,
  SHEET: 'summaries',
  SHEET_COUNTY: '各縣市'
};

var COUNTIES = ['臺北市', '新北市', '桃園市', '臺中市', '臺南市', '高雄市', '基隆市', '新竹市', '嘉義市', '新竹縣', '苗栗縣',
  '彰化縣', '南投縣', '雲林縣', '嘉義縣', '屏東縣', '宜蘭縣', '花蓮縣', '臺東縣', '澎湖縣', '金門縣', '連江縣'];
var TOP_KEYS = ['v', 'app', 'app_version', 'device_id', 'nickname', 'class_code', 'county', 'date', 'camera_fw', 'summary', 'image'];
var SUM_KEYS = ['date', 'photos', 'dark_photos', 'sun_found', 'first_sun_time', 'last_sun_time', 'last_sun_x', 'last_sun_y', 'sun_x_1630', 'sky_avg_rgb_1700'];
var HEADER = ['收到時間', '日期', '縣市', '暱稱', '班級代碼', '裝置代碼', '一般照片', '暗曝光', '找到太陽', '第一次看到太陽', '最後看到太陽',
  '最後太陽 x', '最後太陽 y', '16:30 太陽 x', '17:00 天空 R', '17:00 天空 G', '17:00 天空 B', '韌體', '程式版本', '圖片連結'];

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function fail_(msg) { return json_({ ok: false, error: msg }); }

function doGet() {
  return ContentService.createTextOutput('天空日記夕陽摘要接收端運作中');
}

/** 字串開頭若是 = + - @，前面加 ' 讓試算表當純文字（防公式注入） */
function safe_(s) {
  s = String(s);
  return /^[=+\-@]/.test(s) ? "'" + s : s;
}

function isTime_(s) { return s === '' || /^\d{2}:\d{2}:\d{2}$/.test(s); }
function isNumOrNull_(x, lo, hi) { return x === null || (typeof x === 'number' && isFinite(x) && x >= lo && x <= hi); }
function isInt_(x, lo, hi) { return typeof x === 'number' && Math.floor(x) === x && x >= lo && x <= hi; }
function sameKeys_(obj, keys) {
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return false;
  var k = Object.keys(obj).sort();
  var want = keys.slice().sort();
  return k.length === want.length && k.every(function (v, i) { return v === want[i]; });
}

/** 驗證收到的資料；回傳錯誤訊息（中文）或空字串 */
function validate_(d) {
  if (!sameKeys_(d, TOP_KEYS)) return '欄位不符';
  if (d.v !== 1 || d.app !== 'skycam-desktop') return '版本或程式名稱不符';
  if (CONFIG.TOKEN && d.token !== CONFIG.TOKEN) return '口令不符';
  if (typeof d.app_version !== 'string' || !/^[0-9A-Za-z.\-]{1,16}$/.test(d.app_version)) return 'app_version 不合法';
  if (typeof d.device_id !== 'string' || !/^[0-9a-f]{32}$/.test(d.device_id)) return 'device_id 不合法';
  if (typeof d.nickname !== 'string' || d.nickname.length < 1 || d.nickname.length > 12 || /@/.test(d.nickname) ||
      (d.nickname.match(/\d/g) || []).length > 4 || /[\u0000-\u001f\u007f]/.test(d.nickname)) return '暱稱不合法';
  if (typeof d.class_code !== 'string' || !/^[0-9A-Za-z]{0,12}$/.test(d.class_code)) return '班級代碼不合法';
  if (COUNTIES.indexOf(d.county) < 0) return '縣市不合法';
  if (typeof d.date !== 'string' || !/^20\d{6}$/.test(d.date)) return '日期不合法';
  if (typeof d.camera_fw !== 'string' || !/^[0-9A-Za-z.\-]{0,16}$/.test(d.camera_fw)) return '韌體版本不合法';
  var s = d.summary;
  if (!sameKeys_(s, SUM_KEYS)) return 'summary 欄位不符';
  if (s.date !== d.date) return 'summary 日期不符';
  if (!isInt_(s.photos, 0, 2000) || !isInt_(s.dark_photos, 0, 2000) || !isInt_(s.sun_found, 0, 2000)) return '張數不合法';
  if (!isTime_(s.first_sun_time) || !isTime_(s.last_sun_time)) return '時間格式不合法';
  if (!isNumOrNull_(s.last_sun_x, 0, 5000) || !isNumOrNull_(s.last_sun_y, 0, 5000) || !isNumOrNull_(s.sun_x_1630, 0, 5000)) return '座標不合法';
  if (s.sky_avg_rgb_1700 !== null) {
    if (!Array.isArray(s.sky_avg_rgb_1700) || s.sky_avg_rgb_1700.length !== 3 ||
        !s.sky_avg_rgb_1700.every(function (v) { return isNumOrNull_(v, 0, 255) && v !== null; })) return '天空顏色不合法';
  }
  if (typeof d.image !== 'string' || !/^[A-Za-z0-9+\/]+={0,2}$/.test(d.image)) return '圖片格式不合法';
  return '';
}

function folder_(parent, name) {
  var it = parent.getFoldersByName(name);
  return it.hasNext() ? it.next() : parent.createFolder(name);
}

function doPost(e) {
  var raw = (e && e.postData && e.postData.contents) || '';
  if (raw.length === 0 || raw.length > CONFIG.MAX_BODY) return fail_('資料大小不合法');
  var d;
  try { d = JSON.parse(raw); } catch (err) { return fail_('不是 JSON'); }
  var bad = validate_(d);
  if (bad) return fail_(bad);

  var bytes = Utilities.base64Decode(d.image);
  if (bytes.length > CONFIG.MAX_IMAGE_BYTES) return fail_('圖片太大');
  if (!(bytes.length > 3 && (bytes[0] & 0xff) === 0xff && (bytes[1] & 0xff) === 0xd8)) return fail_('圖片不是 JPEG');

  var lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    var root = folder_(DriveApp.getRootFolder(), CONFIG.FOLDER_NAME);
    var sub = folder_(root, d.county + '_' + d.nickname.replace(/[\/\\]/g, '_') + '_' + d.device_id.slice(0, 6));
    var name = d.date + '.jpg';
    var old = sub.getFilesByName(name);
    while (old.hasNext()) old.next().setTrashed(true);   // 同一天重送：舊圖移到垃圾桶
    var file = sub.createFile(Utilities.newBlob(bytes, 'image/jpeg', name));

    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var sh = ss.getSheetByName(CONFIG.SHEET) || ss.insertSheet(CONFIG.SHEET);
    if (sh.getLastRow() === 0) { sh.appendRow(HEADER); sh.setFrozenRows(1); }
    var s = d.summary, rgb = s.sky_avg_rgb_1700 || ['', '', ''];
    var row = [new Date(), d.date, d.county, safe_(d.nickname), safe_(d.class_code), d.device_id, s.photos, s.dark_photos, s.sun_found,
      s.first_sun_time, s.last_sun_time, s.last_sun_x === null ? '' : s.last_sun_x, s.last_sun_y === null ? '' : s.last_sun_y,
      s.sun_x_1630 === null ? '' : s.sun_x_1630, rgb[0], rgb[1], rgb[2], d.camera_fw, d.app_version, file.getUrl()];
    var values = sh.getDataRange().getValues(), at = -1;
    for (var i = 1; i < values.length; i++) {
      if (values[i][1] === d.date && values[i][5] === d.device_id) { at = i + 1; break; }
    }
    if (at > 0) sh.getRange(at, 1, 1, row.length).setValues([row]);
    else sh.appendRow(row);
  } finally {
    lock.releaseLock();
  }
  return json_({ ok: true, saved: d.date });
}

function onOpen() {
  SpreadsheetApp.getUi().createMenu('天空日記').addItem('重新整理統計', 'refreshStats').addToUi();
}

/** 各縣市：幾台相機、幾天、最常見的最後看到太陽時間 */
function refreshStats() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sh = ss.getSheetByName(CONFIG.SHEET);
  if (!sh || sh.getLastRow() < 2) return;
  var v = sh.getDataRange().getValues().slice(1), by = {};
  v.forEach(function (r) {
    var c = r[2]; by[c] = by[c] || { devices: {}, days: 0, seen: 0 };
    by[c].devices[r[5]] = 1; by[c].days++; if (r[10]) by[c].seen++;
  });
  var out = [['縣市', '相機數', '收到天數', '有看到太陽的天數']];
  Object.keys(by).sort().forEach(function (c) { out.push([c, Object.keys(by[c].devices).length, by[c].days, by[c].seen]); });
  var st = ss.getSheetByName(CONFIG.SHEET_COUNTY) || ss.insertSheet(CONFIG.SHEET_COUNTY);
  st.clear();
  st.getRange(1, 1, out.length, out[0].length).setValues(out);
}
