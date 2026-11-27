// 天空日記接收端 Code.gs 的本機測試：用假的 SpreadsheetApp／DriveApp／Utilities 跑 doPost，不連任何網路。
// 用法：node test_receiver.js
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm"), assert = require("assert");

function makeFakes() {
  const sheets = {}, files = [];
  function Sheet(name) { this.name = name; this.data = []; }
  Sheet.prototype.getLastRow = function () { return this.data.length; };
  Sheet.prototype.appendRow = function (row) {
    row.forEach(v => { if (typeof v === "string" && /^[=+\-@]/.test(v)) throw new Error("公式注入：" + v); });
    this.data.push(row.slice());
  };
  Sheet.prototype.setFrozenRows = function () {};
  Sheet.prototype.clear = function () { this.data = []; };
  Sheet.prototype.getDataRange = function () { const sh = this; return { getValues: () => sh.data.map(r => r.slice()) }; };
  Sheet.prototype.getRange = function (r, c, nr, nc) {
    const sh = this;
    return { setValues(vals) { vals.forEach((row, i) => { sh.data[r - 1 + i] = row.slice(); }); } };
  };
  const ss = {
    getSheetByName: n => sheets[n] || null,
    insertSheet: n => (sheets[n] = new Sheet(n)),
  };
  function Folder(name) { this.name = name; this.sub = {}; this.files = []; }
  Folder.prototype.getFoldersByName = function (n) { const f = this.sub[n]; let used = false; return { hasNext: () => !!f && !used, next: () => { used = true; return f; } }; };
  Folder.prototype.createFolder = function (n) { return (this.sub[n] = new Folder(n)); };
  Folder.prototype.getFilesByName = function (n) { const list = this.files.filter(f => f.name === n && !f.trashed); let i = 0; return { hasNext: () => i < list.length, next: () => list[i++] }; };
  Folder.prototype.createFile = function (blob) { const f = { name: blob.name, bytes: blob.bytes, trashed: false, setTrashed(v) { this.trashed = v; }, getUrl() { return "https://drive.google.com/file/d/FAKE/" + this.name; } }; this.files.push(f); files.push(f); return f; };
  const root = new Folder("root");
  return {
    sheets, files, root,
    ctx: {
      SpreadsheetApp: { getActiveSpreadsheet: () => ss, getUi: () => ({ createMenu: () => ({ addItem() { return this; }, addToUi() {} }) }) },
      DriveApp: { getRootFolder: () => root },
      Utilities: {
        base64Decode: s => Array.from(Buffer.from(s, "base64")).map(b => (b > 127 ? b - 256 : b)),   // Apps Script 回傳有號位元組
        newBlob: (bytes, type, name) => ({ bytes, type, name }),
      },
      LockService: { getScriptLock: () => ({ waitLock() {}, releaseLock() {} }) },
      ContentService: { MimeType: { JSON: "json" }, createTextOutput: t => ({ text: t, setMimeType() { return this; } }) },
      Date, JSON, Object, Array, Math, String, isFinite,
    },
  };
}

const code = fs.readFileSync(path.join(__dirname, "Code.gs"), "utf8");
function fresh() { const f = makeFakes(); vm.createContext(f.ctx); vm.runInContext(code, f.ctx); return f; }
const jpeg = Buffer.concat([Buffer.from([0xff, 0xd8, 0xff, 0xe0]), Buffer.alloc(3000, 7), Buffer.from([0xff, 0xd9])]).toString("base64");
const summary = { date: "20261127", photos: 193, dark_photos: 121, sun_found: 80, first_sun_time: "16:00:00", last_sun_time: "16:58:00",
  last_sun_x: 812.4, last_sun_y: 640.2, sun_x_1630: 503.1, sky_avg_rgb_1700: [180.2, 150.1, 120.4] };
const good = () => ({ v: 1, app: "skycam-desktop", app_version: "1.0.0", device_id: "0123456789abcdef0123456789abcdef", nickname: "小星",
  class_code: "6A", county: "臺北市", date: "20261127", camera_fw: "1.2.0", summary: JSON.parse(JSON.stringify(summary)), image: jpeg });
const post = (f, obj) => JSON.parse(f.ctx.doPost({ postData: { contents: typeof obj === "string" ? obj : JSON.stringify(obj) } }).text);

let f = fresh(), r;
r = post(f, good()); assert.strictEqual(r.ok, true, JSON.stringify(r));
assert.strictEqual(f.sheets.summaries.data.length, 2, "應有表頭＋1 列");
assert.strictEqual(f.root.sub["天空日記_收件"].sub["臺北市_小星_012345"].files.length, 1, "圖應存進對應資料夾");
r = post(f, good()); assert.strictEqual(r.ok, true);
assert.strictEqual(f.sheets.summaries.data.length, 2, "同一天重送應覆蓋，不新增列");
assert.strictEqual(f.files.filter(x => !x.trashed).length, 1, "舊圖應移到垃圾桶");
const g2 = good(); g2.date = g2.summary.date = "20261128"; assert.strictEqual(post(f, g2).ok, true); assert.strictEqual(f.sheets.summaries.data.length, 3);

const cases = [
  ["多一個欄位", o => { o.extra = 1; }],
  ["暱稱是電話", o => { o.nickname = "0912345678"; }],
  ["暱稱有 @", o => { o.nickname = "a@b"; }],
  ["縣市寫錯", o => { o.county = "台北"; }],
  ["裝置代碼不合法", o => { o.device_id = "../../x"; }],
  ["日期不合法", o => { o.date = "2026-11-27"; }],
  ["summary 少鍵", o => { delete o.summary.sun_x_1630; }],
  ["summary 日期不符", o => { o.summary.date = "20261128"; }],
  ["座標是字串", o => { o.summary.last_sun_x = "1"; }],
  ["圖片不是 JPEG", o => { o.image = Buffer.from("hello world").toString("base64"); }],
  ["圖片太大", o => { o.image = Buffer.concat([Buffer.from([0xff, 0xd8]), Buffer.alloc(210 * 1024)]).toString("base64"); }],
  ["v 不對", o => { o.v = 2; }],
];
for (const [name, mut] of cases) {
  const o = good(); mut(o); f = fresh(); r = post(f, o);
  assert.strictEqual(r.ok, false, name + " 應被拒絕"); assert.ok(!f.sheets.summaries, name + " 不應寫入試算表");
}
f = fresh(); assert.strictEqual(post(f, "not json").ok, false);
f = fresh(); const inj = good(); inj.class_code = ""; inj.nickname = "＝SUM"; assert.strictEqual(post(f, inj).ok, true, "全形等號暱稱可以收");
f = fresh(); assert.strictEqual(f.ctx.doGet().text, "天空日記夕陽摘要接收端運作中");
f = fresh(); post(f, good()); f.ctx.refreshStats(); assert.strictEqual(JSON.stringify(f.sheets["各縣市"].data[1]), JSON.stringify(["臺北市", 1, 1, 1]));   // 沙盒裡的陣列和外面的 Array 不是同一個原型，改比 JSON
console.log("接收端測試全部通過（" + (cases.length + 8) + " 項）");
