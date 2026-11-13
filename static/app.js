"use strict";

// sequential ramp (ฟ้าอ่อน -> น้ำเงินเข้ม) — ชุดเดียวกับไฟล์ Excel
const RAMP = [
  "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
  "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
];
const WHITE_TEXT_FROM = 7;
const FONT = '"IBM Plex Sans Thai", "Leelawadee UI", Tahoma, sans-serif';

const $ = (id) => document.getElementById(id);
const state = { result: null, lyrics: "", view: "words", info: null, youtube: null, src: "youtube" };
const ytEnabled = () => Boolean(state.info?.youtube?.enabled);
const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");

const isDark = () => {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : darkQuery.matches;
};

/** สีของช่อง heatmap: โหมดมืดกลับด้าน ramp เพื่อให้ค่าน้อยกลืนกับพื้นหลังเข้ม */
function cellColor(v, dark = isDark()) {
  const step = Math.round(v * (RAMP.length - 1));
  const i = dark ? RAMP.length - 1 - step : step;
  return { bg: RAMP[i], fg: i >= WHITE_TEXT_FROM ? "#ffffff" : "#0b0b0b" };
}

const pct = (v) => `${(v * 100).toFixed(1)}%`;
const fmtInt = (n) => Math.round(n).toLocaleString("th-TH");
const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// ------------------------------------------------------------------ API
async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let msg = `เกิดข้อผิดพลาด (${res.status})`, code = "";
    try { const j = await res.json(); msg = j.error || msg; code = j.code || ""; } catch { /* not json */ }
    throw Object.assign(new Error(msg), { code });
  }
  return res;
}

// ------------------------------------------------------------------ input
function updateCounter() {
  const text = $("lyrics").value;
  const lines = text.split("\n").filter((l) => l.trim()).length;
  $("counter").textContent = `${fmtInt(lines)} บรรทัด · ${fmtInt(text.length)} ตัวอักษร`;
}

function showError(msg) {
  $("error").textContent = msg;
  $("error").hidden = !msg;
}

async function predict() {
  const lyrics = $("lyrics").value.trim();
  if (!lyrics) { showError("กรุณาใส่เนื้อเพลงก่อน"); $("lyrics").focus(); return; }
  showError("");
  const btn = $("btnPredict");
  btn.disabled = true;
  btn.textContent = "กำลังวิเคราะห์…";
  try {
    const result = await (await api("/api/predict", { lyrics })).json();
    state.result = result;
    state.lyrics = lyrics;
    state.youtube = null;
    renderResult(result);
    renderHeatmap();
    if (ytEnabled()) {
      $("ytQuery").value = "";
      searchYoutube();
      refreshLiveViews(result.similar);
    } else {
      renderYoutubeDisabled();
    }
  } catch (e) {
    showError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "ทำนายยอดวิว";
  }
}

async function loadExample() {
  try {
    const ex = await (await api("/api/example")).json();
    $("lyrics").value = ex.lyrics;
    updateCounter();
    showError("");
    predict();
  } catch (e) {
    showError(e.message);
  }
}

// ------------------------------------------------------------------ result
function renderResult(r) {
  const p = r.prediction;
  $("resultEmpty").hidden = true;
  $("resultBody").hidden = false;
  $("heroViews").textContent = p.views_text;
  $("heroSub").innerHTML =
    `≈ ${fmtInt(p.views)} ครั้ง · ระดับ <b>${esc(p.level_name)}</b> (${esc(p.level_range)}) · ความมั่นใจ ${pct(p.confidence)}`;

  $("probs").innerHTML = r.probabilities.map((c, i) => `
    <div class="prob-row${i === p.level ? " is-top" : ""}" role="listitem"
         data-tip="<b>${esc(c.name)}</b> ${esc(c.range)}<br>ความน่าจะเป็น <b>${pct(c.prob)}</b>">
      <div class="prob-label">${esc(c.name)}${i === p.level ? '<span class="tag">คาดการณ์</span>' : ""}<small>${esc(c.range)}</small></div>
      <div class="prob-track"><div class="prob-bar" style="width:${(c.prob * 100).toFixed(2)}%"></div></div>
      <div class="prob-value">${pct(c.prob)}</div>
    </div>`).join("");

  const t = r.tokens;
  $("coverage").textContent =
    `โมเดลรู้จัก ${fmtInt(t.known)} จาก ${fmtInt(t.total)} คำ (หลังตัดคำและตัด stopwords)` +
    (t.known / t.total < 0.5 ? " — คำที่ไม่รู้จักมีมาก ผลอาจคลาดเคลื่อน" : "");

  $("similar").innerHTML = r.similar.length
    ? r.similar.map((s) => {
        const title = s.url
          ? `<a class="sim-title" href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.title)}</a>`
          : `<span class="sim-title">${esc(s.title)}</span>`;
        const link = s.url
          ? `link: <a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.url)}</a>`
          : `<span class="none">ไม่มีลิงก์${state.info?.is_demo ? " (ข้อมูลจำลอง)" : ""}</span>`;
        return `<li${s.url ? ` data-url="${esc(s.url)}"` : ""}>
          <div>
            ${title}
            <div class="sim-meta">${esc(s.artist || "ไม่ระบุศิลปิน")} · ระดับ${esc(s.level_name)}</div>
            <div class="sim-link">${link}</div>
          </div>
          <div class="sim-side">
            <div class="sim-views" title="ยอดวิวตอนเก็บข้อมูล ${fmtInt(s.views)} ครั้ง">${esc(s.views_text)} วิว</div>
            <div class="sim-score">คล้าย ${pct(s.similarity)}</div>
          </div>
        </li>`;
      }).join("")
    : '<li class="empty-row">ไม่พบเพลงที่มีคำซ้ำกัน</li>';
}

// ------------------------------------------------------------------ YouTube (real-time)
const MATCH_REASON = { lyrics: "เนื้อเพลงตรงกัน", title: "ชื่อมีคำในเนื้อเพลง", hook: "YouTube จับคู่ท่อนเพลง", words: "เนื้อหาคล้าย" };

function setSource(src) {
  state.src = src;
  document.querySelectorAll("#simTabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.src === src)));
  $("simYoutube").hidden = src !== "youtube";
  $("simDataset").hidden = src !== "dataset";
}

function ytStatus(msg, isError = false) {
  $("ytStatus").textContent = msg;
  $("ytStatus").classList.toggle("error", isError);
}

function renderYoutubeDisabled() {
  $("ytForm").hidden = true;
  ytStatus("");
  $("ytBench").textContent = "";
  $("ytResults").innerHTML = `<li class="empty-row"><div class="notice">
    ยังไม่ได้เชื่อม <b>YouTube Data API</b> — สร้าง API key ใน Google Cloud Console
    แล้วใส่ <code>YOUTUBE_API_KEY=...</code> ในไฟล์ <code>.env</code> จากนั้นรีสตาร์ทแอป (ดูขั้นตอนใน README)</div></li>`;
}

async function searchYoutube(query = "") {
  if (!state.lyrics) return;
  const btn = $("btnYtSearch");
  btn.disabled = true;
  $("ytForm").hidden = false;
  ytStatus("กำลังค้นหาบน YouTube…");
  $("ytResults").innerHTML = "";
  $("ytBench").textContent = "";
  try {
    const data = await (await api("/api/youtube/search", { lyrics: state.lyrics, query })).json();
    state.youtube = data;
    renderYoutube(data);
  } catch (e) {
    state.youtube = null;
    ytStatus(e.message, true);
    if (e.code === "no_key") renderYoutubeDisabled();
  } finally {
    btn.disabled = false;
  }
}

function renderYoutube(data) {
  $("ytQuery").value = data.query;
  const extra = data.extra_queries?.length ? ` · ค้นเพิ่มด้วย "${data.extra_queries.join('", "')}"` : "";
  ytStatus(data.results.length
    ? `ผลค้นหา ${data.results.length} เพลง เรียงตามความตรงกับเนื้อเพลง รวมเวอร์ชันซ้ำแล้ว${extra}`
    : "ไม่พบวิดีโอ ลองแก้คำค้นแล้วกดค้นใหม่");
  $("ytResults").innerHTML = data.results.map((v) => {
    const year = v.published_at ? ` · ${new Date(v.published_at).getFullYear() + 543}` : "";
    const versions = v.versions > 1 ? ` · รวม ${v.versions} เวอร์ชัน` : "";
    const score = v.lyric_similarity >= 0.01 ? `${MATCH_REASON[v.match_reason] || "ตรงกัน"} ${pct(v.lyric_similarity)}` : "พบจากคำค้น";
    const why = v.signals
      ? `ท่อนเพลงตรงกัน ${pct(v.signals.lines)} · ชื่อเพลงตรง ${pct(v.signals.title)} · คำคล้าย ${pct(v.signals.words)}`
      : "";
    const views = v.views === null ? "ซ่อนยอดวิว" : `${esc(v.views_text)} วิว`;
    return `<li>
      <a class="yt-thumb" href="${esc(v.url)}" target="_blank" rel="noopener noreferrer" tabindex="-1" aria-hidden="true">
        ${v.thumbnail ? `<img src="${esc(v.thumbnail)}" alt="" loading="lazy" referrerpolicy="no-referrer">` : ""}
      </a>
      <div>
        <a class="sim-title" href="${esc(v.url)}" target="_blank" rel="noopener noreferrer">${esc(v.title)}</a>
        <div class="sim-meta">${esc(v.channel)}${year}${versions}</div>
        <div class="sim-link">link: <a href="${esc(v.url)}" target="_blank" rel="noopener noreferrer">${esc(v.url)}</a></div>
      </div>
      <div class="sim-side">
        <div class="sim-views" title="${v.views === null ? "" : fmtInt(v.views) + " ครั้ง"}"><span class="live-dot" aria-hidden="true"></span>${views}</div>
        <div class="sim-score" title="${esc(why)}">${score}</div>
      </div>
    </li>`;
  }).join("");
  $("ytBench").innerHTML = data.median_views_text
    ? `ค่ากลางยอดวิวของเพลงเหล่านี้บน YouTube ตอนนี้ <b>${esc(data.median_views_text)} วิว</b> · โมเดลคาด <b>${esc(state.result.prediction.views_text)} วิว</b>` +
      ` <span class="tt-muted">(ดึงเมื่อ ${esc(new Date(data.fetched_at).toLocaleTimeString("th-TH"))}${data.cached ? ", ใช้ผลค้นหาที่เก็บไว้" : ""})</span>`
    : "";
}

/** ยอดวิวล่าสุดของเพลงที่คล้ายกันในชุดข้อมูล (เฉพาะเพลงที่มีลิงก์ YouTube) */
async function refreshLiveViews(similar) {
  const urls = similar.map((s) => s.url).filter(Boolean);
  if (!urls.length) return;
  try {
    const { stats } = await (await api("/api/youtube/stats", { urls })).json();
    document.querySelectorAll("#similar li[data-url]").forEach((li) => {
      const s = stats[li.dataset.url];
      if (!s) return;
      const el = li.querySelector(".sim-views");
      el.innerHTML = `<span class="live-dot" aria-hidden="true"></span>${esc(s.views_text)} วิว`;
      el.title = `ยอดวิวล่าสุด ${fmtInt(s.views)} ครั้ง (${el.title})`;
    });
  } catch { /* ใช้ยอดวิวจากชุดข้อมูลต่อไป */ }
}

// ------------------------------------------------------------------ heatmap
function heatmapRows(hm, view) {
  return view === "words"
    ? hm.words.map((w) => ({ label: w.word, extra: w.count, scores: w.scores, top: w.top }))
    : hm.lines.map((l, i) => ({ label: l.text, extra: i + 1, scores: l.scores, top: l.top }));
}

function renderHeatmap() {
  const r = state.result;
  if (!r) return;
  const hm = r.heatmap;
  const words = state.view === "words";
  const ranges = r.probabilities.map((p) => p.range);
  const rows = heatmapRows(hm, state.view);

  $("heatmapCard").hidden = false;
  $("heatmapSub").textContent = words
    ? "แต่ละคำสัมพันธ์กับยอดวิวระดับไหน — P(ระดับ | คำ) จาก Naive Bayes (แต่ละแถวรวม 100%)"
    : "ทำนายระดับยอดวิวทีละบรรทัด — บรรทัดสีเทาคือไม่มีคำที่โมเดลรู้จัก";

  const head = words
    ? `<th class="left">คำ</th><th>ความถี่</th>`
    : `<th>#</th><th class="left">เนื้อเพลง</th>`;
  const classHeads = hm.class_names.map((n, i) => `<th>${esc(n)}<small>${esc(ranges[i])}</small></th>`).join("");

  const body = rows.map((row) => {
    const lead = words
      ? `<td class="word left">${esc(row.label)}</td><td class="num">${row.extra}</td>`
      : `<td class="num">${row.extra}</td><td class="line left">${esc(row.label)}</td>`;
    const cells = row.scores
      ? row.scores.map((v, k) => {
          const c = cellColor(v);
          const tip = `<b>${esc(row.label.length > 40 ? row.label.slice(0, 40) + "…" : row.label)}</b><br>` +
            `<span class="tt-muted">ระดับ${esc(hm.class_names[k])} (${esc(ranges[k])})</span><br>ความน่าจะเป็น <b>${pct(v)}</b>`;
          return `<td class="cell" style="background:${c.bg};color:${c.fg}" data-tip="${esc(tip)}">${pct(v)}</td>`;
        }).join("")
      : hm.class_names.map(() => '<td class="cell null">–</td>').join("");
    const top = row.top === null || row.top === undefined ? "–" : esc(hm.class_names[row.top]);
    return `<tr${row.scores ? "" : ' class="muted"'}>${lead}${cells}<td class="top-name">${top}</td></tr>`;
  }).join("");

  $("heatmapTable").innerHTML =
    `<thead><tr>${head}${classHeads}<th>แนวโน้ม</th></tr></thead><tbody>${body ||
      `<tr><td colspan="${hm.class_names.length + 3}">ไม่มีข้อมูล</td></tr>`}</tbody>`;

  const unknown = hm.unknown_words;
  $("unknownWords").textContent = words && unknown.length
    ? `คำที่โมเดลไม่รู้จัก (${unknown.length}): ${unknown.slice(0, 40).join(", ")}${unknown.length > 40 ? " …" : ""}`
    : "";

  const stops = RAMP.map((_, i) => cellColor(i / (RAMP.length - 1)).bg);
  $("legendRamp").style.background = `linear-gradient(90deg, ${stops.join(", ")})`;
}

function setView(view) {
  state.view = view;
  document.querySelectorAll("#viewTabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.view === view)));
  renderHeatmap();
}

// ------------------------------------------------------------------ export
function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function exportXlsx() {
  if (!state.result) return;
  const btn = $("btnXlsx");
  btn.disabled = true;
  try {
    const youtube = state.youtube?.results ?? [];
    const blob = await (await api("/api/export/xlsx", { lyrics: state.lyrics, youtube })).blob();
    download(blob, "lyrics_heatmap.xlsx");
  } catch (e) {
    showError(e.message);
  } finally {
    btn.disabled = false;
  }
}

function fitText(ctx, text, maxWidth) {
  if (ctx.measureText(text).width <= maxWidth) return text;
  let t = text;
  while (t.length > 1 && ctx.measureText(t + "…").width > maxWidth) t = t.slice(0, -1);
  return t + "…";
}

/** วาดตาราง heatmap (มุมมองปัจจุบัน) ลง canvas แล้วบันทึกเป็น PNG — ใช้สีโหมดสว่างเสมอ */
async function exportPng() {
  const r = state.result;
  if (!r) return;
  await document.fonts.ready;
  const hm = r.heatmap;
  const words = state.view === "words";
  const rows = heatmapRows(hm, state.view);
  const ranges = r.probabilities.map((p) => p.range);
  const n = hm.class_names.length;

  const S = 2, PAD = 28, ROW = 32, HEAD = 52, GAP = 2;
  const probe = document.createElement("canvas").getContext("2d");
  probe.font = `500 14px ${FONT}`;
  const labelW = Math.min(words ? 260 : 460,
    Math.max(120, ...rows.map((x) => probe.measureText(String(x.label)).width + 24)));
  const extraW = 64, cellW = 128, topW = 96;
  const cols = words
    ? [{ w: labelW, kind: "label" }, { w: extraW, kind: "extra" }]
    : [{ w: extraW, kind: "extra" }, { w: labelW, kind: "label" }];
  const tableW = cols.reduce((a, c) => a + c.w, 0) + n * cellW + topW + GAP * (cols.length + n);
  const W = PAD * 2 + tableW;
  const titleH = 78, legendH = 56;
  const H = PAD + titleH + HEAD + rows.length * (ROW + GAP) + legendH + PAD;

  const canvas = document.createElement("canvas");
  canvas.width = W * S;
  canvas.height = H * S;
  const ctx = canvas.getContext("2d");
  ctx.scale(S, S);
  ctx.fillStyle = "#fcfcfb";
  ctx.fillRect(0, 0, W, H);
  ctx.textBaseline = "middle";

  const p = r.prediction;
  ctx.fillStyle = "#0b0b0b";
  ctx.font = `600 20px ${FONT}`;
  ctx.fillText(`Heatmap ${words ? "รายคำ" : "รายบรรทัด"} ของเนื้อเพลง`, PAD, PAD + 12);
  ctx.fillStyle = "#52514e";
  ctx.font = `14px ${FONT}`;
  ctx.fillText(`ยอดวิวที่คาด ≈ ${p.views_text} วิว · ระดับ${p.level_name} (${p.level_range}) · ความมั่นใจ ${pct(p.confidence)}`, PAD, PAD + 40);
  ctx.fillText(words ? "ค่าในช่อง = P(ระดับ | คำ) จาก Naive Bayes, แต่ละแถวรวม 100%" : "ค่าในช่อง = ความน่าจะเป็นของระดับยอดวิวเมื่อทำนายทีละบรรทัด", PAD, PAD + 62);

  // header
  let y = PAD + titleH;
  let x = PAD;
  ctx.fillStyle = "#1a1a19";
  ctx.fillRect(PAD, y, tableW, HEAD);
  const headers = [
    ...cols.map((c) => ({ w: c.w, t: c.kind === "label" ? (words ? "คำ" : "เนื้อเพลง") : (words ? "ความถี่" : "#"), left: c.kind === "label" })),
    ...hm.class_names.map((name, i) => ({ w: cellW, t: name, s: ranges[i] })),
    { w: topW, t: "แนวโน้ม" },
  ];
  for (const h of headers) {
    ctx.fillStyle = "#ffffff";
    ctx.font = `600 14px ${FONT}`;
    ctx.textAlign = h.left ? "left" : "center";
    const tx = h.left ? x + 10 : x + h.w / 2;
    ctx.fillText(h.t, tx, y + (h.s ? 18 : HEAD / 2));
    if (h.s) {
      ctx.font = `12px ${FONT}`;
      ctx.fillStyle = "#c3c2b7";
      ctx.fillText(fitText(ctx, h.s, h.w - 8), tx, y + 36);
    }
    x += h.w + GAP;
  }
  y += HEAD + GAP;

  // rows
  for (const row of rows) {
    x = PAD;
    for (const c of cols) {
      const isLabel = c.kind === "label";
      ctx.fillStyle = row.scores || !isLabel ? "#0b0b0b" : "#898781";
      ctx.font = isLabel ? `500 14px ${FONT}` : `14px ${FONT}`;
      ctx.textAlign = isLabel ? "left" : "center";
      const text = isLabel ? fitText(ctx, String(row.label), c.w - 16) : String(row.extra);
      ctx.fillText(text, isLabel ? x + 10 : x + c.w / 2, y + ROW / 2);
      x += c.w + GAP;
    }
    for (let k = 0; k < n; k++) {
      const v = row.scores ? row.scores[k] : null;
      const col = v === null ? { bg: "#f0efec", fg: "#898781" } : cellColor(v, false);
      ctx.fillStyle = col.bg;
      ctx.beginPath();
      ctx.roundRect(x, y, cellW, ROW, 4);
      ctx.fill();
      ctx.fillStyle = col.fg;
      ctx.font = `14px ${FONT}`;
      ctx.textAlign = "center";
      ctx.fillText(v === null ? "–" : pct(v), x + cellW / 2, y + ROW / 2);
      x += cellW + GAP;
    }
    ctx.fillStyle = "#0b0b0b";
    ctx.font = `600 14px ${FONT}`;
    ctx.fillText(row.top === null || row.top === undefined ? "–" : hm.class_names[row.top], x + topW / 2, y + ROW / 2);
    ctx.strokeStyle = "#e1e0d9";
    ctx.beginPath();
    ctx.moveTo(PAD, y + ROW + GAP / 2);
    ctx.lineTo(PAD + tableW, y + ROW + GAP / 2);
    ctx.stroke();
    y += ROW + GAP;
  }

  // legend
  y += 20;
  ctx.textAlign = "left";
  ctx.fillStyle = "#898781";
  ctx.font = `13px ${FONT}`;
  ctx.fillText("0%", PAD, y + 7);
  const gx = PAD + 28, gw = 200;
  const grad = ctx.createLinearGradient(gx, 0, gx + gw, 0);
  RAMP.forEach((c, i) => grad.addColorStop(i / (RAMP.length - 1), c));
  ctx.fillStyle = grad;
  ctx.fillRect(gx, y + 2, gw, 10);
  ctx.fillStyle = "#898781";
  ctx.fillText("100%   สีเข้มขึ้น = สัมพันธ์กับระดับยอดวิวนั้นมากขึ้น", gx + gw + 8, y + 7);

  canvas.toBlob((blob) => download(blob, `lyrics_heatmap_${words ? "words" : "lines"}.png`), "image/png");
}

// ------------------------------------------------------------------ tooltip
function bindTooltip() {
  const tip = $("tooltip");
  document.addEventListener("mouseover", (e) => {
    const el = e.target.closest("[data-tip]");
    if (!el) { tip.hidden = true; return; }
    tip.innerHTML = el.dataset.tip;
    tip.hidden = false;
  });
  document.addEventListener("mousemove", (e) => {
    if (tip.hidden) return;
    const pad = 14;
    const { offsetWidth: w, offsetHeight: h } = tip;
    let x = e.clientX + pad, y = e.clientY + pad;
    if (x + w > window.innerWidth - 8) x = e.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = e.clientY - h - pad;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  });
  document.addEventListener("scroll", () => { tip.hidden = true; }, true);
}

// ------------------------------------------------------------------ model info
async function loadInfo() {
  try {
    const info = await (await api("/api/info")).json();
    state.info = info;
    $("demoBadge").hidden = !info.is_demo;
    const r = info.report;
    $("modelInfo").innerHTML = [
      `ข้อมูลเทรน <b>${fmtInt(r.n_songs)}</b> เพลง${info.is_demo ? " (จำลอง)" : ""}`,
      `คลังคำ <b>${fmtInt(r.vocab_size)}</b> คำ`,
      `Accuracy (test) <b>${pct(r.test_accuracy)}</b> เทียบ baseline ${pct(r.baseline_accuracy)}`,
      `Macro-F1 <b>${r.test_f1_macro.toFixed(3)}</b>`,
      `α = <b>${r.best_params.nb__alpha}</b>`,
      info.youtube.enabled
        ? `YouTube API <b>เชื่อมต่อแล้ว</b> (ค้นวันนี้ ${info.youtube.searches_today}/${info.youtube.daily_search_limit} ครั้ง)`
        : "YouTube API <b>ยังไม่ได้ตั้งค่า</b>",
    ].map((s) => `<span>${s}</span>`).join("");
    const ytTab = document.querySelector('#simTabs button[data-src="youtube"]');
    ytTab.classList.toggle("off", !info.youtube.enabled);
    setSource(info.youtube.enabled ? "youtube" : "dataset");
  } catch (e) {
    showError(e.message);
  }
}

// ------------------------------------------------------------------ init
$("btnPredict").addEventListener("click", predict);
$("btnExample").addEventListener("click", loadExample);
$("btnClear").addEventListener("click", () => {
  $("lyrics").value = "";
  updateCounter();
  showError("");
  $("lyrics").focus();
});
$("lyrics").addEventListener("input", updateCounter);
$("lyrics").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); predict(); }
});
document.querySelectorAll("#viewTabs button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
document.querySelectorAll("#simTabs button").forEach((b) => b.addEventListener("click", () => setSource(b.dataset.src)));
$("ytForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("ytQuery").value.trim();
  // ไม่ได้แก้คำค้น -> ให้เซิร์ฟเวอร์เลือกคำเอง (ค้นได้ 2 รอบ) แทนการค้นด้วยคำเดิมรอบเดียว
  searchYoutube(q === state.youtube?.query ? "" : q);
});
$("btnXlsx").addEventListener("click", exportXlsx);
$("btnPng").addEventListener("click", exportPng);
darkQuery.addEventListener("change", renderHeatmap);
bindTooltip();
updateCounter();
loadInfo();
