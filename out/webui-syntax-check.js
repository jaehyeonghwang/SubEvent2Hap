
const $ = (s) => document.querySelector(s);
const SUB_COLORS = ["#4f46e5", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#a855f7", "#14b8a6", "#f97316"];
const EXAMPLES = ["heartbeat thumping steadily", "rain drops tapping on a window", "cat purring softly",
  "car engine revving", "footsteps on gravel", "notification buzz, two short pulses", "electric shock zap", "ocean waves rolling"];
let library = [], newNames = new Set(), questReady = false, generating = false;
let playState = null; // {name, start, dur}

// ---------- controls
for (const [id, outId, fmt] of [["duration", "durV", v => v + "초"], ["num", "numV", v => v + "개"], ["cfg", "cfgV", v => v],
                                ["temperature", "tempV", v => v], ["gain", "gainV", v => "×" + Number(v).toFixed(2)],
                                ["loop", "loopV", v => v + "회"]]) {
  const el = $("#" + id), out = $("#" + outId);
  const upd = () => out.textContent = fmt(el.value);
  el.addEventListener("input", upd); upd();
  try { const s = localStorage.getItem("hg_" + id); if (s !== null) { el.value = s; upd(); } } catch {}
  el.addEventListener("change", () => { try { localStorage.setItem("hg_" + id, el.value); } catch {} });
}
let hand = "both";
try { hand = localStorage.getItem("hg_hand") || "both"; } catch {}
const setHand = (v) => { hand = v; document.querySelectorAll("#hand button").forEach(b => b.classList.toggle("on", b.dataset.v === v));
  try { localStorage.setItem("hg_hand", v); } catch {} };
document.querySelectorAll("#hand button").forEach(b => b.onclick = () => setHand(b.dataset.v));
setHand(hand);
EXAMPLES.forEach(t => { const c = document.createElement("button"); c.className = "chip"; c.textContent = t;
  c.onclick = () => { $("#prompt").value = t; $("#prompt").focus(); }; $("#examples").appendChild(c); });

function toast(msg, ms = 2600) { const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), ms); }

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail || r.statusText));
  return j;
}

// ---------- status
async function pollStatus() {
  try {
    const s = await api("/api/status");
    const md = $("#mdot"), mt = $("#mtxt");
    md.className = "dot " + (s.model === "ready" ? "ok" : s.model === "error" ? "err" : "warn pulse");
    mt.textContent = s.model === "ready" ? (s.generating ? "생성 중…" : "준비됨") : s.model === "error" ? "오류: " + s.model_error : "로딩 중…";
    const q = s.quest; questReady = q.ready;
    $("#qdot").className = "dot " + (q.playing ? "ok pulse" : q.ready ? "ok" : q.connected ? "warn" : "err");
    $("#qtxt").textContent = q.status;
    $("#genBtn").disabled = generating || s.model !== "ready";
  } catch { $("#mdot").className = "dot err"; $("#mtxt").textContent = "서버 연결 끊김"; }
}
setInterval(pollStatus, 1000); pollStatus();

// ---------- library
async function loadLibrary() { library = await api("/api/library"); render(); }

function render() {
  const f = $("#filter").value.trim().toLowerCase();
  const list = library.filter(it => !f || it.prompt.toLowerCase().includes(f));
  $("#count").textContent = `${list.length}개`;
  const box = $("#items"); box.innerHTML = "";
  if (!list.length) { box.innerHTML = `<div class="empty">${library.length ? "검색 결과가 없습니다." : "아직 생성된 진동이 없습니다. 왼쪽에서 프롬프트를 입력해 생성해 보세요."}</div>`; return; }
  for (const it of list) box.appendChild(card(it));
}

function card(it) {
  const el = document.createElement("div");
  el.className = "card item" + (newNames.has(it.name) ? " new" : "");
  el.dataset.name = it.name;
  const when = new Date(it.created * 1000).toLocaleString("ko-KR", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
  el.innerHTML = `
    <div class="top"><div class="prompt"></div>${newNames.has(it.name) ? '<span class="badge">NEW</span>' : ""}${it.ok ? "" : '<span class="badge weak" title="진폭이 너무 작아 잘 안 느껴질 수 있음">약함</span>'}</div>
    <div class="meta"><span>${it.duration.toFixed(1)}s</span><span>peak ${it.peak.toFixed(2)}</span><span>rms ${it.rms.toFixed(3)}</span><span>${when}</span></div>
    <canvas class="wave"></canvas>
    <div class="actions">
      <button class="play" title="Quest 컨트롤러에서 재생">▶ Quest 재생</button>
      <button class="listen" title="PC 스피커로 미리듣기">🔊</button>
      <button class="regen" title="같은 프롬프트로 프롬프트 칸 채우기">↻</button>
      <button class="danger del" title="삭제">🗑</button>
    </div>`;
  el.querySelector(".prompt").textContent = it.prompt;
  const cv = el.querySelector("canvas");
  requestAnimationFrame(() => drawWave(cv, it));
  cv.onclick = () => playQuest(it);
  el.querySelector(".play").onclick = () => playQuest(it);
  el.querySelector(".listen").onclick = () => listen(it);
  el.querySelector(".regen").onclick = () => { $("#prompt").value = it.prompt; $("#prompt").focus(); };
  el.querySelector(".del").onclick = async () => {
    if (!confirm(`삭제할까요?\n${it.prompt} (${it.name})`)) return;
    await api("/api/library/" + encodeURIComponent(it.name), { method: "DELETE" }); loadLibrary(); };
  return el;
}

function drawWave(cv, it, progress = null) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (!w) return;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr);
  const css = getComputedStyle(document.documentElement);
  const mid = h / 2, n = it.peaks.length, scale = (h / 2 - 4) / Math.max(0.05, it.peak);
  if (it.regions) {
    it.regions.forEach((r, i) => {
      const x0 = r.start / it.duration * w, x1 = (r.start + r.duration) / it.duration * w;
      g.fillStyle = SUB_COLORS[i % SUB_COLORS.length]; g.globalAlpha = .14; g.fillRect(x0, 0, x1 - x0, h);
      g.globalAlpha = .9; g.font = "600 10px system-ui"; g.fillText(String(i + 1), x0 + 3, 11);
    });
    g.globalAlpha = 1;
  }
  g.strokeStyle = css.getPropertyValue("--line"); g.beginPath(); g.moveTo(0, mid); g.lineTo(w, mid); g.stroke();
  g.fillStyle = css.getPropertyValue("--wave");
  for (let i = 0; i < n; i++) {
    const [lo, hi] = it.peaks[i], x = i / n * w;
    g.globalAlpha = progress !== null && i / n > progress ? .45 : 1;
    g.fillRect(x, mid - hi * scale, Math.max(1, w / n - .3), Math.max(1, (hi - lo) * scale));
  }
  g.globalAlpha = 1;
  if (progress !== null) { g.fillStyle = css.getPropertyValue("--playhead"); g.fillRect(progress * w - 1, 0, 2, h); }
}

// ---------- actions
let audioEl = null;
function listen(it) {
  if (audioEl) audioEl.pause();
  audioEl = new Audio(`/audio/${encodeURIComponent(it.name)}?gain=${$("#gain").value}&normalize=${$("#normalize").checked}`);
  audioEl.play().catch(e => toast("미리듣기 실패: " + e.message));
}

async function playQuest(it) {
  if (!questReady) { toast("Quest가 준비되지 않았습니다 – Link를 켜고 헤드셋을 착용하세요"); return; }
  try {
    const loop = +$("#loop").value;
    await api("/api/play", { method: "POST", body: JSON.stringify({ name: it.name, hand, gain: +$("#gain").value, loop,
      normalize: $("#normalize").checked }) });
    playState = { name: it.name, it, start: performance.now(), dur: it.duration * 1000, loop };
    animatePlayhead();
  } catch (e) { toast("재생 실패: " + e.message); }
}

async function stopQuest() {
  clearTimeout(window._abT);
  try { await api("/api/stop", { method: "POST" }); } catch {}
  if (audioEl) audioEl.pause();
  endPlayhead();
}

// the same file can have a hidden library card and a visible compare card; animate the visible one
const visibleCard = (name) => [...document.querySelectorAll(`.item[data-name="${CSS.escape(name)}"]`)].find(e => e.offsetParent !== null);

function endPlayhead() {
  if (!playState) return;
  const el = visibleCard(playState.name);
  if (el) { el.classList.remove("playing"); drawWave(el.querySelector("canvas"), playState.it); }
  playState = null;
}
function animatePlayhead() {
  if (!playState) return;
  const ps = playState, t = performance.now() - ps.start;
  const el = visibleCard(ps.name);
  if (t >= ps.dur * ps.loop) { endPlayhead(); return; }
  if (el) { el.classList.add("playing"); drawWave(el.querySelector("canvas"), ps.it, (t % ps.dur) / ps.dur); }
  requestAnimationFrame(() => { if (playState === ps) animatePlayhead(); });
}

async function doGenerate() {
  const prompt = $("#prompt").value.trim();
  if (!prompt) { $("#prompt").focus(); return; }
  if (generating) return;
  generating = true;
  const btn = $("#genBtn"), t0 = performance.now();
  btn.disabled = true;
  const tick = setInterval(() => btn.innerHTML = `<span class="spinner"></span> 생성 중… ${((performance.now() - t0) / 1000).toFixed(0)}s`, 250);
  try {
    const seed = $("#seed").value.trim();
    const res = await api("/api/generate", { method: "POST", body: JSON.stringify({
      prompt, duration: +$("#duration").value, num: +$("#num").value, cfg: +$("#cfg").value,
      temperature: +$("#temperature").value, top_k: +$("#top_k").value, seed: seed === "" ? null : +seed }) });
    newNames = new Set(res.items.map(i => i.name));
    $("#filter").value = "";
    await loadLibrary();
    window.scrollTo({ top: 0, behavior: "smooth" });
    toast(`${res.items.length}개 생성 완료 (${res.elapsed.toFixed(1)}s)`);
    const first = res.items.find(i => i.ok) || res.items[0];
    if ($("#autoplay").checked && first && questReady) playQuest(first);
  } catch (e) { toast("생성 실패: " + e.message, 4000); }
  finally { clearInterval(tick); generating = false; btn.textContent = "생성하기"; pollStatus(); }
}

// ---------- tabs
let tab = "single";
function setTab(t) {
  tab = t;
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === t));
  $("#tab-single").hidden = t !== "single"; $("#genCard").hidden = t !== "single";
  $("#tab-compose").hidden = t !== "compose";
  try { localStorage.setItem("hg_tab", t); } catch {}
  if (t === "compose") requestAnimationFrame(() => { drawTimeline(); redrawCompare(); }); else render();
}
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => setTab(b.dataset.tab));

// ---------- LLM status
let llmOk = false;
async function pollLlm() {
  try {
    const s = await api("/api/llm/status"); llmOk = s.available;
    $("#ldot").className = "dot " + (s.available ? (s.loaded ? "ok" : "warn") : "err");
    $("#ltxt").textContent = s.status;
  } catch { $("#ldot").className = "dot err"; $("#ltxt").textContent = "확인 실패"; }
}
setInterval(pollLlm, 5000); pollLlm();

// ---------- compose: state & editor
const C_EXAMPLES = ["Someone knocks on a door three times and the door creaks open",
  "A rubber ball is dropped from 1 metre onto a wooden floor and bounces four times. Restitution is 0.6.",
  "A ball bounces three times on a wooden floor and then rolls away", "A sword clashes against a shield, then scrapes off",
  "Heavy rain starts, a thunder strikes nearby, then the rain fades", "A balloon is inflated until it pops",
  "An archer draws a bow, releases the arrow, and it hits a target", "A car engine starts, revs twice, and the car drives away"];
let cSubs = [], cPlan = null, cBusy = false;
let cPhysics = null, cHierarchy = null;
try { const s = JSON.parse(localStorage.getItem("hg_compose") || "null"); if (s) { cSubs = s.subs || []; cPhysics = s.physics || null; cHierarchy = s.hierarchy || null; $("#cEvent").value = s.event || ""; } } catch {}
const saveCompose = () => { try { localStorage.setItem("hg_compose", JSON.stringify({ event: $("#cEvent").value, subs: cSubs, physics: cPhysics, hierarchy: cHierarchy })); } catch {} };
$("#cPlanner").onchange = () => { $("#cBounceOptions").hidden = $("#cPlanner").value !== "ball_physics"; };
function showPhysics() {
  $("#cPhysics").hidden = !cPhysics;
  if (!cPhysics) return;
  const p = cPhysics.parameters, origins = {user_override: "직접 입력", text_extracted_by_lm: "문장에서 LM 추출", assumed_default: "가정값"};
  const names = {drop_height_m: "낙하 높이(m)", restitution: "반발계수", mass_kg: "질량(kg)", release_time_s: "놓는 시각(s)", impact_count: "충돌 수"};
  $("#cPhysicsInfo").textContent = Object.entries(p).map(([k,v]) => `${names[k] || k}: ${v} (${origins[cPhysics.provenance[k]]})`).join("\n")
    + "\n\n충돌 시각(s): " + cPhysics.impacts.map(i => i.time_s.toFixed(3)).join(", ")
    + "\n충돌 간격(s): " + cPhysics.inter_impact_intervals_s.map(t => t.toFixed(3)).join(", ")
    + "\n\n진동 길이는 접촉시간 예측이 아닌 재생 설정입니다. 강도는 정규화 충격량이며 실제 가속도·체감 강도가 아닙니다."
    + "\n재질 이름만으로 반발계수를 추정하지 않습니다.\n" + cPhysics.notes.join("\n");
}
showPhysics();
function showHierarchy() {
  $("#cHierarchy").hidden = !cHierarchy;
  if (!cHierarchy) return;
  const labels = {vertical_bounce: "공의 반발", free_fall: "낙하·비행", single_impact: "단발 충돌",
    sliding: "미끄러짐·마찰", rolling: "굴림 저항", damped_oscillation: "감쇠 진동", periodic: "주기 진동",
    stochastic: "불규칙 발생", rhythmic_action: "의도적 반복", composition: "복합 사건", lm_atomic: "LM 추정"};
  const stops = {model_covers_action: "모델이 전체 동작을 처리하여 분해 중단", atomic_action: "LM이 더 나누지 않기로 판단한 추정 (물리 모델 없음)", depth_limit: "깊이 한도로 중단", no_progress: "동일 사건 재분해 방지", no_valid_split: "추가 분해 근거 없음"};
  const lines = [];
  if (cHierarchy.regime_criterion) lines.push("원자 동작 기준: 접촉 상태·구속 조건·구동 방식이 같으면 한 동작으로 유지합니다. 같은 조건에서 감속해 정지하는 과정은 나누지 않습니다.\n");
  function visit(node, depth) {
    const pad = "  ".repeat(depth);
    lines.push(`${pad}${labels[node.model] || node.model}: ${node.description}`);
    if (node.stop_reason) lines.push(`${pad}  ${stops[node.stop_reason] || node.stop_reason}`);
    if (node.structure_review) lines.push(`${pad}  구조 검토: ${node.structure_review.reason}`);
    if (node.selection_reason) lines.push(`${pad}  근거: ${node.selection_reason}`);
    (node.model_rejections || []).forEach(r => lines.push(`${pad}  적용 거절: ${r.model} · ${r.reason}`));
    if (node.placement) lines.push(`${pad}  상위 시간축 위치: ${node.placement.offset_s.toFixed(3)}s (${node.placement.connection})`);
    if (node.process_duration_s !== undefined) lines.push(`${pad}  물리/과정 종료: ${node.process_duration_s.toFixed(3)}s · 재생 포함: ${node.duration_s.toFixed(3)}s`);
    if (node.state_transfer && node.state_transfer.kind !== "none") lines.push(`${pad}  상태 연결: ${node.state_transfer.kind}\n${pad}  ${JSON.stringify(node.incoming_state)}`);
    if (node.terminal_state && Object.keys(node.terminal_state).length) lines.push(`${pad}  종료 상태: ${JSON.stringify(node.terminal_state)}`);
    if (node.physics) lines.push(`${pad}  조건: ${JSON.stringify(node.physics.parameters)}\n${pad}  출처: ${JSON.stringify(node.physics.provenance)}`);
    (node.children || []).forEach(child => visit(child, depth + 1));
  }
  visit(cHierarchy.tree, 0);
  if (cHierarchy.connections.length) lines.push("\nLM이 지정한 동일 물체의 연속 동작에서 계산된 상태만 전달합니다. 좌표 방향 전환·미지 파라미터는 가정으로 남습니다.");
  lines.push("\n행동 리듬·확률 모델은 물리 법칙이 아니며, 감각 강도와 모델 간 진폭은 보정되지 않았습니다.");
  lines.push("주기·감쇠 모델은 정현파와 포락선을 해석적으로 합성합니다. 마찰 포락선은 생성된 질감 파형에 적용됩니다.");
  $("#cHierarchyInfo").textContent = lines.join("\n");
}
showHierarchy();
$("#cEvent").addEventListener("input", saveCompose);
C_EXAMPLES.forEach(t => { const c = document.createElement("button"); c.className = "chip"; c.textContent = t;
  c.onclick = () => { $("#cEvent").value = t; saveCompose(); }; $("#cExamples").appendChild(c); });
for (const [id, outId, fmt] of [["cCand", "cCandV", v => v + "개"], ["cCfg", "cCfgV", v => v], ["cTemp", "cTempV", v => v],
                                ["cGamma", "cGammaV", v => Number(v).toFixed(1)]]) {
  const el = $("#" + id), upd = () => $("#" + outId).textContent = fmt(el.value); el.addEventListener("input", upd); upd();
}

const total = () => cSubs.reduce((m, s) => Math.max(m, s.start + s.duration), 0);

function renderRows() {
  const tb = $("#cRows"); tb.innerHTML = "";
  cSubs.forEach((s, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><div class="idx" style="background:${SUB_COLORS[i % SUB_COLORS.length]}">${i + 1}</div></td>
      <td><input type="text" data-k="prompt"></td>
      <td><input type="number" data-k="start" step="0.05" min="0" max="10"></td>
      <td><input type="number" data-k="duration" step="0.05" min="0.15" max="10"></td>
      <td><input type="number" data-k="intensity" step="0.05" min="0" max="1"></td>
      <td><select data-k="transition"><option value="cut">cut (급격)</option><option value="crossfade">crossfade</option></select></td>
      <td style="white-space:nowrap"><button data-a="up" title="위로">↑</button> <button data-a="del" class="danger" title="삭제">✕</button></td>`;
    tr.querySelectorAll("[data-k]").forEach(inp => {
      inp.value = s[inp.dataset.k];
      inp.addEventListener(inp.tagName === "SELECT" ? "change" : "input", () => {
        const k = inp.dataset.k;
        s[k] = (k === "prompt" || k === "transition") ? inp.value : (parseFloat(inp.value) || 0);
        drawTimeline(); saveCompose();
      });
    });
    tr.querySelector('[data-a="del"]').onclick = () => { cSubs.splice(i, 1); renderRows(); saveCompose(); };
    tr.querySelector('[data-a="up"]').onclick = () => { if (i > 0) { [cSubs[i - 1], cSubs[i]] = [cSubs[i], cSubs[i - 1]]; autoTime(); } };
    tb.appendChild(tr);
  });
  drawTimeline();
}

function autoTime() {
  let t = 0; for (const s of cSubs) { s.start = Math.round(t * 100) / 100; t += s.duration; }
  renderRows(); saveCompose();
}

function drawTimeline() {
  const cv = $("#cTimeline"), dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (!w) return;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d"); g.scale(dpr, dpr);
  const css = getComputedStyle(document.documentElement), T = Math.max(1, Math.ceil(total()));
  g.fillStyle = css.getPropertyValue("--muted"); g.font = "10px system-ui"; g.strokeStyle = css.getPropertyValue("--line");
  for (let t = 0; t <= T; t++) { const x = t / T * w; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke(); if (t < T) g.fillText(t + "s", x + 3, h - 4); }
  const lane = (h - 16) / 2;
  cSubs.forEach((s, i) => {
    const x = s.start / T * w, bw = Math.max(3, s.duration / T * w), y = 3 + (i % 2) * lane, bh = lane - 3;
    g.fillStyle = SUB_COLORS[i % SUB_COLORS.length]; g.globalAlpha = .25 + .65 * s.intensity; g.fillRect(x, y, bw, bh);
    g.globalAlpha = 1; g.fillStyle = "#fff"; g.font = "600 11px system-ui"; if (bw > 14) g.fillText(String(i + 1), x + 4, y + bh / 2 + 4);
  });
  const tt = total();
  $("#cTotal").textContent = cSubs.length ? `${cSubs.length}개 sub-event · 전체 ${tt.toFixed(2)}초` + (tt > 10 ? " ⚠ 10초 초과" : "") : "분해하거나 직접 추가하세요";
}

$("#addSub").onclick = () => {
  const t = total();
  cSubs.push({ prompt: "", start: Math.round(t * 100) / 100, duration: 0.5, intensity: 0.7, transition: cSubs.length ? "crossfade" : "cut", source: "" });
  renderRows(); saveCompose();
  const inputs = document.querySelectorAll('#cRows input[data-k="prompt"]'); inputs[inputs.length - 1]?.focus();
};
$("#autoTime").onclick = autoTime;

$("#decBtn").onclick = async () => {
  const event = $("#cEvent").value.trim();
  if (!event) { $("#cEvent").focus(); return; }
  if (!llmOk) { toast("로컬 LLM을 찾을 수 없습니다 – LM Studio 서버를 켜세요"); return; }
  const btn = $("#decBtn"); btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> 분해 중…';
  try {
    const planner = $("#cPlanner").value;
    const optionalNumber = id => $(id).value === "" ? null : Number($(id).value);
    const bounce_parameters = planner === "ball_physics" ? {
      drop_height_m: optionalNumber("#bHeight"), restitution: optionalNumber("#bRestitution"),
      mass_kg: optionalNumber("#bMass"), impact_count: optionalNumber("#bCount"), pulse_duration_s: Number($("#bPulse").value) } : null;
    const r = await api("/api/decompose", { method: "POST", body: JSON.stringify({ event, planner, bounce_parameters }) });
    cPhysics = r.physics || null; showPhysics();
    cHierarchy = r.hierarchy || null; showHierarchy();
    if (cPhysics || cHierarchy) {
      $("#cGamma").value = "1"; $("#cGamma").dispatchEvent(new Event("input"));
      $("#cShare").checked = true;
    }
    cSubs = r.sub_events.map(s => ({ ...s })); renderRows(); saveCompose();
    $("#decInfo").textContent = `${r.model} · ${r.elapsed.toFixed(1)}s`;
    const ph = $("#cPhases"); ph.hidden = !r.phases.length;
    ph.innerHTML = "LLM이 파악한 진행 순서:<ol></ol>";
    r.phases.forEach(p => { const li = document.createElement("li"); li.textContent = p; ph.querySelector("ol").appendChild(li); });
    $("#cIssues").textContent = "";
  } catch (e) { toast("분해 실패: " + e.message, 4000); }
  finally { btn.disabled = false; btn.textContent = "로컬 LM으로 분해"; }
};

$("#compBtn").onclick = async () => {
  const event = $("#cEvent").value.trim();
  const subs = cSubs.filter(s => s.prompt.trim());
  if (!event || !subs.length) { toast(cHierarchy ? "현재 계획에는 재생할 진동이 없습니다. 무음 낙하나 빈 확률 실현일 수 있습니다." : "이벤트와 sub-event를 먼저 준비하세요"); return; }
  if (total() > 10) { toast("전체 길이가 10초를 넘습니다"); return; }
  if (cBusy) return; cBusy = true;
  const btn = $("#compBtn"), t0 = performance.now(); btn.disabled = true;
  const tick = setInterval(() => btn.innerHTML = `<span class="spinner"></span> 생성 & 합성 중… ${((performance.now() - t0) / 1000).toFixed(0)}s`, 250);
  try {
    const seed = $("#cSeed").value.trim();
    cPlan = await api("/api/compose", { method: "POST", body: JSON.stringify({
      event, sub_events: subs, candidates: +$("#cCand").value, baseline: $("#cBaseline").checked,
      cfg: +$("#cCfg").value, temperature: +$("#cTemp").value, seed: seed === "" ? null : +seed,
      gamma: +$("#cGamma").value, share_repeats: $("#cShare").checked, physics: cPhysics, hierarchy: cHierarchy }) });
    cPlan.composite_item.regions = cPlan.sub_events;
    newNames = new Set([cPlan.composite, cPlan.baseline].filter(Boolean));
    $("#cResult").hidden = false; redrawCompare();
    $("#cResInfo").textContent = `HapticGen ${cPlan.prompts}개 프롬프트` + (cPlan.analytic_segments ? ` · 해석적 합성 ${cPlan.analytic_segments}구간` : "") + ` · ${cPlan.elapsed.toFixed(1)}s`;
    loadLibrary();
    $("#cResult").scrollIntoView({ behavior: "smooth", block: "start" });
    if ($("#autoplay").checked && questReady) playQuest(cPlan.composite_item);
  } catch (e) { toast("합성 실패: " + e.message, 4000); }
  finally { clearInterval(tick); cBusy = false; btn.disabled = false; btn.textContent = "③ 생성 & 합성"; }
};

function redrawCompare() {
  if (!cPlan) return;
  const box = $("#cCompare"); box.innerHTML = "";
  const a = card(cPlan.composite_item); a.querySelector(".prompt").textContent = "A · Sub-event 합성";
  box.appendChild(a);
  if (cPlan.baseline_item) { const b = card(cPlan.baseline_item); b.querySelector(".prompt").textContent = "B · Baseline (이벤트 문장 한 번에)"; box.appendChild(b); }
  const segs = $("#cSegs"); segs.innerHTML = "";
  cPlan.sub_events.forEach((s, i) => {
    const b = document.createElement("button");
    b.innerHTML = `<span class="sw" style="background:${SUB_COLORS[i % SUB_COLORS.length]}"></span> ${i + 1}`;
    b.title = `${s.prompt}\n후보 ${s.candidate + 1}/${s.scores.length} 선택 (점수 ${s.scores.join(", ")})`;
    b.onclick = async () => { try { playQuest(await api("/api/wave/" + s.file.split("/").map(encodeURIComponent).join("/"))); } catch (e) { toast(e.message); } };
    segs.appendChild(b);
  });
}

$("#abBtn").onclick = () => {
  if (!cPlan) return;
  playQuest(cPlan.composite_item);
  if (cPlan.baseline_item) {
    const wait = (cPlan.composite_item.duration * +$("#loop").value + 1.0) * 1000;
    clearTimeout(window._abT); window._abT = setTimeout(() => playQuest(cPlan.baseline_item), wait);
  }
};

renderRows();
{
  let t = location.hash === "#compose" ? "compose" : location.hash === "#single" ? "single" : null;
  try { t = t || localStorage.getItem("hg_tab"); } catch {}
  setTab(t === "compose" ? "compose" : "single");
}

$("#genBtn").onclick = doGenerate;
$("#stopBtn").onclick = stopQuest;
$("#filter").oninput = render;
try { $("#autoplay").checked = localStorage.getItem("hg_autoplay") !== "0"; } catch {}
$("#autoplay").onchange = () => { try { localStorage.setItem("hg_autoplay", $("#autoplay").checked ? "1" : "0"); } catch {} };
try { $("#normalize").checked = localStorage.getItem("hg_normalize") !== "0"; } catch {}
$("#normalize").onchange = () => { try { localStorage.setItem("hg_normalize", $("#normalize").checked ? "1" : "0"); } catch {} };
document.addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); tab === "compose" ? $("#compBtn").click() : doGenerate(); }
  if (e.key === "Escape") stopQuest();
});
let rz; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => tab === "compose" ? (drawTimeline(), redrawCompare()) : render(), 150); });
loadLibrary();
