import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";

const $ = (s) => document.querySelector(s);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

/* ---------- 내 정보 ---------- */
const hEl = $("#height"), sEl = $("#speed"), smEl = $("#slowmo");
const prof0 = store.get("rc.profile", {});
if (prof0.height) hEl.value = prof0.height;
if (prof0.speed) sEl.value = prof0.speed;
function profile() {
  const height = parseFloat(hEl.value), speed = parseFloat(sEl.value);
  return {
    height: height >= 120 && height <= 220 ? height : null,
    speed: speed >= 3 && speed <= 25 ? speed : null,
    timeScale: parseFloat(smEl.value) || 1,
  };
}
[hEl, sEl].forEach((el) => el.addEventListener("change", () => {
  const p = profile(); store.set("rc.profile", { height: p.height, speed: p.speed });
}));

/* ---------- 탭 ---------- */
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", x === b));
  document.querySelectorAll(".tabpane").forEach((p) => (p.hidden = p.id !== "tab-" + b.dataset.tab));
  if (b.dataset.tab !== "rec") stopCamera();
  if (b.dataset.tab === "history") renderHistory();
}));

/* ---------- 자세 인식 모델 ---------- */
let lmPromise = null;
function getLandmarker() {
  if (!lmPromise) {
    lmPromise = (async () => {
      const fileset = await FilesetResolver.forVisionTasks(WASM);
      const opts = (delegate) => ({
        baseOptions: { modelAssetPath: MODEL, delegate },
        runningMode: "VIDEO", numPoses: 1,
        minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
      });
      try { return await PoseLandmarker.createFromOptions(fileset, opts("GPU")); }
      catch { return await PoseLandmarker.createFromOptions(fileset, opts("CPU")); }
    })();
    lmPromise.catch(() => { lmPromise = null; });
  }
  return lmPromise;
}
let lastTs = 0;
function detect(lmk, source) {
  const ts = Math.max(performance.now(), lastTs + 1);
  lastTs = ts;
  const r = lmk.detectForVideo(source, ts);
  const lm = r.landmarks && r.landmarks[0];
  return lm ? lm.map((p) => [p.x, p.y, p.visibility ?? 1]) : null;
}

/* ---------- 뼈대 그리기 ---------- */
const BONES = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[27,29],[29,31],[27,31],[24,26],[26,28],[28,30],[30,32],[28,32]];
const LEFT = new Set([11,13,15,23,25,27,29,31]);
const RIGHT = new Set([12,14,16,24,26,28,30,32]);
function drawPose(ctx, lm, map, lw, near) {
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  for (const [a, b] of BONES) {
    const side = LEFT.has(a) && LEFT.has(b) ? "L" : RIGHT.has(a) && RIGHT.has(b) ? "R" : "C";
    ctx.strokeStyle = side === "C" ? "rgba(255,255,255,.85)" : side === near ? "#FF5A36" : "#7FD3FF";
    ctx.lineWidth = side === near ? lw * 1.3 : lw;
    const [x1, y1] = map(lm[a]), [x2, y2] = map(lm[b]);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }
  ctx.fillStyle = "#fff";
  for (let i = 11; i < 33; i++) { const [x, y] = map(lm[i]); ctx.beginPath(); ctx.arc(x, y, lw * 0.9, 0, 7); ctx.fill(); }
  const [nx, ny] = map(lm[0]); ctx.beginPath(); ctx.arc(nx, ny, lw * 1.6, 0, 7); ctx.fill();
}
const nearSideOf = (lm) => (lm[11][2] + lm[23][2] + lm[25][2] + lm[27][2] >= lm[12][2] + lm[24][2] + lm[26][2] + lm[28][2] ? "L" : "R");

/* ---------- 실시간 촬영 ---------- */
const cam = $("#cam"), camOv = $("#camOverlay"), stage = $("#stage"), chipsEl = $("#chips"), hintEl = $("#hint");
const camBtn = $("#camBtn"), flipBtn = $("#flipBtn"), recBtn = $("#recBtn"), countEl = $("#count");
let stream = null, facing = "environment", liveOn = false, wakeLock = null, recState = "idle", cancelRec = false;
let tilt = null;

function onMotion(e) {
  const g = e.accelerationIncludingGravity;
  if (!g || g.x == null) return;
  const x = Math.abs(g.x), y = Math.abs(g.y), z = Math.abs(g.z);
  const roll = Math.min(Math.atan2(x, y), Math.atan2(y, x)) * 180 / Math.PI;
  const pitch = Math.atan2(z, Math.hypot(x, y)) * 180 / Math.PI;
  tilt = tilt ? { roll: tilt.roll * 0.8 + roll * 0.2, pitch: tilt.pitch * 0.8 + pitch * 0.2 } : { roll, pitch };
}

async function startCamera() {
  // iOS는 클릭 직후에 권한을 물어야 함
  if (typeof DeviceMotionEvent !== "undefined" && typeof DeviceMotionEvent.requestPermission === "function") {
    try { await DeviceMotionEvent.requestPermission(); } catch {}
  }
  window.addEventListener("devicemotion", onMotion);
  ensureAudio();
  hideError();
  stopCamera();
  camBtn.disabled = true; hint("카메라를 켜는 중…");
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60 } },
    });
  } catch (e) {
    camBtn.disabled = false;
    showError(e && e.name === "NotAllowedError"
      ? "카메라 사용이 허용되지 않았어요. 브라우저 주소창의 권한 설정에서 카메라를 허용한 뒤 다시 눌러 주세요."
      : "카메라를 켜지 못했어요. 다른 앱이 카메라를 쓰고 있는지 확인해 주세요. 카카오톡 안에서 열었다면 Safari나 Chrome으로 열어 주세요.");
    hint("");
    return;
  }
  cam.srcObject = stream;
  await cam.play().catch(() => {});
  $("#camEmpty").hidden = true; chipsEl.hidden = false;
  stage.classList.toggle("mirror", facing === "user");
  camBtn.textContent = "카메라 끄기"; camBtn.disabled = false;
  flipBtn.disabled = false;
  try { wakeLock = await navigator.wakeLock?.request("screen"); } catch {}
  hint("자세 인식 모델을 불러오는 중…");
  let lmk;
  try { lmk = await getLandmarker(); } catch { showError("자세 인식 모델을 불러오지 못했어요. 인터넷 연결을 확인하고 다시 시도해 주세요."); return; }
  if (!stream) return;
  recBtn.disabled = false;
  liveOn = true;
  let last = 0;
  const loop = () => {
    if (!liveOn) return;
    const now = performance.now();
    if (cam.readyState >= 2 && now - last > 60) {
      last = now;
      if (camOv.width !== cam.videoWidth) { camOv.width = cam.videoWidth; camOv.height = cam.videoHeight; }
      const lm = detect(lmk, cam);
      const ctx = camOv.getContext("2d");
      ctx.clearRect(0, 0, camOv.width, camOv.height);
      if (lm) drawPose(ctx, lm, (p) => [p[0] * camOv.width, p[1] * camOv.height], Math.max(3, camOv.height / 160), nearSideOf(lm));
      liveChecks(lm, camOv.width, camOv.height);
    }
    requestAnimationFrame(loop);
  };
  loop();
}

function stopCamera() {
  liveOn = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null; cam.srcObject = null;
  window.removeEventListener("devicemotion", onMotion);
  try { wakeLock?.release(); } catch {} wakeLock = null;
  const ctx = camOv.getContext("2d"); ctx.clearRect(0, 0, camOv.width, camOv.height);
  $("#camEmpty").hidden = false; chipsEl.hidden = true; stage.dataset.ready = "";
  camBtn.textContent = "카메라 켜기"; flipBtn.disabled = true; recBtn.disabled = true;
  if (recState === "idle") hint("");
}

function hint(text, s = "") { hintEl.textContent = text; hintEl.dataset.s = s; }

function liveChecks(lm, W, H) {
  const chips = [];
  let first = null;
  const add = (s, text, msg) => { chips.push([s, text]); if (s !== "good" && msg && !first) first = [s, msg]; };
  if (!lm) {
    add("bad", "사람 없음", "화면에 사람이 보이지 않아요. 러닝머신 위에 서 주세요.");
  } else {
    const inside = (p) => p[0] > 0.02 && p[0] < 0.98 && p[1] > 0.02 && p[1] < 0.98;
    const full = [0, 27, 28, 29, 30, 31, 32].every((i) => inside(lm[i])) && lm[0][2] > 0.4 && Math.max(lm[27][2], lm[28][2]) > 0.4;
    add(full ? "good" : "bad", full ? "전신 보임" : "머리·발 잘림", "머리부터 발끝까지 화면에 들어오게 해 주세요.");

    const top = lm[0][1], bottom = Math.max(lm[29][1], lm[30][1], lm[31][1], lm[32][1]);
    const size = bottom - top;
    if (size > 0.92) add("bad", "너무 가까움", "카메라를 조금 뒤로 옮겨 주세요.");
    else if (size >= 0.45) add("good", "크기 적당");
    else if (size >= 0.3) add("warn", "조금 멀어요", "카메라를 조금 더 가까이 옮기면 정확해져요.");
    else add("bad", "너무 멀어요", "카메라를 더 가까이 옮겨 주세요.");

    const P = (i) => [lm[i][0] * W, lm[i][1] * H];
    const sh = [(P(11)[0] + P(12)[0]) / 2, (P(11)[1] + P(12)[1]) / 2], hp = [(P(23)[0] + P(24)[0]) / 2, (P(23)[1] + P(24)[1]) / 2];
    const torso = Math.hypot(sh[0] - hp[0], sh[1] - hp[1]) || 1;
    const sep = Math.max(Math.abs(P(23)[0] - P(24)[0]), Math.abs(P(11)[0] - P(12)[0])) / torso;
    if (sep < 0.22) add("good", "정측면");
    else if (sep < 0.4) add("warn", "약간 비스듬", "카메라를 몸과 직각이 되게 조금 돌려 주세요.");
    else add("bad", "옆모습 아님", "카메라를 러너의 정확히 옆쪽에 두세요.");
  }
  if (tilt) {
    const r = Math.round(tilt.roll), p = Math.round(tilt.pitch);
    if (r <= 3 && p <= 10) add("good", "수평 맞음");
    else if (r <= 7 && p <= 18) add("warn", `기울어짐 ${Math.max(r, p)}°`, "휴대폰을 똑바로 세워 주세요.");
    else add("bad", `기울어짐 ${Math.max(r, p)}°`, "휴대폰이 많이 기울었어요. 똑바로 세워 주세요.");
  } else chips.push(["", "기울기 센서 없음"]);

  chipsEl.innerHTML = chips.map(([s, t]) => `<span class="chip" data-s="${s}">${t}</span>`).join("");
  const worst = chips.some((c) => c[0] === "bad") ? "bad" : chips.some((c) => c[0] === "warn") ? "warn" : "good";
  stage.dataset.ready = worst === "bad" ? "" : worst;
  if (recState === "idle") {
    if (first) hint(first[1], first[0]);
    else hint("촬영 준비 완료. 녹화 시작을 누르세요.", "good");
  }
}

/* 소리 */
let audioCtx = null;
function ensureAudio() { try { audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)(); audioCtx.resume(); } catch {} }
function beep(freq, dur) {
  if (!audioCtx) return;
  const o = audioCtx.createOscillator(), g = audioCtx.createGain();
  o.frequency.value = freq; o.connect(g); g.connect(audioCtx.destination);
  const t = audioCtx.currentTime;
  g.gain.setValueAtTime(0.25, t); g.gain.exponentialRampToValueAtTime(0.001, t + dur);
  o.start(t); o.stop(t + dur);
}

function pickMime() {
  if (typeof MediaRecorder === "undefined") return null;
  for (const m of ["video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"]) {
    if (MediaRecorder.isTypeSupported?.(m)) return m;
  }
  return "";
}

async function record() {
  if (recState !== "idle") { cancelRec = true; return; }
  if (!stream) return;
  if (typeof MediaRecorder === "undefined") { showError("이 브라우저는 녹화를 지원하지 않아요. ‘영상 올리기’로 찍은 영상을 올려 주세요."); return; }
  ensureAudio();
  cancelRec = false; recState = "count";
  recBtn.textContent = "취소"; flipBtn.disabled = true; camBtn.disabled = true;
  const count = +$("#countSel").value, dur = +$("#durSel").value;
  countEl.hidden = false; countEl.className = "count num";
  for (let i = count; i > 0 && !cancelRec; i--) {
    countEl.textContent = i; hint(`${i}초 뒤 녹화가 시작돼요. 달리기 시작하세요.`, "");
    beep(i <= 3 ? 880 : 660, 0.09); await sleep(1000);
  }
  if (cancelRec || !stream) return endRec();
  beep(1100, 0.3);
  const mime = pickMime();
  const chunks = [];
  let rec;
  try { rec = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 6_000_000 } : undefined); }
  catch { rec = new MediaRecorder(stream); }
  rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
  const stopped = new Promise((r) => (rec.onstop = r));
  rec.start(500);
  recState = "rec"; countEl.className = "count rec num";
  for (let s = dur; s > 0 && !cancelRec; s--) {
    countEl.innerHTML = `<span>● 녹화 중 ${s}</span>`; hint("평소처럼 달려 주세요.", "");
    await sleep(1000);
  }
  rec.stop(); await stopped;
  beep(700, 0.15); setTimeout(() => beep(700, 0.15), 200);
  const cancelled = cancelRec;
  endRec();
  if (cancelled) return;
  const blob = new Blob(chunks, { type: rec.mimeType || mime || "video/webm" });
  stopCamera();
  runAnalysis(blob);
}
function endRec() {
  recState = "idle"; countEl.hidden = true;
  recBtn.textContent = "녹화 시작"; camBtn.disabled = false; flipBtn.disabled = !stream;
  hint("");
}

camBtn.addEventListener("click", () => (stream ? stopCamera() : startCamera()));
flipBtn.addEventListener("click", () => { facing = facing === "environment" ? "user" : "environment"; startCamera(); });
recBtn.addEventListener("click", record);
document.addEventListener("visibilitychange", async () => {
  if (document.visibilityState === "visible" && stream && !wakeLock) { try { wakeLock = await navigator.wakeLock?.request("screen"); } catch {} }
});

/* ---------- 업로드 ---------- */
const fileEl = $("#file"), drop = $("#drop");
fileEl.addEventListener("change", () => { const f = fileEl.files[0]; if (f) runAnalysis(f); fileEl.value = ""; });
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault(); drop.classList.remove("over");
  const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith("video/"));
  if (f) runAnalysis(f); else showError("동영상 파일만 올릴 수 있어요.");
});

/* ---------- 진행 표시·오류 ---------- */
const progEl = $("#progress"), progBar = $("#progBar"), progLabel = $("#progLabel"), progNote = $("#progNote"), errEl = $("#err");
function progress(label, frac, note) {
  progEl.hidden = false; progLabel.textContent = label;
  progBar.style.width = Math.round(Math.max(0, Math.min(1, frac)) * 100) + "%";
  if (note != null) progNote.textContent = note;
}
function showError(msg) { errEl.textContent = msg; errEl.hidden = false; progEl.hidden = true; }
function hideError() { errEl.hidden = true; }

/* ---------- 영상 분석 ---------- */
const anaV = $("#anaVideo");
let currentUrl = null, busy = false;

function once(el, ev, ms = 8000) {
  return new Promise((r) => { const h = () => { el.removeEventListener(ev, h); r(true); }; el.addEventListener(ev, h); setTimeout(() => { el.removeEventListener(ev, h); r(false); }, ms); });
}
function seekTo(v, t) { const p = once(v, "seeked", 3000); v.currentTime = t; return p; }
async function fixDuration(v) {
  if (Number.isFinite(v.duration)) return;
  // MediaRecorder webm은 길이 정보가 없어서 끝으로 한 번 이동해 길이를 얻는다
  const p = once(v, "durationchange", 4000);
  v.currentTime = 1e101;
  await p;
  await seekTo(v, 0);
}

async function runAnalysis(blob) {
  if (busy) return;
  busy = true; hideError();
  const prof = profile();
  progress("자세 인식 모델 준비 중", 0.02, "처음 한 번은 모델(약 9MB)을 내려받느라 조금 걸려요.");
  progEl.scrollIntoView({ behavior: "smooth", block: "center" });
  try {
    const lmk = await getLandmarker();
    if (currentUrl) URL.revokeObjectURL(currentUrl);
    currentUrl = URL.createObjectURL(blob);
    anaV.src = currentUrl;
    if (!(await once(anaV, "loadedmetadata", 15000))) throw new Error("이 영상 형식은 이 브라우저에서 열 수 없어요. mp4로 저장된 영상을 올려 주세요.");
    await fixDuration(anaV);
    const W = anaV.videoWidth, H = anaV.videoHeight;
    const ts = prof.timeScale;
    const maxVT = Math.min(Number.isFinite(anaV.duration) ? anaV.duration : 60, 20 / ts, 45);
    const frames = [];
    progress("영상에서 자세 찾는 중", 0.05, "영상을 천천히 재생하며 한 프레임씩 분석합니다. 화면을 켜 두세요.");

    anaV.playbackRate = ts < 1 ? 1 : 0.5;
    await new Promise((resolve, reject) => {
      let lastT = -1, lastProgress = performance.now(), done = false;
      const finish = () => { if (done) return; done = true; anaV.pause(); resolve(); };
      const handle = (t) => {
        if (t !== lastT) {
          lastT = t; lastProgress = performance.now();
          let lm = null;
          try { lm = detect(lmk, anaV); } catch {}
          frames.push({ vt: t, t: t * ts, lm });
          progress("영상에서 자세 찾는 중", 0.05 + 0.85 * (t / maxVT));
        }
        if (t >= maxVT) finish();
      };
      const rvfc = "requestVideoFrameCallback" in anaV;
      const tick = (now, meta) => {
        if (done) return;
        handle(meta ? meta.mediaTime : anaV.currentTime);
        if (rvfc) anaV.requestVideoFrameCallback(tick); else requestAnimationFrame(() => tick());
      };
      anaV.onended = finish;
      const guard = setInterval(() => {
        if (done) return clearInterval(guard);
        if (performance.now() - lastProgress > 6000) { clearInterval(guard); frames.length > 20 ? finish() : (done = true, reject(new Error("영상을 재생하지 못했어요. 다른 영상으로 다시 시도해 주세요."))); }
      }, 1000);
      anaV.play().then(() => (rvfc ? anaV.requestVideoFrameCallback(tick) : tick())).catch(() => { done = true; clearInterval(guard); reject(new Error("영상을 재생하지 못했어요. 화면을 한 번 누른 뒤 다시 시도해 주세요.")); });
    });

    const posed = frames.filter((f) => f.lm);
    if (posed.length < 15) throw new Error("영상에서 사람을 충분히 찾지 못했어요. 몸 전체가 보이는 옆모습 영상인지 확인해 주세요.");
    progress("결과 계산 중", 0.92);
    const m = analyze(frames, W, H, prof);
    progress("착지 장면 캡처 중", 0.95);
    const thumbs = await captureThumbs(anaV, m, W, H);
    progEl.hidden = true;
    const rec = toRecord(m);
    const prev = store.get("rc.history", [])[0] || null;
    saveHistory(rec);
    renderResults(m, { thumbs, url: currentUrl, frames, W, H, prev });
  } catch (e) {
    showError(e.message || "분석 중 문제가 생겼어요. 다시 시도해 주세요.");
  } finally {
    busy = false;
  }
}

/* --- 수학 도우미 --- */
const finite = (a) => a.filter(Number.isFinite);
const mean = (a) => { const b = finite(a); return b.length ? b.reduce((s, x) => s + x, 0) / b.length : NaN; };
const median = (a) => { const b = finite(a).sort((x, y) => x - y); if (!b.length) return NaN; const k = b.length >> 1; return b.length % 2 ? b[k] : (b[k - 1] + b[k]) / 2; };
const pct = (a, p) => { const b = finite(a).sort((x, y) => x - y); return b.length ? b[Math.min(b.length - 1, Math.max(0, Math.round(p * (b.length - 1))))] : NaN; };
const deg = (r) => (r * 180) / Math.PI;
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const angleAt = (a, b, c) => { const v1x = a.x - b.x, v1y = a.y - b.y, v2x = c.x - b.x, v2y = c.y - b.y; const d = Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y); return d ? deg(Math.acos(Math.max(-1, Math.min(1, (v1x * v2x + v1y * v2y) / d)))) : NaN; };
const smooth = (a) => a.map((_, i) => mean([a[i - 1], a[i], a[i + 1]].filter((x) => x !== undefined)));

const SIDE = { L: { sh: 11, el: 13, wr: 15, hip: 23, kn: 25, an: 27, he: 29, to: 31 }, R: { sh: 12, el: 14, wr: 16, hip: 24, kn: 26, an: 28, he: 30, to: 32 } };
// 신체 분절 비율(Winter): 어깨~엉덩이 0.288, 엉덩이~무릎 0.245, 무릎~발목 0.246 → 합 0.779 × 키
const SEG_RATIO = 0.779;

function analyze(frames, W, H, prof) {
  const F = frames.filter((f) => f.lm);
  const P = (f, i) => ({ x: f.lm[i][0] * W, y: f.lm[i][1] * H, v: f.lm[i][2] });
  const ts = F.map((f) => f.t);
  const span = ts[ts.length - 1] - ts[0];

  const vis = (S) => mean(F.map((f) => mean([S.sh, S.hip, S.kn, S.an].map((i) => f.lm[i][2]))));
  const nearKey = vis(SIDE.L) >= vis(SIDE.R) ? "L" : "R";
  const N = SIDE[nearKey];

  // 달리는 방향: 뒤꿈치 → 발끝
  let vote = 0;
  for (const f of F) for (const S of [SIDE.L, SIDE.R]) vote += Math.sign(P(f, S.to).x - P(f, S.he).x);
  const dir = vote >= 0 ? 1 : -1;

  const segPx = median(F.map((f) => dist(P(f, N.sh), P(f, N.hip)) + dist(P(f, N.hip), P(f, N.kn)) + dist(P(f, N.kn), P(f, N.an))));
  const legPx = median(F.map((f) => dist(P(f, N.hip), P(f, N.kn)) + dist(P(f, N.kn), P(f, N.an))));
  const torsoPx = median(F.map((f) => dist(mid(P(f, 11), P(f, 12)), mid(P(f, 23), P(f, 24)))));
  const cmPerPx = prof.height ? (prof.height * SEG_RATIO) / segPx : null;
  const sideRatio = median(F.map((f) => Math.max(Math.abs(P(f, 23).x - P(f, 24).x), Math.abs(P(f, 11).x - P(f, 12).x)))) / torsoPx;
  const coverage = F.length / frames.length;

  // 착지 찾기: 발 높이가 바닥선 근처로 내려오는 순간
  const footY = (S) => smooth(F.map((f) => Math.max(P(f, S.he).y, P(f, S.to).y)));
  const fy = { L: footY(SIDE.L), R: footY(SIDE.R) };
  const ground = pct([...fy.L, ...fy.R], 0.95);
  const tol = 0.06 * legPx;
  let contacts = [];
  for (const key of ["L", "R"]) {
    const y = fy[key];
    let inC = false, start = 0, lastIC = -1e9;
    for (let i = 0; i <= y.length; i++) {
      const c = i < y.length && y[i] > ground - tol;
      if (c && !inC) { inC = true; start = i; }
      else if (!c && inC) {
        inC = false;
        if (i - start >= 2 && ts[start] - lastIC > 0.4) { contacts.push({ i: start, side: key }); lastIC = ts[start]; }
      }
    }
  }
  contacts.sort((a, b) => a.i - b.i);
  contacts = contacts.filter((c, k, arr) => k === 0 || ts[c.i] - ts[arr[k - 1].i] > 0.15);

  // 케이던스: 엉덩이 상하 움직임의 주기(자기상관) 우선, 착지 수로 보조
  const hipY = smooth(F.map((f) => mid(P(f, 23), P(f, 24)).y));
  let cadence = NaN, cadenceSrc = "";
  const ac = autoCadence(ts, hipY);
  const cc = contacts.length >= 4 ? ((contacts.length - 1) / (ts[contacts[contacts.length - 1].i] - ts[contacts[0].i])) * 60 : NaN;
  if (ac) { cadence = ac; cadenceSrc = "auto"; }
  else if (Number.isFinite(cc)) { cadence = cc; cadenceSrc = "contacts"; }

  const per = contacts.map((c) => {
    const f = F[c.i], S = SIDE[c.side];
    const hipM = mid(P(f, 23), P(f, 24));
    const an = P(f, S.an), kn = P(f, S.kn), hp = P(f, S.hip), he = P(f, S.he), to = P(f, S.to);
    return {
      t: f.t, vt: f.vt, side: c.side, lm: f.lm,
      overPx: (an.x - hipM.x) * dir,
      shank: deg(Math.atan2((an.x - kn.x) * dir, an.y - kn.y)),
      knee: 180 - angleAt(hp, kn, an),
      foot: deg(Math.atan2(he.y - to.y, (to.x - he.x) * dir)),
    };
  });
  const nearPer = per.filter((p) => p.side === nearKey);
  const use = nearPer.length >= 3 ? nearPer : per;

  const trunk = median(F.map((f) => {
    const s = mid(P(f, 11), P(f, 12)), h = mid(P(f, 23), P(f, 24));
    return deg(Math.atan2((s.x - h.x) * dir, h.y - s.y));
  }));
  const elbow = median(F.map((f) => angleAt(P(f, N.sh), P(f, N.el), P(f, N.wr))));

  // 상하 움직임: 착지와 착지 사이 엉덩이 높이 변화
  let voPx = NaN;
  if (contacts.length >= 3) {
    const r = [];
    for (let k = 1; k < contacts.length; k++) {
      const seg = hipY.slice(contacts[k - 1].i, contacts[k].i + 1);
      if (seg.length >= 3) r.push(Math.max(...seg) - Math.min(...seg));
    }
    voPx = median(r);
  } else {
    voPx = pct(hipY, 0.95) - pct(hipY, 0.05);
  }

  const m = {
    nearKey, dir, W, H, span, coverage, sideRatio,
    frames: frames.length, posed: F.length,
    contacts: per, contactCount: contacts.length,
    cadence, cadenceSrc,
    shank: median(use.map((p) => p.shank)),
    knee: median(use.map((p) => p.knee)),
    foot: median(use.map((p) => p.foot)),
    overPct: (median(use.map((p) => p.overPx)) / legPx) * 100,
    overCm: cmPerPx ? median(use.map((p) => p.overPx)) * cmPerPx : null,
    trunk, elbow,
    voCm: cmPerPx ? voPx * cmPerPx : null,
    voPctLeg: (voPx / legPx) * 100,
    height: prof.height, speed: prof.speed,
  };
  if (prof.speed && Number.isFinite(cadence)) {
    m.stepM = prof.speed / 3.6 / (cadence / 60);
    if (m.voCm != null) m.voRatio = (m.voCm / (m.stepM * 100)) * 100;
  }
  return m;
}

function autoCadence(ts, y) {
  const t0 = ts[0], t1 = ts[ts.length - 1];
  if (t1 - t0 < 2.5) return null;
  const dt = 1 / 60, n = Math.floor((t1 - t0) / dt);
  const s = new Array(n);
  let j = 0;
  for (let k = 0; k < n; k++) {
    const t = t0 + k * dt;
    while (j < ts.length - 2 && ts[j + 1] < t) j++;
    const a = ts[j], b = ts[j + 1], u = b > a ? Math.min(1, Math.max(0, (t - a) / (b - a))) : 0;
    s[k] = y[j] + (y[j + 1] - y[j]) * u;
  }
  const w = 30;
  const d = s.map((_, k) => { let sum = 0, c = 0; for (let q = Math.max(0, k - w); q <= Math.min(n - 1, k + w); q++) { sum += s[q]; c++; } return s[k] - sum / c; });
  const lo = Math.round(0.24 / dt), hi = Math.round(0.62 / dt);
  const r = [];
  for (let lag = lo - 1; lag <= hi + 1; lag++) {
    let a = 0, b = 0, c = 0;
    for (let k = 0; k + lag < n; k++) { a += d[k] * d[k + lag]; b += d[k] * d[k]; c += d[k + lag] * d[k + lag]; }
    r.push({ lag, r: a / (Math.sqrt(b * c) || 1) });
  }
  const peaks = r.filter((p, k) => k > 0 && k < r.length - 1 && p.r >= r[k - 1].r && p.r >= r[k + 1].r && p.lag >= lo && p.lag <= hi);
  if (!peaks.length) return null;
  const best = Math.max(...peaks.map((p) => p.r));
  if (best < 0.3) return null;
  const pick = peaks.find((p) => p.r >= best * 0.85);
  return 60 / (pick.lag * dt);
}

/* ---------- 착지 장면 캡처 ---------- */
async function captureThumbs(v, m, W, H) {
  const list = m.contacts.filter((c) => c.side === m.nearKey);
  const src = list.length >= 2 ? list : m.contacts;
  if (!src.length) return [];
  const pickN = Math.min(4, src.length);
  const chosen = Array.from({ length: pickN }, (_, k) => src[Math.round((k * (src.length - 1)) / Math.max(1, pickN - 1))]);
  const out = [];
  for (const c of chosen) {
    await seekTo(v, c.vt);
    await sleep(60);
    out.push({ c, canvas: drawThumb(v, c, m, W, H) });
  }
  return out;
}
function drawThumb(v, c, m, W, H) {
  const lm = c.lm;
  const cv = document.createElement("canvas");
  if (!lm) return cv;
  const xs = lm.slice(0, 33).map((p) => p[0] * W), ys = lm.slice(0, 33).map((p) => p[1] * H);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const bh = y1 - y0, pad = bh * 0.14;
  y0 -= pad; y1 += pad;
  const cx = (x0 + x1) / 2, bw = Math.max(x1 - x0 + pad * 2, (y1 - y0) * 0.62);
  x0 = cx - bw / 2; x1 = cx + bw / 2;
  const outW = 300, sc = outW / (x1 - x0), outH = Math.round((y1 - y0) * sc);
  cv.width = outW; cv.height = outH;
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#05080C"; ctx.fillRect(0, 0, outW, outH);
  ctx.drawImage(v, x0, y0, x1 - x0, y1 - y0, 0, 0, outW, outH);
  const map = (p) => [(p[0] * W - x0) * sc, (p[1] * H - y0) * sc];
  drawPose(ctx, lm, map, 3, c.side);
  const S = SIDE[c.side];
  const hip = [(lm[23][0] + lm[24][0]) / 2, (lm[23][1] + lm[24][1]) / 2];
  const [hx, hy] = map(hip);
  ctx.setLineDash([6, 5]); ctx.strokeStyle = "#FFFFFF"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(hx, outH); ctx.stroke(); ctx.setLineDash([]);
  const [ax, ay] = map(lm[S.an]), [kx, ky] = map(lm[S.kn]);
  ctx.strokeStyle = "#FFD23F"; ctx.lineWidth = 5;
  ctx.beginPath(); ctx.moveTo(kx, ky); ctx.lineTo(ax, ay); ctx.stroke();
  ctx.fillStyle = "#FFD23F"; ctx.beginPath(); ctx.arc(ax, ay, 6, 0, 7); ctx.fill();
  return cv;
}

/* ---------- 평가 기준·코칭 문구 ---------- */
function grade(m) {
  const items = [];
  const fmt = (x, d = 0) => (Number.isFinite(x) ? x.toFixed(d) : "–");

  if (Number.isFinite(m.cadence)) {
    const v = m.cadence, s = v >= 165 ? "good" : v >= 155 ? "warn" : "bad";
    const target = Math.round((v * 1.05) / 2) * 2;
    items.push({
      key: "cadence", label: "케이던스", value: fmt(v), unit: "보/분", s, scale: [140, 200], band: [165, 200],
      sub: m.stepM ? `한 걸음 약 ${Math.round(m.stepM * 100)}cm (러닝머신 속도로 계산)` : "1분 동안 땅을 딛는 횟수",
      tip: s === "good" ? "리듬이 좋습니다. 지금 속도에서 이 정도면 충분해요."
        : `보폭을 조금 줄이고 발을 더 자주 디뎌 보세요. 한 번에 5%씩만 올리는 게 안전합니다. 다음 목표는 분당 ${target}보예요.`,
      drill: `메트로놈 앱을 ${target}bpm에 맞추고 박자에 발을 맞춰 1분 달리기, 1분 걷기를 5번 반복하세요.`,
      title: "발을 더 자주 디디기",
    });
  }
  if (Number.isFinite(m.shank)) {
    const v = m.shank, s = v <= 7 ? "good" : v <= 12 ? "warn" : "bad";
    const over = m.overCm != null ? `발목이 엉덩이보다 ${Math.round(m.overCm)}cm 앞 (다리 길이의 ${Math.round(m.overPct)}%)` : `발목이 엉덩이보다 다리 길이의 ${Math.round(m.overPct)}%만큼 앞`;
    items.push({
      key: "shank", label: "착지 위치 (정강이 각도)", value: fmt(v, 1), unit: "°", s, scale: [-5, 20], band: [-5, 7],
      sub: `착지 순간 ${over}`,
      tip: s === "good" ? "발이 몸 거의 바로 아래에 떨어집니다. 무릎과 정강이에 충격이 덜 가는 착지예요."
        : "발이 몸보다 앞에 떨어져서 걸음마다 브레이크가 걸립니다. 발을 앞으로 뻗지 말고 엉덩이 아래로 내려놓는다는 느낌으로 달려 보세요. 케이던스를 올리면 대부분 함께 좋아집니다.",
      drill: "제자리에서 빠르게 발 구르기 20초 × 3세트, 짧은 오르막 달리기 6 × 20초를 해 보세요. 둘 다 발이 몸 아래에 떨어지는 감각을 익히는 데 좋습니다.",
      title: "발을 몸 아래에 내려놓기",
    });
  }
  if (Number.isFinite(m.knee)) {
    const v = m.knee;
    const s = v >= 15 && v <= 32 ? "good" : v >= 10 && v <= 38 ? "warn" : "bad";
    const stiff = v < 15;
    items.push({
      key: "knee", label: "착지 때 무릎 굽힘", value: fmt(v), unit: "°", s, scale: [0, 45], band: [15, 32],
      sub: "다리를 완전히 폈을 때가 0°",
      tip: s === "good" ? "무릎이 적당히 굽은 채로 착지해 충격을 잘 받아 줍니다."
        : stiff ? "무릎을 거의 편 채로 착지해서 충격이 관절로 바로 전달됩니다. 착지할 때 무릎을 살짝 굽혀 스프링처럼 받는다는 느낌을 가져 보세요."
        : "착지 때 무릎이 많이 굽어 앉은 자세로 달리고 있어요. 허벅지 앞쪽에 부담이 커집니다. 엉덩이를 조금 더 높게 유지해 보세요.",
      drill: stiff ? "A-스킵 20m × 4, 가볍게 제자리 줄넘기 30초 × 3을 해 보세요." : "글루트 브리지 15회 × 3, 한 발 스쿼트 8회 × 3으로 엉덩이 근육을 키워 주세요.",
      title: stiff ? "무릎을 살짝 굽혀 착지하기" : "엉덩이를 높게 유지하기",
    });
  }
  if (Number.isFinite(m.trunk)) {
    const v = m.trunk;
    const s = v >= 3 && v <= 12 ? "good" : v >= 0 && v <= 18 ? "warn" : "bad";
    const back = v < 3;
    items.push({
      key: "trunk", label: "상체 기울기", value: fmt(v, 1), unit: "°", s, scale: [-5, 25], band: [3, 12],
      sub: "앞으로 기울면 +, 뒤로 젖히면 −",
      tip: s === "good" ? "상체가 살짝 앞으로 기울어 자연스럽게 앞으로 나아가는 자세예요."
        : back ? "상체가 곧게 서 있거나 뒤로 젖혀져 있어요. 허리를 꺾지 말고 발목에서부터 몸 전체를 살짝 앞으로 기울여 보세요."
        : "상체가 많이 숙여져 있어요. 허리에서 접히지 말고 가슴을 펴고, 시선을 20~30m 앞에 두세요.",
      drill: back ? "벽에서 한 걸음 떨어져 몸을 일자로 유지한 채 앞으로 기울여 벽을 짚는 연습을 10회 해 보세요." : "플랭크 40초 × 3, 데드버그 10회 × 3으로 코어를 강화하세요.",
      title: back ? "몸 전체를 살짝 앞으로 기울이기" : "가슴 펴고 시선은 멀리",
    });
  }
  {
    let v, unit, s, scale, band, sub;
    if (m.voRatio != null) { v = m.voRatio; unit = "%"; scale = [3, 14]; band = [3, 8]; s = v <= 8 ? "good" : v <= 10 ? "warn" : "bad"; sub = `위아래 ${m.voCm.toFixed(1)}cm, 보폭 대비 비율`; }
    else if (m.voCm != null) { v = m.voCm; unit = "cm"; scale = [3, 14]; band = [3, 9]; s = v <= 9 ? "good" : v <= 11 ? "warn" : "bad"; sub = "걸음마다 엉덩이가 오르내리는 높이"; }
    else { v = m.voPctLeg; unit = "%"; scale = [4, 18]; band = [4, 11]; s = v <= 11 ? "good" : v <= 13 ? "warn" : "bad"; sub = "다리 길이 대비. 키를 입력하면 cm로 보여 드려요"; }
    if (Number.isFinite(v)) items.push({
      key: "vo", label: "상하 움직임", value: v.toFixed(1), unit, s, scale, band, sub,
      tip: s === "good" ? "위아래로 덜 튀어서 힘을 앞으로 가는 데 잘 쓰고 있어요."
        : "몸이 위아래로 많이 튑니다. 위로 뛰어오르는 힘은 앞으로 가는 데 쓰이지 않아 에너지가 낭비돼요. 발을 땅에서 빨리 떼고 앞으로 미끄러지듯 달려 보세요.",
      drill: "머리 위에 물컵을 올렸다고 상상하고 달려 보세요. 케이던스를 올리는 것도 상하 움직임을 줄이는 데 효과적입니다.",
      title: "위아래로 덜 튀기",
    });
  }
  if (Number.isFinite(m.elbow)) {
    const v = m.elbow, s = v >= 65 && v <= 110 ? "good" : "warn";
    items.push({
      key: "elbow", label: "팔꿈치 각도", value: fmt(v), unit: "°", s, scale: [30, 160], band: [65, 110],
      sub: "카메라 쪽 팔 기준",
      tip: s === "good" ? "팔 각도가 적당합니다. 어깨 힘을 빼고 앞뒤로 흔들면 됩니다."
        : v < 65 ? "팔을 너무 많이 굽히고 있어요. 어깨 힘을 빼고 90도 정도로 편하게 흔들어 보세요."
        : "팔이 많이 펴져 있어요. 90도 정도로 굽혀 앞뒤로 흔들면 리듬을 잡기 쉬워집니다.",
      drill: "제자리에 서서 팔만 90도로 흔들기 30초 × 3. 손이 몸 앞을 가로지르지 않게 하세요.",
      title: "팔을 90도로 편하게",
    });
  }
  if (Number.isFinite(m.foot)) {
    const v = m.foot;
    const type = v > 8 ? "뒤꿈치 착지" : v < -5 ? "앞꿈치 착지" : "발 중간 착지";
    items.push({
      key: "foot", label: "착지 부위", value: type, unit: "", s: "info", text: true,
      sub: `착지 순간 발 각도 ${v.toFixed(0)}° (발끝이 들리면 +)`,
      tip: "뒤꿈치 착지 자체가 나쁜 건 아니에요. 발이 몸보다 얼마나 앞에 떨어지는지(정강이 각도)가 더 중요합니다.",
    });
  }
  return items;
}
const PRIORITY = ["shank", "cadence", "knee", "trunk", "vo", "elbow"];

function score(items) {
  const g = items.filter((i) => i.s !== "info");
  if (!g.length) return 0;
  const pts = g.reduce((s, i) => s + (i.s === "good" ? 1 : i.s === "warn" ? 0.55 : 0.15), 0);
  return Math.round((pts / g.length) * 100);
}

/* ---------- 결과 화면 ---------- */
const resultsEl = $("#results");
const STATUS = { good: "좋음", warn: "주의", bad: "개선 필요", info: "참고" };
let replayStop = null;

function renderResults(m, opt = {}) {
  if (replayStop) { replayStop(); replayStop = null; }
  const items = grade(m);
  const sc = score(items);
  const fixes = items.filter((i) => i.s === "bad" || i.s === "warn").sort((a, b) => (a.s === b.s ? PRIORITY.indexOf(a.key) - PRIORITY.indexOf(b.key) : a.s === "bad" ? -1 : 1)).slice(0, 2);
  const C = 2 * Math.PI * 42, ringColor = sc >= 80 ? "var(--good)" : sc >= 55 ? "var(--warn)" : "var(--bad)";
  const summary = fixes.length ? `먼저 ${fixes.map((f) => f.title).join(", ")}부터 연습해 보세요.` : "전반적으로 효율적인 자세입니다. 지금 자세를 유지하세요.";
  const prev = opt.prev;
  const delta = (key, v) => {
    if (!prev || prev[key] == null || !Number.isFinite(v)) return "";
    const d = v - prev[key];
    if (Math.abs(d) < 0.5) return `<span class="delta">지난번과 같음</span>`;
    return `<span class="delta">지난번보다 ${d > 0 ? "+" : ""}${d.toFixed(Math.abs(d) < 10 ? 1 : 0)}</span>`;
  };
  const rawFor = { cadence: m.cadence, shank: m.shank, knee: m.knee, trunk: m.trunk, vo: m.voCm ?? m.voPctLeg, elbow: m.elbow };

  const quality = [];
  if (opt.example) quality.push("아래는 예시 결과입니다. 영상을 찍거나 올리면 내 결과로 바뀝니다.");
  else {
    if (m.sideRatio > 0.4) quality.push("카메라가 정측면이 아니라서 각도 값이 실제와 다를 수 있어요. 다음에는 몸과 직각이 되게 찍어 보세요.");
    else if (m.sideRatio > 0.25) quality.push("카메라가 약간 비스듬했어요. 각도 값에 몇 도 정도 오차가 있을 수 있습니다.");
    if (m.contactCount < 4) quality.push("착지 장면이 충분히 잡히지 않아 착지 관련 값의 정확도가 낮아요. 10초 이상, 발이 잘 보이게 찍어 주세요.");
    if (m.coverage < 0.7) quality.push(`영상의 ${Math.round((1 - m.coverage) * 100)}% 구간에서 사람을 찾지 못했어요.`);
    if (!m.height) quality.push("키를 입력하면 착지 거리와 상하 움직임을 cm로 볼 수 있어요.");
    if (!m.speed) quality.push("러닝머신 속도를 입력하면 보폭과 보폭 대비 상하 움직임 비율까지 계산합니다.");
    quality.push(`분석한 구간 ${m.span.toFixed(1)}초, 프레임 ${m.posed}개, 착지 ${m.contactCount}번.`);
  }

  resultsEl.innerHTML = `
    <div class="card r-head">
      <div class="ring">
        <svg viewBox="0 0 100 100" aria-hidden="true"><circle cx="50" cy="50" r="42" fill="none" stroke="var(--surface-2)" stroke-width="9"/>
        <circle cx="50" cy="50" r="42" fill="none" stroke="${ringColor}" stroke-width="9" stroke-linecap="round" stroke-dasharray="${(C * sc) / 100} ${C}"/></svg>
        <span class="num">${sc}</span><small>자세 점수</small>
      </div>
      <div style="min-width:0">
        <h2>분석 결과${opt.example ? '<span class="badge">예시</span>' : ""}</h2>
        <p>${summary}</p>
      </div>
    </div>
    ${fixes.length ? `<div class="card focus"><h3 class="sec-title">먼저 고칠 것</h3>${fixes.map((f) => `
      <div class="fix"><b>${f.title}</b><p>${f.tip}</p><p class="drill">연습: ${f.drill}</p></div>`).join("")}</div>` : ""}
    <div class="metrics">${items.map((i) => metricCard(i, delta(i.key, rawFor[i.key]))).join("")}</div>
    ${opt.thumbs && opt.thumbs.length ? `<div class="card" style="display:grid;gap:10px"><h3 class="sec-title">착지 순간</h3>
      <p class="note">노란 선이 정강이, 흰 점선이 엉덩이 위치입니다. 노란 점(발목)이 점선에 가까울수록 좋아요.</p>
      <div class="thumbs" id="thumbs"></div></div>` : ""}
    ${opt.url ? `<div style="display:grid;gap:8px"><h3 class="sec-title">분석 영상</h3>
      <div class="replay"><video id="replay" playsinline muted controls loop></video><canvas id="replayOv"></canvas></div></div>` : ""}
    <ul class="quality">${quality.map((q) => `<li>${q}</li>`).join("")}</ul>
    <details class="how card"><summary>어떻게 측정하나요?</summary><div>
      <p>영상의 각 프레임에서 MediaPipe로 관절 33곳의 위치를 찾습니다.</p>
      <p><b>케이던스</b>는 엉덩이가 오르내리는 주기로, <b>착지</b>는 발이 바닥선에 닿는 순간으로 찾습니다.</p>
      <p><b>정강이 각도</b>는 착지 순간 무릎에서 발목으로 내린 선이 수직에서 앞으로 기운 정도입니다. 0~7°면 발이 몸 아래에 떨어진 것입니다.</p>
      <p><b>cm 환산</b>은 입력한 키와 몸통·허벅지·정강이 길이 비율(키의 약 78%)로 화면의 픽셀을 실제 길이로 바꿉니다. 카메라 거리가 달라도 같은 기준이 됩니다.</p>
      <p>2D 영상이라 카메라 각도에 따라 몇 도 정도 오차가 있습니다. 같은 자리에서 찍어 비교하면 변화를 가장 잘 볼 수 있어요.</p>
    </div></details>`;

  if (opt.thumbs && opt.thumbs.length) {
    const box = $("#thumbs");
    for (const { c, canvas } of opt.thumbs) {
      const fig = document.createElement("figure");
      fig.appendChild(canvas);
      const cap = document.createElement("figcaption");
      cap.textContent = `${c.t.toFixed(1)}초 · ${c.side === "L" ? "왼발" : "오른발"} · ${c.shank.toFixed(0)}°`;
      fig.appendChild(cap); box.appendChild(fig);
    }
  }
  if (opt.url) setupReplay(opt.url, opt.frames, opt.W, opt.H, m.nearKey);
  if (!opt.example) resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
}

function metricCard(i, deltaHtml) {
  const [lo, hi] = i.scale || [0, 1];
  const pos = (x) => Math.max(0, Math.min(100, ((x - lo) / (hi - lo)) * 100));
  const val = parseFloat(i.value);
  return `<article class="metric">
    <div class="m-top"><span class="m-label">${i.label}</span><span class="pill" data-s="${i.s}">${STATUS[i.s]}</span></div>
    <div class="m-val ${i.text ? "" : "num"}" style="${i.text ? "font-size:1.35rem;font-weight:700" : ""}">${i.value}${i.unit ? `<small>${i.unit}</small>` : ""} ${deltaHtml || ""}</div>
    ${i.scale ? `<div class="range" role="img" aria-label="권장 범위 ${i.band[0]}~${i.band[1]}${i.unit}, 내 값 ${i.value}${i.unit}">
      <span class="band" style="left:${pos(i.band[0])}%;width:${pos(i.band[1]) - pos(i.band[0])}%"></span>
      <span class="dot" style="left:${pos(val)}%"></span></div>
      <div class="range-lbl"><span>${lo}</span><span>권장 ${i.band[0]}~${i.band[1]}</span><span>${hi}</span></div>` : ""}
    <p class="m-sub">${i.sub}</p>
    <p class="m-tip">${i.tip}</p>
  </article>`;
}

function setupReplay(url, frames, W, H, near) {
  const v = $("#replay"), cv = $("#replayOv");
  v.src = url; cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d");
  const posed = frames.filter((f) => f.lm);
  const find = (t) => {
    let lo = 0, hi = posed.length - 1;
    while (lo < hi) { const md = (lo + hi) >> 1; if (posed[md].vt < t) lo = md + 1; else hi = md; }
    const a = posed[Math.max(0, lo - 1)], b = posed[lo];
    return Math.abs(a.vt - t) < Math.abs(b.vt - t) ? a : b;
  };
  let raf = 0, alive = true;
  const draw = () => {
    ctx.clearRect(0, 0, W, H);
    const f = posed.length && find(v.currentTime);
    if (f && Math.abs(f.vt - v.currentTime) < 0.2) drawPose(ctx, f.lm, (p) => [p[0] * W, p[1] * H], Math.max(3, H / 160), near);
  };
  const loop = () => { if (!alive) return; draw(); raf = requestAnimationFrame(loop); };
  v.addEventListener("loadeddata", () => { v.playbackRate = 0.5; draw(); });
  v.addEventListener("seeked", draw);
  loop();
  replayStop = () => { alive = false; cancelAnimationFrame(raf); };
}

/* ---------- 기록 ---------- */
function toRecord(m) {
  const items = grade(m);
  const r = (x, d = 1) => (Number.isFinite(x) ? +x.toFixed(d) : null);
  return { at: Date.now(), score: score(items), cadence: r(m.cadence, 0), shank: r(m.shank), knee: r(m.knee, 0), trunk: r(m.trunk), vo: r(m.voCm ?? m.voPctLeg), voUnit: m.voCm != null ? "cm" : "%", elbow: r(m.elbow, 0) };
}
function saveHistory(rec) {
  const h = store.get("rc.history", []);
  h.unshift(rec); store.set("rc.history", h.slice(0, 50));
}
function renderHistory() {
  const h = store.get("rc.history", []);
  const body = $("#histBody");
  if (!h.length) { body.innerHTML = `<tr><td colspan="7" style="text-align:left;color:var(--muted)">아직 기록이 없어요. 첫 영상을 분석하면 여기에 쌓입니다.</td></tr>`; return; }
  const f = (x, u = "") => (x == null ? "–" : x + u);
  body.innerHTML = h.map((r) => {
    const d = new Date(r.at);
    return `<tr><td>${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}</td>
      <td>${r.score}</td><td>${f(r.cadence)}</td><td>${f(r.shank, "°")}</td><td>${f(r.knee, "°")}</td><td>${f(r.trunk, "°")}</td><td>${f(r.vo, r.voUnit)}</td></tr>`;
  }).join("");
}
const clearBtn = $("#clearHist");
let clearArmed = false;
clearBtn.addEventListener("click", () => {
  if (!clearArmed) { clearArmed = true; clearBtn.textContent = "한 번 더 누르면 지워집니다"; setTimeout(() => { clearArmed = false; clearBtn.textContent = "기록 지우기"; }, 4000); return; }
  store.set("rc.history", []); clearArmed = false; clearBtn.textContent = "기록 지우기"; renderHistory();
});

/* ---------- 첫 화면: 예시 결과 ---------- */
renderResults({
  cadence: 158, shank: 11.4, knee: 12, foot: 14, overPct: 27, overCm: 23, trunk: 4.6, elbow: 84,
  voCm: 10.3, voPctLeg: 12.4, height: 172, speed: null, nearKey: "L", sideRatio: 0.1, contactCount: 20, coverage: 1, span: 12, posed: 360,
}, { example: true });
