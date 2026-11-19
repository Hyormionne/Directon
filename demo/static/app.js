// 다이렉트몰 상담 체험 — 브라우저 마이크 ↔ 상담원 서버(측정기 프로토콜) 직접 연결.
// 빌드 도구·외부 라이브러리 없음.
"use strict";

(() => {
  const VOICE_RMS = 0.02;          // 마이크 프레임 RMS가 이보다 크면 "말하는 중"(float, -34dBFS 정도)
  const AGENT_VOICE_RMS = 300;     // 상담원 첫 유성음 판정(대회 채점과 같은 int16 RMS 기준)
  const JITTER_S = 0.06;           // 재생 지터 버퍼
  const ECHO_TAIL_S = 0.25;        // 상담원 소리가 끝난 뒤에도 잠깐 마이크를 막는다(잔향)
  const TARGET_MS = 1500;          // 대회 속도 만점 기준

  const $ = (id) => document.getElementById(id);
  const el = {
    conn: $("conn"), phase: $("phase"), caller: $("caller"), now: $("now"), half: $("halfduplex"),
    start: $("start"), stop: $("stop"), meter: $("meter-bar"), notice: $("notice"),
    orders: $("orders"), orderList: $("order-list"), memberPhone: $("member-phone"),
    log: $("log"), sessionId: $("session-id"), latValue: $("lat-value"), latServer: $("lat-server"),
    latHistory: $("lat-history"), fbForm: $("fb-form"), fbSend: $("fb-send"), fbStatus: $("fb-status"),
    fbComment: $("fb-comment"), tpl: $("turn-tpl"),
  };

  let config = { agent_ws: "ws://127.0.0.1:9000", mock_url: "http://127.0.0.1:8100", default_now: "2026-11-20 14:00" };
  let members = [];

  // 통화 상태
  let call = null;  // {ws, ctx, stream, node, source, sessionId, ...}

  // ── 초기화 ────────────────────────────────────────
  async function init() {
    try {
      const r = await fetch("/api/config");
      if (r.ok) config = Object.assign(config, await r.json());
    } catch (e) { /* 기본값 */ }
    if (config.default_now) el.now.value = config.default_now.replace(" ", "T");
    try {
      const r = await fetch("/api/members");
      members = r.ok ? await r.json() : [];
    } catch (e) {
      members = [];
      setNotice("회원 목록을 불러오지 못했습니다(데모 서버 확인).");
    }
    for (const m of members) {
      const o = document.createElement("option");
      o.value = m.phone;
      o.textContent = `${m.name} · ${m.phone} · ${m.grade} (주문 ${m.orders.length}건)`;
      el.caller.appendChild(o);
    }
    el.caller.addEventListener("change", renderOrders);
    el.start.addEventListener("click", startCall);
    el.stop.addEventListener("click", () => endCall("사용자가 통화를 종료했습니다."));
    el.fbForm.addEventListener("submit", sendFeedback);
    if (!window.isSecureContext) {
      setNotice("마이크는 http://127.0.0.1 또는 http://localhost 주소에서만 쓸 수 있습니다.");
    }
    setInterval(updatePhase, 100);
    renderOrders();
  }

  function setNotice(text) { el.notice.textContent = text || ""; }

  function selectedMember() {
    return members.find((m) => m.phone === el.caller.value) || null;
  }

  function renderOrders() {
    const m = selectedMember();
    el.orderList.textContent = "";
    if (!m) { el.orders.hidden = true; return; }
    el.orders.hidden = false;
    el.memberPhone.textContent = `${m.name} · ${m.phone_speech}`;
    for (const o of m.orders) {
      const li = document.createElement("li");
      const top = document.createElement("div");
      const id = document.createElement("span");
      id.className = "order-id";
      id.textContent = o.order_id;
      const st = document.createElement("span");
      st.className = "order-status";
      st.textContent = o.status;
      top.append(id, st);
      const speech = document.createElement("div");
      speech.className = "order-speech";
      speech.textContent = `"${o.order_speech}"`;
      const prod = document.createElement("div");
      prod.className = "muted";
      prod.textContent = `${o.product} · 주문 ${o.ordered_at}`;
      li.append(top, speech, prod);
      el.orderList.appendChild(li);
    }
  }

  function setConn(state, text) {
    el.conn.dataset.state = state;
    el.conn.textContent = text;
  }

  // ── 통화 시작/종료 ─────────────────────────────────
  function makeSessionId() {
    const d = new Date();
    const p = (n) => String(n).padStart(2, "0");
    const ts = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
    return `DEMO-${ts}-${Math.random().toString(36).slice(2, 6)}`;
  }

  async function startCall() {
    if (call) return;
    setNotice("");
    el.start.disabled = true;
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
      });
    } catch (e) {
      el.start.disabled = false;
      setNotice("마이크를 쓸 수 없습니다: " + (e && e.message ? e.message : e) + " — 주소창 왼쪽 자물쇠에서 마이크를 허용하세요.");
      return;
    }
    // 16kHz 컨텍스트면 브라우저가 고품질로 리샘플한다(워클릿은 비율 1로 프레임만 자름).
    // 지원하지 않는 브라우저는 기본 샘플레이트 + 워클릿 다운샘플로 대체.
    let ctx;
    try { ctx = new AudioContext({ sampleRate: 16000 }); } catch (e) { ctx = new AudioContext(); }
    try {
      await ctx.audioWorklet.addModule("mic-worklet.js");
    } catch (e) {
      stream.getTracks().forEach((t) => t.stop());
      ctx.close();
      el.start.disabled = false;
      setNotice("오디오 처리 모듈을 불러오지 못했습니다(크롬·엣지 최신 버전을 쓰세요).");
      return;
    }
    const source = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, "mic-frame-processor");
    const sink = ctx.createGain();
    sink.gain.value = 0;                       // 워클릿을 돌리기 위한 연결(소리는 내보내지 않음)
    source.connect(node);
    node.connect(sink);
    sink.connect(ctx.destination);

    const sessionId = makeSessionId();
    const member = selectedMember();
    call = {
      ws: null, ctx, stream, node, source, sink, sessionId, member,
      now: el.now.value.replace("T", " "),
      halfDuplex: el.half.checked,
      open: false, ended: false, responding: false,
      nextTime: 0, carry: null,
      lastVoiceAt: 0, lastVoiceSeen: 0,
      awaitingFirstSound: true, pendingLatency: null,
      turn: null, lastTurn: null, turns: [], callsSeen: 0,
      startedAt: new Date().toISOString(), endedAt: null, feedbackSent: false,
    };
    el.sessionId.textContent = sessionId;
    el.log.textContent = "";
    el.latHistory.textContent = "";
    el.latServer.textContent = "";
    setLatency(null);
    el.fbSend.disabled = true;
    el.fbStatus.textContent = "";
    el.caller.disabled = el.now.disabled = el.half.disabled = true;

    node.port.onmessage = (ev) => onMicFrame(ev.data);

    setConn("connecting", "연결 중");
    let ws;
    try {
      ws = new WebSocket(config.agent_ws);
    } catch (e) {
      endCall("상담원 서버 주소가 올바르지 않습니다: " + config.agent_ws);
      return;
    }
    ws.binaryType = "arraybuffer";
    call.ws = ws;
    const thisCall = call;
    ws.onopen = () => {
      if (call !== thisCall) return;
      call.open = true;
      setConn("open", "연결됨");
      el.stop.disabled = false;
      const caller = member ? { name: member.name, phone: member.phone, email: null } : null;
      ws.send(JSON.stringify({
        type: "session.start", session_id: sessionId, case_id: "DEMO", now: call.now,
        history: [], caller, tools_url: config.mock_url,
        audio: { encoding: "pcm_s16le", sample_rate: 16000, channels: 1 },
        debug: true,
      }));
      showEmptyHint();
    };
    ws.onmessage = (ev) => {
      if (call !== thisCall) return;
      if (typeof ev.data === "string") onText(ev.data);
      else onAudio(ev.data);
    };
    ws.onerror = () => {
      if (call !== thisCall) return;
      if (!call.open) {
        setConn("error", "연결 실패");
        endCall(`상담원 서버(${config.agent_ws})에 연결할 수 없습니다. 실행 스크립트로 서버를 띄웠는지, 예열이 끝났는지 확인하세요.`, true);
      }
    };
    ws.onclose = () => {
      if (call !== thisCall) return;
      endCall(call.open ? "상담원 서버와 연결이 끊겼습니다." : "", !call.open);
    };
  }

  function showEmptyHint() {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "연결됐습니다. 이제 말해 보세요. 말을 마치면 상담원이 답합니다.";
    el.log.appendChild(li);
  }

  function endCall(message, keepConnState) {
    const c = call;
    if (!c || c.ended) return;
    c.ended = true;
    c.endedAt = new Date().toISOString();
    try { if (c.ws && c.ws.readyState <= 1) c.ws.close(); } catch (e) { /* 무시 */ }
    try { c.node.port.onmessage = null; c.source.disconnect(); c.node.disconnect(); } catch (e) { /* 무시 */ }
    c.stream.getTracks().forEach((t) => t.stop());
    // 남은 상담원 음성은 끝까지 들려주고 닫는다
    const remain = Math.max(0, c.nextTime - c.ctx.currentTime);
    setTimeout(() => { try { c.ctx.close(); } catch (e) { /* 무시 */ } }, remain * 1000 + 300);
    if (!keepConnState) setConn("idle", "연결 안 됨");
    el.start.disabled = false;
    el.stop.disabled = true;
    el.caller.disabled = el.now.disabled = el.half.disabled = false;
    el.meter.style.width = "0";
    if (message) setNotice(message);
    el.fbSend.disabled = c.turns.length === 0;
    if (c.turns.length) el.fbStatus.textContent = "통화가 끝났습니다. 점수와 의견을 남겨 주세요.";
  }

  // ── 마이크 → 서버 ──────────────────────────────────
  function agentAudible(c) {
    return c.ctx.currentTime < c.nextTime + ECHO_TAIL_S;
  }

  function micMuted(c) {
    return c.halfDuplex && (c.responding || agentAudible(c));
  }

  function onMicFrame(data) {
    const c = call;
    if (!c || c.ended) return;
    const muted = micMuted(c);
    const db = data.rms > 0 ? 20 * Math.log10(data.rms) : -90;
    const pct = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
    el.meter.style.width = pct + "%";
    el.meter.classList.toggle("muted-mic", muted);
    const now = performance.now();
    if (!muted && data.rms >= VOICE_RMS) c.lastVoiceAt = now;
    if (!c.open || !c.ws || c.ws.readyState !== 1) return;
    c.ws.send(muted ? new ArrayBuffer(640) : data.pcm);
  }

  // ── 서버 → 재생 ────────────────────────────────────
  function onAudio(buf) {
    const c = call;
    let bytes = new Uint8Array(buf);
    if (c.carry) {                              // 홀수 바이트가 남았던 경우
      const merged = new Uint8Array(c.carry.length + bytes.length);
      merged.set(c.carry, 0);
      merged.set(bytes, c.carry.length);
      bytes = merged;
      c.carry = null;
    }
    if (bytes.length % 2) {
      c.carry = bytes.slice(bytes.length - 1);
      bytes = bytes.slice(0, bytes.length - 1);
    }
    const n = bytes.length / 2;
    if (!n) return;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.length);
    const f32 = new Float32Array(n);
    let sq = 0;
    for (let i = 0; i < n; i++) {
      const s = view.getInt16(i * 2, true);
      sq += s * s;
      f32[i] = s / 32768;
    }
    const rms = Math.sqrt(sq / n);
    if (c.awaitingFirstSound && rms >= AGENT_VOICE_RMS) {
      c.awaitingFirstSound = false;
      const lat = c.lastVoiceAt ? Math.round(performance.now() - c.lastVoiceAt) : null;
      const t = ensureTurn();
      t.clientLatencyMs = lat;
      renderTurn(t);
      setLatency(lat);
    }
    const ab = c.ctx.createBuffer(1, n, 16000);
    ab.copyToChannel(f32, 0);
    const src = c.ctx.createBufferSource();
    src.buffer = ab;
    src.connect(c.ctx.destination);
    const startAt = Math.max(c.nextTime, c.ctx.currentTime + JITTER_S);
    src.start(startAt);
    c.nextTime = startAt + ab.duration;
  }

  // ── 서버 텍스트 이벤트 ──────────────────────────────
  function onText(raw) {
    let msg;
    try { msg = JSON.parse(raw); } catch (e) { return; }
    const c = call;
    switch (msg.type) {
      case "debug.turn":
        if (msg.event === "decided") {
          c.responding = true;
          const t = ensureTurn();
          t.decided = { reason: msg.reason, silence_ms: msg.silence_ms, speech_ms: msg.speech_ms,
                        text: msg.text, opener: msg.opener };
          renderTurn(t);
        } else if (msg.event === "speech_start") {
          c.awaitingFirstSound = true;
        }
        break;
      case "debug.stt": {
        const t = ensureTurn();
        t.user = msg.text || "";
        t.userFinal = true;
        renderTurn(t);
        break;
      }
      case "debug.tool": {
        const t = ensureTurn();
        t.tools.push({ name: msg.name, arguments: msg.arguments, status: msg.status, matched: undefined });
        renderTurn(t);
        break;
      }
      case "agent.transcript": {
        const t = ensureTurn();
        t.agent = msg.text || "";
        renderTurn(t);
        break;
      }
      case "turn.end": {
        const t = ensureTurn();
        t.ended = true;
        c.responding = false;
        c.awaitingFirstSound = true;
        c.lastTurn = t;
        c.turn = null;
        renderTurn(t);
        fetchCalls(t);
        break;
      }
      case "debug.timing": {
        const t = c.turn || c.lastTurn;
        if (t) { t.timing = msg; renderTurn(t); renderServerTiming(msg); addHistory(t); }
        break;
      }
      default:
        break;
    }
  }

  function ensureTurn() {
    const c = call;
    if (c.turn) return c.turn;
    const empty = el.log.querySelector(".empty");
    if (empty) empty.remove();
    const node = el.tpl.content.firstElementChild.cloneNode(true);
    const t = { index: c.turns.length + 1, user: "", userFinal: false, agent: "", tools: [], decided: null,
                timing: null, clientLatencyMs: null, vote: null, note: "", ended: false, node };
    node.querySelectorAll(".vote").forEach((b) => {
      b.setAttribute("aria-pressed", "false");
      b.addEventListener("click", () => {
        t.vote = t.vote === b.dataset.vote ? null : b.dataset.vote;
        node.querySelectorAll(".vote").forEach((x) => x.setAttribute("aria-pressed", String(x.dataset.vote === t.vote)));
      });
    });
    node.querySelector(".note").addEventListener("input", (e) => { t.note = e.target.value; });
    c.turns.push(t);
    c.turn = t;
    el.log.appendChild(node);
    return t;
  }

  async function fetchCalls(turn) {
    const c = call;
    try {
      const r = await fetch("/api/calls?session_id=" + encodeURIComponent(c.sessionId));
      if (!r.ok) return;
      const rows = await r.json();
      const fresh = rows.slice(c.callsSeen);
      c.callsSeen = rows.length;
      for (const row of fresh) {
        const tool = turn.tools.find((x) => x.name === row.name && x.matched === undefined);
        if (tool) tool.matched = row.matched;
        else turn.tools.push({ name: row.name, arguments: row.arguments, status: null, matched: row.matched });
      }
      renderTurn(turn);
    } catch (e) { /* mock 미실행: matched 표시 생략 */ }
  }

  // ── 렌더 ──────────────────────────────────────────
  // 결정기 이유 문자열(예: "end table:final/p0s1(wait=600 ...) silence=620")을 짧은 한국어로
  const GROUPS = {
    strong: "문장이 분명히 끝남", final: "종결 어미", soft: "약한 종결", weak: "약한 종결",
    neutral: "판단 보류(기본 대기)", particle: "조사로 끝남(이어질 수 있음)", hold: "이어지는 말투(연결 어미)",
  };

  function reasonLabel(reason) {
    const r = String(reason || "");
    if (r.startsWith("hard_max")) return "최대 침묵 도달";
    const g = /table:(\w+)/.exec(r);
    const w = /wait=(\d+)/.exec(r);
    if (g) return (GROUPS[g[1]] || g[1]) + (w ? ` → ${w[1]}ms 기다림` : "");
    return r || "?";
  }

  function renderTurn(t) {
    const n = t.node;
    const userP = n.querySelector(".bubble.user .text");
    if (t.userFinal) userP.textContent = t.user || "(알아들은 말 없음)";
    else userP.textContent = (t.decided && t.decided.text) ? t.decided.text + " …" : "…";
    const meta = [];
    if (t.decided) {
      const reason = reasonLabel(t.decided.reason);
      n.querySelector(".meta").title = t.decided.reason || "";
      meta.push(`말 끝 판정: ${reason} · 침묵 ${t.decided.silence_ms}ms · 발화 ${(t.decided.speech_ms / 1000).toFixed(1)}초`);
      if (t.userFinal && t.decided.text && t.decided.text !== t.user) meta.push(`판정 시점 텍스트: "${t.decided.text}"`);
    }
    if (t.clientLatencyMs != null) meta.push(`첫 소리 ${t.clientLatencyMs}ms`);
    n.querySelector(".meta").textContent = meta.join("  |  ");
    const ul = n.querySelector(".tools");
    ul.textContent = "";
    for (const tool of t.tools) {
      const li = document.createElement("li");
      const args = tool.arguments == null ? "" : JSON.stringify(tool.arguments);
      li.append(document.createTextNode(`함수 ${tool.name}(${args})`));
      if (tool.status) li.append(document.createTextNode(` → ${tool.status}`));
      if (tool.matched !== undefined && tool.matched !== null) {
        const m = document.createElement("span");
        m.className = tool.matched ? "ok" : "ng";
        m.textContent = tool.matched ? "  matched" : "  not matched";
        li.append(m);
      }
      ul.appendChild(li);
    }
    n.querySelector(".bubble.agent .text").textContent = t.agent || (t.ended ? "(응답 없음)" : "…");
  }

  function grade(ms) {
    if (ms == null) return "none";
    if (ms <= TARGET_MS) return "good";
    if (ms <= 2500) return "warn";
    return "bad";
  }

  function setLatency(ms) {
    el.latValue.textContent = ms == null ? "—" : (ms / 1000).toFixed(2) + "초";
    el.latValue.dataset.grade = grade(ms);
  }

  const TIMING_LABELS = [
    ["silence_to_commit_ms", "침묵 시작 → 말 끝 판정"],
    ["commit_to_opener_ms", "판정 → 첫 마디 송신"],
    ["commit_to_finalize_ms", "판정 → 최종 인식"],
    ["commit_to_llm_first_ms", "판정 → LLM 첫 응답"],
    ["commit_to_response_audio_ms", "판정 → 본 응답 첫 소리"],
    ["commit_to_turn_end_ms", "판정 → 응답 끝"],
    ["gap_ms", "첫 마디 뒤 공백"],
  ];

  function renderServerTiming(tm) {
    el.latServer.textContent = "";
    for (const [k, label] of TIMING_LABELS) {
      if (tm[k] == null) continue;
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = Math.round(tm[k]) + "ms";
      el.latServer.append(dt, dd);
    }
    const dt = document.createElement("dt");
    dt.textContent = "선호출 재사용";
    const dd = document.createElement("dd");
    dd.textContent = tm.precall_reused ? "예" : "아니요";
    el.latServer.append(dt, dd);
  }

  function addHistory(t) {
    const li = document.createElement("li");
    const ms = t.clientLatencyMs;
    li.className = grade(ms);
    const end = t.timing && t.timing.commit_to_turn_end_ms != null ? ` · 응답 끝 ${(t.timing.commit_to_turn_end_ms / 1000).toFixed(1)}초` : "";
    li.textContent = (ms == null ? "첫 소리 측정 안 됨" : `첫 소리 ${(ms / 1000).toFixed(2)}초`) + end;
    el.latHistory.appendChild(li);
  }

  function updatePhase() {
    const c = call;
    let phase = "idle", label = "대기";
    if (c && !c.ended && c.open) {
      label = "듣는 중";
      if (agentAudible(c) && c.nextTime > c.ctx.currentTime) { phase = "agent"; label = "상담원이 말하는 중"; }
      else if (c.responding) { phase = "thinking"; label = "상담원이 답을 준비하는 중"; }
      else if (performance.now() - c.lastVoiceAt < 300) { phase = "user"; label = "지금 말하는 중"; }
    }
    el.phase.dataset.phase = phase;
    el.phase.textContent = label;
  }

  // ── 피드백 ────────────────────────────────────────
  async function sendFeedback(ev) {
    ev.preventDefault();
    const c = call;
    if (!c || !c.turns.length) return;
    const ratingEl = el.fbForm.querySelector("input[name=rating]:checked");
    const payload = {
      session_id: c.sessionId,
      started_at: c.startedAt,
      ended_at: c.endedAt,
      caller: c.member ? { name: c.member.name, phone: c.member.phone } : null,
      now: c.now,
      half_duplex: c.halfDuplex,
      rating: ratingEl ? Number(ratingEl.value) : null,
      comment: el.fbComment.value.trim(),
      turns: c.turns.map((t) => ({
        index: t.index, user: t.user, agent: t.agent, decided: t.decided, tools: t.tools.map((x) => ({
          name: x.name, arguments: x.arguments, status: x.status, matched: x.matched === undefined ? null : x.matched,
        })),
        client_latency_ms: t.clientLatencyMs, timing: t.timing, vote: t.vote, note: t.note,
      })),
      user_agent: navigator.userAgent,
    };
    el.fbSend.disabled = true;
    try {
      const r = await fetch("/api/feedback", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      });
      const res = await r.json();
      if (!r.ok) throw new Error(res.error || r.status);
      c.feedbackSent = true;
      el.fbStatus.textContent = "고맙습니다! 저장됨: " + res.file;
    } catch (e) {
      el.fbSend.disabled = false;
      el.fbStatus.textContent = "저장 실패: " + (e && e.message ? e.message : e);
    }
  }

  init();
})();
