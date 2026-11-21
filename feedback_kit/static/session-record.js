// 통화 녹음·묶음 도우미(AG-T52, AG-T53 feedback_kit 원본). 브라우저에서는 window.DemoRecord, node 테스트에서는 module.exports.
// 모든 PCM은 16kHz s16le mono(Int16Array). 시각은 통화 연결 순간부터의 ms.
"use strict";

(function (root) {
  const RATE = 16000;
  const SPR = RATE / 1000;           // ms당 샘플 수
  const LEAD_MS = 500;               // 고객 구간: 말 시작보다 이만큼 앞부터
  const TAIL_MS = 200;               // 구간 끝 여유
  const MAX_TRAIL_MS = 800;          // 고객 구간: 말 끝 뒤로 이보다 길게 붙이지 않는다(판정이 늦은 턴)

  // 고객 트랙(보낸 프레임을 순서대로) + 상담원 트랙(재생 예약 위치에 놓은 조각)
  class Recorder {
    constructor() {
      this.mic = [];
      this.micSamples = 0;
      this.agent = [];                 // [{pos, data}]
    }

    addMic(int16) {
      this.mic.push(int16);
      this.micSamples += int16.length;
    }

    // pos: 트랙 안 샘플 위치(재생 시각). 0보다 앞은 잘라낸다.
    addAgent(pos, int16) {
      pos = Math.round(pos);
      if (pos < 0) {
        if (-pos >= int16.length) return;
        int16 = int16.subarray(-pos);
        pos = 0;
      }
      this.agent.push({ pos, data: int16 });
    }

    // 끼어들기로 재생을 멈춘 위치. 그 뒤로 예약돼 있던 소리는 실제로 들리지 않았다.
    cut(pos) {
      pos = Math.max(0, Math.round(pos));
      for (const ch of this.agent) {
        if (ch.pos + ch.data.length > pos) ch.data = ch.data.subarray(0, Math.max(0, pos - ch.pos));
      }
      this.agent = this.agent.filter((ch) => ch.data.length > 0);
    }

    micTrack() {
      const out = new Int16Array(this.micSamples);
      let o = 0;
      for (const f of this.mic) { out.set(f, o); o += f.length; }
      return out;
    }

    agentTrack() {
      let len = 0;
      for (const ch of this.agent) len = Math.max(len, ch.pos + ch.data.length);
      const out = new Int16Array(len);
      for (const ch of this.agent) out.set(ch.data, ch.pos);
      return out;
    }

    micMs() { return Math.ceil(this.micSamples / SPR); }

    durationMs() {
      let len = this.micSamples;
      for (const ch of this.agent) len = Math.max(len, ch.pos + ch.data.length);
      return Math.ceil(len / SPR);
    }
  }

  // 서버가 말 구간을 알려 주지 않을 때(대회 프로토콜만 쓰는 상담원) 브라우저가 마이크 프레임으로 잰다.
  // frame(ms, voiced): 고객 트랙 위치 ms의 20ms 프레임. take(): 지난 take 이후 말 구간 또는 null.
  class SpeechTracker {
    constructor(frameMs) { this.frameMs = frameMs || 20; this.start = null; this.end = null; }
    frame(ms, voiced) {
      if (!voiced) return;
      if (this.start == null) this.start = ms;
      this.end = ms + this.frameMs;
    }
    take() {
      const r = this.start == null ? null : { speech_start_ms: this.start, speech_end_ms: this.end };
      this.start = this.end = null;
      return r;
    }
  }

  function wavBytes(int16, channels) {
    channels = channels || 1;
    const dataLen = int16.length * 2;
    const buf = new ArrayBuffer(44 + dataLen);
    const v = new DataView(buf);
    const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    str(0, "RIFF"); v.setUint32(4, 36 + dataLen, true); str(8, "WAVE");
    str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, channels, true);
    v.setUint32(24, RATE, true); v.setUint32(28, RATE * channels * 2, true);
    v.setUint16(32, channels * 2, true); v.setUint16(34, 16, true);
    str(36, "data"); v.setUint32(40, dataLen, true);
    const bytes = new Uint8Array(buf);
    bytes.set(new Uint8Array(int16.buffer, int16.byteOffset, dataLen), 44);   // little-endian 플랫폼 전제
    return bytes;
  }

  // mono 둘 → 스테레오(왼쪽 a, 오른쪽 b). 짧은 쪽은 무음.
  function interleave(a, b) {
    const n = Math.max(a.length, b.length);
    const out = new Int16Array(n * 2);
    for (let i = 0; i < a.length; i++) out[i * 2] = a[i];
    for (let i = 0; i < b.length; i++) out[i * 2 + 1] = b[i];
    return out;
  }

  function sliceMs(track, startMs, endMs) {
    const a = Math.max(0, Math.min(track.length, Math.round(startMs * SPR)));
    const b = Math.max(a, Math.min(track.length, Math.round(endMs * SPR)));
    return track.subarray(a, b);
  }

  // 턴마다 고객·상담원 오디오 구간. turns[i] = {ts: {...}, decided: {silence_ms}}
  //  고객: 말 시작 LEAD_MS 앞(앞 턴 판정 시각보다 앞으로는 안 감) ~ 판정 + TAIL_MS. speech_end_ms는 구간 시작 기준.
  //  상담원: 판정 ~ 마지막 재생 끝(모르면 turn.end) + TAIL_MS.
  function turnClips(turns, durationMs) {
    let prev = 0;
    return turns.map((t) => {
      const ts = t.ts || {};
      const d = ts.decided_ms;
      const audioEnd = ts.agent_audio_end_ms != null ? ts.agent_audio_end_ms : ts.turn_end_ms;
      if (d == null) {
        // 고객 말 없이 상담원이 먼저 말한 턴(인사 등): 상담원 첫 소리부터
        const fs = ts.first_sound_ms;
        if (fs == null || audioEnd == null) return { customer: null, agent: null };
        const ae = Math.min(durationMs, audioEnd + TAIL_MS);
        return { customer: null, agent: ae > fs ? { start_ms: fs, end_ms: ae } : null };
      }
      const ss = ts.speech_start_ms;
      const cs = Math.max(prev, ss != null ? ss - LEAD_MS : prev);
      const silence = (t.decided && t.decided.silence_ms) || 0;
      const speechEnd = d - silence;
      const ce = Math.min(durationMs, Math.min(d, speechEnd + MAX_TRAIL_MS) + TAIL_MS);
      const customer = ce > cs ? { start_ms: cs, end_ms: ce, speech_end_ms: Math.max(0, speechEnd - cs) } : null;
      let agent = null;
      if (audioEnd != null) {
        const ae = Math.min(durationMs, audioEnd + TAIL_MS);
        if (ae > d) agent = { start_ms: d, end_ms: ae };
      }
      prev = d;
      return { customer, agent };
    });
  }

  function toBase64(bytes) {
    let s = "";
    const CH = 0x8000;
    for (let i = 0; i < bytes.length; i += CH) s += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
    return btoa(s);
  }

  function fromBase64(b64) {
    const s = atob(b64);
    const out = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
    return out;
  }

  // WAV 바이트 → Int16Array. 16kHz mono 16bit가 아니면 Error.
  function wavToInt16(bytes) {
    const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const tag = (o) => String.fromCharCode(bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]);
    if (bytes.length < 12 || tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new Error("WAV 파일이 아닙니다");
    let o = 12, fmt = null;
    while (o + 8 <= bytes.length) {
      const id = tag(o), size = v.getUint32(o + 4, true), body = o + 8;
      if (id === "fmt ") {
        fmt = { ch: v.getUint16(body + 2, true), rate: v.getUint32(body + 4, true), bits: v.getUint16(body + 14, true) };
      } else if (id === "data") {
        if (!fmt || fmt.ch !== 1 || fmt.rate !== RATE || fmt.bits !== 16) {
          throw new Error("16kHz mono 16bit WAV가 아닙니다");
        }
        const n = Math.floor(Math.min(size, bytes.length - body) / 2);
        const out = new Int16Array(n);
        for (let i = 0; i < n; i++) out[i] = v.getInt16(body + i * 2, true);
        return out;
      }
      o = body + size + (size % 2);
    }
    throw new Error("WAV data 청크가 없습니다");
  }

  const api = { RATE, Recorder, SpeechTracker, wavBytes, interleave, sliceMs, turnClips, toBase64, fromBase64, wavToInt16 };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.DemoRecord = api;
})(typeof self !== "undefined" ? self : this);
