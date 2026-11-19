// 마이크 입력(장치 샘플레이트) → 16kHz mono s16le 20ms(320샘플, 640바이트) 프레임.
// 다운샘플은 구간 평균(박스 필터)으로 한다: 48k→16k면 3샘플 평균, 44.1k처럼 나눠떨어지지 않아도 동작.
class MicFrameProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.t = 0;            // 지금까지 받은 입력 샘플 수(구간 경계 계산용)
    this.next = this.ratio;
    this.acc = 0;
    this.cnt = 0;
    this.frame = new Int16Array(320);
    this.n = 0;
    this.sq = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch = input[0];
    for (let i = 0; i < ch.length; i++) {
      this.acc += ch[i];
      this.cnt += 1;
      this.t += 1;
      if (this.t >= this.next) {
        let v = this.acc / this.cnt;
        this.acc = 0;
        this.cnt = 0;
        this.next += this.ratio;
        if (v > 1) v = 1;
        else if (v < -1) v = -1;
        this.sq += v * v;
        this.frame[this.n++] = v < 0 ? v * 0x8000 : v * 0x7fff;
        if (this.n === 320) {
          const rms = Math.sqrt(this.sq / 320);
          const out = this.frame;
          this.port.postMessage({ pcm: out.buffer, rms }, [out.buffer]);
          this.frame = new Int16Array(320);
          this.n = 0;
          this.sq = 0;
        }
      }
    }
    if (this.t > 1e9) {    // 오래 켜 두어도 부동소수 정밀도 유지
      this.t -= 1e9;
      this.next -= 1e9;
    }
    return true;
  }
}

registerProcessor("mic-frame-processor", MicFrameProcessor);
