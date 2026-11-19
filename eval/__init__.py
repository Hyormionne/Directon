"""AG-T06 테스트 음성 도구(합성·통화환경 증강) + 측정기·채점기.

- augment: 통화 환경 증강, speech_end 계산, 세그먼트 조립
- synth: Gemini TTS / edge-tts 합성 + 해시 캐시
- build_testset: cases.jsonl → wav + meta.json
- runner: Starter Kit 측정기 확장판
- score: 속도·함수·LLM 심사(옵션) 채점 → report.md
"""
