"""설정. 값은 환경 변수 또는 맨 위 폴더의 .env 에서 읽는다 (이름=값)."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_env(path=os.path.join(ROOT, ".env")):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
env = os.environ.get

# 행사 게이트웨이(베이즈) 이름으로 넣어도 읽는다: BAZE_API_KEY → Soniox 키, BAZE_WS_URL → Soniox 주소
SONIOX_API_KEY = env("SONIOX_API_KEY", "") or env("BAZE_API_KEY", "")
BAZE_API_KEY = env("BAZE_API_KEY", "")
# 베이즈 게이트웨이 기본 주소 (BAZE_WS_URL 의 호스트에서 만든다): https://<호스트>/v1/gateway
_bws = env("BAZE_WS_URL", "")
BAZE_GATEWAY = env("BAZE_GATEWAY", "") or (("https://" + _bws.split("/")[2] + "/v1/gateway") if _bws.startswith("wss://") else "")
# OpenAI 키가 없으면 베이즈 키로 GPT 도 게이트웨이를 거쳐 부른다
OPENAI_API_KEY = env("OPENAI_API_KEY", "") or BAZE_API_KEY
GEMINI_API_KEY = env("GEMINI_API_KEY", "") or env("GOOGLE_API_KEY", "") or BAZE_API_KEY
USE_BAZE = bool(BAZE_API_KEY) and not env("OPENAI_API_KEY", "")

STT_MODEL = env("STT_MODEL", "stt-rt-v6")
STT_URL = env("STT_URL", "") or env("BAZE_WS_URL", "") or "wss://stt-rt.soniox.com/transcribe-websocket"
STT_MAX_ENDPOINT_DELAY_MS = int(env("STT_MAX_ENDPOINT_DELAY_MS", "500"))   # 500~3000
LLM_MODEL = env("LLM_MODEL", "gpt-5.4-mini")
LLM_URL = env("LLM_URL", "") or (f"{BAZE_GATEWAY}/chat/completions" if USE_BAZE and BAZE_GATEWAY
                                  else "https://api.openai.com/v1/chat/completions")
# 채점(응대 판정)도 같은 곳으로. 채점 모델은 대회 채점기와 같은 gpt-5.5 (게이트웨이에 없으면 JUDGE_MODEL 로 바꾼다)
JUDGE_URL = env("JUDGE_URL", "") or LLM_URL
JUDGE_MODEL = env("JUDGE_MODEL", "gpt-5.5")
LLM_REASONING = env("LLM_REASONING", "none")          # 지원 안 하면 자동으로 빼고 다시 보낸다
TTS_MODEL = env("TTS_MODEL", "gemini-3.8-flash-lite-tts")
TTS_VOICE = env("TTS_VOICE", "Kore")
TTS_RATE = int(env("TTS_RATE", "24000"))               # Gemini 스트리밍 출력 (audio/l16, 24kHz)
TTS_URL = env("TTS_URL", "") or (f"{BAZE_GATEWAY}/gemini/v1beta/models/{{model}}:streamGenerateContent?alt=sse"
                                  if BAZE_API_KEY and not env("GEMINI_API_KEY", "") and BAZE_GATEWAY
                                  else "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse")

# 임시 대체 백엔드 (Soniox·Gemini 키가 없을 때 OpenAI 로 지연 측정). 제출 시에는 soniox / gemini.
STT_BACKEND = env("STT_BACKEND", "soniox")            # soniox | openai
TTS_BACKEND = env("TTS_BACKEND", "gemini")            # gemini | openai
OPENAI_STT_MODEL = env("OPENAI_STT_MODEL", "gpt-4o-transcribe")
OPENAI_VAD_SILENCE_MS = int(env("OPENAI_VAD_SILENCE_MS", "500"))
OPENAI_TTS_MODEL = env("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = env("OPENAI_TTS_VOICE", "marin")

# 모든 문항에서 말 끝 직후 짧은 채움말을 먼저 낸다 (첫 소리 점수). 0 이면 함수 문항에서만.
FILLER_ALWAYS = env("FILLER_ALWAYS", "1") == "1"
# 로컬 시험용: STT/LLM/TTS 를 가짜로 대체한다 (API 키 없이 배관만 확인). 제출 시 0.
FAKE = env("FAKE_APIS", "0") == "1"

RATE = 16000
