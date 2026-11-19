"""턴 감지: Silero VAD, Smart Turn v3.2, 한국어 텍스트 규칙, 말 끝 결정기.

onnxruntime을 쓰는 vad/smart_turn은 필요할 때 직접 import한다(가벼운 import 유지).
"""
from agent.turn.detector import DEFAULT_CONFIG, RuleTurnDetector, TurnConfig, decide, wait_ms
from agent.turn.rules import RuleResult, analyze, end_score, is_strong_hold

__all__ = [
    "DEFAULT_CONFIG", "RuleTurnDetector", "TurnConfig", "decide", "wait_ms",
    "RuleResult", "analyze", "end_score", "is_strong_hold",
]
