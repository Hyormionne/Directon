#!/bin/bash
# 단순 조합(naive) 172문항 측정. 사용: bash bench/run_naive.sh [이름=naive] [full|sentence]
cd "c:/Users/bdmg1/Files/Work/Directon"
NAME=${1:-naive}
MODE=${2:-full}
OUT=runs/${NAME}_172_$(date +%H%M)
mkdir -p $OUT
uv run python -m mock_api.server --port 8100 > $OUT/mock.log 2>&1 &
MP=$!
uv run python -m bench.naive_server --port 9000 --tts-mode $MODE > $OUT/agent.log 2>&1 &
AP=$!
for i in $(seq 1 60); do grep -qi "naive server on" $OUT/agent.log && break; sleep 1; done
sleep 3
uv run python -m eval.runner --server ws://127.0.0.1:9000 --cases data/testset/cases.jsonl --audio data/testset/audio --mock http://127.0.0.1:8100 --out $OUT --concurrency 4 > $OUT/runner.log 2>&1
uv run python -m eval.score --run $OUT > $OUT/score.log 2>&1
kill $AP $MP 2>/dev/null; sleep 1
taskkill //F //T //PID $AP >/dev/null 2>&1; taskkill //F //T //PID $MP >/dev/null 2>&1
# uv가 띄운 python 자식이 남는 경우 대비: 포트로 찾아 종료
for p in $(netstat -ano | grep -E ":(9000|8100) .*LISTEN" | awk '{print $5}' | sort -u); do
  taskkill //F //T //PID $p >/dev/null 2>&1
done
echo DONE $OUT
tail -30 $OUT/score.log
