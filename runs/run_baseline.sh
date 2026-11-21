#!/bin/bash
cd "c:/Users/bdmg1/Files/Work/Directon"
OUT=runs/${1:-baseline}_172_$(date +%H%M)
mkdir -p $OUT
uv run python -m mock_api.server --port 8100 > $OUT/mock.log 2>&1 &
MP=$!
uv run python -m agent.server --port 9000 > $OUT/agent.log 2>&1 &
AP=$!
for i in $(seq 1 60); do grep -qi "listening\|serving\|ready\|9000" $OUT/agent.log && break; sleep 1; done
sleep 3
uv run python -m eval.runner --server ws://127.0.0.1:9000 --cases legacy/testset/cases.jsonl --audio legacy/testset/audio --mock http://127.0.0.1:8100 --out $OUT --concurrency 4 > $OUT/runner.log 2>&1
uv run python -m eval.score --run $OUT > $OUT/score.log 2>&1
kill $AP $MP 2>/dev/null; sleep 1
taskkill //F //PID $AP >/dev/null 2>&1; taskkill //F //PID $MP >/dev/null 2>&1
echo DONE $OUT
tail -30 $OUT/score.log
