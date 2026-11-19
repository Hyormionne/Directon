# 외부 접속용 Cloudflare quick tunnel (AG-T18). 측정기(주최 측 노트북)가 우리 서버에 붙을 wss 주소를 만든다.
#
# 사용:  powershell -ExecutionPolicy Bypass -File scripts\tunnel.ps1 [-Port 9000] [-Cloudflared cloudflared]
#   1) 먼저 상담원 서버를 띄운다:  uv run python -m agent.server --port 9000
#   2) 이 스크립트를 실행하면 wss://xxxx.trycloudflare.com 주소를 출력한다. 그 주소를 주최 측에 전달.
#   3) 측정이 끝날 때까지 이 창을 닫지 않는다(Ctrl+C로 종료하면 주소도 사라진다). 다시 켜면 주소가 바뀐다.
#
# cloudflared 설치(Windows, 계정·로그인 불필요):
#   winget install --id Cloudflare.cloudflared
#   (또는 https://github.com/cloudflare/cloudflared/releases 에서 cloudflared-windows-amd64.exe를 받아
#    cloudflared.exe로 이름을 바꾸고 PATH에 두거나 -Cloudflared 경로로 지정)
#   설치 뒤 새 터미널에서 `cloudflared --version`으로 확인.
#
# 참고: quick tunnel은 무료·무계정이지만 가용성 보장이 없다. 당일에는 같은 와이파이 IP 직접 접속
#       (ws://<노트북IP>:9000, 방화벽 9000 허용)을 1순위, 터널을 2순위로 준비한다(docs/decisions/stability.md).
param(
    [int]$Port = 9000,
    [string]$Cloudflared = "cloudflared",
    [int]$WaitSeconds = 60
)
$ErrorActionPreference = "Stop"

$exe = Get-Command $Cloudflared -ErrorAction SilentlyContinue
if (-not $exe) {
    Write-Host "cloudflared를 찾을 수 없습니다. 설치: winget install --id Cloudflare.cloudflared" -ForegroundColor Red
    exit 1
}

$listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $listening) {
    Write-Host "경고: localhost:$Port 에서 대기 중인 서버가 없습니다. 먼저 'uv run python -m agent.server --port $Port'를 실행하세요." -ForegroundColor Yellow
}

$logDir = Join-Path (Split-Path -Parent $PSScriptRoot) "runs/tunnel"   # runs/는 .gitignore 대상
New-Item -ItemType Directory -Force $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$errLog = Join-Path $logDir "tunnel_$stamp.err.log"
$outLog = Join-Path $logDir "tunnel_$stamp.out.log"

$proc = Start-Process -FilePath $exe.Source -ArgumentList @("tunnel", "--no-autoupdate", "--url", "http://localhost:$Port") `
    -RedirectStandardError $errLog -RedirectStandardOutput $outLog -NoNewWindow -PassThru

try {
    $url = $null
    $deadline = (Get-Date).AddSeconds($WaitSeconds)
    while (-not $url -and (Get-Date) -lt $deadline -and -not $proc.HasExited) {
        Start-Sleep -Milliseconds 500
        foreach ($f in @($errLog, $outLog)) {
            if (Test-Path $f) {
                $m = Select-String -Path $f -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -AllMatches | Select-Object -First 1
                if ($m) { $url = $m.Matches[0].Value; break }
            }
        }
    }
    if (-not $url) {
        Write-Host "터널 주소를 받지 못했습니다. 로그: $errLog" -ForegroundColor Red
        exit 1
    }
    $wss = $url -replace "^https://", "wss://"
    Write-Host ""
    Write-Host "측정기 접속 주소:  $wss" -ForegroundColor Green
    Write-Host "(HTTP 확인용: $url)  로그: $errLog"
    Write-Host "점검: uv run python -m eval.stress --server $wss --sessions 1 --rounds 1"
    Write-Host "이 창을 닫거나 Ctrl+C를 누르면 터널이 끝납니다."
    Set-Content -Path (Join-Path $logDir "tunnel_url.txt") -Value $wss -Encoding utf8
    Wait-Process -Id $proc.Id
}
finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
}
