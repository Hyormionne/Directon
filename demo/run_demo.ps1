# 체험 데모 한 번에 실행 (Windows PowerShell)
#   powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1
# 순서: Mock API(8100) -> 상담원 서버(9000, 예열 포함) -> 데모 페이지(8080) -> 브라우저 열기
# Ctrl+C 를 누르면 세 서버를 모두 끈다(포트 기준 정리 포함).
# 옵션: -Force  이미 포트를 쓰는 프로세스가 있으면 끄고 시작
#       -NoBrowser  브라우저를 자동으로 열지 않음
param(
    [switch]$Force,
    [switch]$NoBrowser,
    [int]$MockPort = 8100,
    [int]$AgentPort = 9000,
    [int]$WebPort = 8080
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$LogDir = Join-Path $Root "runs\logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
# 공식 연습 데이터·매뉴얼로 띄운다(official/은 scripts/build_manual.py로 만든다). 데모는 실제 소리가 필요하니 무음 TTS를 끈다.
$env:DIRECTON_MANUAL = "official/manual.md"
$env:DIRECTON_STYLE = "official/style.md"
$env:DIRECTON_MOCK_DATA = "official/mock_db.json"
$env:DIRECTON_SILENT_TTS = $null
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$DemoNow = "2026-10-16 14:00"   # 공식 연습 데이터의 주문 날짜(9~10월) 뒤

function Test-Port([int]$Port) {
    $c = New-Object System.Net.Sockets.TcpClient
    try {
        $iar = $c.BeginConnect("127.0.0.1", $Port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(300)
        if ($ok) { $c.EndConnect($iar) }
        return $ok -and $c.Connected
    } catch { return $false } finally { $c.Close() }
}

function Get-PortPids([int]$Port) {
    try {
        return @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop |
                 Select-Object -ExpandProperty OwningProcess -Unique)
    } catch { return @() }
}

function Stop-Port([int]$Port) {
    foreach ($procId in (Get-PortPids $Port)) {
        if ($procId -and $procId -ne 0) {
            try { Stop-Process -Id $procId -Force -ErrorAction Stop } catch {}
        }
    }
}

function Wait-Port([int]$Port, [int]$TimeoutSec, [string]$Name, $Proc) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-Port $Port) { return $true }
        if ($Proc -and $Proc.HasExited) {
            Write-Host "  $Name 이(가) 시작하자마자 종료됐습니다. 로그를 확인하세요." -ForegroundColor Red
            return $false
        }
        Start-Sleep -Milliseconds 500
    }
    Write-Host "  $Name 이(가) ${TimeoutSec}초 안에 준비되지 않았습니다." -ForegroundColor Red
    return $false
}

$ports = @($MockPort, $AgentPort, $WebPort)
foreach ($p in $ports) {
    if (Test-Port $p) {
        if ($Force) {
            Write-Host "포트 $p 사용 중 -> 끕니다(-Force)." -ForegroundColor Yellow
            Stop-Port $p
            Start-Sleep -Milliseconds 800
        } else {
            Write-Host "포트 $p 를 이미 쓰는 프로세스가 있습니다(PID: $((Get-PortPids $p) -join ', '))." -ForegroundColor Red
            Write-Host "이전 데모가 남았다면 -Force 옵션으로 다시 실행하세요: powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1 -Force"
            exit 1
        }
    }
}

$procs = @()
function Start-Server([string]$Name, [string[]]$ArgList) {
    $out = Join-Path $LogDir "demo_$Name.log"
    $err = Join-Path $LogDir "demo_$Name.err.log"
    $p = Start-Process -FilePath $Python -ArgumentList $ArgList -WorkingDirectory $Root -NoNewWindow `
        -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
    Write-Host "  $Name 시작 (PID $($p.Id), 로그 runs\logs\demo_$Name*.log)"
    return $p
}

try {
    Write-Host "[1/3] Mock API :$MockPort" -ForegroundColor Cyan
    $mock = Start-Server "mock" @("-m", "mock_api.server", "--port", "$MockPort", "--host", "127.0.0.1",
        "--db", "official/mock_db.json", "--tools", "official/tools.json")
    $procs += $mock
    if (-not (Wait-Port $MockPort 30 "Mock API" $mock)) { throw "mock failed" }

    Write-Host "[2/3] 상담원 서버 :$AgentPort (모델 예열 때문에 20~60초 걸릴 수 있습니다)" -ForegroundColor Cyan
    $agent = Start-Server "agent" @("-m", "agent.server", "--port", "$AgentPort", "--host", "127.0.0.1")
    $procs += $agent
    if (-not (Wait-Port $AgentPort 180 "상담원 서버" $agent)) { throw "agent failed" }

    Write-Host "[3/3] 데모 페이지 :$WebPort" -ForegroundColor Cyan
    $web = Start-Server "web" @("-m", "demo.server", "--port", "$WebPort",
        "--agent-ws", "ws://127.0.0.1:$AgentPort", "--mock-url", "http://127.0.0.1:$MockPort",
        "--data", "official/mock_db.json", "--now", "`"$DemoNow`"", "--presets", "official/demo_presets.json")
    $procs += $web
    if (-not (Wait-Port $WebPort 30 "데모 페이지" $web)) { throw "web failed" }

    $url = "http://127.0.0.1:$WebPort"
    Write-Host ""
    Write-Host "준비 완료: $url" -ForegroundColor Green
    Write-Host "끝내려면 이 창에서 Ctrl+C 를 누르세요."
    if (-not $NoBrowser) { Start-Process $url }

    while ($true) {
        Start-Sleep -Seconds 1
        foreach ($p in $procs) {
            if ($p.HasExited) { throw "서버 하나가 종료됐습니다(PID $($p.Id)). runs\logs\demo_*.log 를 확인하세요." }
        }
    }
} catch {
    if ($_.Exception.Message -notmatch "failed$") { Write-Host $_.Exception.Message -ForegroundColor Red }
    else { Write-Host "시작 실패. runs\logs\demo_*.err.log 를 확인하세요(예: GEMINI_API_KEY)." -ForegroundColor Red }
} finally {
    Write-Host "서버를 끄는 중..." -ForegroundColor Yellow
    foreach ($p in $procs) {
        if ($p -and -not $p.HasExited) {
            try { Stop-Process -Id $p.Id -Force -ErrorAction Stop } catch {}
        }
    }
    # 자식 python이 남을 수 있어 포트 기준으로 한 번 더 정리
    foreach ($p in $ports) { Stop-Port $p }
    Write-Host "종료했습니다."
}
