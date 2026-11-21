param(
    [switch]$NoBrowser,
    [switch]$StopOnly
)

$ErrorActionPreference = 'Stop'
try {
    [Console]::InputEncoding = [System.Text.Encoding]::UTF8
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
} catch {}

$ProjectName = 'sentiment-engine'
$OpenUrl = 'http://localhost:8001/dashboard'
$ProjectDir = $PSScriptRoot

function Get-ListeningPids {
    param([Parameter(Mandatory = $true)][int]$Port)

    $pids = New-Object 'System.Collections.Generic.HashSet[int]'
    try {
        $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        foreach ($conn in $conns) {
            if ($conn.OwningProcess -gt 4 -and $conn.OwningProcess -ne $PID) {
                [void]$pids.Add([int]$conn.OwningProcess)
            }
        }
    } catch {}

    $prevEap = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $netstatLines = & netstat.exe -ano -p TCP 2>$null
    } finally {
        $ErrorActionPreference = $prevEap
    }

    foreach ($line in $netstatLines) {
        if ($line -match "^\s*TCP\s+\S+:$Port\s+\S+\s+LISTENING\s+(\d+)\s*$") {
            $foundPid = [int]$Matches[1]
            if ($foundPid -gt 4 -and $foundPid -ne $PID) {
                [void]$pids.Add($foundPid)
            }
        }
    }
    return @($pids)
}

function Stop-ProcessTreeSafe {
    param([Parameter(Mandatory = $true)][int]$TargetPid)
    $prevEap = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & taskkill.exe /F /T /PID $TargetPid *> $null
    } finally {
        $ErrorActionPreference = $prevEap
    }
    Stop-Process -Id $TargetPid -Force -ErrorAction SilentlyContinue
}

function Test-DockerDaemon {
    $dockerCmd = Get-Command docker.exe -ErrorAction SilentlyContinue
    if (-not $dockerCmd) { return $false }
    $prevEap = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $dockerCmd.Source info *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $prevEap
    }
}

function Stop-PortProcess {
    param([Parameter(Mandatory = $true)][int]$Port)

    $pidFile = Join-Path $env:TEMP ("projects-launcher-port-{0}.pid" -f $Port)
    if (Test-Path -LiteralPath $pidFile) {
        $rawPidLine = (Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
        if ($rawPidLine) {
            $parts = $rawPidLine.Trim().Split('|')
            $wrapperPid = 0
            [long]$expectedTicks = 0
            if ([int]::TryParse($parts[0], [ref]$wrapperPid) -and $wrapperPid -gt 4 -and $wrapperPid -ne $PID) {
                if ($parts.Length -gt 1) { [void][long]::TryParse($parts[1], [ref]$expectedTicks) }
                $proc = Get-Process -Id $wrapperPid -ErrorAction SilentlyContinue
                if ($proc -and $proc.ProcessName -match '^(powershell|pwsh)$') {
                    $matchesStart = $true
                    if ($expectedTicks -gt 0) {
                        try { $matchesStart = ($proc.StartTime.ToFileTimeUtc() -eq $expectedTicks) } catch { $matchesStart = $false }
                    }
                    if ($matchesStart) {
                        Write-Host "  [Temizlik] Onceki baslatici penceresi kapatiliyor (PID: $wrapperPid, Port: $Port)..." -ForegroundColor Yellow
                        Stop-ProcessTreeSafe -TargetPid $wrapperPid
                    }
                }
            }
        }
        Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    }

    if (Test-DockerDaemon) {
        $prevEap = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            $runningContainer = & docker.exe ps --filter "name=sentiment-engine-app" --filter "status=running" -q 2>$null
            if ($runningContainer) {
                Write-Host "  [Temizlik] Docker konteyneri (sentiment-engine-app) durduruluyor..." -ForegroundColor Yellow
                Push-Location $ProjectDir
                try {
                    & docker.exe compose stop sentiment-engine *> $null
                } finally {
                    Pop-Location
                }
            }
        } finally {
            $ErrorActionPreference = $prevEap
        }
    }

    $listeningPids = Get-ListeningPids -Port $Port
    foreach ($targetPid in $listeningPids) {
        $proc = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
        $procName = if ($proc) { $proc.ProcessName } else { 'bilinmeyen' }
        if ($procName -match 'docker|vpnkit|wsl') {
            Write-Host "  [Uyari] Port $Port Docker/WSL tarafindan tutuluyor ($procName, PID: $targetPid); Docker Desktop kapatilmiyor." -ForegroundColor Yellow
            continue
        }
        Write-Host "  [Temizlik] Port $Port uzerindeki eski surec sonlandiriliyor ($procName, PID: $targetPid)..." -ForegroundColor Yellow
        Stop-ProcessTreeSafe -TargetPid $targetPid
    }

    $deadline = (Get-Date).AddSeconds(6)
    while ((Get-Date) -lt $deadline) {
        if ((Get-ListeningPids -Port $Port).Count -eq 0) {
            return
        }
        Start-Sleep -Milliseconds 200
    }

    $remaining = Get-ListeningPids -Port $Port
    if ($remaining.Count -gt 0) {
        throw "Port $Port bosaltilamadi. Kalan PID'ler: $($remaining -join ', ')"
    }
}

function Test-TcpPort {
    param([Parameter(Mandatory = $true)][int]$Port)
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $task = $client.ConnectAsync('127.0.0.1', $Port)
        return ($task.Wait(300) -and $client.Connected)
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Wait-ServiceReady {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][int]$Port,
        [Parameter(Mandatory = $true)][string]$HealthUrl,
        [int]$TimeoutSec = 60,
        [System.Diagnostics.Process]$WrapperProcess = $null
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if ($null -ne $WrapperProcess -and $WrapperProcess.HasExited -and -not (Test-TcpPort -Port $Port)) {
            throw "$Name servisi beklenmedik sekilde kapandi (Cikis kodu: $($WrapperProcess.ExitCode))."
        }
        if (Test-TcpPort -Port $Port) {
            try {
                $response = Invoke-WebRequest -UseBasicParsing -Uri $HealthUrl -TimeoutSec 3 -ErrorAction Stop
                if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400) {
                    return
                }
            } catch {}
        }
        Start-Sleep -Milliseconds 400
    }
    throw "$Name servisi $TimeoutSec saniye icinde hazir olmadi ($HealthUrl)."
}

function Resolve-VenvPython {
    foreach ($venvDir in @('.venv', 'venv')) {
        $candidate = Join-Path $ProjectDir "$venvDir\Scripts\python.exe"
        if (Test-Path -LiteralPath $candidate) {
            $prevEap = $ErrorActionPreference
            try {
                $ErrorActionPreference = 'Continue'
                & $candidate -c "import uvicorn, fastapi" *> $null
                if ($LASTEXITCODE -eq 0) {
                    return $candidate
                }
            } catch {
            } finally {
                $ErrorActionPreference = $prevEap
            }
        }
    }
    return $null
}

function Resolve-SystemPython {
    $candidates = @()
    $pyCmd = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pyCmd) { $candidates += @{ Exe = $pyCmd.Source; Prefix = @() } }
    $pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($pyLauncher) { $candidates += @{ Exe = $pyLauncher.Source; Prefix = @('-3') } }
    foreach ($glob in @("$env:LOCALAPPDATA\Programs\Python\Python*\python.exe", "C:\Python*\python.exe")) {
        foreach ($match in (Get-Item -Path $glob -ErrorAction SilentlyContinue)) {
            $candidates += @{ Exe = $match.FullName; Prefix = @() }
        }
    }

    foreach ($cand in $candidates) {
        $prevEap = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            $argsList = @($cand.Prefix) + @('-c', 'import sys; assert sys.version_info >= (3,11)')
            & $cand.Exe @argsList *> $null
            if ($LASTEXITCODE -eq 0) {
                return $cand
            }
        } catch {
        } finally {
            $ErrorActionPreference = $prevEap
        }
    }
    return $null
}

try {
    Write-Host "=== $ProjectName ===" -ForegroundColor Cyan

    Stop-PortProcess -Port 8001
    if ($StopOnly) {
        Write-Host "[OK] $ProjectName durduruldu (Port: 8001)." -ForegroundColor Green
        exit 0
    }

    $venvPython = Resolve-VenvPython
    $useDocker = $false

    if (-not $venvPython) {
        if (Test-DockerDaemon) {
            $useDocker = $true
        } else {
            $sysPy = Resolve-SystemPython
            if ($null -ne $sysPy) {
                Write-Host "[Kurulum] .venv olusturuluyor ve requirements.txt yukleniyor..." -ForegroundColor Yellow
                Push-Location $ProjectDir
                try {
                    $venvArgs = @($sysPy.Prefix) + @('-m', 'venv', '.venv')
                    & $sysPy.Exe @venvArgs
                    if ($LASTEXITCODE -ne 0) { throw ".venv olusturulamadi." }
                    & (Join-Path $ProjectDir '.venv\Scripts\python.exe') -m pip install -r requirements.txt
                    if ($LASTEXITCODE -ne 0) { throw "requirements.txt kurulumu basarisiz oldu." }
                } finally {
                    Pop-Location
                }
                $venvPython = Resolve-VenvPython
            }
        }
    }

    if ($venvPython) {
        $Services = @(
            @{
                Name         = 'Sentiment Engine API & Dashboard (Uvicorn)'
                Port         = 8001
                WorkDir      = $ProjectDir
                HealthUrl    = 'http://127.0.0.1:8001/api/v1/health'
                WindowTitle  = 'sentiment-engine [Port 8001]'
                TimeoutSec   = 60
                StartCommand = "& '$($venvPython.Replace("'", "''"))' -m uvicorn app.main:app --host 127.0.0.1 --port 8001"
                IsDocker     = $false
            }
        )
    } elseif ($useDocker) {
        $Services = @(
            @{
                Name         = 'Sentiment Engine API & Dashboard (Docker Compose)'
                Port         = 8001
                WorkDir      = $ProjectDir
                HealthUrl    = 'http://127.0.0.1:8001/api/v1/health'
                WindowTitle  = 'sentiment-engine [Docker Port 8001]'
                TimeoutSec   = 120
                StartCommand = "docker compose up -d && docker compose logs -f sentiment-engine"
                IsDocker     = $true
            }
        )
    } else {
        throw "Ne calisan bir Python 3.11+ ortami (.venv) ne de calisan bir Docker Desktop bulundu. Lutfen Python 3.11+ kurun veya Docker Desktop'i baslatin."
    }

    $launched = @()
    foreach ($svc in $Services) {
        $titleEscaped = $svc.WindowTitle.Replace("'", "''")
        $dirEscaped = $svc.WorkDir.Replace("'", "''")
        $childScript = @"
try { `$Host.UI.RawUI.WindowTitle = '$titleEscaped' } catch {}
Set-Location -LiteralPath '$dirEscaped'
`$env:PORT = '$($svc.Port)'
Write-Host '========================================' -ForegroundColor Cyan
Write-Host '  $titleEscaped' -ForegroundColor Green
Write-Host '  URL: $OpenUrl' -ForegroundColor Cyan
Write-Host '  Swagger: http://localhost:8001/docs' -ForegroundColor Cyan
Write-Host '  Kapatmak icin Ctrl+C veya pencereyi kapatin.' -ForegroundColor DarkGray
Write-Host '========================================' -ForegroundColor Cyan
$($svc.StartCommand)
"@
        $encoded = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($childScript))
        $proc = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', $encoded) -WorkingDirectory $svc.WorkDir -PassThru
        $pidFile = Join-Path $env:TEMP ("projects-launcher-port-{0}.pid" -f $svc.Port)
        $startTicks = try { $proc.StartTime.ToFileTimeUtc() } catch { 0 }
        Set-Content -LiteralPath $pidFile -Value ("{0}|{1}" -f $proc.Id, $startTicks) -Encoding ASCII
        $launched += @{ Service = $svc; Process = $proc }
        Write-Host "[Baslatildi] $($svc.Name) -> Port $($svc.Port) (Pencere PID: $($proc.Id))" -ForegroundColor Green
    }

    foreach ($item in $launched) {
        $svc = $item.Service
        Write-Host "[Bekleniyor] $($svc.Name) saglik kontrolu ($($svc.HealthUrl))..." -ForegroundColor Cyan
        $wrapper = if ($svc.IsDocker) { $null } else { $item.Process }
        Wait-ServiceReady -Name $svc.Name -Port $svc.Port -HealthUrl $svc.HealthUrl -TimeoutSec $svc.TimeoutSec -WrapperProcess $wrapper
        Write-Host "[Hazir] $($svc.Name) -> $($svc.HealthUrl)" -ForegroundColor Green
    }

    if (-not $NoBrowser) {
        Write-Host "[Tarayici] Aciliyor: $OpenUrl" -ForegroundColor Cyan
        Start-Process $OpenUrl
    }
    exit 0
} catch {
    Write-Host "[HATA] $($_.Exception.Message)" -ForegroundColor Red
    if (-not $NoBrowser -and [Environment]::UserInteractive) {
        try { $null = Read-Host 'Cikmak icin Enter tusuna basin' } catch {}
    }
    exit 1
}
