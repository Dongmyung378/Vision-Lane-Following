# Windows CARLA lifecycle for this project. This script does not connect to RPC,
# change firewall rules, or terminate a process it cannot verify as its own.
[CmdletBinding()]
param(
    [ValidateSet('Start', 'Stop', 'Status')]
    [string]$Action = 'Start'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$packageRoot = Join-Path $projectRoot 'carla\CARLA_0.9.16'
$launcherPath = Join-Path $packageRoot 'CarlaUE4.exe'
$shippingPath = Join-Path $packageRoot 'CarlaUE4\Binaries\Win64\CarlaUE4-Win64-Shipping.exe'
$localResults = Join-Path $projectRoot 'results\local'
$statePath = Join-Path $localResults 'carla_server_state.json'
$stdoutPath = Join-Path $localResults 'carla_server_stdout.log'
$stderrPath = Join-Path $localResults 'carla_server_stderr.log'
$rpcPort = 2000

function Assert-PinnedConfiguration {
    try {
        $manifest = Get-Content -LiteralPath (Join-Path $projectRoot 'config\environment.json') -Raw | ConvertFrom-Json
        if ($manifest.carla_version -ne '0.9.16' -or
            ([string]$manifest.windows_package.install_directory).Replace('\', '/') -ne 'carla/CARLA_0.9.16' -or
            $manifest.runtime.rpc_port -ne 2000 -or $manifest.runtime.streaming_port -ne 2001 -or
            $manifest.runtime.map -ne 'Town01' -or $manifest.runtime.quality -ne 'Low' -or
            $manifest.runtime.offscreen -ne $true) {
            throw 'Configuration mismatch.'
        }
    }
    catch {
        throw 'config/environment.json is missing or differs from the pinned Windows lifecycle settings. Update the manifest and lifecycle together before running.'
    }
}

function Get-CreationStamp([datetime]$Value) {
    # CIM uses microseconds, while .NET may expose finer precision. Millisecond
    # normalization lets both probes compare the same recorded creation time.
    return $Value.ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ss.fffZ', [System.Globalization.CultureInfo]::InvariantCulture)
}

function Test-SamePath([string]$Left, [string]$Right) {
    return [string]::Equals($Left, $Right, [System.StringComparison]::OrdinalIgnoreCase)
}

function Test-ExpectedExecutable([string]$Path) {
    return ((Test-SamePath $Path $launcherPath) -or (Test-SamePath $Path $shippingPath))
}

function Get-ListeningPorts {
    $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
    return @($listeners | Where-Object { $_.Port -in @($rpcPort, ($rpcPort + 1)) } | ForEach-Object { $_.Port } | Sort-Object -Unique)
}

function Wait-ProjectTcpListeners([int]$TimeoutSeconds = 45) {
    Write-Output ("Waiting up to {0} seconds for TCP listeners {1} and {2}; RPC is not yet verified." -f $TimeoutSeconds, $rpcPort, ($rpcPort + 1))
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        do {
            $ports = @(Get-ListeningPorts)
            if ($ports -contains $rpcPort -and $ports -contains ($rpcPort + 1)) {
                Write-Output 'Both CARLA TCP listeners are present. Verify RPC, server/API versions, and sensors with the separate connection probe.'
                return
            }
            if ($timer.Elapsed.TotalSeconds -ge $TimeoutSeconds) {
                break
            }
            Start-Sleep -Milliseconds 250
        } while ($timer.Elapsed.TotalSeconds -lt $TimeoutSeconds)
    }
    finally {
        $timer.Stop()
    }
    throw 'Startup timed out waiting for both TCP ports 2000 and 2001. Owned process state and logs were preserved; RPC readiness has not been verified.'
}

function Set-PinnedStartupMap {
    # The Windows Shipping build does not honor the positional map URL. Set the
    # package's own default map; never touch user profiles, AppData, or registry.
    $engineConfigPath = Join-Path $packageRoot 'CarlaUE4\Config\DefaultEngine.ini'
    if (-not (Test-Path -LiteralPath $engineConfigPath -PathType Leaf)) {
        throw 'CARLA package DefaultEngine.ini is missing; refusing to launch with an unverified startup map.'
    }
    $contents = [System.IO.File]::ReadAllText($engineConfigPath, [System.Text.Encoding]::UTF8)
    $mapEntry = [regex]::new('^([ \t]*GameDefaultMap[ \t]*=[ \t]*)([^\r\n]*)(\r?)$', [System.Text.RegularExpressions.RegexOptions]::Multiline -bor [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    $entries = $mapEntry.Matches($contents)
    if ($entries.Count -ne 1) {
        throw 'Expected exactly one GameDefaultMap in package DefaultEngine.ini; custom or ambiguous settings were preserved.'
    }
    $desiredMap = '/Game/Carla/Maps/Town01.Town01'
    $officialDefault = '/Game/Carla/Maps/Town10HD_Opt.Town10HD_Opt'
    $currentMap = $entries[0].Groups[2].Value.Trim()
    if ($currentMap -ceq $desiredMap) {
        Write-Output 'Package startup map: Town01 (already pinned).'
        return
    }
    if ($currentMap -cne $officialDefault) {
        throw 'Package GameDefaultMap differs from the official default and Town01. Custom map settings were preserved; start was refused.'
    }
    $backupPath = Join-Path $localResults 'carla_default_engine_original.ini'
    if (-not (Test-Path -LiteralPath $backupPath)) {
        Copy-Item -LiteralPath $engineConfigPath -Destination $backupPath
    }
    $updated = $mapEntry.Replace($contents, [System.Text.RegularExpressions.MatchEvaluator]{
        param($entry)
        return $entry.Groups[1].Value + $desiredMap + $entry.Groups[3].Value
    }, 1)
    $temporaryPath = Join-Path $packageRoot 'CarlaUE4\Config\DefaultEngine.vlf.pending.ini'
    try {
        [System.IO.File]::WriteAllText($temporaryPath, $updated, [System.Text.UTF8Encoding]::new($false))
        # PowerShell converts ordinary $null to an empty string for a .NET string
        # parameter. NullString passes a real null (no second backup file).
        [System.IO.File]::Replace($temporaryPath, $engineConfigPath, [System.Management.Automation.Language.NullString]::Value)
    }
    finally {
        if (Test-Path -LiteralPath $temporaryPath -PathType Leaf) {
            Remove-Item -LiteralPath $temporaryPath -Force
        }
    }
    Write-Output 'Package startup map: Town01 pinned; original DefaultEngine.ini backed up under results/local.'
}

function Get-ProcessRecord([int]$ProcessNumber) {
    return Get-CimInstance -ClassName Win32_Process -Filter "ProcessId=$ProcessNumber" -OperationTimeoutSec 3
}

function Read-State {
    if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
        return $null
    }
    try {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        if ($state.schemaVersion -ne 1 -or
            -not (Test-SamePath ([string]$state.projectRoot) $projectRoot) -or
            -not (Test-SamePath ([string]$state.packageRoot) $packageRoot) -or
            @($state.processes).Count -lt 1) {
            throw 'Invalid state.'
        }
        foreach ($entry in $state.processes) {
            if ([int]$entry.processId -le 0 -or
                -not (Test-ExpectedExecutable ([string]$entry.executablePath)) -or
                [string]$entry.creationTimeUtc -notmatch '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$') {
                throw 'Invalid process identity.'
            }
        }
        return $state
    }
    catch {
        throw 'The local CARLA state file is invalid. Refusing to start or stop a process using unverified state.'
    }
}

function Write-State($State) {
    $temporaryPath = Join-Path $localResults 'carla_server_state.pending.json'
    $State | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporaryPath -Encoding UTF8
    Move-Item -LiteralPath $temporaryPath -Destination $statePath -Force
}

function Get-StateSnapshot($State) {
    $active = @()
    $missing = @()
    $mismatched = @()
    foreach ($entry in $State.processes) {
        $observed = Get-ProcessRecord ([int]$entry.processId)
        if ($null -eq $observed) {
            $missing += [int]$entry.processId
            continue
        }
        if (-not (Test-SamePath ([string]$observed.ExecutablePath) ([string]$entry.executablePath)) -or
            (Get-CreationStamp $observed.CreationDate) -ne [string]$entry.creationTimeUtc) {
            $mismatched += [int]$entry.processId
            continue
        }
        $active += $entry
    }
    return [pscustomobject]@{ active = @($active); missing = @($missing); mismatched = @($mismatched) }
}

function Show-Status($State) {
    if ($null -eq $State) {
        Write-Output 'CARLA: stopped (no process owned by this script).'
    }
    else {
        $snapshot = Get-StateSnapshot $State
        if ($snapshot.mismatched.Count -gt 0) {
            throw 'Recorded process identity no longer matches. Refusing to treat reused or inaccessible PIDs as owned.'
        }
        if ($snapshot.active.Count -eq 0) {
            Write-Output 'CARLA: stopped (recorded processes have exited; stale state retained until Start or Stop).'
        }
        else {
            Write-Output ("CARLA: running; {0} verified owned process(es)." -f $snapshot.active.Count)
        }
    }
    $ports = @(Get-ListeningPorts)
    if ($ports.Count -gt 0) {
        Write-Output ("Listening project ports: {0}. RPC connection and server version are not verified by this script." -f ($ports -join ', '))
    }
    else {
        Write-Output 'Ports 2000 and 2001: no TCP listeners. RPC connection has not been verified.'
    }
}

function Stop-OwnedServer($State) {
    if ($null -eq $State) {
        Write-Output 'CARLA: already stopped (no owned state).'
        return
    }
    $snapshot = Get-StateSnapshot $State
    if ($snapshot.mismatched.Count -gt 0) {
        throw 'Stop refused: a recorded PID has a different path or creation time. No processes were terminated.'
    }

    # Validate every live identity before stopping any, then pin OS handles to
    # prevent a subsequent PID reuse from targeting an unrelated process.
    $handles = @()
    try {
        foreach ($entry in $snapshot.active) {
            $process = Get-Process -Id ([int]$entry.processId) -ErrorAction SilentlyContinue
            if ($null -eq $process) {
                continue
            }
            $null = $process.Handle
            if ($process.HasExited) {
                $process.Dispose()
                continue
            }
            if ((Get-CreationStamp $process.StartTime) -ne [string]$entry.creationTimeUtc -or
                -not (Test-SamePath $process.MainModule.FileName ([string]$entry.executablePath))) {
                $process.Dispose()
                throw 'Stop refused: process identity changed during validation. No processes were terminated.'
            }
            $handles += [pscustomobject]@{ process = $process; shipping = (Test-SamePath ([string]$entry.executablePath) $shippingPath) }
        }
        # Stop the actual simulator before its small launcher, where both exist.
        foreach ($item in @($handles | Sort-Object -Property shipping -Descending)) {
            $process = $item.process
            if ($process.HasExited) {
                continue
            }
            $closed = $process.CloseMainWindow()
            if ($closed) {
                $null = $process.WaitForExit(1500)
            }
            if (-not $process.HasExited) {
                $process.Kill()
                if (-not $process.WaitForExit(5000)) {
                    throw 'An owned process did not exit within the stop timeout; state was preserved.'
                }
            }
        }
    }
    finally {
        foreach ($item in $handles) {
            $item.process.Dispose()
        }
    }
    # This exact nonrecursive file operation stays inside results/local.
    Remove-Item -LiteralPath $statePath -Force
    Write-Output 'CARLA: stopped; only recorded, verified owned processes were terminated.'
}

function Start-OwnedServer($State) {
    if ($null -ne $State) {
        $snapshot = Get-StateSnapshot $State
        if ($snapshot.mismatched.Count -gt 0) {
            throw 'Start refused: recorded process identity is stale or mismatched. No process was terminated.'
        }
        if ($snapshot.active.Count -gt 0) {
            Write-Output 'CARLA: already running; no duplicate server was launched.'
            Show-Status $State
            return
        }
    }
    if (-not (Test-Path -LiteralPath $launcherPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $shippingPath -PathType Leaf)) {
        throw 'CARLA 0.9.16 Windows package is missing. Extract it to the project carla/CARLA_0.9.16 directory first.'
    }
    $ports = @(Get-ListeningPorts)
    if ($ports.Count -gt 0) {
        throw ("Start refused: port(s) {0} are already occupied. No unowned process will be stopped." -f ($ports -join ', '))
    }
    $null = New-Item -ItemType Directory -Path $localResults -Force
    Set-PinnedStartupMap
    if ($null -ne $State) {
        Remove-Item -LiteralPath $statePath -Force
    }

    $arguments = @('/Game/Carla/Maps/Town01', '-RenderOffScreen', '-quality-level=Low', '-nosound', "-carla-rpc-port=$rpcPort")
    Write-Output 'Starting CARLA 0.9.16 candidate: Town01, hidden, offscreen, Low quality, no sound, RPC port 2000.'
    $launcher = Start-Process -FilePath $launcherPath -ArgumentList $arguments -WorkingDirectory $packageRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    try {
        $null = $launcher.Handle
        $launchStamp = Get-CreationStamp $launcher.StartTime
        $launcherNumber = $launcher.Id
        $state = [pscustomobject]@{
            schemaVersion = 1
            projectRoot = $projectRoot
            packageRoot = $packageRoot
            candidateVersion = '0.9.16'
            serverVersionVerified = $false
            rpcPort = $rpcPort
            map = 'Town01'
            startupMapStrategy = 'package-default-engine-ini'
            arguments = $arguments
            processes = @([pscustomobject]@{
                processId = $launcherNumber
                parentProcessId = $null
                executablePath = $launcherPath
                creationTimeUtc = $launchStamp
            })
        }
        Write-State $state

        # The launcher starts a distinct shipping executable. Capture its real
        # PID/path/creation time while observing only this launch's direct tree.
        $deadline = [datetime]::UtcNow.AddSeconds(15)
        $shippingFound = $false
        do {
            $children = @(Get-CimInstance -ClassName Win32_Process -Filter "ParentProcessId=$launcherNumber" -OperationTimeoutSec 3)
            $parent = Get-ProcessRecord $launcherNumber
            if ($null -ne $parent -and
                ((Get-CreationStamp $parent.CreationDate) -ne $launchStamp -or
                -not (Test-SamePath ([string]$parent.ExecutablePath) $launcherPath))) {
                throw 'Launcher PID was reused before child ownership could be captured; local state was preserved.'
            }
            foreach ($child in $children) {
                $childStamp = Get-CreationStamp $child.CreationDate
                if ((Test-SamePath ([string]$child.ExecutablePath) $shippingPath) -and
                    [string]::CompareOrdinal($childStamp, $launchStamp) -ge 0 -and
                    $child.CreationDate.ToUniversalTime() -le $deadline) {
                    $state.processes += [pscustomobject]@{
                        processId = [int]$child.ProcessId
                        parentProcessId = $launcherNumber
                        executablePath = $shippingPath
                        creationTimeUtc = $childStamp
                    }
                    $shippingFound = $true
                }
            }
            if ($shippingFound) {
                Write-State $state
                break
            }
            Start-Sleep -Milliseconds 250
        } while ([datetime]::UtcNow -lt $deadline)

        if (-not $shippingFound) {
            throw 'No owned CARLA shipping child was found during startup. Local logs and launcher state were preserved for diagnosis.'
        }
        Wait-ProjectTcpListeners
        Show-Status $state
        Write-Output 'Run the separate CARLA connection probe to verify RPC, server/API versions, and RGB sensors.'
    }
    finally {
        $launcher.Dispose()
    }
}

try {
    if ($env:OS -ne 'Windows_NT') {
        throw 'This lifecycle script supports Windows only.'
    }
    Assert-PinnedConfiguration
    $state = Read-State
    switch ($Action) {
        'Start' { Start-OwnedServer $state }
        'Stop' { Stop-OwnedServer $state }
        'Status' { Show-Status $state }
    }
    exit 0
}
catch {
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
    exit 1
}
