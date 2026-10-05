[CmdletBinding()]
param(
    [string]$PythonExecutable = '',
    [string]$OutputPath = '',
    [string]$CarlaRoot = '',
    [switch]$Strict
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$doctorPath = Join-Path $PSScriptRoot 'doctor.py'
$prefixArgs = @()

try {
    if ($PythonExecutable) {
        $runner = (Get-Command $PythonExecutable -ErrorAction Stop).Source
    } else {
        $venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
        if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
            $runner = $venvPython
        } elseif (Get-Command py -ErrorAction SilentlyContinue) {
            $runner = (Get-Command py -ErrorAction Stop).Source
            $prefixArgs = @('-3.12')
        } elseif (Get-Command python -ErrorAction SilentlyContinue) {
            $runner = (Get-Command python -ErrorAction Stop).Source
        } else {
            throw 'No Python interpreter found. Install CPython 3.12 x64 or supply -PythonExecutable.'
        }
    }

    $doctorArgs = @()
    if ($OutputPath) { $doctorArgs += @('--output', $OutputPath) }
    if ($CarlaRoot) { $doctorArgs += @('--carla-root', $CarlaRoot) }
    if ($Strict) { $doctorArgs += '--strict' }

    & $runner @prefixArgs $doctorPath @doctorArgs
    exit $LASTEXITCODE
} catch {
    Write-Error 'Environment audit could not start. Check CPython 3.12 x64 or -PythonExecutable. No software was installed.' -ErrorAction Continue
    exit 2
}
