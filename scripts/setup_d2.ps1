[CmdletBinding()]
param([switch]$SkipSimulator)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $projectRoot '.venv\Scripts\python.exe'
$requirements = Join-Path $projectRoot 'requirements-d2.txt'

try {
    if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
        & py -3.12 -m venv (Join-Path $projectRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the project Python 3.12 environment.' }
    }
    & $pythonExecutable -c "import platform,struct,sys; sys.exit(0 if platform.python_implementation()=='CPython' and sys.version_info[:2]==(3,12) and struct.calcsize('P')==8 and platform.machine().lower() in ('amd64','x86_64') else 1)"
    if ($LASTEXITCODE -ne 0) { throw 'The existing .venv must use CPython 3.12 x64. Inspect it before recreating it.' }
    & $pythonExecutable -m pip install --only-binary=:all: --require-hashes -r $requirements
    if ($LASTEXITCODE -ne 0) { throw 'Pinned CARLA API installation failed.' }
    & $pythonExecutable -c "import carla; from importlib.metadata import version; assert version('carla')=='0.9.16'; print('CARLA 0.9.16 native API import: PASS')"
    if ($LASTEXITCODE -ne 0) { throw 'CARLA API import or version verification failed.' }
    if (-not $SkipSimulator) {
        & $pythonExecutable (Join-Path $PSScriptRoot 'install_carla_windows.py')
        if ($LASTEXITCODE -ne 0) { throw 'CARLA Windows package installation failed.' }
    }
    Write-Output 'D2 installation prepared. Start the server and run check_carla_connection.py to verify RPC.'
    exit 0
}
catch {
    Write-Error $_
    exit 1
}
