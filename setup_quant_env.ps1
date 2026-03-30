param(
    [string]$PythonCmd = "py",
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $RepoRoot ".venv"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"
$PipPath = Join-Path $VenvPath "Scripts\pip.exe"
$RequirementsPath = Join-Path $RepoRoot "requirements-quant.txt"

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment at $VenvPath"
    & $PythonCmd "-$PythonVersion" -m venv $VenvPath
}

Write-Host "Upgrading pip"
& $PythonPath -m pip install --upgrade pip

Write-Host "Installing quantitative research dependencies"
& $PipPath install -r $RequirementsPath

Write-Host ""
Write-Host "Environment ready."
Write-Host "Note: full qlib model training still requires Microsoft Visual C++ Build Tools"
Write-Host "because qlib.data._libs.rolling must be compiled on this machine."
