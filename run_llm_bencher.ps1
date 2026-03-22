# Run LLM-Bencher
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "Starting LLM-Bencher..." -ForegroundColor Cyan

# Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Error: 'uv' is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install it from: https://docs.astral.sh/uv/" -ForegroundColor Yellow
    Pause
    exit
}

# Run the app using uv
# This ensures it uses the virtual environment managed by uv
uv run python -m llm_bencher
