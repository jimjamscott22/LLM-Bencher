# Create Desktop Shortcut for LLM-Bencher

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runScriptPath = Join-Path $projectDir "run_llm_bencher.ps1"
$shortcutName = "LLM-Bencher.lnk"
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath $shortcutName

Write-Host "Creating desktop shortcut for LLM-Bencher..." -ForegroundColor Cyan

# Check if run script exists
if (-not (Test-Path $runScriptPath)) {
    Write-Host "Error: Run script '$runScriptPath' not found." -ForegroundColor Red
    Pause
    exit
}

# Create WScript Shell Object
$shell = New-Object -ComObject WScript.Shell

# Create the shortcut
$shortcut = $shell.CreateShortcut($shortcutPath)

# Target: PowerShell.exe
$shortcut.TargetPath = "powershell.exe"

# Arguments: -ExecutionPolicy Bypass -File "path\to\run_llm_bencher.ps1"
# -NoExit keeps the window open so logs can be seen after Ctrl+C
$shortcut.Arguments = "-ExecutionPolicy Bypass -NoExit -File ""$runScriptPath"""

# Working Directory: Project folder
$shortcut.WorkingDirectory = $projectDir

# Icon (using Python icon if available, or default)
$pythonPath = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
if ($pythonPath) {
    $shortcut.IconLocation = "$pythonPath, 0"
}

$shortcut.Save()

Write-Host "Shortcut created on desktop: $shortcutPath" -ForegroundColor Green
Write-Host "You can now run LLM-Bencher directly from your desktop." -ForegroundColor Yellow
Write-Host "Press any key to close this window..."
Pause
