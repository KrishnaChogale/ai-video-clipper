$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot "..\..\venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Shared project Python was not found at $pythonPath. Install dependencies using this folder's requirements.txt."
}
& $pythonPath -m streamlit run (Join-Path $PSScriptRoot "app.py")
