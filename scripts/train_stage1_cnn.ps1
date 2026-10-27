param(
    [string]$Output = 'build/stage1-cnn-first-run',
    [int]$Epochs = 100,
    [int]$Seed = 11
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$pythonExe = Join-Path $repoRoot 'build/stage1-gpu-env/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'Create the isolated GPU environment described in docs/stage1_cnn_linucb_8gb.md first.'
}
$env:PYTHONPATH = Join-Path $repoRoot 'src'
& $pythonExe -m chiplet_layout_experiment.stage1_cnn_train `
    --labels-root (Join-Path $repoRoot 'docs/figures/stage0_same_state') `
    --train-cases Case1 Case3 Case4 Case5 Case6 `
    --validation-cases Case7 --test-cases Case8 Case9 Case10 `
    --device cuda --epochs $Epochs --seed $Seed --output (Join-Path $repoRoot $Output)
if ($LASTEXITCODE -ne 0) { throw "CNN run failed with exit code $LASTEXITCODE" }
