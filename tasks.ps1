# Windows/PowerShell equivalents of the Makefile targets.
# Usage: .\tasks.ps1 <task>   e.g.  .\tasks.ps1 test
param(
    [Parameter(Position = 0)]
    [ValidateSet("install", "lint", "fmt", "test", "test-live")]
    [string]$Task = "test"
)

switch ($Task) {
    "install"   { pip install -e "./core[dev]" }
    "lint"      { ruff check }
    "fmt"       { ruff format }
    "test"      { python -m pytest core/tests -m "not live" }
    "test-live" { python -m pytest core/tests -m live }
}
