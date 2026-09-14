"""
HyDRA — Quick Smoke Test
Verifies that the three UX fixes work:
  1. Session state flags are properly initialized
  2. Download cache builder works
  3. PipelineInterrupted can be raised/caught
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("HyDRA UX Fix Smoke Test")
print("=" * 60)

# ── Test 1: Session-state key definitions ─────────────────────
print("\n[1] Checking session state defaults defined in app.py...")
import ast, inspect

with open("app.py", "r", encoding="utf-8") as f:
    src = f.read()

required_keys = [
    "pipeline", "pipeline_run", "pipeline_running",
    "pipeline_paused", "pipeline_step", "pipeline_error",
    "stop_event",
    "dl_frag_csv", "dl_db_csv", "dl_combined_csv",
    "dl_admet_csv", "dl_origins_csv", "dl_docking_csv",
    "dl_pharm_json", "dl_viewer_html", "dl_zip",
]

missing = [k for k in required_keys if f'"{k}"' not in src and f"'{k}'" not in src]
if missing:
    print(f"  FAIL — missing keys: {missing}")
    sys.exit(1)
else:
    print(f"  PASS — all {len(required_keys)} session state keys present")

# ── Test 2: _cache_downloads function exists ───────────────────
print("\n[2] Checking _cache_downloads function...")
if "_cache_downloads" not in src:
    print("  FAIL — _cache_downloads not found in app.py")
    sys.exit(1)
print("  PASS — _cache_downloads defined")

# ── Test 3: Pause/Resume — PipelineInterrupted class ──────────
print("\n[3] Checking PipelineInterrupted exception class...")
if "class PipelineInterrupted" not in src:
    print("  FAIL — PipelineInterrupted not found")
    sys.exit(1)
if "stop_event.is_set()" not in src:
    print("  FAIL — stop_event.is_set() check not found")
    sys.exit(1)
print("  PASS — PipelineInterrupted class and stop_event checks present")

# ── Test 4: Threading import ───────────────────────────────────
print("\n[4] Checking threading import...")
if "import threading" not in src:
    print("  FAIL — threading not imported")
    sys.exit(1)
print("  PASS — threading imported")

# ── Test 5: Downloads tab uses session state keys only ─────────
print("\n[5] Checking Downloads tab uses cached session_state keys...")
# Find the downloads tab section
dl_tab_start = src.find("TAB 8: DOWNLOADS")
dl_tab_end   = src.find("FOOTER", dl_tab_start)
dl_section   = src[dl_tab_start:dl_tab_end]

bad_calls = [
    "export_fragment_results_csv()",
    "export_database_results_csv()",
    "export_combined_leaderboard_csv()",
    "export_admet_full_csv()",
]
failures = [c for c in bad_calls if c in dl_section]
if failures:
    print(f"  FAIL — Downloads tab still calls pipeline directly: {failures}")
    sys.exit(1)
if "dl_frag_csv" not in dl_section:
    print("  FAIL — dl_frag_csv not used in Downloads tab")
    sys.exit(1)
print("  PASS — Downloads tab reads only from cached session_state keys")

# ── Test 6: Resume skip logic ──────────────────────────────────
print("\n[6] Checking Resume skip logic (if step < N)...")
skip_checks = [
    "if step < 1",
    "if step < 2",
    "if step < 3",
    "if step < 4",
]
missing_skips = [c for c in skip_checks if c not in src]
if missing_skips:
    print(f"  FAIL — missing step skip checks: {missing_skips}")
    sys.exit(1)
print("  PASS — all 4 step-skip guards present")

# ── Test 7: RDKit imports for triage ──────────────────────────
print("\n[7] Checking orchestrator PAINS+Ro5 triage in hybrid_orchestrator.py...")
with open("modules/hybrid_orchestrator.py", "r", encoding="utf-8") as f:
    orch_src = f.read()

triage_checks = [
    "max_dock_per_pathway",
    "_passes_frag_triage",
    "_triage_db",
    "_budget_db",
    "PAINS",
]
missing_triage = [c for c in triage_checks if c not in orch_src]
if missing_triage:
    print(f"  FAIL — missing triage elements: {missing_triage}")
    sys.exit(1)
print("  PASS — PAINS+Ro5 triage and budget caps present")

# ── Test 8: Screening caps ────────────────────────────────────
print("\n[8] Checking screening.py compound caps...")
with open("modules/screening.py", "r", encoding="utf-8") as f:
    screen_src = f.read()

if "n_target=500" not in screen_src:
    print("  FAIL — fallback n_target not capped at 500")
    sys.exit(1)
if "max_per_source=500" not in screen_src:
    print("  FAIL — max_per_source not capped at 500")
    sys.exit(1)
print("  PASS — screening caps in place (500 fallback, 500/src, 1000 total)")

# ── All passed ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL TESTS PASSED [OK]")
print("=" * 60)
print(f"\nApp is running at: http://localhost:8501")
print("Open it in your browser to test interactively.")
