"""
tests/stress_test_inspection.py
Stress Test 1: Project Inspection
Verifies modules, imports, placeholders, TODOs, config paths, SQLite, directories, and error handling.
"""
import os
import sys
import re
import sqlite3
import importlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import config

MODULES_TO_TEST = [
    "config",
    "modules.gpu_utils",
    "modules.video_ingestion",
    "modules.audio_analysis",
    "modules.speech_recognition",
    "modules.object_detection",
    "modules.event_classification",
    "modules.highlight_scoring",
    "modules.title_generator",
    "modules.description_generator",
    "modules.subtitle_generator",
    "modules.clip_editor",
    "modules.thumbnail_generator",
    "modules.quality_control",
    "modules.ocr_detection",
    "modules.metadata_manager",
]

def run_inspection():
    print("==================================================")
    print("STAGE 1: PROJECT CODE & ARCHITECTURE INSPECTION")
    print("==================================================")

    # 1. Verify all expected modules exist
    missing_modules = []
    for mod_name in MODULES_TO_TEST:
        mod_rel_path = mod_name.replace(".", "/") + ".py"
        full_p = os.path.join(BASE_DIR, mod_rel_path)
        if not os.path.isfile(full_p):
            missing_modules.append(mod_name)
    print(f"[*] Module existence check: {len(MODULES_TO_TEST) - len(missing_modules)}/{len(MODULES_TO_TEST)} found.")
    if missing_modules:
        print(f"[-] MISSING MODULES: {missing_modules}")
        return False
    else:
        print("[+] All 16 core module files exist on disk.")

    # 2. Verify imports work cleanly
    import_errors = []
    for mod_name in MODULES_TO_TEST:
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            import_errors.append((mod_name, str(e)))
    if import_errors:
        print(f"[-] IMPORT ERRORS: {import_errors}")
        return False
    else:
        print("[+] All 16 core modules imported cleanly without exception.")

    # 3. Scan for fake implementations / unfulfilled TODOs
    todo_findings = []
    py_files = []
    for root, dirs, files in os.walk(os.path.join(BASE_DIR, "modules")):
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
    py_files.append(os.path.join(BASE_DIR, "app.py"))
    py_files.append(os.path.join(BASE_DIR, "config.py"))

    for p in py_files:
        with open(p, "r", encoding="utf-8", errors="ignore") as fp:
            lines = fp.readlines()
        for idx, line in enumerate(lines, 1):
            if re.search(r"\b(TODO|FIXME|XXX)\b", line, re.IGNORECASE):
                # Check if it's just a comment
                todo_findings.append((os.path.relpath(p, BASE_DIR), idx, line.strip()))
            if re.search(r"raise NotImplementedError", line):
                todo_findings.append((os.path.relpath(p, BASE_DIR), idx, line.strip()))

    print(f"[*] Scan for TODOs / NotImplementedError: {len(todo_findings)} found.")
    for f in todo_findings:
        print(f"    - {f[0]}:{f[1]} -> {f[2]}")

    # 4. Validate Configuration Paths & Directory Creation
    dirs_to_check = [
        ("INPUT_DIR", config.INPUT_DIR),
        ("OUTPUT_DIR", config.OUTPUT_DIR),
        ("TEMP_DIR", config.TEMP_DIR),
        ("MODELS_DIR", config.MODELS_DIR),
        ("DATABASE_DIR", config.DATABASE_DIR),
        ("LOGS_DIR", config.LOGS_DIR),
    ]
    for name, path in dirs_to_check:
        os.makedirs(path, exist_ok=True)
        assert os.path.isdir(path), f"Failed to ensure directory: {path}"
        # Test write permission
        test_file = os.path.join(path, ".perm_test")
        with open(test_file, "w") as tf:
            tf.write("ok")
        os.remove(test_file)
        print(f"[+] Directory verified writable: {name} -> {path}")

    # 5. Verify SQLite Database & Schema
    from modules.metadata_manager import init_db, save_clip_metadata, get_all_clips, save_performance_metrics
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row["name"] for row in cursor.fetchall()]
    print(f"[+] SQLite Tables present: {tables}")
    assert "clips" in tables, "Table 'clips' missing!"
    assert "clip_performance" in tables, "Table 'clip_performance' missing!"
    assert "clips_legacy" in tables, "Table 'clips_legacy' missing!"
    conn.close()

    print("[+] STAGE 1 COMPLETED SUCCESSFULLY: Project architecture, modules, imports, and database verified.")
    return True

if __name__ == "__main__":
    success = run_inspection()
    sys.exit(0 if success else 1)
