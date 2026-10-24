"""Verify every run, checkpoint, plotted artifact and fresh figure QA report."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from collections import Counter
import pymupdf


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert len(manifest["runs"]) == manifest["expected_runs"] == 30
    validity = Counter()
    checkpoints = 0
    sums = []
    for relative in manifest["runs"]:
        # Manifest paths are portable even when collected by Windows Python.
        path = root / Path(relative.replace("\\", "/"))
        summary = json.loads(path.read_text(encoding="utf-8"))
        assert summary["status"] == "completed-configured-budget"
        assert summary["checkpoint_steps"] == [0, 200, 400, 600, 800]
        assert summary["evaluator_calls_total"] == 801
        assert summary["diagnostic_evaluator_calls"] == 0
        code_root = Path(__file__).resolve().parents[1] / "src/chiplet_layout_experiment"
        for name, expected_hash in summary["algorithm_source_files"].items():
            assert hashlib.sha256((code_root / name).read_bytes()).hexdigest() == expected_hash
        if summary["source"].get("proven_infeasible_by_area"):
            assert not any(c["metrics"]["valid"] for c in summary["checkpoints"])
        expected_gradient = 800 if summary["algorithm"] == "analytical" else 0
        assert summary["gradient_calls"] == expected_gradient
        rows = [json.loads(line) for line in gzip.decompress(
            (path.parent / "trace.jsonl.gz").read_bytes()).splitlines()]
        assert [r["step"] for r in rows] == list(range(801))
        validity[summary["algorithm"]] += bool(summary["checkpoints"][-1]["metrics"]["valid"])
        for c in summary["checkpoints"]:
            prefix = path.parent / f"checkpoint_{c['step']:04d}"
            for ext in (".png", ".pdf", ".svg", ".json"):
                assert prefix.with_suffix(ext).is_file(), prefix.with_suffix(ext)
            with pymupdf.open(prefix.with_suffix(".pdf")) as pdf:
                text = " ".join(pdf[0].get_text().split())
            displayed = (f"overlap={c['metrics']['overlap_fraction']:.5f} | "
                         f"valid={c['metrics']['valid']}")
            assert displayed in text, (prefix, displayed)
            assert c["map_shape"] == [3, 256, 256]
            relative_error = abs(c["hpwl_map_sum_mm"] - c["metrics"]["hpwl_mm"]) / max(
                c["metrics"]["hpwl_mm"], 1e-12)
            sums.append(relative_error)
            assert relative_error < 1e-6
            checkpoints += 1
        if summary["algorithm"] == "rl":
            impl = summary["implementation"]
            assert impl["training_steps"] == 800
            assert impl["initial_weight_hash"] != impl["final_weight_hash"]
    images = sorted(root.rglob("*.png"))
    assert len(images) == 230, len(images)
    warnings = []
    for png in images:
        # Header dimensions without loading the full potentially large bitmap.
        header = png.read_bytes()[:24]
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        assert width >= 3000 and height >= 900, (png, width, height)
        assert png.with_suffix(".pdf").is_file()
        assert png.with_suffix(".svg").is_file()
        alignment = json.loads(png.with_suffix(".alignment.json").read_text(encoding="utf-8"))
        assert alignment["verdict"] == "PASS", (png, alignment["verdict"])
        collision = json.loads(png.with_suffix(".collision.json").read_text(encoding="utf-8"))
        assert collision["summary"]["fail"] == 0, png
        if collision["summary"]["warn"]:
            warnings.append({"image": str(png), "findings": collision["findings"]})
        reports = json.loads(png.with_suffix(".qa.json").read_text(encoding="utf-8"))
        assert all(r["returncode"] == 0 for r in reports), png
        for report in reports:
            if report["tool"] == "audit_pdf_text":
                glyph = json.loads(report["stdout"])
                assert glyph["below_minimum_count"] == 0, png
    source_root = Path(__file__).resolve().parent
    rendering_sources = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [source_root / name for name in
                  ("render_trajectory_runs.py", "figure_export.py", "plot_grid.py",
                   "plot_snapshot_maps.py", "plot_state_flow.py", "plot_algorithm_effects.py")]
    }
    result = {
        "status": "pass" if not warnings else "review-required",
        "completed_runs": 30, "full_trace_records": 30 * 801,
        "triptych_checkpoints": checkpoints, "raw_maps_in_checkpoints": checkpoints * 3,
        "png_count": len(images), "pdf_count": len(list(root.rglob("*.pdf"))),
        "svg_count": len(list(root.rglob("*.svg"))),
        "all_alignment_reports_pass": True, "all_glyph_reports_pass": True,
        "all_algorithm_source_hashes_match": True,
        "collision_failures": 0, "collision_warnings": warnings,
        "final_legal_counts": dict(validity),
        "max_hpwl_map_sum_relative_error": max(sums),
        "rendering_source_sha256": rendering_sources,
        "static_preflight_notes": [
            "Large research-inspection figures, not resized journal-submission panels.",
            "PNG300dpi plus vectorPDF/SVG; TIFF not requested.",
            "Single-seed individual trajectories, no uncertainty or population superiority claims.",
        ],
    }
    (root / "bundle_audit.json").write_text(json.dumps(result, indent=2) + "\n",
                                           encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2))
