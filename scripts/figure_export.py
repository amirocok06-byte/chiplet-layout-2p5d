"""Export real-data figures and optionally run the installed scientific QA tools."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def export_figure(fig, output_path, dpi=300, axes=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if axes is not None:
        axes = list(axes)
        # Colorbar tick-label widths can make constrained_layout allocate unequal
        # gutters. Freeze image-grid cells to equal physical geometry before QA.
        if len(axes) > 1:
            fig.canvas.draw()
            groups = {}
            for ax in axes:
                spec = ax.get_subplotspec()
                groups.setdefault(spec.get_gridspec(), []).append((ax, spec))
            fig.set_layout_engine(None)
            for grid, cells in groups.items():
                rows, cols = grid.nrows, grid.ncols
                widths = [ax.get_position().width for ax, _ in cells]
                heights = [ax.get_position().height for ax, _ in cells]
                left = min(ax.get_position().x0 for ax, _ in cells)
                right = max(ax.get_position().x1 for ax, _ in cells)
                bottom = min(ax.get_position().y0 for ax, _ in cells)
                top = max(ax.get_position().y1 for ax, _ in cells)
                width, height = min(widths), min(heights)
                dx = (right - left - width) / max(cols - 1, 1)
                dy = (top - bottom - height) / max(rows - 1, 1)
                for ax, spec in cells:
                    old = ax.get_position()
                    x = left + spec.colspan.start * dx
                    y = top - height - spec.rowspan.start * dy
                    ax.set_position([x, y, width, height])
                    colorbar = getattr(ax.images[0], "colorbar", None) if ax.images else None
                    if colorbar is not None:
                        pos = colorbar.ax.get_position()
                        # Follow its reference panel horizontally; shared bars
                        # keep their span over their original row/column group.
                        colorbar.ax.set_position([pos.x0 + x - old.x0, pos.y0,
                                                  pos.width, pos.height])
    qa_tools = os.environ.get("CHIPLET_FIGURE_QA_TOOLS")
    if qa_tools:
        path = Path(qa_tools) / "audit_panel_alignment.py"
        spec = importlib.util.spec_from_file_location("chiplet_alignment_audit", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.require_matplotlib_panel_alignment(
            fig, axes=list(axes) if axes is not None else None,
            json_out=output_path.with_suffix(".alignment.json"), strict=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    pdf = output_path.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    if qa_tools:
        commands = [
            [sys.executable, str(Path(qa_tools) / "audit_pdf_text.py"),
             str(pdf), "--min-pt", "5", "--json"],
            [sys.executable, str(Path(qa_tools) / "audit_figure_collisions.py"),
             str(pdf), "--json-out", str(output_path.with_suffix(".collision.json"))],
        ]
        results = []
        for cmd in commands:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            results.append({"tool": Path(cmd[1]).stem, "returncode": completed.returncode,
                            "stdout": completed.stdout, "stderr": completed.stderr})
        output_path.with_suffix(".qa.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8")
        failed = [r for r in results if r["returncode"] != 0]
        if failed:
            raise RuntimeError(f"Figure QA failed: {pdf}: {failed}")
