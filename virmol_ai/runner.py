# -*- coding: utf-8 -*-
"""Sequential SOP runner for VirMol Copilot (P0).

The runner walks a :class:`Plan` (see :mod:`virmol_ai.sop`) step by step.
Each step invokes a tool from :mod:`virmol_ai.tools`; the tool may complete
synchronously or wait on a Qt worker thread before reporting back.

Design goals:

* **Single-threaded control flow.** All tools run on the Qt main thread via
  ``QTimer.singleShot``; there is no extra QThread in the runner itself.
* **UI-independent.** The runner emits callbacks; the AI Assistant tab
  decides how to render progress, errors, and final summaries.
* **Pause / step / abort.** The runner can be paused between steps, advanced
  one step at a time, or aborted (current async tool is not killed forcibly,
  but its eventual ``on_done`` is ignored).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .sop import Plan, PlanStep
from .tools import TOOL_REGISTRY, ToolResult


# ---------- Public dataclasses --------------------------------------------------

@dataclass
class StepRecord:
    """Per-step outcome as recorded by the runner."""
    index: int
    step: PlanStep
    status: str = "pending"          # pending / running / success / failed / skipped / aborted
    result: Optional[ToolResult] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at


# Callback signatures
StepStartCb    = Callable[[StepRecord], None]
StepDoneCb     = Callable[[StepRecord], None]
PlanDoneCb     = Callable[[bool, List[StepRecord]], None]
LogCb          = Callable[[str, str], None]   # (level, message)


# ---------- Runner --------------------------------------------------------------

class SOPRunner:
    """Sequential plan executor.

    Typical usage::

        runner = SOPRunner(gui, plan,
                           on_step_start=ui.append_running,
                           on_step_done=ui.append_done,
                           on_plan_done=ui.show_summary)
        runner.run_all()
    """

    def __init__(
        self,
        gui: Any,
        plan: Plan,
        *,
        on_step_start: Optional[StepStartCb] = None,
        on_step_done: Optional[StepDoneCb] = None,
        on_plan_done: Optional[PlanDoneCb] = None,
        on_log: Optional[LogCb] = None,
        stop_on_failure: bool = True,
    ) -> None:
        self.gui = gui
        self.plan = plan
        self.on_step_start = on_step_start
        self.on_step_done = on_step_done
        self.on_plan_done = on_plan_done
        self.on_log = on_log
        self.stop_on_failure = stop_on_failure

        self.records: List[StepRecord] = [
            StepRecord(index=i, step=s) for i, s in enumerate(plan.steps)
        ]
        self._cursor: int = 0
        self._mode: str = "idle"        # idle / run_all / single / paused / done / aborted
        self._aborted: bool = False
        self._token: int = 0            # incremented on abort to invalidate pending callbacks
        self._on_done_dispatched: bool = False

    # ------------- Public API ----------------------------------------------

    def run_all(self) -> None:
        """Start (or resume) and advance until the plan finishes or fails."""
        if self._mode == "done":
            return
        self._mode = "run_all"
        self._aborted = False
        self._advance()

    def run_one_step(self) -> None:
        """Run only the next pending step then stop."""
        if self._mode == "done":
            return
        self._mode = "single"
        self._aborted = False
        self._advance()

    def pause(self) -> None:
        """Pause after the currently-running step completes (no forcible kill)."""
        if self._mode in ("idle", "done", "aborted"):
            return
        self._mode = "paused"
        self._log("info", "Runner paused — current step will finish before stopping.")

    def abort(self) -> None:
        """Abort the plan. Pending async results from the current step are ignored."""
        self._aborted = True
        self._mode = "aborted"
        self._token += 1
        rec = self._current_record()
        if rec is not None and rec.status == "running":
            rec.status = "aborted"
            self._emit_done(rec)
        self._finalize(ok=False)

    def skip_current(self) -> None:
        """Mark the current step as skipped and continue (only when paused/idle)."""
        if self._mode == "running":
            self._log("warn", "Cannot skip while a step is running.")
            return
        if self._cursor >= len(self.records):
            return
        rec = self.records[self._cursor]
        rec.status = "skipped"
        self._emit_done(rec)
        self._cursor += 1
        if self._mode == "run_all":
            self._advance()

    @property
    def status(self) -> str:
        return self._mode

    @property
    def cursor(self) -> int:
        return self._cursor

    # ------------- Internals -----------------------------------------------

    def _current_record(self) -> Optional[StepRecord]:
        if 0 <= self._cursor < len(self.records):
            return self.records[self._cursor]
        return None

    def _advance(self) -> None:
        if self._aborted:
            return
        if self._cursor >= len(self.records):
            self._finalize(ok=True)
            return

        rec = self.records[self._cursor]
        if rec.status in ("success", "skipped", "failed", "aborted"):
            self._cursor += 1
            self._advance()
            return

        step = rec.step
        spec = TOOL_REGISTRY.get(step.tool_name)
        if spec is None:
            rec.status = "failed"
            rec.result = ToolResult(
                False,
                f"Unknown tool: {step.tool_name}",
                {},
                "unknown_tool",
            )
            self._emit_done(rec)
            self._handle_failure(rec)
            return

        rec.status = "running"
        rec.started_at = self._now()
        if self.on_step_start is not None:
            try:
                self.on_step_start(rec)
            except Exception as exc:  # noqa: BLE001
                self._log("warn", f"on_step_start raised: {exc}")
        self._log("info", f"▶ {step.label}")

        token_snapshot = self._token

        def _on_tool_done(result: ToolResult) -> None:
            # Ignore stale callbacks from aborted runs
            if token_snapshot != self._token:
                return
            rec.result = result
            rec.finished_at = self._now()
            rec.status = "success" if result.ok else "failed"
            self._emit_done(rec)

            if not result.ok:
                self._handle_failure(rec)
                return

            self._cursor += 1
            if self._mode == "run_all":
                self._advance()
            elif self._mode == "single":
                self._mode = "paused"
            # else: do nothing, paused / aborted

        try:
            spec.fn(self.gui, step.args or {}, _on_tool_done)
        except Exception as exc:  # noqa: BLE001
            rec.result = ToolResult(False, f"tool raised: {exc}", {}, str(exc))
            rec.finished_at = self._now()
            rec.status = "failed"
            self._emit_done(rec)
            self._handle_failure(rec)

    def _handle_failure(self, rec: StepRecord) -> None:
        self._log("error", f"Step {rec.index + 1} failed: {rec.result.summary if rec.result else ''}")
        if self.stop_on_failure:
            self._finalize(ok=False)
        else:
            self._cursor += 1
            if self._mode == "run_all":
                self._advance()

    def _finalize(self, *, ok: bool) -> None:
        if self._on_done_dispatched:
            return
        self._on_done_dispatched = True
        self._mode = "done" if ok else ("aborted" if self._aborted else "done")
        if self.on_plan_done is not None:
            try:
                self.on_plan_done(ok, list(self.records))
            except Exception as exc:  # noqa: BLE001
                self._log("warn", f"on_plan_done raised: {exc}")

    def _emit_done(self, rec: StepRecord) -> None:
        if self.on_step_done is not None:
            try:
                self.on_step_done(rec)
            except Exception as exc:  # noqa: BLE001
                self._log("warn", f"on_step_done raised: {exc}")

    def _log(self, level: str, msg: str) -> None:
        if self.on_log is not None:
            try:
                self.on_log(level, msg)
            except Exception as exc:  # noqa: BLE001
                print(f"on_log raised: {exc}")

    @staticmethod
    def _now() -> float:
        import time
        return time.time()


# ---------- Utilities ----------------------------------------------------------

def render_plan_outline(plan: Plan) -> str:
    """Return a multi-line ASCII outline of the plan for previewing in chat."""
    lines = [f"## Plan: {plan.name}", "", plan.description, ""]
    for i, step in enumerate(plan.steps):
        eta = f"{step.estimated_seconds:.0f}s"
        danger = " [confirm]" if step.danger_level in ("medium", "high") else ""
        lines.append(f"  {i + 1}. **{step.label}** — `{step.tool_name}` · ~{eta}{danger}")
        if step.note:
            lines.append(f"     _{step.note}_")
    lines.append("")
    lines.append(f"Total estimated time: ~{plan.total_eta_seconds:.0f}s")
    return "\n".join(lines)


def render_run_summary(records: List[StepRecord]) -> str:
    """Markdown summary table of step outcomes."""
    lines = ["## Run summary", "", "| # | Step | Status | Notes |", "|---|------|--------|-------|"]
    for rec in records:
        status_icon = {
            "success": "OK",
            "failed": "FAIL",
            "skipped": "skip",
            "aborted": "abort",
            "pending": "pending",
            "running": "...",
        }.get(rec.status, rec.status)
        notes = (rec.result.summary if rec.result else "").replace("|", "/")
        lines.append(
            f"| {rec.index + 1} | {rec.step.label} | {status_icon} | {notes} |"
        )
    return "\n".join(lines)
