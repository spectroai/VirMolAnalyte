# -*- coding: utf-8 -*-
"""End-to-end smoke test for the Copilot stack (intake → sop → runner → tools).

Runs offline with a stub ``QApplication`` and a mock GUI that mimics just
enough of :class:`ModernVirMolAnalyteGUI` for the runner to walk the plan.
Verifies:

* SOP construction from an :class:`IntakePayload`
* Runner sequencing (sync + async tools)
* The masking batch callback hook fires correctly
"""

from __future__ import annotations

import sys
import traceback
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from virmol_ai.intake import from_text
from virmol_ai.runner import SOPRunner, render_plan_outline, render_run_summary
from virmol_ai.sop import IntakePayload, build_plan


class FakeWidget:
    """Tiny stand-in for QSpinBox/QCheckBox/QLineEdit/QComboBox."""
    def __init__(self, value=None, max_val: int = 100):
        self._value = value
        self._max = max_val
        self._items = []
        self._idx = 0
        self._text = ""

    # Spin / radio / checkbox
    def value(self): return self._value
    def setValue(self, v): self._value = v
    def maximum(self): return self._max
    def isChecked(self): return bool(self._value)
    def setChecked(self, v): self._value = bool(v)

    # Combo
    def count(self): return len(self._items)
    def itemText(self, i): return self._items[i]
    def setCurrentIndex(self, i): self._idx = i
    def currentText(self): return self._items[self._idx] if self._items else ""

    # LineEdit / TextEdit
    def text(self): return self._text
    def setText(self, v): self._text = v
    def toPlainText(self): return self._text
    def setPlainText(self, v): self._text = v


class StubResults:
    def __init__(self, n): self._n = n
    def __len__(self): return self._n
    def iloc(self): raise NotImplementedError


class MockGui:
    """Minimal subset of ModernVirMolAnalyteGUI used by the tools."""

    def __init__(self):
        # peak input
        self.manual_peak_input = FakeWidget(value="")
        self.manual_peak_input._text = ""

        # database
        self.db_combo = FakeWidget()
        self.db_combo._items = [
            "Plant Database (188,478 NPs)",
            "Human Database (217,347 NPs)",
            "Microbial Database (36,427 NPs)",
            "Drug Database",
            "All Database (605,735 NPs)",
        ]
        self.other_db_input = FakeWidget(value="")
        self.other_db_input._text = ""
        self.database1 = None

        # filters
        self.cnf_checkbox = FakeWidget(value=True)
        self.ctnf_checkbox = FakeWidget(value=True)
        self.mw_checkbox = FakeWidget(value=False)
        self.cnf_bias = FakeWidget(value=5)
        self.ctnf_bias = FakeWidget(value=2)
        self.mw_list = FakeWidget(value="")
        self.mw_list._text = "300,400"

        # evaluator
        self.evaluator_css = FakeWidget(value=False)
        self.evaluator_aas = FakeWidget(value=False)
        self.evaluator_fps = FakeWidget(value=False)
        self.evaluator_fpaacs = FakeWidget(value=True)
        self.fpaacs_weights = FakeWidget(value="")
        self.fpaacs_weights._text = "0.2,0.3,0.5"

        # masking
        self.frag_mask_topn_spin = FakeWidget(value=5, max_val=50)

        # results / state
        self.result = None
        self.masking_all_results: List[Dict[str, Any]] = []
        self.masking_fragments_all: List[List[Any]] = []
        self.masking_fragments: List[Any] = []
        self.fusion_results: List[Any] = []
        self._threads_running = set()
        self._masking_batch_callbacks: List[Any] = []
        self._log: List[str] = []

    def _log_evt(self, msg): self._log.append(msg)

    def parse_manual_peak_input_strict(self, text):
        from virmol_gui import ModernVirMolAnalyteGUI
        return ModernVirMolAnalyteGUI.parse_manual_peak_input_strict(self, text)

    def _is_thread_running(self, name): return name in self._threads_running

    def load_database(self):
        self._log_evt("load_database called")
        self.database1 = {"fake": True}

    def load_other_database(self):
        self._log_evt("load_other_database called")
        self.database1 = {"fake_custom": True}

    def start_analysis(self):
        self._log_evt("start_analysis called")
        self._threads_running.add("analysis_thread")
        def _finish():
            self.result = StubResults(7)   # 7 fake hits
            self._threads_running.discard("analysis_thread")
            self._log_evt("analysis_thread finished")
        QTimer.singleShot(50, _finish)

    def run_masking_attribution(self):
        self._log_evt("run_masking_attribution called")
        self._threads_running.add("masking_thread")
        def _finish():
            self.masking_all_results = [{"candidate_rank_index": i, "aas_full": 0.5} for i in range(5)]
            self._threads_running.discard("masking_thread")
            self._log_evt("masking batch finished, firing callback")
            for cb in list(self._masking_batch_callbacks):
                cb(True, None)
            self._masking_batch_callbacks = []
        QTimer.singleShot(80, _finish)

    def register_masking_batch_callback(self, cb):
        self._masking_batch_callbacks.append(cb)

    def extract_masking_fragments(self):
        self._log_evt("extract_masking_fragments called")
        self.masking_fragments_all = [
            [{"fragment_id": 1}], [{"fragment_id": 1}, {"fragment_id": 2}],
        ]
        self.masking_fragments = self.masking_fragments_all[0]

    def run_fragment_fusion_analysis(self):
        self._log_evt("run_fragment_fusion_analysis called")
        self.fusion_results = [{"combo": "a+b"}]

    def run_intra_fragment_fusion(self):
        self._log_evt("run_intra_fragment_fusion called")
        self.fusion_results = [{"combo": "intra"}]

    def export_molecular_analysis_folder(self):
        self._log_evt("export_molecular_analysis_folder called")


def main() -> int:
    app = QApplication(sys.argv)

    # Step 1: parse a paper-style peak text via intake
    report = from_text("172.5 (s), 145.2 (d), 21.3 (q)")
    assert report.ok, f"intake failed: {report.warnings}"
    print(f"intake: {report.n_peaks} peaks -> {report.text!r}")

    # Step 2: build a full pipeline plan
    payload = IntakePayload(
        peak_text=report.text,
        database="plant",
        masking_top_n=5,
        do_masking=True,
        do_fragment_extraction=True,
        do_fusion=True,
        fusion_mode="cross",
        do_export=False,  # skip filesystem write in the smoke test
    )
    plan = build_plan("full_pipeline", payload)
    print()
    print(render_plan_outline(plan))

    # Step 3: run the plan against a mock GUI
    gui = MockGui()

    def on_step_start(rec):
        print(f"  > step {rec.index + 1} start: {rec.step.tool_name}")

    def on_step_done(rec):
        print(f"  + step {rec.index + 1} {rec.status}: "
              f"{(rec.result.summary if rec.result else '-')}")

    finished = {"done": False, "ok": False, "records": []}

    def on_plan_done(ok, records):
        finished.update(done=True, ok=ok, records=records)
        QTimer.singleShot(0, app.quit)

    def on_log(level, msg):
        if level != "info":
            print(f"  [{level}] {msg}")

    runner = SOPRunner(
        gui, plan,
        on_step_start=on_step_start,
        on_step_done=on_step_done,
        on_plan_done=on_plan_done,
        on_log=on_log,
    )
    runner.run_all()

    # Safety net: bail out after 5s
    QTimer.singleShot(5000, app.quit)
    app.exec_()

    print()
    print(render_run_summary(finished["records"]))
    print()
    print(f"plan finished ok={finished['ok']}")
    return 0 if finished["ok"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
