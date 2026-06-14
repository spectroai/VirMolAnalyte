#!/usr/bin/env python
# coding: utf-8

import sys
import os
import json
import math
import re
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib
# Set matplotlib backend - use PyQt5 backend
matplotlib.use('Qt5Agg', force=True)
# Completely disable matplotlib's interactive mode and display to prevent pop-up windows
matplotlib.interactive(False)

# Import matplotlib components
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.ioff()  # Turn off pyplot interactive mode

# Set font for matplotlib
plt.rcParams['font.sans-serif'] = ['Arial']  # Use Arial font
plt.rcParams['axes.unicode_minus'] = False   # Fix minus sign display

print("Using Qt5Agg backend")

# PyQt5 imports
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
                             QLabel, QPushButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
                             QCheckBox, QRadioButton, QGroupBox, QTabWidget, QScrollArea,
                             QTableWidget, QTableWidgetItem, QTextEdit, QPlainTextEdit,
                             QFileDialog, QMessageBox, QProgressBar, QApplication, QSizePolicy, QStyle, QDialog,
                             QFrame, QGraphicsDropShadowEffect,
                             QSplashScreen, QMenu, QFormLayout, QDialogButtonBox,
                             QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QEvent, QSize
from PyQt5.QtGui import QFont, QIcon, QPixmap, QImage, QColor, QPainter, QLinearGradient, QPen, QBrush
print("Using PyQt5")


def _app_install_dir():
    """App root: exe directory when frozen, else virmol_gui.py directory."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _toolkit_dir():
    """
    External data/toolkit root.

    Packaged layout:
      VirMolAnalyte-GUI/VirMolAnalyte.exe
      VirMolAnalyte-toolkit/Database, NMR2FP, GUI_result_files, ...

    The toolkit can also be selected explicitly with VIRMOL_TOOLKIT_DIR.
    """
    env_dir = os.environ.get("VIRMOL_TOOLKIT_DIR", "").strip()
    if env_dir:
        return os.path.abspath(env_dir)
    install_dir = _app_install_dir()
    sibling = os.path.abspath(os.path.join(install_dir, os.pardir, "VirMolAnalyte-toolkit"))
    if os.path.isdir(sibling):
        return sibling
    return install_dir


def _configure_runtime_paths():
    toolkit = _toolkit_dir()
    if os.path.isdir(toolkit):
        os.chdir(toolkit)
    return toolkit


def _default_gui_result_dir():
    return os.path.join(_toolkit_dir(), "GUI_result_files")


def _app_icon_path():
    return os.path.join(_default_gui_result_dir(), "logo.ico")


def _app_logo_path():
    """Find the preferred raster logo for splash and in-app branding."""
    install_dir = _app_install_dir()
    toolkit_dir = _toolkit_dir()
    candidates = [
        os.path.join(install_dir, "Logo1.png"),
        os.path.join(toolkit_dir, "Logo1.png"),
        os.path.join(toolkit_dir, "GUI_result_files", "Logo1.png"),
        os.path.join(os.getcwd(), "Logo1.png"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return ""


def _load_app_icon():
    path = _app_icon_path()
    if os.path.isfile(path):
        return QIcon(path)
    return QIcon()


_MANUAL_PEAK_EXAMPLE_TEXT = (
    "131.2, s\n"
    "116.3, d\n"
    "146.1, s\n"
    "144.7, s\n"
    "117.0, d\n"
    "121.1, d\n"
    "36.8, t\n"
    "72.6, t\n"
    "104.6, d\n"
    "75.0, d\n"
    "78.0, d\n"
    "72.0, d\n"
    "75.3, d\n"
    "64.7, t\n"
    "131.1, s\n"
    "141.6, d\n"
    "28.5, t\n"
    "45.4, d\n"
    "24.4, t\n"
    "26.3, t\n"
    "168.8, s\n"
    "72.9, s\n"
    "27.1, q\n"
    "26.4, q"
)


class VirMolSplashScreen(QSplashScreen):
    """Custom splash screen with loading animation"""
    
    def __init__(self):
        # Create a larger pixmap for the splash screen
        pixmap = QPixmap(800, 500)
        pixmap.fill(QColor(248, 249, 250))  # Light gray background
        
        super().__init__(pixmap)
        
        # Set window properties
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.SplashScreen)
        self.setWindowTitle("VirMolAnalyte")
        
        # Load logo image
        self.logo_pixmap = None
        try:
            logo_path = _app_logo_path()
            if os.path.exists(logo_path):
                self.logo_pixmap = QPixmap(logo_path)
                self.logo_pixmap = self.logo_pixmap.scaled(420, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                print(f"Logo loaded successfully: {logo_path}")
            else:
                print(f"Logo file not found: {logo_path}")
        except Exception as e:
            print(f"Failed to load logo: {str(e)}")
        
        # Center the splash screen
        self.center_splash()
        
        # Initialize progress
        self.progress = 0
        self.loading_text = "Loading VirMolAnalyte..."
        
        # Create timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(50)  # Update every 50ms
        
    def center_splash(self):
        """Center the splash screen on the screen"""
        screen = QApplication.desktop().screenGeometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )
    
    def update_progress(self):
        """Update loading progress"""
        self.progress += 1
        if self.progress <= 100:
            self.showMessage(
                f"{self.loading_text} {self.progress}%",
                Qt.AlignBottom | Qt.AlignCenter,
                QColor(54, 75, 82)
            )
        else:
            self.timer.stop()
    
    def drawContents(self, painter):
        """Custom drawing for the splash screen"""
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background gradient
        gradient = QLinearGradient(0, 0, 0, 500)
        gradient.setColorAt(0, QColor(238, 244, 245))
        gradient.setColorAt(1, QColor(215, 226, 229))
        painter.fillRect(self.rect(), gradient)
        
        # Calculate center positions
        center_x = 400  # Half of 800
        center_y = 250  # Half of 500
        
        # Draw logo if available
        if self.logo_pixmap and not self.logo_pixmap.isNull():
            # Center logo horizontally
            logo_x = center_x - self.logo_pixmap.width() // 2
            logo_y = 58
            painter.drawPixmap(logo_x, logo_y, self.logo_pixmap)
            
            # Adjust text positions to be below logo
            text_start_y = logo_y + self.logo_pixmap.height() + 30
        else:
            # If no logo, start text higher
            text_start_y = 100
        
        # Draw title - centered
        painter.setPen(QColor(36, 78, 87))
        painter.setFont(QFont("Arial", 28, QFont.Bold))
        title_text = "VirMolAnalyte"
        title_metrics = painter.fontMetrics()
        title_width = title_metrics.width(title_text)
        painter.drawText(center_x - title_width // 2, text_start_y, title_text)
        
        # Draw subtitle - centered
        painter.setPen(QColor(72, 91, 98))
        painter.setFont(QFont("Arial", 16))
        subtitle_text = "NMR-Guided Natural Product Analysis Platform"
        subtitle_metrics = painter.fontMetrics()
        subtitle_width = subtitle_metrics.width(subtitle_text)
        painter.drawText(center_x - subtitle_width // 2, text_start_y + 40, subtitle_text)
        
        # Draw version - centered
        painter.setFont(QFont("Arial", 12))
        version_text = "Version 2.0"
        version_metrics = painter.fontMetrics()
        version_width = version_metrics.width(version_text)
        painter.drawText(center_x - version_width // 2, text_start_y + 70, version_text)
        
        # Draw loading bar - centered
        bar_width = 400
        bar_height = 8
        bar_x = center_x - bar_width // 2
        bar_y = text_start_y + 120
        
        # Background bar
        painter.setPen(QColor(222, 226, 230))
        painter.setBrush(QColor(222, 226, 230))
        painter.drawRoundedRect(bar_x, bar_y, bar_width, bar_height, 4, 4)
        
        # Progress bar
        progress_width = int(bar_width * self.progress / 100)
        painter.setPen(QColor(47, 111, 106))
        painter.setBrush(QColor(47, 111, 106))
        painter.drawRoundedRect(bar_x, bar_y, progress_width, bar_height, 4, 4)
        
        # Draw loading text - centered
        painter.setPen(QColor(54, 75, 82))
        painter.setFont(QFont("Arial", 14))
        loading_text = f"{self.loading_text} {self.progress}%"
        loading_metrics = painter.fontMetrics()
        loading_width = loading_metrics.width(loading_text)
        painter.drawText(center_x - loading_width // 2, text_start_y + 150, loading_text)
        
        # Draw decorative elements
        # Top border
        painter.setPen(QColor(47, 111, 106))
        painter.setBrush(QColor(47, 111, 106))
        painter.drawRect(0, 0, 800, 3)
        
        # Bottom border
        painter.drawRect(0, 497, 800, 3)

# Import original function modules
sys.path.append(r'.\VirMolAnalyte\VirDBcreator\NMRprediction')
import VirMolAnalyte
from VirMolAnalyte.Filter_evaluator import *
from VirMolAnalyte.NMR1D import *
from VirMolAnalyte.DataPrepare import *

from virmol_ai import (
    build_analysis_context,
    build_fragment_analysis_context,
    build_results_draft,
    chat_completion,
    format_diagnosis_text,
    format_preflight_text,
    has_fragment_data,
    is_llm_configured,
    load_llm_config,
    run_preflight,
    run_screening_diagnosis,
    save_llm_config,
)
from virmol_ai.config import LLMConfig, load_provider_id, normalize_model_name
from virmol_ai.prompts import DISCLAIMER_ZH, build_messages
from virmol_ai.providers import (
    DEFAULT_PROVIDER_ID,
    LLM_PROVIDERS,
    apply_provider,
    get_provider,
    guess_provider_id,
    provider_choices,
)

class MultiLabelDNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiLabelDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc4(x))
        return x

class DatabaseCreationThread(QThread):
    """Background database creation thread"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, smiles_list, output_path=""):
        super().__init__()
        self.smiles_list = smiles_list
        self.output_path = output_path  # Not used in GUI.py approach
    
    def run(self):
        """Run database creation in background - following GUI.py approach"""
        try:
            # Following the exact approach from GUI.py
            self.status_update.emit("The process of creating a database takes some time, please be patient and wait.")
            self.progress_update.emit(20)
            
            # Import and run VirDBGenerator directly (like in GUI.py)
            from VirMolAnalyte.DataPrepare import VirDBGenerator
            self.status_update.emit(f"Processing {len(self.smiles_list)} SMILES strings...")
            self.progress_update.emit(50)
            
            # Ensure we're working in the project root directory
            import os
            original_dir = os.getcwd()
            
            # Get the project root directory (where virmol_gui.py is located)
            project_root = os.path.dirname(os.path.abspath(__file__))
            
            # Change to project root directory
            os.chdir(project_root)
            self.status_update.emit(f"Changed to project root directory: {os.getcwd()}")
            
            # Use the provided output path (which is already absolute)
            output_file = self.output_path
            
            self.status_update.emit(f"Output file path: {output_file}")
            
            try:
                # Run VirDBGenerator with complete file path
                VirDBGenerator(self.smiles_list, output_file)
                
                self.progress_update.emit(100)
                final_path = os.path.abspath(output_file)
                self.status_update.emit(f"Job finish! File created at: {final_path}")
                self.status_update.emit("This file can be imported and used for structural identification.")
                self.finished.emit()
                
            finally:
                # Always restore original directory
                os.chdir(original_dir)
                self.status_update.emit(f"Restored to original directory: {os.getcwd()}")
            
        except Exception as e:
            self.error.emit(str(e))


class AnalysisThread(QThread):
    """Background analysis thread"""
    progress = pyqtSignal(int)
    result_ready = pyqtSignal(object)  # do not override QThread.finished
    error = pyqtSignal(str)
    
    def __init__(self, shifts, ctype, database, filters, evaluator, params):
        super().__init__()
        self.shifts = shifts
        self.ctype = ctype
        self.database = database
        self.filters = filters
        self.evaluator = evaluator
        self.params = params
        self.result_csv_path = self.params.get("result_csv_path")
    
    def run(self):
        try:
            self.progress.emit(10)
            task = Filters_and_evaluators(np.array(self.shifts), np.array(self.ctype), self.database)
            self.progress.emit(30)
            
            # Debug: Print database structure
            print(f"Database keys: {list(self.database.keys())}")
            if 'Vir_shifts' in self.database:
                print(f"Vir_shifts sample: {self.database['Vir_shifts'][:2] if len(self.database['Vir_shifts']) > 0 else 'Empty'}")
            if 'Ctype' in self.database:
                print(f"Ctype sample: {self.database['Ctype'][:2] if len(self.database['Ctype']) > 0 else 'Empty'}")
            
            if "CNF" in self.filters:
                task.CarbonNumFilter(bias=self.params.get('CNFbias', 5))
            self.progress.emit(50)
            
            if "CTNF" in self.filters:
                CarbonTypeNum = [self.ctype.count("q"), self.ctype.count("t"), 
                                self.ctype.count("d"), self.ctype.count("s")]
                task.CarbonTypeNumFilter(CarbonTypeNum=CarbonTypeNum, bias=self.params.get('CTNFbias', 2))
            self.progress.emit(70)
            
            if "MW" in self.filters:
                task.MWFilter(MW=self.params.get('MWlist', [300]), bias=self.params.get('MWbias', 5))
            self.progress.emit(80)
            
            if len(task.database["DBindex"]) == 0:
                self.error.emit("No compounds meet the search criteria!")
                return
            
            if self.evaluator == "FPS":
                task.FPS_evaluator()
                task.ShowTopN(task.FPSscore, TopN=self.params.get('TopN', 80))
                result = task.TOPN
            elif self.evaluator in ["AAS", "CSS"]:
                task.CSS_AAS_evaluator()
                if self.evaluator == "AAS":
                    task.ShowTopN(task.AASscore, TopN=self.params.get('TopN', 80))
                    result = task.TOPN
                else:
                    task.ShowTopN(task.CSSscore, TopN=self.params.get('TopN', 80))
                    result = task.TOPN
            elif self.evaluator == "FPAACS":
                task.FPAACS_evaluator(weights=self.params.get('weights', [0.2, 0.3, 0.5]))
                task.ShowTopN(task.FPAACSscore, TopN=self.params.get('TopN', 80))
                result = task.TOPN
            
            # Debug: Print result structure
            print(f"Result columns: {list(result.columns)}")
            print(f"Result shape: {result.shape}")
            if 'Vir_shifts' in result.columns:
                print(f"Result Vir_shifts sample: {result['Vir_shifts'].iloc[0] if len(result) > 0 else 'Empty'}")
            if 'Ctype' in result.columns:
                print(f"Result Ctype sample: {result['Ctype'].iloc[0] if len(result) > 0 else 'Empty'}")
            
            # Save results
            if self.result_csv_path:
                os.makedirs(os.path.dirname(self.result_csv_path), exist_ok=True)
                result.to_csv(self.result_csv_path, index=False)
            else:
                fallback_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "GUI_result_files",
                )
                os.makedirs(fallback_dir, exist_ok=True)
                result.to_csv(os.path.join(fallback_dir, "Result.csv"), index=False)
            
            self.progress.emit(100)
            self.result_ready.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))


class LLMThread(QThread):
    """Background cloud LLM request (P0 AI assistant)."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, messages, max_tokens=None, reasoning_effort=None):
        super().__init__()
        self.messages = messages
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

    def run(self):
        try:
            text = chat_completion(
                self.messages,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )
            self.finished.emit(text)
        except Exception as e:
            self.error.emit(str(e))


class MaskingAttributionThread(QThread):
    """Random-mask Monte Carlo AAS attribution (background)."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, smiles, vir_shifts, ctype, exp_peaks, candidate_rank_index, options_dict):
        super().__init__()
        self.smiles = smiles
        self.vir_shifts = vir_shifts
        self.ctype = ctype
        self.exp_peaks = exp_peaks
        self.candidate_rank_index = candidate_rank_index
        self.options_dict = options_dict

    def run(self):
        try:
            self.progress.emit(5)
            opts_in = dict(self.options_dict or {})
            mode = str(opts_in.get("mask_mode", "random_fraction") or "random_fraction").strip().lower()

            if mode == "threshold_baseline":
                payload = self._run_threshold_baseline(opts_in)
                self.progress.emit(100)
                self.finished.emit(payload)
                return

            from VirMolAnalyte.masking_aas_attribution import (
                run_masking_attribution,
                MaskingAttributionOptions,
            )

            mc_opts = {k: v for k, v in opts_in.items() if k != "tau_threshold"}
            opts = MaskingAttributionOptions(**mc_opts)
            self.progress.emit(20)
            result = run_masking_attribution(
                self.smiles,
                self.vir_shifts,
                self.exp_peaks,
                options=opts,
                candidate_rank_index=self.candidate_rank_index,
                ctype_raw=self.ctype,
            )
            self.progress.emit(100)
            self.finished.emit(result.to_serializable())
        except Exception as e:
            self.error.emit(str(e))

    def _run_threshold_baseline(self, opts_dict):
        """Compare_CSS-style threshold baseline scoring (deterministic, no Monte Carlo).

        Per-carbon score is τ − |Δδ|, where |Δδ| is the absolute error from greedy peak
        matching. Unmatched carbons (a defensive case) get -inf so the downstream
        extractor (which filters via np.isfinite) excludes them. The returned payload
        mirrors MaskingAttributionResult.to_serializable() to keep all GUI consumers
        (table, structure colouring, fragment extraction, fusion, methodology text)
        compatible without further changes.
        """
        import math
        from rdkit import Chem
        from VirMolAnalyte.masking_aas_attribution import (
            _match_pred_to_exp_pairs,
            carbon_atom_indices_in_shift_order,
            compute_aas,
            normalize_peak_type,
        )

        self.progress.emit(15)
        smiles = str(self.smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        atom_idx_full = carbon_atom_indices_in_shift_order(mol)
        try:
            vir_list = [
                float(x) for x in (
                    self.vir_shifts.tolist()
                    if hasattr(self.vir_shifts, "tolist")
                    else list(self.vir_shifts)
                )
            ]
        except Exception as exc:
            raise ValueError(f"Invalid virtual shifts: {exc}") from exc

        n = min(len(atom_idx_full), len(vir_list))
        if n == 0:
            raise ValueError("No carbon shifts or no carbons in molecule.")
        atom_idx = [int(x) for x in atom_idx_full[:n]]
        vir = vir_list[:n]

        ctype_raw = self.ctype
        if ctype_raw is None:
            lib_types = ["s"] * n
        else:
            try:
                ctype_seq = list(ctype_raw)
            except TypeError:
                ctype_seq = [ctype_raw]
            lib_types = [str(x) if x is not None else "s" for x in ctype_seq[:n]]
            while len(lib_types) < n:
                lib_types.append("s")

        if not self.exp_peaks:
            raise ValueError("No experimental peaks.")

        use_dept = bool(opts_dict.get("use_dept_constraint", False))
        greedy_unique = bool(opts_dict.get("greedy_unique_matching", False))
        try:
            tau = float(opts_dict.get("tau_threshold", 3.0))
        except (TypeError, ValueError):
            tau = 3.0
        if not math.isfinite(tau) or tau <= 0:
            tau = 3.0

        self.progress.emit(40)
        pairs = _match_pred_to_exp_pairs(
            vir,
            lib_types,
            self.exp_peaks,
            use_dept_constraint=use_dept,
            greedy_unique_matching=greedy_unique,
        )
        abs_err_by_pred = [float("inf")] * n
        for p in pairs:
            i = int(p["pred_local_index"])
            if 0 <= i < n:
                abs_err_by_pred[i] = float(p["abs_err"])
        mean_scores = [
            (tau - e) if math.isfinite(e) else float("-inf")
            for e in abs_err_by_pred
        ]

        self.progress.emit(75)
        exp_ppms = [float(p["ppm"]) for p in self.exp_peaks]
        exp_types = [normalize_peak_type(p.get("type")) for p in self.exp_peaks]
        try:
            aas_full = float(
                compute_aas(
                    vir,
                    exp_ppms,
                    lib_types=lib_types,
                    exp_types=exp_types,
                    use_dept_constraint=use_dept,
                    greedy_unique_matching=greedy_unique,
                )
            )
        except Exception:
            aas_full = float("nan")

        self.progress.emit(95)
        return {
            "smiles": smiles,
            "candidate_rank_index": int(self.candidate_rank_index),
            "mean_scores": [float(x) for x in mean_scores],
            "carbon_atom_indices": [int(x) for x in atom_idx],
            "vir_shifts": [float(x) for x in vir],
            "lib_types": [str(x) for x in lib_types],
            "aas_full": aas_full,
            "n_carbons": int(n),
            "n_iterations": 0,
            "mask_fraction": float(opts_dict.get("mask_fraction", 0.0)),
            "mask_mode": "threshold_baseline",
            "connected_mask_size": int(opts_dict.get("connected_mask_size", 0)),
            "tau_threshold": float(tau),
        }


class CollapsibleBox(QWidget):
    """Collapsible parameter component"""
    def __init__(self, title="", parent=None, start_expanded=False):
        super().__init__(parent)
        
        self.toggle_button = QPushButton(title)
        self.toggle_button.setObjectName("CollapsibleHeaderButton")
        self.toggle_button.clicked.connect(self.toggle_content)
        
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(20, 0, 0, 0)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_area)
        
        self.content_area.setVisible(bool(start_expanded))
        
    def toggle_content(self):
        self.content_area.setVisible(not self.content_area.isVisible())
    
    def addWidget(self, widget):
        self.content_layout.addWidget(widget)
    
    def addLayout(self, layout):
        self.content_layout.addLayout(layout)

class ModernVirMolAnalyteGUI(QMainWindow):
    """Modern VirMolAnalyte GUI Interface"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VirMolAnalyte - Molecular Analysis Tool")
        self.setGeometry(100, 100, 1800, 1000)  # Increase window width for English text

        # Ensure GUI auto-generated files are centralized here
        self.gui_output_dir = _default_gui_result_dir()
        os.makedirs(self.gui_output_dir, exist_ok=True)

        # Set application icon (GUI_result_files/logo.ico)
        self.set_app_icon()
        
        # Initialize variables
        self.test = None
        self.database1 = None
        self.result = None
        self.current_plot_type = "peak"  # Current plot type
        self.peak_plot_data = None  # Peak detection plot data
        self.merge_plot_data = None  # Merged plot data
        self.compound_detail_dialog = None
        self._compound_detail_current_row = -1
        self.llm_thread = None
        self._ai_pending_append_disclaimer = True
        
        # 设置样式
        self.setStyleSheet(self.get_modern_style())
        
        # 创建主界面
        self.init_ui()
        
        # 加载模型
        self.load_model()
        
        # 初始化图谱切换按钮状态
        self.update_plot_buttons()
    
    def set_app_icon(self):
        """Set window and application icon from GUI_result_files/logo.ico."""
        try:
            icon_path = _app_icon_path()
            icon = _load_app_icon()
            if icon.isNull():
                print(f"Icon file not found: {icon_path}")
                return
            self.setWindowIcon(icon)
            app = QApplication.instance()
            if app is not None:
                app.setWindowIcon(icon)
            print(f"Application icon set successfully: {icon_path}")
        except Exception as e:
            print(f"Failed to set application icon: {str(e)}")

    def _gui_output_path(self, filename):
        """Absolute path under GUI_result_files."""
        return os.path.join(self.gui_output_dir, filename)

    def _nmr_csv_path(self):
        """
        Preferred NMR csv path in GUI_result_files.
        Falls back to legacy location if needed.
        """
        preferred = self._gui_output_path("NMR-1D.csv")
        if os.path.exists(preferred):
            return preferred
        legacy = os.path.abspath("NMR-1D.csv")
        if os.path.exists(legacy):
            return legacy
        return preferred
    
    def get_modern_style(self):
        """获取现代化样式"""
        return """
        QMainWindow {
            background: #e9eff1;
        }
        
        QWidget {
            font-family: "Segoe UI", "Microsoft YaHei UI", "PingFang SC", sans-serif;
        }

        QMenuBar {
            background: #244e57;
            color: #eef7f7;
            padding: 4px 8px;
            border-bottom: 1px solid #1b3b43;
            font-size: 14px;
        }

        QMenuBar::item {
            padding: 7px 12px;
            border-radius: 4px;
            background: transparent;
        }

        QMenuBar::item:selected {
            background: rgba(255, 255, 255, 0.12);
        }

        QMenu {
            background: #fbfdfd;
            color: #22363d;
            border: 1px solid #bccbd0;
            padding: 6px;
        }

        QMenu::item {
            padding: 7px 28px 7px 18px;
            border-radius: 4px;
        }

        QMenu::item:selected {
            background: #dfecee;
            color: #183036;
        }

        QToolBar {
            background: #f8fbfb;
            border: 1px solid #c9d7db;
            border-radius: 8px;
            spacing: 10px;
            padding: 10px 8px;
            min-width: 64px;
        }

        QToolButton {
            background: #ffffff;
            color: #284950;
            border: 1px solid #c6d5d9;
            border-radius: 9px;
            padding: 8px;
            font-weight: 600;
            min-width: 42px;
            min-height: 42px;
        }

        QToolButton:hover {
            background: #e4eff1;
            border-color: #8fb7bd;
        }

        QToolButton:pressed {
            background: #d6e7e9;
            border-color: #2f6f6a;
        }

        QStatusBar {
            background: #f8fbfb;
            color: #536970;
            border-top: 1px solid #c9d7db;
            font-size: 13px;
        }

        QFrame#BrandHeader {
            border: 1px solid #b8cbd1;
            border-radius: 10px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #163840, stop:0.62 #244e57, stop:1 #2f6f6a);
        }

        QLabel#BrandLogo {
            margin: 0;
            padding: 0;
        }

        QLabel#BrandEyebrow {
            color: #9fe2d9;
            font-size: 13px;
            font-weight: 800;
            letter-spacing: 1px;
            margin: 0;
        }

        QLabel#BrandTitle {
            color: #ffffff;
            font-size: 27px;
            font-weight: 800;
            margin: 0;
        }

        QLabel#BrandSubtitle {
            color: rgba(238, 247, 247, 0.86);
            font-size: 14px;
            font-weight: 500;
            margin: 0;
        }

        QLabel#BrandMeta {
            color: rgba(238, 247, 247, 0.74);
            font-size: 12px;
            font-weight: 500;
            margin: 0;
            max-width: 360px;
        }
        
        QTabWidget::pane {
            border: 1px solid #c9d7db;
            background: #f8fbfb;
            border-radius: 8px;
            margin-top: 3px;
        }
        
        QTabBar::tab {
            background: #dce6e8;
            color: #29464d;
            padding: 13px 24px;
            margin-right: 3px;
            border: 1px solid #b8cbd1;
            border-bottom: none;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            font-weight: bold;
            font-size: 20px;
        }
        
        QTabBar::tab:hover:!selected {
            background: #c9dcdf;
            color: #1f2c33;
            border-color: #91b6bf;
        }
        
        QTabBar::tab:selected {
            background: #244e57;
            color: #f3f8fa;
            border-color: #183840;
            border-bottom: 2px solid #244e57;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 1px solid #c9d7db;
            border-radius: 10px;
             margin-top: 25px;
             padding-top: 30px;
             padding-bottom: 15px;
            background: #fbfdfd;
            color: #3d4852;
             font-size: 18px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
             padding: 0 15px;
            background: #fbfdfd;
            color: #244e57;
             font-size: 20px;
        }
        
        /* 折叠区块标题：中性冷灰，与功能按钮明显区分 */
        QPushButton#CollapsibleHeaderButton {
            text-align: left;
            padding: 14px 18px;
            background: #506a72;
            color: #f6fbfb;
            border: 1px solid #40575f;
            border-radius: 6px;
            font-weight: bold;
            font-size: 17px;
            min-width: 0px;
        }
        
        QPushButton#CollapsibleHeaderButton:hover {
            background: #466169;
            border-color: #344d55;
        }
        
        QPushButton#CollapsibleHeaderButton:pressed {
            background: #3c565e;
        }
        
        /* 主功能按钮：低饱和青灰蓝 */
        QPushButton {
            background: #2f6f6a;
            color: #f2f6f7;
            border: 1px solid #265a56;
            padding: 14px 28px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 17px;
            min-width: 120px;
        }
        
        QPushButton:hover {
            background: #2a635f;
            border-color: #214f4b;
        }
        
        QPushButton:pressed {
            background: #244e57;
        }
        
        QPushButton:disabled {
            background: #d8dde2;
            color: #9aa3ad;
            border-color: #c5ccd4;
        }
        
        /* 主流程强调（开始分析等） */
        QPushButton#PrimaryActionButton {
            background: #244e57;
            border-color: #183840;
            font-weight: bold;
        }
        
        QPushButton#PrimaryActionButton:hover {
            background: #1d444d;
        }
        
        QPushButton#PrimaryActionButton:disabled {
            background: #d8dde2;
            color: #9aa3ad;
            border-color: #c5ccd4;
        }
        
        /* 图标浏览按钮 */
        QPushButton#IconToolButton {
            min-width: 44px;
            max-width: 52px;
            padding: 10px;
        }
        
        /* 与参数同行的次要操作：更小、略浅 */
        QPushButton#RecolorOnlyButton {
            min-width: 0;
            padding: 6px 12px;
            font-size: 13px;
            font-weight: 500;
            background: #6a7a82;
            border: 1px solid #586770;
            color: #f5f7f8;
        }
        
        QPushButton#RecolorOnlyButton:hover {
            background: #5d6d75;
            border-color: #4a5a61;
        }
        
        QPushButton#RecolorOnlyButton:disabled {
            background: #d8dde2;
            color: #9aa3ad;
            border-color: #c5ccd4;
        }
        
        /* 窄条 Submit 等 */
        QPushButton#CompactActionButton {
            min-width: 52px;
            max-width: 90px;
            padding: 10px 14px;
            font-size: 15px;
        }
        
        /* AI Assistant: compact action chips */
        QPushButton#AiCompactButton {
            padding: 5px 8px;
            font-size: 12px;
            min-height: 28px;
            max-height: 32px;
        }
        /* Buttons that trigger cloud LLM calls (highlighted) */
        QPushButton#AiLlmButton {
            padding: 5px 8px;
            font-size: 12px;
            min-height: 28px;
            max-height: 32px;
            background: #8f2d56;
            border-color: #6f2343;
            color: #fff7fb;
            font-weight: bold;
        }
        QPushButton#AiLlmButton:hover {
            background: #a13362;
            border-color: #7b264a;
        }
        QPushButton#AiLlmButton:pressed {
            background: #7c274b;
            border-color: #5f1f39;
        }
        QPushButton#AiLlmButton:disabled {
            background: #d6c7cf;
            border-color: #c3b2bb;
            color: #f8f2f5;
        }
        QGroupBox#AiPanelGroup {
            font-size: 12px;
            font-weight: bold;
            margin-top: 4px;
            padding-top: 14px;
        }
        QGroupBox#AiPanelGroup::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
        }
        
        /* 图谱切换：未选中略浅，选中略深 */
        QPushButton#PlotToggleButton {
            min-width: 150px;
            font-size: 15px;
        }
        
        QPushButton#PlotToggleButton:checked {
            background: #2f4d55;
            border-color: #1f3339;
            color: #f2f6f7;
        }
        
        QPushButton#PlotToggleButton:!checked {
            background: #5a6d75;
            border-color: #4a5a61;
            color: #eef2f4;
        }
        
        QPushButton#PlotToggleButton:!checked:hover {
            background: #4f6169;
        }
        
        QPushButton#PlotToggleButton:disabled {
            background: #d8dde2;
            color: #9aa3ad;
            border-color: #c5ccd4;
        }
        
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
             padding: 12px;
            border: 1px solid #c6d5d9;
             border-radius: 8px;
            background: #ffffff;
            color: #263d44;
             font-size: 18px;
             min-height: 35px;
        }
        
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid #2f6f6a;
            background: #f6fbfb;
        }
        
        QLabel {
            color: #2c434a;
             font-size: 18px;
            font-weight: 500;
             margin: 8px 0;
        }
        
        QProgressBar {
            border: 1px solid #c6d5d9;
            border-radius: 6px;
            text-align: center;
            background: #edf4f5;
            color: #2c434a;
            font-weight: bold;
        }
        
        QProgressBar::chunk {
            background: #2f6f6a;
            border-radius: 4px;
        }
        
        QTableWidget {
            border: 1px solid #c6d5d9;
            border-radius: 6px;
            background: #ffffff;
            alternate-background-color: #f4f8f9;
            gridline-color: #dce8eb;
            selection-background-color: #d8e9e9;
            selection-color: #1d343b;
            font-size: 16px;
        }
        
        QHeaderView::section {
            background: #e4eff1;
            color: #244e57;
            padding: 10px;
            border: none;
            border-bottom: 1px solid #c6d5d9;
            font-weight: bold;
            font-size: 14px;
        }
        
        QCheckBox, QRadioButton {
            color: #3d4852;
            font-size: 18px;
            spacing: 8px;
        }
        """
    
    def init_ui(self):
        """初始化用户界面"""
        # Create professional menu bar
        self.create_menu_bar()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(12)
        main_layout.addWidget(self.create_brand_header())
        
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Add tabs (AI Assistant first, then analysis workflow tabs)
        self.create_ai_assistant_tab()
        self.create_analysis_tab()
        self.create_fragment_analysis_tab()
        self.create_database_tab()
        self.create_other_tool_tab()
        
        # Enhanced status bar
        self.create_enhanced_status_bar()

    def create_brand_header(self):
        """Create a compact academic brand header without changing workflow controls."""
        header = QFrame()
        header.setObjectName("BrandHeader")
        header.setMinimumHeight(112)

        shadow = QGraphicsDropShadowEffect(header)
        shadow.setBlurRadius(22)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(18, 42, 48, 42))
        header.setGraphicsEffect(shadow)

        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 14, 24, 14)
        layout.setSpacing(22)

        logo_label = QLabel()
        logo_label.setObjectName("BrandLogo")
        logo_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        logo_path = _app_logo_path()
        if logo_path:
            pixmap = QPixmap(logo_path)
            if not pixmap.isNull():
                logo_label.setPixmap(pixmap.scaled(420, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        layout.addWidget(logo_label, 0, Qt.AlignVCenter)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        eyebrow = QLabel("VIRMOLANALYTE V2.0")
        eyebrow.setObjectName("BrandEyebrow")
        title = QLabel("NMR-Guided Molecular Analysis Workbench")
        title.setObjectName("BrandTitle")
        subtitle = QLabel("Natural product screening · masking AAS fragment attribution · AI-assisted interpretation")
        subtitle.setObjectName("BrandSubtitle")

        text_layout.addWidget(eyebrow)
        text_layout.addWidget(title)
        text_layout.addWidget(subtitle)
        layout.addLayout(text_layout, 1)

        output_label = QLabel(f"Output: {self.gui_output_dir}")
        output_label.setObjectName("BrandMeta")
        output_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        output_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(output_label, 0, Qt.AlignVCenter)

        return header
    
    def create_menu_bar(self):
        """Create professional menu bar"""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu('&File')
        
        # New Project
        new_action = file_menu.addAction('&New Project')
        new_action.setShortcut('Ctrl+N')
        new_action.setStatusTip('Create a new project')
        new_action.triggered.connect(self.new_project)
        
        # Open Project
        open_action = file_menu.addAction('&Open Project...')
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open an existing project')
        open_action.triggered.connect(self.open_project)
        
        # Save Project
        save_action = file_menu.addAction('&Save Project')
        save_action.setShortcut('Ctrl+S')
        save_action.setStatusTip('Save current project')
        save_action.triggered.connect(self.save_project)
        
        file_menu.addSeparator()
        
        # Import Data
        import_menu = file_menu.addMenu('&Import')
        import_nmr_action = import_menu.addAction('&NMR Data...')
        import_nmr_action.triggered.connect(self.import_nmr_data)
        import_smiles_action = import_menu.addAction('&SMILES File...')
        import_smiles_action.triggered.connect(self.import_smiles_file)
        
        # Export Results
        export_menu = file_menu.addMenu('&Export')
        export_csv_action = export_menu.addAction('&Results to CSV...')
        export_csv_action.triggered.connect(self.export_results_csv)
        export_pdf_action = export_menu.addAction('&Report to PDF...')
        export_pdf_action.triggered.connect(self.export_report_pdf)
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = file_menu.addAction('E&xit')
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        
        # Edit Menu
        edit_menu = menubar.addMenu('&Edit')
        
        # Preferences
        preferences_action = edit_menu.addAction('&Preferences...')
        preferences_action.setStatusTip('Configure application settings')
        preferences_action.triggered.connect(self.show_preferences)
        
        # View Menu
        view_menu = menubar.addMenu('&View')
        
        # Status bar toggle
        self.statusbar_action = view_menu.addAction('&Status Bar')
        self.statusbar_action.setCheckable(True)
        self.statusbar_action.setChecked(True)
        self.statusbar_action.triggered.connect(self.toggle_statusbar)
        
        view_menu.addSeparator()
        
        # Window management
        reset_layout_action = view_menu.addAction('&Reset Layout')
        reset_layout_action.setStatusTip('Reset window layout to default')
        reset_layout_action.triggered.connect(self.reset_window_layout)
        
        # Tools Menu
        tools_menu = menubar.addMenu('&Tools')
        
        # Peak Detection
        peak_detection_action = tools_menu.addAction('&Peak Detection')
        peak_detection_action.setStatusTip('Perform peak detection on NMR data')
        peak_detection_action.triggered.connect(self.run_peak_detection)
        
        # Data Analysis
        data_analysis_action = tools_menu.addAction('&Data Analysis')
        data_analysis_action.setStatusTip('Run molecular analysis')
        data_analysis_action.triggered.connect(self.run_data_analysis)
        
        # Database Management
        db_management_action = tools_menu.addAction('&Database Management')
        db_management_action.setStatusTip('Manage molecular databases')
        db_management_action.triggered.connect(self.show_database_management)
        
        tools_menu.addSeparator()
        
        # Settings
        settings_action = tools_menu.addAction('&Settings...')
        settings_action.setStatusTip('Configure analysis parameters')
        settings_action.triggered.connect(self.show_settings)
        
        # Help Menu
        help_menu = menubar.addMenu('&Help')
        
        # Documentation
        docs_action = help_menu.addAction('&Documentation')
        docs_action.setStatusTip('View application documentation')
        docs_action.triggered.connect(self.show_documentation)
        
        # About
        about_action = help_menu.addAction('&About VirMolAnalyte')
        about_action.setStatusTip('About this application')
        about_action.triggered.connect(self.show_about)

    def _make_toolbar_icon(self, kind):
        """Create consistent flat icons for the docked analysis toolbar."""
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        colors = {
            "new": (QColor("#2f6f6a"), QColor("#e7f4f2")),
            "open": (QColor("#b4822b"), QColor("#fff5df")),
            "save": (QColor("#345f8c"), QColor("#e7f0fb")),
            "peak": (QColor("#5a7a94"), QColor("#edf4f8")),
            "analysis": (QColor("#2f6f6a"), QColor("#e7f4f2")),
            "database": (QColor("#557a8c"), QColor("#edf5f8")),
            "output": (QColor("#8b6f3d"), QColor("#fff8e8")),
            "methods": (QColor("#6b5a8c"), QColor("#f1edf8")),
            "export": (QColor("#2f7d4f"), QColor("#eaf7ef")),
            "prefs": (QColor("#60717a"), QColor("#eef3f4")),
            "docs": (QColor("#345f8c"), QColor("#e9f1fb")),
            "about": (QColor("#1689c7"), QColor("#e5f5fc")),
            "help": (QColor("#1689c7"), QColor("#e5f5fc")),
        }
        accent, fill = colors.get(kind, (QColor("#2f6f6a"), QColor("#eef6f6")))

        painter.setPen(QPen(QColor("#b8cbd1"), 2))
        painter.setBrush(QBrush(fill))
        painter.drawRoundedRect(7, 7, 50, 50, 11, 11)

        painter.setPen(QPen(accent, 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)

        if kind == "new":
            painter.drawRoundedRect(21, 15, 22, 32, 3, 3)
            painter.drawLine(27, 23, 37, 23)
            painter.drawLine(27, 31, 37, 31)
            painter.setPen(QPen(accent, 5, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(32, 36, 32, 47)
            painter.drawLine(26, 41, 38, 41)
        elif kind == "open":
            painter.setBrush(QBrush(QColor("#ffd66e")))
            painter.setPen(QPen(QColor("#d2a139"), 2))
            painter.drawRoundedRect(14, 25, 36, 22, 4, 4)
            painter.drawRoundedRect(17, 19, 17, 10, 3, 3)
            painter.setBrush(QBrush(QColor("#ffe28f")))
            painter.drawRoundedRect(14, 27, 36, 20, 4, 4)
        elif kind == "save":
            painter.setPen(QPen(accent, 3))
            painter.setBrush(QBrush(QColor("#dcecff")))
            painter.drawRoundedRect(16, 14, 32, 36, 4, 4)
            painter.setBrush(QBrush(QColor("#ffffff")))
            painter.drawRect(22, 16, 20, 12)
            painter.setBrush(QBrush(QColor("#345f8c")))
            painter.drawRoundedRect(23, 37, 18, 10, 2, 2)
        elif kind == "peak":
            painter.setPen(QPen(accent, 3, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(14, 43, 50, 43)
            points = [(16, 39), (21, 35), (25, 20), (29, 38), (34, 31), (38, 16), (43, 37), (49, 30)]
            for a, b in zip(points, points[1:]):
                painter.drawLine(a[0], a[1], b[0], b[1])
        elif kind == "analysis":
            painter.setPen(QPen(accent, 3))
            for x in (18, 31, 44):
                painter.drawLine(x, 18, x, 46)
            for y in (18, 31, 44):
                painter.drawLine(18, y, 46, y)
            painter.setBrush(QBrush(accent))
            painter.drawEllipse(26, 26, 10, 10)
        elif kind == "database":
            painter.setPen(QPen(accent, 3))
            painter.setBrush(QBrush(QColor("#e7f2f6")))
            painter.drawEllipse(16, 14, 32, 12)
            painter.drawRect(16, 20, 32, 25)
            painter.drawEllipse(16, 38, 32, 12)
            painter.drawLine(16, 26, 48, 26)
            painter.drawLine(16, 34, 48, 34)
        elif kind == "help":
            painter.setPen(QPen(QColor("#ffffff"), 5, Qt.SolidLine, Qt.RoundCap))
            painter.setBrush(QBrush(accent))
            painter.drawEllipse(12, 12, 40, 40)
            painter.setFont(QFont("Arial", 26, QFont.Bold))
            painter.drawText(QRect(12, 10, 40, 42), Qt.AlignCenter, "?")
        elif kind == "output":
            painter.setPen(QPen(QColor("#d4a94c"), 2))
            painter.setBrush(QBrush(QColor("#ffe08a")))
            painter.drawRoundedRect(15, 23, 34, 22, 4, 4)
            painter.drawRoundedRect(18, 17, 14, 9, 3, 3)
            painter.setPen(QPen(accent, 3, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(32, 29, 32, 39)
            painter.drawLine(27, 34, 32, 39)
            painter.drawLine(37, 34, 32, 39)
        elif kind == "methods":
            painter.setPen(QPen(accent, 3))
            painter.setBrush(QBrush(QColor("#ffffff")))
            painter.drawRoundedRect(19, 14, 28, 36, 4, 4)
            painter.drawLine(25, 24, 40, 24)
            painter.drawLine(25, 31, 40, 31)
            painter.drawLine(25, 38, 35, 38)
            painter.setPen(QPen(QColor("#b4822b"), 3, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(38, 43, 48, 33)
        elif kind == "export":
            painter.setPen(QPen(accent, 3))
            painter.setBrush(QBrush(QColor("#ffffff")))
            painter.drawRoundedRect(18, 16, 28, 34, 4, 4)
            painter.drawLine(25, 25, 39, 25)
            painter.drawLine(25, 32, 39, 32)
            painter.setPen(QPen(accent, 5, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(32, 34, 32, 47)
            painter.drawLine(26, 41, 32, 47)
            painter.drawLine(38, 41, 32, 47)
        elif kind == "prefs":
            painter.setPen(QPen(accent, 4, Qt.SolidLine, Qt.RoundCap))
            painter.drawEllipse(22, 22, 20, 20)
            painter.drawEllipse(29, 29, 6, 6)
            for x1, y1, x2, y2 in ((32, 13, 32, 20), (32, 44, 32, 51), (13, 32, 20, 32), (44, 32, 51, 32),
                                   (19, 19, 24, 24), (40, 40, 45, 45), (45, 19, 40, 24), (24, 40, 19, 45)):
                painter.drawLine(x1, y1, x2, y2)
        elif kind == "docs":
            painter.setPen(QPen(accent, 3))
            painter.setBrush(QBrush(QColor("#ffffff")))
            painter.drawRoundedRect(17, 15, 24, 34, 3, 3)
            painter.drawLine(23, 25, 36, 25)
            painter.drawLine(23, 32, 36, 32)
            painter.drawLine(23, 39, 33, 39)
            painter.setBrush(QBrush(QColor("#d9e8f8")))
            painter.drawRoundedRect(29, 19, 18, 30, 3, 3)
        elif kind == "about":
            painter.setBrush(QBrush(accent))
            painter.setPen(QPen(accent, 3))
            painter.drawEllipse(13, 13, 38, 38)
            painter.setPen(QPen(QColor("#ffffff"), 4, Qt.SolidLine, Qt.RoundCap))
            painter.setFont(QFont("Arial", 25, QFont.Bold))
            painter.drawText(QRect(13, 10, 38, 42), Qt.AlignCenter, "i")

        painter.end()
        return QIcon(pixmap)
    
    def create_toolbar(self):
        """Create professional toolbar"""
        self.toolbar = self.addToolBar('Main Toolbar')
        self.toolbar.setMovable(True)
        self.toolbar.setFloatable(True)
        self.toolbar.setAllowedAreas(Qt.LeftToolBarArea | Qt.RightToolBarArea | Qt.TopToolBarArea)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.toolbar.setIconSize(QSize(36, 36))
        self.addToolBar(Qt.LeftToolBarArea, self.toolbar)
        
        # New Project
        new_action = self.toolbar.addAction('New')
        new_action.setIcon(self._make_toolbar_icon("new"))
        new_action.setToolTip('New Project (Ctrl+N)')
        new_action.triggered.connect(self.new_project)
        
        # Open Project
        open_action = self.toolbar.addAction('Open')
        open_action.setIcon(self._make_toolbar_icon("open"))
        open_action.setToolTip('Open Project (Ctrl+O)')
        open_action.triggered.connect(self.open_project)
        
        # Save Project
        save_action = self.toolbar.addAction('Save')
        save_action.setIcon(self._make_toolbar_icon("save"))
        save_action.setToolTip('Save Project (Ctrl+S)')
        save_action.triggered.connect(self.save_project)
        
        self.toolbar.addSeparator()
        
        # Peak Detection
        peak_action = self.toolbar.addAction('Peak Detection')
        peak_action.setIcon(self._make_toolbar_icon("peak"))
        peak_action.setToolTip('Run Peak Detection')
        peak_action.triggered.connect(self.run_peak_detection)
        
        # Analysis
        analysis_action = self.toolbar.addAction('Analysis')
        analysis_action.setIcon(self._make_toolbar_icon("analysis"))
        analysis_action.setToolTip('Run Data Analysis')
        analysis_action.triggered.connect(self.run_data_analysis)
        
        # Database
        db_action = self.toolbar.addAction('Database')
        db_action.setIcon(self._make_toolbar_icon("database"))
        db_action.setToolTip('Database Management')
        db_action.triggered.connect(self.show_database_management)

        # Output folder
        output_action = self.toolbar.addAction('Output Folder')
        output_action.setIcon(self._make_toolbar_icon("output"))
        output_action.setToolTip('Open GUI_result_files output folder')
        output_action.triggered.connect(self.open_output_folder)
        
        self.toolbar.addSeparator()

        # Copy methods text
        methods_action = self.toolbar.addAction('Copy Methods')
        methods_action.setIcon(self._make_toolbar_icon("methods"))
        methods_action.setToolTip('Copy manuscript-ready methods text')
        methods_action.triggered.connect(self.copy_methodology_to_clipboard)

        # Export analysis package
        export_action = self.toolbar.addAction('Export Results')
        export_action.setIcon(self._make_toolbar_icon("export"))
        export_action.setToolTip('Export results, peaks, structures, and analysis parameters')
        export_action.triggered.connect(self.export_molecular_analysis_folder)

        self.toolbar.addSeparator()

        # Preferences
        prefs_action = self.toolbar.addAction('Preferences')
        prefs_action.setIcon(self._make_toolbar_icon("prefs"))
        prefs_action.setToolTip('AI / LLM preferences')
        prefs_action.triggered.connect(self.show_preferences)

        # Documentation
        docs_action = self.toolbar.addAction('Documentation')
        docs_action.setIcon(self._make_toolbar_icon("docs"))
        docs_action.setToolTip('Open documentation')
        docs_action.triggered.connect(self.show_documentation)
        
        # About
        about_action = self.toolbar.addAction('About')
        about_action.setIcon(self._make_toolbar_icon("about"))
        about_action.setToolTip('About VirMolAnalyte')
        about_action.triggered.connect(self.show_about)
    
    def create_enhanced_status_bar(self):
        """Create enhanced status bar with multiple sections"""
        # Main status message
        self.statusBar().showMessage("Ready")
        
        # Add permanent widgets to status bar
        # Memory usage indicator
        self.memory_label = QLabel("Memory: 0 MB")
        self.statusBar().addPermanentWidget(self.memory_label)
        
        # Progress indicator
        self.progress_indicator = QLabel("")
        self.statusBar().addPermanentWidget(self.progress_indicator)
        
        # Update memory usage periodically
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_usage)
        self.memory_timer.start(5000)  # Update every 5 seconds
    
    # Menu action methods
    def new_project(self):
        """Create new project"""
        reply = QMessageBox.question(self, 'New Project', 
                                   'Are you sure you want to create a new project? Current data will be lost.',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.reset_project()
            self.statusBar().showMessage("New project created")
    
    def open_project(self):
        """Open existing project"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "Project Files (*.vmp)")
        if file_path:
            self.load_project(file_path)
            self.statusBar().showMessage(f"Project loaded: {file_path}")
    
    def save_project(self):
        """Save current project"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "Project Files (*.vmp)")
        if file_path:
            self.save_project_data(file_path)
            self.statusBar().showMessage(f"Project saved: {file_path}")
    
    def import_nmr_data(self):
        """Import NMR data"""
        folder = QFileDialog.getExistingDirectory(self, "Select NMR Data Folder")
        if folder:
            self.c13_input.setText(folder)
            self.statusBar().showMessage(f"NMR data imported from: {folder}")
    
    def import_smiles_file(self):
        """Import SMILES file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Import SMILES File", "", "CSV Files (*.csv)")
        if file_path:
            self.smiles_input.setText(file_path)
            self.statusBar().showMessage(f"SMILES file imported: {file_path}")
    
    def export_results_csv(self):
        """Export results to CSV"""
        if hasattr(self, 'result') and self.result is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "CSV Files (*.csv)")
            if file_path:
                self.result.to_csv(file_path, index=False)
                self.statusBar().showMessage(f"Results exported to: {file_path}")
        else:
            QMessageBox.warning(self, "Warning", "No results to export")

    def open_output_folder(self):
        """Open the GUI output folder from the main toolbar."""
        import subprocess

        folder = getattr(self, "gui_output_dir", "") or os.getcwd()
        try:
            os.makedirs(folder, exist_ok=True)
            if sys.platform.startswith("win"):
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
            self.statusBar().showMessage(f"Opened output folder: {folder}")
        except Exception as exc:
            QMessageBox.warning(self, "Output folder", f"Could not open output folder:\n{exc}")
            self.statusBar().showMessage("Could not open output folder")

    def _resolve_analysis_peaks(self):
        """Return (shifts, ctypes) from manual input or NMR-1D.csv, or None."""
        if hasattr(self, "manual_peak_input"):
            manual_text = self.manual_peak_input.toPlainText().strip()
            if manual_text:
                try:
                    parsed = self.parse_manual_peak_input_strict(manual_text)
                except ValueError:
                    return None
                if parsed is not None:
                    return parsed
        csv_file = self._nmr_csv_path()
        if not os.path.isfile(csv_file):
            return None
        try:
            df = pd.read_csv(csv_file, header=None)
            shifts, ctypes = [], []
            for _, row in df.iterrows():
                try:
                    ppm = float(row.iloc[0])
                    ct = str(row.iloc[1]).strip().lower()
                    if ct in ("q", "t", "d", "s"):
                        shifts.append(ppm)
                        ctypes.append(ct)
                except (ValueError, IndexError):
                    continue
            if shifts:
                return shifts, ctypes
        except Exception:
            return None
        return None

    @staticmethod
    def _format_peak_lines_for_manual_input(shifts, ctypes):
        """Format peaks as '146.1, s' lines for Manual peak input reuse."""
        lines = []
        for ppm, ct in zip(shifts, ctypes):
            try:
                p = float(ppm)
                line_ppm = f"{p:.1f}" if abs(p - round(p, 1)) < 1e-6 else str(p)
            except (TypeError, ValueError):
                line_ppm = str(ppm)
            lines.append(f"{line_ppm}, {str(ct).strip().lower()}")
        return lines

    def _write_nmr_1d_csv_file(self, path, shifts, ctypes):
        """Write peak table (ppm, type, intensity) without header."""
        exp = self.get_experimental_data()
        if exp and len(exp) == len(shifts):
            intensities = [float(d.get("intensity", 1.0)) for d in exp]
        else:
            intensities = [1.0] * len(shifts)
        pd.DataFrame(list(zip(shifts, ctypes, intensities))).to_csv(
            path, index=False, header=False
        )

    def _export_top_structures_grid(self, result_df, path, n_mols=20, n_rows=5, n_cols=4):
        """Save Top-N structures on one image (default 5×4)."""
        n = min(n_mols, len(result_df))
        if n <= 0:
            return False
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.4, n_rows * 2.4))
        axs = np.atleast_2d(axs)
        for slot in range(n_rows * n_cols):
            ax = axs[slot // n_cols, slot % n_cols]
            ax.axis("off")
            if slot >= n:
                continue
            row = result_df.iloc[slot]
            smi = str(row.get("smiles", "") or "").strip()
            mol = Chem.MolFromSmiles(smi) if smi else None
            rank = slot + 1
            db_idx = row.get("DBindex", "")
            try:
                score_txt = f"{float(row.get('score', 0)):.2f}"
            except (TypeError, ValueError):
                score_txt = str(row.get("score", ""))
            legend = f"#{rank}  ID {db_idx}\nscore {score_txt}"
            if mol is not None:
                img = Draw.MolToImage(mol, size=(240, 240), kekulize=True)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Invalid\nSMILES", ha="center", va="center", fontsize=9)
            ax.text(0.5, -0.06, legend, ha="center", va="top", fontsize=8, transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True

    def _build_molecular_analysis_parameters_txt(self, shifts, ctypes):
        """Human-readable analysis snapshot with copy-paste peak block."""
        from datetime import datetime

        lines = [
            "VirMolAnalyte V2.0 — Molecular Analysis export",
            f"Exported at: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "========== Peak list (Manual peak input) ==========",
            "# Paste the lines below into Molecular Analysis → Manual peak input:",
            "",
        ]
        lines.extend(self._format_peak_lines_for_manual_input(shifts, ctypes))
        lines.extend(["", "========== Analysis settings =========="])

        if hasattr(self, "db_combo"):
            lines.append(f"Database selection: {self.db_combo.currentText()}")
        if hasattr(self, "other_db_input"):
            odp = self.other_db_input.text().strip()
            if odp:
                lines.append(f"Other database path: {odp}")

        filters = []
        if getattr(self, "cnf_checkbox", None) and self.cnf_checkbox.isChecked():
            filters.append(f"CNF (bias={self.cnf_bias.value()})")
        if getattr(self, "ctnf_checkbox", None) and self.ctnf_checkbox.isChecked():
            filters.append(f"CTNF (bias={self.ctnf_bias.value()})")
        if getattr(self, "mw_checkbox", None) and self.mw_checkbox.isChecked():
            filters.append(f"MW (list={self.mw_list.text().strip()})")
        lines.append(f"Filters: {', '.join(filters) if filters else 'none'}")

        evaluator = "none"
        if getattr(self, "evaluator_css", None) and self.evaluator_css.isChecked():
            evaluator = "CSS"
        elif getattr(self, "evaluator_aas", None) and self.evaluator_aas.isChecked():
            evaluator = "AAS"
        elif getattr(self, "evaluator_fps", None) and self.evaluator_fps.isChecked():
            evaluator = "FPS"
        elif getattr(self, "evaluator_fpaacs", None) and self.evaluator_fpaacs.isChecked():
            evaluator = f"FPAACS (weights={self.fpaacs_weights.text().strip()})"
        lines.append(f"Evaluator: {evaluator}")
        lines.append("TopN (screening): 80")

        if hasattr(self, "c13_input"):
            lines.extend([
                "",
                "========== Spectral paths (preprocessing) ==========",
                f"13C_NMR: {self.c13_input.text().strip()}",
                f"DEPT90: {self.dept90_input.text().strip()}",
                f"DEPT135: {self.dept135_input.text().strip()}",
                f"Threshold C: {self.threshold_c.value()}",
                f"Threshold C90: {self.threshold_c90.value()}",
                f"Threshold C135pos: {self.threshold_c135pos.value()}",
                f"Threshold C135neg: {self.threshold_c135neg.value()}",
            ])

        peak_src = "manual_peak_input" if (
            hasattr(self, "manual_peak_input") and self.manual_peak_input.toPlainText().strip()
        ) else "NMR-1D.csv"
        lines.append(f"Peak source used for analysis: {peak_src}")
        if self.result is not None:
            lines.append(f"Result rows exported: {len(self.result)}")
        return "\n".join(lines) + "\n"

    def export_molecular_analysis_folder(self):
        """One-click export: results, peaks, Top-20 grid, and parameters txt."""
        import shutil
        from datetime import datetime

        try:
            if self.result is None or len(self.result) == 0:
                QMessageBox.warning(
                    self,
                    "Export",
                    "No analysis results. Run Start Analysis first.",
                )
                return

            peaks = self._resolve_analysis_peaks()
            if peaks is None:
                QMessageBox.warning(
                    self,
                    "Export",
                    "No peak data. Complete preprocessing (NMR-1D.csv) or enter manual peaks.",
                )
                return
            shifts, ctypes = peaks

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(self.gui_output_dir, f"analysis_export_{stamp}")
            os.makedirs(out_dir, exist_ok=True)

            # 1) Result list
            result_path = os.path.join(out_dir, "Result.csv")
            self.result.to_csv(result_path, index=False)

            # 2) Peak file
            nmr_path = os.path.join(out_dir, "NMR-1D.csv")
            src_nmr = self._nmr_csv_path()
            manual_active = (
                hasattr(self, "manual_peak_input")
                and self.manual_peak_input.toPlainText().strip()
            )
            if not manual_active and os.path.isfile(src_nmr):
                shutil.copy2(src_nmr, nmr_path)
            else:
                self._write_nmr_1d_csv_file(nmr_path, shifts, ctypes)

            # 3) Top 20 structures — 5 rows × 4 columns
            grid_path = os.path.join(out_dir, "Top20_structures_5x4.png")
            if not self._export_top_structures_grid(self.result, grid_path):
                QMessageBox.warning(self, "Export", "Could not render structure grid.")
                return

            # 4) Analysis parameters
            params_path = os.path.join(out_dir, "analysis_parameters.txt")
            with open(params_path, "w", encoding="utf-8") as f:
                f.write(self._build_molecular_analysis_parameters_txt(shifts, ctypes))

            readme_path = os.path.join(out_dir, "README.txt")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(
                    "VirMolAnalyte V2.0 analysis export\n"
                    "--------------------------------\n"
                    "Result.csv              — screening hit list\n"
                    "NMR-1D.csv              — experimental peaks (ppm, type, intensity)\n"
                    "Top20_structures_5x4.png — Top 20 structures (5×4 grid)\n"
                    "analysis_parameters.txt — settings + peaks for Manual peak input\n"
                )

            self.statusBar().showMessage(f"Analysis export saved: {out_dir}")
            QMessageBox.information(
                self,
                "Export complete",
                f"Files saved to:\n{out_dir}\n\n"
                "• Result.csv\n"
                "• NMR-1D.csv\n"
                "• Top20_structures_5x4.png\n"
                "• analysis_parameters.txt",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
            self.statusBar().showMessage("Export failed")
    
    def export_report_pdf(self):
        """Export report to PDF"""
        QMessageBox.information(self, "Info", "PDF export functionality will be implemented in future versions")
    
    def _fill_provider_combobox(self, combo, selected_id=None):
        combo.blockSignals(True)
        combo.clear()
        for pid, label in provider_choices():
            combo.addItem(label, pid)
        sid = selected_id or load_provider_id() or guess_provider_id(
            load_llm_config().base_url, load_llm_config().model
        )
        if not sid or sid not in LLM_PROVIDERS:
            sid = DEFAULT_PROVIDER_ID
        idx = combo.findData(sid)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _apply_provider_to_fields(self, provider_id, base_edit, model_combo, *, block_signals=False):
        spec = get_provider(provider_id)
        custom = provider_id == "custom"
        if block_signals:
            base_edit.blockSignals(True)
            model_combo.blockSignals(True)
        base_edit.setReadOnly(not custom)
        model_combo.clear()
        if custom:
            base_edit.setPlaceholderText("https://...")
            model_combo.setEditable(True)
            cfg = load_llm_config()
            base_edit.setText(cfg.base_url)
            model_combo.addItem(cfg.model)
            model_combo.setCurrentText(cfg.model)
        else:
            base, model = apply_provider(provider_id)
            base_edit.setText(base)
            for m in spec.get("models") or [model]:
                model_combo.addItem(m)
            mi = model_combo.findText(model)
            model_combo.setCurrentIndex(mi if mi >= 0 else 0)
            model_combo.setEditable(False)
        if block_signals:
            base_edit.blockSignals(False)
            model_combo.blockSignals(False)

    def _save_llm_from_fields(self, provider_id, key_edit, base_edit, model_combo, temp_spin):
        custom = provider_id == "custom"
        base_saved = base_edit.text().strip().rstrip("/")
        model_saved = model_combo.currentText().strip()
        if not custom:
            base_saved, model_saved = apply_provider(
                provider_id, model_override=model_saved or None
            )
        model_saved = normalize_model_name(model_saved, base_saved)
        new_cfg = LLMConfig(
            api_key=key_edit.text().strip(),
            base_url=base_saved,
            model=model_saved,
            temperature=float(temp_spin.value()),
        )
        save_llm_config(new_cfg, provider_id=provider_id)
        self._refresh_ai_status_label()
        if hasattr(self, "ai_provider_combo"):
            self._fill_provider_combobox(self.ai_provider_combo, provider_id)

    def show_preferences(self):
        """LLM API preferences (P0) — pick provider, only paste API Key."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Preferences — AI / LLM")
        dlg.setMinimumWidth(520)
        form = QFormLayout(dlg)
        cfg = load_llm_config()

        provider_combo = QComboBox()
        self._fill_provider_combobox(provider_combo)

        key_edit = QLineEdit(cfg.api_key)
        key_edit.setEchoMode(QLineEdit.Password)
        key_edit.setPlaceholderText("Paste key from provider console, or set VIRMOL_LLM_API_KEY")

        base_edit = QLineEdit()
        model_combo = QComboBox()

        def _on_provider_changed(_index=None):
            pid = provider_combo.currentData()
            if pid:
                self._apply_provider_to_fields(pid, base_edit, model_combo, block_signals=True)

        provider_combo.currentIndexChanged.connect(_on_provider_changed)
        _on_provider_changed()

        temp_spin = QDoubleSpinBox()
        temp_spin.setRange(0.0, 1.5)
        temp_spin.setSingleStep(0.1)
        temp_spin.setValue(float(cfg.temperature))

        form.addRow("Provider (required):", provider_combo)
        form.addRow("API Key:", key_edit)
        form.addRow("Model:", model_combo)
        form.addRow("API base URL:", base_edit)
        form.addRow("Temperature:", temp_spin)
        hint = QLabel(
            "Select a provider and paste your API Key; base URL and model are filled automatically.\n"
            "DeepSeek keys: https://platform.deepseek.com/api_keys"
        )
        hint.setWordWrap(True)
        hint.setOpenExternalLinks(True)
        hint.setStyleSheet("color: #6c757d; font-size: 12px;")
        form.addRow(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec_() != QDialog.Accepted:
            return
        pid = provider_combo.currentData() or DEFAULT_PROVIDER_ID
        self._save_llm_from_fields(pid, key_edit, base_edit, model_combo, temp_spin)
        self.statusBar().showMessage("LLM settings saved")
    
    def toggle_toolbar(self, checked):
        """Toggle toolbar visibility"""
        toolbar = getattr(self, "toolbar", None)
        if toolbar is not None:
            toolbar.setVisible(checked)
    
    def toggle_statusbar(self, checked):
        """Toggle status bar visibility"""
        self.statusBar().setVisible(checked)
    
    def reset_window_layout(self):
        """Reset window layout to default"""
        self.resize(1800, 1000)
        self.move(100, 100)
        self.statusBar().showMessage("Window layout reset")
    
    def run_peak_detection(self):
        """Run peak detection from menu"""
        self.submit_peaks()
    
    def run_data_analysis(self):
        """Run data analysis from menu"""
        if hasattr(self, 'database1') and self.database1 is not None:
            self.start_analysis()
        else:
            QMessageBox.warning(self, "Warning", "Please load a database first")
    
    def show_database_management(self):
        """Show database management tab"""
        self.tab_widget.setCurrentIndex(1)
        self.statusBar().showMessage("Database management tab activated")
    
    def show_settings(self):
        """Show settings dialog"""
        QMessageBox.information(self, "Settings", "Settings dialog will be implemented in future versions")
    
    def show_documentation(self):
        """Open built user manual (website/site/index.html) in the default browser."""
        import webbrowser
        from pathlib import Path

        doc_index = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "website",
            "site",
            "index.html",
        )
        if os.path.isfile(doc_index):
            webbrowser.open(Path(doc_index).as_uri())
            self.statusBar().showMessage(f"Opened documentation: {doc_index}")
            return
        QMessageBox.information(
            self,
            "Documentation",
            "User manual site not found.\n\n"
            "Build it first:\n"
            "  cd website\n"
            "  python docs.py\n\n"
            "Then use Help → Documentation again.\n\n"
            "Live preview: python docs.py serve",
        )
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About VirMolAnalyte", 
                         "VirMolAnalyte v2.0\n\n"
                         "A professional molecular analysis tool for NMR spectroscopy.\n\n"
                         "© 2024 VirMolAnalyte Team")
    
    def show_help(self):
        """Open user manual (same as Documentation)."""
        self.show_documentation()
    
    def reset_project(self):
        """Reset project data"""
        self.test = None
        self.database1 = None
        self.result = None
        self.peak_plot_data = None
        self.merge_plot_data = None
        
        # Clear inputs
        if hasattr(self, 'c13_input'):
            self.c13_input.clear()
        if hasattr(self, 'dept90_input'):
            self.dept90_input.clear()
        if hasattr(self, 'dept135_input'):
            self.dept135_input.clear()
        if hasattr(self, 'smiles_input'):
            self.smiles_input.clear()
        
        # Clear results
        if hasattr(self, 'results_table'):
            self.results_table.setRowCount(0)
        if hasattr(self, 'peak_info_table'):
            self.peak_info_table.setRowCount(0)
        
        # Reset plot
        if hasattr(self, 'figure'):
            self.figure.clear()
            self.canvas.draw()
        
        self.statusBar().showMessage("Project reset completed")
    
    def load_project(self, file_path):
        """Load project from file"""
        try:
            # This is a placeholder for project loading functionality
            QMessageBox.information(self, "Info", f"Loading project from: {file_path}\nProject loading will be implemented in future versions")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")
    
    def save_project_data(self, file_path):
        """Save project data to file"""
        try:
            # This is a placeholder for project saving functionality
            QMessageBox.information(self, "Info", f"Project saved to: {file_path}\nProject saving will be implemented in future versions")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project: {str(e)}")
    
    def update_memory_usage(self):
        """Update memory usage display"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_label.setText(f"Memory: {memory_mb:.1f} MB")
        except ImportError:
            self.memory_label.setText("Memory: N/A")
    
    def create_analysis_tab(self):
        """创建分析标签页 - 左侧参数输入，右侧结果显示"""
        analysis_widget = QWidget()
        analysis_layout = QHBoxLayout(analysis_widget)
        analysis_layout.setContentsMargins(10, 10, 10, 10)
        analysis_layout.setSpacing(15)
        
        # 左侧参数输入区域
        left_panel = QWidget()
        left_panel.setMaximumWidth(650)  # 增加左侧面板宽度
        left_layout = QVBoxLayout(left_panel)
        
        # Spectrum preprocessing: spectral input + peak detection + impurity removal
        spectrum_prep_box = CollapsibleBox("Spectrum Preprocessing", start_expanded=False)

        # --- Spectral Data Input (inside preprocessing) ---
        step1_widget = QWidget()
        step1_layout = QGridLayout(step1_widget)
        
        self.c13_input = QLineEdit()
        self.c13_input.setText("test\\NMRsample\\4\\pdata\\1")
        self.c13_browse_btn = QPushButton()
        self.c13_browse_btn.setObjectName("IconToolButton")
        self.c13_browse_btn.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.c13_browse_btn.setToolTip("Browse 13C_NMR Data Folder")
        self.c13_browse_btn.clicked.connect(lambda: self.browse_folder(self.c13_input))
        
        self.dept90_input = QLineEdit()
        self.dept90_input.setText("test\\NMRsample\\8\\pdata\\1")
        self.dept90_browse_btn = QPushButton()
        self.dept90_browse_btn.setObjectName("IconToolButton")
        self.dept90_browse_btn.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.dept90_browse_btn.setToolTip("Browse DEPT90 Data Folder")
        self.dept90_browse_btn.clicked.connect(lambda: self.browse_folder(self.dept90_input))
        
        self.dept135_input = QLineEdit()
        self.dept135_input.setText("test\\NMRsample\\6\\pdata\\1")
        self.dept135_browse_btn = QPushButton()
        self.dept135_browse_btn.setObjectName("IconToolButton")
        self.dept135_browse_btn.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.dept135_browse_btn.setToolTip("Browse DEPT135 Data Folder")
        self.dept135_browse_btn.clicked.connect(lambda: self.browse_folder(self.dept135_input))
        
        step1_layout.addWidget(QLabel("13C_NMR:"), 0, 0)
        step1_layout.addWidget(self.c13_input, 0, 1)
        step1_layout.addWidget(self.c13_browse_btn, 0, 2)
        step1_layout.addWidget(QLabel("DEPT90:"), 1, 0)
        step1_layout.addWidget(self.dept90_input, 1, 1)
        step1_layout.addWidget(self.dept90_browse_btn, 1, 2)
        step1_layout.addWidget(QLabel("DEPT135:"), 2, 0)
        step1_layout.addWidget(self.dept135_input, 2, 1)
        step1_layout.addWidget(self.dept135_browse_btn, 2, 2)

        spectral_input_group = QGroupBox("Spectral Data Input")
        spectral_input_group_l = QVBoxLayout(spectral_input_group)
        spectral_input_group_l.addWidget(step1_widget)
        spectrum_prep_box.addWidget(spectral_input_group)

        # --- Peak Detection Parameters ---
        step2_widget = QWidget()
        step2_layout = QGridLayout(step2_widget)
        
        self.threshold_c = QDoubleSpinBox()
        self.threshold_c.setRange(0, 1e10)
        self.threshold_c.setValue(3000000.0)
        self.threshold_c.setDecimals(0)
        
        self.threshold_c90 = QDoubleSpinBox()
        self.threshold_c90.setRange(0, 1e10)
        self.threshold_c90.setValue(90000000.0)
        self.threshold_c90.setDecimals(0)
        
        self.threshold_c135pos = QDoubleSpinBox()
        self.threshold_c135pos.setRange(0, 1e10)
        self.threshold_c135pos.setValue(50000000.0)
        self.threshold_c135pos.setDecimals(0)
        
        self.threshold_c135neg = QDoubleSpinBox()
        self.threshold_c135neg.setRange(-1e10, 0)
        self.threshold_c135neg.setValue(-50000000.0)
        self.threshold_c135neg.setDecimals(0)
        
        step2_layout.addWidget(QLabel("C:"), 0, 0)
        step2_layout.addWidget(self.threshold_c, 0, 1)
        step2_layout.addWidget(QLabel("C90:"), 0, 2)
        step2_layout.addWidget(self.threshold_c90, 0, 3)
        step2_layout.addWidget(QLabel("C135pos:"), 1, 0)
        step2_layout.addWidget(self.threshold_c135pos, 1, 1)
        step2_layout.addWidget(QLabel("C135neg:"), 1, 2)
        step2_layout.addWidget(self.threshold_c135neg, 1, 3)
        
        # 设置列宽比例，让输入框更宽
        step2_layout.setColumnStretch(1, 2)  # 第一行第二列（threshold_c）
        step2_layout.setColumnStretch(3, 2)  # 第一行第四列（threshold_c90）
        step2_layout.setColumnStretch(1, 2)  # 第二行第二列（threshold_c135pos）
        step2_layout.setColumnStretch(3, 2)  # 第二行第四列（threshold_c135neg）
        
        # Peak detection buttons
        button_layout = QHBoxLayout()
        self.submit_btn = QPushButton("Submit")
        self.submit_btn.clicked.connect(self.submit_peaks)
        self.combine_btn = QPushButton("Merge")
        self.combine_btn.clicked.connect(self.combine_data)
        self.combine_btn.setEnabled(False)
        
        button_layout.addWidget(self.submit_btn)
        button_layout.addWidget(self.combine_btn)
        button_layout.addStretch()
        
        step2_layout.addLayout(button_layout, 2, 0, 1, 4)

        peak_detect_group = QGroupBox("Peak Detection Parameters")
        peak_detect_group_l = QVBoxLayout(peak_detect_group)
        peak_detect_group_l.addWidget(step2_widget)
        spectrum_prep_box.addWidget(peak_detect_group)

        # --- Impurity Signal Removal ---
        step3_widget = QWidget()
        step3_layout = QVBoxLayout(step3_widget)
        
        # Solvent removal section
        solvent_group = QGroupBox("Solvent Removal")
        solvent_layout = QGridLayout(solvent_group)
        
        self.solvent_combo = QComboBox()
        self.solvent_combo.addItems(["Chloroform", "Methanol", "DMSO", "Pyridine"])
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["CH3", "CH2", "CH", "C"])
        
        self.solvent_submit_btn = QPushButton("Submit")
        self.solvent_submit_btn.setObjectName("CompactActionButton")
        self.solvent_submit_btn.clicked.connect(self.submit_solvent_removal)
        self.solvent_submit_btn.setEnabled(False)
        self.solvent_submit_btn.setFixedWidth(50)  # Set fixed compact width
        
        solvent_layout.addWidget(QLabel("Solvent:"), 0, 0)
        solvent_layout.addWidget(self.solvent_combo, 0, 1)
        solvent_layout.addWidget(QLabel("Type:"), 0, 2)
        solvent_layout.addWidget(self.type_combo, 0, 3)
        solvent_layout.addWidget(self.solvent_submit_btn, 0, 4)
        
        # Set column stretch to make submit button narrower
        solvent_layout.setColumnStretch(4, 0)  # Don't stretch submit button column
        
        step3_layout.addWidget(solvent_group)
        
        # Impurity removal section
        impurity_group = QGroupBox("Impurity Removal")
        impurity_layout = QGridLayout(impurity_group)
        
        self.impurity_type_combo = QComboBox()
        self.impurity_type_combo.addItems(["CH3", "CH2", "CH", "C"])
        
        self.impurity_threshold1 = QLineEdit()
        self.impurity_threshold1.setPlaceholderText("Threshold 1")
        
        self.impurity_threshold2 = QLineEdit()
        self.impurity_threshold2.setText("1e7")
        
        self.impurity_submit_btn = QPushButton("Submit")
        self.impurity_submit_btn.setObjectName("CompactActionButton")
        self.impurity_submit_btn.clicked.connect(self.submit_impurity_removal)
        self.impurity_submit_btn.setEnabled(False)
        self.impurity_submit_btn.setFixedWidth(50)  # Set fixed compact width
        
        impurity_layout.addWidget(QLabel("Type:"), 0, 0)
        impurity_layout.addWidget(self.impurity_type_combo, 0, 1)
        impurity_layout.addWidget(QLabel("Threshold:"), 0, 2)
        impurity_layout.addWidget(self.impurity_threshold1, 0, 3)
        impurity_layout.addWidget(self.impurity_threshold2, 0, 4)
        impurity_layout.addWidget(self.impurity_submit_btn, 0, 5)
        
        # Set column stretch to make submit button narrower
        impurity_layout.setColumnStretch(5, 0)  # Don't stretch submit button column
        
        step3_layout.addWidget(impurity_group)

        impurity_signal_group = QGroupBox("Impurity Signal Removal")
        impurity_signal_group_l = QVBoxLayout(impurity_signal_group)
        impurity_signal_group_l.addWidget(step3_widget)
        spectrum_prep_box.addWidget(impurity_signal_group)

        left_layout.addWidget(spectrum_prep_box)

                # Step 5: Database Analysis
        db_box = CollapsibleBox("Database Analysis")
        step5_widget = QWidget()
        step5_layout = QVBoxLayout(step5_widget)
        
        manual_peak_label = QLabel("Manual peak input (optional)")
        manual_peak_hint = QLabel(
            "One peak per line: ppm, then comma, then carbon type q / t / d / s. "
            "If provided, analysis uses this list and does not require NMR-1D.csv or prior spectral steps."
        )
        manual_peak_hint.setWordWrap(True)
        manual_peak_hint.setStyleSheet("color: #6c757d; font-size: 12px;")
        manual_peak_btn_row = QHBoxLayout()
        self.manual_peak_example_btn = QPushButton("Use Example Case")
        self.manual_peak_example_btn.setToolTip(
            "Fill the box with a demo peak list. Leave empty to use NMR-1D.csv from preprocessing."
        )
        self.manual_peak_example_btn.clicked.connect(self._fill_manual_peak_example)
        manual_peak_btn_row.addWidget(self.manual_peak_example_btn)
        manual_peak_btn_row.addStretch()
        self.manual_peak_input = QPlainTextEdit()
        self.manual_peak_input.setPlaceholderText("ppm, q / t / d / s — one peak per line")
        self.manual_peak_input.setMinimumHeight(110)
        self.manual_peak_input.setMaximumHeight(200)
        step5_layout.addWidget(manual_peak_label)
        step5_layout.addWidget(manual_peak_hint)
        step5_layout.addLayout(manual_peak_btn_row)
        step5_layout.addWidget(self.manual_peak_input)
        
        # Database Selection
        db_select_box = CollapsibleBox("Database Selection")
        db_select_widget = QWidget()
        db_select_layout = QVBoxLayout(db_select_widget)
        
        # Database selection dropdown
        db_select_layout.addWidget(QLabel("Select Database:"))
        self.db_combo = QComboBox()
        self.db_combo.addItems([
            "Plant Database (188,478 NPs)",
            "Human Database (217,347 NPs)", 
            "Microbial Database (36,427 NPs)",
            "Drug Database",
            "All Database (605,735 NPs)"
        ])
        self.db_combo.setCurrentIndex(0)  # Default to Plant Database
        db_select_layout.addWidget(self.db_combo)
        
        # Other Database
        other_db_layout = QHBoxLayout()
        self.other_db_input = QLineEdit()
        self.other_db_input.setPlaceholderText("Select Other Database File")
        self.other_db_browse_btn = QPushButton()
        self.other_db_browse_btn.setObjectName("IconToolButton")
        self.other_db_browse_btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogStart))
        self.other_db_browse_btn.setToolTip("Browse Other Database")
        self.other_db_browse_btn.clicked.connect(lambda: self.browse_file(self.other_db_input, "NPZ files (*.npz)"))
        
        other_db_layout.addWidget(self.other_db_input)
        other_db_layout.addWidget(self.other_db_browse_btn)
        
        # Database load buttons
        db_load_layout = QHBoxLayout()
        self.load_db_btn = QPushButton("Load Database")
        self.load_db_btn.clicked.connect(self.load_database)
        self.load_other_db_btn = QPushButton("Load Other Database")
        self.load_other_db_btn.clicked.connect(self.load_other_database)
        
        db_load_layout.addWidget(self.load_db_btn)
        db_load_layout.addWidget(self.load_other_db_btn)
        
        db_select_layout.addLayout(other_db_layout)
        db_select_layout.addLayout(db_load_layout)
        db_select_widget.setLayout(db_select_layout)
        db_select_box.addWidget(db_select_widget)
        step5_layout.addWidget(db_select_box)
        
        # Filter Parameters
        filter_box = CollapsibleBox("Filter Parameters")
        filter_widget = QWidget()
        filter_layout = QGridLayout(filter_widget)
        
        self.cnf_checkbox = QCheckBox("CNF")
        self.cnf_checkbox.setChecked(True)
        self.ctnf_checkbox = QCheckBox("CTNF")
        self.ctnf_checkbox.setChecked(True)
        self.mw_checkbox = QCheckBox("MW")
        
        self.cnf_bias = QSpinBox()
        self.cnf_bias.setRange(1, 100)
        self.cnf_bias.setValue(5)
        
        self.ctnf_bias = QSpinBox()
        self.ctnf_bias.setRange(1, 100)
        self.ctnf_bias.setValue(2)
        
        self.mw_list = QLineEdit()
        self.mw_list.setText("300,400")
        
        filter_layout.addWidget(self.cnf_checkbox, 0, 0)
        filter_layout.addWidget(self.ctnf_checkbox, 0, 1)
        filter_layout.addWidget(self.mw_checkbox, 0, 2)
        filter_layout.addWidget(QLabel("CNF bias:"), 1, 0)
        filter_layout.addWidget(self.cnf_bias, 1, 1)
        filter_layout.addWidget(QLabel("CTNF bias:"), 1, 2)
        filter_layout.addWidget(self.ctnf_bias, 1, 3)
        filter_layout.addWidget(QLabel("MW list:"), 2, 0)
        filter_layout.addWidget(self.mw_list, 2, 1, 1, 3)
        
        filter_box.addWidget(filter_widget)
        step5_layout.addWidget(filter_box)
        
        # Evaluator Parameters
        evaluator_box = CollapsibleBox("Evaluator Parameters")
        evaluator_widget = QWidget()
        evaluator_layout = QGridLayout(evaluator_widget)
        
        self.evaluator_css = QRadioButton("CSS")
        self.evaluator_aas = QRadioButton("AAS")
        self.evaluator_fps = QRadioButton("FPS")
        self.evaluator_fpaacs = QRadioButton("FPAACS")
        self.evaluator_fpaacs.setChecked(True)
        
        self.css_threshold = QDoubleSpinBox()
        self.css_threshold.setRange(0, 1)
        self.css_threshold.setValue(0.5)
        self.css_threshold.setSingleStep(0.1)
        
        self.fpaacs_weights = QLineEdit()
        self.fpaacs_weights.setText("0.2,0.3,0.5")
        
        evaluator_layout.addWidget(self.evaluator_css, 0, 0)
        evaluator_layout.addWidget(self.evaluator_aas, 0, 1)
        evaluator_layout.addWidget(self.evaluator_fps, 0, 2)
        evaluator_layout.addWidget(self.evaluator_fpaacs, 0, 3)
        evaluator_layout.addWidget(QLabel("CSS threshold:"), 1, 0)
        evaluator_layout.addWidget(self.css_threshold, 1, 1)
        evaluator_layout.addWidget(QLabel("FPAACS weights:"), 1, 2)
        evaluator_layout.addWidget(self.fpaacs_weights, 1, 3)
        
        evaluator_box.addWidget(evaluator_widget)
        step5_layout.addWidget(evaluator_box)
        
        # Start Analysis + copy manuscript methods paragraph
        analysis_btn_row = QHBoxLayout()
        self.analysis_submit_btn = QPushButton("Start Analysis")
        self.analysis_submit_btn.setObjectName("PrimaryActionButton")
        self.analysis_submit_btn.clicked.connect(self.start_analysis)
        self.analysis_submit_btn.setEnabled(False)
        analysis_btn_row.addWidget(self.analysis_submit_btn)
        self.copy_methodology_btn = QPushButton("Copy methods text")
        self.copy_methodology_btn.setToolTip(
            "一键复制英文方法学段落：含数据库搜索与 Fragment 分析（Monte Carlo masking、"
            "片段提取、跨分子/分子内 fusion 等当前界面参数），供论文 Methods 使用。"
        )
        self.copy_methodology_btn.clicked.connect(self.copy_methodology_to_clipboard)
        analysis_btn_row.addWidget(self.copy_methodology_btn)
        self.goto_ai_assistant_btn = QPushButton("AI: Top 5 →")
        self.goto_ai_assistant_btn.setObjectName("AiLlmButton")
        self.goto_ai_assistant_btn.setToolTip(
            "Open AI Assistant and run LLM interpretation of the top 5 screening hits"
        )
        self.goto_ai_assistant_btn.clicked.connect(self._ai_interpret_top5_from_analysis)
        analysis_btn_row.addWidget(self.goto_ai_assistant_btn)
        step5_layout.addLayout(analysis_btn_row)
        
        step5_widget.setLayout(step5_layout)
        db_box.addWidget(step5_widget)
        left_layout.addWidget(db_box)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        left_layout.addStretch()
        
        # Right panel for result display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Plot title and switch buttons
        plot_header_layout = QHBoxLayout()
        
        plot_title = QLabel("NMR Spectrum Display")
        plot_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #3d5a66; margin: 10px;")
        
        # Plot switch buttons
        self.peak_plot_btn = QPushButton("Peak Detection Plot")
        self.peak_plot_btn.setObjectName("PlotToggleButton")
        self.peak_plot_btn.setCheckable(True)
        self.peak_plot_btn.setChecked(True)
        self.peak_plot_btn.clicked.connect(lambda: self.switch_plot("peak"))
        
        self.merge_plot_btn = QPushButton("Merged Spectrum")
        self.merge_plot_btn.setObjectName("PlotToggleButton")
        self.merge_plot_btn.setCheckable(True)
        self.merge_plot_btn.clicked.connect(lambda: self.switch_plot("merge"))
        
        plot_header_layout.addWidget(plot_title)
        plot_header_layout.addStretch()
        plot_header_layout.addWidget(self.peak_plot_btn)
        plot_header_layout.addWidget(self.merge_plot_btn)
        
        right_layout.addLayout(plot_header_layout)
        
        # Left panel for plots (larger area)
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        
        # Create matplotlib canvas and toolbar for plots
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, plot_panel)
        
        # Set canvas style
        self.canvas.setStyleSheet("""
            QWidget {
                border: 1px solid #c5ccd4;
                border-radius: 8px;
                background: white;
            }
        """)
        
        # Keep plot height stable when the left parameter panel grows
        self.canvas.setMinimumSize(500, 320)
        self.canvas.setMaximumHeight(420)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        # Add plot components to left panel
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        plot_layout.addStretch(1)
        
        # Right panel for peak information table
        table_panel = QWidget()
        table_layout = QVBoxLayout(table_panel)
        
        # Peak information table title
        table_title = QLabel("Peak Information (NMR-1D.csv)")
        table_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #3d5a66; margin: 5px;")
        table_layout.addWidget(table_title)
        
        # Peak information table
        self.peak_info_table = QTableWidget()
        self.peak_info_table.setColumnCount(3)
        self.peak_info_table.setHorizontalHeaderLabels(["Chemical Shift (ppm)", "Carbon Type", "Intensity"])
        self.peak_info_table.horizontalHeader().setStretchLastSection(True)
        self.peak_info_table.setMinimumHeight(300)
        self.peak_info_table.setMaximumHeight(400)
        table_layout.addWidget(self.peak_info_table)
        
        # Cap NMR row height so plots are not stretched with the left panel
        nmr_row = QWidget()
        nmr_row.setMaximumHeight(540)
        nmr_row_layout = QHBoxLayout(nmr_row)
        nmr_row_layout.setContentsMargins(0, 0, 0, 0)
        nmr_row_layout.addWidget(plot_panel, 2)
        nmr_row_layout.addWidget(table_panel, 1)
        
        right_layout.addWidget(nmr_row)
        
        # Results section
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)

        results_toolbar = QHBoxLayout()
        self.export_analysis_folder_btn = QPushButton("Export Results Folder")
        self.export_analysis_folder_btn.setToolTip(
            "Export Result.csv, NMR-1D.csv, Top-20 structure grid (5×4), and analysis_parameters.txt "
            "into a timestamped folder under GUI_result_files/"
        )
        self.export_analysis_folder_btn.clicked.connect(self.export_molecular_analysis_folder)
        results_toolbar.addWidget(self.export_analysis_folder_btn)
        results_toolbar.addStretch()
        self.goto_ai_assistant_results_btn = QPushButton("AI: Top 5 →")
        self.goto_ai_assistant_results_btn.setObjectName("AiLlmButton")
        self.goto_ai_assistant_results_btn.setToolTip(
            "Open AI Assistant and run LLM interpretation of the top 5 screening hits"
        )
        self.goto_ai_assistant_results_btn.clicked.connect(self._ai_interpret_top5_from_analysis)
        results_toolbar.addWidget(self.goto_ai_assistant_results_btn)
        results_layout.addLayout(results_toolbar)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(["Compound ID", "SMILES", "Score", "Chemical Shifts", "Carbon Types"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setMinimumHeight(200)
        self.results_table.cellClicked.connect(self.show_selected_compound)
        self.results_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self._results_table_context_menu)
        results_layout.addWidget(self.results_table)
        
        popup_hint = QLabel(
            "Click any result row to open the compound detail popup "
            "(with Previous/Next and Export Image)."
        )
        popup_hint.setStyleSheet("color: #6c757d; padding: 6px 2px;")
        results_layout.addWidget(popup_hint)
        
        # Add results group to right layout
        right_layout.addWidget(results_group)
        right_layout.setStretch(1, 0)  # NMR plot row: fixed height
        right_layout.setStretch(2, 1)  # Results table: use remaining space
        
        # Left: scrollable parameters; right: fixed-height plots, top-aligned
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setMinimumWidth(400)
        left_scroll.setMaximumWidth(650)

        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        analysis_layout.addWidget(left_scroll, 1)
        analysis_layout.addWidget(right_panel, 2, Qt.AlignTop)
        analysis_layout.setAlignment(Qt.AlignTop)

        self.tab_widget.addTab(analysis_widget, "Molecular Analysis")

    def create_fragment_analysis_tab(self):
        """Fragment Analysis: Random masking AAS attribution — see masking_aas_attribution.py."""
        frag_root = QWidget()
        root_layout = QVBoxLayout(frag_root)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(15)

        self.frag_progress = QProgressBar()
        self.frag_progress.setVisible(False)
        root_layout.addWidget(self.frag_progress)

        # ----- Random masking AAS (left: params, right: structure + table) -----
        mask_page = QWidget()
        mask_page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        page_m_layout = QHBoxLayout(mask_page)
        page_m_layout.setSpacing(15)
        page_m_layout.setAlignment(Qt.AlignTop)

        left_mask = QWidget()
        left_mask.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        left_mask.setMaximumWidth(650)
        mask_layout = QVBoxLayout(left_mask)
        mask_mc_collapsible = CollapsibleBox(
            "Random masking AAS — Monte Carlo parameters", start_expanded=True
        )

        mg = QGridLayout()
        mr = 0
        mg.addWidget(QLabel("Top N candidates to analyze:"), mr, 0)
        self.frag_mask_topn_spin = QSpinBox()
        self.frag_mask_topn_spin.setRange(1, 500)
        self.frag_mask_topn_spin.setValue(10)
        mg.addWidget(self.frag_mask_topn_spin, mr, 1)
        mr += 1

        mg.addWidget(QLabel("Monte Carlo mask mode:"), mr, 0)
        self.frag_mask_mode_combo = QComboBox()
        self.frag_mask_mode_combo.addItem("Random fraction (ceil(f·N) carbons)", "random_fraction")
        self.frag_mask_mode_combo.addItem("Random connected subgraph (k bonded carbons)", "random_connected")
        self.frag_mask_mode_combo.addItem("Threshold baseline (CSS, τ − |Δδ|)", "threshold_baseline")
        self.frag_mask_mode_combo.setCurrentIndex(1)  # default: connected subgraph
        self.frag_mask_mode_combo.setToolTip(
            "Random fraction: each iteration uses ⌈f·N⌉ arbitrary carbons.\n"
            "Random connected: each iteration uses k carbons that form one bonded fragment (same molecule graph).\n"
            "Threshold baseline (CSS): deterministic per-carbon score = τ − |Δδ| from greedy peak matching "
            "(no Monte Carlo). R / f / k / seed / shuffle-δ are not used in this mode."
        )
        mg.addWidget(self.frag_mask_mode_combo, mr, 1)
        mr += 1

        mg.addWidget(QLabel("Connected mask size k (carbons):"), mr, 0)
        self.frag_mask_connected_k_spin = QSpinBox()
        self.frag_mask_connected_k_spin.setRange(1, 30)
        self.frag_mask_connected_k_spin.setValue(3)
        self.frag_mask_connected_k_spin.setToolTip(
            "Only used when mask mode is Random connected subgraph. k must be mutually reachable by bonds."
        )
        mg.addWidget(self.frag_mask_connected_k_spin, mr, 1)
        mr += 1
        self.frag_mask_mode_combo.currentIndexChanged.connect(self._update_frag_mask_mode_ui)
        self._update_frag_mask_mode_ui()

        mg.addWidget(QLabel("Random subset fraction f:"), mr, 0)
        self.frag_mask_fraction = QDoubleSpinBox()
        self.frag_mask_fraction.setRange(0.1, 0.95)
        self.frag_mask_fraction.setDecimals(2)
        self.frag_mask_fraction.setSingleStep(0.05)
        self.frag_mask_fraction.setValue(0.5)
        mg.addWidget(self.frag_mask_fraction, mr, 1)
        mr += 1

        mg.addWidget(QLabel("Threshold τ (ppm):"), mr, 0)
        self.frag_mask_tau_spin = QDoubleSpinBox()
        self.frag_mask_tau_spin.setRange(0.05, 50.0)
        self.frag_mask_tau_spin.setDecimals(2)
        self.frag_mask_tau_spin.setSingleStep(0.1)
        self.frag_mask_tau_spin.setValue(3.00)
        self.frag_mask_tau_spin.setEnabled(False)
        self.frag_mask_tau_spin.setToolTip(
            "Threshold-baseline (CSS) mode only. Per-carbon score is τ − |Δδ|, where |Δδ| comes "
            "from greedy peak matching (DEPT and greedy-unique flags below still apply). "
            "Carbons with score > 0 (i.e. |Δδ| < τ) are eligible for the downstream fragment extractor."
        )
        mg.addWidget(self.frag_mask_tau_spin, mr, 1)
        mr += 1

        mg.addWidget(QLabel("Monte Carlo iterations R:"), mr, 0)
        self.frag_mask_iterations = QSpinBox()
        self.frag_mask_iterations.setRange(10, 5000)
        self.frag_mask_iterations.setValue(300)
        mg.addWidget(self.frag_mask_iterations, mr, 1)
        mr += 1

        mg.addWidget(QLabel("Random seed (optional):"), mr, 0)
        self.frag_mask_seed_edit = QLineEdit()
        self.frag_mask_seed_edit.setPlaceholderText("empty = non-deterministic")
        mg.addWidget(self.frag_mask_seed_edit, mr, 1)
        mr += 1

        self.frag_mask_dept_check = QCheckBox("Consider DEPT type matching")
        self.frag_mask_dept_check.setChecked(True)
        self.frag_mask_dept_check.setToolTip(
            "If enabled, only compare predicted carbon to experimental peaks with matching q/t/d/s when available."
        )
        mg.addWidget(self.frag_mask_dept_check, mr, 0, 1, 2)
        mr += 1

        self.frag_mask_unique_match_check = QCheckBox("Greedy unique peak matching (remove matched peak each step)")
        self.frag_mask_unique_match_check.setChecked(True)
        self.frag_mask_unique_match_check.setToolTip(
            "If enabled, each step uses nearest available peak and removes it, reducing repeated peak assignment."
        )
        mg.addWidget(self.frag_mask_unique_match_check, mr, 0, 1, 2)
        mr += 1

        self.frag_mask_shuffle_aas_check = QCheckBox(
            "Shuffle virtual δ order before each AAS (greedy unique only)"
        )
        self.frag_mask_shuffle_aas_check.setChecked(True)
        self.frag_mask_shuffle_aas_check.setToolTip(
            "When greedy unique is on, randomly permute predicted shifts (and matching Ctypes) "
            "before AAS(full) and before each subset AAS, so matching order does not follow fixed carbon order."
        )
        self.frag_mask_shuffle_aas_check.setEnabled(False)
        self.frag_mask_unique_match_check.toggled.connect(self._update_frag_mask_shuffle_aas_enabled)
        mg.addWidget(self.frag_mask_shuffle_aas_check, mr, 0, 1, 2)
        mr += 1
        self._update_frag_mask_shuffle_aas_enabled(self.frag_mask_unique_match_check.isChecked())

        mg.addWidget(QLabel("Color scale (Mean score):"), mr, 0)
        color_row = QHBoxLayout()
        self.frag_mask_color_fixed_check = QCheckBox("Fixed ±T (cross-molecule)")
        self.frag_mask_color_fixed_check.setChecked(True)
        self.frag_mask_color_fixed_check.setToolTip(
            "On: same numeric range for all molecules (red = negative, green = positive). "
            "Off: stretch colors to this molecule only."
        )
        color_row.addWidget(self.frag_mask_color_fixed_check)
        color_row.addWidget(QLabel("T ="))
        self.frag_mask_color_halfrange = QDoubleSpinBox()
        self.frag_mask_color_halfrange.setRange(0.1, 200.0)
        self.frag_mask_color_halfrange.setDecimals(2)
        self.frag_mask_color_halfrange.setValue(5.0)
        self.frag_mask_color_halfrange.setToolTip("Maps [-T, +T] to red→yellow→green; values clip outside.")
        color_row.addWidget(self.frag_mask_color_halfrange)
        self.frag_mask_recolor_btn = QPushButton("Apply color scale only (no re-run MC)")
        self.frag_mask_recolor_btn.setObjectName("RecolorOnlyButton")
        self.frag_mask_recolor_btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.frag_mask_recolor_btn.clicked.connect(self.apply_masking_color_scale)
        self.frag_mask_recolor_btn.setEnabled(False)
        color_row.addWidget(self.frag_mask_recolor_btn)
        color_row.addStretch()
        cw = QWidget()
        cw.setLayout(color_row)
        mg.addWidget(cw, mr, 1)
        mr += 1

        mask_mc_collapsible.addLayout(mg)

        self.frag_mask_run_btn = QPushButton("Run masking attribution")
        self.frag_mask_run_btn.clicked.connect(self.run_masking_attribution)
        self.frag_mask_run_btn.setEnabled(False)
        mask_mc_collapsible.addWidget(self.frag_mask_run_btn)

        mask_layout.addWidget(mask_mc_collapsible)

        frag_extract_collapsible = CollapsibleBox("Positive fragment extraction", start_expanded=False)

        extract_grid = QGridLayout()
        er = 0
        extract_grid.addWidget(QLabel("Threshold mode:"), er, 0)
        self.frag_mask_thresh_mode = QComboBox()
        self.frag_mask_thresh_mode.addItem("Manual (>T)", "manual")
        self.frag_mask_thresh_mode.addItem("Robust Z-score (median + k·MAD)", "robust_z")
        self.frag_mask_thresh_mode.setCurrentIndex(0)
        extract_grid.addWidget(self.frag_mask_thresh_mode, er, 1)
        er += 1
        extract_grid.addWidget(QLabel("Score threshold (>T):"), er, 0)
        self.frag_mask_frag_thresh = QDoubleSpinBox()
        self.frag_mask_frag_thresh.setRange(-200.0, 200.0)
        self.frag_mask_frag_thresh.setDecimals(3)
        self.frag_mask_frag_thresh.setSingleStep(0.05)
        self.frag_mask_frag_thresh.setValue(0.0)
        extract_grid.addWidget(self.frag_mask_frag_thresh, er, 1)
        er += 1
        extract_grid.addWidget(QLabel("Robust Z k:"), er, 0)
        self.frag_mask_robust_k = QDoubleSpinBox()
        self.frag_mask_robust_k.setRange(0.0, 10.0)
        self.frag_mask_robust_k.setDecimals(2)
        self.frag_mask_robust_k.setSingleStep(0.1)
        self.frag_mask_robust_k.setValue(1.0)
        self.frag_mask_robust_k.setToolTip("Threshold = median(score) + k * 1.4826 * MAD(score).")
        extract_grid.addWidget(self.frag_mask_robust_k, er, 1)
        er += 1
        self.frag_mask_thresh_info = QLabel("Auto threshold: --")
        self.frag_mask_thresh_info.setStyleSheet("color: #6c757d;")
        extract_grid.addWidget(self.frag_mask_thresh_info, er, 0, 1, 2)
        er += 1
        extract_grid.addWidget(QLabel("Min carbons in fragment:"), er, 0)
        self.frag_mask_frag_min_c = QSpinBox()
        self.frag_mask_frag_min_c.setRange(1, 30)
        self.frag_mask_frag_min_c.setValue(2)
        extract_grid.addWidget(self.frag_mask_frag_min_c, er, 1)
        er += 1

        self.frag_mask_bridge_check = QCheckBox(
            "Bridge via low-score carbons (merge adjacent high-score clusters)"
        )
        self.frag_mask_bridge_check.setToolTip(
            "Allow merging high-score clusters through low-score carbons. "
            "Bridge carbons are also included in fragment-level spectrum matching."
        )
        extract_grid.addWidget(self.frag_mask_bridge_check, er, 0, 1, 2)
        er += 1

        self.frag_mask_allow_hetero_bridge_check = QCheckBox(
            "Allow connectivity across hetero atom bridge (C-X-C, X=O/N/S...)"
        )
        self.frag_mask_allow_hetero_bridge_check.setChecked(True)
        self.frag_mask_allow_hetero_bridge_check.setToolTip(
            "When enabled, fragment extraction connectivity treats C-X-C as adjacent "
            "(X is non-hydrogen hetero atom)."
        )
        extract_grid.addWidget(self.frag_mask_allow_hetero_bridge_check, er, 0, 1, 2)
        er += 1

        extract_grid.addWidget(QLabel("Max bridge low-score carbons:"), er, 0)
        self.frag_mask_bridge_max_low = QSpinBox()
        self.frag_mask_bridge_max_low.setRange(1, 10)
        self.frag_mask_bridge_max_low.setValue(1)
        self.frag_mask_bridge_max_low.setEnabled(False)
        self.frag_mask_bridge_max_low.setToolTip(
            "When bridging is enabled, connect high-score clusters if a path exists "
            "through at most N low-score carbons."
        )
        self.frag_mask_bridge_check.toggled.connect(
            lambda checked: self.frag_mask_bridge_max_low.setEnabled(bool(checked))
        )
        extract_grid.addWidget(self.frag_mask_bridge_max_low, er, 1)
        self.frag_mask_bridge_check.setChecked(True)
        er += 1

        self.frag_mask_extract_dept_check = QCheckBox(
            "Fragment ↔ spectrum: DEPT type match (q/t/d/s only to same-type peaks)"
        )
        self.frag_mask_extract_dept_check.setChecked(True)
        self.frag_mask_extract_dept_check.setToolTip(
            "Applies only to fragment-to-experimental matching (not the Monte Carlo masking run)."
        )
        extract_grid.addWidget(self.frag_mask_extract_dept_check, er, 0, 1, 2)
        er += 1

        self.frag_mask_extract_greedy_check = QCheckBox(
            "Fragment ↔ spectrum: greedy unique peak matching"
        )
        self.frag_mask_extract_greedy_check.setChecked(True)
        self.frag_mask_extract_greedy_check.setToolTip(
            "Sequential nearest match with peak removal (same option as AAS). "
            "Independent from the masking MC settings above."
        )
        extract_grid.addWidget(self.frag_mask_extract_greedy_check, er, 0, 1, 2)
        er += 1
        self.frag_mask_thresh_mode.currentIndexChanged.connect(self._update_frag_mask_threshold_ui)
        self.frag_mask_frag_thresh.valueChanged.connect(self._update_frag_mask_threshold_ui)
        self.frag_mask_robust_k.valueChanged.connect(self._update_frag_mask_threshold_ui)
        self._update_frag_mask_threshold_ui()

        extract_grid.addWidget(QLabel("Greedy shuffle repeats:"), er, 0)
        self.frag_mask_greedy_shuffle_repeats = QSpinBox()
        self.frag_mask_greedy_shuffle_repeats.setRange(0, 10000)
        self.frag_mask_greedy_shuffle_repeats.setValue(100)
        self.frag_mask_greedy_shuffle_repeats.setToolTip(
            "Only when greedy unique matching is on. 0 or 1 = single pass (original order). "
            "≥2 = try this many random orderings of fragment virtual shifts; keep the match "
            "with smallest Σ(Δδ)² (same MSE term as AAS)."
        )
        extract_grid.addWidget(self.frag_mask_greedy_shuffle_repeats, er, 1)
        er += 1

        extract_grid.addWidget(QLabel("Shuffle RNG seed (optional):"), er, 0)
        self.frag_mask_greedy_shuffle_seed = QLineEdit()
        self.frag_mask_greedy_shuffle_seed.setPlaceholderText("empty = non-deterministic")
        self.frag_mask_greedy_shuffle_seed.setToolTip(
            "Integer seed for shuffling order across fragments (same extraction is reproducible)."
        )
        extract_grid.addWidget(self.frag_mask_greedy_shuffle_seed, er, 1)
        er += 1

        extract_grid.addWidget(QLabel("Max fragments listed:"), er, 0)
        self.frag_mask_max_frags = QSpinBox()
        self.frag_mask_max_frags.setRange(1, 500)
        self.frag_mask_max_frags.setValue(50)
        self.frag_mask_max_frags.setToolTip(
            "After ranking by Σscore, keep at most this many fragments in the table."
        )
        extract_grid.addWidget(self.frag_mask_max_frags, er, 1)
        er += 1

        self.frag_mask_extract_btn = QPushButton("Extract positive fragments + map")
        self.frag_mask_extract_btn.setEnabled(False)
        self.frag_mask_extract_btn.clicked.connect(self.extract_masking_fragments)
        extract_grid.addWidget(self.frag_mask_extract_btn, er, 0, 1, 2)
        er += 1

        intra_hdr = QLabel(
            "Intra-molecular fragment fusion (same algorithm as cross-compound fusion, "
            "fragments from the current candidate only)"
        )
        intra_hdr.setWordWrap(True)
        intra_hdr.setStyleSheet("color: #495057; font-weight: bold; margin-top: 8px;")
        extract_grid.addWidget(intra_hdr, er, 0, 1, 2)
        er += 1

        extract_grid.addWidget(QLabel("Pool Top fragments:"), er, 0)
        self.frag_intra_fusion_pool_topn = QSpinBox()
        self.frag_intra_fusion_pool_topn.setRange(1, 200)
        self.frag_intra_fusion_pool_topn.setValue(30)
        extract_grid.addWidget(self.frag_intra_fusion_pool_topn, er, 1)
        er += 1

        extract_grid.addWidget(QLabel("Max merged fragments:"), er, 0)
        self.frag_intra_fusion_max_merge = QSpinBox()
        self.frag_intra_fusion_max_merge.setRange(1, 10)
        self.frag_intra_fusion_max_merge.setValue(5)
        extract_grid.addWidget(self.frag_intra_fusion_max_merge, er, 1)
        er += 1

        extract_grid.addWidget(QLabel("Greedy shuffle repeats:"), er, 0)
        self.frag_intra_fusion_shuffle_repeats = QSpinBox()
        self.frag_intra_fusion_shuffle_repeats.setRange(1, 500)
        self.frag_intra_fusion_shuffle_repeats.setValue(10)
        extract_grid.addWidget(self.frag_intra_fusion_shuffle_repeats, er, 1)
        er += 1

        extract_grid.addWidget(QLabel("Coverage weight λ:"), er, 0)
        self.frag_intra_fusion_cov_weight = QDoubleSpinBox()
        self.frag_intra_fusion_cov_weight.setRange(0.0, 5.0)
        self.frag_intra_fusion_cov_weight.setDecimals(2)
        self.frag_intra_fusion_cov_weight.setSingleStep(0.05)
        self.frag_intra_fusion_cov_weight.setValue(0.3)
        extract_grid.addWidget(self.frag_intra_fusion_cov_weight, er, 1)
        er += 1

        self.frag_intra_fusion_run_btn = QPushButton(
            "Run intra-molecular fusion (current candidate)"
        )
        self.frag_intra_fusion_run_btn.setEnabled(False)
        self.frag_intra_fusion_run_btn.clicked.connect(self.run_intra_fragment_fusion)
        extract_grid.addWidget(self.frag_intra_fusion_run_btn, er, 0, 1, 2)

        frag_extract_collapsible.addLayout(extract_grid)
        mask_layout.addWidget(frag_extract_collapsible)

        frag_fusion_collapsible = CollapsibleBox("Fragment fusion (Top combinations)", start_expanded=False)
        fusion_grid = QGridLayout()
        fr = 0
        fusion_grid.addWidget(QLabel("Fusion pool Top fragments:"), fr, 0)
        self.frag_fusion_pool_topn = QSpinBox()
        self.frag_fusion_pool_topn.setRange(1, 200)
        self.frag_fusion_pool_topn.setValue(30)
        fusion_grid.addWidget(self.frag_fusion_pool_topn, fr, 1)
        fr += 1

        fusion_grid.addWidget(QLabel("Fusion max merged fragments:"), fr, 0)
        self.frag_fusion_max_merge = QSpinBox()
        self.frag_fusion_max_merge.setRange(1, 10)
        self.frag_fusion_max_merge.setValue(5)
        fusion_grid.addWidget(self.frag_fusion_max_merge, fr, 1)
        fr += 1

        fusion_grid.addWidget(QLabel("Fusion greedy shuffle repeats:"), fr, 0)
        self.frag_fusion_shuffle_repeats = QSpinBox()
        self.frag_fusion_shuffle_repeats.setRange(1, 500)
        self.frag_fusion_shuffle_repeats.setValue(10)
        fusion_grid.addWidget(self.frag_fusion_shuffle_repeats, fr, 1)
        fr += 1

        fusion_grid.addWidget(QLabel("Fusion coverage weight λ:"), fr, 0)
        self.frag_fusion_cov_weight = QDoubleSpinBox()
        self.frag_fusion_cov_weight.setRange(0.0, 5.0)
        self.frag_fusion_cov_weight.setDecimals(2)
        self.frag_fusion_cov_weight.setSingleStep(0.05)
        self.frag_fusion_cov_weight.setValue(0.3)
        fusion_grid.addWidget(self.frag_fusion_cov_weight, fr, 1)
        fr += 1

        self.frag_fusion_run_btn = QPushButton("Run fusion analysis (Top combinations)")
        self.frag_fusion_run_btn.setEnabled(False)
        self.frag_fusion_run_btn.clicked.connect(self.run_fragment_fusion_analysis)
        fusion_grid.addWidget(self.frag_fusion_run_btn, fr, 0, 1, 2)

        frag_fusion_collapsible.addLayout(fusion_grid)
        mask_layout.addWidget(frag_fusion_collapsible)

        self.frag_mask_status = QLabel("")
        self.frag_mask_status.setStyleSheet("color: #6c757d;")
        mask_layout.addWidget(self.frag_mask_status)

        right_mask = QWidget()
        right_mask.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        right_mask_layout = QVBoxLayout(right_mask)
        right_mask_layout.setSpacing(4)
        right_mask_layout.setContentsMargins(0, 0, 0, 0)
        right_mask_layout.setAlignment(Qt.AlignTop)

        view_row = QHBoxLayout()
        view_row.setContentsMargins(0, 0, 0, 0)
        view_row.addWidget(QLabel("Result view:"))
        self.frag_mask_view_combo = QComboBox()
        self.frag_mask_view_combo.addItems(["Compound attribution view", "Fusion combinations view"])
        self.frag_mask_view_combo.currentIndexChanged.connect(self._on_frag_mask_view_changed)
        view_row.addWidget(self.frag_mask_view_combo, 1)
        right_mask_layout.addLayout(view_row)

        self.frag_mask_img_title = QLabel("Attribution structure")
        self.frag_mask_img_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #3d5a66;")
        right_mask_layout.addWidget(self.frag_mask_img_title)

        self.frag_mask_result_row_wrap = QWidget()
        result_sel_row = QHBoxLayout(self.frag_mask_result_row_wrap)
        result_sel_row.setContentsMargins(0, 0, 0, 0)
        result_sel_row.addWidget(QLabel(""))
        self.frag_mask_result_combo = QComboBox()
        self.frag_mask_result_combo.setEnabled(False)
        self.frag_mask_result_combo.currentIndexChanged.connect(
            self._on_masking_result_combo_changed
        )
        result_sel_row.addWidget(self.frag_mask_result_combo, 1)
        self.frag_mask_infer_btn = QPushButton("AI: Infer structure")
        self.frag_mask_infer_btn.setObjectName("AiLlmButton")
        self.frag_mask_infer_btn.setEnabled(False)
        self.frag_mask_infer_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.frag_mask_infer_btn.setToolTip(
            "Send the selected candidate's masking data (SMILES, per-carbon score, DEPT type, "
            "virtual shifts) + experimental peaks to the LLM to infer plausible structures "
            "(LLM API required; Edit → Preferences)."
        )
        self.frag_mask_infer_btn.clicked.connect(self._ai_infer_structure_from_selected)
        result_sel_row.addWidget(self.frag_mask_infer_btn)
        self.frag_mask_assign_btn = QPushButton("AI: Assign signals")
        self.frag_mask_assign_btn.setObjectName("AiLlmButton")
        self.frag_mask_assign_btn.setEnabled(False)
        self.frag_mask_assign_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.frag_mask_assign_btn.setToolTip(
            "After Monte-Carlo carbon scoring, extract high-score fragments and assign "
            "experimental 13C/DEPT signals to the selected candidate's carbons; "
            "LLM returns a structured assignment report."
        )
        self.frag_mask_assign_btn.clicked.connect(self._ai_assign_signals_from_selected)
        result_sel_row.addWidget(self.frag_mask_assign_btn)
        self.frag_mask_review_btn = QPushButton("AI: Fragment evidence review")
        self.frag_mask_review_btn.setObjectName("AiLlmButton")
        self.frag_mask_review_btn.setEnabled(False)
        self.frag_mask_review_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.frag_mask_review_btn.setToolTip(
            "Analyze extracted positive fragments and mapping evidence: reliability ranking, "
            "redundancy/complementarity, unexplained peaks, and fusion-ready subset."
        )
        self.frag_mask_review_btn.clicked.connect(self._ai_fragment_evidence_review)
        result_sel_row.addWidget(self.frag_mask_review_btn)
        right_mask_layout.addWidget(self.frag_mask_result_row_wrap)

        self.frag_mask_image_label = QLabel()
        self.frag_mask_image_label.setMinimumSize(400, 320)
        self.frag_mask_image_label.setAlignment(Qt.AlignCenter)
        self.frag_mask_image_label.setStyleSheet("border: 1px solid #dee2e6; background: #fff;")
        self.frag_mask_image_label.setText("Colored structure appears here after run.")
        self.frag_mask_image_label.setSizePolicy(
            QSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        )

        self.frag_mask_label_opts_wrap = QWidget()
        label_opts_row = QHBoxLayout(self.frag_mask_label_opts_wrap)
        label_opts_row.setContentsMargins(0, 0, 0, 0)
        self.frag_mask_show_atom_idx_check = QCheckBox("Show RDKit atom index on structure")
        self.frag_mask_show_atom_idx_check.setChecked(False)
        self.frag_mask_show_atom_idx_check.setToolTip(
            "Overlay each carbon’s RDKit atom index on the 2D structure."
        )
        self.frag_mask_show_atom_idx_check.toggled.connect(
            self._refresh_masking_structure_image
        )
        label_opts_row.addWidget(self.frag_mask_show_atom_idx_check)
        self.frag_mask_show_pred_shift_check = QCheckBox("Show predicted δ (ppm) on structure")
        self.frag_mask_show_pred_shift_check.setChecked(False)
        self.frag_mask_show_pred_shift_check.setToolTip(
            "Overlay predicted ¹³C chemical shift (ppm) for each carbon on the structure."
        )
        self.frag_mask_show_pred_shift_check.toggled.connect(
            self._refresh_masking_structure_image
        )
        label_opts_row.addWidget(self.frag_mask_show_pred_shift_check)
        label_opts_row.addStretch()
        self.frag_mask_transform_opts_wrap = QWidget()
        transform_opts_row = QHBoxLayout(self.frag_mask_transform_opts_wrap)
        transform_opts_row.setContentsMargins(0, 0, 0, 0)
        transform_opts_row.addWidget(QLabel("Rotate (deg):"))
        self.frag_mask_rotate_spin = QDoubleSpinBox()
        self.frag_mask_rotate_spin.setRange(-360.0, 360.0)
        self.frag_mask_rotate_spin.setDecimals(1)
        self.frag_mask_rotate_spin.setSingleStep(5.0)
        self.frag_mask_rotate_spin.setValue(0.0)
        self.frag_mask_rotate_spin.setToolTip(
            "Rotate structure image clockwise by this angle before preview/export."
        )
        self.frag_mask_rotate_spin.valueChanged.connect(self._refresh_masking_structure_image)
        transform_opts_row.addWidget(self.frag_mask_rotate_spin)
        self.frag_mask_flip_h_check = QCheckBox("Flip H")
        self.frag_mask_flip_h_check.setToolTip("Flip structure image horizontally.")
        self.frag_mask_flip_h_check.toggled.connect(self._refresh_masking_structure_image)
        transform_opts_row.addWidget(self.frag_mask_flip_h_check)
        self.frag_mask_flip_v_check = QCheckBox("Flip V")
        self.frag_mask_flip_v_check.setToolTip("Flip structure image vertically.")
        self.frag_mask_flip_v_check.toggled.connect(self._refresh_masking_structure_image)
        transform_opts_row.addWidget(self.frag_mask_flip_v_check)
        self.frag_mask_export_hd_btn = QPushButton("Save HD (SVG/PNG)")
        self.frag_mask_export_hd_btn.setToolTip(
            "Export current attribution structure with current labels/orientation."
        )
        self.frag_mask_export_hd_btn.clicked.connect(self._export_mask_structure_hd_image)
        transform_opts_row.addWidget(self.frag_mask_export_hd_btn)
        transform_opts_row.addStretch()

        self.frag_mask_scores_heading = QLabel("")
        self.frag_mask_scores_heading.setStyleSheet("font-size: 16px; font-weight: bold; color: #3d5a66;")
        self.frag_mask_scores_table = QTableWidget()
        self.frag_mask_scores_table.setColumnCount(5)
        self.frag_mask_scores_table.setHorizontalHeaderLabels(
            [
                "Shift index",
                "RDKit atom index",
                "Type (s/d/t/q)",
                "Mean score",
                "δ_pred (ppm)",
            ]
        )
        self.frag_mask_scores_table.horizontalHeader().setStretchLastSection(True)
        self.frag_mask_scores_table.setMinimumHeight(300)
        self.frag_mask_scores_table.setSizePolicy(
            QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        )
        self.frag_mask_scores_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # 结构图与 per-carbon 分数表左右排列
        self.frag_mask_struct_scores_row = QWidget()
        _ss_h = QHBoxLayout(self.frag_mask_struct_scores_row)
        _ss_h.setContentsMargins(0, 0, 0, 0)
        _ss_h.setSpacing(14)
        _left_struct = QWidget()
        _left_struct_l = QVBoxLayout(_left_struct)
        _left_struct_l.setContentsMargins(0, 0, 0, 0)
        _left_struct_l.setSpacing(6)
        _left_struct_l.addWidget(self.frag_mask_image_label)
        _left_struct_l.addWidget(self.frag_mask_label_opts_wrap)
        _left_struct_l.addWidget(self.frag_mask_transform_opts_wrap)
        _left_struct_l.addStretch(1)
        _left_struct.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        _right_scores = QWidget()
        _right_scores_l = QVBoxLayout(_right_scores)
        _right_scores_l.setContentsMargins(0, 0, 0, 0)
        _right_scores_l.setSpacing(4)
        _right_scores_l.addWidget(self.frag_mask_scores_heading)
        _right_scores_l.addWidget(self.frag_mask_scores_table, 1)
        _right_scores.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        _ss_h.addWidget(_left_struct, 0, Qt.AlignTop)
        _ss_h.addWidget(_right_scores, 1)
        right_mask_layout.addWidget(self.frag_mask_struct_scores_row)

        self.frag_mask_frag_table = QTableWidget()
        self.frag_mask_frag_table.setColumnCount(9)
        self.frag_mask_frag_table.setHorizontalHeaderLabels(
            [
                "Frag #",
                "N C",
                "N(high)",
                "Σscore",
                "Mean score",
                "Hit(unique)",
                "MAE (ppm)",
                "ΣΔδ²",
                "Matched exp/pred pairs",
            ]
        )
        self.frag_mask_frag_table.horizontalHeader().setStretchLastSection(True)
        self.frag_mask_frag_table.setMinimumHeight(140)
        self.frag_mask_frag_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.frag_mask_frag_table.setSelectionMode(QTableWidget.SingleSelection)
        self.frag_mask_frag_table.cellClicked.connect(self._on_mask_fragment_table_clicked)
        self.frag_mask_frag_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.frag_mask_frag_table.customContextMenuRequested.connect(
            self._frag_mask_table_context_menu
        )
        right_mask_layout.addWidget(self.frag_mask_frag_table)
        self.frag_mask_global_title = QLabel("All extracted fragments (all candidates, sorted by Σscore)")
        self.frag_mask_global_title.setStyleSheet("color: #495057; font-weight: bold; margin-top: 6px;")
        right_mask_layout.addWidget(self.frag_mask_global_title)
        self.frag_mask_global_ai_row = QWidget()
        _gai = QHBoxLayout(self.frag_mask_global_ai_row)
        _gai.setContentsMargins(0, 0, 0, 0)
        _gai.setSpacing(6)
        _gai.addStretch()
        self.frag_mask_global_review_btn = QPushButton("AI: Global fragment evidence review")
        self.frag_mask_global_review_btn.setObjectName("AiLlmButton")
        self.frag_mask_global_review_btn.setEnabled(False)
        self.frag_mask_global_review_btn.setToolTip(
            "Analyze all extracted fragments across candidates: evidence ranking, redundancy/"
            "complementarity, missing motifs and fusion-ready subset."
        )
        self.frag_mask_global_review_btn.clicked.connect(self._ai_global_fragment_evidence_review)
        _gai.addWidget(self.frag_mask_global_review_btn)
        right_mask_layout.addWidget(self.frag_mask_global_ai_row)
        self.frag_mask_global_table = QTableWidget()
        self.frag_mask_global_table.setColumnCount(9)
        self.frag_mask_global_table.setHorizontalHeaderLabels(
            ["Rank", "Candidate", "Frag #", "N C", "Σscore", "Mean", "Hit(unique)", "MAE", "ΣΔδ²"]
        )
        self.frag_mask_global_table.horizontalHeader().setStretchLastSection(True)
        self.frag_mask_global_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.frag_mask_global_table.setSelectionMode(QTableWidget.SingleSelection)
        self.frag_mask_global_table.setMinimumHeight(180)
        self.frag_mask_global_table.setMaximumHeight(280)
        self.frag_mask_global_table.cellClicked.connect(self._on_global_fragment_table_clicked)
        right_mask_layout.addWidget(self.frag_mask_global_table)
        self.frag_mask_global_plot_title = QLabel("All extracted fragments by Σscore (sorted)")
        self.frag_mask_global_plot_title.setStyleSheet("color: #495057; font-weight: bold; margin-top: 4px;")
        right_mask_layout.addWidget(self.frag_mask_global_plot_title)
        self.frag_mask_global_opts_wrap = QWidget()
        _gopt_l = QHBoxLayout(self.frag_mask_global_opts_wrap)
        _gopt_l.setContentsMargins(0, 0, 0, 0)
        _gopt_l.setSpacing(10)
        self.frag_mask_global_no_highlight = QCheckBox("No highlight")
        self.frag_mask_global_no_ppm = QCheckBox("No ppm labels")
        self.frag_mask_global_no_highlight.setChecked(False)
        self.frag_mask_global_no_ppm.setChecked(False)
        self.frag_mask_global_no_highlight.toggled.connect(self._render_global_fragments_score_plot)
        self.frag_mask_global_no_ppm.toggled.connect(self._render_global_fragments_score_plot)
        _gopt_l.addWidget(self.frag_mask_global_no_highlight)
        _gopt_l.addWidget(self.frag_mask_global_no_ppm)
        _gopt_l.addStretch()
        right_mask_layout.addWidget(self.frag_mask_global_opts_wrap)
        self.frag_mask_global_struct_label = QLabel("All fragment structures will appear here after extraction.")
        self.frag_mask_global_struct_label.setAlignment(Qt.AlignCenter)
        self.frag_mask_global_struct_label.setStyleSheet(
            "border: 1px solid #dee2e6; border-radius: 8px; background: #fff;"
        )
        self.frag_mask_global_struct_label.setScaledContents(False)
        self.frag_mask_global_scroll = QScrollArea()
        self.frag_mask_global_scroll.setWidgetResizable(False)
        self.frag_mask_global_scroll.setMinimumHeight(260)
        self.frag_mask_global_scroll.setMaximumHeight(420)
        self.frag_mask_global_scroll.setStyleSheet(
            "QScrollArea { border: 1px solid #dde3ea; border-radius: 8px; background: #f8fafc; }"
        )
        self.frag_mask_global_scroll.setWidget(self.frag_mask_global_struct_label)
        right_mask_layout.addWidget(self.frag_mask_global_scroll)
        self.frag_mask_plot_hint_label = QLabel("Click a fragment row to open NMR matching plot.")
        self.frag_mask_plot_hint_label.setStyleSheet("color: #6c757d;")
        right_mask_layout.addWidget(self.frag_mask_plot_hint_label)

        self.frag_fusion_title = QLabel("")
        self.frag_fusion_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #3d5a66;")
        right_mask_layout.addWidget(self.frag_fusion_title)
        self.frag_fusion_toolbar = QWidget()
        _ftb = QHBoxLayout(self.frag_fusion_toolbar)
        _ftb.setContentsMargins(0, 0, 0, 6)
        _ftb.addStretch(1)
        self.frag_fusion_ai_top5_btn = QPushButton("AI: Fusion → Top-5")
        self.frag_fusion_ai_top5_btn.setObjectName("AiLlmButton")
        self.frag_fusion_ai_top5_btn.setToolTip(
            "LLM assembles fragment motifs + experimental peaks into five proposed full "
            "structures (de novo; not a database hit re-ranking)."
        )
        self.frag_fusion_ai_top5_btn.setEnabled(False)
        self.frag_fusion_ai_top5_btn.clicked.connect(self._ai_fusion_infer_top5)
        _ftb.addWidget(self.frag_fusion_ai_top5_btn)
        self.frag_fusion_ai_review_btn = QPushButton("AI: Fusion evidence review")
        self.frag_fusion_ai_review_btn.setObjectName("AiLlmButton")
        self.frag_fusion_ai_review_btn.setToolTip(
            "Review fusion top combinations: ranking rationale, fragment roles, unexplained/conflicting "
            "signals, and priority verification plan."
        )
        self.frag_fusion_ai_review_btn.setEnabled(False)
        self.frag_fusion_ai_review_btn.clicked.connect(self._ai_fusion_evidence_review)
        _ftb.addWidget(self.frag_fusion_ai_review_btn)
        right_mask_layout.addWidget(self.frag_fusion_toolbar)
        self.frag_fusion_table = QTableWidget()
        self.frag_fusion_table.setColumnCount(8)
        self.frag_fusion_table.setHorizontalHeaderLabels(
            ["Rank", "k", "Na/Nb", "Coverage", "AAS(best)", "Final score", "ΣΔδ²", "Fragments"]
        )
        self.frag_fusion_table.horizontalHeader().setStretchLastSection(True)
        self.frag_fusion_table.setMinimumHeight(240)
        self.frag_fusion_table.setMaximumHeight(450)
        self.frag_fusion_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.frag_fusion_table.setSizePolicy(
            QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        )
        self.frag_fusion_table.cellClicked.connect(self._on_fusion_table_clicked)
        self.frag_fusion_table.cellDoubleClicked.connect(self._on_fusion_table_double_clicked)
        self.frag_fusion_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.frag_fusion_table.customContextMenuRequested.connect(
            self._frag_fusion_table_context_menu
        )
        right_mask_layout.addWidget(self.frag_fusion_table)
        self.frag_fusion_preview_card = QFrame()
        self.frag_fusion_preview_card.setObjectName("FusionInlinePreviewCard")
        self.frag_fusion_preview_card.setStyleSheet(
            """
            QFrame#FusionInlinePreviewCard {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #ffffff, stop:1 #f7fafc);
                border: 1px solid #d6dee6;
                border-radius: 10px;
            }
            """
        )
        _fp_l = QVBoxLayout(self.frag_fusion_preview_card)
        _fp_l.setContentsMargins(10, 8, 10, 10)
        _fp_l.setSpacing(6)
        self.frag_fusion_preview_title = QLabel("All matched fragments (combined view)")
        self.frag_fusion_preview_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #3d5a66;")
        _fp_l.addWidget(self.frag_fusion_preview_title)
        self.frag_fusion_combo = QComboBox()
        self.frag_fusion_combo.setEnabled(False)
        self.frag_fusion_combo.currentIndexChanged.connect(self._on_fusion_combo_changed)
        self.frag_fusion_combo.setVisible(False)
        self.frag_fusion_preview_label = QLabel("Fusion fragment combination preview appears here.")
        self.frag_fusion_preview_label.setMinimumHeight(120)
        self.frag_fusion_preview_label.setMaximumHeight(300)
        self.frag_fusion_preview_label.setAlignment(Qt.AlignCenter)
        self.frag_fusion_preview_label.setStyleSheet(
            "border: 1px solid #d9e1e8; border-radius: 8px; background: #ffffff;"
        )
        self.frag_fusion_preview_label.setScaledContents(False)
        self.frag_fusion_preview_label.setSizePolicy(
            QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        )
        _fp_l.addWidget(self.frag_fusion_preview_label)
        right_mask_layout.addWidget(self.frag_fusion_preview_card)
        self.frag_fusion_preview_card.setVisible(False)

        self.frag_fusion_map_opts_wrap = QWidget()
        fusion_map_opts_l = QHBoxLayout(self.frag_fusion_map_opts_wrap)
        fusion_map_opts_l.setContentsMargins(0, 0, 0, 0)
        self.frag_fusion_show_fragment_structs_check = QCheckBox(
            "Show fragment structures under spectrum (one per fragment)"
        )
        self.frag_fusion_show_fragment_structs_check.setChecked(True)
        self.frag_fusion_show_fragment_structs_check.setToolTip(
            "Off: spectrum + exp/pred links only. On: one thumbnail per fragment (neighbor shell), "
            "dashed lines from each δ_pred to that thumbnail."
        )
        self.frag_fusion_show_fragment_structs_check.toggled.connect(
            self._on_fusion_fragment_structs_toggled
        )
        fusion_map_opts_l.addWidget(self.frag_fusion_show_fragment_structs_check)
        fusion_map_opts_l.addStretch()
        self.frag_fusion_map_opts_wrap.setVisible(False)

        self.frag_fusion_map_canvas_top = FigureCanvas(Figure(figsize=(6, 4.2)))
        self.frag_fusion_map_canvas_top.setMinimumHeight(340)
        fusion_plot_fp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.frag_fusion_map_canvas_top.setSizePolicy(fusion_plot_fp)
        self.frag_fusion_map_canvas_top.setVisible(False)
        self.frag_fusion_map_canvas = FigureCanvas(Figure(figsize=(6, 2.8)))
        self.frag_fusion_map_canvas.setMinimumHeight(220)
        self.frag_fusion_map_canvas.setSizePolicy(fusion_plot_fp)
        self.frag_fusion_map_canvas.setVisible(False)

        # 左窄右高时，QHBoxLayout 默认子控件为 AlignVCenter，会把左侧整块顶到垂直中间
        page_m_layout.addWidget(left_mask, 1, Qt.AlignTop)
        page_m_layout.addWidget(right_mask, 2, Qt.AlignTop)

        root_layout.addWidget(mask_page, 0)

        self.masking_result_data = None
        self.masking_all_results = []
        self.masking_fragments_all = []
        self.masking_batch_rows = []
        self.masking_batch_cursor = 0
        self.masking_batch_total = 0
        self.masking_batch_exp = None
        self.masking_batch_options = None
        self.masking_fragments = []
        self.masking_fragments_global_rows = []
        self.masking_mapping_dialog = None
        self.fusion_detail_dialog = None
        self.fusion_results = []
        self._on_frag_mask_view_changed(0)

        scroll = QScrollArea()
        scroll.setWidget(frag_root)
        scroll.setWidgetResizable(True)
        self.tab_widget.addTab(scroll, "Fragment Analysis")

    def _update_frag_mask_shuffle_aas_enabled(self, checked):
        """Enable shuffle-δ checkbox only when greedy unique is on AND we are in a MC mode."""
        if not hasattr(self, "frag_mask_shuffle_aas_check"):
            return
        mode = (
            self.frag_mask_mode_combo.currentData()
            if hasattr(self, "frag_mask_mode_combo")
            else "random_fraction"
        )
        is_threshold = mode == "threshold_baseline"
        self.frag_mask_shuffle_aas_check.setEnabled(bool(checked) and not is_threshold)

    def _update_frag_mask_mode_ui(self, _idx=None):
        """Enable per-mode controls: fraction f, connected k, τ, and gray out MC-only inputs in CSS mode."""
        if not hasattr(self, "frag_mask_mode_combo"):
            return
        mode = self.frag_mask_mode_combo.currentData()
        if mode is None:
            mode = "random_fraction"
        is_fraction = mode == "random_fraction"
        is_connected = mode == "random_connected"
        is_threshold = mode == "threshold_baseline"
        is_mc = is_fraction or is_connected
        if hasattr(self, "frag_mask_fraction"):
            self.frag_mask_fraction.setEnabled(is_fraction)
        if hasattr(self, "frag_mask_connected_k_spin"):
            self.frag_mask_connected_k_spin.setEnabled(is_connected)
        if hasattr(self, "frag_mask_tau_spin"):
            self.frag_mask_tau_spin.setEnabled(is_threshold)
        if hasattr(self, "frag_mask_iterations"):
            self.frag_mask_iterations.setEnabled(is_mc)
        if hasattr(self, "frag_mask_seed_edit"):
            self.frag_mask_seed_edit.setEnabled(is_mc)
        if hasattr(self, "frag_mask_shuffle_aas_check") and hasattr(
            self, "frag_mask_unique_match_check"
        ):
            self._update_frag_mask_shuffle_aas_enabled(
                self.frag_mask_unique_match_check.isChecked()
            )

    def _update_frag_mask_threshold_ui(self, _idx=None):
        if not hasattr(self, "frag_mask_thresh_mode"):
            return
        mode = self.frag_mask_thresh_mode.currentData()
        is_manual = mode == "manual"
        if hasattr(self, "frag_mask_frag_thresh"):
            self.frag_mask_frag_thresh.setEnabled(is_manual)
        if hasattr(self, "frag_mask_robust_k"):
            self.frag_mask_robust_k.setEnabled(not is_manual)
        if hasattr(self, "frag_mask_thresh_info"):
            if is_manual:
                self.frag_mask_thresh_info.setText(
                    f"Manual threshold: {float(self.frag_mask_frag_thresh.value()):.3f}"
                )
            else:
                self.frag_mask_thresh_info.setText(
                    f"Robust Z mode: k={float(self.frag_mask_robust_k.value()):.2f}"
                )

    def _resolve_fragment_score_threshold(self, scores):
        mode = (
            self.frag_mask_thresh_mode.currentData()
            if hasattr(self, "frag_mask_thresh_mode")
            else "manual"
        )
        if mode == "manual":
            t = float(self.frag_mask_frag_thresh.value())
            return t, f"Manual threshold T={t:.3f}"

        arr_full = np.asarray(scores, dtype=float)
        arr = arr_full[np.isfinite(arr_full)] if arr_full.size else arr_full
        if arr.size == 0:
            t = float(self.frag_mask_frag_thresh.value())
            return t, f"No finite scores; fallback manual T={t:.3f}"

        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        k = float(self.frag_mask_robust_k.value()) if hasattr(self, "frag_mask_robust_k") else 1.0
        robust_sigma = 1.4826 * mad
        if robust_sigma <= 1e-12:
            t = med
            return t, f"Robust Z fallback: MAD≈0, using median={med:.3f}"

        t = med + k * robust_sigma
        return t, f"Robust Z: T={t:.3f} (median={med:.3f}, MAD={mad:.3f}, k={k:.2f})"

    def _on_masking_result_combo_changed(self, idx):
        if idx < 0 or idx >= len(getattr(self, "masking_all_results", [])):
            return
        payload = self.masking_all_results[idx]
        if idx < len(getattr(self, "masking_fragments_all", [])):
            self.masking_fragments = self.masking_fragments_all[idx]
        else:
            self.masking_fragments = []
        self._display_masking_payload(payload)
        if getattr(self, "masking_mapping_dialog", None) and self.masking_mapping_dialog.isVisible():
            self._refresh_masking_mapping_dialog()

    def _render_current_fragments_table(self):
        self.frag_mask_frag_table.setRowCount(0)
        fragments = list(getattr(self, "masking_fragments", []))
        for i, frag in enumerate(fragments):
            self.frag_mask_frag_table.insertRow(i)
            pairs = frag.get("pairs", [])
            peaks_txt = "; ".join(
                f"{float(row.get('exp_ppm', 0.0)):.2f}<-{float(row.get('pred_ppm', 0.0)):.2f}"
                for row in pairs
            )
            vals = [
                str(frag.get("fragment_id", i + 1)),
                str(frag.get("n_carbons", "")),
                str(frag.get("n_core_high_carbons", frag.get("n_carbons", ""))),
                f"{float(frag.get('score_sum', 0.0)):.4f}",
                f"{float(frag.get('score_mean', 0.0)):.4f}",
                f"{100.0 * float(frag.get('hit_rate_unique', 0.0)):.1f}%",
                f"{float(frag.get('mean_abs_err', float('nan'))):.3f}",
                f"{float(frag.get('match_mse', float('nan'))):.4f}",
                peaks_txt,
            ]
            for col, txt in enumerate(vals):
                self.frag_mask_frag_table.setItem(i, col, QTableWidgetItem(txt))
        self.frag_mask_frag_table.resizeColumnsToContents()
        if hasattr(self, "frag_mask_review_btn"):
            self.frag_mask_review_btn.setEnabled(bool(fragments))
        if fragments:
            self.frag_mask_frag_table.selectRow(0)
            try:
                self._refresh_masking_structure_image()
            except Exception:
                pass
        else:
            if getattr(self, "masking_mapping_dialog", None):
                self.masking_mapping_dialog.close()
                self.masking_mapping_dialog = None

    def _render_global_fragments_table(self):
        tbl = getattr(self, "frag_mask_global_table", None)
        if tbl is None:
            return
        rows = []
        all_frags = list(getattr(self, "masking_fragments_all", []) or [])
        all_payload = list(getattr(self, "masking_all_results", []) or [])
        for cand_idx, frags in enumerate(all_frags):
            candidate_rank = cand_idx + 1
            if cand_idx < len(all_payload):
                try:
                    candidate_rank = int(all_payload[cand_idx].get("candidate_rank_index", cand_idx)) + 1
                except Exception:
                    candidate_rank = cand_idx + 1
            for frag_idx, frag in enumerate(list(frags or [])):
                rows.append(
                    {
                        "candidate_idx": cand_idx,
                        "candidate_rank": candidate_rank,
                        "fragment_idx": frag_idx,
                        "frag": frag,
                        "score_sum": float(frag.get("score_sum", 0.0)),
                    }
                )
        rows.sort(key=lambda x: x["score_sum"], reverse=True)
        self.masking_fragments_global_rows = rows
        if hasattr(self, "frag_mask_global_review_btn"):
            self.frag_mask_global_review_btn.setEnabled(bool(rows))

        tbl.setRowCount(0)
        for i, rec in enumerate(rows):
            frag = rec["frag"]
            tbl.insertRow(i)
            vals = [
                str(i + 1),
                f"#{rec['candidate_rank']}",
                str(frag.get("fragment_id", rec["fragment_idx"] + 1)),
                str(frag.get("n_carbons", "")),
                f"{float(frag.get('score_sum', 0.0)):.4f}",
                f"{float(frag.get('score_mean', 0.0)):.4f}",
                f"{100.0 * float(frag.get('hit_rate_unique', 0.0)):.1f}%",
                f"{float(frag.get('mean_abs_err', float('nan'))):.3f}",
                f"{float(frag.get('match_mse', float('nan'))):.4f}",
            ]
            for col, txt in enumerate(vals):
                tbl.setItem(i, col, QTableWidgetItem(txt))
        tbl.resizeColumnsToContents()
        self._render_global_fragments_score_plot()

    def _render_global_fragments_score_plot(self):
        # Keep function name for compatibility; now renders one combined
        # structure image (all fragments sorted by Σscore) instead of bars.
        lbl = getattr(self, "frag_mask_global_struct_label", None)
        if lbl is None:
            return
        rows = list(getattr(self, "masking_fragments_global_rows", []) or [])
        if not rows:
            lbl.setPixmap(QPixmap())
            lbl.setText("No extracted fragments yet.")
            return

        try:
            from PIL import Image

            palette = self._fusion_nature_palette()
            all_payload = list(getattr(self, "masking_all_results", []) or [])
            # Use larger tiles and scroll view so each fragment is fully visible.
            tile_w, tile_h = 300, 220
            gap = 14
            scroll = getattr(self, "frag_mask_global_scroll", None)
            vp_w = int(scroll.viewport().width()) if scroll is not None else 1200
            ncols = max(1, min(4, (max(420, vp_w) - gap) // (tile_w + gap)))
            max_show = len(rows)
            show_highlight = not (
                getattr(self, "frag_mask_global_no_highlight", None)
                and self.frag_mask_global_no_highlight.isChecked()
            )
            show_ppm = not (
                getattr(self, "frag_mask_global_no_ppm", None)
                and self.frag_mask_global_no_ppm.isChecked()
            )
            # Build a synthetic "fusion row" and reuse the exact same renderer
            # as Fusion combination detail popup for consistent style.
            frags_for_view = []
            pairs_for_view = []
            dedup_seen = {}
            for rec in rows[:max_show]:
                cand_idx = int(rec.get("candidate_idx", -1))
                frag = dict(rec.get("frag", {}) or {})
                if cand_idx < 0 or cand_idx >= len(all_payload):
                    continue
                p = all_payload[cand_idx]
                smiles = str(p.get("smiles", "")).strip()
                atom_ids = [int(x) for x in (frag.get("atom_indices", []) or [])]
                if not smiles or not atom_ids:
                    continue
                # Merge fully identical fragments by canonical fragment structure.
                # Keep first one because rows are already sorted by Σscore (high -> low).
                try:
                    mol0 = Chem.MolFromSmiles(smiles)
                    if mol0 is None:
                        continue
                    frag_key = Chem.MolFragmentToSmiles(
                        mol0,
                        atomsToUse=sorted(set(atom_ids)),
                        canonical=True,
                        isomericSmiles=False,
                    )
                except Exception:
                    frag_key = f"{smiles}|{tuple(sorted(set(atom_ids)))}"
                if frag_key in dedup_seen:
                    dedup_seen[frag_key] += 1
                    continue
                dedup_seen[frag_key] = 1
                fid = int(frag.get("fragment_id", rec.get("fragment_idx", 0) + 1))
                frag["compound_idx"] = cand_idx
                frag["fragment_id"] = fid
                frags_for_view.append(frag)
                for pr in list(frag.get("pairs", []) or []):
                    pp = dict(pr)
                    pp["compound_idx"] = cand_idx
                    pp["fragment_id"] = fid
                    pairs_for_view.append(pp)

            if not frags_for_view:
                lbl.setPixmap(QPixmap())
                lbl.setText("No fragment structure preview available.")
                return

            pseudo_row = {"fragments": frags_for_view, "pairs": pairs_for_view}
            if show_highlight:
                canvas = self._build_fusion_matched_preview_image(
                    pseudo_row,
                    show_idx=False,
                    show_matched=bool(show_ppm),
                    fragments_only=True,
                    show_base_colors=True,
                )
            else:
                # If highlight is disabled, keep fragment-only geometry but use neutral color.
                tiles = []
                for f in frags_for_view:
                    ci = int(f.get("compound_idx", 0))
                    p = all_payload[ci]
                    local_pairs = [pr for pr in pairs_for_view if int(pr.get("compound_idx", -1)) == ci and int(pr.get("fragment_id", -1)) == int(f.get("fragment_id", 0))]
                    best_map = {}
                    for pr in local_pairs:
                        try:
                            aid = int(pr.get("atom_index", -1))
                            exp = float(pr.get("exp_ppm"))
                            ae = float(pr.get("abs_err", 1e18))
                        except Exception:
                            continue
                        old = best_map.get(aid)
                        if old is None or ae < old[0]:
                            best_map[aid] = (ae, exp)
                    matched_map = {a: v[1] for a, v in best_map.items()} if show_ppm else {}
                    try:
                        tile = self._draw_fusion_fragment_only_tile(
                            p.get("smiles", ""),
                            f.get("atom_indices", []),
                            matched_map,
                            show_idx=False,
                            show_matched=bool(show_ppm),
                            frag_rgb=(0.55, 0.55, 0.55),
                            use_fragment_coloring=False,
                            annotation_font_scale=0.68,
                            size=(tile_w, tile_h),
                        )
                    except Exception:
                        tile = Image.new("RGB", (tile_w, tile_h), (252, 253, 254))
                    tiles.append(tile)
                if not tiles:
                    lbl.setPixmap(QPixmap())
                    lbl.setText("No fragment structure preview available.")
                    return
                w = sum(im.width for im in tiles)
                h = max(im.height for im in tiles)
                canvas = Image.new("RGB", (w, h), (250, 252, 254))
                x = 0
                for im in tiles:
                    canvas.paste(im, (x, (h - im.height) // 2))
                    x += im.width

            if canvas is None:
                lbl.setPixmap(QPixmap())
                lbl.setText("No fragment structure preview available.")
                return
            pix = self.pil_image_to_pixmap(canvas)
            lbl.setPixmap(pix)
            lbl.setText("")
            lbl.resize(pix.size())
            lbl.setMinimumSize(0, 0)
            lbl.setFixedSize(pix.size())
            if scroll is not None:
                scroll.horizontalScrollBar().setValue(0)
                scroll.verticalScrollBar().setValue(0)
            self.frag_mask_global_plot_title.setText(
                f"All extracted fragments by Σscore (sorted, unique {len(frags_for_view)}/{len(rows)})"
            )
        except Exception as e:
            lbl.setPixmap(QPixmap())
            lbl.setText(f"Preview error: {e}")

    def _on_global_fragment_table_clicked(self, row, _col):
        rows = list(getattr(self, "masking_fragments_global_rows", []) or [])
        if row < 0 or row >= len(rows):
            return
        rec = rows[row]
        cand_idx = int(rec.get("candidate_idx", -1))
        frag_idx = int(rec.get("fragment_idx", -1))
        if cand_idx < 0 or frag_idx < 0:
            return
        combo = getattr(self, "frag_mask_result_combo", None)
        if combo is None or cand_idx >= combo.count():
            return
        combo.setCurrentIndex(cand_idx)
        self._on_mask_fragment_table_clicked(frag_idx, 0)

    def _start_masking_batch_item(self):
        if self.masking_batch_cursor >= self.masking_batch_total:
            self.frag_progress.setValue(100)
            self.frag_mask_run_btn.setEnabled(True)
            if hasattr(self, "frag_mask_recolor_btn"):
                self.frag_mask_recolor_btn.setEnabled(True)
            if hasattr(self, "frag_mask_extract_btn"):
                self.frag_mask_extract_btn.setEnabled(True)
            if hasattr(self, "frag_fusion_run_btn"):
                self.frag_fusion_run_btn.setEnabled(False)
            self.frag_progress.setVisible(False)
            self.frag_mask_result_combo.clear()
            for p in self.masking_all_results:
                rank = int(p.get("candidate_rank_index", 0)) + 1
                aas = float(p.get("aas_full", 0.0))
                self.frag_mask_result_combo.addItem(f"Candidate #{rank} | AAS(full)={aas:.4f}")
            self.frag_mask_result_combo.setEnabled(len(self.masking_all_results) > 0)
            if hasattr(self, "frag_mask_infer_btn"):
                self.frag_mask_infer_btn.setEnabled(len(self.masking_all_results) > 0)
            if hasattr(self, "frag_mask_assign_btn"):
                self.frag_mask_assign_btn.setEnabled(len(self.masking_all_results) > 0)
            if hasattr(self, "frag_mask_review_btn"):
                self.frag_mask_review_btn.setEnabled(len(getattr(self, "masking_fragments", [])) > 0)
            if self.masking_all_results:
                self.frag_mask_result_combo.setCurrentIndex(0)
                self._display_masking_payload(self.masking_all_results[0])
            self.statusBar().showMessage(
                f"Masking attribution completed for {self.masking_batch_total} candidates"
            )
            self._fire_masking_batch_callbacks(ok=True, error=None)
            return

        row = int(self.masking_batch_rows[self.masking_batch_cursor])
        r = self.result.iloc[row]
        smiles = str(r.get("smiles", "")).strip()
        vir = r.get("Vir_shifts")
        ctype = r.get("Ctype")
        if not smiles:
            raise ValueError(f"Candidate #{row + 1} has empty SMILES.")

        self.masking_thread = MaskingAttributionThread(
            smiles, vir, ctype, self.masking_batch_exp, row, self.masking_batch_options
        )
        self.masking_thread.progress.connect(self._on_masking_batch_progress)
        self.masking_thread.finished.connect(self._on_masking_batch_item_finished)
        self.masking_thread.error.connect(self.fragment_masking_error)
        self.masking_thread.start()

    def _on_masking_batch_progress(self, p):
        if self.masking_batch_total <= 0:
            self.frag_progress.setValue(int(p))
            return
        done = float(self.masking_batch_cursor)
        val = int(((done + float(p) / 100.0) / float(self.masking_batch_total)) * 100.0)
        self.frag_progress.setValue(max(0, min(100, val)))

    def _on_masking_batch_item_finished(self, payload):
        self.masking_all_results.append(payload)
        self.masking_batch_cursor += 1
        self.frag_mask_status.setText(
            f"Running Monte Carlo masking… ({self.masking_batch_cursor}/{self.masking_batch_total})"
        )
        self._start_masking_batch_item()

    def run_masking_attribution(self):
        try:
            if self._is_thread_running("masking_thread"):
                QMessageBox.information(
                    self,
                    "Please wait",
                    "Masking attribution is still running. Please wait for it to finish.",
                )
                return
            if self.result is None or len(self.result) == 0:
                QMessageBox.warning(
                    self, "Warning", "No molecular analysis results. Run Molecular Analysis first."
                )
                return
            exp = self.get_experimental_data()
            if not exp:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "No experimental peaks. Use preprocessing (NMR-1D.csv) or manual peak input on Molecular Analysis.",
                )
                return

            seed_txt = self.frag_mask_seed_edit.text().strip()
            seed = int(seed_txt) if seed_txt else None

            mode = self.frag_mask_mode_combo.currentData()
            if mode is None:
                mode = "random_fraction"
            options_dict = {
                "mask_fraction": float(self.frag_mask_fraction.value()),
                "n_iterations": int(self.frag_mask_iterations.value()),
                "random_seed": seed,
                "use_dept_constraint": self.frag_mask_dept_check.isChecked(),
                "greedy_unique_matching": self.frag_mask_unique_match_check.isChecked(),
                "shuffle_before_each_aas_if_greedy": self.frag_mask_shuffle_aas_check.isChecked(),
                "mask_mode": str(mode),
                "connected_mask_size": int(self.frag_mask_connected_k_spin.value()),
                "tau_threshold": float(self.frag_mask_tau_spin.value())
                if hasattr(self, "frag_mask_tau_spin")
                else 3.0,
            }

            self.frag_mask_run_btn.setEnabled(False)
            self.frag_progress.setVisible(True)
            self.frag_progress.setValue(0)
            n_total = len(self.result)
            topn = min(int(self.frag_mask_topn_spin.value()), n_total)
            self.masking_all_results = []
            self.masking_fragments_all = []
            self.masking_fragments_global_rows = []
            self.masking_fragments = []
            self.fusion_results = []
            self.frag_fusion_table.setRowCount(0)
            if hasattr(self, "frag_mask_global_table"):
                self.frag_mask_global_table.setRowCount(0)
            if hasattr(self, "frag_mask_global_review_btn"):
                self.frag_mask_global_review_btn.setEnabled(False)
            if hasattr(self, "frag_mask_global_struct_label"):
                self._render_global_fragments_score_plot()
            if hasattr(self, "frag_intra_fusion_run_btn"):
                self.frag_intra_fusion_run_btn.setEnabled(False)
            self.masking_batch_rows = list(range(topn))
            self.masking_batch_cursor = 0
            self.masking_batch_total = topn
            self.masking_batch_exp = exp
            self.masking_batch_options = options_dict
            self.frag_mask_result_combo.clear()
            self.frag_mask_result_combo.setEnabled(False)
            if hasattr(self, "frag_mask_infer_btn"):
                self.frag_mask_infer_btn.setEnabled(False)
            if hasattr(self, "frag_mask_assign_btn"):
                self.frag_mask_assign_btn.setEnabled(False)
            if hasattr(self, "frag_mask_review_btn"):
                self.frag_mask_review_btn.setEnabled(False)
            self.frag_mask_status.setText(f"Running Monte Carlo masking… (0/{topn})")
            self._start_masking_batch_item()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.frag_mask_run_btn.setEnabled(True)
            self.frag_progress.setVisible(False)

    def _masking_color_scale_bounds(self):
        """Return (vmin, vmax) for fixed scale, or (None, None) for per-molecule auto."""
        if not getattr(self, "frag_mask_color_fixed_check", None):
            return None, None
        if not self.frag_mask_color_fixed_check.isChecked():
            return None, None
        t = float(self.frag_mask_color_halfrange.value())
        if t <= 0:
            return None, None
        return -t, t

    def _refresh_masking_structure_image(self):
        """Redraw attribution structure from cached payload (color scale + label options)."""
        if not getattr(self, "masking_result_data", None):
            return
        p = self.masking_result_data
        idx_cb = getattr(self, "frag_mask_show_atom_idx_check", None)
        shift_cb = getattr(self, "frag_mask_show_pred_shift_check", None)
        show_idx = idx_cb.isChecked() if idx_cb else False
        show_shift = shift_cb.isChecked() if shift_cb else False
        rotate_deg, flip_h, flip_v = self._mask_structure_transform_params()
        try:
            from VirMolAnalyte.masking_aas_attribution import draw_mol_attribution_png

            vmin, vmax = self._masking_color_scale_bounds()
            pil_img = draw_mol_attribution_png(
                p["smiles"],
                p["carbon_atom_indices"],
                p["mean_scores"],
                p.get("vir_shifts", []),
                score_vmin=vmin,
                score_vmax=vmax,
                show_rdkit_atom_index=show_idx,
                show_pred_shift_label=show_shift,
                rotate_deg=rotate_deg,
                flip_horizontal=flip_h,
                flip_vertical=flip_v,
            )
            pix = self.pil_image_to_pixmap(pil_img)
            self.frag_mask_image_label.setPixmap(pix)
            self.frag_mask_image_label.setText("")
        except Exception as e:
            self.frag_mask_image_label.setText(f"Image error: {e}")
            raise

    def apply_masking_color_scale(self):
        if not getattr(self, "masking_result_data", None):
            return
        try:
            self._refresh_masking_structure_image()
            vmin, vmax = self._masking_color_scale_bounds()
            if vmin is None:
                self.frag_mask_status.setText(
                    self.frag_mask_status.text().split("  |  Color")[0]
                    + "  |  Color: per-molecule auto"
                )
            else:
                base = self.frag_mask_status.text().split("  |  Color")[0]
                self.frag_mask_status.setText(
                    f"{base}  |  Color: fixed [{vmin:g}, {vmax:g}]"
                )
        except Exception as e:
            QMessageBox.warning(self, "Recolor", str(e))

    def _current_selected_mask_fragment(self):
        idx = self.frag_mask_frag_table.currentRow() if hasattr(self, "frag_mask_frag_table") else -1
        if idx < 0 or idx >= len(getattr(self, "masking_fragments", [])):
            return None
        return self.masking_fragments[idx]

    def _build_matched_ppm_by_atom(self, frag):
        pairs = frag.get("pairs", []) if frag else []
        best = {}
        for pr in pairs:
            try:
                aid = int(pr.get("atom_index"))
                exp_ppm = float(pr.get("exp_ppm"))
                abs_err = float(pr.get("abs_err", 1e18))
            except Exception:
                continue
            prev = best.get(aid)
            if prev is None or abs_err < prev[0]:
                best[aid] = (abs_err, exp_ppm)
        return {aid: v[1] for aid, v in best.items()}

    def _render_mask_fragment_mapping_axes(self, ax, frag):
        if frag is None:
            ax.text(
                0.5,
                0.5,
                "No fragment selected.\nRun extraction first.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#6c757d",
            )
            ax.set_axis_off()
            return

        exp = self.get_experimental_data() or []
        exp_ppm = [float(p.get("ppm", 0.0)) for p in exp]
        if exp_ppm:
            ax.vlines(exp_ppm, 0.25, 0.95, color="#c8ced4", linewidth=1.0, alpha=0.7)
            for p in exp_ppm:
                ax.scatter([p], [0.95], s=18, color="#adb5bd", zorder=2)

        pairs = frag.get("pairs", [])
        pred_ppm = [float(p.get("pred_ppm", 0.0)) for p in pairs]
        exp_match = [float(p.get("exp_ppm", 0.0)) for p in pairs]
        atom_idx = [int(p.get("atom_index", -1)) for p in pairs]
        for x in pred_ppm:
            ax.scatter([x], [-0.85], s=28, color="#1f77b4", zorder=4)
            ax.vlines([x], -0.85, -0.25, color="#1f77b4", linewidth=1.0, alpha=0.65)
        for x in exp_match:
            ax.scatter([x], [0.95], s=34, color="#d9480f", zorder=5)
        for x0, x1 in zip(pred_ppm, exp_match):
            ax.plot([x0, x1], [-0.25, 0.25], color="#fa5252", alpha=0.55, linewidth=1.1)
        for x, a in zip(pred_ppm, atom_idx):
            ax.text(x, -0.98, f"C{a}", ha="center", va="top", fontsize=8, color="#1f77b4")

        all_ppm = exp_ppm + pred_ppm + exp_match
        if all_ppm:
            xmin, xmax = min(all_ppm) - 5.0, max(all_ppm) + 5.0
        else:
            xmin, xmax = 0.0, 220.0
        ax.set_xlim(xmax, xmin)  # NMR style: high ppm on the left
        ax.set_ylim(-1.15, 1.15)
        ax.axhline(0, color="#868e96", linewidth=1.0)
        ax.set_yticks([-0.85, 0.95])
        ax.set_yticklabels(["Fragment δ_pred", "Experimental peaks"])
        ax.set_xlabel("Chemical Shift (ppm)")
        ax.grid(axis="x", linestyle="--", alpha=0.2)
        mse_txt = f"{float(frag.get('match_mse', float('nan'))):.4f}"
        ax.set_title(
            f"Fragment #{frag.get('fragment_id')} | N={frag.get('n_carbons')} | "
            f"MAE={float(frag.get('mean_abs_err', float('nan'))):.3f} ppm | ΣΔδ²={mse_txt}",
            fontsize=10,
        )

    def _render_mask_fragment_mapping_plot(self, fig, frag):
        fig.clear()
        ax = fig.add_subplot(111)
        self._render_mask_fragment_mapping_axes(ax, frag)

    def _refresh_masking_mapping_dialog(self):
        dlg = getattr(self, "masking_mapping_dialog", None)
        if dlg is None or not getattr(self, "masking_result_data", None):
            return

        idx = int(getattr(dlg, "_fragment_index", -1))
        total = len(getattr(self, "masking_fragments", []))
        if idx < 0 or idx >= total:
            return
        frag = self.masking_fragments[idx]
        payload = self.masking_result_data
        if hasattr(dlg, "_index_label"):
            dlg._index_label.setText(f"{idx + 1}/{total}")
        if hasattr(dlg, "_prev_btn"):
            dlg._prev_btn.setEnabled(idx > 0)
        if hasattr(dlg, "_next_btn"):
            dlg._next_btn.setEnabled(idx < total - 1)

        try:
            from VirMolAnalyte.masking_aas_attribution import draw_mol_attribution_png

            vmin, vmax = self._masking_color_scale_bounds()
            matched_ppm = self._build_matched_ppm_by_atom(frag)
            rotate_deg, flip_h, flip_v = self._mask_structure_transform_params()

            pil_img_pred = draw_mol_attribution_png(
                payload["smiles"],
                payload["carbon_atom_indices"],
                payload["mean_scores"],
                payload.get("vir_shifts", []),
                score_vmin=vmin,
                score_vmax=vmax,
                show_rdkit_atom_index=dlg._show_atom_idx.isChecked(),
                show_pred_shift_label=dlg._show_pred_shift.isChecked(),
                show_matched_shift_label=False,
                rotate_deg=rotate_deg,
                flip_horizontal=flip_h,
                flip_vertical=flip_v,
            )
            pil_img_match = draw_mol_attribution_png(
                payload["smiles"],
                payload["carbon_atom_indices"],
                payload["mean_scores"],
                payload.get("vir_shifts", []),
                score_vmin=vmin,
                score_vmax=vmax,
                show_rdkit_atom_index=dlg._show_atom_idx.isChecked(),
                show_pred_shift_label=False,
                show_matched_shift_label=dlg._show_matched_shift.isChecked(),
                matched_ppm_by_atom=matched_ppm,
                rotate_deg=rotate_deg,
                flip_horizontal=flip_h,
                flip_vertical=flip_v,
            )
            dlg._structure_label_pred.setPixmap(self.pil_image_to_pixmap(pil_img_pred))
            dlg._structure_label_pred.setText("")
            dlg._structure_label_match.setPixmap(self.pil_image_to_pixmap(pil_img_match))
            dlg._structure_label_match.setText("")
        except Exception as e:
            dlg._structure_label_pred.setText(f"Structure draw error: {e}")
            dlg._structure_label_match.setText(f"Structure draw error: {e}")

        self._render_mask_fragment_mapping_plot(dlg._map_canvas.figure, frag)
        dlg._map_canvas.draw()
        dlg._title_label.setText(
            f"Fragment #{frag.get('fragment_id')}  |  N={frag.get('n_carbons')}  |  "
            f"MAE={float(frag.get('mean_abs_err', float('nan'))):.3f} ppm"
        )

    def _json_safe_export(self, obj):
        """Convert nested structures to JSON-serializable forms."""
        if obj is None:
            return None
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, str):
            return obj
        if isinstance(obj, (np.integer, int)) and type(obj) is not bool:
            return int(obj)
        if isinstance(obj, (float, np.floating)):
            v = float(obj)
            return None if not np.isfinite(v) else v
        if isinstance(obj, np.ndarray):
            return self._json_safe_export(obj.tolist())
        if isinstance(obj, dict):
            return {str(k): self._json_safe_export(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._json_safe_export(x) for x in obj]
        if isinstance(obj, set):
            return [self._json_safe_export(x) for x in sorted(obj, key=lambda x: str(x))]
        return str(obj)

    def _collect_analysis_params_base(self):
        """Snapshot of GUI analysis settings (database, filters, evaluator, peaks)."""
        from datetime import datetime

        peak_manual = ""
        if hasattr(self, "manual_peak_input"):
            peak_manual = self.manual_peak_input.toPlainText().strip()
        nmr_path = self._nmr_csv_path()
        nmr_exists = os.path.isfile(nmr_path)
        if peak_manual:
            peak_source = "manual_peak_input"
        elif nmr_exists:
            peak_source = "NMR-1D.csv"
        else:
            peak_source = "none"

        payload = {
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "application": "VirMolAnalyte GUI",
            "database": {},
            "filters": {},
            "evaluator": {},
            "run_params": {},
            "peak_input": {
                "source": peak_source,
                "manual_peak_text": peak_manual if peak_manual else None,
                "nmr_1d_csv_resolved_path": nmr_path if nmr_exists else None,
            },
            "results": {},
            "output_files_note": (
                "After molecular analysis, results are written to "
                f"{self._gui_output_path('Result.csv')}."
            ),
        }

        if hasattr(self, "db_combo"):
            payload["database"]["combo_selection"] = self.db_combo.currentText()
        if hasattr(self, "other_db_input"):
            odp = self.other_db_input.text().strip()
            payload["database"]["other_database_path"] = odp or None
        payload["database"]["database_loaded_in_memory"] = getattr(self, "database1", None) is not None

        if hasattr(self, "cnf_checkbox"):
            payload["filters"]["CNF"] = self.cnf_checkbox.isChecked()
            payload["filters"]["CNF_bias"] = self.cnf_bias.value()
        if hasattr(self, "ctnf_checkbox"):
            payload["filters"]["CTNF"] = self.ctnf_checkbox.isChecked()
            payload["filters"]["CTNF_bias"] = self.ctnf_bias.value()
        if hasattr(self, "mw_checkbox"):
            payload["filters"]["MW"] = self.mw_checkbox.isChecked()
        if hasattr(self, "mw_list"):
            try:
                payload["filters"]["MW_list"] = [
                    float(x.strip()) for x in self.mw_list.text().split(",") if x.strip()
                ]
            except Exception:
                payload["filters"]["MW_list_raw"] = self.mw_list.text()

        evaluator = None
        if hasattr(self, "evaluator_css") and self.evaluator_css.isChecked():
            evaluator = "CSS"
        elif hasattr(self, "evaluator_aas") and self.evaluator_aas.isChecked():
            evaluator = "AAS"
        elif hasattr(self, "evaluator_fps") and self.evaluator_fps.isChecked():
            evaluator = "FPS"
        elif hasattr(self, "evaluator_fpaacs") and self.evaluator_fpaacs.isChecked():
            evaluator = "FPAACS"
        payload["evaluator"]["selected"] = evaluator

        if hasattr(self, "fpaacs_weights"):
            try:
                payload["evaluator"]["fpaacs_weights"] = [
                    float(x.strip()) for x in self.fpaacs_weights.text().split(",") if x.strip()
                ]
            except Exception:
                payload["evaluator"]["fpaacs_weights_raw"] = self.fpaacs_weights.text()

        payload["run_params"]["TopN"] = 80
        payload["run_params"]["MWbias"] = 5
        if hasattr(self, "cnf_bias"):
            payload["run_params"]["CNFbias"] = self.cnf_bias.value()
        if hasattr(self, "ctnf_bias"):
            payload["run_params"]["CTNFbias"] = self.ctnf_bias.value()
        if hasattr(self, "mw_list"):
            try:
                payload["run_params"]["MWlist"] = [
                    float(x.strip()) for x in self.mw_list.text().split(",") if x.strip()
                ]
            except Exception:
                pass
        if hasattr(self, "fpaacs_weights"):
            try:
                payload["run_params"]["weights"] = [
                    float(x.strip()) for x in self.fpaacs_weights.text().split(",") if x.strip()
                ]
            except Exception:
                pass

        if hasattr(self, "result") and self.result is not None:
            try:
                payload["results"]["molecular_analysis_result_rows"] = len(self.result)
            except Exception:
                payload["results"]["molecular_analysis_result_rows"] = None

        frag_ui = {}
        for name in (
            "frag_fusion_pool_topn",
            "frag_fusion_max_merge",
            "frag_fusion_shuffle_repeats",
            "frag_fusion_cov_weight",
            "frag_intra_fusion_pool_topn",
            "frag_intra_fusion_max_merge",
            "frag_intra_fusion_shuffle_repeats",
            "frag_intra_fusion_cov_weight",
        ):
            w = getattr(self, name, None)
            if w is not None:
                try:
                    frag_ui[name] = w.value()
                except Exception:
                    pass
        if frag_ui:
            payload["fragment_analysis_ui"] = frag_ui

        return payload

    def _append_fragment_methodology_sentences(self, sentences):
        """Append English sentences for Fragment Analysis: MC masking, extraction, intra/cross fusion."""
        if not hasattr(self, "frag_mask_topn_spin"):
            return

        k_top = int(self.frag_mask_topn_spin.value())
        _mode_for_intro = (
            self.frag_mask_mode_combo.currentData()
            if hasattr(self, "frag_mask_mode_combo")
            else "random_fraction"
        )
        if _mode_for_intro == "threshold_baseline":
            intro_method = "deterministic threshold-baseline (Compare_CSS)"
        else:
            intro_method = "random-mask Monte Carlo"
        sentences.append(
            f"Fragment Analysis (downstream of screening) was configured to run {intro_method} "
            f"¹³C attribution on up to the top {k_top} ranked candidates from the hit list."
        )

        mode = self.frag_mask_mode_combo.currentData()
        if mode is None:
            mode = "random_fraction"
        n_iter = int(self.frag_mask_iterations.value())
        seed_txt = self.frag_mask_seed_edit.text().strip()
        seed_phrase = (
            f"a fixed Monte Carlo RNG seed ({seed_txt})"
            if seed_txt
            else "no fixed RNG seed (stochastic runs)"
        )

        if mode == "threshold_baseline":
            tau = (
                float(self.frag_mask_tau_spin.value())
                if hasattr(self, "frag_mask_tau_spin")
                else 3.0
            )
            sentences.append(
                f"Per-carbon scoring used the deterministic threshold baseline (Compare_CSS-style) "
                f"with no Monte Carlo: each predicted carbon was greedily matched to an experimental "
                f"peak and scored as τ − |Δδ| (τ = {tau:.2f} ppm); unmatched carbons received a "
                f"non-finite score and were excluded from the downstream fragment extractor."
            )
        elif mode == "random_connected":
            k_bonded = int(self.frag_mask_connected_k_spin.value())
            sentences.append(
                f"Each Monte Carlo iteration masked a random connected subgraph of {k_bonded} carbon atoms; "
                f"AAS-style spectrum matching was evaluated on the unmasked region "
                f"({n_iter} iterations; {seed_phrase})."
            )
        else:
            frac = float(self.frag_mask_fraction.value())
            sentences.append(
                f"Each Monte Carlo iteration masked a random fraction f = {frac} of carbons "
                f"(ceil(f·N) atoms); AAS-style matching was evaluated on the unmasked region "
                f"({n_iter} iterations; {seed_phrase})."
            )

        mc_bits = []
        if self.frag_mask_dept_check.isChecked():
            mc_bits.append("DEPT multiplicity (q/t/d/s) consistency in peak comparisons")
        if self.frag_mask_unique_match_check.isChecked():
            g = "greedy unique peak matching between predicted and experimental peaks"
            if mode != "threshold_baseline" and self.frag_mask_shuffle_aas_check.isChecked():
                g += ", with optional shuffle-before-each-AAS pass"
            mc_bits.append(g)
        if mc_bits:
            label = (
                "Per-carbon scoring options included"
                if mode == "threshold_baseline"
                else "Monte Carlo masking options included"
            )
            sentences.append(label + ": " + "; ".join(mc_bits) + ".")

        tm = self.frag_mask_thresh_mode.currentData()
        if tm == "manual":
            thr = float(self.frag_mask_frag_thresh.value())
            thr_txt = f"a manual per-carbon score cutoff (> {thr})"
        else:
            rk = float(self.frag_mask_robust_k.value())
            thr_txt = (
                f"a robust Z-style cutoff (median + {rk} × 1.4826 × MAD of per-carbon scores)"
            )
        nmin = int(self.frag_mask_frag_min_c.value())
        sentences.append(
            f"Positive fragments were extracted by thresholding high-scoring carbons using {thr_txt}, "
            f"retaining connected clusters with at least {nmin} carbons."
        )

        if self.frag_mask_bridge_check.isChecked():
            nb = int(self.frag_mask_bridge_max_low.value())
            sentences.append(
                f"High-score clusters could be merged across up to {nb} intervening low-score carbon "
                f"'bridge' atom(s), which were included in fragment-level spectrum matching."
            )
        else:
            sentences.append(
                "Bridging of clusters through low-score carbons was turned off for fragment construction."
            )

        ext_bits = []
        if self.frag_mask_extract_dept_check.isChecked():
            ext_bits.append("fragment↔experimental matching enforced DEPT-type agreement")
        if self.frag_mask_extract_greedy_check.isChecked():
            rs = int(self.frag_mask_greedy_shuffle_repeats.value())
            seed_g = self.frag_mask_greedy_shuffle_seed.text().strip()
            sg = f"; shuffle RNG seed {seed_g}" if seed_g else ""
            ext_bits.append(
                f"greedy unique peak assignment with {rs} random-order repeat(s) minimizing Σ(Δδ)²{sg}"
            )
        if ext_bits:
            sentences.append("Fragment-to-spectrum mapping for extracted fragments used: " + "; ".join(ext_bits) + ".")

        max_fr = int(self.frag_mask_max_frags.value())
        sentences.append(
            f"After ranking fragments by total attribution score, at most {max_fr} fragments were listed "
            f"for inspection and fusion."
        )

        fp = int(self.frag_fusion_pool_topn.value())
        fm = int(self.frag_fusion_max_merge.value())
        fs = int(self.frag_fusion_shuffle_repeats.value())
        fcw = float(self.frag_fusion_cov_weight.value())
        sentences.append(
            f"Cross-compound fragment fusion (optional) explored combinations drawn from the top {fp} "
            f"fragments by score, allowing up to {fm} merged fragments per combination, "
            f"{fs} greedy shuffle repeats, and coverage weight λ = {fcw} in the objective."
        )

        ip = int(self.frag_intra_fusion_pool_topn.value())
        im = int(self.frag_intra_fusion_max_merge.value())
        isr = int(self.frag_intra_fusion_shuffle_repeats.value())
        icw = float(self.frag_intra_fusion_cov_weight.value())
        sentences.append(
            f"Intra-molecular fragment fusion (optional, single candidate) used the same fusion kernel "
            f"with pool top-{ip}, max merge {im}, {isr} shuffle repeats, and λ = {icw}."
        )

    def _build_methodology_paragraph(self):
        """English manuscript-style summary of current Molecular Analysis (and related GUI) settings."""
        p = self._collect_analysis_params_base()
        sentences = []

        sentences.append(
            "Automated ¹³C NMR-based molecular candidate screening was performed using "
            "VirMolAnalyte (GUI workflow)."
        )

        db = p.get("database") or {}
        combo = db.get("combo_selection") or "unspecified database preset"
        sentences.append(f"The compound collection was specified as: {combo}.")
        if db.get("database_loaded_in_memory"):
            sentences.append("The corresponding structure library was loaded into memory before screening.")
        else:
            sentences.append(
                "Note: at the time this text was generated, no database was loaded in memory; "
                "load a library before running analysis or revise this sentence."
            )

        odp = db.get("other_database_path")
        if odp:
            sentences.append(
                f"A user-provided library file was additionally referenced ({os.path.basename(odp)})."
            )

        pk = p.get("peak_input") or {}
        src = pk.get("source")
        if src == "manual_peak_input":
            sentences.append(
                "Experimental ¹³C chemical shifts and DEPT-style multiplicity labels (q, t, d, s) "
                "were supplied via the manual peak list in the software."
            )
        elif src == "NMR-1D.csv":
            npath = pk.get("nmr_1d_csv_resolved_path") or "NMR-1D.csv"
            sentences.append(
                f"Experimental peaks were read from the preprocessed file ({os.path.basename(npath)}) "
                "in the working directory after spectral preprocessing."
            )
        else:
            sentences.append(
                "Experimental peak input was not yet configured (neither a non-empty manual list nor "
                "NMR-1D.csv); configure peaks before analysis or edit this paragraph manually."
            )

        flt = p.get("filters") or {}
        rp = p.get("run_params") or {}
        mwbias = rp.get("MWbias", 5)
        bits = []
        if flt.get("CNF"):
            bits.append(f"carbon count filter (CNF, bias = {flt.get('CNF_bias')})")
        if flt.get("CTNF"):
            bits.append(f"carbon-type-count filter (CTNF, bias = {flt.get('CTNF_bias')})")
        if flt.get("MW"):
            mwlist = flt.get("MW_list")
            if isinstance(mwlist, list) and mwlist:
                mw_parts = []
                for x in mwlist:
                    xf = float(x)
                    mw_parts.append(str(int(xf)) if abs(xf - round(xf)) < 1e-9 else str(xf))
                mw_str = ", ".join(mw_parts)
                bits.append(
                    f"molecular-weight band filter (MW, reference values {mw_str} Da; MW bias = {mwbias})"
                )
            else:
                bits.append(f"molecular-weight filter (MW; MW bias = {mwbias})")
        if bits:
            sentences.append("Database pre-filtering comprised: " + "; ".join(bits) + ".")
        else:
            sentences.append("No structural pre-filters (CNF / CTNF / MW) were enabled.")

        ev = (p.get("evaluator") or {}).get("selected")
        topn = rp.get("TopN", 80)
        if ev == "FPAACS":
            w = rp.get("weights") or (p.get("evaluator") or {}).get("fpaacs_weights")
            if isinstance(w, list) and len(w) >= 3:
                sentences.append(
                    f"Candidates were ranked with the FPAACS composite score using weights "
                    f"w₁–w₃ = {w[0]}, {w[1]}, and {w[2]} (as set in the interface); "
                    f"the top {topn} hits were retained."
                )
            else:
                sentences.append(
                    f"Candidates were ranked with the FPAACS composite score; "
                    f"the top {topn} hits were retained."
                )
        elif ev == "CSS":
            th = None
            if hasattr(self, "css_threshold"):
                try:
                    th = float(self.css_threshold.value())
                except Exception:
                    th = None
            if th is not None:
                sentences.append(
                    f"Candidates were ranked using the CSS score (threshold τ = {th} in the interface); "
                    f"the top {topn} structures were retained."
                )
            else:
                sentences.append(
                    f"Candidates were ranked using the CSS score; the top {topn} structures were retained."
                )
        elif ev == "AAS":
            sentences.append(
                f"Candidates were ranked using the AAS score; the top {topn} hits were retained."
            )
        elif ev == "FPS":
            sentences.append(
                f"Candidates were ranked using the FPS score; the top {topn} hits were retained."
            )
        else:
            sentences.append(
                "The scoring backend was not determined from the current GUI state; "
                "describe the evaluator manually."
            )

        self._append_fragment_methodology_sentences(sentences)

        res = p.get("results") or {}
        nrows = res.get("molecular_analysis_result_rows")
        if isinstance(nrows, int) and nrows >= 0:
            sentences.append(
                f"After the last completed screen, the ranked table contained {nrows} entries "
                f"({self._gui_output_path('Result.csv')})."
            )

        return " ".join(sentences)

    def copy_methodology_to_clipboard(self):
        """Copy English methods paragraph to the system clipboard."""
        try:
            text = self._build_methodology_paragraph()
            QApplication.clipboard().setText(text)
            self.statusBar().showMessage("Methods description (English) copied to clipboard.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to build or copy methods text: {str(e)}")
            self.statusBar().showMessage("Failed to copy methods text")

    def _write_analysis_params_sidecar(self, image_path, export_kind, extra=None):
        """Write `{stem}_analysis_params.json` next to an exported image."""
        json_path = os.path.splitext(image_path)[0] + "_analysis_params.json"
        try:
            payload = self._collect_analysis_params_base()
            payload["export"] = {"kind": export_kind}
            if extra is not None:
                payload["export"]["detail"] = self._json_safe_export(extra)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            return json_path
        except Exception as e:
            print(f"Failed to write analysis parameters sidecar: {e}")
            return None

    def _fit_svg_to_physical_width(self, svg_text, target_width_mm=70.0):
        """Set SVG physical width (mm), preserving aspect ratio via viewBox."""
        if not isinstance(svg_text, str) or "<svg" not in svg_text:
            return svg_text
        m = re.search(r'viewBox\\s*=\\s*"\\s*[-+0-9.eE]+\\s+[-+0-9.eE]+\\s+([-+0-9.eE]+)\\s+([-+0-9.eE]+)\\s*"', svg_text)
        if m is None:
            return svg_text
        try:
            vb_w = float(m.group(1))
            vb_h = float(m.group(2))
            if vb_w <= 1e-12 or vb_h <= 1e-12:
                return svg_text
            width_mm = float(target_width_mm)
            height_mm = width_mm * (vb_h / vb_w)
            svg_text = re.sub(r'\\swidth\\s*=\\s*"[^"]*"', "", svg_text, count=1)
            svg_text = re.sub(r'\\sheight\\s*=\\s*"[^"]*"', "", svg_text, count=1)
            svg_text = re.sub(
                r"<svg\\b",
                f'<svg width="{width_mm:.2f}mm" height="{height_mm:.2f}mm"',
                svg_text,
                count=1,
            )
            return svg_text
        except Exception:
            return svg_text

    def _export_masking_mapping_hd_image(self):
        dlg = getattr(self, "masking_mapping_dialog", None)
        if dlg is None or not getattr(self, "masking_result_data", None):
            QMessageBox.warning(self, "Export", "No popup mapping data to export.")
            return
        idx = int(getattr(dlg, "_fragment_index", -1))
        if idx < 0 or idx >= len(getattr(self, "masking_fragments", [])):
            QMessageBox.warning(self, "Export", "No fragment selected.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export HD mapping image",
            f"fragment_{idx + 1}_mapping_hd.png",
            "PNG Files (*.png);;TIFF Files (*.tif *.tiff);;JPEG Files (*.jpg *.jpeg);;SVG Files (*.svg);;PDF Files (*.pdf)",
        )
        if not file_path:
            return

        frag = self.masking_fragments[idx]
        payload = self.masking_result_data
        try:
            from VirMolAnalyte.masking_aas_attribution import draw_mol_attribution_png

            vmin, vmax = self._masking_color_scale_bounds()
            matched_ppm = self._build_matched_ppm_by_atom(frag)
            rotate_deg, flip_h, flip_v = self._mask_structure_transform_params()
            # Render larger structure images for high-resolution export.
            pred_img = draw_mol_attribution_png(
                payload["smiles"],
                payload["carbon_atom_indices"],
                payload["mean_scores"],
                payload.get("vir_shifts", []),
                size=(1900, 1300),
                score_vmin=vmin,
                score_vmax=vmax,
                show_rdkit_atom_index=dlg._show_atom_idx.isChecked(),
                show_pred_shift_label=dlg._show_pred_shift.isChecked(),
                show_matched_shift_label=False,
                rotate_deg=rotate_deg,
                flip_horizontal=flip_h,
                flip_vertical=flip_v,
            )
            match_img = draw_mol_attribution_png(
                payload["smiles"],
                payload["carbon_atom_indices"],
                payload["mean_scores"],
                payload.get("vir_shifts", []),
                size=(1900, 1300),
                score_vmin=vmin,
                score_vmax=vmax,
                show_rdkit_atom_index=dlg._show_atom_idx.isChecked(),
                show_pred_shift_label=False,
                show_matched_shift_label=dlg._show_matched_shift.isChecked(),
                matched_ppm_by_atom=matched_ppm,
                rotate_deg=rotate_deg,
                flip_horizontal=flip_h,
                flip_vertical=flip_v,
            )

            fig = Figure(figsize=(12.5, 14.5), dpi=150)
            gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.45, 1.15], hspace=0.10)
            ax_pred = fig.add_subplot(gs[0, 0])
            ax_match = fig.add_subplot(gs[1, 0])
            ax_map = fig.add_subplot(gs[2, 0])

            ax_pred.imshow(np.asarray(pred_img))
            ax_pred.set_title("Predicted δ on structure", fontsize=13, fontweight="bold")
            ax_pred.axis("off")
            ax_match.imshow(np.asarray(match_img))
            ax_match.set_title("Matched experimental ppm on structure", fontsize=13, fontweight="bold")
            ax_match.axis("off")

            self._render_mask_fragment_mapping_axes(ax_map, frag)

            fig.suptitle(
                dlg._title_label.text(),
                fontsize=14,
                fontweight="bold",
                y=0.985,
            )
            lower = file_path.lower()
            if lower.endswith(".svg"):
                fig.savefig(file_path, format="svg", bbox_inches="tight")
            elif lower.endswith(".pdf"):
                fig.savefig(file_path, format="pdf", bbox_inches="tight")
            else:
                fig.savefig(file_path, dpi=600, bbox_inches="tight")
            frag_summary = {
                k: frag.get(k)
                for k in (
                    "fragment_id",
                    "n_carbons",
                    "mean_abs_err",
                    "compound_idx",
                    "mse",
                    "coverage",
                )
                if k in frag
            }
            extra_detail = {
                "masking_fragment_table_index": idx,
                "fragment_summary": self._json_safe_export(frag_summary),
                "masking_smiles": payload.get("smiles") if isinstance(payload, dict) else None,
                "export_dialog": {
                    "show_atom_idx": dlg._show_atom_idx.isChecked(),
                    "show_pred_shift": dlg._show_pred_shift.isChecked(),
                    "show_matched_shift": dlg._show_matched_shift.isChecked(),
                    "rotate_deg": rotate_deg,
                    "flip_horizontal": flip_h,
                    "flip_vertical": flip_v,
                },
            }
            param_path = self._write_analysis_params_sidecar(
                file_path, "masking_mapping_hd", extra=extra_detail
            )
            self.statusBar().showMessage(f"HD image exported: {file_path}")
            msg = f"HD image exported:\n{file_path}"
            if param_path:
                msg += f"\n\nAnalysis parameters:\n{param_path}"
            else:
                msg += "\n\n(Could not save analysis parameters JSON; see console.)"
            QMessageBox.information(self, "Export", msg)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def _export_mask_structure_hd_image(self):
        """Export main attribution structure as SVG (preferred) or high-res raster."""
        payload = getattr(self, "masking_result_data", None)
        if not payload:
            QMessageBox.warning(self, "Export", "No attribution structure to export.")
            return
        default_name = "attribution_structure.svg"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save attribution structure",
            default_name,
            "SVG Files (*.svg);;PNG Files (*.png);;TIFF Files (*.tif *.tiff);;JPEG Files (*.jpg *.jpeg)",
        )
        if not file_path:
            return
        try:
            from VirMolAnalyte.masking_aas_attribution import (
                draw_mol_attribution_png,
                draw_mol_attribution_svg,
            )

            vmin, vmax = self._masking_color_scale_bounds()
            rotate_deg, flip_h, flip_v = self._mask_structure_transform_params()
            show_idx = bool(getattr(self, "frag_mask_show_atom_idx_check", None) and self.frag_mask_show_atom_idx_check.isChecked())
            show_shift = bool(getattr(self, "frag_mask_show_pred_shift_check", None) and self.frag_mask_show_pred_shift_check.isChecked())
            lower = file_path.lower()
            if lower.endswith(".svg"):
                svg_text = draw_mol_attribution_svg(
                    payload["smiles"],
                    payload["carbon_atom_indices"],
                    payload["mean_scores"],
                    payload.get("vir_shifts", []),
                    size=(2200, 1700),
                    score_vmin=vmin,
                    score_vmax=vmax,
                    show_rdkit_atom_index=show_idx,
                    show_pred_shift_label=show_shift,
                    rotate_deg=rotate_deg,
                    flip_horizontal=flip_h,
                    flip_vertical=flip_v,
                )
                svg_text = self._fit_svg_to_physical_width(svg_text, target_width_mm=70.0)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(svg_text)
            else:
                img = draw_mol_attribution_png(
                    payload["smiles"],
                    payload["carbon_atom_indices"],
                    payload["mean_scores"],
                    payload.get("vir_shifts", []),
                    size=(2400, 1800),
                    score_vmin=vmin,
                    score_vmax=vmax,
                    show_rdkit_atom_index=show_idx,
                    show_pred_shift_label=show_shift,
                    rotate_deg=rotate_deg,
                    flip_horizontal=flip_h,
                    flip_vertical=flip_v,
                )
                img.save(file_path, dpi=(300, 300))

            extra_detail = {
                "smiles": payload.get("smiles"),
                "show_atom_idx": show_idx,
                "show_pred_shift": show_shift,
                "rotate_deg": rotate_deg,
                "flip_horizontal": flip_h,
                "flip_vertical": flip_v,
                "svg_target_width_mm": 70.0 if lower.endswith(".svg") else None,
                "color_scale_vmin": vmin,
                "color_scale_vmax": vmax,
            }
            param_path = self._write_analysis_params_sidecar(
                file_path, "masking_structure_hd", extra=extra_detail
            )
            self.statusBar().showMessage(f"Attribution structure exported: {file_path}")
            msg = f"Attribution structure exported:\n{file_path}"
            if param_path:
                msg += f"\n\nAnalysis parameters:\n{param_path}"
            QMessageBox.information(self, "Export", msg)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def _on_mask_fragment_table_clicked(self, row, _col):
        if row < 0 or row >= len(getattr(self, "masking_fragments", [])):
            return
        self.frag_mask_frag_table.selectRow(row)
        try:
            self._refresh_masking_structure_image()
        except Exception:
            pass

        if self.masking_mapping_dialog is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("NMR matching map")
            dlg.resize(1220, 780)
            lay = QVBoxLayout(dlg)
            lay.setContentsMargins(10, 10, 10, 10)

            dlg._title_label = QLabel("Fragment mapping detail")
            dlg._title_label.setStyleSheet("font-size: 17px; font-weight: bold; color: #3d5a66;")
            lay.addWidget(dlg._title_label)

            nav_row = QHBoxLayout()
            dlg._prev_btn = QPushButton("Previous")
            dlg._next_btn = QPushButton("Next")
            dlg._index_label = QLabel("")
            dlg._index_label.setStyleSheet("color: #6c757d; font-size: 14px;")
            nav_row.addWidget(dlg._prev_btn)
            nav_row.addWidget(dlg._next_btn)
            nav_row.addWidget(dlg._index_label)
            nav_row.addStretch()
            lay.addLayout(nav_row)

            opts = QHBoxLayout()
            dlg._show_atom_idx = QCheckBox("Atom index")
            dlg._show_atom_idx.setChecked(False)
            dlg._show_pred_shift = QCheckBox("Pred δ")
            dlg._show_pred_shift.setChecked(False)
            dlg._show_matched_shift = QCheckBox("Matched ppm")
            dlg._show_matched_shift.setChecked(True)
            dlg._export_hd_btn = QPushButton("Export")
            opts.addWidget(dlg._show_atom_idx)
            opts.addWidget(dlg._show_pred_shift)
            opts.addWidget(dlg._show_matched_shift)
            opts.addStretch()
            opts.addWidget(dlg._export_hd_btn)
            lay.addLayout(opts)

            struct_row = QHBoxLayout()
            struct_row.setSpacing(10)

            pred_wrap = QWidget()
            pred_l = QVBoxLayout(pred_wrap)
            pred_l.setContentsMargins(0, 0, 0, 0)
            pred_l.setSpacing(4)
            pred_title = QLabel("Predicted δ on structure")
            pred_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #3d5a66;")
            pred_l.addWidget(pred_title)
            dlg._structure_label_pred = QLabel("Predicted structure preview")
            dlg._structure_label_pred.setMinimumHeight(360)
            dlg._structure_label_pred.setAlignment(Qt.AlignCenter)
            dlg._structure_label_pred.setStyleSheet("border: 1px solid #dee2e6; background: #fff;")
            pred_l.addWidget(dlg._structure_label_pred)
            struct_row.addWidget(pred_wrap, 1)

            match_wrap = QWidget()
            match_l = QVBoxLayout(match_wrap)
            match_l.setContentsMargins(0, 0, 0, 0)
            match_l.setSpacing(4)
            match_title = QLabel("Matched experimental ppm on structure")
            match_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #3d5a66;")
            match_l.addWidget(match_title)
            dlg._structure_label_match = QLabel("Matched structure preview")
            dlg._structure_label_match.setMinimumHeight(360)
            dlg._structure_label_match.setAlignment(Qt.AlignCenter)
            dlg._structure_label_match.setStyleSheet("border: 1px solid #dee2e6; background: #fff;")
            match_l.addWidget(dlg._structure_label_match)
            struct_row.addWidget(match_wrap, 1)

            lay.addLayout(struct_row)

            dlg._map_canvas = FigureCanvas(Figure(figsize=(9.8, 3.4)))
            dlg._map_canvas.setMinimumHeight(260)
            lay.addWidget(dlg._map_canvas)

            dlg._show_atom_idx.toggled.connect(self._refresh_masking_mapping_dialog)
            dlg._show_pred_shift.toggled.connect(self._refresh_masking_mapping_dialog)
            dlg._show_matched_shift.toggled.connect(self._refresh_masking_mapping_dialog)
            dlg._export_hd_btn.clicked.connect(self._export_masking_mapping_hd_image)
            dlg._prev_btn.clicked.connect(lambda: self._navigate_masking_fragment(-1))
            dlg._next_btn.clicked.connect(lambda: self._navigate_masking_fragment(1))
            self.masking_mapping_dialog = dlg

        self.masking_mapping_dialog._fragment_index = int(row)
        self._refresh_masking_mapping_dialog()
        self.masking_mapping_dialog.show()
        self.masking_mapping_dialog.raise_()
        self.masking_mapping_dialog.activateWindow()

    def _navigate_masking_fragment(self, step):
        dlg = getattr(self, "masking_mapping_dialog", None)
        frags = list(getattr(self, "masking_fragments", []))
        if dlg is None or not frags:
            return
        idx = int(getattr(dlg, "_fragment_index", -1))
        if idx < 0:
            idx = 0
        idx = max(0, min(len(frags) - 1, idx + int(step)))
        dlg._fragment_index = idx
        if hasattr(self, "frag_mask_frag_table"):
            self.frag_mask_frag_table.selectRow(idx)
        self._refresh_masking_mapping_dialog()

    def _on_frag_mask_view_changed(self, idx):
        compound_vis = int(idx) == 0
        compound_widgets = [
            getattr(self, "frag_mask_img_title", None),
            getattr(self, "frag_mask_result_row_wrap", None),
            getattr(self, "frag_mask_struct_scores_row", None),
            getattr(self, "frag_mask_frag_table", None),
            getattr(self, "frag_mask_global_title", None),
            getattr(self, "frag_mask_global_table", None),
            getattr(self, "frag_mask_global_plot_title", None),
            getattr(self, "frag_mask_global_opts_wrap", None),
            getattr(self, "frag_mask_global_scroll", None),
            getattr(self, "frag_mask_plot_hint_label", None),
        ]
        for w in compound_widgets:
            if w is not None:
                w.setVisible(compound_vis)
        # Force hidden compound widgets to collapse height in fusion view.
        if getattr(self, "frag_mask_image_label", None) is not None:
            if compound_vis:
                self.frag_mask_image_label.setMinimumHeight(320)
                self.frag_mask_image_label.setMaximumHeight(16777215)
            else:
                self.frag_mask_image_label.setMinimumHeight(0)
                self.frag_mask_image_label.setMaximumHeight(0)
        if getattr(self, "frag_mask_scores_table", None) is not None:
            if compound_vis:
                self.frag_mask_scores_table.setMinimumHeight(300)
                self.frag_mask_scores_table.setMaximumHeight(16777215)
            else:
                self.frag_mask_scores_table.setMinimumHeight(0)
                self.frag_mask_scores_table.setMaximumHeight(0)
        if getattr(self, "frag_mask_frag_table", None) is not None:
            if compound_vis:
                self.frag_mask_frag_table.setMinimumHeight(140)
                self.frag_mask_frag_table.setMaximumHeight(16777215)
            else:
                self.frag_mask_frag_table.setMinimumHeight(0)
                self.frag_mask_frag_table.setMaximumHeight(0)
        fusion_vis = not compound_vis
        for w in [
            getattr(self, "frag_fusion_title", None),
            getattr(self, "frag_fusion_toolbar", None),
            getattr(self, "frag_fusion_table", None),
            getattr(self, "frag_fusion_preview_card", None),
        ]:
            if w is not None:
                w.setVisible(fusion_vis)
        if fusion_vis:
            self._update_fusion_ai_top5_btn_state()

    def _on_fusion_fragment_structs_toggled(self, _checked=False):
        if getattr(self, "fusion_detail_dialog", None) and self.fusion_detail_dialog.isVisible():
            self._refresh_fusion_detail_dialog()

    def _current_fusion_row_index(self):
        idx = -1
        if hasattr(self, "frag_fusion_table"):
            idx = self.frag_fusion_table.currentRow()
        if idx < 0 and getattr(self, "fusion_results", None):
            idx = 0
        return idx

    def _fusion_fragment_color_keys(self, row):
        """Stable (compound_idx, fragment_id) order and hex colors for fusion attribution plots."""
        palette = self._fusion_nature_palette()
        key_order = []
        seen = set()
        for p in row.get("pairs", []):
            k = (int(p.get("compound_idx", 0)), int(p.get("fragment_id", 0)))
            if k not in seen:
                seen.add(k)
                key_order.append(k)
        if not key_order:
            for f in row.get("fragments", []):
                k = (int(f.get("compound_idx", 0)), int(f.get("fragment_id", 0)))
                if k not in seen:
                    seen.add(k)
                    key_order.append(k)
        key_color = {k: palette[i % len(palette)] for i, k in enumerate(key_order)}
        return key_order, key_color

    def _fusion_nature_palette(self):
        """Muted, publication-like palette inspired by Nature-style figures."""
        return [
            "#3E5C76",  # muted navy
            "#6C8E6F",  # sage green
            "#8F6D9A",  # dusty purple
            "#A87C5F",  # warm brown
            "#5E8D9E",  # teal blue-gray
            "#B34E5A",  # muted crimson
            "#9CA86A",  # olive
            "#7A7A7A",  # neutral gray
        ]

    def _update_fusion_mapping_plot(self):
        fig = self.frag_fusion_map_canvas_top.figure
        fig.clear()
        ax = fig.add_subplot(111)
        idx = self._current_fusion_row_index()
        if idx < 0 or idx >= len(getattr(self, "fusion_results", [])):
            ax.text(0.5, 0.5, "No fusion result.", ha="center", va="center", transform=ax.transAxes, color="#6c757d")
            ax.set_axis_off()
            self.frag_fusion_map_canvas_top.draw()
            return
        row = self.fusion_results[idx]
        exp = self.get_experimental_data() or []
        exp_ppm = [float(p.get("ppm", 0.0)) for p in exp]
        if exp_ppm:
            ax.vlines(exp_ppm, 0.25, 0.95, color="#ced4da", linewidth=1.0, alpha=0.8)
        pairs = row.get("pairs", [])
        _, key_color = self._fusion_fragment_color_keys(row)
        palette = self._fusion_nature_palette()
        frags_by_key = {}
        for f in row.get("fragments", []):
            kk = (int(f.get("compound_idx", 0)), int(f.get("fragment_id", 0)))
            frags_by_key[kk] = f

        sorted_pairs = sorted(
            pairs,
            key=lambda p: (round(float(p.get("pred_ppm", 0.0)), 3), int(p.get("fragment_id", 0))),
        )
        pred_ppm = []
        exp_match = []
        all_res = list(getattr(self, "masking_all_results", None) or [])
        if not all_res and getattr(self, "masking_result_data", None):
            all_res = [self.masking_result_data]

        for p in sorted_pairs:
            k = (int(p.get("compound_idx", 0)), int(p.get("fragment_id", 0)))
            col = key_color.get(k, palette[len(key_color) % len(palette)])
            x0 = float(p.get("pred_ppm", 0.0))
            x1 = float(p.get("exp_ppm", 0.0))
            pred_ppm.append(x0)
            exp_match.append(x1)
            ax.scatter([x0], [-0.85], s=40, color=col, zorder=9)
            ax.scatter([x1], [0.95], s=42, color=col, zorder=9, edgecolor="#2f2f2f", linewidth=0.25)
            ax.plot([x0, x1], [-0.25, 0.25], color=col, alpha=0.42, linewidth=1.1)

        show_struct = bool(
            getattr(self, "frag_fusion_show_fragment_structs_check", None)
            and self.frag_fusion_show_fragment_structs_check.isChecked()
        )
        fusion_frag_band_h = 0.88
        from collections import defaultdict

        by_frag = defaultdict(list)
        for p in pairs:
            kk = (int(p.get("compound_idx", 0)), int(p.get("fragment_id", 0)))
            by_frag[kk].append(p)

        min_y_for_thumb = None
        if show_struct and by_frag:
            import matplotlib.colors as mcolors
            import numpy as np

            from VirMolAnalyte.masking_aas_attribution import (
                draw_fusion_fragment_env_group_image_and_atom_pixels,
            )

            width_ppm = 28.0
            height_y = fusion_frag_band_h
            row_gap = 0.16
            img_w, img_h = 260, 200

            unique_keys = sorted(by_frag.keys(), key=lambda kk: (kk[0], kk[1]))
            layout_row = 0
            for k in unique_keys:
                plist = by_frag[k]
                col = key_color.get(k, palette[len(key_color) % len(palette)])
                frag = frags_by_key.get(k)
                ci = k[0]
                if not frag or ci < 0 or ci >= len(all_res):
                    continue
                smi = str(all_res[ci].get("smiles", "")).strip()
                if not smi:
                    continue
                xs = [float(x.get("pred_ppm", 0.0)) for x in plist]
                if not xs:
                    continue
                x_center = float(np.mean(xs))
                y_img = -1.52 - layout_row * (height_y + row_gap)
                drew_thumb = False
                pixel_map = {}
                iw, ih = img_w, img_h
                try:
                    rgb = mcolors.to_rgb(col)
                    pil_img, pixel_map, iw, ih = draw_fusion_fragment_env_group_image_and_atom_pixels(
                        smi,
                        frag.get("atom_indices", []),
                        rgb,
                        size=(img_w, img_h),
                    )
                    arr = np.asarray(pil_img.convert("RGBA"))
                    half_w = width_ppm / 2.0
                    el, er = x_center + half_w, x_center - half_w
                    eb, et = y_img - height_y / 2.0, y_img + height_y / 2.0
                    ax.imshow(
                        arr,
                        extent=[el, er, eb, et],
                        origin="upper",
                        zorder=5,
                        aspect="auto",
                        interpolation="bilinear",
                    )
                    drew_thumb = True
                except Exception:
                    pass
                if drew_thumb:
                    layout_row += 1
                    min_y_for_thumb = (
                        y_img if min_y_for_thumb is None else min(min_y_for_thumb, y_img)
                    )
                    for pr in plist:
                        x0 = float(pr.get("pred_ppm", 0.0))
                        aid = int(pr.get("atom_index", -1))
                        px, py = pixel_map.get(aid, (iw / 2.0, ih / 2.0))
                        px = max(0.0, min(float(iw) - 1e-6, float(px)))
                        py = max(0.0, min(float(ih) - 1e-6, float(py)))
                        x_atom = el + (px / float(iw)) * (er - el)
                        y_atom = et - (py / float(ih)) * (et - eb)
                        ax.plot(
                            [x0, x_atom],
                            [-0.85, y_atom],
                            color=col,
                            linestyle="--",
                            linewidth=0.85,
                            alpha=0.38,
                            zorder=7,
                        )

        all_ppm = exp_ppm + pred_ppm + exp_match
        if all_ppm:
            xmin, xmax = min(all_ppm) - 5.0, max(all_ppm) + 5.0
        else:
            xmin, xmax = 0.0, 220.0
        ax.set_xlim(xmax, xmin)
        y_bottom = -1.15
        if show_struct and min_y_for_thumb is not None:
            y_bottom = min(y_bottom, min_y_for_thumb - fusion_frag_band_h / 2.0 - 0.22)
        ax.set_ylim(y_bottom, 1.15)
        ax.axhline(0, color="#7f8a95", linewidth=1.0)
        ax.set_yticks([-0.85, 0.95])
        ax.set_yticklabels(["Fusion δ_pred", "Experimental peaks"], fontsize=11)
        ax.set_xlabel("Chemical Shift (ppm)", fontsize=12)
        ax.tick_params(axis="x", labelsize=11)
        ax.grid(axis="x", linestyle="--", alpha=0.2)
        ax.set_title(
            f"Fusion #{idx + 1} | k={row.get('combo_size')} | Na/Nb={row.get('na')}/{row.get('nb')} | "
            f"AAS={float(row.get('aas_best', 0.0)):.3f} | Final={float(row.get('score_final', 0.0)):.3f}",
            fontsize=12,
        )
        self.frag_fusion_map_canvas_top.draw()

    def _on_fusion_combo_changed(self, _idx):
        if getattr(self, "fusion_detail_dialog", None) and self.fusion_detail_dialog.isVisible():
            self.fusion_detail_dialog._fusion_index = int(self._current_fusion_row_index())
            self._refresh_fusion_detail_dialog()
        self._update_fusion_structure_preview()

    def _on_fusion_table_clicked(self, row, _col):
        if row < 0 or row >= len(getattr(self, "fusion_results", [])):
            return
        if hasattr(self, "frag_fusion_table"):
            self.frag_fusion_table.selectRow(row)
        self._update_fusion_structure_preview()

    def _update_fusion_structure_preview(self):
        idx = self._current_fusion_row_index()
        if idx < 0 or idx >= len(getattr(self, "fusion_results", [])):
            self.frag_fusion_preview_label.setText("Fusion fragment combination preview appears here.")
            self.frag_fusion_preview_label.setPixmap(QPixmap())
            return
        row = self.fusion_results[idx]
        try:
            # Reuse the same fragment extraction / rendering strategy as
            # Fusion combination detail popup ("Matched fragments only").
            img = self._build_fusion_matched_preview_image(
                row,
                show_idx=False,
                show_matched=True,
                fragments_only=True,
                show_base_colors=True,
            )
            if img is None:
                self.frag_fusion_preview_label.setText("No structure preview for this combination.")
                self.frag_fusion_preview_label.setPixmap(QPixmap())
                return
            pix = self.pil_image_to_pixmap(img)
            mh = int(self.frag_fusion_preview_label.maximumHeight()) - 4
            if mh > 40 and pix.height() > mh:
                pix = pix.scaledToHeight(mh, Qt.SmoothTransformation)
            self.frag_fusion_preview_label.setPixmap(pix)
            self.frag_fusion_preview_label.setText("")
        except Exception as e:
            self.frag_fusion_preview_label.setText(f"Preview error: {e}")

    def _patch_fusion_result_compound_indices(self, results, compound_idx: int):
        """Map fusion meta compound_idx to UI candidate row (needed for single-compound fusion pool)."""
        for r in results:
            for f in r.get("fragments", []):
                f["compound_idx"] = int(compound_idx)
            for p in r.get("pairs", []):
                p["compound_idx"] = int(compound_idx)

    def _build_fusion_pred_preview_image(self, row, show_idx=False, show_pred=False, show_base_colors=True):
        from VirMolAnalyte.masking_aas_attribution import draw_mol_attribution_png
        from PIL import Image

        frag_list = row.get("fragments", [])
        _, key_color = self._fusion_fragment_color_keys(row)
        tiles = []
        for f in frag_list:
            ci = int(f.get("compound_idx", 0))
            if ci < 0 or ci >= len(getattr(self, "masking_all_results", [])):
                continue
            p = self.masking_all_results[ci]
            fid = int(f.get("fragment_id", 0))
            kk = (ci, fid)
            hxc = key_color.get(kk) or "#3E5C76"
            tile = draw_mol_attribution_png(
                p["smiles"],
                p["carbon_atom_indices"],
                p["mean_scores"],
                p.get("vir_shifts", []),
                size=(480, 340),
                score_vmin=-5.0,
                score_vmax=5.0,
                show_rdkit_atom_index=bool(show_idx),
                show_pred_shift_label=bool(show_pred),
                selected_atom_indices=f.get("atom_indices", []),
                selected_atom_color=self._hex_color_to_rgb01(hxc),
                annotation_font_scale=0.96,
                use_score_coloring=bool(show_base_colors),
            ).convert("RGB")
            tiles.append(tile)
        if not tiles:
            return None
        w = sum(im.width for im in tiles)
        h = max(im.height for im in tiles)
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        x = 0
        for im in tiles:
            canvas.paste(im, (x, (h - im.height) // 2))
            x += im.width
        return canvas

    def _draw_fusion_fragment_only_tile(
        self,
        smiles,
        fragment_atom_indices,
        matched_map,
        *,
        show_idx=False,
        show_matched=True,
        frag_rgb=(0.20, 0.60, 0.25),
        use_fragment_coloring=True,
        annotation_font_scale=0.82,
        size=(480, 340),
    ):
        """Draw fragment with first-shell neighbors (non-fragment atoms kept as context)."""
        import io
        from PIL import Image
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        frag_old = sorted({int(x) for x in (fragment_atom_indices or [])})
        if not frag_old:
            raise ValueError("Empty fragment atom indices")

        env_old = set(frag_old)
        for old_idx in frag_old:
            atom = mol.GetAtomWithIdx(int(old_idx))
            for nb in atom.GetNeighbors():
                env_old.add(int(nb.GetIdx()))
        env_old = sorted(env_old)
        env_set = set(env_old)
        frag_set = set(frag_old)
        boundary_old = set()
        for old_idx in env_old:
            if old_idx in frag_set:
                continue
            atom = mol.GetAtomWithIdx(int(old_idx))
            if any(int(nb.GetIdx()) not in env_set for nb in atom.GetNeighbors()):
                boundary_old.add(int(old_idx))

        rw = Chem.RWMol()
        old_to_new = {}
        r_dummy_new_idxs = []
        # Keep all first-shell context atoms (including hetero atoms like O/N)
        # and only append R outside of this kept shell.
        for old_idx in env_old:
            old_to_new[old_idx] = rw.AddAtom(mol.GetAtomWithIdx(old_idx))
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in old_to_new and j in old_to_new:
                ni, nj = old_to_new[i], old_to_new[j]
                if rw.GetBondBetweenAtoms(ni, nj) is None:
                    rw.AddBond(ni, nj, bond.GetBondType())
        # For boundary atoms, add explicit R placeholders for truncated neighbors.
        for old_idx in boundary_old:
            atom = mol.GetAtomWithIdx(int(old_idx))
            n_new = old_to_new[old_idx]
            # Only add R when truncated neighbors contain non-H atoms.
            ext_non_h = []
            for nb in atom.GetNeighbors():
                nb_old = int(nb.GetIdx())
                if nb_old in env_set:
                    continue
                if int(nb.GetAtomicNum()) == 1:
                    continue
                ext_non_h.append(nb_old)
            if not ext_non_h:
                continue
            for nb in atom.GetNeighbors():
                nb_old = int(nb.GetIdx())
                if nb_old in env_set:
                    continue
                if int(nb.GetAtomicNum()) == 1:
                    continue
                r_new = rw.AddAtom(Chem.Atom(0))
                r_dummy_new_idxs.append(int(r_new))
                if rw.GetBondBetweenAtoms(int(n_new), int(r_new)) is None:
                    # Preserve visual bond type when possible.
                    b = mol.GetBondBetweenAtoms(int(old_idx), int(nb_old))
                    bt = b.GetBondType() if b is not None else Chem.BondType.SINGLE
                    rw.AddBond(int(n_new), int(r_new), bt)
        sub = rw.GetMol()
        Chem.SanitizeMol(sub)
        AllChem.Compute2DCoords(sub)

        hm = {}
        radii = {}
        bond_colors = {}
        fr, fg, fb = float(frag_rgb[0]), float(frag_rgb[1]), float(frag_rgb[2])
        # Harmonized label bubble color (pastel tint from fragment color)
        lbl_r = min(1.0, 0.84 + 0.16 * fr)
        lbl_g = min(1.0, 0.84 + 0.16 * fg)
        lbl_b = min(1.0, 0.84 + 0.16 * fb)
        if bool(use_fragment_coloring):
            for old_idx, new_idx in old_to_new.items():
                # Bond-first style: keep atom circles subtle.
                if int(old_idx) in frag_set and int(old_idx) in matched_map:
                    hm[int(new_idx)] = (lbl_r, lbl_g, lbl_b)
                    radii[int(new_idx)] = 0.18

        d2d = rdMolDraw2D.MolDraw2DCairo(int(size[0]), int(size[1]))
        opts = d2d.drawOptions()
        opts.clearBackground = True
        try:
            opts.useBWAtomPalette()
        except Exception:
            pass
        try:
            opts.setBackgroundColour((1.0, 1.0, 1.0))
        except Exception:
            pass
        try:
            # Use a muted light-blue tone for score/ppm annotations.
            opts.setAnnotationColour((0.24, 0.31, 0.40))
        except Exception:
            pass

        if bool(show_idx) or bool(show_matched) or bool(r_dummy_new_idxs):
            amap = rdMolDraw2D.IntStringMap()
            for old_idx, new_idx in old_to_new.items():
                parts = []
                if bool(show_idx):
                    parts.append(str(int(old_idx)))
                if bool(show_matched) and int(old_idx) in matched_map:
                    parts.append(f"{float(matched_map[int(old_idx)]):.1f}")
                if parts:
                    amap[int(new_idx)] = "\n".join(parts)
            for ridx in r_dummy_new_idxs:
                amap[int(ridx)] = "R"
            opts.atomLabels = amap
            opts.annotationFontScale = float(annotation_font_scale)

        new_to_old = {new_idx: old_idx for old_idx, new_idx in old_to_new.items()}
        if bool(use_fragment_coloring):
            for b in sub.GetBonds():
                a1 = int(b.GetBeginAtomIdx())
                a2 = int(b.GetEndAtomIdx())
                o1 = int(new_to_old.get(a1, -1))
                o2 = int(new_to_old.get(a2, -1))
                if o1 in frag_set and o2 in frag_set:
                    bid = int(b.GetIdx())
                    bond_colors[bid] = (fr, fg, fb)

        rdMolDraw2D.PrepareAndDrawMolecule(
            d2d,
            Chem.Mol(sub),
            highlightAtoms=sorted(hm.keys()),
            highlightAtomColors=hm,
            highlightAtomRadii=radii,
            highlightBonds=sorted(bond_colors.keys()),
            highlightBondColors=bond_colors,
        )
        d2d.FinishDrawing()
        return Image.open(io.BytesIO(d2d.GetDrawingText())).convert("RGB")

    def _hex_color_to_rgb01(self, color_hex, fallback=(0.30, 0.65, 0.30)):
        """Convert #RRGGBB to normalized RGB tuple."""
        try:
            s = str(color_hex).strip().lstrip("#")
            if len(s) != 6:
                return fallback
            return (
                int(s[0:2], 16) / 255.0,
                int(s[2:4], 16) / 255.0,
                int(s[4:6], 16) / 255.0,
            )
        except Exception:
            return fallback

    def _build_fusion_matched_preview_image(self, row, show_idx=False, show_matched=True, fragments_only=False, show_base_colors=True):
        from VirMolAnalyte.masking_aas_attribution import draw_mol_attribution_png
        from PIL import Image, ImageDraw

        pairs = row.get("pairs", [])
        by_comp_frag = {}
        for pr in pairs:
            try:
                k = (int(pr.get("compound_idx", 0)), int(pr.get("fragment_id", 0)))
                aid = int(pr.get("atom_index", -1))
                exp = float(pr.get("exp_ppm"))
                ae = float(pr.get("abs_err", 1e18))
            except Exception:
                continue
            if k not in by_comp_frag:
                by_comp_frag[k] = {}
            old = by_comp_frag[k].get(aid)
            if old is None or ae < old[0]:
                by_comp_frag[k][aid] = (ae, exp)

        frag_list = row.get("fragments", [])
        _, key_color = self._fusion_fragment_color_keys(row)
        tiles = []
        for f in frag_list:
            ci = int(f.get("compound_idx", 0))
            fid = int(f.get("fragment_id", 0))
            placeholder = None
            if ci < 0 or ci >= len(getattr(self, "masking_all_results", [])):
                placeholder = f"Missing candidate C{ci + 1}"
            else:
                p = self.masking_all_results[ci]
                matched_map = {a: v[1] for a, v in by_comp_frag.get((ci, fid), {}).items()}
                try:
                    if bool(fragments_only):
                        kk = (ci, fid)
                        hxc = key_color.get(kk) or "#3E5C76"
                        tile = self._draw_fusion_fragment_only_tile(
                            p["smiles"],
                            f.get("atom_indices", []),
                            matched_map,
                            show_idx=bool(show_idx),
                            show_matched=bool(show_matched),
                            frag_rgb=self._hex_color_to_rgb01(hxc),
                            annotation_font_scale=0.82,
                            size=(480, 340),
                        )
                    else:
                        tile = draw_mol_attribution_png(
                            p["smiles"],
                            p["carbon_atom_indices"],
                            p["mean_scores"],
                            p.get("vir_shifts", []),
                            size=(480, 340),
                            score_vmin=-5.0,
                            score_vmax=5.0,
                            show_rdkit_atom_index=bool(show_idx),
                            show_pred_shift_label=False,
                            show_matched_shift_label=bool(show_matched),
                            matched_ppm_by_atom=matched_map,
                            annotation_font_scale=0.95,
                            use_score_coloring=bool(show_base_colors),
                        ).convert("RGB")
                    tiles.append(tile)
                    continue
                except Exception:
                    placeholder = f"Render failed C{ci + 1}-F{fid}"

            # Keep slot visible to avoid "missing structure in the middle" confusion.
            ph = Image.new("RGB", (480, 340), (251, 252, 254))
            d = ImageDraw.Draw(ph)
            d.rectangle([8, 8, 472, 332], outline=(213, 222, 232), width=1)
            d.text((20, 20), str(placeholder or "Missing fragment"), fill=(93, 106, 120))
            tile = ph
            tiles.append(tile)
        if not tiles:
            return None
        w = sum(im.width for im in tiles)
        h = max(im.height for im in tiles)
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        x = 0
        for im in tiles:
            canvas.paste(im, (x, (h - im.height) // 2))
            x += im.width
        return canvas

    def _update_fusion_ai_top5_btn_state(self):
        if not hasattr(self, "frag_fusion_ai_top5_btn"):
            return
        ok = bool(getattr(self, "fusion_results", None)) and bool(self.get_experimental_data())
        self.frag_fusion_ai_top5_btn.setEnabled(ok)
        if hasattr(self, "frag_fusion_ai_review_btn"):
            self.frag_fusion_ai_review_btn.setEnabled(ok)

    def _populate_fusion_results_ui(self, status_message: str):
        self.frag_fusion_table.setRowCount(0)
        for i, r in enumerate(self.fusion_results):
            self.frag_fusion_table.insertRow(i)
            frag_txt = ", ".join(
                f"C{int(f.get('compound_idx', 0)) + 1}-F{int(f.get('fragment_id', 0))}"
                for f in r.get("fragments", [])
            )
            vals = [
                str(i + 1),
                str(r.get("combo_size", "")),
                f"{int(r.get('na', 0))}/{int(r.get('nb', 0))}",
                f"{100.0 * float(r.get('coverage', 0.0)):.1f}%",
                f"{float(r.get('aas_best', 0.0)):.4f}",
                f"{float(r.get('score_final', 0.0)):.4f}",
                f"{float(r.get('sse', 0.0)):.4f}",
                frag_txt,
            ]
            for c, t in enumerate(vals):
                self.frag_fusion_table.setItem(i, c, QTableWidgetItem(t))
        self.frag_fusion_table.resizeColumnsToContents()
        if self.fusion_results:
            self.frag_fusion_table.selectRow(0)
            self.frag_mask_view_combo.setCurrentIndex(1)
        self._update_fusion_structure_preview()
        self._update_fusion_ai_top5_btn_state()
        self.statusBar().showMessage(status_message)

    def _refresh_fusion_detail_dialog(self):
        dlg = getattr(self, "fusion_detail_dialog", None)
        if dlg is None:
            return
        idx = int(getattr(dlg, "_fusion_index", -1))
        total = len(getattr(self, "fusion_results", []))
        if idx < 0 or idx >= total:
            return
        row = self.fusion_results[idx]
        if hasattr(dlg, "_index_label"):
            dlg._index_label.setText(f"{idx + 1}/{total}")
        if hasattr(dlg, "_prev_btn"):
            dlg._prev_btn.setEnabled(idx > 0)
        if hasattr(dlg, "_next_btn"):
            dlg._next_btn.setEnabled(idx < total - 1)
        dlg._title_label.setText(
            f"Fusion #{idx + 1} | k={row.get('combo_size')} | Na/Nb={row.get('na')}/{row.get('nb')} | "
            f"AAS={float(row.get('aas_best', 0.0)):.3f} | Final={float(row.get('score_final', 0.0)):.3f}"
        )
        try:
            img_pred = self._build_fusion_pred_preview_image(
                row,
                show_idx=dlg._show_atom_idx.isChecked(),
                show_pred=dlg._show_pred.isChecked(),
                show_base_colors=dlg._show_base_colors.isChecked(),
            )
            if img_pred is not None:
                self._set_label_pixmap_fit(dlg._pred_label, self.pil_image_to_pixmap(img_pred))
                dlg._pred_label.setText("")
            else:
                dlg._pred_label.setText("No predicted structure preview.")
        except Exception as e:
            dlg._pred_label.setText(f"Predicted preview error: {e}")
        try:
            img_match = self._build_fusion_matched_preview_image(
                row,
                show_idx=dlg._show_atom_idx.isChecked(),
                show_matched=dlg._show_matched.isChecked(),
                fragments_only=dlg._matched_frag_only.isChecked(),
                show_base_colors=dlg._show_base_colors.isChecked(),
            )
            if img_match is not None:
                self._set_label_pixmap_fit(dlg._match_label, self.pil_image_to_pixmap(img_match))
                dlg._match_label.setText("")
            else:
                dlg._match_label.setText("No matched structure preview.")
        except Exception as e:
            dlg._match_label.setText(f"Matched preview error: {e}")

        fig = dlg._map_canvas.figure
        fig.clear()
        ax_map = fig.add_subplot(111)
        self._render_fusion_mapping_axes(
            ax_map,
            row,
            idx,
            show_struct=dlg._show_struct_thumb.isChecked(),
        )
        fig.tight_layout()
        dlg._map_canvas.draw()

    def _render_fusion_mapping_axes(self, ax, row, idx, show_struct=True):
        ax.clear()
        exp = self.get_experimental_data() or []
        exp_ppm = [float(p.get("ppm", 0.0)) for p in exp]
        if exp_ppm:
            ax.vlines(exp_ppm, 0.25, 0.95, color="#ced4da", linewidth=1.0, alpha=0.8)
        pairs = row.get("pairs", [])
        _, key_color = self._fusion_fragment_color_keys(row)
        palette = self._fusion_nature_palette()
        sorted_pairs = sorted(
            pairs,
            key=lambda p: (round(float(p.get("pred_ppm", 0.0)), 3), int(p.get("fragment_id", 0))),
        )
        pred_ppm, exp_match = [], []
        frags_by_key = {}
        for f in row.get("fragments", []):
            kk = (int(f.get("compound_idx", 0)), int(f.get("fragment_id", 0)))
            frags_by_key[kk] = f
        all_res = list(getattr(self, "masking_all_results", None) or [])
        if not all_res and getattr(self, "masking_result_data", None):
            all_res = [self.masking_result_data]
        for p in sorted_pairs:
            k = (int(p.get("compound_idx", 0)), int(p.get("fragment_id", 0)))
            col = key_color.get(k, palette[len(key_color) % len(palette)])
            x0 = float(p.get("pred_ppm", 0.0))
            x1 = float(p.get("exp_ppm", 0.0))
            pred_ppm.append(x0)
            exp_match.append(x1)
            ax.scatter([x0], [-0.85], s=32, color=col, zorder=9)
            ax.scatter([x1], [0.95], s=34, color=col, zorder=9, edgecolor="black", linewidth=0.2)
            ax.plot([x0, x1], [-0.25, 0.25], color=col, alpha=0.55, linewidth=1.0)

        fusion_frag_band_h = 0.88
        from collections import defaultdict
        by_frag = defaultdict(list)
        for p in pairs:
            kk = (int(p.get("compound_idx", 0)), int(p.get("fragment_id", 0)))
            by_frag[kk].append(p)
        min_y_for_thumb = None
        if bool(show_struct) and by_frag:
            import matplotlib.colors as mcolors
            import numpy as np

            from VirMolAnalyte.masking_aas_attribution import (
                draw_fusion_fragment_env_group_image_and_atom_pixels,
            )

            width_ppm = 28.0
            height_y = fusion_frag_band_h
            row_gap = 0.16
            img_w, img_h = 320, 246
            unique_keys = sorted(by_frag.keys(), key=lambda kk: (kk[0], kk[1]))
            layout_row = 0
            for k in unique_keys:
                plist = by_frag[k]
                col = key_color.get(k, palette[len(key_color) % len(palette)])
                frag = frags_by_key.get(k)
                ci = k[0]
                if not frag or ci < 0 or ci >= len(all_res):
                    continue
                smi = str(all_res[ci].get("smiles", "")).strip()
                if not smi:
                    continue
                xs = [float(x.get("pred_ppm", 0.0)) for x in plist]
                if not xs:
                    continue
                x_center = float(np.mean(xs))
                y_img = -1.52 - layout_row * (height_y + row_gap)
                drew_thumb = False
                pixel_map = {}
                iw, ih = img_w, img_h
                try:
                    rgb = mcolors.to_rgb(col)
                    pil_img, pixel_map, iw, ih = draw_fusion_fragment_env_group_image_and_atom_pixels(
                        smi,
                        frag.get("atom_indices", []),
                        rgb,
                        size=(img_w, img_h),
                        font_scale=1.28,
                    )
                    arr = np.asarray(pil_img.convert("RGBA"))
                    half_w = width_ppm / 2.0
                    el, er = x_center + half_w, x_center - half_w
                    eb, et = y_img - height_y / 2.0, y_img + height_y / 2.0
                    ax.imshow(
                        arr,
                        extent=[el, er, eb, et],
                        origin="upper",
                        zorder=5,
                        aspect="auto",
                        interpolation="bilinear",
                    )
                    drew_thumb = True
                except Exception:
                    pass
                if drew_thumb:
                    layout_row += 1
                    min_y_for_thumb = y_img if min_y_for_thumb is None else min(min_y_for_thumb, y_img)
                    for pr in plist:
                        x0 = float(pr.get("pred_ppm", 0.0))
                        aid = int(pr.get("atom_index", -1))
                        px, py = pixel_map.get(aid, (iw / 2.0, ih / 2.0))
                        px = max(0.0, min(float(iw) - 1e-6, float(px)))
                        py = max(0.0, min(float(ih) - 1e-6, float(py)))
                        x_atom = el + (px / float(iw)) * (er - el)
                        y_atom = et - (py / float(ih)) * (et - eb)
                        ax.plot(
                            [x0, x_atom],
                            [-0.85, y_atom],
                            color=col,
                            linestyle="--",
                            linewidth=0.85,
                            alpha=0.38,
                            zorder=7,
                        )

        all_ppm = exp_ppm + pred_ppm + exp_match
        if all_ppm:
            xmin, xmax = min(all_ppm) - 5.0, max(all_ppm) + 5.0
        else:
            xmin, xmax = 0.0, 220.0
        ax.set_xlim(xmax, xmin)
        y_bottom = -1.15
        if bool(show_struct) and min_y_for_thumb is not None:
            y_bottom = min(y_bottom, min_y_for_thumb - fusion_frag_band_h / 2.0 - 0.22)
        ax.set_ylim(y_bottom, 1.15)
        ax.axhline(0, color="#868e96", linewidth=1.0)
        ax.set_yticks([-0.85, 0.95])
        ax.set_yticklabels(["Fusion δ_pred", "Experimental peaks"], fontsize=14)
        ax.set_xlabel("Chemical Shift (ppm)", fontsize=14)
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(axis="x", linestyle="--", alpha=0.2)
        ax.set_title(
            f"Fusion #{idx + 1} | k={row.get('combo_size')} | Na/Nb={row.get('na')}/{row.get('nb')} | "
            f"AAS={float(row.get('aas_best', 0.0)):.3f} | Final={float(row.get('score_final', 0.0)):.3f}",
            fontsize=15,
        )

    def _export_fusion_detail_hd(self):
        dlg = getattr(self, "fusion_detail_dialog", None)
        if dlg is None:
            return
        idx = int(getattr(dlg, "_fusion_index", -1))
        if idx < 0 or idx >= len(getattr(self, "fusion_results", [])):
            return
        row = self.fusion_results[idx]
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export fusion detail image",
            f"fusion_{idx + 1}_detail_hd.png",
            "PNG Files (*.png);;TIFF Files (*.tif *.tiff);;JPEG Files (*.jpg *.jpeg);;SVG Files (*.svg);;PDF Files (*.pdf)",
        )
        if not file_path:
            return
        try:
            pred_img = self._build_fusion_pred_preview_image(
                row,
                show_idx=dlg._show_atom_idx.isChecked(),
                show_pred=dlg._show_pred.isChecked(),
                show_base_colors=dlg._show_base_colors.isChecked(),
            )
            match_img = self._build_fusion_matched_preview_image(
                row,
                show_idx=dlg._show_atom_idx.isChecked(),
                show_matched=dlg._show_matched.isChecked(),
                fragments_only=dlg._matched_frag_only.isChecked(),
                show_base_colors=dlg._show_base_colors.isChecked(),
            )
            fig = Figure(figsize=(15.5, 11.0), dpi=150)
            gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.2], hspace=0.08, wspace=0.04)
            ax_pred = fig.add_subplot(gs[0, 0])
            ax_match = fig.add_subplot(gs[0, 1])
            ax_map = fig.add_subplot(gs[1, :])
            if pred_img is not None:
                ax_pred.imshow(np.asarray(pred_img))
                ax_pred.set_title("Predicted δ on structure", fontsize=15, fontweight="bold")
            else:
                ax_pred.text(0.5, 0.5, "No predicted structure preview.", ha="center", va="center", transform=ax_pred.transAxes)
            ax_pred.axis("off")
            if match_img is not None:
                ax_match.imshow(np.asarray(match_img))
                ax_match.set_title("Matched experimental ppm on structure", fontsize=15, fontweight="bold")
            else:
                ax_match.text(0.5, 0.5, "No matched structure preview.", ha="center", va="center", transform=ax_match.transAxes)
            ax_match.axis("off")
            self._render_fusion_mapping_axes(
                ax_map,
                row,
                idx,
                show_struct=dlg._show_struct_thumb.isChecked(),
            )
            fig.suptitle(dlg._title_label.text(), fontsize=17, fontweight="bold", y=0.985)
            lower = file_path.lower()
            if lower.endswith(".svg"):
                fig.savefig(file_path, format="svg", bbox_inches="tight")
            elif lower.endswith(".pdf"):
                fig.savefig(file_path, format="pdf", bbox_inches="tight")
            else:
                fig.savefig(file_path, dpi=600, bbox_inches="tight")
            extra_detail = {
                "fusion_index": idx,
                "fusion_row": self._json_safe_export(row),
                "export_dialog": {
                    "show_atom_idx": dlg._show_atom_idx.isChecked(),
                    "show_pred": dlg._show_pred.isChecked(),
                    "show_matched": dlg._show_matched.isChecked(),
                    "matched_fragments_only": dlg._matched_frag_only.isChecked(),
                    "show_structure_thumbnail_in_mapping": dlg._show_struct_thumb.isChecked(),
                    "carbon_base_colors": dlg._show_base_colors.isChecked(),
                },
            }
            param_path = self._write_analysis_params_sidecar(
                file_path, "fusion_detail_hd", extra=extra_detail
            )
            self.statusBar().showMessage(f"Fusion detail exported: {file_path}")
            msg = f"Fusion detail exported:\n{file_path}"
            if param_path:
                msg += f"\n\nAnalysis parameters:\n{param_path}"
            else:
                msg += "\n\n(Could not save analysis parameters JSON; see console.)"
            QMessageBox.information(self, "Export", msg)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def _on_fusion_table_double_clicked(self, row, _col):
        if row < 0 or row >= len(getattr(self, "fusion_results", [])):
            return
        if hasattr(self, "frag_fusion_table"):
            self.frag_fusion_table.selectRow(row)
        if getattr(self, "fusion_detail_dialog", None) is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("Fusion combination detail")
            dlg.resize(1240, 820)
            lay = QVBoxLayout(dlg)
            lay.setContentsMargins(14, 12, 14, 12)
            lay.setSpacing(10)
            dlg.setStyleSheet(
                """
                QDialog { background: #f6f7f8; }
                QLabel { color: #334155; }
                QCheckBox { spacing: 6px; font-size: 13px; color: #34495e; }
                QPushButton {
                    background: #3f6270; color: #f5f7fa; border: 1px solid #36535f;
                    border-radius: 8px; padding: 8px 14px; font-weight: 600;
                }
                QPushButton:hover { background: #486f7e; }
                QPushButton:disabled { background: #c6ced6; color: #eef2f6; border-color: #c0c8d0; }
                QFrame#FusionCard {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                stop:0 #ffffff, stop:1 #f7fafc);
                    border: 1px solid #d6dee6;
                    border-radius: 12px;
                }
                QLabel#FusionCardTitle {
                    font-size: 15px; font-weight: 700; color: #3d5a66;
                    padding-top: 3px; padding-left: 2px;
                }
                """
            )
            dlg._title_label = QLabel("Fusion detail")
            dlg._title_label.setStyleSheet("font-size: 17px; font-weight: bold; color: #3d5a66;")
            lay.addWidget(dlg._title_label)
            nav_row = QHBoxLayout()
            nav_row.setSpacing(8)
            dlg._prev_btn = QPushButton("Previous")
            dlg._next_btn = QPushButton("Next")
            dlg._prev_btn.setFixedWidth(96)
            dlg._next_btn.setFixedWidth(96)
            dlg._index_label = QLabel("")
            dlg._index_label.setStyleSheet("color: #6c757d; font-size: 14px;")
            nav_row.addWidget(dlg._prev_btn)
            nav_row.addWidget(dlg._next_btn)
            nav_row.addWidget(dlg._index_label)
            nav_row.addStretch()
            dlg._close_btn = QPushButton("Close")
            dlg._close_btn.setFixedWidth(96)
            nav_row.addWidget(dlg._close_btn)
            lay.addLayout(nav_row)
            opts = QHBoxLayout()
            opts.setSpacing(10)
            dlg._show_atom_idx = QCheckBox("Atom index")
            dlg._show_pred = QCheckBox("Pred δ")
            dlg._show_matched = QCheckBox("Matched ppm")
            dlg._matched_frag_only = QCheckBox("Matched fragments only")
            dlg._show_base_colors = QCheckBox("Carbon base colors")
            dlg._show_struct_thumb = QCheckBox("Fragment thumbnails")
            dlg._show_atom_idx.setChecked(False)
            dlg._show_pred.setChecked(False)
            dlg._show_matched.setChecked(True)
            dlg._matched_frag_only.setChecked(False)
            dlg._show_base_colors.setChecked(True)
            dlg._show_struct_thumb.setChecked(True)
            dlg._export_btn = QPushButton("Export")
            opts.addWidget(dlg._show_atom_idx)
            opts.addWidget(dlg._show_pred)
            opts.addWidget(dlg._show_matched)
            opts.addWidget(dlg._matched_frag_only)
            opts.addWidget(dlg._show_base_colors)
            opts.addWidget(dlg._show_struct_thumb)
            opts.addStretch()
            dlg._copy_smiles_btn = QPushButton("Copy SMILES")
            opts.addWidget(dlg._export_btn)
            opts.addWidget(dlg._copy_smiles_btn)
            lay.addLayout(opts)
            dlg._hint_label = QLabel(
                "Tip: Left/Right arrows switch fusion result. Esc closes this dialog."
            )
            dlg._hint_label.setStyleSheet("color: #667788; font-size: 12px;")
            lay.addWidget(dlg._hint_label)
            def _apply_card_shadow(w, blur=20, yoff=3):
                eff = QGraphicsDropShadowEffect(w)
                eff.setBlurRadius(blur)
                eff.setOffset(0, yoff)
                eff.setColor(QColor(24, 39, 56, 45))
                w.setGraphicsEffect(eff)
            row2 = QVBoxLayout()
            row2.setSpacing(8)
            pw = QFrame(); pw.setObjectName("FusionCard")
            pl = QVBoxLayout(pw); pl.setContentsMargins(12, 10, 12, 12); pl.setSpacing(5)
            pt = QLabel("Predicted δ on structure"); pt.setObjectName("FusionCardTitle")
            pl.addWidget(pt)
            dlg._pred_label = QLabel("Predicted structure preview")
            dlg._pred_label.setMinimumHeight(250)
            dlg._pred_label.setAlignment(Qt.AlignCenter)
            dlg._pred_label.setStyleSheet(
                "border: 1px solid #d9e1e8; border-radius: 9px; background: #ffffff;"
            )
            pl.addWidget(dlg._pred_label)
            row2.addWidget(pw)
            mw = QFrame(); mw.setObjectName("FusionCard")
            ml = QVBoxLayout(mw); ml.setContentsMargins(12, 10, 12, 12); ml.setSpacing(5)
            mt = QLabel("Matched experimental ppm on structure"); mt.setObjectName("FusionCardTitle")
            ml.addWidget(mt)
            dlg._match_label = QLabel("Matched structure preview")
            dlg._match_label.setMinimumHeight(330)
            dlg._match_label.setAlignment(Qt.AlignCenter)
            dlg._match_label.setStyleSheet(
                "border: 1px solid #d9e1e8; border-radius: 9px; background: #ffffff;"
            )
            ml.addWidget(dlg._match_label)
            row2.addWidget(mw)
            lay.addLayout(row2)
            map_card = QFrame(); map_card.setObjectName("FusionCard")
            map_l = QVBoxLayout(map_card); map_l.setContentsMargins(12, 10, 12, 12); map_l.setSpacing(5)
            map_t = QLabel("Fusion mapping")
            map_t.setObjectName("FusionCardTitle")
            map_l.addWidget(map_t)
            dlg._map_canvas = FigureCanvas(Figure(figsize=(11.0, 4.2)))
            dlg._map_canvas.setMinimumHeight(300)
            dlg._map_canvas.setStyleSheet(
                "border: 1px solid #d9e1e8; border-radius: 9px; background: #ffffff;"
            )
            map_l.addWidget(dlg._map_canvas)
            lay.addWidget(map_card)
            _apply_card_shadow(pw)
            _apply_card_shadow(mw)
            _apply_card_shadow(map_card, blur=22, yoff=4)
            dlg._show_atom_idx.toggled.connect(self._refresh_fusion_detail_dialog)
            dlg._show_pred.toggled.connect(self._refresh_fusion_detail_dialog)
            dlg._show_matched.toggled.connect(self._refresh_fusion_detail_dialog)
            dlg._matched_frag_only.toggled.connect(self._refresh_fusion_detail_dialog)
            dlg._show_base_colors.toggled.connect(self._refresh_fusion_detail_dialog)
            dlg._show_struct_thumb.toggled.connect(self._refresh_fusion_detail_dialog)
            dlg._export_btn.clicked.connect(self._export_fusion_detail_hd)
            dlg._copy_smiles_btn.clicked.connect(self._copy_fusion_smiles)
            dlg._prev_btn.clicked.connect(lambda: self._navigate_fusion_detail(-1))
            dlg._next_btn.clicked.connect(lambda: self._navigate_fusion_detail(1))
            dlg._close_btn.clicked.connect(dlg.close)
            def _fusion_resize_event(event):
                QDialog.resizeEvent(dlg, event)
                if dlg.isVisible():
                    self._refresh_fusion_detail_dialog()
            dlg.resizeEvent = _fusion_resize_event
            def _fusion_key_press(event):
                if event.key() == Qt.Key_Left:
                    self._navigate_fusion_detail(-1)
                    event.accept()
                    return
                if event.key() == Qt.Key_Right:
                    self._navigate_fusion_detail(1)
                    event.accept()
                    return
                if event.key() == Qt.Key_Escape:
                    dlg.close()
                    event.accept()
                    return
                QDialog.keyPressEvent(dlg, event)
            dlg.keyPressEvent = _fusion_key_press
            self.fusion_detail_dialog = dlg
        self.fusion_detail_dialog._fusion_index = int(row)
        self._refresh_fusion_detail_dialog()
        self.fusion_detail_dialog.show()
        self.fusion_detail_dialog.raise_()
        self.fusion_detail_dialog.activateWindow()

    def _navigate_fusion_detail(self, step):
        dlg = getattr(self, "fusion_detail_dialog", None)
        rows = list(getattr(self, "fusion_results", []))
        if dlg is None or not rows:
            return
        idx = int(getattr(dlg, "_fusion_index", -1))
        if idx < 0:
            idx = 0
        idx = max(0, min(len(rows) - 1, idx + int(step)))
        dlg._fusion_index = idx
        if hasattr(self, "frag_fusion_table"):
            self.frag_fusion_table.selectRow(idx)
        self._refresh_fusion_detail_dialog()

    def _copy_fusion_smiles(self):
        """Copy all unique compound SMILES involved in current fusion row."""
        dlg = getattr(self, "fusion_detail_dialog", None)
        rows = list(getattr(self, "fusion_results", []))
        if dlg is None or not rows:
            QMessageBox.warning(self, "Copy SMILES", "No fusion result available.")
            return
        idx = int(getattr(dlg, "_fusion_index", -1))
        if idx < 0 or idx >= len(rows):
            QMessageBox.warning(self, "Copy SMILES", "No fusion result selected.")
            return
        row = rows[idx]
        fragments = list(row.get("fragments", []))
        if not fragments:
            QMessageBox.warning(self, "Copy SMILES", "No fragment data in current fusion result.")
            return

        smiles_list = []
        seen = set()
        all_res = list(getattr(self, "masking_all_results", []))
        for f in fragments:
            try:
                ci = int(f.get("compound_idx", -1))
            except Exception:
                continue
            if ci < 0 or ci >= len(all_res):
                continue
            smi = str(all_res[ci].get("smiles", "")).strip()
            if smi and smi not in seen:
                seen.add(smi)
                smiles_list.append(smi)

        if not smiles_list:
            QMessageBox.warning(self, "Copy SMILES", "Failed to resolve SMILES from this fusion result.")
            return

        text = "\n".join(smiles_list)
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage(f"Copied {len(smiles_list)} SMILES to clipboard")
        QMessageBox.information(
            self,
            "Copy SMILES",
            f"Copied {len(smiles_list)} SMILES to clipboard.\n(One line per compound)",
        )

    def _refresh_intra_fusion_run_enabled(self):
        if not hasattr(self, "frag_intra_fusion_run_btn"):
            return
        frags = getattr(self, "masking_fragments", None)
        self.frag_intra_fusion_run_btn.setEnabled(bool(frags and len(frags) > 0))

    def run_intra_fragment_fusion(self):
        frags = list(getattr(self, "masking_fragments", []))
        if not frags:
            QMessageBox.warning(
                self,
                "Warning",
                "No fragments for the current candidate. Extract positive fragments first.",
            )
            return
        exp = self.get_experimental_data()
        if not exp:
            QMessageBox.warning(
                self, "Warning", "No experimental peaks. Use preprocessing (NMR-1D.csv) or manual peak input."
            )
            return
        cur = self.frag_mask_result_combo.currentIndex()
        if cur < 0:
            cur = 0

        btn_text = self.frag_intra_fusion_run_btn.text()
        self.frag_intra_fusion_run_btn.setEnabled(False)
        self.frag_intra_fusion_run_btn.setText("Running intra-molecular fusion…")
        QApplication.processEvents()
        try:
            from VirMolAnalyte.masking_aas_attribution import run_fragment_fusion

            self.fusion_results = run_fragment_fusion(
                [frags],
                exp,
                top_fragment_pool=int(self.frag_intra_fusion_pool_topn.value()),
                max_merge_fragments=int(self.frag_intra_fusion_max_merge.value()),
                greedy_shuffle_repeats=int(self.frag_intra_fusion_shuffle_repeats.value()),
                coverage_weight=float(self.frag_intra_fusion_cov_weight.value()),
                top_k_results=10,
                use_dept_constraint=True,
                random_seed=None,
            )
            self._patch_fusion_result_compound_indices(self.fusion_results, cur)
            if hasattr(self, "frag_fusion_title"):
                self.frag_fusion_title.setText(
                    f"Intra-molecular fusion — candidate #{cur + 1} only"
                )
            self._populate_fusion_results_ui(
                f"Intra-molecular fusion done: {len(self.fusion_results)} top combinations"
            )
        except Exception as e:
            QMessageBox.critical(self, "Intra-molecular fusion failed", str(e))
        finally:
            self.frag_intra_fusion_run_btn.setText(btn_text)
            self._refresh_intra_fusion_run_enabled()

    def run_fragment_fusion_analysis(self):
        if not getattr(self, "masking_fragments_all", None):
            QMessageBox.warning(self, "Warning", "Run masking and extract fragments first.")
            return
        exp = self.get_experimental_data()
        if not exp:
            QMessageBox.warning(
                self, "Warning", "No experimental peaks. Use preprocessing (NMR-1D.csv) or manual peak input."
            )
            return
        btn_text = self.frag_fusion_run_btn.text()
        self.frag_fusion_run_btn.setEnabled(False)
        self.frag_fusion_run_btn.setText("Analyzing fusion combinations...")
        QApplication.processEvents()
        try:
            from VirMolAnalyte.masking_aas_attribution import run_fragment_fusion

            self.fusion_results = run_fragment_fusion(
                self.masking_fragments_all,
                exp,
                top_fragment_pool=int(self.frag_fusion_pool_topn.value()),
                max_merge_fragments=int(self.frag_fusion_max_merge.value()),
                greedy_shuffle_repeats=int(self.frag_fusion_shuffle_repeats.value()),
                coverage_weight=float(self.frag_fusion_cov_weight.value()),
                top_k_results=10,
                use_dept_constraint=True,
                random_seed=None,
            )
            if hasattr(self, "frag_fusion_title"):
                self.frag_fusion_title.setText("Cross-compound fragment fusion")
            self._populate_fusion_results_ui(
                f"Fusion analysis done: {len(self.fusion_results)} top combinations"
            )
        except Exception as e:
            QMessageBox.critical(self, "Fusion analysis failed", str(e))
        finally:
            self.frag_fusion_run_btn.setText(btn_text)
            self.frag_fusion_run_btn.setEnabled(bool(getattr(self, "masking_fragments_all", None)))

    def extract_masking_fragments(self):
        if not getattr(self, "masking_result_data", None):
            QMessageBox.warning(self, "Warning", "Run masking attribution first.")
            return
        exp = self.get_experimental_data()
        if not exp:
            QMessageBox.warning(
                self,
                "Warning",
                "No experimental peaks. Use preprocessing (NMR-1D.csv) or manual peak input.",
            )
            return
        try:
            from VirMolAnalyte.masking_aas_attribution import extract_positive_fragments

            seed_txt = self.frag_mask_greedy_shuffle_seed.text().strip()
            shuffle_seed = int(seed_txt) if seed_txt else None

            targets = (
                list(getattr(self, "masking_all_results", []))
                if getattr(self, "masking_all_results", None)
                else [self.masking_result_data]
            )
            all_frags = []
            for p in targets:
                score_threshold, threshold_note = self._resolve_fragment_score_threshold(
                    p.get("mean_scores", [])
                )
                fragments = extract_positive_fragments(
                    p["smiles"],
                    p["carbon_atom_indices"],
                    p["mean_scores"],
                    p.get("vir_shifts", []),
                    p.get("lib_types", []),
                    exp,
                    score_threshold=score_threshold,
                    min_carbons=int(self.frag_mask_frag_min_c.value()),
                    use_dept_constraint=self.frag_mask_extract_dept_check.isChecked(),
                    greedy_unique_matching=self.frag_mask_extract_greedy_check.isChecked(),
                    bridge_max_low_carbons=(
                        int(self.frag_mask_bridge_max_low.value())
                        if self.frag_mask_bridge_check.isChecked()
                        else 0
                    ),
                    allow_hetero_bridge_neighbors=(
                        self.frag_mask_allow_hetero_bridge_check.isChecked()
                        if hasattr(self, "frag_mask_allow_hetero_bridge_check")
                        else True
                    ),
                    max_fragments=int(self.frag_mask_max_frags.value()),
                    greedy_shuffle_repeats=int(self.frag_mask_greedy_shuffle_repeats.value()),
                    greedy_shuffle_seed=shuffle_seed,
                )
                all_frags.append(fragments)
            if hasattr(self, "frag_mask_thresh_info"):
                self.frag_mask_thresh_info.setText(threshold_note)

            self.masking_fragments_all = all_frags
            self._render_global_fragments_table()
            cur = self.frag_mask_result_combo.currentIndex()
            if cur < 0:
                cur = 0
            if cur >= len(all_frags):
                cur = max(0, len(all_frags) - 1)
            self.masking_fragments = all_frags[cur] if all_frags else []
            self._render_current_fragments_table()
            if hasattr(self, "frag_fusion_run_btn"):
                self.frag_fusion_run_btn.setEnabled(len(all_frags) > 0)
            self._refresh_intra_fusion_run_enabled()
            self.statusBar().showMessage(
                f"Extracted fragments for {len(all_frags)} compounds (current: {len(self.masking_fragments)})"
            )
        except Exception as e:
            QMessageBox.critical(self, "Extract fragments failed", str(e))

    def _display_masking_payload(self, payload):
        self.masking_result_data = payload
        if hasattr(self, "frag_mask_infer_btn"):
            self.frag_mask_infer_btn.setEnabled(bool(payload))
        if hasattr(self, "frag_mask_assign_btn"):
            self.frag_mask_assign_btn.setEnabled(bool(payload))
        if hasattr(self, "frag_mask_review_btn"):
            self.frag_mask_review_btn.setEnabled(len(getattr(self, "masking_fragments", [])) > 0)

        try:
            self._refresh_masking_structure_image()
        except Exception:
            pass

        # Keep manual threshold value user-controlled; default remains 0.0.

        self.frag_mask_scores_table.setRowCount(0)
        shifts = payload.get("vir_shifts", [])
        ctypes = payload.get("lib_types", [])
        for i, (aid, sc) in enumerate(
            zip(payload["carbon_atom_indices"], payload["mean_scores"])
        ):
            self.frag_mask_scores_table.insertRow(i)
            dppm = shifts[i] if i < len(shifts) else ""
            dtxt = f"{dppm:.3f}" if isinstance(dppm, (int, float)) else str(dppm)
            ct = ctypes[i] if i < len(ctypes) else ""
            try:
                sc_f = float(sc)
                sc_txt = f"{sc_f:.6f}" if math.isfinite(sc_f) else "—"
            except Exception:
                sc_txt = str(sc)
            row_vals = [str(i), str(aid), str(ct), sc_txt, dtxt]
            for col, text in enumerate(row_vals):
                self.frag_mask_scores_table.setItem(i, col, QTableWidgetItem(text))
        self.frag_mask_scores_table.resizeColumnsToContents()
        self._render_current_fragments_table()

        vmin, vmax = self._masking_color_scale_bounds()
        color_note = (
            f"fixed [{vmin:g},{vmax:g}]" if vmin is not None else "per-molecule auto"
        )
        self.frag_mask_status.setText(
            f"AAS(full)={payload.get('aas_full', 0):.4f}  |  R={payload.get('n_iterations')}  "
            f"f={payload.get('mask_fraction')}  |  candidate #{payload.get('candidate_rank_index', 0) + 1}  "
            f"|  Color: {color_note}  |  DEPT={self.frag_mask_dept_check.isChecked()}  "
            f"|  unique-match={self.frag_mask_unique_match_check.isChecked()}"
        )
        self._refresh_intra_fusion_run_enabled()

    def fragment_masking_finished(self, payload):
        # Backward-compatible single-result entrypoint.
        self.masking_thread = None
        self.frag_progress.setVisible(False)
        self.frag_mask_run_btn.setEnabled(True)
        if hasattr(self, "frag_mask_recolor_btn"):
            self.frag_mask_recolor_btn.setEnabled(True)
        if hasattr(self, "frag_mask_extract_btn"):
            self.frag_mask_extract_btn.setEnabled(True)
        self._display_masking_payload(payload)
        self.statusBar().showMessage("Masking attribution completed")

    def fragment_masking_error(self, msg):
        self.masking_thread = None
        self.masking_batch_cursor = self.masking_batch_total
        self.frag_progress.setVisible(False)
        self.frag_mask_run_btn.setEnabled(True)
        if hasattr(self, "frag_mask_recolor_btn"):
            self.frag_mask_recolor_btn.setEnabled(bool(getattr(self, "masking_result_data", None)))
        if hasattr(self, "frag_mask_extract_btn"):
            self.frag_mask_extract_btn.setEnabled(bool(getattr(self, "masking_result_data", None)))
        if hasattr(self, "frag_fusion_run_btn"):
            self.frag_fusion_run_btn.setEnabled(bool(getattr(self, "masking_fragments_all", None)))
        self._refresh_intra_fusion_run_enabled()
        self.frag_mask_status.setText("Error.")
        QMessageBox.critical(self, "Masking attribution failed", msg)
        self._fire_masking_batch_callbacks(ok=False, error=msg)

    def register_masking_batch_callback(self, callback):
        """Register a one-shot callback fired when the masking batch finishes.

        Used by the SOP runner (virmol_ai.runner) to wait for the
        ``_start_masking_batch_item`` recursive batch to complete. ``callback``
        is invoked with ``(ok: bool, error: Optional[str])`` exactly once and
        then removed from the queue.
        """
        if not hasattr(self, "_masking_batch_callbacks"):
            self._masking_batch_callbacks = []
        self._masking_batch_callbacks.append(callback)

    def _fire_masking_batch_callbacks(self, *, ok, error):
        callbacks = list(getattr(self, "_masking_batch_callbacks", []) or [])
        self._masking_batch_callbacks = []
        for cb in callbacks:
            try:
                cb(ok, error)
            except Exception as exc:
                print(f"masking batch callback failed: {exc}")

    def create_database_tab(self):
        """Create database management tab"""
        db_widget = QWidget()
        db_layout = QVBoxLayout(db_widget)
        db_layout.setContentsMargins(20, 20, 20, 20)
        
        # Database information
        info_group = QGroupBox("Database Statistics")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel("• All Database: 605,735 NPs\n"
                          "• Plant Database: 188,478 NPs\n"
                          "• Human Database: 217,347 NPs\n"
                          "• Microbial Database: 36,427 NPs")
        info_text.setStyleSheet("font-size: 16px; padding: 15px; line-height: 1.6;")
        info_layout.addWidget(info_text)
        
        db_layout.addWidget(info_group)
        
        # Database creation
        create_group = QGroupBox("Create New Database")
        create_layout = QGridLayout(create_group)
        
        self.smiles_input = QLineEdit()
        self.smiles_input.setPlaceholderText("Select CSV file containing SMILES")
        self.smiles_browse_btn = QPushButton()
        self.smiles_browse_btn.setObjectName("IconToolButton")
        self.smiles_browse_btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogStart))
        self.smiles_browse_btn.setToolTip("Browse SMILES File")
        self.smiles_browse_btn.clicked.connect(lambda: self.browse_file(self.smiles_input, "CSV files (*.csv)"))
        
        # Output path selection
        self.output_path_input = QLineEdit()
        self.output_path_input.setPlaceholderText(
            "Database output path (optional, defaults to GUI_result_files/generated_virDB.npz)"
        )
        self.output_path_browse_btn = QPushButton()
        self.output_path_browse_btn.setObjectName("IconToolButton")
        self.output_path_browse_btn.setIcon(self.style().standardIcon(QStyle.SP_FileDialogStart))
        self.output_path_browse_btn.setToolTip("Browse Output Path")
        self.output_path_browse_btn.clicked.connect(lambda: self.browse_output_path())
        
        self.create_db_btn = QPushButton("Create Database")
        self.create_db_btn.clicked.connect(self.create_database)
        
        create_layout.addWidget(QLabel("SMILES File:"), 0, 0)
        create_layout.addWidget(self.smiles_input, 0, 1)
        create_layout.addWidget(self.smiles_browse_btn, 0, 2)
        create_layout.addWidget(QLabel("Output Path:"), 1, 0)
        create_layout.addWidget(self.output_path_input, 1, 1)
        create_layout.addWidget(self.output_path_browse_btn, 1, 2)
        create_layout.addWidget(self.create_db_btn, 2, 0, 1, 3)
        
        db_layout.addWidget(create_group)
        
        # Progress and status display
        status_group = QGroupBox("Progress & Status")
        status_layout = QVBoxLayout(status_group)
        
        # Model file status
        self.model_status_label = QLabel("Model Status: Checking...")
        self.model_status_label.setStyleSheet("font-weight: bold; color: #6c757d;")
        status_layout.addWidget(self.model_status_label)
        
        # Progress bar
        self.db_progress_bar = QProgressBar()
        self.db_progress_bar.setVisible(False)
        self.db_progress_bar.setRange(0, 100)
        self.db_progress_bar.setValue(0)
        
        # Status text display
        self.db_status_text = QTextEdit()
        self.db_status_text.setMaximumHeight(150)
        self.db_status_text.setReadOnly(True)
        self.db_status_text.setPlaceholderText("Status will appear here during database creation...")
        self.db_status_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                font-size: 14px;
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        
        status_layout.addWidget(QLabel("Progress:"))
        status_layout.addWidget(self.db_progress_bar)
        status_layout.addWidget(QLabel("Status Log:"))
        status_layout.addWidget(self.db_status_text)
        
        # Check model file status
        self.check_model_status()
        
        db_layout.addWidget(status_group)
        db_layout.addStretch()
        
        self.tab_widget.addTab(db_widget, "Database Management")

    def create_other_tool_tab(self):
        """Create other utility tab (SMILES drawing + HD export)."""
        tool_widget = QWidget()
        tool_layout = QVBoxLayout(tool_widget)
        tool_layout.setContentsMargins(16, 16, 16, 16)
        tool_layout.setSpacing(10)

        input_group = QGroupBox("SMILES Structure Drawer")
        input_layout = QGridLayout(input_group)
        self.other_tool_smiles_input = QLineEdit()
        self.other_tool_smiles_input.setPlaceholderText("Input a SMILES string and click Draw")
        self.other_tool_draw_btn = QPushButton("Draw Structure")
        self.other_tool_draw_btn.clicked.connect(self.draw_other_tool_smiles)
        self.other_tool_smiles_input.returnPressed.connect(self.draw_other_tool_smiles)
        input_layout.addWidget(QLabel("SMILES:"), 0, 0)
        input_layout.addWidget(self.other_tool_smiles_input, 0, 1)
        input_layout.addWidget(self.other_tool_draw_btn, 0, 2)
        tool_layout.addWidget(input_group)

        self.other_tool_image_label = QLabel("Structure preview appears here.")
        self.other_tool_image_label.setMinimumSize(560, 420)
        self.other_tool_image_label.setAlignment(Qt.AlignCenter)
        self.other_tool_image_label.setStyleSheet(
            "border: 1px solid #dee2e6; background: #fff; border-radius: 4px;"
        )
        tool_layout.addWidget(self.other_tool_image_label, 1)

        ctrl_wrap = QWidget()
        ctrl_row = QHBoxLayout(ctrl_wrap)
        ctrl_row.setContentsMargins(0, 0, 0, 0)
        self.other_tool_show_atom_idx_check = QCheckBox("Show RDKit atom index on structure")
        self.other_tool_show_atom_idx_check.setChecked(False)
        self.other_tool_show_atom_idx_check.toggled.connect(self._refresh_other_tool_structure_image)
        ctrl_row.addWidget(self.other_tool_show_atom_idx_check)
        ctrl_row.addStretch()
        tool_layout.addWidget(ctrl_wrap)

        tf_wrap = QWidget()
        tf_row = QHBoxLayout(tf_wrap)
        tf_row.setContentsMargins(0, 0, 0, 0)
        tf_row.addWidget(QLabel("Rotate (deg):"))
        self.other_tool_rotate_spin = QDoubleSpinBox()
        self.other_tool_rotate_spin.setRange(-360.0, 360.0)
        self.other_tool_rotate_spin.setDecimals(1)
        self.other_tool_rotate_spin.setSingleStep(5.0)
        self.other_tool_rotate_spin.setValue(0.0)
        self.other_tool_rotate_spin.valueChanged.connect(self._refresh_other_tool_structure_image)
        tf_row.addWidget(self.other_tool_rotate_spin)
        self.other_tool_flip_h_check = QCheckBox("Flip H")
        self.other_tool_flip_h_check.toggled.connect(self._refresh_other_tool_structure_image)
        tf_row.addWidget(self.other_tool_flip_h_check)
        self.other_tool_flip_v_check = QCheckBox("Flip V")
        self.other_tool_flip_v_check.toggled.connect(self._refresh_other_tool_structure_image)
        tf_row.addWidget(self.other_tool_flip_v_check)
        self.other_tool_export_btn = QPushButton("Save HD (SVG/PNG)")
        self.other_tool_export_btn.clicked.connect(self._export_other_tool_hd_image)
        tf_row.addWidget(self.other_tool_export_btn)
        tf_row.addStretch()
        tool_layout.addWidget(tf_wrap)

        self.other_tool_status_label = QLabel("")
        self.other_tool_status_label.setStyleSheet("color: #6c757d;")
        tool_layout.addWidget(self.other_tool_status_label)

        self.other_tool_current_smiles = ""
        self.tab_widget.addTab(tool_widget, "Other Tool")

    def draw_other_tool_smiles(self):
        smiles = self.other_tool_smiles_input.text().strip() if hasattr(self, "other_tool_smiles_input") else ""
        if not smiles:
            QMessageBox.warning(self, "SMILES", "Please input a SMILES string.")
            return
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            QMessageBox.warning(self, "SMILES", "Invalid SMILES string.")
            return
        self.other_tool_current_smiles = smiles
        self._refresh_other_tool_structure_image()

    def _other_tool_transform_params(self):
        rotate_deg = float(self.other_tool_rotate_spin.value()) if hasattr(self, "other_tool_rotate_spin") else 0.0
        flip_h = bool(self.other_tool_flip_h_check.isChecked()) if hasattr(self, "other_tool_flip_h_check") else False
        flip_v = bool(self.other_tool_flip_v_check.isChecked()) if hasattr(self, "other_tool_flip_v_check") else False
        return rotate_deg, flip_h, flip_v

    def _draw_other_tool_structure_png(self, smiles, size=(1000, 760)):
        from PIL import Image
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D
        from rdkit.Geometry import Point3D
        import io
        import math

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        mol_draw = Chem.Mol(mol)
        AllChem.Compute2DCoords(mol_draw)
        rotate_deg, flip_h, flip_v = self._other_tool_transform_params()
        if abs(float(rotate_deg)) > 1e-9 or flip_h or flip_v:
            conf = mol_draw.GetConformer()
            xs = [conf.GetAtomPosition(i).x for i in range(mol_draw.GetNumAtoms())]
            ys = [conf.GetAtomPosition(i).y for i in range(mol_draw.GetNumAtoms())]
            cx = float(np.mean(xs)) if xs else 0.0
            cy = float(np.mean(ys)) if ys else 0.0
            ang = math.radians(-float(rotate_deg))
            ca, sa = math.cos(ang), math.sin(ang)
            for i in range(mol_draw.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                x = p.x - cx
                y = p.y - cy
                if flip_h:
                    x = -x
                if flip_v:
                    y = -y
                xr = x * ca - y * sa
                yr = x * sa + y * ca
                conf.SetAtomPosition(i, Point3D(float(xr), float(yr), float(p.z)))

        w, h = int(size[0]), int(size[1])
        d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
        opts = d2d.drawOptions()
        opts.clearBackground = True
        try:
            opts.useBWAtomPalette()
        except Exception:
            pass
        if self.other_tool_show_atom_idx_check.isChecked():
            amap = rdMolDraw2D.IntStringMap()
            for aid in range(mol_draw.GetNumAtoms()):
                amap[int(aid)] = str(int(aid))
            opts.atomLabels = amap
            opts.annotationFontScale = 0.7
        rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol_draw)
        d2d.FinishDrawing()
        png = d2d.GetDrawingText()
        return Image.open(io.BytesIO(png)).convert("RGB")

    def _draw_other_tool_structure_svg(self, smiles, size=(2200, 1700)):
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D
        from rdkit.Geometry import Point3D
        import math

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        mol_draw = Chem.Mol(mol)
        AllChem.Compute2DCoords(mol_draw)
        rotate_deg, flip_h, flip_v = self._other_tool_transform_params()
        if abs(float(rotate_deg)) > 1e-9 or flip_h or flip_v:
            conf = mol_draw.GetConformer()
            xs = [conf.GetAtomPosition(i).x for i in range(mol_draw.GetNumAtoms())]
            ys = [conf.GetAtomPosition(i).y for i in range(mol_draw.GetNumAtoms())]
            cx = float(np.mean(xs)) if xs else 0.0
            cy = float(np.mean(ys)) if ys else 0.0
            ang = math.radians(-float(rotate_deg))
            ca, sa = math.cos(ang), math.sin(ang)
            for i in range(mol_draw.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                x = p.x - cx
                y = p.y - cy
                if flip_h:
                    x = -x
                if flip_v:
                    y = -y
                xr = x * ca - y * sa
                yr = x * sa + y * ca
                conf.SetAtomPosition(i, Point3D(float(xr), float(yr), float(p.z)))

        w, h = int(size[0]), int(size[1])
        try:
            d2d = rdMolDraw2D.MolDraw2DSVG(w, h, -1, -1, True)
        except Exception:
            d2d = rdMolDraw2D.MolDraw2DSVG(w, h)
        opts = d2d.drawOptions()
        opts.clearBackground = True
        # SVG export: enlarge atom element symbols for better editability/readability.
        try:
            opts.baseFontSize = float(opts.baseFontSize) * 2.0
        except Exception:
            pass
        try:
            opts.fixedFontSize = float(opts.fixedFontSize) * 2.0
        except Exception:
            pass
        try:
            opts.useBWAtomPalette()
        except Exception:
            pass
        if self.other_tool_show_atom_idx_check.isChecked():
            amap = rdMolDraw2D.IntStringMap()
            for aid in range(mol_draw.GetNumAtoms()):
                amap[int(aid)] = str(int(aid))
            opts.atomLabels = amap
            opts.annotationFontScale = 0.7
        rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol_draw)
        d2d.FinishDrawing()
        svg = d2d.GetDrawingText()
        return svg if isinstance(svg, str) else svg.decode("utf-8")

    def _refresh_other_tool_structure_image(self):
        smiles = getattr(self, "other_tool_current_smiles", "")
        if not smiles:
            return
        try:
            img = self._draw_other_tool_structure_png(smiles, size=(1000, 760))
            self.other_tool_image_label.setPixmap(self.pil_image_to_pixmap(img))
            self.other_tool_image_label.setText("")
            r, fh, fv = self._other_tool_transform_params()
            self.other_tool_status_label.setText(
                f"Rendered. Rotate={r:.1f}°, FlipH={fh}, FlipV={fv}, AtomIndex={self.other_tool_show_atom_idx_check.isChecked()}"
            )
        except Exception as e:
            self.other_tool_image_label.setText(f"Image error: {e}")

    def _export_other_tool_hd_image(self):
        smiles = getattr(self, "other_tool_current_smiles", "")
        if not smiles:
            QMessageBox.warning(self, "Export", "Please draw a SMILES structure first.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Other Tool structure",
            "other_tool_structure.svg",
            "SVG Files (*.svg);;PNG Files (*.png);;TIFF Files (*.tif *.tiff);;JPEG Files (*.jpg *.jpeg)",
        )
        if not file_path:
            return
        try:
            lower = file_path.lower()
            if lower.endswith(".svg"):
                svg_text = self._draw_other_tool_structure_svg(smiles, size=(2200, 1700))
                svg_text = self._fit_svg_to_physical_width(svg_text, target_width_mm=70.0)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(svg_text)
            else:
                img = self._draw_other_tool_structure_png(smiles, size=(2400, 1800))
                img.save(file_path, dpi=(300, 300))
            extra_detail = {
                "smiles": smiles,
                "show_atom_idx": self.other_tool_show_atom_idx_check.isChecked(),
                "rotate_deg": self._other_tool_transform_params()[0],
                "flip_horizontal": self._other_tool_transform_params()[1],
                "flip_vertical": self._other_tool_transform_params()[2],
                "svg_target_width_mm": 70.0 if lower.endswith(".svg") else None,
            }
            param_path = self._write_analysis_params_sidecar(
                file_path, "other_tool_structure_hd", extra=extra_detail
            )
            self.statusBar().showMessage(f"Other Tool structure exported: {file_path}")
            msg = f"Other Tool structure exported:\n{file_path}"
            if param_path:
                msg += f"\n\nAnalysis parameters:\n{param_path}"
            QMessageBox.information(self, "Export", msg)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
    

    
    def browse_folder(self, line_edit):
        """Browse folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            line_edit.setText(folder)
    
    def browse_file(self, line_edit, filter_str):
        """Browse file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", filter_str)
        if file_path:
            line_edit.setText(file_path)
    
    def browse_output_path(self):
        """Browse output path for database"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Database As", 
            "generated_virDB.npz", 
            "NPZ files (*.npz);;All files (*.*)"
        )
        if file_path:
            self.output_path_input.setText(file_path)
    

    
    def switch_plot(self, plot_type):
        """Switch plot display"""
        if plot_type == "peak":
            self.current_plot_type = "peak"
            self.peak_plot_btn.setChecked(True)
            self.merge_plot_btn.setChecked(False)
            if self.peak_plot_data is not None:
                self.update_spectral_plot("peak")
        elif plot_type == "merge":
            self.current_plot_type = "merge"
            self.peak_plot_btn.setChecked(False)
            self.merge_plot_btn.setChecked(True)
            if self.merge_plot_data is not None:
                self.update_spectral_plot("merge")
    
    def update_plot_buttons(self):
        """Update plot switch button states"""
        if self.peak_plot_data is not None:
            self.peak_plot_btn.setEnabled(True)
        else:
            self.peak_plot_btn.setEnabled(False)
            self.peak_plot_btn.setChecked(False)
        
        if self.merge_plot_data is not None:
            self.merge_plot_btn.setEnabled(True)
        else:
            self.merge_plot_btn.setEnabled(False)
            self.merge_plot_btn.setChecked(False)
    

    
    def update_spectral_plot(self, plot_type=None):
        """Update spectral plot display"""
        try:
            if plot_type is None:
                plot_type = self.current_plot_type
            
            # Clear current figure
            self.figure.clear()
            
            if plot_type == "peak" and hasattr(self.test, 'data'):
                try:
                    # DEPT135 subplot
                    ax1 = self.figure.add_subplot(3,1,1)
                    ax1.plot(self.test.uc135.ppm_scale(), self.test.data135)
                    for peak135 in self.test.peaks135:
                        height = self.test.data135[int(peak135["X_AXIS"])]
                        ppm = self.test.uc135.ppm(peak135["X_AXIS"])
                        ax1.scatter(ppm, height, marker="o", color="r", s=50, alpha=0.5)
                    ax1.hlines(self.test.threshold_C135pos, *self.test.uc135.ppm_limits(), linestyle="--", color="k")
                    ax1.hlines(self.test.threshold_C135neg, *self.test.uc135.ppm_limits(), linestyle="--", color="k")
                    ax1.set_xlim(200, -2)
                    ax1.set_xticks([])
                    ax1.grid(True, alpha=0.3)
                    
                    # DEPT90 subplot
                    ax2 = self.figure.add_subplot(3,1,2)
                    ax2.plot(self.test.uc90.ppm_scale(), self.test.data90)
                    for peak90 in self.test.peaks90:
                        height = self.test.data90[int(peak90["X_AXIS"])]
                        ppm = self.test.uc90.ppm(peak90["X_AXIS"])
                        ax2.scatter(ppm, height, marker="o", color="r", s=50, alpha=0.5)
                    ax2.hlines(self.test.threshold_C90, *self.test.uc90.ppm_limits(), linestyle="--", color="k")
                    ax2.set_xlim(200, -2)
                    ax2.set_xticks([])
                    ax2.grid(True, alpha=0.3)
                    
                    # 13C_NMR subplot
                    ax3 = self.figure.add_subplot(3,1,3)
                    ax3.plot(self.test.uc.ppm_scale(), self.test.data)
                    for peak in self.test.peaks:
                        height = self.test.data[int(peak["X_AXIS"])]
                        ppm = self.test.uc.ppm(peak["X_AXIS"])
                        ax3.scatter(ppm, height, marker="o", color="r", s=50, alpha=0.5)
                    ax3.hlines(self.test.threshold_C, *self.test.uc.ppm_limits(), linestyle="--", color="k")
                    ax3.set_xlim(200, -2)
                    ax3.set_ylim(0, 1e8)
                    ax3.grid(True, alpha=0.3)
                    ax3.set_xlabel('Chemical Shift (ppm)')
                    
                    # Set ax3 as the last used ax for subsequent label settings
                    ax = ax3
                    
                except Exception as e:
                    print(f"Failed to draw peak detection plot: {str(e)}")
                    # Create single subplot for error message
                    ax = self.figure.add_subplot(111)
                    ax.text(0.5, 0.5, f'Failed to draw peak detection plot: {str(e)}', 
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=12, color='red')
                
            elif plot_type == "merge" and self.merge_plot_data is not None:
                # Create single subplot for merged spectrum
                ax = self.figure.add_subplot(111)
                
                # Define better colors for carbon types with higher contrast
                ch3_color = '#FF0000'  # Pure red
                ch2_color = '#00FF00'  # Pure green
                ch_color = '#0000FF'   # Pure blue
                c_color = '#FF8C00'    # Dark orange
                
                # CH3 (coral red) - positive peaks
                ch3_bars = []
                for ppm in self.merge_plot_data['hit_CH3']:
                    index = self.merge_plot_data['ppmcarbon'].index(ppm)
                    height = self.merge_plot_data['heightcarbon'][index]
                    bar = ax.bar(ppm, height, width=0.6, color=ch3_color, alpha=0.8)
                    ch3_bars.append(bar)
                
                # CH2 (turquoise) - negative peaks
                ch2_bars = []
                for ppm in self.merge_plot_data['hit_CH2']:
                    index = self.merge_plot_data['ppmcarbon'].index(ppm)
                    height = self.merge_plot_data['heightcarbon'][index]
                    bar = ax.bar(ppm, -height, width=0.6, color=ch2_color, alpha=0.8)
                    ch2_bars.append(bar)
                
                # CH (sky blue) - positive peaks
                ch_bars = []
                for ppm in self.merge_plot_data['hit_CH']:
                    index = self.merge_plot_data['ppmcarbon'].index(ppm)
                    height = self.merge_plot_data['heightcarbon'][index]
                    bar = ax.bar(ppm, height, width=0.6, color=ch_color, alpha=0.8)
                    ch_bars.append(bar)
                
                # C (mint green) - negative peaks
                c_bars = []
                for ppm in self.merge_plot_data['Hit_C_or_unhited']:
                    index = self.merge_plot_data['ppmcarbon'].index(ppm)
                    height = self.merge_plot_data['heightcarbon'][index]
                    bar = ax.bar(ppm, -height, width=0.6, color=c_color, alpha=0.8)
                    c_bars.append(bar)
                
                # Add legend for carbon types
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor=ch3_color, alpha=0.8, label='CH3 (Methyl)'),
                    Patch(facecolor=ch2_color, alpha=0.8, label='CH2 (Methylene)'),
                    Patch(facecolor=ch_color, alpha=0.8, label='CH (Methine)'),
                    Patch(facecolor=c_color, alpha=0.8, label='C (Quaternary)')
                ]
                ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
                
                ax.set_title('Merged Spectrum')
                ax.hlines(0, *self.test.uc.ppm_limits(), linestyle='-', color='k')
                ax.set_xlim(200, -2)
                ax.set_ylim(-0.6e8, 0.6e8)
                
            else:
                # Create single subplot for prompt
                ax = self.figure.add_subplot(111)
                # Show waiting prompt
                ax.text(0.5, 0.5, 'Please perform peak detection or data merge first', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=14, color='gray')
                ax.set_title('Waiting for Data...')
                # Set axis range
                ax.set_xlim(200, -2)
            
            # Set axes
            ax.set_xlabel('Chemical Shift (ppm)')
            ax.set_ylabel('Intensity')
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()  # Invert x-axis
            ax.set_xlim(200, -2)  # Set x-axis range
            
            # Adjust layout and refresh canvas
            self.figure.tight_layout()
            self.canvas.draw()
            
            print(f"Plot updated: {plot_type}")
            
        except Exception as e:
            print(f"Failed to update spectral plot: {str(e)}")
            import traceback
            traceback.print_exc()
    

    

    
    def generate_peak_plot_data(self):
        """Generate peak detection plot data"""
        try:
            if self.test is not None and hasattr(self.test, 'hit_CH3'):
                # Get actual peak detection data
                x_data = self.test.x_data  # Assume x_data is stored in test object
                y_data = self.test.y_data  # Assume y_data is stored in test object
                return y_data
            else:
                return None
        except Exception as e:
            print(f"Failed to generate peak detection plot data: {str(e)}")
            return None
    
    def generate_merge_plot_data(self):
        """Generate merged plot data"""
        try:
            if self.test is not None and hasattr(self.test, 'hit_CH3'):
                # Get actual merged data
                return {
                    'hit_CH3': self.test.hit_CH3,
                    'hit_CH2': self.test.hit_CH2,
                    'hit_CH': self.test.hit_CH,
                    'Hit_C_or_unhited': self.test.Hit_C_or_unhited,
                    'ppmcarbon': self.test.ppmcarbon,
                    'heightcarbon': self.test.heightcarbon
                }
            else:
                return None
        except Exception as e:
            print(f"Failed to generate merged plot data: {str(e)}")
            return None
    
    def update_peak_info_table(self):
        """Fill peak table from manual input if set, else from NMR-1D.csv."""
        try:
            if hasattr(self, "manual_peak_input"):
                mt = self.manual_peak_input.toPlainText().strip()
                if mt:
                    try:
                        parsed = self.parse_manual_peak_input_strict(mt)
                    except ValueError as e:
                        self.statusBar().showMessage(f"Manual peak input invalid: {e}")
                        return
                    if parsed is not None:
                        shifts, ctypes = parsed
                        self.peak_info_table.setRowCount(0)
                        self.peak_info_table.setColumnCount(3)
                        self.peak_info_table.setHorizontalHeaderLabels(["ppm", "Type", "Intensity"])
                        n = len(shifts)
                        self.peak_info_table.setRowCount(n)
                        for row in range(n):
                            ppm_value = float(shifts[row])
                            ppm_item = QTableWidgetItem(f"{ppm_value:.2f}")
                            self.peak_info_table.setItem(row, 0, ppm_item)
                            type_value = str(ctypes[row]).strip()
                            type_item = QTableWidgetItem(type_value)
                            if type_value == "q":
                                type_item.setBackground(QColor(255, 182, 193))
                                type_item.setText("CH3")
                            elif type_value == "t":
                                type_item.setBackground(QColor(144, 238, 144))
                                type_item.setText("CH2")
                            elif type_value == "d":
                                type_item.setBackground(QColor(173, 216, 230))
                                type_item.setText("CH")
                            elif type_value == "s":
                                type_item.setBackground(QColor(255, 218, 185))
                                type_item.setText("C")
                            else:
                                type_item.setText(f"Unknown ({type_value})")
                            self.peak_info_table.setItem(row, 1, type_item)
                            intensity_item = QTableWidgetItem("1.00")
                            self.peak_info_table.setItem(row, 2, intensity_item)
                        self.peak_info_table.sortItems(0, Qt.DescendingOrder)
                        self.peak_info_table.setColumnWidth(0, 80)
                        self.peak_info_table.setColumnWidth(1, 60)
                        self.peak_info_table.setColumnWidth(2, 120)
                        self.peak_info_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                        self.statusBar().showMessage(f"Peak table: {n} manual peaks")
                        return
            
            # Read NMR-1D.csv file directly
            csv_file = self._nmr_csv_path()
            print(f"Attempting to read CSV file: {csv_file}")
            
            if not os.path.exists(csv_file):
                print(f"CSV file not found: {csv_file}")
                # Show message in status bar
                self.statusBar().showMessage(f"CSV file not found: {csv_file}")
                return
            
            # Read CSV data
            import pandas as pd
            df = pd.read_csv(csv_file, header=None)
            print(f"CSV file loaded successfully. Shape: {df.shape}, Columns: {len(df.columns)}")
            print(f"First few rows: {df.head()}")
            
            if df.empty:
                print("CSV file is empty")
                return
            
            if len(df.columns) < 3:
                print(f"CSV file has insufficient columns: {len(df.columns)}")
                print(f"Column data: {df.columns.tolist()}")
                return
            
            # Skip the first row (0, 1, 2) and use data from second row onwards
            if len(df) > 1:
                df = df.iloc[1:]  # Skip first row
                print(f"Removed first row, now using {len(df)} data rows")
            
            # Clear existing data
            self.peak_info_table.setRowCount(0)
            
            # Set column headers with simpler titles
            self.peak_info_table.setColumnCount(3)
            self.peak_info_table.setHorizontalHeaderLabels(["ppm", "Type", "Intensity"])
            
            # Populate table with CSV data
            self.peak_info_table.setRowCount(len(df))
            print(f"Setting table to {len(df)} rows")
            
            for row, (_, csv_row) in enumerate(df.iterrows()):
                try:
                    # Column 0: ppm
                    ppm_value = float(csv_row[0])
                    ppm_item = QTableWidgetItem(f"{ppm_value:.2f}")
                    self.peak_info_table.setItem(row, 0, ppm_item)
                    
                    # Column 1: type
                    type_value = str(csv_row[1]).strip()
                    type_item = QTableWidgetItem(type_value)
                    
                    # Color code based on carbon type
                    if type_value == 'q':
                        type_item.setBackground(QColor(255, 182, 193))  # Light red for CH3
                        type_item.setText('CH3')
                    elif type_value == 't':
                        type_item.setBackground(QColor(144, 238, 144))  # Light green for CH2
                        type_item.setText('CH2')
                    elif type_value == 'd':
                        type_item.setBackground(QColor(173, 216, 230))  # Light blue for CH
                        type_item.setText('CH')
                    elif type_value == 's':
                        type_item.setBackground(QColor(255, 218, 185))  # Light orange for C
                        type_item.setText('C')
                    else:
                        type_item.setText(f"Unknown ({type_value})")
                    
                    self.peak_info_table.setItem(row, 1, type_item)
                    
                    # Column 2: intensity
                    intensity_value = float(csv_row[2])
                    intensity_item = QTableWidgetItem(f"{intensity_value:,.0f}")
                    self.peak_info_table.setItem(row, 2, intensity_item)
                    
                    print(f"Row {row}: ppm={ppm_value:.2f}, type={type_value}, intensity={intensity_value:,.0f}")
                    
                except Exception as row_error:
                    print(f"Error processing row {row}: {row_error}")
                    print(f"Row data: {csv_row.tolist()}")
                    continue
            
            # Sort by chemical shift (ppm) - highest to lowest
            self.peak_info_table.sortItems(0, Qt.DescendingOrder)
            
            # Set optimal column widths to fit all information without scrolling
            self.peak_info_table.setColumnWidth(0, 80)   # ppm column - enough for 2 decimal places
            self.peak_info_table.setColumnWidth(1, 60)   # Type column - enough for CH3, CH2, CH, C
            self.peak_info_table.setColumnWidth(2, 120)  # Intensity column - enough for large numbers
            
            # Disable horizontal scrolling to ensure all columns are visible
            self.peak_info_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            
            print(f"Peak info table updated successfully with {len(df)} rows from {csv_file}")
            self.statusBar().showMessage(f"Peak info table updated with {len(df)} rows")
            
        except Exception as e:
            print(f"Failed to update peak info table: {str(e)}")
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f"Failed to update peak info table: {str(e)}")
    
    def load_model(self):
        """Load model"""
        try:
            self.statusBar().showMessage("Model loaded successfully")
        except Exception as e:
            self.statusBar().showMessage(f"Failed to load model: {str(e)}")

    def _set_label_pixmap_fit(self, label, pixmap):
        """Set pixmap fitted into label while keeping aspect ratio."""
        if label is None or pixmap is None or pixmap.isNull():
            return
        w = max(1, int(label.width()))
        h = max(1, int(label.height()))
        fitted = pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(fitted)
    
    def submit_peaks(self):
        """Submit peak detection"""
        try:
            if not all([self.c13_input.text(), self.dept90_input.text(), self.dept135_input.text()]):
                QMessageBox.warning(self, "Warning", "Please select all required data folders")
                return
            
            self.statusBar().showMessage("Processing peak detection...")
            
            # Create CNMR_process object
            self.test = CNMR_process(
                self.c13_input.text(),
                self.dept135_input.text(),
                self.dept90_input.text(),
                pipe_output_dir=self.gui_output_dir,
            )
            
            # Get peak data
            self.test.get_13CDEPT_peak(
                threshold_C=self.threshold_c.value(),
                threshold_C90=self.threshold_c90.value(),
                threshold_C135pos=self.threshold_c135pos.value(),
                threshold_C135neg=self.threshold_c135neg.value(),
                DrawWin="False"  # Don't plot in NMR1D.py
            )
            
            # Check if data was correctly obtained
            if not hasattr(self.test, 'data') or not hasattr(self.test, 'data135') or not hasattr(self.test, 'data90'):
                raise Exception("Failed to get spectral data correctly")
            if not hasattr(self.test, 'peaks') or not hasattr(self.test, 'peaks135') or not hasattr(self.test, 'peaks90'):
                raise Exception("Failed to get peak data correctly")
            
            # Update plot display
            self.update_spectral_plot("peak")
            
            # Update plot display
            self.update_spectral_plot("peak")
            
            # Close all possible matplotlib windows
            plt.close('all')
            
            # Update button states
            self.update_plot_buttons()
            
            self.combine_btn.setEnabled(True)
            self.statusBar().showMessage("Peak detection completed")
            QMessageBox.information(self, "Success", "Peak detection completed")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Peak detection failed: {str(e)}")
            self.statusBar().showMessage("Peak detection failed")
    
    def combine_data(self):
        """Combine data"""
        try:
            if not self.test:
                QMessageBox.warning(self, "Warning", "Please perform peak detection first")
                return
            
            self.statusBar().showMessage("Merging data...")
            
            # Sort and reconstruct
            self.test.sort_Ctype_mindelta()
            self.test.CNMR_reconstract(
                width=0.6, fontsize='medium', chemical_shift="False",
                label="False", summary="True", x_left=200, x_right=-2,
                y_top=0.6*1e8, y_bottom=-0.6*1e8, DrawWin="True"  # Use original plotting method
            )
            # Update plot display
            self.update_spectral_plot("merge")
            
            # Combine data
            self.test.combine_data(CarbonFileName=self._gui_output_path("NMR-1D.csv"))
            
            # Get merged data
            self.merge_plot_data = self.generate_merge_plot_data()
            
            # Update plot display
            self.update_spectral_plot("merge")
            
            # Update peak information table
            self.update_peak_info_table()
            
            # Close all possible matplotlib windows
            plt.close('all')
            
            # Update button states
            self.update_plot_buttons()
            
            self.solvent_submit_btn.setEnabled(True)
            self.impurity_submit_btn.setEnabled(True)
            self.statusBar().showMessage("Data merge completed")
            QMessageBox.information(self, "Success", "Data merge completed")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Data merge failed: {str(e)}")
            self.statusBar().showMessage("Data merge failed")
    
    def submit_solvent_removal(self):
        """Submit solvent removal"""
        try:
            if not self.test:
                QMessageBox.warning(self, "Warning", "Please complete data merge first")
                return
            
            self.statusBar().showMessage("Removing solvent...")
            
            solvent_type = self.type_combo.currentText()
            solvent = self.solvent_combo.currentText()
            
            if solvent_type == "CH3":
                self.test.solvent_remove(self.test.hit_CH3, type="CH3", solvent=solvent)
            elif solvent_type == "CH2":
                self.test.solvent_remove(self.test.hit_CH2, type="CH2", solvent=solvent)
            elif solvent_type == "CH":
                self.test.solvent_remove(self.test.hit_CH, type="CH", solvent=solvent)
            elif solvent_type == "C":
                self.test.solvent_remove(self.test.Hit_C_or_unhited, type="C", solvent=solvent)
            
            # Reconstruct image
            self.test.CNMR_reconstract(
                width=0.6, fontsize='medium', chemical_shift="False",
                label="False", summary="True", x_left=200, x_right=-2,
                y_top=0.6*1e8, y_bottom=-0.6*1e8, DrawWin="False"  # Set to False to prevent pop-up window
            )
            
            # Recombine data
            self.test.combine_data(CarbonFileName=self._gui_output_path("NMR-1D.csv"))
            
            # Update plot display
            self.merge_plot_data = self.generate_merge_plot_data()
            self.update_spectral_plot("merge")
            
            # Update peak information table
            self.update_peak_info_table()
            
            # Update button states
            self.update_plot_buttons()
            
            # Close all possible matplotlib windows
            plt.close('all')
            
            self.statusBar().showMessage("Solvent removal completed")
            QMessageBox.information(self, "Success", "Solvent removal completed")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Solvent removal failed: {str(e)}")
            self.statusBar().showMessage("Solvent removal failed")
    
    def submit_impurity_removal(self):
        """Submit impurity removal"""
        try:
            if not self.test:
                QMessageBox.warning(self, "Warning", "Please complete data merge first")
                return
            
            if not self.impurity_threshold1.text():
                QMessageBox.warning(self, "Warning", "Please enter threshold 1")
                return
            
            self.statusBar().showMessage("Removing impurities...")
            
            impurity_type = self.impurity_type_combo.currentText()
            threshold1 = float(self.impurity_threshold1.text())
            threshold2 = float(self.impurity_threshold2.text())
            threshold = threshold1 * threshold2
            
            if impurity_type == "CH3":
                self.test.impurity_removal(self.test.hit_CH3, type="CH3", rate=0.5, 
                                        Standsrd_num=0, method="absolute", threshold=threshold)
            elif impurity_type == "CH2":
                self.test.impurity_removal(self.test.hit_CH2, type="CH2", rate=0.5, 
                                        Standsrd_num=0, method="absolute", threshold=threshold)
            elif impurity_type == "CH":
                self.test.impurity_removal(self.test.hit_CH, type="CH", rate=0.5, 
                                        Standsrd_num=0, method="absolute", threshold=threshold)
            elif impurity_type == "C":
                self.test.impurity_removal(self.test.Hit_C_or_unhited, type="C", rate=0.5, 
                                        Standsrd_num=0, method="absolute", threshold=threshold)
            
            # Reconstruct image
            self.test.CNMR_reconstract(
                width=0.6, fontsize='medium', chemical_shift="False",
                label="False", summary="True", x_left=200, x_right=-2,
                y_top=0.6*1e8, y_bottom=-0.6*1e8, DrawWin="False"  # Set to False to prevent pop-up window
            )
            
            # Recombine data
            self.test.combine_data(CarbonFileName=self._gui_output_path("NMR-1D.csv"))
            
            # Update plot display
            self.merge_plot_data = self.generate_merge_plot_data()
            self.update_spectral_plot("merge")
            
            # Update peak information table
            self.update_peak_info_table()
            
            # Update button states
            self.update_plot_buttons()
            
            # Close all possible matplotlib windows
            plt.close('all')
            
            self.statusBar().showMessage("Impurity removal completed")
            QMessageBox.information(self, "Success", "Impurity removal completed")
            
        except ValueError as ve:
            QMessageBox.critical(self, "Error", "Please enter valid thresholds")
            self.statusBar().showMessage("Impurity removal failed: Invalid thresholds")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Impurity removal failed: {str(e)}")
            self.statusBar().showMessage("Impurity removal failed")
    
    def load_database(self):
        """Load database"""
        try:
            # Determine selected database from combo box
            selected_text = self.db_combo.currentText()
            
            if "Plant Database" in selected_text:
                db_name = "Plant"
            elif "Human Database" in selected_text:
                db_name = "Human"
            elif "Microbial Database" in selected_text:
                db_name = "Microorganism"
            elif "Drug Database" in selected_text:
                db_name = "Drug"
            elif "All Database" in selected_text:
                db_name = "AllDB"
            else:
                db_name = "Plant"  # Default fallback
            
            db_path = f"Database/{db_name}.npz"
            self.database1 = np.load(db_path, allow_pickle=True)["data"][()]
            
            # Change button color to medium gray to indicate successful loading
            self.load_db_btn.setStyleSheet(
                "QPushButton { background-color: #4e6a5c; color: #f4f8f6; font-weight: bold; "
                "border: 1px solid #3d5548; border-radius: 8px; padding: 14px 28px; }"
            )
            self.load_db_btn.setText("✓ Database Loaded")
            
            self.analysis_submit_btn.setEnabled(True)
            self.statusBar().showMessage(f"Database {db_name} loaded successfully")
            QMessageBox.information(self, "Success", f"Database {db_name} loaded successfully")
            
        except Exception as e:
            # Reset button color on error
            self.load_db_btn.setStyleSheet("")
            self.load_db_btn.setText("Load Database")
            QMessageBox.critical(self, "Error", f"Failed to load database: {str(e)}")
            self.statusBar().showMessage("Failed to load database")
    
    def load_other_database(self):
        """Load other database"""
        try:
            if not self.other_db_input.text():
                QMessageBox.warning(self, "Warning", "Please select a database file")
                return
            
            self.database1 = np.load(self.other_db_input.text(), allow_pickle=True)["data"][()]
            
            # Change button color to medium gray to indicate successful loading
            self.load_other_db_btn.setStyleSheet(
                "QPushButton { background-color: #4e6a5c; color: #f4f8f6; font-weight: bold; "
                "border: 1px solid #3d5548; border-radius: 8px; padding: 14px 28px; }"
            )
            self.load_other_db_btn.setText("✓ Other DB Loaded")
            
            self.analysis_submit_btn.setEnabled(True)
            self.statusBar().showMessage("Other database loaded successfully")
            QMessageBox.information(self, "Success", "Other database loaded successfully")
            
        except Exception as e:
            # Reset button color on error
            self.load_other_db_btn.setStyleSheet("")
            self.load_other_db_btn.setText("Load Other Database")
            QMessageBox.critical(self, "Error", f"Failed to load other database: {str(e)}")
            self.statusBar().showMessage("Failed to load other database")
    
    def start_analysis(self):
        """Start analysis"""
        try:
            if self._is_thread_running("analysis_thread"):
                QMessageBox.information(
                    self,
                    "Please wait",
                    "Molecular analysis is still running. Please wait for it to finish.",
                )
                return
            self._release_analysis_thread()
            if not self.database1:
                QMessageBox.warning(self, "Warning", "Please load a database first")
                return
            
            manual_text = self.manual_peak_input.toPlainText()
            try:
                manual_parsed = self.parse_manual_peak_input_strict(manual_text)
            except ValueError as e:
                QMessageBox.warning(self, "Invalid manual peak input", str(e))
                return
            
            if manual_parsed is not None:
                shifts, ctype = manual_parsed
            else:
                csv_file = self._nmr_csv_path()
                if not os.path.exists(csv_file):
                    QMessageBox.warning(
                        self,
                        "Warning",
                        f"No peak data: either complete preprocessing ({csv_file}) or enter manual peaks above.",
                    )
                    return
                df = pd.read_csv(csv_file)
                shifts = df.iloc[:, 0].tolist()
                ctype = df.iloc[:, 1].tolist()
            
            self.update_peak_info_table()
            
            # Determine filters and evaluator
            filters = []
            if self.cnf_checkbox.isChecked():
                filters.append("CNF")
            if self.ctnf_checkbox.isChecked():
                filters.append("CTNF")
            if self.mw_checkbox.isChecked():
                filters.append("MW")
            
            evaluator = ""
            if self.evaluator_css.isChecked():
                evaluator = "CSS"
            elif self.evaluator_aas.isChecked():
                evaluator = "AAS"
            elif self.evaluator_fps.isChecked():
                evaluator = "FPS"
            elif self.evaluator_fpaacs.isChecked():
                evaluator = "FPAACS"
            
            # Prepare parameters
            params = {
                'CNFbias': self.cnf_bias.value(),
                'CTNFbias': self.ctnf_bias.value(),
                'MWlist': [float(x.strip()) for x in self.mw_list.text().split(',')],
                'MWbias': 5,
                'TopN': 80,
                'weights': [float(x.strip()) for x in self.fpaacs_weights.text().split(',')],
                'result_csv_path': self._gui_output_path("Result.csv"),
            }
            
            # Create analysis thread
            self.analysis_thread = AnalysisThread(shifts, ctype, self.database1, filters, evaluator, params)
            self.analysis_thread.progress.connect(self.progress_bar.setValue)
            self.analysis_thread.result_ready.connect(self.analysis_finished)
            self.analysis_thread.error.connect(self.analysis_error)
            
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.analysis_submit_btn.setEnabled(False)
            
            # Start analysis
            self.analysis_thread.start()
            self.statusBar().showMessage("Analysis in progress...")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start analysis: {str(e)}")
            self.statusBar().showMessage("Failed to start analysis")
    
    def analysis_finished(self, result):
        """Analysis completed"""
        try:
            self.result = result
            self.progress_bar.setVisible(False)
            
            # Update results table
            self.update_results_table(result)
            
            # Select first row to show details
            if self.results_table.rowCount() > 0:
                self.results_table.selectRow(0)
            
            self.statusBar().showMessage("Analysis completed. Click a result row to open details.")
            QMessageBox.information(
                self,
                "Success",
                f"Analysis completed! Results have been saved to {self._gui_output_path('Result.csv')}",
            )

            if hasattr(self, "frag_mask_run_btn"):
                self.frag_mask_run_btn.setEnabled(True)
            if hasattr(self, "frag_mask_topn_spin"):
                self.frag_mask_topn_spin.setMaximum(max(1, len(result)))
                if self.frag_mask_topn_spin.value() > len(result):
                    self.frag_mask_topn_spin.setValue(len(result))
            if hasattr(self, "frag_mask_status"):
                self.frag_mask_status.setText(
                    "Molecular analysis results ready — run masking attribution when peaks are set."
                )
            
            print(f"Analysis completed successfully with {len(result)} results")
            if len(result) == 0:
                self._ai_show_rule_diagnosis_in_panel()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display results: {str(e)}")
            self.statusBar().showMessage("Failed to display results")
            print(f"Error in analysis_finished: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self._release_analysis_thread()
            if self.database1 is not None:
                self.analysis_submit_btn.setEnabled(True)
    
    def analysis_error(self, error_msg):
        """Analysis error"""
        try:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Analysis failed: {error_msg}")
            self.statusBar().showMessage("Analysis failed")
            self._ai_show_rule_diagnosis_in_panel()
        finally:
            self._release_analysis_thread()
            if self.database1 is not None:
                self.analysis_submit_btn.setEnabled(True)
    
    def _ensure_compound_detail_dialog(self):
        """Create detail popup lazily and reuse it."""
        if self.compound_detail_dialog is not None:
            return self.compound_detail_dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Compound Details")
        dlg.resize(1300, 760)

        root = QVBoxLayout(dlg)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("SMILES:"))
        dlg._smiles_text = QLineEdit()
        dlg._smiles_text.setReadOnly(True)
        dlg._copy_btn = QPushButton("Copy SMILES")
        dlg._copy_btn.clicked.connect(self.copy_smiles)
        top_bar.addWidget(dlg._smiles_text)
        top_bar.addWidget(dlg._copy_btn)
        root.addLayout(top_bar)

        nav_bar = QHBoxLayout()
        dlg._prev_btn = QPushButton("Previous")
        dlg._next_btn = QPushButton("Next")
        dlg._export_btn = QPushButton("Export Image")
        dlg._ai_btn = QPushButton("AI: Interpret candidate")
        dlg._ai_btn.setToolTip("Narrative for this hit vs experimental peaks (P0)")
        dlg._ai_btn.clicked.connect(self._ai_compound_from_dialog)
        dlg._close_btn = QPushButton("Close")
        nav_bar.addWidget(dlg._prev_btn)
        nav_bar.addWidget(dlg._next_btn)
        nav_bar.addStretch()
        nav_bar.addWidget(dlg._ai_btn)
        nav_bar.addWidget(dlg._export_btn)
        nav_bar.addWidget(dlg._close_btn)
        root.addLayout(nav_bar)

        content = QHBoxLayout()
        dlg._structure_label = QLabel("No structure")
        dlg._structure_label.setMinimumSize(440, 340)
        dlg._structure_label.setAlignment(Qt.AlignCenter)
        content.addWidget(dlg._structure_label, 1)

        spectrum_side = QVBoxLayout()
        dlg._spectrum_canvas = FigureCanvas(Figure(figsize=(7, 5)))
        dlg._spectrum_toolbar = NavigationToolbar(dlg._spectrum_canvas, dlg)
        spectrum_side.addWidget(dlg._spectrum_toolbar)
        spectrum_side.addWidget(dlg._spectrum_canvas)
        content.addLayout(spectrum_side, 2)
        root.addLayout(content)

        dlg._prev_btn.clicked.connect(self._show_previous_compound)
        dlg._next_btn.clicked.connect(self._show_next_compound)
        dlg._export_btn.clicked.connect(self._export_selected_compound_image)
        dlg._close_btn.clicked.connect(dlg.close)

        self.compound_detail_dialog = dlg
        return dlg

    def _get_compound_row_data(self, current_row):
        """Extract row data robustly from result table/dataframe."""
        smiles = self.results_table.item(current_row, 1).text()
        shifts_text = self.results_table.item(current_row, 3).text()
        carbon_types = []
        shifts = []
        if hasattr(self, 'result') and self.result is not None:
            try:
                result_row = self.result.iloc[current_row]
                if 'Ctype' in result_row:
                    carbon_types_data = result_row['Ctype']
                    if isinstance(carbon_types_data, str):
                        carbon_types = [x.strip() for x in carbon_types_data.split(',') if x.strip()]
                    elif isinstance(carbon_types_data, (list, tuple)):
                        carbon_types = list(carbon_types_data)
                    elif isinstance(carbon_types_data, np.ndarray):
                        carbon_types = carbon_types_data.tolist()
                    carbon_types = [str(x).strip() for x in carbon_types if str(x).strip() in ['q', 't', 'd', 's']]
                if 'Vir_shifts' in result_row:
                    shifts_data = result_row['Vir_shifts']
                    if isinstance(shifts_data, np.ndarray):
                        shifts = shifts_data.tolist()
                    elif isinstance(shifts_data, (list, tuple)):
                        shifts = list(shifts_data)
                    elif isinstance(shifts_data, (int, float)):
                        shifts = [float(shifts_data)]
            except Exception as e:
                print(f"Failed to parse row data: {str(e)}")
        if not shifts and shifts_text and shifts_text.strip():
            shifts = self.parse_shifts_from_string(shifts_text)
        return smiles, shifts, carbon_types

    def _render_compound_spectrum(self, ax, shifts, carbon_types):
        """Render merged comparison spectrum for popup and export."""
        experimental_data = self.get_experimental_data()
        if shifts and experimental_data:
            ax.set_xlim(220, 0)
            avg_intensity = sum([d['intensity'] for d in experimental_data]) / len(experimental_data)
            self.plot_experimental_data_merged(ax, experimental_data)
            self.plot_database_data_merged(ax, shifts, carbon_types, avg_intensity)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
            exp_src = (
                "Manual peak input"
                if (
                    hasattr(self, "manual_peak_input")
                    and self.manual_peak_input.toPlainText().strip()
                )
                else os.path.basename(self._nmr_csv_path())
            )
            ax.set_title(
                f"Chemical Shifts Comparison: ↑ Experimental ({exp_src}) | ↓ Virtual Database Data",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlabel('Chemical Shift (ppm)', fontsize=10)
            ax.set_ylabel('Intensity', fontsize=10)
            ax.grid(True, alpha=0.3)
        elif shifts:
            ax.set_xlim(220, 0)
            self.plot_database_data_merged(ax, shifts, carbon_types, 1.0)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
            ax.set_title('Virtual Database Data (Predicted)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Chemical Shift (ppm)', fontsize=10)
            ax.set_ylabel('Intensity', fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                'No chemical shift data available',
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=12,
                color='gray',
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Chemical Shifts Comparison')
            ax.set_xlabel('No Data')
            ax.set_ylabel('No Data')

    def _update_compound_detail_dialog(self, current_row):
        """Update popup content for selected row."""
        dlg = self._ensure_compound_detail_dialog()
        if current_row < 0 or current_row >= self.results_table.rowCount():
            return
        smiles, shifts, carbon_types = self._get_compound_row_data(current_row)
        if not smiles or smiles.strip() == '':
            QMessageBox.warning(self, "Warning", "No SMILES data available")
            return
        self._compound_detail_current_row = current_row
        dlg._smiles_text.setText(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            QMessageBox.warning(self, "Warning", f"Invalid SMILES: {smiles}")
            return
        draw_opts = Draw.rdMolDraw2D.MolDrawOptions()
        try:
            draw_opts.useBWAtomPalette()
        except Exception:
            pass
        img = Draw.MolToImage(mol, size=(420, 320), options=draw_opts)
        dlg._structure_label.setPixmap(self.pil_image_to_pixmap(img))

        dlg._spectrum_canvas.figure.clear()
        ax = dlg._spectrum_canvas.figure.add_subplot(111)
        self._render_compound_spectrum(ax, shifts, carbon_types)
        dlg._spectrum_canvas.figure.tight_layout()
        dlg._spectrum_canvas.draw()

        has_prev = current_row > 0
        has_next = current_row < self.results_table.rowCount() - 1
        dlg._prev_btn.setEnabled(has_prev)
        dlg._next_btn.setEnabled(has_next)
        dlg.setWindowTitle(f"Compound Details ({current_row + 1}/{self.results_table.rowCount()})")

    def _show_previous_compound(self):
        new_row = self._compound_detail_current_row - 1
        if new_row < 0:
            return
        self.results_table.selectRow(new_row)
        self._update_compound_detail_dialog(new_row)

    def _show_next_compound(self):
        new_row = self._compound_detail_current_row + 1
        if new_row >= self.results_table.rowCount():
            return
        self.results_table.selectRow(new_row)
        self._update_compound_detail_dialog(new_row)

    def _export_selected_compound_image(self):
        """Export current popup compound image and spectrum as PNG."""
        if self._compound_detail_current_row < 0:
            QMessageBox.warning(self, "Export", "No compound selected.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Compound Image",
            f"compound_{self._compound_detail_current_row + 1}.png",
            "PNG files (*.png)",
        )
        if not file_path:
            return
        try:
            row = self._compound_detail_current_row
            smiles, shifts, carbon_types = self._get_compound_row_data(row)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")

            export_fig = Figure(figsize=(14, 6), dpi=200)
            ax_img = export_fig.add_subplot(1, 2, 1)
            ax_sp = export_fig.add_subplot(1, 2, 2)
            draw_opts = Draw.rdMolDraw2D.MolDrawOptions()
            try:
                draw_opts.useBWAtomPalette()
            except Exception:
                pass
            img = Draw.MolToImage(mol, size=(800, 600), options=draw_opts)
            ax_img.imshow(np.asarray(img))
            ax_img.set_title(f"Structure (DB row {row + 1})", fontsize=12, fontweight='bold')
            ax_img.axis("off")
            self._render_compound_spectrum(ax_sp, shifts, carbon_types)
            export_fig.tight_layout()
            export_fig.savefig(file_path, dpi=300, bbox_inches="tight")
            cid_it = self.results_table.item(row, 0)
            score_it = self.results_table.item(row, 2)
            extra_detail = {
                "results_table_row_1based": row + 1,
                "compound_id": cid_it.text() if cid_it is not None else "",
                "smiles": smiles,
                "score": score_it.text() if score_it is not None else "",
                "predicted_shifts_ppm": self._json_safe_export(list(shifts) if shifts else []),
                "predicted_carbon_types": self._json_safe_export(list(carbon_types) if carbon_types else []),
            }
            param_path = self._write_analysis_params_sidecar(
                file_path, "compound_detail_image", extra=extra_detail
            )
            self.statusBar().showMessage(f"Compound image exported: {file_path}")
            msg = f"Compound image exported:\n{file_path}"
            if param_path:
                msg += f"\n\nAnalysis parameters:\n{param_path}"
            else:
                msg += "\n\n(Could not save analysis parameters JSON; see console.)"
            QMessageBox.information(self, "Export", msg)
        except Exception as e:
            QMessageBox.critical(self, "Export", f"Failed to export image: {str(e)}")

    def show_selected_compound(self, row=None, column=None):
        """Open compound details in popup when a result row is clicked."""
        try:
            current_row = self.results_table.currentRow() if row is None else int(row)
            if current_row < 0:
                return
            dlg = self._ensure_compound_detail_dialog()
            self._update_compound_detail_dialog(current_row)
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to show compound details: {str(e)}")
            self.statusBar().showMessage("Failed to show compound details")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def get_carbon_type_label(self, ctype):
        """Convert carbon type code to readable label"""
        ctype_map = {
            'q': 'CH3',
            't': 'CH2', 
            'd': 'CH',
            's': 'C'
        }
        return ctype_map.get(ctype, ctype)
    
    def _fill_manual_peak_example(self):
        """Load demo peaks into manual input (only when user clicks Use Example Case)."""
        if hasattr(self, "manual_peak_input"):
            self.manual_peak_input.setPlainText(_MANUAL_PEAK_EXAMPLE_TEXT)
            self.statusBar().showMessage("Manual peak input: example case loaded")

    def parse_manual_peak_input_strict(self, text):
        """Parse manual peak list. Returns (shifts, ctypes) or None if empty. Raises ValueError if invalid."""
        if text is None:
            return None
        text = text.strip()
        if not text:
            return None
        shifts = []
        ctypes = []
        valid = {"q", "t", "d", "s"}
        for line_no, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "," not in line:
                raise ValueError(f"Line {line_no}: expected 'ppm, type' (comma missing).")
            left, right = line.split(",", 1)
            try:
                ppm = float(left.strip())
            except ValueError:
                raise ValueError(f"Line {line_no}: invalid ppm '{left.strip()}'.")
            ct = right.strip().lower()
            if len(ct) != 1 or ct not in valid:
                raise ValueError(
                    f"Line {line_no}: carbon type must be q, t, d, or s (got '{right.strip()}')."
                )
            shifts.append(ppm)
            ctypes.append(ct)
        if not shifts:
            raise ValueError("No peaks parsed. Add lines like: 13.31, q")
        return shifts, ctypes
    
    def get_experimental_data(self):
        """Experimental peaks: manual input if set, else NMR-1D.csv."""
        try:
            if hasattr(self, "manual_peak_input"):
                mt = self.manual_peak_input.toPlainText().strip()
                if mt:
                    try:
                        parsed = self.parse_manual_peak_input_strict(mt)
                    except ValueError:
                        return None
                    if parsed is not None:
                        shifts, ctypes = parsed
                        return [
                            {"ppm": p, "type": ct, "intensity": 1.0}
                            for p, ct in zip(shifts, ctypes)
                        ]
            csv_file = self._nmr_csv_path()
            if not os.path.exists(csv_file):
                return None
            
            import pandas as pd
            df = pd.read_csv(csv_file, header=None)
            
            if df.empty or len(df.columns) < 3:
                return None
            
            # Skip the first row and get data
            if len(df) > 1:
                df = df.iloc[1:]
            
            experimental_data = []
            for _, row in df.iterrows():
                try:
                    ppm = float(row.iloc[0])
                    ctype = str(row.iloc[1]).strip()
                    intensity = float(row.iloc[2])
                    experimental_data.append({
                        'ppm': ppm,
                        'type': ctype,
                        'intensity': intensity
                    })
                except (ValueError, IndexError):
                    continue
            
            return experimental_data if experimental_data else None
            
        except Exception as e:
            print(f"Error reading experimental data: {str(e)}")
            return None
    
    def plot_experimental_data_merged(self, ax, experimental_data):
        """Plot experimental data on merged plot (positive values, upward)"""
        bar_width = 2.0
        
        for data in experimental_data:
            ppm = data['ppm']
            ctype = data['type']
            intensity = data['intensity']
            
            # Determine color based on carbon type
            color = self.get_carbon_color(ctype)
            
            # Create bar pointing upward (positive intensity)
            ax.bar(ppm, intensity, width=bar_width, color=color, alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
            
            # Add carbon type annotation
            ctype_label = self.get_carbon_type_label(ctype)
            ax.text(ppm, intensity/2, ctype_label, ha='center', va='center', 
                   fontsize=8, color='white', weight='bold')
    
    def plot_database_data_merged(self, ax, shifts, carbon_types, avg_intensity=1.0):
        """Plot database data on merged plot (negative values, downward)"""
        bar_width = 2.0
        
        for i, shift in enumerate(shifts):
            # Determine color based on carbon type
            if i < len(carbon_types):
                ctype = carbon_types[i]
                color = self.get_carbon_color(ctype)
                ctype_label = self.get_carbon_type_label(ctype)
            else:
                color = 'skyblue'
                ctype_label = 'Unknown'
            
            # Create bar pointing downward (negative intensity) using average intensity
            ax.bar(shift, -avg_intensity, width=bar_width, color=color, alpha=0.7, 
                   edgecolor='gray', linewidth=0.5)
            
            # Add carbon type annotation
            ax.text(shift, -avg_intensity/2, ctype_label, ha='center', va='center', 
                   fontsize=8, color='white', weight='bold')
    
    def plot_experimental_data(self, ax, experimental_data):
        """Plot experimental data on the top subplot"""
        bar_width = 2.0
        
        for data in experimental_data:
            ppm = data['ppm']
            ctype = data['type']
            intensity = data['intensity']
            
            # Determine color based on carbon type
            color = self.get_carbon_color(ctype)
            
            # Create bar pointing upward (positive intensity)
            ax.bar(ppm, intensity, width=bar_width, color=color, alpha=0.7)
            
            # Add carbon type annotation
            ctype_label = self.get_carbon_type_label(ctype)
            ax.text(ppm, intensity/2, ctype_label, ha='center', va='center', 
                   fontsize=8, color='white', weight='bold')
        
        # Set y-axis limits
        if experimental_data:
            max_intensity = max([d['intensity'] for d in experimental_data])
            ax.set_ylim(0, max_intensity * 1.2)
        else:
            ax.set_ylim(0, 1)
    
    def plot_database_data(self, ax, shifts, carbon_types):
        """Plot database data on the bottom subplot"""
        bar_width = 2.0
        
        for i, shift in enumerate(shifts):
            # Determine color based on carbon type
            if i < len(carbon_types):
                ctype = carbon_types[i]
                color = self.get_carbon_color(ctype)
                ctype_label = self.get_carbon_type_label(ctype)
            else:
                color = 'skyblue'
                ctype_label = 'Unknown'
            
            # Create bar pointing downward (negative intensity)
            ax.bar(shift, -1, width=bar_width, color=color, alpha=0.7)
            
            # Add carbon type annotation
            ax.text(shift, -0.5, ctype_label, ha='center', va='center', 
                   fontsize=8, color='white', weight='bold')
        
        # Set y-axis limits
        ax.set_ylim(-1.5, 0)
    
    def get_carbon_color(self, ctype):
        """Get color for carbon type"""
        color_map = {
            'q': 'red',      # CH3
            't': 'green',    # CH2
            'd': 'blue',     # CH
            's': 'orange'    # C
        }
        return color_map.get(ctype, 'skyblue')
    
    def parse_shifts_from_string(self, shifts_text):
        """Parse chemical shifts from various string formats"""
        try:
            # Remove common brackets and parentheses
            cleaned_text = shifts_text.strip()
            if cleaned_text.startswith('[') and cleaned_text.endswith(']'):
                cleaned_text = cleaned_text[1:-1]
            elif cleaned_text.startswith('(') and cleaned_text.endswith(')'):
                cleaned_text = cleaned_text[1:-1]
            
            # Split by common separators
            separators = [',', ';', '|', '\t', '\n']
            for sep in separators:
                if sep in cleaned_text:
                    parts = cleaned_text.split(sep)
                    shifts = []
                    for part in parts:
                        part = part.strip()
                        if part:
                            try:
                                # Handle scientific notation and regular numbers
                                if 'e' in part.lower() or 'E' in part:
                                    shifts.append(float(part))
                                else:
                                    shifts.append(float(part))
                            except ValueError:
                                continue
                    return shifts
            
            # If no separators found, try to parse as single number
            try:
                return [float(cleaned_text)]
            except ValueError:
                return []
                
        except Exception as e:
            print(f"Failed to parse shifts from string '{shifts_text}': {str(e)}")
            return []
    
    def copy_smiles(self):
        """Copy SMILES to clipboard"""
        try:
            smiles = ""
            dlg = getattr(self, "compound_detail_dialog", None)
            if dlg is not None and hasattr(dlg, "_smiles_text"):
                smiles = dlg._smiles_text.text()
            if smiles:
                QApplication.clipboard().setText(smiles)
                self.statusBar().showMessage("SMILES copied to clipboard")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to copy SMILES: {str(e)}")
            self.statusBar().showMessage("Failed to copy SMILES")
    
    def pil_image_to_pixmap(self, pil_image):
        """Convert PIL image to QPixmap"""
        try:
            # Convert PIL image to bytes
            data = pil_image.tobytes('raw', 'RGB')
            qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGB888)
            return QPixmap.fromImage(qimage)
        except Exception as e:
            print(f"Failed to convert image: {str(e)}")
            return QPixmap()

    def _mask_structure_transform_params(self):
        """Return structure image transform params: (rotate_deg, flip_h, flip_v)."""
        rotate_widget = getattr(self, "frag_mask_rotate_spin", None)
        flip_h_widget = getattr(self, "frag_mask_flip_h_check", None)
        flip_v_widget = getattr(self, "frag_mask_flip_v_check", None)
        rotate_deg = float(rotate_widget.value()) if rotate_widget is not None else 0.0
        flip_h = bool(flip_h_widget.isChecked()) if flip_h_widget is not None else False
        flip_v = bool(flip_v_widget.isChecked()) if flip_v_widget is not None else False
        return rotate_deg, flip_h, flip_v

    def _apply_mask_structure_transform(self, pil_image):
        """Apply rotation/flip settings to structure image for preview/export."""
        if pil_image is None:
            return pil_image
        from PIL import Image

        rotate_deg, flip_h, flip_v = self._mask_structure_transform_params()
        out = pil_image.copy()
        if flip_h:
            out = out.transpose(method=Image.FLIP_LEFT_RIGHT)
        if flip_v:
            out = out.transpose(method=Image.FLIP_TOP_BOTTOM)
        if abs(rotate_deg) > 1e-9:
            bg = (255, 255, 255, 255) if out.mode == "RGBA" else ((255, 255, 255) if out.mode == "RGB" else 255)
            out = out.rotate(-rotate_deg, expand=True, fillcolor=bg)
        if out.mode != "RGB":
            out = out.convert("RGB")
        return out
            

    
    def update_results_table(self, result):
        """Update results table"""
        try:
            if not isinstance(result, pd.DataFrame):
                print("Result is not a DataFrame")
                return
            
            # Clear table
            self.results_table.setRowCount(0)
            
            # Add result data
            for i, (idx, row) in enumerate(result.iterrows()):
                self.results_table.insertRow(i)
                
                # Create and set table items
                db_index_item = QTableWidgetItem(str(row.get('DBindex', '')))
                smiles_item = QTableWidgetItem(str(row.get('smiles', '')))
                score_item = QTableWidgetItem(f"{row.get('score', 0):.3f}")
                
                # Format chemical shifts for better display
                shifts_data = row.get('Vir_shifts', '')
                if isinstance(shifts_data, (list, tuple)):
                    shifts_text = ', '.join([f"{x:.2f}" for x in shifts_data])
                elif isinstance(shifts_data, (int, float)):
                    shifts_text = f"{shifts_data:.2f}"
                else:
                    shifts_text = str(shifts_data)
                shifts_item = QTableWidgetItem(shifts_text)
                
                # Format carbon types for better display
                ctype_data = row.get('Ctype', '')
                if isinstance(ctype_data, (list, tuple)):
                    ctype_text = ', '.join([self.get_carbon_type_label(x) for x in ctype_data])
                elif isinstance(ctype_data, str):
                    ctype_text = ', '.join([self.get_carbon_type_label(x) for x in ctype_data])
                else:
                    ctype_text = str(ctype_data)
                ctype_item = QTableWidgetItem(ctype_text)
                
                # Set items alignment
                for item in [db_index_item, smiles_item, score_item, shifts_item, ctype_item]:
                    item.setTextAlignment(Qt.AlignCenter)
                
                # Add items to table
                self.results_table.setItem(i, 0, db_index_item)
                self.results_table.setItem(i, 1, smiles_item)
                self.results_table.setItem(i, 2, score_item)
                self.results_table.setItem(i, 3, shifts_item)
                self.results_table.setItem(i, 4, ctype_item)
            
            # Adjust column widths
            self.results_table.resizeColumnsToContents()
            
            # Set column headers to be more descriptive
            self.results_table.setHorizontalHeaderLabels([
                "Compound ID", "SMILES", "Score", "Chemical Shifts (ppm)", "Carbon Types"
            ])
            
            print(f"Results table updated with {len(result)} compounds")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update results table: {str(e)}")
            self.statusBar().showMessage("Failed to update results table")
            print(f"Error updating results table: {str(e)}")
            import traceback
            traceback.print_exc()
    

    
    def create_database(self):
        """Create database"""
        try:
            if self._is_thread_running("db_thread"):
                QMessageBox.information(
                    self,
                    "Please wait",
                    "Database creation is still running. Please wait for it to finish.",
                )
                return
            if not self.smiles_input.text():
                QMessageBox.warning(self, "Warning", "Please select a SMILES file")
                return
            
            # Check if file exists
            if not os.path.exists(self.smiles_input.text()):
                QMessageBox.critical(self, "Error", "Selected file does not exist")
                return
            
            # Disable create button and show progress
            self.create_db_btn.setEnabled(False)
            self.create_db_btn.setText("Creating...")
            self.db_progress_bar.setVisible(True)
            self.db_progress_bar.setValue(0)
            self.db_status_text.clear()
            
            # Update status
            self.statusBar().showMessage("Creating database, please wait...")
            self.db_status_text.append("Starting database creation...")
            self.db_status_text.append(f"Reading SMILES file: {self.smiles_input.text()}")
            
            # Read SMILES file
            try:
                df = pd.read_csv(self.smiles_input.text())
                if df.empty:
                    raise ValueError("CSV file is empty")
                
                smiles_list = df.iloc[:, 0].tolist()
                # Filter out empty or invalid SMILES
                smiles_list = [s for s in smiles_list if s and str(s).strip() and str(s).strip() != 'nan']
                
                if not smiles_list:
                    raise ValueError("No valid SMILES strings found in the file")
                
                self.db_status_text.append(f"Loaded {len(smiles_list)} valid SMILES strings")
                self.db_progress_bar.setValue(10)
                
            except Exception as e:
                raise ValueError(f"Failed to read SMILES file: {str(e)}")
            
            # Use user-specified output path or default
            user_output_path = self.output_path_input.text().strip()
            if user_output_path:
                output_path = user_output_path
                self.db_status_text.append(f"Using user-specified output path: {output_path}")
            else:
                output_path = self._gui_output_path("generated_virDB.npz")  # Default file path
                self.db_status_text.append(f"Using default output path: {output_path}")
            
            # Check if output path is writable
            try:
                # Get project root directory
                project_root = os.path.dirname(os.path.abspath(__file__))
                
                # If output path is relative, make it absolute
                if not os.path.isabs(output_path):
                    full_output_path = os.path.join(project_root, output_path)
                else:
                    full_output_path = output_path
                
                # Extract directory from file path and create it if needed
                output_dir = os.path.dirname(full_output_path)
                if output_dir:  # If there's a directory part
                    os.makedirs(output_dir, exist_ok=True)
                
                # Test if we can write to the directory
                test_file = os.path.join(output_dir, "test_write.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                
                self.db_status_text.append(f"Output file path: {full_output_path}")
            except Exception as e:
                raise ValueError(f"Output path is not writable: {str(e)}")
            
            # Create database in background thread
            self.db_thread = DatabaseCreationThread(smiles_list, full_output_path)
            self.db_thread.progress_update.connect(self.update_db_progress)
            self.db_thread.status_update.connect(self.update_db_status)
            self.db_thread.finished.connect(self.database_creation_finished)
            self.db_thread.error.connect(self.database_creation_error)
            self.db_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Database creation failed: {str(e)}")
            self.statusBar().showMessage("Database creation failed")
            self.reset_database_ui()
    
    def update_db_progress(self, value):
        """Update database creation progress"""
        self.db_progress_bar.setValue(value)
    
    def update_db_status(self, message):
        """Update database creation status"""
        self.db_status_text.append(message)
        # Auto-scroll to bottom
        self.db_status_text.verticalScrollBar().setValue(
            self.db_status_text.verticalScrollBar().maximum()
        )
    
    def database_creation_finished(self):
        """Database creation completed successfully"""
        self.db_thread = None
        self.db_progress_bar.setValue(100)
        self.db_status_text.append("Database creation completed successfully!")
        
        # Use the actual output path from the input field
        user_output_path = self.output_path_input.text().strip()
        if user_output_path:
            output_path = user_output_path
        else:
            output_path = self._gui_output_path("generated_virDB.npz")
        
        # Make sure it's absolute
        if not os.path.isabs(output_path):
            project_root = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(project_root, output_path)
        
        # Get the directory where the file was saved
        output_dir = os.path.dirname(output_path)
        if not output_dir:
            output_dir = "."  # Current directory
        
        self.db_status_text.append(f"Database file saved as: {os.path.abspath(output_path)}")
        self.db_status_text.append(f"File location: {os.path.abspath(output_dir)}")
        self.db_status_text.append("Generated files:")
        self.db_status_text.append("  - generated_virDB.npz (main database)")
        self.db_status_text.append("  - virdb.smi (SMILES file)")
        self.db_status_text.append("  - descriptors.csv (molecular descriptors)")
        
        self.statusBar().showMessage("Database creation completed")
        QMessageBox.information(self, "Success", f"Database creation completed! File saved as {output_path}")
        
        self.reset_database_ui()
    
    def database_creation_error(self, error_message):
        """Database creation failed"""
        self.db_thread = None
        self.db_status_text.append(f"ERROR: {error_message}")
        QMessageBox.critical(self, "Error", f"Database creation failed: {error_message}")
        self.statusBar().showMessage("Database creation failed")
        self.reset_database_ui()
    
    def check_model_status(self):
        """Check if the required model file exists"""
        model_path = os.path.join(os.getcwd(), 'VirMolAnalyte', 'VirDBcreator', 'NMRprediction', 'model', 'nmr_model.pt')
        if os.path.exists(model_path):
            self.model_status_label.setText("Model Status: ✓ Available")
            self.model_status_label.setStyleSheet("font-weight: bold; color: #28a745;")
        else:
            self.model_status_label.setText("Model Status: ✗ Missing (nmr_model.pt)")
            self.model_status_label.setStyleSheet("font-weight: bold; color: #dc3545;")
            self.create_db_btn.setEnabled(False)
            self.create_db_btn.setToolTip("Cannot create database: Model file missing")
    
    def reset_database_ui(self):
        """Reset database creation UI to initial state"""
        self.create_db_btn.setEnabled(True)
        self.create_db_btn.setText("Create Database")
        self.create_db_btn.setToolTip("")
        self.db_progress_bar.setVisible(False)

    # ----- P0 AI assistant (virmol_ai) -----

    def _ai_make_compact_button(self, text, tooltip=None, callback=None):
        btn = QPushButton(text)
        btn.setObjectName("AiCompactButton")
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if tooltip:
            btn.setToolTip(tooltip)
        if callback:
            btn.clicked.connect(callback)
        return btn

    @staticmethod
    def _ai_mark_llm_button(btn: QPushButton):
        """Apply highlighted style for buttons that call cloud LLM."""
        btn.setObjectName("AiLlmButton")

    def _ai_fill_button_grid(self, grid, specs, cols=3, row_start=0):
        """Place (label, callback[, tooltip[, colspan]]) buttons on a grid."""
        row, col = row_start, 0
        for spec in specs:
            text, cb = spec[0], spec[1]
            tip = spec[2] if len(spec) > 2 else None
            span = min(int(spec[3]) if len(spec) > 3 else 1, cols)
            is_llm = bool(spec[4]) if len(spec) > 4 else False
            btn = self._ai_make_compact_button(text, tip, cb)
            if is_llm:
                self._ai_mark_llm_button(btn)
            grid.addWidget(btn, row, col, 1, span)
            col += span
            if col >= cols:
                col = 0
                row += 1
        return row + (1 if col > 0 else 0)

    # ----- Default SOP parameters (used when the user just pastes peaks) -----
    _SOP_DEFAULTS = {
        "sop_id": "full_pipeline",
        "database": "plant",
        "evaluator": "FPAACS",
        "use_cnf": True,
        "use_ctnf": True,
        "use_mw": False,
        "masking_top_n": 5,
        "do_masking": True,
        "do_fusion": True,
        "do_export": True,
        "sample_note": "",
    }

    def create_ai_assistant_tab(self):
        """AI Assistant — a single chat box that drives the whole pipeline.

        - Paste a ¹³C peak list (or load a file) → the assistant runs the full
          screening + masking + fusion + export pipeline using sensible defaults.
        - Type a question → the cloud LLM answers using the current GUI state.
        - All progress, results, and answers stream into one conversation view.
        """
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # ---------- Top bar: status + small action buttons ----------
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        self.ai_status_label = QLabel("")
        self.ai_status_label.setStyleSheet("font-size: 12px;")
        self._refresh_ai_status_label()
        top_row.addWidget(self.ai_status_label, 1)

        self.ai_provider_combo = QComboBox()
        self.ai_provider_combo.setMaximumHeight(28)
        self._fill_provider_combobox(self.ai_provider_combo)
        self.ai_provider_combo.currentIndexChanged.connect(self._ai_on_provider_combo_changed)
        top_row.addWidget(self.ai_provider_combo)
        if not load_provider_id():
            self._ai_apply_provider_preset(DEFAULT_PROVIDER_ID, notify=False)

        prefs_btn = self._ai_make_compact_button("API settings", None, self.show_preferences)
        prefs_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        top_row.addWidget(prefs_btn)

        adv_btn = self._ai_make_compact_button(
            "Advanced…",
            "Override analysis defaults (database, filters, masking, …)",
            self._ai_show_advanced_dialog,
        )
        adv_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        top_row.addWidget(adv_btn)
        root.addLayout(top_row)

        # Defaults summary line (auto-updated)
        self.ai_defaults_label = QLabel("")
        self.ai_defaults_label.setStyleSheet("color: #6c757d; font-size: 11px;")
        self.ai_defaults_label.setWordWrap(True)
        self._ai_refresh_defaults_label()
        root.addWidget(self.ai_defaults_label)

        # ---------- Quick actions (collapsed by default) ----------
        # Original LLM-driven shortcuts are kept here for power users; click
        # the header to expand.
        quick_box = CollapsibleBox(
            "Quick actions — Top hits · Methods · Fragments (click to expand)",
            start_expanded=False,
        )
        quick_inner = QWidget()
        quick_v = QVBoxLayout(quick_inner)
        quick_v.setContentsMargins(0, 4, 0, 4)
        quick_v.setSpacing(6)

        llm_grp = QGroupBox("LLM actions  (API key required)")
        llm_grp.setObjectName("AiPanelGroup")
        llm_grid = QGridLayout(llm_grp)
        llm_grid.setContentsMargins(8, 6, 8, 8)
        llm_grid.setHorizontalSpacing(6)
        llm_grid.setVerticalSpacing(6)
        self._ai_fill_button_grid(
            llm_grid,
            [
                ("Top 5", lambda: self._ai_run_task("topn", top_n=5), "Interpret top 5 screening hits", 1, True),
                ("Top 10", lambda: self._ai_run_task("topn", top_n=10), "Interpret top 10 screening hits", 1, True),
                ("Top 20", lambda: self._ai_run_task("topn", top_n=20), "Interpret top 20 screening hits", 1, True),
            ],
            cols=3,
        )
        quick_v.addWidget(llm_grp)

        frag_grp = QGroupBox("Fragment  (requires masking + fragment extraction)")
        frag_grp.setObjectName("AiPanelGroup")
        frag_grid = QGridLayout(frag_grp)
        frag_grid.setContentsMargins(8, 6, 8, 8)
        frag_grid.setHorizontalSpacing(6)
        frag_grid.setVerticalSpacing(6)
        self._ai_fill_button_grid(
            frag_grid,
            [
                ("Selected → infer structure", self._ai_infer_structure_from_selected,
                 "Send the selected candidate's masking data (SMILES, per-carbon score, DEPT type, "
                 "virtual shifts) + experimental peaks to the LLM to infer plausible structures", 1, True),
                ("Selected → assign signals", self._ai_assign_signals_from_selected,
                 "Use Monte-Carlo carbon scores to extract high-score fragments and assign "
                 "experimental 13C/DEPT peaks to the selected candidate's carbons "
                 "with a structured LLM report", 1, True),
                ("Fragment evidence review", self._ai_fragment_evidence_review,
                 "Review extracted fragments: evidence ranking, redundancy/complementarity, "
                 "missing motifs from unassigned peaks, and fusion-ready subset", 1, True),
                ("Fusion → Top-5", self._ai_fusion_infer_top5,
                 "LLM assembles fusion fragments + ¹³C peaks into 5 proposed full structures (de novo)", 1, True),
            ],
            cols=2,
        )
        quick_v.addWidget(frag_grp)

        offline_grp = QGroupBox("Offline tools  (no API key)")
        offline_grp.setObjectName("AiPanelGroup")
        offline_grid = QGridLayout(offline_grp)
        offline_grid.setContentsMargins(8, 6, 8, 8)
        offline_grid.setHorizontalSpacing(6)
        offline_grid.setVerticalSpacing(6)
        self._ai_fill_button_grid(
            offline_grid,
            [
                ("Screening diag.", self._ai_rule_diagnosis,
                 "Rule-based filter/peak/hit check"),
                ("Copy Methods", self.copy_methodology_to_clipboard,
                 "Rule-based Methods paragraph"),
                ("Results draft", self._ai_copy_results_draft,
                 "Rule-based Results draft"),
                ("Fragment JSON", self._ai_copy_fragment_summary,
                 "Copy fragment snapshot (no API)"),
            ],
            cols=4,
        )
        quick_v.addWidget(offline_grp)

        opts_row = QHBoxLayout()
        opts_row.setSpacing(8)
        self.ai_include_fragment_check = QCheckBox("Include fragment context (Top N)")
        self.ai_include_fragment_check.setChecked(False)
        self.ai_include_fragment_check.setToolTip(
            "Attach Phase A fragment data when calling Top 5 / Top 10 / Top 20"
        )
        self.ai_include_fragment_check.setStyleSheet("font-size: 12px;")
        opts_row.addWidget(self.ai_include_fragment_check)
        self.ai_sample_note = QLineEdit()
        self.ai_sample_note.setPlaceholderText("Sample note (optional, used in prompts)")
        self.ai_sample_note.setMaximumHeight(28)
        opts_row.addWidget(self.ai_sample_note, 1)
        quick_v.addLayout(opts_row)

        quick_box.addWidget(quick_inner)
        root.addWidget(quick_box)

        # ---------- Conversation view + right jump-to column ----------
        chat_row = QHBoxLayout()
        chat_row.setSpacing(8)

        # Left/main: chat view. Use QTextEdit so we can embed structure images.
        self.ai_chat_view = QTextEdit()
        self.ai_chat_view.setReadOnly(True)
        self.ai_chat_view.setPlaceholderText(
            "Paste a ¹³C peak list here, or type a question.\n\n"
            "Examples that work:\n"
            "  172.5, s\n  21.3, q\n\n"
            "or paper-style:\n  172.5 (s), 145.2 (d), 21.3 (q)"
        )
        self.ai_chat_view.setMinimumHeight(380)
        self.ai_chat_view.setStyleSheet(
            "QTextEdit { font-family: 'Consolas', 'Menlo', 'Courier New', monospace; "
            "font-size: 15px; line-height: 1.45; "
            "background: #ffffff; color: #212529; "
            "padding: 8px; }"
        )
        chat_font = self.ai_chat_view.font()
        chat_font.setPointSize(11)
        self.ai_chat_view.setFont(chat_font)
        chat_row.addWidget(self.ai_chat_view, 1)

        # Right: vertical column of quick-jump buttons
        nav_holder = QFrame()
        nav_holder.setFrameShape(QFrame.NoFrame)
        nav_holder.setMaximumWidth(170)
        nav_v = QVBoxLayout(nav_holder)
        nav_v.setContentsMargins(0, 0, 0, 0)
        nav_v.setSpacing(8)

        nav_lbl = QLabel("Jump to")
        nav_lbl.setStyleSheet(
            "font-size: 12px; color: #212529; font-weight: bold;"
            " padding: 2px 0 4px 4px;"
        )
        nav_v.addWidget(nav_lbl)

        nav_button_specs = [
            ("Top-5 structures",
             self._ai_show_top5_structures,
             "Render the top 5 hit molecules and show them here in the chat"),
            ("Hits table",
             lambda: self._ai_goto_tab("Molecular Analysis"),
             "Show the screening hit list (Molecular Analysis tab)"),
            ("Fragments",
             lambda: self._ai_goto_tab("Fragment Analysis"),
             "Open the Fragment / masking analysis tab"),
            ("Attribution view",
             self._ai_goto_attribution_view,
             "Open Fragment Analysis and switch to the Compound attribution view"),
            ("Database",
             lambda: self._ai_goto_tab("Database Management"),
             "Open the Database management tab"),
            ("Other tools",
             lambda: self._ai_goto_tab("Other Tool"),
             "Misc utilities"),
            ("Output folder",
             self._ai_open_output_folder,
             "Open the GUI output folder in the file manager"),
        ]
        nav_style = (
            "QPushButton { text-align: left; padding: 8px 12px; "
            "background: #3d5a66; color: #ffffff; "
            "border: 1px solid #2f4651; border-radius: 6px; "
            "font-size: 13px; font-weight: 600; } "
            "QPushButton:hover { background: #4d6e7d; } "
            "QPushButton:pressed { background: #2f4651; } "
            "QPushButton:disabled { background: #adb5bd; color: #f8f9fa; }"
        )
        for label, slot, tip in nav_button_specs:
            btn = QPushButton(label)
            btn.setMinimumHeight(36)
            btn.setStyleSheet(nav_style)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            nav_v.addWidget(btn)
        nav_v.addStretch()
        chat_row.addWidget(nav_holder)

        root.addLayout(chat_row, 1)
        self._ai_show_welcome_message()

        # ---------- Bottom: multi-line input + Send / Stop / Clear ----------
        input_grp = QFrame()
        input_grp.setFrameShape(QFrame.StyledPanel)
        input_grp.setStyleSheet(
            "QFrame { background: #f8f9fa; border: 1px solid #ced4da; border-radius: 8px; }"
        )
        input_l = QVBoxLayout(input_grp)
        input_l.setContentsMargins(8, 8, 8, 8)
        input_l.setSpacing(6)

        self.ai_user_input = QPlainTextEdit()
        self.ai_user_input.setPlaceholderText(
            "Paste peaks or ask a question…  (Ctrl+Enter to send)"
        )
        self.ai_user_input.setMinimumHeight(70)
        self.ai_user_input.setMaximumHeight(140)
        self.ai_user_input.installEventFilter(self)
        input_l.addWidget(self.ai_user_input)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.ai_open_file_btn = QPushButton("Open file…")
        self.ai_open_file_btn.setToolTip("Open a CSV / Excel peak list")
        self.ai_open_file_btn.clicked.connect(self._ai_open_peak_file)
        btn_row.addWidget(self.ai_open_file_btn)

        self.ai_use_nmr_csv_btn = QPushButton("Use NMR-1D.csv")
        self.ai_use_nmr_csv_btn.setToolTip(
            "Run analysis using the NMR-1D.csv produced by spectrum preprocessing."
        )
        self.ai_use_nmr_csv_btn.clicked.connect(self._ai_run_with_nmr_csv)
        btn_row.addWidget(self.ai_use_nmr_csv_btn)

        self.ai_example_btn = QPushButton("Example case")
        self.ai_example_btn.setToolTip(
            "Fill the input box with a sample ¹³C peak list (24 peaks)\n"
            "so you can see the expected format and try the pipeline."
        )
        self.ai_example_btn.clicked.connect(self._ai_load_example)
        btn_row.addWidget(self.ai_example_btn)

        self.ai_direct_llm_check = QCheckBox("Direct LLM Top5 (skip DB)")
        self.ai_direct_llm_check.setChecked(False)
        self.ai_direct_llm_check.setToolTip(
            "When enabled, pasted peak lists are sent directly to the LLM to propose\n"
            "Top-5 candidate SMILES without database screening/masking.\n"
            "Use this for side-by-side comparison with the default SOP pipeline."
        )
        self.ai_direct_llm_check.setStyleSheet("font-size: 12px;")
        btn_row.addWidget(self.ai_direct_llm_check)

        btn_row.addStretch()

        self.ai_send_btn = QPushButton("Send")
        self.ai_send_btn.setObjectName("PrimaryActionButton")
        self.ai_send_btn.setStyleSheet(
            "QPushButton#PrimaryActionButton { background-color: #3d5a66; color: white; "
            "border-radius: 6px; padding: 6px 18px; font-weight: bold; } "
            "QPushButton#PrimaryActionButton:hover { background-color: #4d6e7d; } "
            "QPushButton#PrimaryActionButton:disabled { background-color: #adb5bd; }"
        )
        self.ai_send_btn.clicked.connect(self._ai_smart_send)
        btn_row.addWidget(self.ai_send_btn)

        self.ai_stop_btn = QPushButton("Stop")
        self.ai_stop_btn.setEnabled(False)
        self.ai_stop_btn.clicked.connect(self._ai_stop_all)
        btn_row.addWidget(self.ai_stop_btn)

        self.ai_clear_btn = QPushButton("Clear")
        self.ai_clear_btn.clicked.connect(self._ai_clear_conversation)
        btn_row.addWidget(self.ai_clear_btn)

        input_l.addLayout(btn_row)
        root.addWidget(input_grp)

        # Runtime state for the SOP pipeline triggered from the chat
        self._ai_sop_runner = None
        self._ai_sop_plan = None
        self._ai_sop_overrides: dict = {}
        self._ai_sop_active: bool = False

        self.tab_widget.addTab(page, "AI Assistant")

    def _refresh_ai_status_label(self):
        if not hasattr(self, "ai_status_label"):
            return
        cfg = load_llm_config()
        pid = load_provider_id() or guess_provider_id(cfg.base_url, cfg.model)
        label = get_provider(pid).get("label", pid) if pid else ""
        if is_llm_configured(cfg):
            short = label.split("·")[0].strip() if "·" in label else label
            self.ai_status_label.setText(f"Connected: {short} · {cfg.model}")
            self.ai_status_label.setStyleSheet("color: #198754; font-weight: bold;")
        else:
            self.ai_status_label.setText(
                f"Model: {cfg.model} — add API Key (API settings…)"
            )
            self.ai_status_label.setStyleSheet("color: #856404; font-weight: bold;")

    def _ai_apply_provider_preset(self, provider_id, notify=True):
        base, model = apply_provider(provider_id)
        cfg = load_llm_config()
        new_cfg = LLMConfig(
            api_key=cfg.api_key,
            base_url=base,
            model=model,
            temperature=cfg.temperature,
        )
        save_llm_config(new_cfg, provider_id=provider_id)
        self._refresh_ai_status_label()
        if notify:
            self.statusBar().showMessage(f"AI provider: {get_provider(provider_id).get('label', '')}")

    def _ai_on_provider_combo_changed(self):
        if not hasattr(self, "ai_provider_combo"):
            return
        pid = self.ai_provider_combo.currentData()
        if not pid:
            return
        self._ai_apply_provider_preset(pid, notify=True)

    def _ai_refresh_workflow_banner(self):
        if not hasattr(self, "ai_workflow_label"):
            return
        ctx = build_analysis_context(self, top_n=1)
        self.ai_workflow_label.setText(format_preflight_text(ctx.get("preflight") or []))

    def _ai_append_chat(self, role: str, text: str):
        if not hasattr(self, "ai_chat_view"):
            return
        cursor = self.ai_chat_view.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(f"\n[{role}]\n{text}\n")
        self.ai_chat_view.setTextCursor(cursor)
        sb = self.ai_chat_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _ai_append_chat_image(self, role: str, image_path: str, caption: str = ""):
        """Insert an inline image into the chat view (right after a role banner)."""
        if not hasattr(self, "ai_chat_view"):
            return
        from PyQt5.QtGui import QImage
        img = QImage(image_path)
        if img.isNull():
            self._ai_append_chat(role, f"(could not load image: {image_path})")
            return
        # Cap rendering width so the image fits the chat view.
        max_w = max(360, self.ai_chat_view.viewport().width() - 24)
        if img.width() > max_w:
            img = img.scaledToWidth(max_w, Qt.SmoothTransformation)
        cursor = self.ai_chat_view.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(f"\n[{role}]\n")
        if caption:
            cursor.insertText(caption + "\n")
        cursor.insertImage(img)
        cursor.insertText("\n")
        self.ai_chat_view.setTextCursor(cursor)
        sb = self.ai_chat_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _ai_show_top5_structures(self, *, silent: bool = False) -> bool:
        """Render the top 5 hit molecules and insert them into the chat view.

        Returns True if an image was inserted, False otherwise. When ``silent``
        is True, suppresses ``System`` chat messages for the empty/no-data case
        (used by the auto-after-pipeline flow).
        """
        import os as _os
        import tempfile as _tempfile
        try:
            import pandas as _pd  # noqa: F401
        except Exception:
            if not silent:
                self._ai_append_chat("System", "pandas is not available.")
            return False
        result = getattr(self, "result", None)
        if result is None or not hasattr(result, "iloc") or len(result) == 0:
            if not silent:
                self._ai_append_chat(
                    "System",
                    "No screening results yet — paste peaks and Send first, or run analysis."
                )
            return False
        if "smiles" not in result.columns:
            if not silent:
                self._ai_append_chat("System", "Hit table has no 'smiles' column.")
            return False
        n_mols = min(5, len(result))
        try:
            out_dir = getattr(self, "gui_output_dir", "") or _tempfile.gettempdir()
            _os.makedirs(out_dir, exist_ok=True)
            out_path = _os.path.join(out_dir, "ai_top5_structures.png")
            ok = self._export_top_structures_grid(
                result, out_path, n_mols=n_mols, n_rows=1, n_cols=n_mols
            )
        except Exception as exc:
            self._ai_append_chat("System", f"Failed to render Top-{n_mols} structures: {exc}")
            return False
        if not ok or not _os.path.isfile(out_path):
            self._ai_append_chat("System", "Could not render Top structures.")
            return False
        self._ai_append_chat_image(
            "Assistant",
            out_path,
            caption=f"Top {n_mols} candidate structures (ranked by score):",
        )
        return True

    def _ai_show_fragments_overview(self, *, silent: bool = False) -> bool:
        """Insert the *All extracted positive fragments* overview image into chat.

        Reuses the fragment-analysis tab's structure label: triggers a refresh
        of :meth:`_render_global_fragments_score_plot`, then saves the produced
        pixmap to ``gui_output_dir/ai_fragments_overview.png`` and embeds it.
        """
        import os as _os
        rows = list(getattr(self, "masking_fragments_global_rows", []) or [])
        if not rows:
            if not silent:
                self._ai_append_chat(
                    "System",
                    "No extracted fragments yet — run masking + extract_fragments first."
                )
            return False
        lbl = getattr(self, "frag_mask_global_struct_label", None)
        if lbl is None:
            if not silent:
                self._ai_append_chat("System", "Fragment overview widget not available.")
            return False
        try:
            self._render_global_fragments_score_plot()
        except Exception as exc:
            if not silent:
                self._ai_append_chat("System", f"Fragment overview render failed: {exc}")
            return False
        pix = lbl.pixmap()
        if pix is None or pix.isNull():
            if not silent:
                self._ai_append_chat(
                    "System", "Fragment overview is empty — nothing to display."
                )
            return False
        out_dir = getattr(self, "gui_output_dir", "") or "."
        try:
            _os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
        out_path = _os.path.join(out_dir, "ai_fragments_overview.png")
        if not pix.save(out_path, "PNG"):
            if not silent:
                self._ai_append_chat("System", "Could not save fragment overview image.")
            return False
        self._ai_append_chat_image(
            "Assistant",
            out_path,
            caption=f"All extracted positive fragments — {len(rows)} entries (sorted by Σscore):",
        )
        return True

    def _ai_log_path(self):
        return self._gui_output_path("ai_log.jsonl")

    def _ai_write_log(self, task: str, context: dict, reply: str):
        try:
            import hashlib
            from datetime import datetime

            h = hashlib.sha256(
                json.dumps(context, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:16]
            cfg = load_llm_config()
            line = {
                "at": datetime.now().isoformat(timespec="seconds"),
                "task": task,
                "model": cfg.model,
                "context_hash": h,
                "reply_chars": len(reply or ""),
            }
            with open(self._ai_log_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _ai_stop_llm(self):
        self._ai_llm_cancelled = True
        if self._is_thread_running("llm_thread"):
            th = self.llm_thread
            try:
                th.requestInterruption()
                th.terminate()
            except Exception:
                pass
            self.llm_thread = None
        if hasattr(self, "ai_stop_btn"):
            self.ai_stop_btn.setEnabled(False)
        self._ai_append_chat(
            "System",
            "AI request stopped. If the reply was empty, run Top 5/10 again without clicking Stop.",
        )
        self.statusBar().showMessage("AI request cancelled")

    def _ai_topn_max_tokens(self, context: dict) -> int:
        n = int(context.get("analysis_top_n") or len(context.get("top_candidates") or []) or 5)
        if n <= 5:
            return 4096
        if n <= 10:
            return 5500
        return 7000

    @staticmethod
    def _ai_compact_context_for_retry(context: dict) -> dict:
        """Trim large context fields for one-shot length-limit retry."""
        c = dict(context or {})
        peaks = list(c.get("experimental_peaks") or [])
        if len(peaks) > 40:
            c["experimental_peaks"] = peaks[:40]
            c["experimental_peaks_note"] = f"truncated to first 40 of {len(peaks)}"

        top = list(c.get("top_candidates") or [])
        if top:
            trimmed = []
            for cand in top[:5]:
                item = dict(cand)
                shifts = list(item.get("predicted_shifts_ppm") or [])
                if len(shifts) > 16:
                    item["predicted_shifts_ppm"] = shifts[:16]
                    item["predicted_shifts_note"] = f"truncated to first 16 of {len(shifts)}"
                trimmed.append(item)
            c["top_candidates"] = trimmed
            if len(top) > 5:
                c["top_candidates_note"] = f"truncated to top 5 of {len(top)}"

        # fragment payloads can be very large; keep only compact summaries
        if isinstance(c.get("fragment_analysis"), dict):
            frag = dict(c["fragment_analysis"])
            for heavy_key in ("fragments_all", "pairs_all", "raw_rows", "global_rows"):
                if heavy_key in frag:
                    frag.pop(heavy_key, None)
            c["fragment_analysis"] = frag
        return c

    def _ai_compact_fusion_top5_context(self, context: dict) -> dict:
        """Fusion retry context: peaks + one combination, no parent SMILES list."""
        c = dict(context or {})
        peaks = list(c.get("experimental_peaks") or [])
        if len(peaks) > 35:
            peaks = peaks[:35]
        fc = dict((c.get("fusion_combination") or {}))
        motifs = []
        for m in list(fc.get("fragment_motifs") or fc.get("fragments") or [])[:6]:
            motifs.append({
                "label": m.get("label"),
                "n_carbons": m.get("n_carbons"),
                "fragment_smiles": m.get("fragment_smiles"),
                "carbon_profile": m.get("carbon_profile"),
                "matched_peak_summary": m.get("matched_peak_summary"),
            })
        return {
            "experimental_peaks": peaks,
            "experimental_summary": c.get("experimental_summary"),
            "unmatched_exp_peaks": list(c.get("unmatched_exp_peaks") or [])[:20],
            "fusion_combination": {
                "coverage_pct": fc.get("coverage_pct"),
                "aas_best": fc.get("aas_best"),
                "matched_pairs": list(fc.get("matched_pairs") or [])[:25],
                "fragment_motifs": motifs,
            },
            "constraints": c.get("constraints"),
            "retry_note": "Compact context — still assemble NEW full structures from the selected "
                          "fragments only, matching the carbon budget; rank best-first (1-5).",
        }

    def _ai_compact_direct_top5_context(self, context: dict) -> dict:
        """Minimal peaks-only context for direct_top5 length-limit retries."""
        peaks = list((context or {}).get("experimental_peaks") or [])
        if len(peaks) > 30:
            peaks = peaks[:30]
        out = {
            "experimental_peaks": peaks,
            "analysis_top_n": 5,
            "constraints": {
                "require_exactly_five": True,
                "must_output_smiles": True,
            },
        }
        summary = (context or {}).get("experimental_summary")
        if isinstance(summary, dict):
            out["experimental_summary"] = summary
        return out

    def _ai_max_tokens_for_task(self, task: str, context: dict, *, retry_tier: int = 0):
        """Task-aware token budget. Direct-top5 and topn need larger outputs."""
        if task == "topn":
            base = self._ai_topn_max_tokens(context)
            return base + (2048 if retry_tier > 0 else 0)
        if task in ("direct_top5", "fusion_infer_top5", "infer_structure", "signal_assignment", "fragment_evidence_review", "fusion_evidence_review"):
            return 20480 if retry_tier > 0 else 16384
        if task == "chat":
            return 2048
        return None

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        m = (model or "").lower()
        tokens = ("reasoner", "-pro", "think", "o1", "o3", "o4-mini", "qwq", "-r1", "deepseek-r")
        return any(tok in m for tok in tokens)

    def _ai_reasoning_effort_for_task(self, task: str):
        """Effort hint for generation tasks. User config wins; else nudge reasoning models low.

        Returns a string ("low"/"medium"/"high"/"minimal") or None (do not send the field).
        Structure-generation tasks fail when a reasoning model spends the whole budget on
        visible chain-of-thought, so we lower effort there (with a 400-safe fallback in the client).
        """
        try:
            cfg = load_llm_config()
        except Exception:
            return None
        configured = (getattr(cfg, "reasoning_effort", "") or "").strip().lower()
        if configured in ("low", "medium", "high", "minimal"):
            return configured
        if task in ("direct_top5", "fusion_infer_top5", "infer_structure", "signal_assignment", "fragment_evidence_review", "fusion_evidence_review") and \
                self._is_reasoning_model(getattr(cfg, "model", "")):
            return "low"
        return None

    def _ai_start_llm(
        self,
        task: str,
        context: dict,
        extra_user: str = "",
        *,
        _retry: bool = False,
        _retry_tier: int = 0,
    ):
        if self._is_thread_running("llm_thread"):
            QMessageBox.information(self, "AI", "Wait for the current AI request to finish, or click Stop.")
            return
        if not is_llm_configured():
            QMessageBox.warning(
                self,
                "AI",
                "LLM API not configured.\nUse Edit → Preferences or set VIRMOL_LLM_API_KEY.",
            )
            return
        self._ai_llm_cancelled = False
        # Keep request metadata for error-retry handling.
        self._ai_last_task = task
        self._ai_last_context = context
        self._ai_last_extra_user = extra_user
        if not _retry:
            self._ai_length_retry_tier = 0
        retry_tier = _retry_tier if _retry else 0
        compact_ctx = _retry and task in (
            "topn", "direct_top5", "fusion_infer_top5", "signal_assignment", "fragment_evidence_review", "fusion_evidence_review"
        )
        messages = build_messages(
            task, context, extra_user=extra_user, compact_context=compact_ctx
        )
        # Show the task banner for analytical jobs only — "chat" is a normal
        # conversation and the banner would just be noise.
        if task != "chat":
            self._ai_append_chat("Task", task)
        self.ai_stop_btn.setEnabled(True)
        self.statusBar().showMessage("AI request in progress…")
        max_tokens = self._ai_max_tokens_for_task(task, context, retry_tier=retry_tier)
        effort = self._ai_reasoning_effort_for_task(task)
        self.llm_thread = LLMThread(messages, max_tokens=max_tokens, reasoning_effort=effort)
        self.llm_thread.finished.connect(
            lambda text, t=task, c=context: self._ai_on_llm_finished(t, c, text)
        )
        self.llm_thread.error.connect(self._ai_on_llm_error)
        self.llm_thread.start()

    def _ai_on_llm_finished(self, task: str, context: dict, text: str):
        self.llm_thread = None
        if hasattr(self, "ai_stop_btn"):
            self.ai_stop_btn.setEnabled(False)
        if getattr(self, "_ai_llm_cancelled", False):
            self.statusBar().showMessage("AI request cancelled")
            return
        body = (text or "").strip()
        if not body:
            n = int(context.get("analysis_top_n") or 0)
            body = (
                "**No text was returned from the LLM.**\n\n"
                "Common causes:\n"
                "- The request was interrupted (Stop) or timed out\n"
                "- The prompt was too large (try **Top 5** or **Top 10** instead of Top 20)\n"
                "- The API returned empty content (model limit or provider issue)\n\n"
                f"Context sent: {n} candidate(s). Check status bar / API settings, then retry."
            )
            self._ai_append_chat("Assistant", body)
            self._ai_write_log(task, context, body)
            self.statusBar().showMessage("AI returned empty reply")
            QMessageBox.warning(
                self,
                "AI Assistant",
                "The model returned no text. Try Top 5/10, wait for completion, "
                "or verify API key and model in API settings.",
            )
            return
        if self._ai_output_incomplete_for_task(task, body):
            if self._ai_launch_concise_retry(task, "Reply was truncated before a usable answer"):
                return
        if (
            self._ai_pending_append_disclaimer
            and task not in ("polish_methods", "chat")
        ):
            body = body.rstrip() + DISCLAIMER_ZH
        self._ai_append_chat("Assistant", body)
        self._ai_write_log(task, context, body)
        self.statusBar().showMessage("AI reply complete")
        self._ai_refresh_workflow_banner()

    def _ai_on_llm_error(self, msg: str):
        self.llm_thread = None
        if hasattr(self, "ai_stop_btn"):
            self.ai_stop_btn.setEnabled(False)
        task = getattr(self, "_ai_last_task", "")
        lower_msg = str(msg or "").lower()
        if "finish_reason=length" in lower_msg and task in (
            "topn", "direct_top5", "fusion_infer_top5", "infer_structure", "signal_assignment", "fragment_evidence_review", "fusion_evidence_review"
        ):
            if self._ai_launch_concise_retry(task, "Model output hit length limit"):
                return
        self._ai_append_chat("Error", msg)
        QMessageBox.warning(self, "AI request failed", msg)
        self.statusBar().showMessage("AI request failed")

    def _ai_launch_concise_retry(self, task: str, reason: str) -> bool:
        """Relaunch a truncated/incomplete generative task with compact context.

        Returns True if a retry was scheduled, False if the retry budget is exhausted.
        """
        tier = int(getattr(self, "_ai_length_retry_tier", 0) or 0)
        max_retries = 2 if task in (
            "direct_top5", "fusion_infer_top5", "infer_structure", "signal_assignment", "fragment_evidence_review", "fusion_evidence_review"
        ) else 1
        if tier >= max_retries:
            return False
        self._ai_length_retry_tier = tier + 1
        last_ctx = getattr(self, "_ai_last_context", {}) or {}
        if task == "direct_top5":
            compact_ctx = self._ai_compact_direct_top5_context(last_ctx)
        elif task == "fusion_infer_top5":
            compact_ctx = self._ai_compact_fusion_top5_context(last_ctx)
        elif task == "infer_structure":
            compact_ctx = self._ai_compact_infer_structure_context(last_ctx)
        elif task == "signal_assignment":
            compact_ctx = self._ai_compact_signal_assignment_context(last_ctx)
        elif task == "fragment_evidence_review":
            compact_ctx = self._ai_compact_fragment_review_context(last_ctx)
        elif task == "fusion_evidence_review":
            compact_ctx = self._ai_compact_fusion_review_context(last_ctx)
        else:
            compact_ctx = self._ai_compact_context_for_retry(last_ctx)
        style_note = (
            "a structure-first, answer-only instruction."
            if task in ("direct_top5", "fusion_infer_top5", "infer_structure")
            else "a table-first, schema-locked instruction."
            if task in ("signal_assignment", "fragment_evidence_review", "fusion_evidence_review")
            else "a concise answer-only instruction."
        )
        self._ai_append_chat(
            "System",
            f"{reason}; retry {tier + 1}/{max_retries} with compact context and {style_note}",
        )
        extra = (getattr(self, "_ai_last_extra_user", "") or "").strip()
        extra_retry = (
            (extra + "\n\n" if extra else "")
            + "Be concise and fit within the output limit. Do NOT narrate reasoning."
        )
        if task in ("direct_top5", "fusion_infer_top5", "infer_structure"):
            extra_retry += (
                "\n\nOutput the '## Most likely structures' block FIRST and emit the #1 "
                "SMILES immediately. Keep any reasoning to one short line per structure."
            )
        elif task == "signal_assignment":
            extra_retry += (
                "\n\nOutput ONLY the required five sections for signal assignment, keep the "
                "assignment table complete, and keep narrative concise."
            )
        elif task == "fragment_evidence_review":
            extra_retry += (
                "\n\nOutput ONLY the required six sections for fragment evidence review "
                "(likely motifs & compound class FIRST, then ranking, redundancy/complementarity, "
                "unexplained peaks, fusion-ready set, risks). Keep section 1 under 120 words."
            )
        elif task == "fusion_evidence_review":
            extra_retry += (
                "\n\nOutput ONLY the required five sections for fusion evidence review "
                "(ranking rationale, fragment roles, unexplained/conflicting signals, recommendations, next checks)."
            )
        self._ai_start_llm(
            task,
            compact_ctx,
            extra_user=extra_retry,
            _retry=True,
            _retry_tier=tier + 1,
        )
        return True

    @staticmethod
    def _ai_structure_output_incomplete(task: str, body: str) -> bool:
        """Detect a truncated / SMILES-less reply for structure-generation tasks."""
        if task not in ("direct_top5", "fusion_infer_top5", "infer_structure"):
            return False
        low = (body or "").lower()
        if "reply was truncated by the model output token limit" in low:
            return True
        # No usable structure emitted at all.
        if "smiles:" not in low:
            return True
        return False

    @staticmethod
    def _ai_signal_assignment_output_incomplete(body: str) -> bool:
        """Detect incomplete structured output for signal_assignment task."""
        low = (body or "").lower()
        if "reply was truncated by the model output token limit" in low:
            return True
        required = (
            "## assignment table",
            "## confident assignments",
            "## uncertain or conflicting",
            "## unassigned experimental peaks",
            "## summary",
        )
        return not all(h in low for h in required)

    @staticmethod
    def _ai_fragment_review_output_incomplete(body: str) -> bool:
        low = (body or "").lower()
        if "reply was truncated by the model output token limit" in low:
            return True
        required = (
            "## likely structural motifs & compound class",
            "## fragment evidence ranking",
            "## redundant vs complementary fragments",
            "## unexplained peaks and likely missing motifs",
            "## recommended fusion-ready fragment set",
            "## risks and verification priorities",
        )
        return not all(h in low for h in required)

    @staticmethod
    def _ai_fusion_review_output_incomplete(body: str) -> bool:
        low = (body or "").lower()
        if "reply was truncated by the model output token limit" in low:
            return True
        required = (
            "## combination ranking rationale",
            "## fragment-role interpretation",
            "## unexplained / conflicting signals",
            "## recommended combinations for verification",
            "## next experimental checks",
        )
        return not all(h in low for h in required)

    @classmethod
    def _ai_output_incomplete_for_task(cls, task: str, body: str) -> bool:
        """Task-aware completion check used before accepting LLM output."""
        if task == "signal_assignment":
            return cls._ai_signal_assignment_output_incomplete(body)
        if task == "fragment_evidence_review":
            return cls._ai_fragment_review_output_incomplete(body)
        if task == "fusion_evidence_review":
            return cls._ai_fusion_review_output_incomplete(body)
        return cls._ai_structure_output_incomplete(task, body)

    def _ai_build_context(self, **kwargs):
        if kwargs.pop("include_fragment", None) is None and hasattr(self, "ai_include_fragment_check"):
            kwargs["include_fragment"] = self.ai_include_fragment_check.isChecked()
        return build_analysis_context(self, **kwargs)

    def _ai_run_task(self, task: str, **kwargs):
        ctx = self._ai_build_context(**kwargs)
        if task == "polish_methods":
            draft = ctx.get("methodology_draft_en") or ""
            if not draft.strip():
                QMessageBox.warning(self, "Methods", "Cannot build Methods text — configure analysis settings first.")
                return
            ctx = {"methodology_draft_en": draft, "rule_only": False}
        if task == "diagnosis_llm":
            items = run_screening_diagnosis(ctx)
            ctx = {"rule_diagnosis": items, "workflow": ctx.get("workflow")}
        self._ai_start_llm(task, ctx)

    def _ai_send_freeform(self):
        text = self.ai_user_input.text().strip()
        if not text:
            return
        self.ai_user_input.clear()
        ctx = self._ai_build_context(top_n=10)
        ctx["user_question"] = text
        self._ai_start_llm("wizard", ctx, extra_user=text)

    def _ai_rule_diagnosis(self):
        ctx = self._ai_build_context(top_n=15)
        items = run_screening_diagnosis(ctx)
        self._ai_append_chat("Rule diagnosis", format_diagnosis_text(items))
        self._ai_refresh_workflow_banner()

    def _ai_show_rule_diagnosis_in_panel(self):
        if not hasattr(self, "ai_chat_view"):
            return
        ctx = self._ai_build_context(top_n=5)
        items = run_screening_diagnosis(ctx)
        self._ai_append_chat("Rule diagnosis", format_diagnosis_text(items))
        if hasattr(self, "tab_widget"):
            for i in range(self.tab_widget.count()):
                if self.tab_widget.tabText(i) == "AI Assistant":
                    self.tab_widget.setCurrentIndex(i)
                    break

    def _ai_copy_results_draft(self):
        ctx = self._ai_build_context(top_n=3)
        note = self.ai_sample_note.text() if hasattr(self, "ai_sample_note") else ""
        text = build_results_draft(ctx, sample_note=note)
        QApplication.clipboard().setText(text)
        self._ai_append_chat("System", "Results draft copied to clipboard.")
        self.statusBar().showMessage("Results draft copied")

    def _results_table_context_menu(self, pos):
        menu = QMenu(self)
        row = self.results_table.rowAt(pos.y())
        act_interpret = menu.addAction("AI: Interpret Top 5")
        act_interpret.triggered.connect(lambda: self._ai_run_task("topn", top_n=5))
        if row >= 0:
            act_row = menu.addAction(f"AI: Interpret row {row + 1}")
            act_row.triggered.connect(
                lambda r=row: self._ai_run_task(
                    "compound", selected_row=r, top_n=5
                )
            )
            if row + 1 < self.results_table.rowCount():
                act_cmp = menu.addAction(f"Compare #{row + 1} vs #{row + 2}")
                act_cmp.triggered.connect(
                    lambda r=row: self._ai_compare_two(r, r + 1)
                )
        menu.exec_(self.results_table.viewport().mapToGlobal(pos))

    def _ai_compare_two(self, row_a: int, row_b: int):
        ctx = self._ai_build_context(top_n=max(row_a, row_b) + 2, compare_rows=(row_a, row_b))
        extra = (
            f"Compare #rank {row_a + 1} vs #rank {row_b + 1}: differences and verification advice."
        )
        if not is_llm_configured():
            self._ai_append_chat("System", extra + "\n(Configure API to run LLM.)")
            return
        self._ai_start_llm("topn", ctx, extra_user=extra)

    def _ai_compound_from_dialog(self):
        row = getattr(self, "_compound_detail_current_row", -1)
        if row < 0:
            QMessageBox.warning(self, "AI", "No candidate selected.")
            return
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "AI Assistant":
                self.tab_widget.setCurrentIndex(i)
                break
        self._ai_run_task("compound", selected_row=row, top_n=5)

    def _ai_goto_assistant_tab(self):
        if not hasattr(self, "tab_widget"):
            return
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "AI Assistant":
                self.tab_widget.setCurrentIndex(i)
                break
        if hasattr(self, "_ai_refresh_workflow_banner"):
            self._ai_refresh_workflow_banner()
        self.statusBar().showMessage("Switched to AI Assistant tab")

    def _ai_goto_tab(self, tab_label: str) -> bool:
        """Switch to the tab whose visible text matches ``tab_label``.

        Used by the AI Assistant's left-side quick-jump column. Returns True
        on success so callers can react if the tab is missing.
        """
        if not hasattr(self, "tab_widget"):
            return False
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == tab_label:
                self.tab_widget.setCurrentIndex(i)
                self.statusBar().showMessage(f"Switched to {tab_label} tab")
                return True
        if hasattr(self, "_ai_append_chat"):
            self._ai_append_chat("System", f"Tab '{tab_label}' is not available.")
        return False

    def _ai_goto_attribution_view(self):
        """Switch to the Fragment Analysis tab and force the *Compound
        attribution view* selection (the per-candidate masking image)."""
        if not self._ai_goto_tab("Fragment Analysis"):
            return
        combo = getattr(self, "frag_mask_view_combo", None)
        if combo is not None:
            for i in range(combo.count()):
                if combo.itemText(i) == "Compound attribution view":
                    if combo.currentIndex() != i:
                        combo.setCurrentIndex(i)
                    break
        # If a candidate is already selected, ensure the result combo points
        # at it so the view renders immediately.
        rc = getattr(self, "frag_mask_result_combo", None)
        if rc is not None and rc.isEnabled() and rc.count() > 0 and rc.currentIndex() < 0:
            rc.setCurrentIndex(0)
        self.statusBar().showMessage("Switched to Compound attribution view")

    def _ai_open_output_folder(self):
        """Open the GUI output folder (where analysis exports are written)."""
        import os as _os
        import sys as _sys
        import subprocess as _sp
        folder = getattr(self, "gui_output_dir", "") or _os.getcwd()
        if not _os.path.isdir(folder):
            try:
                _os.makedirs(folder, exist_ok=True)
            except Exception as exc:
                self._ai_append_chat("System", f"Output folder not available: {exc}")
                return
        try:
            if _sys.platform.startswith("win"):
                _os.startfile(folder)  # type: ignore[attr-defined]
            elif _sys.platform == "darwin":
                _sp.Popen(["open", folder])
            else:
                _sp.Popen(["xdg-open", folder])
            self.statusBar().showMessage(f"Opened {folder}")
            self._ai_append_chat("System", f"Opened output folder: {folder}")
        except Exception as exc:
            self._ai_append_chat("System", f"Could not open folder: {exc}")

    def _ai_interpret_top5_from_analysis(self):
        """From Molecular Analysis: switch to AI Assistant and run Top 5 LLM task."""
        n_rows = self.results_table.rowCount() if hasattr(self, "results_table") else 0
        if n_rows == 0:
            if getattr(self, "result", None) is None:
                QMessageBox.warning(
                    self,
                    "AI Assistant",
                    "No screening results yet.\n\n"
                    "Run Database Analysis (Start Analysis) first, then interpret Top 5 hits.",
                )
            else:
                QMessageBox.warning(
                    self,
                    "AI Assistant",
                    "The results table has no hits — Top 5 interpretation needs at least one candidate.",
                )
            return
        self._ai_goto_assistant_tab()
        self._ai_run_task("topn", top_n=5)
        self.statusBar().showMessage("AI Assistant: interpreting top 5 screening hits…")

    def _ai_fragment_list_index(self) -> Optional[int]:
        if hasattr(self, "frag_mask_result_combo") and self.frag_mask_result_combo.isEnabled():
            return int(self.frag_mask_result_combo.currentIndex())
        if getattr(self, "masking_result_data", None):
            return int(self.masking_result_data.get("candidate_rank_index", 0))
        return None

    def _ai_build_fragment_llm_context(
        self,
        *,
        fragment_row_index: Optional[int] = None,
        candidate_list_index: Optional[int] = None,
        include_screening: bool = True,
    ) -> Optional[dict]:
        if not has_fragment_data(self):
            return None
        frag = build_fragment_analysis_context(
            self,
            candidate_list_index=candidate_list_index
            if candidate_list_index is not None
            else self._ai_fragment_list_index(),
            fragment_row_index=fragment_row_index,
        )
        if not frag:
            return None
        if include_screening:
            base = build_analysis_context(self, top_n=8, include_fragment=False)
            ctx = {
                "workflow": base.get("workflow"),
                "experimental_peaks": frag.get("experimental_peaks") or base.get("experimental_peaks"),
                "top_candidates": base.get("top_candidates"),
                "fragment_analysis": frag,
            }
            return ctx
        return {"fragment_analysis": frag, "experimental_peaks": frag.get("experimental_peaks")}

    def _ai_run_fragment_task(
        self,
        task: str,
        *,
        fragment_row_index: Optional[int] = None,
        candidate_list_index: Optional[int] = None,
    ):
        if not has_fragment_data(self):
            QMessageBox.warning(
                self,
                "Fragment Analysis",
                "No masking data. Run Random masking attribution on the Fragment Analysis tab, "
                "then Extract positive fragments for richer context.",
            )
            return
        ctx = self._ai_build_fragment_llm_context(
            fragment_row_index=fragment_row_index,
            candidate_list_index=candidate_list_index,
        )
        if not ctx:
            QMessageBox.warning(self, "Fragment Analysis", "Could not build fragment context.")
            return
        self._ai_goto_assistant_tab()
        self._ai_start_llm(task, ctx)

    def _ai_fragment_narrate(self):
        self._ai_run_fragment_task("fragment")

    def _ai_fragment_explain_row(self, row: int):
        self._ai_run_fragment_task("fragment_explain", fragment_row_index=row)

    def _ai_selected_masking_payload(self) -> Optional[dict]:
        """The masking result currently selected on the Fragment Analysis tab."""
        all_res = list(getattr(self, "masking_all_results", []) or [])
        idx = self._ai_fragment_list_index()
        if all_res:
            if idx is not None and 0 <= int(idx) < len(all_res):
                return all_res[int(idx)]
            return all_res[0]
        return getattr(self, "masking_result_data", None)

    def _ai_build_infer_structure_context(self, payload: dict) -> dict:
        """Per-carbon table of the selected candidate + original screening peaks."""
        import math as _math

        scores = list(payload.get("mean_scores") or [])
        shifts = list(payload.get("vir_shifts") or [])
        types = list(payload.get("lib_types") or [])
        atom_idx = list(payload.get("carbon_atom_indices") or [])
        n = max(len(scores), len(shifts), len(types))

        per_carbon = []
        for i in range(n):
            sc = None
            if i < len(scores):
                try:
                    f = float(scores[i])
                    sc = round(f, 4) if _math.isfinite(f) else None
                except (TypeError, ValueError):
                    sc = None
            per_carbon.append({
                "carbon_slot": i,
                "rdkit_atom_index": int(atom_idx[i]) if i < len(atom_idx) else None,
                "vir_shift_ppm": round(float(shifts[i]), 2) if i < len(shifts) else None,
                "dept_type": str(types[i]) if i < len(types) else None,
                "attribution_score": sc,
            })

        exp = self.get_experimental_data() or []
        exp_peaks = [
            {"ppm": round(float(p.get("ppm", 0.0)), 2), "type": str(p.get("type", "")).lower()}
            for p in exp
        ]

        aas_full = payload.get("aas_full")
        try:
            aas_full = round(float(aas_full), 4)
        except (TypeError, ValueError):
            aas_full = None

        return {
            "workflow": {"mode": "infer_structure", "analysis_done": True},
            "candidate": {
                "smiles": payload.get("smiles"),
                "note": "Closest database analog from screening — not necessarily the true structure",
                "screening_rank": int(payload.get("candidate_rank_index", 0)) + 1,
                "n_carbons": payload.get("n_carbons"),
                "aas_full": aas_full,
                "carbon_type_counts": self._ai_count_carbon_types(
                    [{"type": t} for t in types]
                ),
            },
            "per_carbon_table": per_carbon,
            "experimental_peaks": exp_peaks,
            "experimental_summary": {
                "n_peaks": len(exp_peaks),
                "carbon_type_counts": self._ai_count_carbon_types(exp_peaks),
            },
            "constraints": {
                "must_output_smiles": True,
                "candidate_smiles_is_analog_clue": True,
                "use_attribution_scores_and_shift_mismatch_to_locate_differences": True,
            },
        }

    def _ai_compact_infer_structure_context(self, context: dict) -> dict:
        c = dict(context or {})
        tbl = list(c.get("per_carbon_table") or [])
        if len(tbl) > 40:
            c["per_carbon_table"] = tbl[:40]
            c["per_carbon_table_note"] = f"truncated to first 40 of {len(tbl)}"
        peaks = list(c.get("experimental_peaks") or [])
        if len(peaks) > 40:
            c["experimental_peaks"] = peaks[:40]
        return c

    def _ai_extract_fragments_for_payload(self, payload: dict, exp: list) -> list:
        """Run score-based fragment extraction for one candidate using the GUI's params."""
        from VirMolAnalyte.masking_aas_attribution import extract_positive_fragments

        seed_txt = (
            self.frag_mask_greedy_shuffle_seed.text().strip()
            if hasattr(self, "frag_mask_greedy_shuffle_seed")
            else ""
        )
        shuffle_seed = int(seed_txt) if seed_txt else None
        score_threshold, _note = self._resolve_fragment_score_threshold(
            payload.get("mean_scores", [])
        )
        return extract_positive_fragments(
            payload["smiles"],
            payload["carbon_atom_indices"],
            payload["mean_scores"],
            payload.get("vir_shifts", []),
            payload.get("lib_types", []),
            exp,
            score_threshold=score_threshold,
            min_carbons=int(self.frag_mask_frag_min_c.value()),
            use_dept_constraint=self.frag_mask_extract_dept_check.isChecked(),
            greedy_unique_matching=self.frag_mask_extract_greedy_check.isChecked(),
            bridge_max_low_carbons=(
                int(self.frag_mask_bridge_max_low.value())
                if self.frag_mask_bridge_check.isChecked()
                else 0
            ),
            allow_hetero_bridge_neighbors=(
                self.frag_mask_allow_hetero_bridge_check.isChecked()
                if hasattr(self, "frag_mask_allow_hetero_bridge_check")
                else True
            ),
            max_fragments=int(self.frag_mask_max_frags.value()),
            greedy_shuffle_repeats=int(self.frag_mask_greedy_shuffle_repeats.value()),
            greedy_shuffle_seed=shuffle_seed,
        )

    def _ai_build_signal_assignment_context(self, payload: dict) -> dict:
        """Score-based fragment extraction + whole-molecule signal assignment for the LLM.

        Combines the deterministic peak↔carbon matching (_match_pred_to_exp_pairs) with the
        attribution scores and high-score fragment membership, so the LLM only has to validate
        and structure the result.
        """
        import math as _math
        from VirMolAnalyte.masking_aas_attribution import _match_pred_to_exp_pairs

        scores = list(payload.get("mean_scores") or [])
        shifts = list(payload.get("vir_shifts") or [])
        types = [str(t) for t in (payload.get("lib_types") or [])]
        atom_idx = list(payload.get("carbon_atom_indices") or [])
        n = max(len(scores), len(shifts), len(types))

        exp = self.get_experimental_data() or []
        exp_peaks = [
            {"ppm": round(float(p.get("ppm", 0.0)), 2), "type": str(p.get("type", "")).lower()}
            for p in exp
        ]

        # Map each carbon (slot) -> which high-score fragment it belongs to.
        slot_to_fragment = {}
        high_fragments = []
        try:
            fragments = self._ai_extract_fragments_for_payload(payload, exp) or []
        except Exception:
            fragments = []
        for fi, frag in enumerate(fragments):
            label = f"F{int(frag.get('fragment_id', fi + 1))}"
            fa_idx = [int(x) for x in (frag.get("atom_indices") or [])]
            for a in fa_idx:
                slot = atom_idx.index(a) if a in atom_idx else None
                if slot is not None:
                    slot_to_fragment[slot] = label
            high_fragments.append({
                "label": label,
                "fragment_smiles": frag.get("fragment_smiles") or frag.get("smiles"),
                "n_carbons": int(frag.get("n_carbons", len(fa_idx))),
            })

        # Deterministic signal assignment: each carbon -> best experimental peak.
        use_dept = (
            self.frag_mask_dept_check.isChecked()
            if hasattr(self, "frag_mask_dept_check") else True
        )
        pairs = _match_pred_to_exp_pairs(
            shifts, types, exp_peaks,
            use_dept_constraint=use_dept,
            greedy_unique_matching=True,
        )
        pair_by_slot = {int(p.get("pred_local_index", -1)): p for p in pairs}
        assigned_exp_idx = {int(p.get("exp_index", -1)) for p in pairs}

        assignment_table = []
        for i in range(n):
            sc = None
            if i < len(scores):
                try:
                    f = float(scores[i])
                    sc = round(f, 3) if _math.isfinite(f) else None
                except (TypeError, ValueError):
                    sc = None
            pr = pair_by_slot.get(i)
            assignment_table.append({
                "atom": int(atom_idx[i]) if i < len(atom_idx) else i,
                "dept_type": types[i] if i < len(types) else None,
                "vir_shift_ppm": round(float(shifts[i]), 2) if i < len(shifts) else None,
                "exp_ppm": round(float(pr["exp_ppm"]), 2) if pr else None,
                "abs_err": round(float(pr["abs_err"]), 2) if pr else None,
                "attribution_score": sc,
                "fragment_label": slot_to_fragment.get(i),
                "in_high_score_fragment": i in slot_to_fragment,
            })

        unassigned = [
            ep for j, ep in enumerate(exp_peaks) if j not in assigned_exp_idx
        ]

        aas_full = payload.get("aas_full")
        try:
            aas_full = round(float(aas_full), 4)
        except (TypeError, ValueError):
            aas_full = None

        return {
            "workflow": {"mode": "signal_assignment", "analysis_done": True},
            "candidate": {
                "smiles": payload.get("smiles"),
                "note": "Closest database analog from screening — not necessarily the true structure",
                "screening_rank": int(payload.get("candidate_rank_index", 0)) + 1,
                "n_carbons": payload.get("n_carbons"),
                "aas_full": aas_full,
            },
            "assignment_table": assignment_table,
            "high_score_fragments": high_fragments,
            "unassigned_exp_peaks": unassigned,
            "experimental_summary": {
                "n_peaks": len(exp_peaks),
                "carbon_type_counts": self._ai_count_carbon_types(exp_peaks),
                "dept_constraint_used": bool(use_dept),
            },
            "constraints": {
                "validate_only_no_new_smiles": True,
                "use_only_provided_values": True,
            },
        }

    def _ai_compact_signal_assignment_context(self, context: dict) -> dict:
        c = dict(context or {})
        tbl = list(c.get("assignment_table") or [])
        if len(tbl) > 45:
            c["assignment_table"] = tbl[:45]
            c["assignment_table_note"] = f"truncated to first 45 of {len(tbl)}"
        return c

    def _ai_build_fragment_review_context(self, payload: dict) -> dict:
        """Context for reviewing extracted positive fragments + mapping evidence."""
        exp = self.get_experimental_data() or []
        exp_peaks = [
            {"ppm": round(float(p.get("ppm", 0.0)), 2), "type": str(p.get("type", "")).lower()}
            for p in exp
        ]
        fragments = list(getattr(self, "masking_fragments", []) or [])
        if not fragments:
            # Fallback: derive fragments for the selected payload when table is empty.
            try:
                fragments = self._ai_extract_fragments_for_payload(payload, exp) or []
            except Exception:
                fragments = []
        rows = []
        assigned_exp_idx = set()
        for i, frag in enumerate(fragments):
            pairs = list(frag.get("pairs") or [])
            pair_examples = []
            for pr in pairs[:10]:
                try:
                    exp_idx = int(pr.get("exp_index", -1))
                except Exception:
                    exp_idx = -1
                if exp_idx >= 0:
                    assigned_exp_idx.add(exp_idx)
                pair_examples.append({
                    "atom_index": pr.get("atom_index"),
                    "ctype": pr.get("ctype"),
                    "pred_ppm": round(float(pr.get("pred_ppm", 0.0)), 2),
                    "exp_ppm": round(float(pr.get("exp_ppm", 0.0)), 2),
                    "abs_err": round(float(pr.get("abs_err", 0.0)), 2),
                })
            enrich = self._ai_enrich_fragment_row_for_review(frag)
            rows.append({
                "fragment_label": f"F{int(frag.get('fragment_id', i + 1))}",
                "fragment_id": int(frag.get("fragment_id", i + 1)),
                "fragment_smiles": frag.get("fragment_smiles") or frag.get("smiles"),
                "n_carbons": int(frag.get("n_carbons", 0)),
                "n_core_high_carbons": int(frag.get("n_core_high_carbons", frag.get("n_carbons", 0))),
                "score_sum": round(float(frag.get("score_sum", 0.0)), 4),
                "score_mean": round(float(frag.get("score_mean", 0.0)), 4),
                "hit_rate_unique_pct": round(100.0 * float(frag.get("hit_rate_unique", 0.0)), 2),
                "mean_abs_err": round(float(frag.get("mean_abs_err", 0.0)), 3),
                "match_mse": round(float(frag.get("match_mse", 0.0)), 4),
                "pair_examples": pair_examples,
                **enrich,
            })
        unassigned = [ep for j, ep in enumerate(exp_peaks) if j not in assigned_exp_idx]
        rows.sort(key=lambda r: (r["score_sum"], r["hit_rate_unique_pct"]), reverse=True)
        return {
            "workflow": {"mode": "fragment_evidence_review", "analysis_done": True},
            "candidate": {
                "smiles": payload.get("smiles"),
                "screening_rank": int(payload.get("candidate_rank_index", 0)) + 1,
                "n_carbons": payload.get("n_carbons"),
                "aas_full": payload.get("aas_full"),
            },
            "fragment_rows": rows,
            "unassigned_exp_peaks": unassigned,
            "experimental_summary": {
                "n_peaks": len(exp_peaks),
                "carbon_type_counts": self._ai_count_carbon_types(exp_peaks),
            },
            "constraints": {
                "no_new_smiles": True,
                "review_extracted_fragments_only": True,
            },
        }

    def _ai_compact_fragment_review_context(self, context: dict) -> dict:
        c = dict(context or {})
        rows = list(c.get("fragment_rows") or [])
        if len(rows) > 10:
            c["fragment_rows"] = rows[:10]
            c["fragment_rows_note"] = f"truncated to top 10 of {len(rows)}"
        for r in c.get("fragment_rows", []):
            ex = list(r.get("pair_examples") or [])
            if len(ex) > 6:
                r["pair_examples"] = ex[:6]
        un = list(c.get("unassigned_exp_peaks") or [])
        if len(un) > 20:
            c["unassigned_exp_peaks"] = un[:20]
        return c

    def _ai_infer_structure_from_selected(self):
        """Send the selected candidate's masking data + screening peaks to the LLM."""
        if not has_fragment_data(self):
            QMessageBox.warning(
                self,
                "Fragment Analysis",
                "No masking data. Run Random masking attribution on the Fragment Analysis tab first, "
                "then select a candidate in the result table.",
            )
            return
        payload = self._ai_selected_masking_payload()
        if not payload:
            QMessageBox.warning(
                self,
                "Fragment Analysis",
                "Could not read the selected candidate. Select a row in the masking result table.",
            )
            return
        if not self.get_experimental_data():
            QMessageBox.warning(
                self,
                "Fragment Analysis",
                "No experimental peaks available (the original screening data). "
                "Load NMR-1D.csv or enter manual peaks.",
            )
            return
        ctx = self._ai_build_infer_structure_context(payload)
        self._ai_goto_assistant_tab()
        rank = ctx["candidate"]["screening_rank"]
        self._ai_append_chat(
            "Assistant",
            f"Inferring plausible structure(s) from selected candidate #{rank} "
            f"(SMILES analog + {len(ctx['per_carbon_table'])} carbons with attribution scores, "
            f"DEPT types and virtual shifts, plus {ctx['experimental_summary']['n_peaks']} "
            f"experimental peaks)."
        )
        self._ai_start_llm(
            "infer_structure",
            ctx,
            extra_user=(
                "Use the per-carbon attribution scores, virtual shifts vs experimental peaks, and "
                "DEPT types to propose the most plausible real structure(s) as valid SMILES. "
                "Treat the candidate SMILES as an analog clue, not the final answer."
            ),
        )

    def _ai_assign_signals_from_selected(self):
        """Score-based fragment extraction + signal assignment → structured LLM report."""
        if not has_fragment_data(self):
            QMessageBox.warning(
                self,
                "Fragment Analysis",
                "No masking data. Run Random masking attribution on the Fragment Analysis tab first, "
                "then select a candidate in the result table.",
            )
            return
        payload = self._ai_selected_masking_payload()
        if not payload:
            QMessageBox.warning(
                self,
                "Fragment Analysis",
                "Could not read the selected candidate. Select a row in the masking result table.",
            )
            return
        if not self.get_experimental_data():
            QMessageBox.warning(
                self,
                "Fragment Analysis",
                "No experimental peaks available (the original screening data). "
                "Load NMR-1D.csv or enter manual peaks.",
            )
            return
        ctx = self._ai_build_signal_assignment_context(payload)
        self._ai_goto_assistant_tab()
        rank = ctx["candidate"]["screening_rank"]
        n_frag = len(ctx.get("high_score_fragments") or [])
        n_unassigned = len(ctx.get("unassigned_exp_peaks") or [])
        self._ai_append_chat(
            "Assistant",
            f"Assigning ¹³C/DEPT signals for selected candidate #{rank} "
            f"({len(ctx['assignment_table'])} carbons, {n_frag} high-score fragment(s), "
            f"{n_unassigned} unassigned experimental peak(s))."
        )
        self._ai_start_llm(
            "signal_assignment",
            ctx,
            extra_user=(
                "Validate and structure the rule-based assignment_table. Rate each carbon's "
                "confidence from attribution_score, Δδ and DEPT agreement; interpret any "
                "unassigned_exp_peaks. Do not propose a new SMILES."
            ),
        )

    def _ai_fragment_evidence_review(self):
        """LLM review for extracted fragment evidence after Extract+map."""
        if not has_fragment_data(self):
            QMessageBox.warning(
                self,
                "Fragment Analysis",
                "No masking data. Run Random masking attribution first.",
            )
            return
        payload = self._ai_selected_masking_payload()
        if not payload:
            QMessageBox.warning(
                self,
                "Fragment Analysis",
                "Could not read the selected candidate. Select a row in the masking result table.",
            )
            return
        if not self.get_experimental_data():
            QMessageBox.warning(
                self,
                "Fragment Analysis",
                "No experimental peaks available. Load NMR-1D.csv or enter manual peaks.",
            )
            return
        if not list(getattr(self, "masking_fragments", []) or []):
            QMessageBox.warning(
                self,
                "Fragment Analysis",
                "No extracted fragments found for current candidate. "
                "Please click 'Extract positive fragments + map' first.",
            )
            return
        ctx = self._ai_build_fragment_review_context(payload)
        self._ai_goto_assistant_tab()
        self._ai_append_chat(
            "Assistant",
            f"Reviewing extracted fragment evidence for candidate #{ctx['candidate']['screening_rank']} "
            f"({len(ctx.get('fragment_rows') or [])} fragment(s), "
            f"{len(ctx.get('unassigned_exp_peaks') or [])} unassigned peak(s))."
        )
        self._ai_start_llm(
            "fragment_evidence_review",
            ctx,
            extra_user=(
                "Provide the six-section structured review in English only. Start with 'Likely "
                "structural motifs & compound class' (≤6 bullets: compound class, 2–5 substructure "
                "units with ppm+DEPT or fragment support, one line on overall features). Then rank "
                "fragments, redundancy vs complementarity, unassigned peaks, fusion-ready subset. "
                "Do not output new SMILES."
            ),
        )

    def _ai_build_global_fragment_review_context(self) -> dict:
        """Context for all-candidates fragment evidence review."""
        rows = list(getattr(self, "masking_fragments_global_rows", []) or [])
        exp = self.get_experimental_data() or []
        exp_peaks = [
            {"ppm": round(float(p.get("ppm", 0.0)), 2), "type": str(p.get("type", "")).lower()}
            for p in exp
        ]
        assigned_exp_idx = set()
        fragment_rows = []
        for rec in rows:
            frag = rec.get("frag") or {}
            cand_rank = int(rec.get("candidate_rank", 0))
            pairs = list(frag.get("pairs") or [])
            pair_examples = []
            for pr in pairs[:8]:
                exp_idx = int(pr.get("exp_index", -1))
                if exp_idx >= 0:
                    assigned_exp_idx.add(exp_idx)
                pair_examples.append({
                    "atom_index": pr.get("atom_index"),
                    "ctype": pr.get("ctype"),
                    "pred_ppm": round(float(pr.get("pred_ppm", 0.0)), 2),
                    "exp_ppm": round(float(pr.get("exp_ppm", 0.0)), 2),
                    "abs_err": round(float(pr.get("abs_err", 0.0)), 2),
                })
            enrich = self._ai_enrich_fragment_row_for_review(frag)
            fragment_rows.append({
                "candidate_rank": cand_rank,
                "fragment_label": f"C{cand_rank}-F{int(frag.get('fragment_id', rec.get('fragment_idx', 0) + 1))}",
                "fragment_id": int(frag.get("fragment_id", rec.get("fragment_idx", 0) + 1)),
                "fragment_smiles": frag.get("fragment_smiles") or frag.get("smiles"),
                "n_carbons": int(frag.get("n_carbons", 0)),
                "n_core_high_carbons": int(frag.get("n_core_high_carbons", frag.get("n_carbons", 0))),
                "score_sum": round(float(frag.get("score_sum", 0.0)), 4),
                "score_mean": round(float(frag.get("score_mean", 0.0)), 4),
                "hit_rate_unique_pct": round(100.0 * float(frag.get("hit_rate_unique", 0.0)), 2),
                "mean_abs_err": round(float(frag.get("mean_abs_err", 0.0)), 3),
                "match_mse": round(float(frag.get("match_mse", 0.0)), 4),
                "pair_examples": pair_examples,
                **enrich,
            })
        unassigned = [ep for j, ep in enumerate(exp_peaks) if j not in assigned_exp_idx]
        return {
            "workflow": {"mode": "fragment_evidence_review", "scope": "global_all_candidates", "analysis_done": True},
            "global_scope": {
                "n_fragments_total": len(fragment_rows),
                "n_candidates_with_fragments": len({int(r.get("candidate_rank", 0)) for r in fragment_rows}),
            },
            "fragment_rows": fragment_rows,
            "unassigned_exp_peaks": unassigned,
            "experimental_summary": {
                "n_peaks": len(exp_peaks),
                "carbon_type_counts": self._ai_count_carbon_types(exp_peaks),
            },
            "presentation": {
                "language": "en",
                "motif_section_first": True,
            },
            "constraints": {
                "no_new_smiles": True,
                "review_extracted_fragments_only": True,
                "global_scope_all_candidates": True,
            },
        }

    def _ai_global_fragment_evidence_review(self):
        """LLM review for all extracted fragments across all candidates."""
        rows = list(getattr(self, "masking_fragments_global_rows", []) or [])
        if not rows:
            QMessageBox.warning(
                self,
                "Fragment Analysis",
                "No global extracted fragments found. Please run 'Extract positive fragments + map' first.",
            )
            return
        if not self.get_experimental_data():
            QMessageBox.warning(
                self,
                "Fragment Analysis",
                "No experimental peaks available. Load NMR-1D.csv or enter manual peaks.",
            )
            return
        ctx = self._ai_build_global_fragment_review_context()
        self._ai_goto_assistant_tab()
        self._ai_append_chat(
            "Assistant",
            f"Reviewing global extracted fragments across candidates "
            f"({ctx['global_scope']['n_fragments_total']} fragment(s), "
            f"{ctx['global_scope']['n_candidates_with_fragments']} candidate(s))."
        )
        self._ai_start_llm(
            "fragment_evidence_review",
            ctx,
            extra_user=(
                "Global scope: compare fragments across all candidates. Six sections required; "
                "English only throughout. Section 1 (Likely structural motifs & compound class): "
                "≤6 concise bullets — plausible compound class, 2–5 likely substructures (e.g. "
                "β-D-glucopyranosyl, caffeoyl, aromatic ring) with supporting ppm+DEPT or "
                "high-confidence fragments, one line on overall structural features. Then global "
                "ranking, redundancy/complementarity across candidates, and a compact fusion-ready "
                "subset. Do not output new SMILES."
            ),
        )

    def _fusion_fragment_smiles(self, compound_idx: int, atom_indices) -> str:
        """Derive a fragment SMILES from its parent masking candidate + atom indices."""
        try:
            from rdkit import Chem
        except Exception:
            return ""
        results = getattr(self, "masking_all_results", None) or []
        if not (0 <= int(compound_idx) < len(results)):
            return ""
        parent = results[int(compound_idx)]
        smi = str(parent.get("smiles", "") or "").strip()
        atoms = sorted({int(x) for x in (atom_indices or [])})
        if not smi or not atoms:
            return ""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        try:
            return Chem.MolFragmentToSmiles(
                mol, atomsToUse=atoms, canonical=True, isomericSmiles=False
            )
        except Exception:
            return ""

    def _ai_fusion_motif_label(self, compound_idx: int, fragment_id: int) -> str:
        return f"C{int(compound_idx) + 1}-F{int(fragment_id)}"

    def _ai_fusion_fragment_motif_entry(self, compound_idx: int, frag: dict, pairs: list) -> dict:
        """One analog fragment as a spectral motif (no parent SMILES — avoids DB copy-paste)."""
        fid = int(frag.get("fragment_id", 0))
        ci = int(compound_idx)
        frag_pairs = [
            p for p in pairs
            if int(p.get("fragment_id", 0)) == fid
            and int(p.get("compound_idx", 0)) == ci
        ]
        peak_bits = [
            f"{p.get('ctype','?')} exp {p.get('exp_ppm')} (Δ{p.get('abs_err')})"
            for p in frag_pairs[:8]
        ]
        carbon_profile = self._ai_fragment_carbon_profile(frag_pairs)
        smi = self._fusion_fragment_smiles(ci, frag.get("atom_indices"))
        hints = {}
        if smi:
            try:
                from virmol_ai.context import _structure_facts
                hints = _structure_facts(smi)
                hints.pop("smiles", None)
            except Exception:
                hints = {}
        return {
            "label": self._ai_fusion_motif_label(ci, fid),
            "n_carbons": int(frag.get("n_carbons", 0)),
            "fragment_smiles": smi,
            "fragment_smiles_note": (
                "Carbon-skeleton approximation only; it MAY omit O/N/heteroatoms. "
                "Trust carbon_profile (ctype + exp_ppm) for each carbon's TRUE environment."
            ),
            "carbon_profile": carbon_profile,
            "substructure_hints": hints.get("substructure_hints") or [],
            "molecular_formula_fragment": hints.get("molecular_formula"),
            "matched_peak_summary": "; ".join(peak_bits) if peak_bits else "",
        }

    def _ai_enrich_fragment_row_for_review(self, frag: dict) -> dict:
        """Add carbon_profile and RDKit substructure_hints for fragment evidence review."""
        pairs = list(frag.get("pairs") or [])
        carbon_profile = self._ai_fragment_carbon_profile(pairs)
        smi = (frag.get("fragment_smiles") or frag.get("smiles") or "").strip()
        hints: list = []
        if smi:
            try:
                from virmol_ai.context import _structure_facts
                facts = _structure_facts(smi)
                hints = list(facts.get("substructure_hints") or [])[:8]
            except Exception:
                hints = []
        return {
            "carbon_profile": carbon_profile,
            "substructure_hints": hints,
            "fragment_smiles_note": (
                "Skeleton SMILES may omit heteroatoms; use carbon_profile (ctype + exp_ppm) "
                "for each carbon's environment."
            ),
        }

    @staticmethod
    def _ai_fragment_carbon_profile(frag_pairs: list) -> dict:
        """Per-fragment carbon-type profile derived from matched_pairs (authoritative).

        Returns DEPT-type counts plus the experimental ppm grouped by type, so the
        LLM does not have to reverse-engineer functional groups from a bare skeleton.
        """
        by_type: dict = {"s": [], "d": [], "t": [], "q": []}
        for p in frag_pairs:
            ct = str(p.get("ctype", "")).lower().strip()
            if ct not in by_type:
                continue
            try:
                by_type[ct].append(round(float(p.get("exp_ppm", 0.0)), 1))
            except (TypeError, ValueError):
                continue
        for ct in by_type:
            by_type[ct].sort(reverse=True)
        counts = {ct: len(vals) for ct, vals in by_type.items()}
        return {
            "carbon_type_counts": counts,
            "n_carbons_from_peaks": sum(counts.values()),
            "exp_ppm_by_type": by_type,
        }

    def _ai_build_fusion_top5_context(self, row: dict) -> Optional[dict]:
        """Context for fusion_infer_top5: motifs + spectrum; LLM assembles full structures."""
        exp = self.get_experimental_data() or []
        exp_peaks = [
            {"ppm": round(float(p.get("ppm", 0.0)), 3), "type": str(p.get("type", "")).lower()}
            for p in exp
        ]

        pairs = list(row.get("pairs", []))
        fragment_motifs = []
        for f in row.get("fragments", []):
            ci = int(f.get("compound_idx", 0))
            fragment_motifs.append(self._ai_fusion_fragment_motif_entry(ci, f, pairs))

        matched_pairs = []
        matched_exp_idx = set()
        for p in pairs:
            matched_exp_idx.add(int(p.get("exp_index", -1)))
            matched_pairs.append({
                "fragment_label": self._ai_fusion_motif_label(
                    int(p.get("compound_idx", 0)),
                    int(p.get("fragment_id", 0)),
                ),
                "ctype": str(p.get("ctype", "")).lower(),
                "pred_ppm": round(float(p.get("pred_ppm", 0.0)), 3),
                "exp_ppm": round(float(p.get("exp_ppm", 0.0)), 3),
                "abs_err": round(float(p.get("abs_err", 0.0)), 3),
            })

        unmatched = [ep for i, ep in enumerate(exp_peaks) if i not in matched_exp_idx]

        ctx = {
            "workflow": {
                "mode": "fusion_infer_top5",
                "task": "de_novo_assembly_from_fragment_motifs",
                "analysis_done": True,
            },
            "experimental_peaks": exp_peaks,
            "experimental_summary": {
                "n_carbons": len(exp_peaks),
                "carbon_type_counts": self._ai_count_carbon_types(exp_peaks),
            },
            "fusion_combination": {
                "combo_size": int(row.get("combo_size", 0)),
                "matched_carbons": int(row.get("na", 0)),
                "total_exp_peaks": int(row.get("nb", 0)),
                "coverage_pct": round(100.0 * float(row.get("coverage", 0.0)), 1),
                "aas_best": round(float(row.get("aas_best", 0.0)), 4),
                "score_final": round(float(row.get("score_final", 0.0)), 4),
                "fragment_motifs": fragment_motifs,
                "matched_pairs": matched_pairs,
            },
            "unmatched_exp_peaks": unmatched,
            "unmatched_summary": {
                "count": len(unmatched),
                "carbon_type_counts": self._ai_count_carbon_types(unmatched),
                "hint": "These peaks are NOT explained by the selected fragments. For these carbons "
                        "ONLY, infer the likely small group/linker from chemical experience — do not "
                        "introduce other fragments or other fusion combinations.",
            },
            "analysis_top_n": 5,
            "constraints": {
                "max_structures": 5,
                "rank_best_first": True,
                "no_padding_with_near_duplicates": True,
                "must_output_smiles": True,
                "de_novo_full_structures": True,
                "use_only_selected_fragment_motifs": True,
                "forbid_other_fragments_or_combinations": True,
                "match_carbon_budget_and_dept": True,
                "infer_unmatched_peaks_from_chemistry_knowledge": True,
            },
        }
        return ctx

    def _ai_build_fusion_review_context(self, rows: list, selected_idx: int) -> dict:
        """Context for reviewing fusion combination evidence (no structure generation)."""
        exp = self.get_experimental_data() or []
        exp_peaks = [
            {"ppm": round(float(p.get("ppm", 0.0)), 2), "type": str(p.get("type", "")).lower()}
            for p in exp
        ]
        fusion_rows = []
        covered_exp_idx = set()
        for i, r in enumerate(list(rows or [])[:12]):
            pairs = list(r.get("pairs") or [])
            pair_examples = []
            for pr in pairs[:14]:
                ei = int(pr.get("exp_index", -1))
                if ei >= 0:
                    covered_exp_idx.add(ei)
                pair_examples.append({
                    "fragment_id": int(pr.get("fragment_id", 0)),
                    "compound_idx": int(pr.get("compound_idx", 0)),
                    "ctype": str(pr.get("ctype", "")).lower(),
                    "pred_ppm": round(float(pr.get("pred_ppm", 0.0)), 2),
                    "exp_ppm": round(float(pr.get("exp_ppm", 0.0)), 2),
                    "abs_err": round(float(pr.get("abs_err", 0.0)), 2),
                })
            fusion_rows.append({
                "rank": i + 1,
                "combo_size": int(r.get("combo_size", 0)),
                "na": int(r.get("na", 0)),
                "nb": int(r.get("nb", 0)),
                "coverage_pct": round(100.0 * float(r.get("coverage", 0.0)), 2),
                "aas_best": round(float(r.get("aas_best", 0.0)), 4),
                "score_final": round(float(r.get("score_final", 0.0)), 4),
                "sse": round(float(r.get("sse", 0.0)), 4),
                "fragments": [
                    f"C{int(f.get('compound_idx', 0)) + 1}-F{int(f.get('fragment_id', 0))}"
                    for f in list(r.get("fragments") or [])
                ],
                "pair_examples": pair_examples,
            })
        selected = None
        if 0 <= int(selected_idx) < len(fusion_rows):
            selected = fusion_rows[int(selected_idx)]
        unmatched = [ep for j, ep in enumerate(exp_peaks) if j not in covered_exp_idx]
        return {
            "workflow": {"mode": "fusion_evidence_review", "analysis_done": True},
            "fusion_rows": fusion_rows,
            "selected_combination": selected,
            "experimental_summary": {
                "n_peaks": len(exp_peaks),
                "carbon_type_counts": self._ai_count_carbon_types(exp_peaks),
            },
            "unmatched_exp_peaks": unmatched,
            "constraints": {
                "no_new_smiles": True,
                "evidence_review_only": True,
            },
        }

    def _ai_compact_fusion_review_context(self, context: dict) -> dict:
        c = dict(context or {})
        rows = list(c.get("fusion_rows") or [])
        if len(rows) > 8:
            c["fusion_rows"] = rows[:8]
            c["fusion_rows_note"] = f"truncated to top 8 of {len(rows)}"
        for r in c.get("fusion_rows", []):
            ex = list(r.get("pair_examples") or [])
            if len(ex) > 8:
                r["pair_examples"] = ex[:8]
        up = list(c.get("unmatched_exp_peaks") or [])
        if len(up) > 20:
            c["unmatched_exp_peaks"] = up[:20]
        return c

    def _ai_fusion_infer_top5(self):
        """From the selected fusion combination + experimental peaks, infer Top-5 full structures."""
        if not getattr(self, "fusion_results", None):
            QMessageBox.information(
                self,
                "Fusion analysis",
                "No fusion results. Run Fragment fusion (or intra-molecular fusion) "
                "on the Fragment Analysis tab first, then select a combination row.",
            )
            return
        idx = self._current_fusion_row_index()
        if idx < 0 or idx >= len(self.fusion_results):
            QMessageBox.information(
                self,
                "Fusion analysis",
                "Select a fusion combination row in the Fragment fusion table first.",
            )
            return
        if not self.get_experimental_data():
            QMessageBox.warning(
                self,
                "Fusion analysis",
                "No experimental peaks available. Load NMR-1D.csv or enter manual peaks.",
            )
            return
        ctx = self._ai_build_fusion_top5_context(self.fusion_results[idx])
        if not ctx:
            QMessageBox.warning(self, "Fusion analysis", "Could not build fusion context.")
            return
        self._ai_goto_assistant_tab()
        self._ai_append_chat(
            "Assistant",
            f"Fusion → Top-5: de novo assembly for combination #{idx + 1} "
            f"(coverage {ctx['fusion_combination']['coverage_pct']}%, "
            f"AAS {ctx['fusion_combination']['aas_best']}, "
            f"{ctx['unmatched_summary']['count']} unmatched peak(s)). "
            f"The LLM will combine fragment motifs + experimental data — "
            f"not re-list database screening hits."
        )
        self._ai_start_llm(
            "fusion_infer_top5",
            ctx,
            extra_user=(
                "Propose exactly five NEW full structures (valid SMILES each) by chemically "
                "assembling the fragment_motifs and explaining unmatched_exp_peaks. "
                "Do NOT return parent database molecules or unchanged screening candidates. "
                "Rank by spectral/chemical plausibility, not screening score."
            ),
        )

    def _ai_fusion_evidence_review(self):
        """LLM review of fusion combination evidence after fusion analysis."""
        rows = list(getattr(self, "fusion_results", []) or [])
        if not rows:
            QMessageBox.information(
                self,
                "Fusion analysis",
                "No fusion results. Run fusion analysis on the Fragment Analysis tab first.",
            )
            return
        if not self.get_experimental_data():
            QMessageBox.warning(
                self,
                "Fusion analysis",
                "No experimental peaks available. Load NMR-1D.csv or enter manual peaks.",
            )
            return
        idx = self._current_fusion_row_index()
        if idx < 0:
            idx = 0
        ctx = self._ai_build_fusion_review_context(rows, idx)
        self._ai_goto_assistant_tab()
        self._ai_append_chat(
            "Assistant",
            f"Fusion evidence review: top {len(ctx.get('fusion_rows') or [])} combinations "
            f"(selected #{idx + 1}, unmatched peaks {len(ctx.get('unmatched_exp_peaks') or [])})."
        )
        self._ai_start_llm(
            "fusion_evidence_review",
            ctx,
            extra_user=(
                "Provide ONLY the five requested sections. Prioritize ranking rationale from "
                "coverage/AAS/score/sse, fragment-role interpretation, unresolved/conflicting "
                "signals, and verification-focused recommendations. Do not output new SMILES."
            ),
        )

    def _frag_mask_table_context_menu(self, pos):
        row = self.frag_mask_frag_table.rowAt(pos.y())
        menu = QMenu(self)
        if row >= 0:
            act = menu.addAction(f"AI: Explain fragment row {row + 1}")
            act.triggered.connect(lambda r=row: self._ai_fragment_explain_row(r))
        act_all = menu.addAction("AI: Current candidate attribution")
        act_all.triggered.connect(self._ai_fragment_narrate)
        menu.exec_(self.frag_mask_frag_table.viewport().mapToGlobal(pos))

    def _frag_fusion_table_context_menu(self, pos):
        row = self.frag_fusion_table.rowAt(pos.y())
        if row >= 0:
            self.frag_fusion_table.selectRow(row)
        menu = QMenu(self)
        act = menu.addAction("AI: Fusion → Top-5")
        act.setEnabled(bool(getattr(self, "fusion_results", None)))
        act.triggered.connect(self._ai_fusion_infer_top5)
        menu.exec_(self.frag_fusion_table.viewport().mapToGlobal(pos))

    def _ai_copy_fragment_summary(self):
        if not has_fragment_data(self):
            QMessageBox.warning(self, "Fragment Analysis", "No fragment data to export.")
            return
        frag = build_fragment_analysis_context(
            self, candidate_list_index=self._ai_fragment_list_index()
        )
        if not frag:
            QMessageBox.warning(self, "Fragment Analysis", "Could not build fragment summary.")
            return
        text = json.dumps(frag, ensure_ascii=False, indent=2)
        QApplication.clipboard().setText(text)
        self._ai_goto_assistant_tab()
        self._ai_append_chat("System", "Fragment summary JSON copied to clipboard.")
        self.statusBar().showMessage("Fragment summary copied")

    # ===== Chat-driven SOP pipeline (powers the AI Assistant tab) ==============

    @staticmethod
    def _ai_format_defaults_summary(defaults: dict) -> str:
        filters_on = [n for n, on in (
            ("CNF", defaults.get("use_cnf")),
            ("CTNF", defaults.get("use_ctnf")),
            ("MW", defaults.get("use_mw")),
        ) if on]
        bits = [
            f"db={str(defaults.get('database', 'plant')).capitalize()}",
            f"evaluator={defaults.get('evaluator', 'FPAACS')}",
            "filters=" + (", ".join(filters_on) if filters_on else "none"),
        ]
        if defaults.get("do_masking"):
            bits.append(f"masking Top-{defaults.get('masking_top_n', 5)}")
            if defaults.get("do_fusion"):
                bits.append("fragment fusion")
        if defaults.get("do_export"):
            bits.append("export")
        return "Defaults: " + " · ".join(bits)

    @staticmethod
    def _ai_set_combo_data(combo, target_data):
        for i in range(combo.count()):
            if combo.itemData(i) == target_data:
                combo.setCurrentIndex(i)
                return

    def _ai_effective_defaults(self) -> dict:
        """Merge class defaults with overrides set via the Advanced dialog."""
        eff = dict(self._SOP_DEFAULTS)
        eff.update(getattr(self, "_ai_sop_overrides", {}) or {})
        return eff

    def _ai_refresh_defaults_label(self):
        if hasattr(self, "ai_defaults_label"):
            self.ai_defaults_label.setText(
                self._ai_format_defaults_summary(self._ai_effective_defaults())
            )

    def _ai_show_welcome_message(self):
        if not hasattr(self, "ai_chat_view"):
            return
        self._ai_append_chat(
            "Assistant",
            "Hi! Paste a ¹³C peak list below and I'll run the full pipeline "
            "(screening → masking → fragment fusion → export) and report results here. "
            "Or just ask me a question and I'll use the cloud LLM.\n"
            + self._ai_format_defaults_summary(self._ai_effective_defaults())
        )

    def _ai_clear_conversation(self):
        if hasattr(self, "ai_chat_view"):
            self.ai_chat_view.clear()
        self._ai_show_welcome_message()

    def _ai_stop_all(self):
        """Stop both LLM streaming and any in-flight SOP pipeline."""
        self._ai_stop_llm()
        runner = getattr(self, "_ai_sop_runner", None)
        if runner is not None:
            runner.abort()
            self._ai_append_chat("System", "Pipeline abort requested.")

    def _ai_set_sop_running(self, running: bool):
        self._ai_sop_active = bool(running)
        if hasattr(self, "ai_send_btn"):
            self.ai_send_btn.setEnabled(not running)
            self.ai_send_btn.setText("Running…" if running else "Send")
        if hasattr(self, "ai_stop_btn"):
            self.ai_stop_btn.setEnabled(running)

    def eventFilter(self, obj, event):
        """Ctrl+Enter (or Cmd+Enter) in the chat input sends the message."""
        if (event.type() == QEvent.KeyPress
                and hasattr(self, "ai_user_input") and obj is self.ai_user_input):
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
                    self._ai_smart_send()
                    return True
        return super().eventFilter(obj, event)

    # ---------- Input handling ----------

    def _ai_open_peak_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open peak file", "", "CSV / Excel (*.csv *.xlsx *.xls);;All files (*)"
        )
        if not path:
            return
        from virmol_ai.intake import from_file
        report = from_file(path)
        if not report.ok:
            warns = "\n".join(f"- {w}" for w in (report.warnings or ["(unknown error)"]))
            QMessageBox.warning(self, "Open file",
                                "Could not parse this file.\n" + warns)
            return
        self.ai_user_input.setPlainText(report.text)
        self._ai_append_chat("System",
            f"Loaded {report.n_peaks} peaks from {path}. Click Send to run.")

    # Built-in example case shown via the "Example case" button under the
    # chat input.  Format is the same as what _ai_smart_send / intake expects.
    _AI_EXAMPLE_PEAKS = (
        "131.2, s\n"
        "116.3, d\n"
        "146.1, s\n"
        "144.7, s\n"
        "117.0, d\n"
        "121.1, d\n"
        "36.8, t\n"
        "72.6, t\n"
        "104.6, d\n"
        "75.0, d\n"
        "78.0, d\n"
        "72.0, d\n"
        "75.3, d\n"
        "64.7, t\n"
        "131.1, s\n"
        "141.6, d\n"
        "28.5, t\n"
        "45.4, d\n"
        "24.4, t\n"
        "26.3, t\n"
        "168.8, s\n"
        "72.9, s\n"
        "27.1, q\n"
        "26.4, q"
    )

    def _ai_load_example(self):
        """Fill the chat input with a built-in example peak list."""
        if not hasattr(self, "ai_user_input"):
            return
        self.ai_user_input.setPlainText(self._AI_EXAMPLE_PEAKS)
        try:
            self.ai_user_input.setFocus()
            cursor = self.ai_user_input.textCursor()
            cursor.movePosition(cursor.End)
            self.ai_user_input.setTextCursor(cursor)
        except Exception:
            pass
        self._ai_append_chat(
            "System",
            "Loaded a 24-peak example (¹³C NMR). Each line is\n"
            "  ppm, type   where type is one of s / d / t / q.\n"
            "Click Send to run the full analysis on this example."
        )

    def _ai_direct_mode_enabled(self) -> bool:
        return bool(
            hasattr(self, "ai_direct_llm_check")
            and self.ai_direct_llm_check.isChecked()
        )

    @staticmethod
    def _ai_peaks_text_to_context_peaks(text: str):
        peaks = []
        for raw in str(text or "").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                ppm = round(float(parts[0]), 2)
            except (TypeError, ValueError):
                continue
            ctype = str(parts[1]).strip().lower()
            if ctype not in ("q", "t", "d", "s"):
                continue
            peaks.append({"ppm": ppm, "type": ctype})
        return peaks

    @staticmethod
    def _ai_count_carbon_types(peaks):
        out = {"q": 0, "t": 0, "d": 0, "s": 0}
        for p in peaks or []:
            key = str(p.get("type", "")).lower().strip()
            if key in out:
                out[key] += 1
        return out

    def _ai_start_direct_top5_from_peaks(self, peak_text: str, source_label: str = ""):
        """Direct LLM mode: propose Top-5 SMILES from peaks only (no DB/SOP)."""
        peaks = self._ai_peaks_text_to_context_peaks(peak_text)
        if len(peaks) < 2:
            QMessageBox.warning(
                self,
                "AI Assistant",
                "Direct LLM mode needs at least 2 valid peaks in 'ppm, type' format.",
            )
            return
        summary = {
            "n_carbons": len(peaks),
            "carbon_type_counts": self._ai_count_carbon_types(peaks),
        }
        ctx = {
            "workflow": {
                "mode": "llm_direct_top5",
                "database_loaded": False,
                "analysis_done": False,
                "result_rows": 0,
            },
            "experimental_peaks": peaks,
            "experimental_summary": summary,
            "analysis_top_n": 5,
            "constraints": {
                "source": source_label or "manual peaks",
                "require_exactly_five": True,
                "must_output_smiles": True,
                "no_database_screening_used": True,
            },
        }
        self._ai_append_chat(
            "Assistant",
            "Direct LLM mode is ON: skipping database screening and generating "
            "Top-5 hypothetical SMILES from spectral evidence only."
        )
        self._ai_start_llm("direct_top5", ctx, extra_user="Return exactly five candidate SMILES.")

    def _ai_run_with_nmr_csv(self):
        import os as _os
        path = self._nmr_csv_path()
        if not _os.path.isfile(path):
            QMessageBox.warning(
                self, "NMR-1D.csv missing",
                f"No NMR-1D.csv at:\n{path}\n\n"
                "Finish spectrum preprocessing on the Molecular Analysis tab first, "
                "or paste a peak list above.",
            )
            return
        if getattr(self, "_ai_sop_active", False):
            return
        if self._ai_direct_mode_enabled():
            try:
                from virmol_ai.intake import from_file
                rpt = from_file(path)
            except Exception as exc:
                QMessageBox.warning(self, "AI Assistant", f"Could not read NMR-1D.csv: {exc}")
                return
            if not rpt.ok:
                QMessageBox.warning(
                    self,
                    "AI Assistant",
                    "Direct LLM mode could not parse NMR-1D.csv into peak lines.",
                )
                return
            self._ai_append_chat("You", f"(Direct LLM mode, from NMR-1D.csv: {path})")
            self._ai_start_direct_top5_from_peaks(rpt.text, source_label=f"NMR-1D.csv: {path}")
            return
        self._ai_append_chat("You", f"(Use the existing NMR-1D.csv at {path})")
        self._ai_start_sop_pipeline(peak_text="", source_label=f"NMR-1D.csv: {path}")

    def _ai_smart_send(self):
        """Smart entry: peak text → run SOP, otherwise → LLM wizard."""
        if getattr(self, "_ai_sop_active", False):
            return
        text = self.ai_user_input.toPlainText().strip()
        if not text:
            return
        # 1) Try to parse as a peak list
        try:
            from virmol_ai.intake import from_text
            report = from_text(text)
        except Exception:
            report = None
        if report is not None and report.ok and report.n_peaks >= 2:
            self._ai_append_chat("You", text)
            self.ai_user_input.clear()
            if self._ai_direct_mode_enabled():
                self._ai_append_chat(
                    "Assistant",
                    f"Parsed {report.n_peaks} peaks. Starting Direct LLM Top-5 generation…"
                )
                self._ai_start_direct_top5_from_peaks(
                    report.text,
                    source_label=f"{report.n_peaks} pasted peaks",
                )
                return
            self._ai_append_chat(
                "Assistant",
                f"Parsed {report.n_peaks} peaks. Starting full analysis…"
            )
            self._ai_start_sop_pipeline(
                peak_text=report.text,
                source_label=f"{report.n_peaks} pasted peaks",
            )
            return
        # 2) Otherwise treat as a free-form question for the LLM.
        #    Use the lightweight "chat" task so the model just answers the
        #    user instead of running the workflow-wizard template.
        self.ai_user_input.clear()
        self._ai_append_chat("You", text)
        ctx = self._ai_build_context()
        note = (self.ai_sample_note.text() if hasattr(self, "ai_sample_note") else "").strip()
        if note:
            ctx["sample_note"] = note
        if hasattr(self, "_ai_refresh_workflow_banner"):
            self._ai_refresh_workflow_banner()
        self._ai_start_llm("chat", ctx, extra_user=text)

    # ---------- SOP pipeline ----------

    def _ai_build_sop_payload(self, peak_text: str):
        from virmol_ai.sop import IntakePayload
        d = self._ai_effective_defaults()
        return IntakePayload(
            peak_text=peak_text,
            database=d["database"],
            use_cnf=bool(d["use_cnf"]),
            use_ctnf=bool(d["use_ctnf"]),
            use_mw=bool(d["use_mw"]),
            evaluator=d["evaluator"],
            masking_top_n=int(d["masking_top_n"]),
            do_masking=bool(d["do_masking"]),
            do_fragment_extraction=bool(d["do_masking"]),
            do_fusion=bool(d["do_fusion"]),
            do_export=bool(d["do_export"]),
            sample_note=str(d.get("sample_note", "")).strip(),
        )

    def _ai_start_sop_pipeline(self, peak_text: str, source_label: str = ""):
        from virmol_ai.sop import build_plan
        from virmol_ai.runner import SOPRunner
        d = self._ai_effective_defaults()
        payload = self._ai_build_sop_payload(peak_text)
        try:
            plan = build_plan(d["sop_id"], payload)
        except Exception as exc:
            self._ai_append_chat("System", f"Plan generation failed: {exc}")
            return
        self._ai_sop_plan = plan
        suffix = f" (from {source_label})" if source_label else ""
        self._ai_append_chat(
            "System",
            f"Plan{suffix}: {plan.name} — {len(plan.steps)} steps · "
            f"ETA ~{plan.total_eta_seconds:.0f}s"
        )
        for i, step in enumerate(plan.steps):
            self._ai_append_chat("System", f"   {i + 1}. {step.label}")

        runner = SOPRunner(
            self,
            plan,
            on_step_start=self._ai_sop_on_step_start,
            on_step_done=self._ai_sop_on_step_done,
            on_plan_done=self._ai_sop_on_plan_done,
            on_log=self._ai_sop_on_log,
        )
        self._ai_sop_runner = runner
        self._ai_set_sop_running(True)
        runner.run_all()

    def _ai_sop_on_step_start(self, rec):
        total = len(self._ai_sop_plan.steps) if self._ai_sop_plan else "?"
        self._ai_append_chat(
            "Assistant",
            f"▶ Step {rec.index + 1}/{total}: {rec.step.label}"
        )
        self.statusBar().showMessage(f"AI pipeline: {rec.step.label}")

    def _ai_sop_on_step_done(self, rec):
        icon = {"success": "✓", "failed": "✗", "skipped": "→", "aborted": "✗"}.get(
            rec.status, "?"
        )
        summary = rec.result.summary if rec.result is not None else rec.status
        self._ai_append_chat("Assistant", f"   {icon} {summary}")

    def _ai_sop_on_plan_done(self, ok, records):
        self._ai_set_sop_running(False)
        self._ai_sop_runner = None
        n_done = sum(1 for r in records if r.status == "success")
        n_fail = sum(1 for r in records if r.status == "failed")
        msg = f"Pipeline finished: {n_done}/{len(records)} succeeded"
        if n_fail:
            msg += f", {n_fail} failed"
        self._ai_append_chat("Assistant", msg + ".")
        self.statusBar().showMessage(msg)
        if ok:
            self._ai_append_chat(
                "Assistant",
                "Results are now on the Molecular Analysis and Fragment Analysis tabs."
            )
            # Inline Top-5 structures + fragments overview right inside the chat
            try:
                self._ai_show_top5_structures(silent=True)
            except Exception as exc:
                self._ai_append_chat("System", f"Top-5 structure render skipped: {exc}")
            try:
                self._ai_show_fragments_overview(silent=True)
            except Exception as exc:
                self._ai_append_chat("System", f"Fragment overview skipped: {exc}")
            # Auto-run the cloud LLM Top-5 interpretation if configured
            try:
                cfg = load_llm_config()
                if is_llm_configured(cfg):
                    self._ai_append_chat(
                        "Assistant",
                        "Asking the cloud LLM for a Top-5 interpretation…"
                    )
                    QTimer.singleShot(0, lambda: self._ai_run_task("topn", top_n=5))
            except Exception:
                pass

    def _ai_sop_on_log(self, level: str, message: str):
        if not message:
            return
        if level in ("warn", "error"):
            prefix = "⚠ " if level == "warn" else "✗ "
            self._ai_append_chat("System", prefix + str(message))

    # ---------- Advanced settings dialog ----------

    def _ai_show_advanced_dialog(self):
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox
        from virmol_ai.sop import list_sops

        d = self._ai_effective_defaults()
        dlg = QDialog(self)
        dlg.setWindowTitle("AI Assistant — Advanced settings")
        dlg.setMinimumWidth(420)
        grid = QGridLayout(dlg)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        r = 0
        grid.addWidget(QLabel("SOP:"), r, 0)
        sop_cb = QComboBox()
        for e in list_sops():
            sop_cb.addItem(e["label"], e["id"])
        self._ai_set_combo_data(sop_cb, d["sop_id"])
        grid.addWidget(sop_cb, r, 1, 1, 2)
        r += 1

        grid.addWidget(QLabel("Database:"), r, 0)
        db_cb = QComboBox()
        for name in ("plant", "human", "microbial", "drug", "all"):
            db_cb.addItem(name.capitalize(), name)
        self._ai_set_combo_data(db_cb, d["database"])
        grid.addWidget(db_cb, r, 1, 1, 2)
        r += 1

        grid.addWidget(QLabel("Evaluator:"), r, 0)
        ev_cb = QComboBox()
        for name in ("FPAACS", "CSS", "AAS", "FPS"):
            ev_cb.addItem(name, name)
        self._ai_set_combo_data(ev_cb, d["evaluator"])
        grid.addWidget(ev_cb, r, 1, 1, 2)
        r += 1

        cnf = QCheckBox("CNF"); cnf.setChecked(bool(d["use_cnf"]))
        ctnf = QCheckBox("CTNF"); ctnf.setChecked(bool(d["use_ctnf"]))
        mw = QCheckBox("MW"); mw.setChecked(bool(d["use_mw"]))
        grid.addWidget(cnf, r, 0); grid.addWidget(ctnf, r, 1); grid.addWidget(mw, r, 2)
        r += 1

        grid.addWidget(QLabel("Masking Top-N:"), r, 0)
        topn = QSpinBox(); topn.setRange(1, 50); topn.setValue(int(d["masking_top_n"]))
        grid.addWidget(topn, r, 1, 1, 2)
        r += 1

        msk = QCheckBox("Run masking attribution"); msk.setChecked(bool(d["do_masking"]))
        fus = QCheckBox("Run fragment fusion"); fus.setChecked(bool(d["do_fusion"]))
        exp = QCheckBox("Export final results folder"); exp.setChecked(bool(d["do_export"]))
        grid.addWidget(msk, r, 0, 1, 3); r += 1
        grid.addWidget(fus, r, 0, 1, 3); r += 1
        grid.addWidget(exp, r, 0, 1, 3); r += 1

        grid.addWidget(QLabel("Sample note (for report):"), r, 0, 1, 3); r += 1
        note = QLineEdit()
        note.setText(str(d.get("sample_note", "")))
        note.setPlaceholderText("e.g. EtOAc fraction 3, plant X leaves")
        grid.addWidget(note, r, 0, 1, 3); r += 1

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Reset
        )
        grid.addWidget(buttons, r, 0, 1, 3)

        def _reset_dialog_to_class_defaults():
            base = self._SOP_DEFAULTS
            self._ai_set_combo_data(sop_cb, base["sop_id"])
            self._ai_set_combo_data(db_cb, base["database"])
            self._ai_set_combo_data(ev_cb, base["evaluator"])
            cnf.setChecked(base["use_cnf"])
            ctnf.setChecked(base["use_ctnf"])
            mw.setChecked(base["use_mw"])
            topn.setValue(int(base["masking_top_n"]))
            msk.setChecked(base["do_masking"])
            fus.setChecked(base["do_fusion"])
            exp.setChecked(base["do_export"])
            note.clear()
        buttons.button(QDialogButtonBox.Reset).clicked.connect(_reset_dialog_to_class_defaults)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec_() == QDialog.Accepted:
            self._ai_sop_overrides = {
                "sop_id": sop_cb.currentData(),
                "database": db_cb.currentData(),
                "evaluator": ev_cb.currentData(),
                "use_cnf": cnf.isChecked(),
                "use_ctnf": ctnf.isChecked(),
                "use_mw": mw.isChecked(),
                "masking_top_n": topn.value(),
                "do_masking": msk.isChecked(),
                "do_fusion": fus.isChecked(),
                "do_export": exp.isChecked(),
                "sample_note": note.text().strip(),
            }
            self._ai_refresh_defaults_label()
            self._ai_append_chat(
                "System",
                self._ai_format_defaults_summary(self._ai_effective_defaults())
            )

    def _shutdown_worker_threads(self):
        """Stop background worker threads before window closes."""
        self._release_analysis_thread(wait_ms=5000)
        for name in ("fragment_thread", "masking_thread", "db_thread", "llm_thread"):
            th = getattr(self, name, None)
            if th is None:
                continue
            if isinstance(th, QThread) and th.isRunning():
                try:
                    th.requestInterruption()
                except Exception:
                    pass
                try:
                    th.quit()
                except Exception:
                    pass
                if not th.wait(2000):
                    try:
                        th.terminate()
                    except Exception:
                        pass
                    th.wait(1000)
            setattr(self, name, None)

    def _is_thread_running(self, name):
        th = getattr(self, name, None)
        return isinstance(th, QThread) and th.isRunning()

    def _release_analysis_thread(self, wait_ms=15000):
        """Wait for the worker to exit, then delete the QThread safely."""
        th = getattr(self, "analysis_thread", None)
        if th is None:
            return
        for sig in (getattr(th, "progress", None), getattr(th, "result_ready", None), getattr(th, "error", None)):
            if sig is None:
                continue
            try:
                sig.disconnect()
            except Exception:
                pass
        if th.isRunning():
            th.wait(wait_ms)
        th.deleteLater()
        self.analysis_thread = None

    def closeEvent(self, event):
        self._shutdown_worker_threads()
        super().closeEvent(event)



if __name__ == "__main__":
    # Run GUI directly without additional script
    try:
        _configure_runtime_paths()
        app = QApplication(sys.argv)

        app_icon = _load_app_icon()
        if not app_icon.isNull():
            app.setWindowIcon(app_icon)
        
        # Set application information
        app.setApplicationName("VirMolAnalyte")
        app.setApplicationVersion("2.0")
        app.setOrganizationName("VirMolAnalyte Team")
    
        # Set application font
        font = app.font()
        font.setPointSize(12)  # Increase default font size
        app.setFont(font)
        
        # Create and show splash screen
        splash = VirMolSplashScreen()
        if not app_icon.isNull():
            splash.setWindowIcon(app_icon)
        splash.show()
        
        # Process events to show splash screen
        app.processEvents()
        
        # Simulate loading time
        import time
        time.sleep(2)  # Show splash for 2 seconds
        
        # Create main window
        window = ModernVirMolAnalyteGUI()
        
        # Close splash screen and show main window
        splash.finish(window)
        window.show()
        
        # Run application
        sys.exit(app.exec())

    except Exception as e:
        print(f"Startup failed: {e}")
        print("Please check environment configuration and dependencies")
        input("Press Enter to exit...")
