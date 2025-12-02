import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QDoubleSpinBox, QSlider,
                             QGroupBox, QComboBox, QFormLayout, QSizePolicy, QLineEdit)
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QPixmap

# QApplicationの作成前に高DPIスケーリングを有効化
QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

class CircuitCalculator:
    """回路計算ロジッククラス"""
    @staticmethod
    def calculate_gain(R1, R2, R3, R4, R5, RL, RC, RJ, A):
        try:
            # RCとRJをmΩからΩに変換
            RC = RC / 1000.0
            RJ = RJ / 1000.0

            # AをdBから線形値に変換
            A = 10 ** (A / 20.0)

            # ゼロ除算回避
            if (R1 + R2) == 0 or (R3 + R4) == 0 or R5 == 0 or RC == 0:
                return 0.0

            # 1. 分圧比 alpha, beta
            alpha = R3 / (R3 + R4)
            beta = R1 / (R1 + R2)

            # 2. RT, RF, Gamma
            # 1/RT = 2/(R3+R4) + 1/R5
            inv_RT = (2 / (R3 + R4)) + (1 / R5)
            if inv_RT == 0: return 0.0
            RT = 1 / inv_RT

            RF = RT + RJ
            if RF == 0: return 0.0
            Gamma = RT / RF

            # 3. 係数 K
            K = 2 + RL * ((1 / RC) + (1 / RF))

            # 4. メインの計算 (V2-VC)/Vin
            # 分子: Num = A * beta * [ A * alpha * Gamma - (1 + A * beta) ]
            term_common = 1 + A * beta
            term_A_alpha_Gamma = A * alpha * Gamma

            numerator = A * beta * (term_A_alpha_Gamma - term_common)

            # 分母: Den = (1 + A * beta) * [ (1 + A * beta) * K - 2 * A * alpha * Gamma ]
            denominator = term_common * (term_common * K - 2 * term_A_alpha_Gamma)

            if denominator == 0:
                return 0.0

            return numerator / denominator
        except Exception:
            return 0.0

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("クロストーク除去回路理論解析")
        self.resize(1200, 850)

        # パラメータの初期値と設定範囲
        self.params = {
            'A':  {'val': 40.0, 'label': 'Gain A (dB)'},
            'R1': {'val': 1000.0,  'label': 'R1 (Ω)'},
            'R2': {'val': 1000.0, 'label': 'R2 (Ω)'},
            'R3': {'val': 1000.0,  'label': 'R3 (Ω)'},
            'R4': {'val': 1000.0, 'label': 'R4 (Ω)'},
            'R5': {'val': 100.0,  'label': 'R5 (Ω)'},
            'RC': {'val': 100.0,   'label': 'RC (mΩ)'},
            'RJ': {'val': 100.0,    'label': 'RJ (mΩ)'},
            'RL': {'val': 10.0,     'label': 'RL (Ω)'},
        }

        self.controls = {} # ウィジェットへの参照を保持

        # メインウィジェット
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- 左側: コントロールパネル ---
        control_panel = QGroupBox("パラメータ設定")
        control_panel.setFixedWidth(400)
        control_layout = QVBoxLayout(control_panel)

        # 各パラメータの入力フィールド生成
        form_layout = QFormLayout()
        for key, info in self.params.items():
            # テキストボックス (指数表記対応)
            textbox = QLineEdit()
            textbox.setText(str(info['val']))

            # 変更時に再計算をトリガー
            textbox.textChanged.connect(self.update_calculation)

            self.controls[key] = textbox
            form_layout.addRow(info['label'], textbox)

        control_layout.addLayout(form_layout)

        # 入力エリア直下に回路図画像を表示
        image_path = os.path.join(os.path.dirname(__file__), 'sch.png')
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(150)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        if os.path.exists(image_path):
            self._orig_pixmap = QPixmap(image_path)
            if not self._orig_pixmap.isNull():
                scaled = self._orig_pixmap.scaled(360, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled)
            else:
                self._orig_pixmap = None
                self.image_label.setText('sch.png の読み込みに失敗しました')
        else:
            self._orig_pixmap = None
            self.image_label.setText('sch.png が見つかりません（このファイルと同じフォルダに置いてください）')

        control_layout.addWidget(self.image_label)

        # スイープ設定
        sweep_group = QGroupBox("グラフ設定 (パラメータスイープ)")
        sweep_layout = QVBoxLayout(sweep_group)

        self.combo_sweep = QComboBox()
        self.combo_sweep.addItems(list(self.params.keys()))
        self.combo_sweep.currentTextChanged.connect(self.update_calculation)

        sweep_layout.addWidget(QLabel("横軸にするパラメータ:"))
        sweep_layout.addWidget(self.combo_sweep)

        control_layout.addWidget(sweep_group)
        control_layout.addStretch() # 下部の余白

        # --- 右側: 結果表示とグラフ ---
        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)

        # 結果数値表示エリア
        self.label_result_lin = QLabel("Linear: --- V/V")
        self.label_result_lin.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        self.label_result_db = QLabel("dB: --- dB")
        self.label_result_db.setStyleSheet("font-size: 18px; font-weight: bold; color: #0066cc;")

        result_box = QGroupBox("計算結果 (V2 - VC) / Vin")
        result_box_layout = QVBoxLayout(result_box)
        result_box_layout.addWidget(self.label_result_lin)
        result_box_layout.addWidget(self.label_result_db)

        result_layout.addWidget(result_box)

        # Matplotlib グラフエリア
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        graph_box = QGroupBox("パラメータ感度解析")
        graph_box_layout = QVBoxLayout(graph_box)
        graph_box_layout.addWidget(self.canvas)

        result_layout.addWidget(graph_box)

        # レイアウトへの追加
        main_layout.addWidget(control_panel)
        main_layout.addWidget(result_panel)

        # 初回計算
        self.update_calculation()

    def get_current_params(self):
        """現在のGUI入力値を取得"""
        values = {}
        for key, widget in self.controls.items():
            try:
                # QLineEdit から文字列を取得し float に変換（指数表記可）
                values[key] = float(widget.text())
            except Exception:
                # 変換に失敗した場合は初期値を使用
                values[key] = self.params[key]['val']
        return values

    def update_calculation(self):
        """計算を実行し、UIを更新する"""
        p = self.get_current_params()

        # 1. 現在値の計算
        val = CircuitCalculator.calculate_gain(
            p['R1'], p['R2'], p['R3'], p['R4'], p['R5'],
            p['RL'], p['RC'], p['RJ'], p['A']
        )

        # 結果ラベル更新
        self.label_result_lin.setText(f"Linear: {val:.6f} V/V")
        if val != 0:
            db_val = 20 * np.log10(abs(val))
            self.label_result_db.setText(f"dB: {db_val:.4f} dB")
        else:
            self.label_result_db.setText("dB: -∞ dB")

        # 2. グラフの更新 (スイープ解析)
        self.update_plot(p, val)

    def update_plot(self, current_params, current_val):
        """グラフを描画"""
        target_param = self.combo_sweep.currentText()
        center_val = current_params[target_param]

        # スイープ範囲: 中心値の 10% ～ 200% (線形スイープ)
        # パラメータによっては0を含めないように注意
        start_val = center_val * 0.5
        end_val = center_val * 1.5
        if start_val <= 0 and target_param != 'RJ': start_val = 1e-3 # RJ以外は正の値と仮定

        steps = 100
        x_values = np.linspace(start_val, end_val, steps)
        y_values_db = []

        # パラメータを変化させて計算
        for x in x_values:
            temp_params = current_params.copy()
            temp_params[target_param] = x

            v = CircuitCalculator.calculate_gain(
                temp_params['R1'], temp_params['R2'], temp_params['R3'], temp_params['R4'],
                temp_params['R5'], temp_params['RL'], temp_params['RC'], temp_params['RJ'],
                temp_params['A']
            )

            if v != 0:
                y_values_db.append(20 * np.log10(abs(v)))
            else:
                y_values_db.append(np.nan) # プロットしない

        # Matplotlib描画処理
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.plot(x_values, y_values_db, label='Gain (dB)', color='#1f77b4', linewidth=2)

        # 現在値のポイント
        if current_val != 0:
            current_db = 20 * np.log10(abs(current_val))
            ax.scatter([center_val], [current_db], color='red', zorder=5, label='Current Point')
            ax.text(center_val, current_db, f' {current_db:.2f}dB', color='red', verticalalignment='bottom')

        ax.set_title(f"Gain vs {target_param}")
        ax.set_xlabel(f"{target_param} Value")
        ax.set_ylabel("Gain (dB)")
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend()

        self.figure.tight_layout()
        self.canvas.draw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, '_orig_pixmap', None) and self.image_label is not None:
            if not self._orig_pixmap.isNull():
                target_size = self.image_label.size()
                if target_size.width() > 0 and target_size.height() > 0:
                    scaled = self._orig_pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.image_label.setPixmap(scaled)

if __name__ == '__main__':
    # 上で QCoreApplication に対して属性を設定済み
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())