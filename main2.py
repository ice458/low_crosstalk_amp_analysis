import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QGroupBox, QFormLayout,
                             QSizePolicy, QLineEdit, QComboBox)
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QPixmap

# QApplicationの作成前に高DPIスケーリングを有効化
QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

class ComplexCircuitCalculator:
    """複素インピーダンスを含む回路計算ロジック"""
    @staticmethod
    def calculate_frequency_response(params, frequencies):
        # 周波数ベクトル (Hz -> rad/s)
        w = 2 * np.pi * frequencies
        s = 1j * w

        # パラメータ展開
        R1, C1 = params['R1'], params['C1']
        R2, C2 = params['R2'], params['C2']
        R3, C3 = params['R3'], params['C3']
        R4, C4 = params['R4'], params['C4']
        R5 = params['R5']
        RC = params['RC']
        RJ = params['RJ']
        RL = params['RL']
        A_db  = params['A']
        A_GBW = params.get('A_GBW', 0.0)
        # A は dB 入力。GBW>0 なら単一極A(s)に、そうでなければ定数ゲイン。
        A0 = 10 ** (A_db / 20.0)
        if A_GBW and A_GBW > 0:
            # fp = GBW / A0, wp = 2π fp
            fp = A_GBW / A0
            wp = 2 * np.pi * fp
            # ベクトルsに対してブロードキャストされる複素ゲイン
            A = A0 / (1 + s / wp)
        else:
            A = A0

        # 1. インピーダンス Z1~Z4 の計算 (Z = R // C)
        # Z = R / (1 + sCR)
        Z1 = R1 / (1 + s * C1 * R1)
        Z2 = R2 / (1 + s * C2 * R2)
        Z3 = R3 / (1 + s * C3 * R3)
        Z4 = R4 / (1 + s * C4 * R4)

        # 2. 分圧比 alpha, beta (複素数)
        # alpha = Z3 / (Z3 + Z4)
        denom_alpha = Z3 + Z4
        alpha = np.divide(Z3, denom_alpha, out=np.zeros_like(s), where=denom_alpha!=0)

        # beta = Z1 / (Z1 + Z2)
        denom_beta = Z1 + Z2
        beta = np.divide(Z1, denom_beta, out=np.zeros_like(s), where=denom_beta!=0)

        # 3. ZT, ZF, Gamma
        # 1/ZT = 2/(Z3+Z4) + 1/R5
        inv_ZT = (2 / (Z3 + Z4)) + (1 / R5)
        ZT = np.divide(1, inv_ZT, out=np.zeros_like(s), where=inv_ZT!=0)

        ZF = ZT + RJ

        # Gamma = ZT / ZF
        Gamma = np.divide(ZT, ZF, out=np.zeros_like(s), where=ZF!=0)

        # 4. 係数 K
        # K = 2 + RL * (1/RC + 1/ZF)
        K = 2 + RL * ((1 / RC) + np.divide(1, ZF, out=np.zeros_like(s), where=ZF!=0))

        # 5. 伝達関数 (V2-VC)/Vin
        # Numerator
        term_common = 1 + A * beta
        term_A_alpha_Gamma = A * alpha * Gamma
        numerator = A * beta * (term_A_alpha_Gamma - term_common)

        # Denominator
        denominator = term_common * (term_common * K - 2 * term_A_alpha_Gamma)

        # Result
        H = np.divide(numerator, denominator, out=np.zeros_like(s), where=denominator!=0)

        return H

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("周波数特性解析 (R+C並列モデル)")
        self.resize(1200, 880)

        # パラメータ初期値
        self.params = {
            'A':  {'val': 120.0, 'scale': 1.0, 'label': 'Gain A (dB)'},
            'A_GBW': {'val': 0.0, 'scale': 1e6, 'label': 'A GBW (MHz)'},
            'R1': {'val': 1000.0,  'scale': 1.0, 'label': 'R1 (Ω)'},
            'C1': {'val': 0.0,     'scale': 1e-12, 'label': 'C1 (pF)'},
            'R2': {'val': 1000.0,  'scale': 1.0, 'label': 'R2 (Ω)'},
            'C2': {'val': 10.0,     'scale': 1e-12, 'label': 'C2 (pF)'},
            'R3': {'val': 1000.0,  'scale': 1.0, 'label': 'R3 (Ω)'},
            'C3': {'val': 0.0,     'scale': 1e-12, 'label': 'C3 (pF)'},
            'R4': {'val': 1000.0,  'scale': 1.0, 'label': 'R4 (Ω)'},
            'C4': {'val': 10.0,     'scale': 1e-12, 'label': 'C4 (pF)'},
            'R5': {'val': 100.0,  'scale': 1.0, 'label': 'R5 (Ω)'},
            'RC': {'val': 10.0,   'scale': 1e-3, 'label': 'RC (mΩ)'},
            'RJ': {'val': 10.0,    'scale': 1e-3, 'label': 'RJ (mΩ)'},
            'RL': {'val': 10.0,     'scale': 1.0, 'label': 'RL (Ω)'},
        }

        self.controls = {}

        # メインウィジェット
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- 左側: コントロールパネル ---
        control_panel = QGroupBox("回路定数設定")
        control_panel.setFixedWidth(400)
        control_layout = QVBoxLayout(control_panel)

        # 各パラメータの入力フィールド生成
        form_layout = QFormLayout()
        for key, info in self.params.items():
            # テキストボックス
            textbox = QLineEdit()
            textbox.setText(str(info['val']))

            # 変更時に再計算をトリガー
            textbox.textChanged.connect(self.update_plot)

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
            self.image_label.setText('sch.png が見つかりません')

        control_layout.addWidget(self.image_label)

        # スイープ設定
        sweep_group = QGroupBox("グラフ設定 (パラメータスイープ)")
        sweep_layout = QVBoxLayout(sweep_group)

        self.combo_sweep = QComboBox()
        self.combo_sweep.addItems(list(self.params.keys()))
        self.combo_sweep.currentTextChanged.connect(self.update_plot)
        sweep_layout.addWidget(QLabel("スイープするパラメータ:"))
        sweep_layout.addWidget(self.combo_sweep)

        self.sweep_percent = QLineEdit()
        self.sweep_percent.setText("50")  # デフォルト±50%
        self.sweep_percent.textChanged.connect(self.update_plot)
        sweep_layout.addWidget(QLabel("スイープ範囲 (±%):"))
        sweep_layout.addWidget(self.sweep_percent)

        control_layout.addWidget(sweep_group)
        control_layout.addStretch() # 下部の余白

        # --- 右側: 結果表示とグラフ ---
        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)

        # 結果数値表示エリア
        self.dc_label = QLabel("Low Freq Gain (1Hz): --- dB")
        self.dc_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")

        result_box = QGroupBox("計算結果")
        result_box_layout = QVBoxLayout(result_box)
        result_box_layout.addWidget(self.dc_label)

        result_layout.addWidget(result_box)

        # Matplotlib グラフエリア
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        graph_box = QGroupBox("周波数特性 (Bode Plot)")
        graph_box_layout = QVBoxLayout(graph_box)
        graph_box_layout.addWidget(self.canvas)

        result_layout.addWidget(graph_box)

        # レイアウトへの追加
        main_layout.addWidget(control_panel)
        main_layout.addWidget(result_panel)

        # 初回計算
        self.update_plot()

    def get_current_params(self):
        """現在のGUI入力値を取得"""
        values = {}
        for key, widget in self.controls.items():
            try:
                # QLineEdit から文字列を取得し float に変換
                val = float(widget.text())
                # スケールを適用 (例: pF -> F)
                values[key] = val * self.params[key]['scale']
            except Exception:
                # 変換に失敗した場合は初期値を使用
                values[key] = self.params[key]['val'] * self.params[key]['scale']
        return values

    def update_plot(self):
        """グラフを描画"""
        params = self.get_current_params()

        # 周波数範囲 (1Hz ~ 100MHz) 対数スケール
        freqs = np.logspace(0, 8, 500)

        # プロット
        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)

        # スイープ設定の取得
        target_param = getattr(self, 'combo_sweep', None).currentText() if hasattr(self, 'combo_sweep') else None
        percent_text = getattr(self, 'sweep_percent', None).text().strip() if hasattr(self, 'sweep_percent') else "0"
        try:
            if percent_text.endswith('%'):
                p = float(percent_text[:-1]) / 100.0
            else:
                p = float(percent_text) / 100.0
        except Exception:
            p = 0.0
        if p < 0:
            p = 0.0
        if p > 10:
            p = 10.0

        # まず現在値の応答を計算
        H_cur = ComplexCircuitCalculator.calculate_frequency_response(params, freqs)
        mag_db_cur = 20 * np.log10(np.abs(H_cur) + 1e-20)
        phase_deg_cur = np.angle(H_cur, deg=True)

        # スイープ曲線を先に薄色で描画
        if target_param is not None and target_param in params and p > 0:
            center_val = params[target_param]
            start_val = center_val * (1.0 - p)
            end_val = center_val * (1.0 + p)
            if target_param in {'R1','R2','R3','R4','R5','RC','RJ','RL','C1','C2','C3','C4','A_GBW'} and start_val <= 0:
                start_val = max(center_val * 0.1, 1e-24)

            n_curves = 100
            sweep_vals = np.linspace(start_val, end_val, n_curves)

            for x in sweep_vals:
                if np.isclose(x, center_val):
                    continue
                temp_params = dict(params)
                temp_params[target_param] = x
                H_sw = ComplexCircuitCalculator.calculate_frequency_response(temp_params, freqs)
                mag_db_sw = 20 * np.log10(np.abs(H_sw) + 1e-20)
                phase_deg_sw = np.angle(H_sw, deg=True)
                ax1.semilogx(freqs, mag_db_sw, color='#1f77b4', alpha=0.18, linewidth=1)
                ax2.semilogx(freqs, phase_deg_sw, color='#d62728', alpha=0.18, linewidth=1)

        # 現在値を強調表示
        ax1.semilogx(freqs, mag_db_cur, color='#1f77b4', linewidth=2, label='Current Magnitude')
        ax2.semilogx(freqs, phase_deg_cur, color='#d62728', linewidth=2, label='Current Phase')

        # 軸やタイトル
        sweep_note = ""
        if p > 0 and target_param:
            sweep_note = f"  [Sweep: {target_param} ±{int(p*100)}%]"
        ax1.set_title('Bode Plot: (V2 - VC) / Vin' + sweep_note)
        ax1.set_ylabel('Magnitude (dB)')
        ax1.grid(True, which="both", linestyle='--', alpha=0.6)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (deg)')
        ax2.grid(True, which="both", linestyle='--', alpha=0.6)

        self.figure.tight_layout()
        self.canvas.draw()

        # 低周波(1Hz)での値を表示（現在値）
        dc_val_db = mag_db_cur[0]
        self.dc_label.setText(f"Low Freq Gain (1Hz): {dc_val_db:.4f} dB")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, '_orig_pixmap', None) and self.image_label is not None:
            if not self._orig_pixmap.isNull():
                target_size = self.image_label.size()
                if target_size.width() > 0 and target_size.height() > 0:
                    scaled = self._orig_pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.image_label.setPixmap(scaled)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
