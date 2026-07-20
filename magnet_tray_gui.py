#!/usr/bin/env python3
"""
magnet_tray_gui.py
==================
Magnet Tray ジェネレータの PyQt5 GUI。

  python magnet_tray_gui.py

magnet_tray_generator.py と同じディレクトリに置いて実行してください。
起動時に前回使用した元モデル、またはスクリプトと同じフォルダの
1x1x1 モデル (*.3mf / box.stl) を自動で読み込みます。

3Dプレビュー:
  - pyqtgraph + PyOpenGL があればマウスで回転/ズームできる高速ビュー
      pip install pyqtgraph PyOpenGL
  - 無ければ matplotlib 埋め込みビュー(視点ボタン切替)に自動フォールバック

依存: PyQt5, trimesh, manifold3d, shapely, numpy, lxml, matplotlib
"""
import glob
import os
import sys
import time
import traceback

import numpy as np
from PyQt5.QtCore import Qt, QThread, QSettings, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSpinBox,
    QDoubleSpinBox, QLineEdit, QFileDialog, QCheckBox, QGroupBox,
    QHBoxLayout, QVBoxLayout, QFormLayout, QMessageBox, QSizePolicy)

import magnet_tray_generator as gen

LIGHT1 = np.array([0.45, -0.5, 0.74])
LIGHT2 = np.array([-0.5, 0.7, 0.3])
TRAY_RGB = np.array([0.82, 0.82, 0.85])
LID_RGB = np.array([0.55, 0.72, 0.95])
BG = (30, 33, 36)

# ------------------------------------------------------------
# プレビューバックエンド判定
# ------------------------------------------------------------
USE_GL = False
if not os.environ.get('MAGNET_GUI_NO_GL'):
    try:
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl
        USE_GL = True
    except Exception:
        USE_GL = False

if not USE_GL:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def face_colors(mesh, base_rgb):
    """面法線からフラットシェーディング色を計算 (ライティング焼き込み)"""
    n = mesh.face_normals
    d1 = np.clip(n @ (LIGHT1 / np.linalg.norm(LIGHT1)), 0, 1)
    d2 = np.clip(n @ (LIGHT2 / np.linalg.norm(LIGHT2)), 0, 1)
    inten = np.clip(0.32 + 0.62 * d1 + 0.22 * d2, 0, 1.15)
    rgb = np.clip(np.outer(inten, base_rgb), 0, 1)
    return np.c_[rgb, np.ones(len(rgb))]


# ------------------------------------------------------------
# 生成ワーカースレッド
# ------------------------------------------------------------
class BuildWorker(QThread):
    done = pyqtSignal(object, object, str, dict)   # tray, lid, info, params
    failed = pyqtSignal(str)

    def __init__(self, base_box, W, L, H, sections, div_t, make_lid):
        super().__init__()
        self.args = (base_box, W, L, H, sections, div_t, make_lid)

    def run(self):
        base_box, W, L, H, sections, div_t, make_lid = self.args
        try:
            t0 = time.time()
            tray = gen.build_box(base_box, W, L, H, sections=sections,
                                 divider_t=div_t, verbose=False)
            lid = gen.build_lid(W, L, verbose=False) if make_lid else None
            nw, nl = sections
            n_side = 2 * (W + L) * H
            iw = W * gen.MODULE - 2 * gen.WALL_T
            il = L * gen.MODULE - 2 * gen.WALL_T
            cw = (iw - (nw - 1) * div_t) / nw
            cl = (il - (nl - 1) * div_t) / nl
            info = (
                f"トレイ外寸: {W*gen.MODULE:.1f} x {L*gen.MODULE:.1f} x "
                f"{gen.BASE_H + (H-1)*gen.HEIGHT_UNIT:.2f} mm\n"
                f"区画: {nw} x {nl}  (各内寸 {cw:.1f} x {cl:.1f} mm, "
                f"深さ {19.05 + (H-1)*gen.HEIGHT_UNIT:.1f} mm)\n"
                f"必要磁石 (5x2mm): トレイ {n_side + 4} 個"
                f"{' + 蓋 2 個' if make_lid else ''}\n"
                f"生成時間: {time.time()-t0:.1f} 秒  "
                f"watertight: {tray.is_watertight}")
            params = {'W': W, 'L': L, 'H': H, 'nw': nw, 'nl': nl,
                      'divider_t': div_t, 'make_lid': make_lid}
            self.done.emit(tray, lid, info, params)
        except Exception:
            self.failed.emit(traceback.format_exc())


# ------------------------------------------------------------
# プレビュー (OpenGL版)
# ------------------------------------------------------------
class GLPreview(QWidget):
    def __init__(self):
        super().__init__()
        self.view = gl.GLViewWidget()
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setBackgroundColor(*BG)
        self.view.setCameraPosition(distance=220, elevation=30, azimuth=-60)
        self.grid = gl.GLGridItem()
        self.grid.setSize(300, 300)
        self.grid.setSpacing(gen.MODULE, gen.MODULE)
        self.grid.setColor((110, 110, 120, 90))
        self.view.addItem(self.grid)
        self.items = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view, stretch=1)
        hint = QLabel('左ドラッグ: 回転  /  ホイール: ズーム  /  中ドラッグ: 移動')
        hint.setStyleSheet('color: #888; padding: 2px;')
        hint.setAlignment(Qt.AlignCenter)
        lay.addWidget(hint)

    def show_meshes(self, tray, lid):
        for it in self.items:
            self.view.removeItem(it)
        self.items = []

        def add(mesh, rgb, dx=0.0):
            v = np.asarray(mesh.vertices, dtype=np.float32).copy()
            v[:, 0] += dx
            v[:, 2] -= mesh.bounds[0][2]
            md = gl.MeshData(vertexes=v, faces=np.asarray(mesh.faces),
                             faceColors=face_colors(mesh, rgb))
            # ライティングは色に焼き込み済みなので shader なしで描画
            item = gl.GLMeshItem(meshdata=md, smooth=False, shader=None,
                                 glOptions='opaque', drawEdges=False)
            self.view.addItem(item)
            self.items.append(item)

        add(tray, TRAY_RGB)
        if lid is not None:
            off = tray.extents[0] / 2 + lid.extents[0] / 2 + 12
            add(lid, LID_RGB, dx=off)

        # カメラを対象にフィット
        w = tray.extents[0] + (lid.extents[0] + 12 if lid is not None else 0)
        cx = (w - tray.extents[0]) / 2
        self.view.opts['center'] = pg.Vector(cx, 0, tray.extents[2] / 2)
        self.view.setCameraPosition(distance=max(w, tray.extents[1]) * 1.7,
                                    elevation=32, azimuth=-60)
        size = max(150., w * 1.6)
        self.grid.setSize(size, size)


# ------------------------------------------------------------
# プレビュー (matplotlib版フォールバック)
# ------------------------------------------------------------
class MplPreview(QWidget):
    def __init__(self):
        super().__init__()
        self.fig = Figure(facecolor='#1e2124')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.tray = self.lid = None
        self.elev, self.azim = 32, -60
        btns = QHBoxLayout()
        for label, e, a in (('斜め', 32, -60), ('上面', 88, -90),
                            ('正面', 5, -90), ('底面', -88, -90)):
            b = QPushButton(label)
            b.clicked.connect(lambda _, e=e, a=a: self.set_view(e, a))
            btns.addWidget(b)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas, stretch=1)
        lay.addLayout(btns)
        self.redraw()   # 起動時に暗背景を描画

    def set_view(self, elev, azim):
        self.elev, self.azim = elev, azim
        self.redraw()

    def show_meshes(self, tray, lid):
        self.tray, self.lid = tray, lid
        self.redraw()

    def redraw(self):
        self.ax.clear()
        self.ax.set_facecolor('#1e2124')
        self.ax.set_axis_off()
        if self.tray is None:
            self.canvas.draw()
            return

        def draw(mesh, rgb, dx=0.0):
            tri = mesh.vertices[mesh.faces].copy()
            tri[:, :, 0] += dx
            pc = Poly3DCollection(tri, facecolors=face_colors(mesh, rgb),
                                  edgecolors='none')
            self.ax.add_collection3d(pc)
            return tri.reshape(-1, 3)

        pts = [draw(self.tray, TRAY_RGB)]
        if self.lid is not None:
            off = self.tray.extents[0] / 2 + self.lid.extents[0] / 2 + 12
            pts.append(draw(self.lid, LID_RGB, dx=off))
        allp = np.vstack(pts)
        c = (allp.max(0) + allp.min(0)) / 2
        r = (allp.max(0) - allp.min(0)).max() / 2
        self.ax.set_xlim(c[0]-r, c[0]+r)
        self.ax.set_ylim(c[1]-r, c[1]+r)
        self.ax.set_zlim(c[2]-r, c[2]+r)
        self.ax.view_init(elev=self.elev, azim=self.azim)
        self.canvas.draw()


# ------------------------------------------------------------
# メインウィンドウ
# ------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Magnet Tray ジェネレータ')
        self.settings = QSettings('MagnetTray', 'Generator')
        self.base_box = None
        self.tray = self.lid = None
        self.built_params = None   # 保存名は生成時パラメータから決める (画面値は使わない)
        self.worker = None

        # ----- 元モデル -----
        self.src_edit = QLineEdit()
        self.src_edit.setPlaceholderText('1x1x1 の 3MF / STL')
        self.src_edit.setReadOnly(True)
        self.btn_browse = QPushButton('参照...')
        self.btn_browse.clicked.connect(self.browse_source)
        self.src_status = QLabel('')
        src_lay = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(self.src_edit)
        row.addWidget(self.btn_browse)
        src_lay.addLayout(row)
        src_lay.addWidget(self.src_status)
        grp_src = QGroupBox('元モデル (1x1x1)')
        grp_src.setLayout(src_lay)

        # ----- サイズ / 区画 -----
        def spin(lo, hi, val):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setFixedWidth(80)
            s.setAlignment(Qt.AlignRight)
            return s
        # 上限は Bambu Lab X2D の造形サイズ (256 x 256 x 260mm) 由来:
        #   W/L: 6モジュール = 228.6mm ≤ 256mm (7モジュール266.7mmは不可)
        #   H:   10段 = 250.8mm ≤ 260mm
        self.spin_w = spin(1, 6, 2)
        self.spin_l = spin(1, 6, 2)
        self.spin_h = spin(1, 10, 1)
        self.spin_w.setToolTip('最大6 (=228.6mm、X2Dベッド256mmに収まる上限)')
        self.spin_l.setToolTip('最大6 (=228.6mm、X2Dベッド256mmに収まる上限)')
        self.spin_h.setToolTip('最大10 (=250.8mm、X2D造形高さ260mmに収まる上限)\n'
                               'H1=22.2 / H2=47.6 / H3=73.0mm... (+25.4mm/段)')
        self.spin_sw = spin(1, 6, 1)
        self.spin_sl = spin(1, 6, 1)
        self.spin_dt = QDoubleSpinBox()
        self.spin_dt.setRange(1.0, 6.0)
        self.spin_dt.setSingleStep(0.5)
        self.spin_dt.setValue(gen.WALL_T)
        self.spin_dt.setSuffix(' mm')
        self.spin_dt.setFixedWidth(80)
        self.spin_dt.setAlignment(Qt.AlignRight)
        self.spin_w.valueChanged.connect(self.update_section_limits)
        self.spin_l.valueChanged.connect(self.update_section_limits)

        size_form = QFormLayout()
        size_form.setLabelAlignment(Qt.AlignRight)
        size_form.addRow('横 W (x38.1mm):', self.spin_w)
        size_form.addRow('縦 L (x38.1mm):', self.spin_l)
        size_form.addRow('高さ H:', self.spin_h)
        grp_size = QGroupBox('サイズ')
        grp_size.setLayout(size_form)

        sec_form = QFormLayout()
        sec_form.setLabelAlignment(Qt.AlignRight)
        sec_form.addRow('幅方向 区画数:', self.spin_sw)
        sec_form.addRow('奥行方向 区画数:', self.spin_sl)
        sec_form.addRow('仕切り厚:', self.spin_dt)
        grp_sec = QGroupBox('内部区画 (仕切り)')
        grp_sec.setLayout(sec_form)

        self.chk_lid = QCheckBox('蓋も生成する')
        self.chk_lid.setChecked(True)

        self.btn_build = QPushButton('プレビュー生成')
        self.btn_build.setMinimumHeight(34)
        self.btn_build.clicked.connect(self.build)
        self.btn_save = QPushButton('STL 保存...')
        self.btn_save.setMinimumHeight(30)
        self.btn_save.clicked.connect(self.save)
        self.btn_save.setEnabled(False)

        self.info = QLabel('')
        self.info.setWordWrap(True)
        self.info.setStyleSheet(
            'padding: 8px; background: #2b2f33; color: #ddd; border-radius: 4px;')
        self.info.setMinimumHeight(96)
        self.info.setAlignment(Qt.AlignTop)

        left = QVBoxLayout()
        left.setSpacing(10)
        left.addWidget(grp_src)
        left.addWidget(grp_size)
        left.addWidget(grp_sec)
        left.addWidget(self.chk_lid)
        left.addWidget(self.btn_build)
        left.addWidget(self.btn_save)
        left.addWidget(self.info)
        left.addStretch()
        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(330)

        # ----- プレビュー -----
        try:
            self.preview = GLPreview() if USE_GL else MplPreview()
        except Exception:
            self.preview = MplPreview()

        root = QHBoxLayout()
        root.addWidget(left_w)
        root.addWidget(self.preview, stretch=1)
        cw = QWidget()
        cw.setLayout(root)
        self.setCentralWidget(cw)

        self.setMinimumSize(980, 620)
        geo = self.settings.value('geometry')
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1280, 760)

        self.update_section_limits()
        if not USE_GL:
            self.statusBar().showMessage(
                'matplotlib プレビュー使用中 (pip install pyqtgraph PyOpenGL で高速ビュー)')
        self.autoload_base()

    def closeEvent(self, ev):
        if self.worker is not None and self.worker.isRunning():
            ret = QMessageBox.question(
                self, '生成中です',
                '生成処理が実行中です。完了を待ってから終了しますか?\n'
                '(いいえ を選ぶとウィンドウは閉じません)',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ret != QMessageBox.Yes:
                ev.ignore()
                return
            self.worker.wait()   # QThread実行中の破棄を防ぐ
        self.settings.setValue('geometry', self.saveGeometry())
        super().closeEvent(ev)

    # ------------------------------------------------------------
    def set_inputs_enabled(self, on):
        """生成中はサイズ・区画・元モデルの変更を禁止する
        (画面値と生成物の乖離を防ぐ)"""
        for w in (self.spin_w, self.spin_l, self.spin_h,
                  self.spin_dt, self.chk_lid, self.btn_browse):
            w.setEnabled(on)
        if on:
            self.update_section_limits()
        else:
            self.spin_sw.setEnabled(False)
            self.spin_sl.setEnabled(False)

    def invalidate_result(self, reason):
        """生成済みモデルを無効化する (元モデル変更時など)"""
        self.tray = self.lid = None
        self.built_params = None
        self.btn_save.setEnabled(False)
        self.info.setText(reason)

    def set_src_status(self, ok, msg):
        color = '#7ec87e' if ok else '#e0a04a'
        mark = '✔' if ok else '⚠'
        self.src_status.setText(f'{mark} {msg}')
        self.src_status.setStyleSheet(f'color: {color}; padding: 2px;')

    def autoload_base(self):
        """前回パス → スクリプトフォルダ/カレントの順で1x1x1を自動探索"""
        candidates = []
        last = self.settings.value('source_path', '')
        if last:
            candidates.append(last)
        here = os.path.dirname(os.path.abspath(__file__))
        for d in (here, os.getcwd()):
            candidates += sorted(glob.glob(os.path.join(d, '*1x1x1*.3mf')))
            candidates += sorted(glob.glob(os.path.join(d, '*.3mf')))
            candidates.append(os.path.join(d, 'box.stl'))
        seen = set()
        for path in candidates:
            if not path or path in seen or not os.path.exists(path):
                continue
            seen.add(path)
            try:
                self.set_source(path)
                return
            except Exception:
                continue
        self.set_src_status(False,
            '1x1x1 モデルが見つかりません。「参照...」で指定してください')
        self.info.setText('⚠ 元モデル未設定\n\n'
                          '1x1x1_Storage_Box.3mf をこのスクリプトと同じ\n'
                          'フォルダに置くと起動時に自動読み込みされます。')

    def _load(self, path):
        if path.lower().endswith('.3mf'):
            box, _ = gen.load_bambu_3mf(path)
            if box is None:
                raise ValueError('3MF内にトレイのメッシュが見つかりません')
        else:
            import trimesh
            box = trimesh.load(path)
        gen.validate_base(box)   # 寸法・原点・向き・特徴位置を検証
        return box

    def browse_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '元の 1x1x1 モデルを選択',
            os.path.dirname(self.src_edit.text()) or '',
            '3Dモデル (*.3mf *.stl);;すべて (*)')
        if not path:
            return
        try:
            self.set_source(path)
        except Exception as e:
            QMessageBox.warning(self, '読み込みエラー', str(e))

    def set_source(self, path):
        """元モデルを読み込む。成功時は既存の生成物を無効化する"""
        try:
            new_box = self._load(path)
        except Exception as e:
            self.base_box = None
            self.set_src_status(False, str(e))
            raise
        self.base_box = new_box
        self.src_edit.setText(path)
        self.settings.setValue('source_path', path)
        self.set_src_status(True, f'読み込み済み ({os.path.basename(path)})')
        if self.tray is not None:
            self.invalidate_result(
                '元モデルを変更しました。再生成してください。')

    def update_section_limits(self):
        for s, size in ((self.spin_sw, self.spin_w.value()),
                        (self.spin_sl, self.spin_l.value())):
            if size < 2:
                s.setValue(1)
                s.setEnabled(False)
            else:
                s.setEnabled(True)

    # ------------------------------------------------------------
    def build(self):
        if self.base_box is None:
            QMessageBox.warning(self, '元モデル未設定',
                                '先に 1x1x1 モデルを読み込んでください')
            return
        W, L, H = (self.spin_w.value(), self.spin_l.value(),
                   self.spin_h.value())
        self.btn_build.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.set_inputs_enabled(False)
        self.info.setText(f'{W}x{L}x{H} を生成中...\n'
                          '(ブーリアン演算に数秒かかります)')
        self.worker = BuildWorker(
            self.base_box, W, L, H,
            (self.spin_sw.value(), self.spin_sl.value()),
            self.spin_dt.value(), self.chk_lid.isChecked())
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_done(self, tray, lid, info, params):
        self.tray, self.lid = tray, lid
        self.built_params = params
        self.preview.show_meshes(tray, lid)
        self.info.setText(info)
        self.btn_build.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.set_inputs_enabled(True)

    def on_failed(self, tb):
        self.invalidate_result('生成に失敗しました')
        QMessageBox.critical(self, '生成エラー', tb[-1500:])
        self.btn_build.setEnabled(True)
        self.set_inputs_enabled(True)

    # ------------------------------------------------------------
    def save(self):
        if self.tray is None or self.built_params is None:
            return
        out = QFileDialog.getExistingDirectory(
            self, '保存先フォルダを選択',
            self.settings.value('out_dir', ''))
        if not out:
            return
        self.settings.setValue('out_dir', out)
        saved = self._do_save(out)
        QMessageBox.information(
            self, '保存完了', '保存しました:\n' + '\n'.join(saved))

    def _do_save(self, out):
        """ファイル名は必ず生成時パラメータ (built_params) から決める。
        画面のスピンボックス値は生成後に変更され得るため使わない。"""
        p = self.built_params
        W, L, H, nw, nl = p['W'], p['L'], p['H'], p['nw'], p['nl']
        tag = f'_S{nw}x{nl}' if (nw > 1 or nl > 1) else ''
        p1 = os.path.join(out, f'B_{W}x{L}x{H}{tag}.stl')
        self.tray.export(p1)
        saved = [p1]
        if self.lid is not None:
            p2 = os.path.join(out, f'T_{W}x{L}.stl')
            self.lid.export(p2)
            saved.append(p2)
        return saved


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
