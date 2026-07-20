#!/usr/bin/env python3
"""
test_modular_tray.py
===================
modular Tray ジェネレータの自動テスト。

実行方法 (どちらでも可):
    pytest test_modular_tray.py -v
    python test_modular_tray.py

元モデル (box.stl または *1x1x1*.3mf) がスクリプトと同じフォルダに
必要です。見つからない場合、生成テストはスキップされます。
"""
import glob
import os
import sys

import numpy as np
import trimesh

import modular_tray_generator as gen

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_base():
    for pat in ('box.stl', '*1x1x1*.3mf', '*.3mf'):
        for p in sorted(glob.glob(os.path.join(HERE, pat))):
            try:
                if p.lower().endswith('.3mf'):
                    b, _ = gen.load_bambu_3mf(p)
                else:
                    b = trimesh.load(p)
                gen.validate_base(b)
                return b
            except Exception:
                continue
    return None


BASE = _find_base()


def _check_solid(m, name=''):
    """健全性の共通チェック (レビュー指摘の検査項目)"""
    assert m.is_watertight, f'{name}: not watertight'
    assert m.is_winding_consistent, f'{name}: winding inconsistent'
    assert m.is_volume, f'{name}: not a valid volume'
    assert m.volume > 0, f'{name}: volume <= 0'
    assert m.body_count == 1, f'{name}: body_count={m.body_count}'


# ------------------------------------------------------------
# 単体テスト (ジオメトリ計算)
# ------------------------------------------------------------
def test_module_centers():
    assert gen.module_centers(1) == [0.0]
    assert np.allclose(gen.module_centers(2), [-19.05, 19.05])
    assert np.allclose(gen.module_centers(3), [-38.1, 0.0, 38.1])


def test_cell_size_math():
    # 2モジュール(内寸69.85)を2分割・仕切り3.175 → 各33.3375
    iw = 2 * gen.MODULE - 2 * gen.WALL_T
    cell = (iw - 1 * gen.WALL_T) / 2
    assert abs(cell - 33.3375) < 1e-6


def test_height_dimensions():
    for h, expect in ((1, 22.225), (2, 47.625), (3, 73.025), (10, 250.825)):
        assert abs(gen.BASE_H + (h - 1) * gen.HEIGHT_UNIT - expect) < 1e-6


# ------------------------------------------------------------
# 元モデル検証 (validate_base)
# ------------------------------------------------------------
def test_validate_base_accepts_good():
    if BASE is None:
        return
    gen.validate_base(BASE)   # 例外が出なければOK




def test_validate_base_rejects_transforms():
    if BASE is None:
        return
    cases = {
        'flip_x': trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]),
        'flip_y': trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]),
        'rot_z90': trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1]),
    }
    for name, T in cases.items():
        m = BASE.copy()
        m.apply_transform(T)
        try:
            gen.validate_base(m)
            raise AssertionError(f'{name}: 不正な向きを検出できなかった')
        except ValueError:
            pass
    m = BASE.copy()
    m.apply_translation([5, 0, 0])
    try:
        gen.validate_base(m)
        raise AssertionError('原点ずれを検出できなかった')
    except ValueError:
        pass


# ------------------------------------------------------------
# 生成テスト (代表寸法マトリクス)
# ------------------------------------------------------------
GEN_MATRIX = [
    # (W, L, H, sections)
    (1, 1, 1, (1, 1)),
    (2, 1, 1, (1, 1)),
    (1, 2, 1, (1, 1)),
    (2, 2, 1, (2, 2)),
    (1, 1, 2, (1, 1)),
    (2, 3, 2, (2, 1)),
]


def test_generation_matrix():
    if BASE is None:
        return
    for W, L, H, sec in GEN_MATRIX:
        tray = gen.build_box(BASE, W, L, H, sections=sec, verbose=False)
        name = f'{W}x{L}x{H}_S{sec[0]}x{sec[1]}'
        _check_solid(tray, name)
        exp = np.array([W * gen.MODULE, L * gen.MODULE,
                        gen.BASE_H + (H - 1) * gen.HEIGHT_UNIT])
        assert np.allclose(tray.extents, exp, atol=0.01), \
            f'{name}: extents {tray.extents} != {exp}'


def test_side_hole_count_2x2x1():
    """側壁磁石穴の全数をレイキャストで確認 (2*(W+L)*H = 8)"""
    if BASE is None:
        return
    tray = gen.build_box(BASE, 2, 2, 1, verbose=False)
    ray = trimesh.ray.ray_triangle.RayMeshIntersector(tray)
    half = 2 * gen.MODULE / 2
    found = 0
    for lvl in range(1):
        z = gen.HOLE_Z0 + lvl * gen.HEIGHT_UNIT
        for sign in (1, -1):
            for pos in gen.module_centers(2):
                for d in (0, 1):
                    o = [sign * 100, pos, z] if d == 0 else [pos, sign * 100, z]
                    dr = [-sign, 0, 0] if d == 0 else [0, -sign, 0]
                    locs, _, _ = ray.intersects_location([o], [dr])
                    if len(locs) and max(np.abs(locs[:, d])) < half - 0.001:
                        found += 1
    assert found == 8, f'側壁穴 {found}/8'


def test_divider_rejects_too_small():
    if BASE is None:
        return
    try:
        # 2モジュール(内寸69.85)を12分割 → 区画2.9mm → 拒否されるべき
        gen.build_box(BASE, 2, 2, 1, sections=(12, 1), verbose=False)
        raise AssertionError('過剰分割が拒否されなかった')
    except ValueError:
        pass


def test_regression_1x1x1_matches_base():
    """W=L=H=1 の生成物は元モデルとほぼ同一 (回帰基準)"""
    if BASE is None:
        return
    out = gen.build_box(BASE, 1, 1, 1, verbose=False)
    assert abs(out.volume - BASE.volume) < 1e-3, \
        f'体積差 {abs(out.volume - BASE.volume)}'
    assert np.allclose(out.extents, BASE.extents, atol=1e-6)
    assert np.allclose(out.bounds, BASE.bounds, atol=1e-6)


# ------------------------------------------------------------
# 蓋テスト
# ------------------------------------------------------------
def test_lid_dimensions_and_recess():
    lid = gen.build_lid(2, 2, verbose=False)
    _check_solid(lid, 'lid_2x2')
    assert np.allclose(lid.extents,
                       [76.2, 76.2, gen.LID_THICK], atol=0.01)
    # 磁石凹み: 上面開口・凹み底 z = THICK/2 - DEPTH (元モデル実測仕様)
    V = lid.vertices
    zfloor = gen.LID_THICK / 2 - gen.HOLE_DEPTH
    hx = 76.2 / 2 - gen.CORNER_INSET
    for cx, cy in ((-hx, -hx), (hx, hx)):
        r = np.sqrt((V[:, 0] - cx) ** 2 + (V[:, 1] - cy) ** 2)
        m = (np.abs(V[:, 2] - zfloor) < 0.01) & (np.abs(r - gen.R_HOLE) < 0.1)
        assert np.sum(m) >= 8, f'蓋の磁石凹みが ({cx:.1f},{cy:.1f}) にない'


# ------------------------------------------------------------
# スタンドアロン実行
# ------------------------------------------------------------
def _run_all():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith('test_') and callable(f)]
    if BASE is None:
        print('警告: 元モデルが見つからないため生成テストはスキップされます')
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f'  PASS  {name}')
        except AssertionError as e:
            failed += 1
            print(f'  FAIL  {name}: {e}')
        except Exception as e:
            failed += 1
            print(f'  ERROR {name}: {type(e).__name__}: {e}')
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    return failed


if __name__ == '__main__':
    sys.exit(1 if _run_all() else 0)
