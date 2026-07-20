#!/usr/bin/env python3
"""
magnet_tray_generator.py
========================
1x1x1 Magnet Tray (Storage Box) の3MF/STLを元に、任意サイズ
(横W x 縦L x 高さH モジュール) のトレイと蓋のSTLを生成する。

  1モジュール = 38.1mm (1.5"),  高さ1単位 = +25.4mm (1")
  H=1: 22.225mm / H=2: 47.625mm / H=3: 73.025mm ...

手法:
  - 元メッシュを「断面が位置不変」なカット面で2分割し、
    カット断面ポリゴンを正確に押し出したプリズムを挟んで拡張する。
    (フィレット・リップ・磁石ポストなどの形状を完全保持)
  - カット位置は実測により決定:
      XY: ±3.3mm  (側壁磁石穴 |u|<2.65 とポスト根元スカート半径~10.2 を回避)
      Z : +4.5mm  (側壁穴上端 4.242 とリム形状開始 4.762 の間)
  - 不足する側壁磁石穴 (モジュール中心 x 高さ25.4mm刻み) をブーリアンで追加。
  - 底面磁石穴(対角2箇所)とポスト(逆対角2箇所)は分割保持で自動的に四隅へ。
  - 蓋はハニカム構造をパラメトリック再構築 (実測パラメータ使用)。

使い方:
  python magnet_tray_generator.py --source 1x1x1_Storage_Box.3mf \
      --width 2 --length 3 --height 2 --out-dir ./out
  python magnet_tray_generator.py --box box.stl --lid lid.stl -W 2 -L 2 -H 1

依存: trimesh, manifold3d, shapely, numpy, lxml
"""
import argparse
import sys
import zipfile
import io
import numpy as np
import trimesh
from shapely.geometry import Polygon, Point, box as shapely_box
from shapely.ops import unary_union

# ============================================================
# 実測定数 (1x1x1 Storage Box 解析値)
# ============================================================
MODULE = 38.1          # 1モジュール寸法 (1.5")
HEIGHT_UNIT = 25.4     # 高さ1単位の増分 (1")
BASE_H = 22.225        # H=1 の外高 (0.875")
Z_BOT = -11.1125       # 元モデルの底Z
Z_TOP = 11.1125        # 元モデルの上Z

CUT_XY = 3.3           # XY方向カット位置 (断面不変ゾーン)
CUT_Z = 4.5            # Z方向カット位置 (断面不変ゾーン)

R_HOLE = 2.6543        # 磁石穴半径 (φ5.309 : 5x2mm磁石用)
HOLE_DEPTH = 2.0320    # 磁石穴深さ
HOLE_Z0 = 1.5875       # 最下段の側壁穴中心Z
HOLE_SECTIONS = 96     # 穴円筒の分割数

# 蓋 (実測パラメータ)
LID_THICK = 3.175
LID_CORNER_R = 6.35
LID_BORDER = 3.175
HEX_PITCH = 7.9375     # 5/16"
HEX_FLAT = 6.3736      # 六角対辺
HEX_CORNER_R = 0.15
PAD_R = 3.797
CORNER_INSET = 4.5699  # 底穴/蓋凹みの角からの対角インセット (19.05-14.4801)

SECTION_TOL = 0.5      # 断面不変性チェック閾値 (対称差面積) [mm²]

WALL_T = 3.175         # 壁厚 (仕切りのデフォルト厚もこれに合わせる)
FLOOR_TOP = -7.9375    # 内底面Z (元モデル座標)


# ============================================================
# 3MF / STL 読み込み
# ============================================================
def load_bambu_3mf(path):
    """Bambu Studio形式3MFから (box, lid) メッシュを抽出"""
    from lxml import etree
    ns = {'m': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
    meshes = []
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist()
                 if n.startswith('3D/') and n.endswith('.model')]
        # Bambu分割形式 (3D/Objects/*.model) を優先、なければ本体
        obj_files = [n for n in names if '/Objects/' in n] or names
        for n in obj_files:
            tree = etree.parse(io.BytesIO(zf.read(n)))
            for obj in tree.findall('.//m:object', ns):
                verts = [[float(v.get('x')), float(v.get('y')), float(v.get('z'))]
                         for v in obj.findall('.//m:vertex', ns)]
                tris = [[int(t.get('v1')), int(t.get('v2')), int(t.get('v3'))]
                        for t in obj.findall('.//m:triangle', ns)]
                if verts and tris:
                    m = trimesh.Trimesh(vertices=np.array(verts),
                                        faces=np.array(tris), process=False)
                    m.merge_vertices()
                    m.process()
                    meshes.append(m)
    if not meshes:
        raise ValueError(f'{path}: メッシュが見つかりません')
    # 分類: Z高さで箱(>10mm)と蓋(~3mm)を判別
    box = lid = None
    for m in meshes:
        if m.extents[2] > 10:
            box = m
        elif m.extents[2] < 6:
            lid = m
    return box, lid


# ============================================================
# コア: 断面不変カット + プリズム挿入による軸方向拡張
# ============================================================
def check_section_invariance(mesh, axis, cut, delta=0.15):
    """カット面近傍で断面が位置不変か検証 (対称差面積)"""
    def sec_poly(c):
        normal = [0, 0, 0]; normal[axis] = 1
        origin = [0, 0, 0]; origin[axis] = c
        sec = mesh.section(plane_origin=origin, plane_normal=normal)
        other = [i for i in range(3) if i != axis]
        polys = [Polygon(np.c_[lp[:, other[0]], lp[:, other[1]]])
                 for lp in sec.discrete]
        polys.sort(key=lambda p: -abs(p.area))
        ext = polys[0]
        holes = [p for p in polys[1:] if ext.contains(p)]
        return Polygon(ext.exterior.coords, [h.exterior.coords for h in holes])
    p0, p1, p2 = (sec_poly(cut - delta), sec_poly(cut), sec_poly(cut + delta))
    d = max(p0.symmetric_difference(p1).area, p2.symmetric_difference(p1).area)
    return d


def validate_base(box):
    """元1x1x1メッシュの寸法・原点・向き・特徴位置を検証する。

    寸法とwatertightだけでは、原点ずれ・回転・反転したSTLを
    検出できないため、既知の特徴 (床面・側壁磁石穴・底面磁石穴の
    対角位置) が期待座標に実在することを頂点ベースで確認する。
    """
    import numpy as _np
    errs = []
    exp_ext = _np.array([MODULE, MODULE, BASE_H])
    if not _np.allclose(box.extents, exp_ext, atol=0.1):
        errs.append(f'外寸が想定外: {_np.round(box.extents, 2)} (期待 {exp_ext})')
    exp_bounds = _np.array([[-MODULE / 2, -MODULE / 2, Z_BOT],
                            [MODULE / 2, MODULE / 2, Z_TOP]])
    if not _np.allclose(box.bounds, exp_bounds, atol=0.1):
        errs.append(f'原点位置が想定外: bounds={_np.round(box.bounds, 2)} '
                    '(原点中心・Z範囲-11.11..11.11のメッシュが必要)')
    if not box.is_watertight:
        errs.append('メッシュがwatertightではありません')
    if errs:
        raise ValueError(' / '.join(errs))

    V = box.vertices
    # 床面: 中央領域 (|x|,|y|<8) に z=FLOOR_TOP の頂点があること。
    # 反転メッシュではリムのリップ (z=+7.94→-7.94) が壁際にのみ現れるため、
    # 中央領域に限定して判別する。
    m = ((_np.abs(V[:, 2] - FLOOR_TOP) < 0.05)
         & (_np.abs(V[:, 0]) < 8) & (_np.abs(V[:, 1]) < 8))
    if _np.sum(m) < 3:
        errs.append(f'内底面 (z={FLOOR_TOP}) が見つかりません (上下反転の可能性)')
    # 側壁磁石穴の奥面リング: 中心高さ z=HOLE_Z0 の近傍 (±0.4) に限定。
    # 反転時は穴中心が z=-1.59 に移るため、このリングは z<1.07 となり検出されない。
    m = ((_np.abs(V[:, 0] - (MODULE / 2 - HOLE_DEPTH)) < 0.05)
         & (_np.abs(V[:, 1]) < 3.0)
         & (_np.abs(V[:, 2] - HOLE_Z0) < 0.4))
    if _np.sum(m) < 2:
        errs.append('側壁磁石穴が期待位置 (X+壁中央) にありません')
    # 底面磁石穴の開口リング: 底面 (z=Z_BOT) 上で半径 R_HOLE の円周頂点。
    # 反転時にポスト上面が同座標に来る偽陽性を、半径条件で排除する。
    hx = MODULE / 2 - CORNER_INSET
    onbot = _np.abs(V[:, 2] - Z_BOT) < 0.05
    r = _np.sqrt((V[:, 0] - hx) ** 2 + (V[:, 1] - hx) ** 2)
    if _np.sum(onbot & (_np.abs(r - R_HOLE) < 0.1)) < 8:
        errs.append('底面磁石穴が期待位置 (+,+対角) にありません '
                    '(反転・回転の可能性)')
    if errs:
        raise ValueError(' / '.join(errs))


def validate_result(mesh, name):
    """生成メッシュの形状健全性を検査する (watertightだけでは不十分)。"""
    errs = []
    if not mesh.is_watertight:
        errs.append('watertightではない')
    if not mesh.is_winding_consistent:
        errs.append('面の向きが不整合')
    if not mesh.is_volume:
        errs.append('有効なソリッドではない')
    if mesh.volume <= 0:
        errs.append(f'体積が不正 ({mesh.volume:.1f})')
    if mesh.body_count != 1:
        errs.append(f'分離ボディが存在 (body_count={mesh.body_count})')
    if errs:
        raise RuntimeError(f'{name} の健全性検査に失敗: ' + ' / '.join(errs))


def section_polygon(mesh, axis, cut):
    """カット面の断面をshapely Polygonとして取得 ((other0, other1)座標系)"""
    normal = [0, 0, 0]; normal[axis] = 1
    origin = [0, 0, 0]; origin[axis] = cut
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    other = [i for i in range(3) if i != axis]
    polys = [Polygon(np.c_[lp[:, other[0]], lp[:, other[1]]])
             for lp in sec.discrete]
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    polys = [p for p in polys if abs(p.area) > 0.01]
    polys.sort(key=lambda p: -abs(p.area))
    ext = polys[0]
    holes = [p for p in polys[1:] if ext.contains(p)]
    poly = Polygon(ext.exterior.coords, [h.exterior.coords for h in holes])
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def widen_polygon(poly, cut, shift, coord=0):
    """断面ポリゴンを2Dで拡張 (coord軸の cut より大きい頂点を shift)。

    拡張ゾーンではカット線を横切る境界エッジが全て軸平行のため、
    頂点シフトは厳密に正しい widened 断面を与える。
    """
    if shift <= 0:
        return poly
    def f(coords):
        c = np.array(coords)
        c[c[:, coord] > cut, coord] += shift
        return c
    return Polygon(f(poly.exterior.coords),
                   [f(h.coords) for h in poly.interiors])


def widen_axis(mesh, axis, cut, shift, poly2d):
    """mesh を axis 方向に cut 位置で分割し、poly2d を押し出したプリズムを挟んで拡張。

    poly2d は必ず「クリーンな元メッシュ由来の断面」を widen_polygon で
    拡張したものを渡すこと。ブーリアン出力メッシュからの断面抽出は
    継ぎ目の微細三角形でポリライン組み立てが崩れることがあるため行わない。
    """
    if shift <= 0:
        return mesh
    big = 10000.0
    lo = [-big] * 3; hi = [big] * 3; lo[axis] = cut
    cutbox = trimesh.creation.box(bounds=[lo, hi])
    half_lo = trimesh.boolean.difference([mesh, cutbox], engine='manifold')
    half_hi = trimesh.boolean.intersection([mesh, cutbox], engine='manifold')
    T = np.eye(4); T[axis, 3] = shift
    half_hi.apply_transform(T)

    other = [i for i in range(3) if i != axis]
    prism = trimesh.creation.extrude_polygon(poly2d, height=shift)
    M = np.zeros((4, 4)); M[3, 3] = 1
    M[other[0], 0] = 1
    M[other[1], 1] = 1
    M[axis, 2] = 1
    M[axis, 3] = cut
    prism.apply_transform(M)
    if prism.volume < 0:
        prism.invert()
    return trimesh.boolean.union([half_lo, prism, half_hi], engine='manifold')


# ============================================================
# 磁石穴
# ============================================================
def hole_cylinder(tx, ty, tz, rot=None, overcut=1.0):
    c = trimesh.creation.cylinder(radius=R_HOLE, height=HOLE_DEPTH + overcut,
                                  sections=HOLE_SECTIONS)
    M = np.eye(4)
    if rot is not None:
        M[:3, :3] = rot
    M[:3, 3] = [tx, ty, tz]
    c.apply_transform(M)
    return c


def module_centers(n):
    """nモジュール軸のモジュール中心座標 (原点センタリング後)"""
    return [-(n - 1) * MODULE / 2 + k * MODULE for k in range(n)]


# ============================================================
# トレイ生成
# ============================================================
def add_dividers(mesh, W, L, H, nw, nl, t):
    """内寸を等分割する仕切りリブを追加 (nw: 幅方向区画数, nl: 奥行方向区画数)"""
    if nw <= 1 and nl <= 1:
        return mesh
    half_w = W * MODULE / 2
    half_l = L * MODULE / 2
    iw = W * MODULE - 2 * WALL_T   # 内寸
    il = L * MODULE - 2 * WALL_T
    z_top = Z_TOP + (H - 1) * HEIGHT_UNIT
    z_bot = FLOOR_TOP - 1.0        # 床に1mm食い込ませて融合
    OVL = 2.0                      # 壁への食い込み
    slabs = []
    def positions(inner, n):
        cell = (inner - (n - 1) * t) / n
        if cell <= 5.0:
            raise ValueError(f'区画が小さすぎます (区画内寸 {cell:.1f}mm)')
        return [-inner / 2 + cell * (k + 1) + t * k + t / 2 for k in range(n - 1)], cell
    if nw > 1:
        xs, cw = positions(iw, nw)
        for xd in xs:
            slabs.append(trimesh.creation.box(
                bounds=[[xd - t / 2, -half_l + WALL_T - OVL, z_bot],
                        [xd + t / 2, half_l - WALL_T + OVL, z_top]]))
    else:
        cw = iw
    if nl > 1:
        ys, cl = positions(il, nl)
        for yd in ys:
            slabs.append(trimesh.creation.box(
                bounds=[[-half_w + WALL_T - OVL, yd - t / 2, z_bot],
                        [half_w - WALL_T + OVL, yd + t / 2, z_top]]))
    else:
        cl = il
    # 注意: 交差する仕切り板を concatenate すると自己交差メッシュになり
    # manifold の入力として不正。必ず個別メッシュとして union に渡す。
    mesh = trimesh.boolean.union([mesh] + slabs, engine='manifold')
    return mesh, cw, cl


def build_box(base_box, W, L, H, sections=(1, 1), divider_t=WALL_T, verbose=True):
    # 断面不変性は元メッシュに対して一度だけ検証する
    for axis, cut, n in ((0, CUT_XY, W), (1, CUT_XY, L), (2, CUT_Z, H)):
        if n > 1:
            d = check_section_invariance(base_box, axis, cut)
            if d > SECTION_TOL:
                raise RuntimeError(
                    f'axis{axis} cut={cut}: 元メッシュの断面が位置不変ではありません '
                    f'(対称差 {d:.4f}mm²)')
    Sx = (W - 1) * MODULE
    Sy = (L - 1) * MODULE
    Sz = (H - 1) * HEIGHT_UNIT
    m = base_box.copy()
    # 断面はすべて元メッシュから取得し、2Dで拡張してから押し出す
    if Sx > 0:
        m = widen_axis(m, 0, CUT_XY, Sx, section_polygon(base_box, 0, CUT_XY))
    if Sy > 0:
        pY = section_polygon(base_box, 1, CUT_XY)      # (x, z)
        pY = widen_polygon(pY, CUT_XY, Sx, coord=0)    # x方向を拡張
        m = widen_axis(m, 1, CUT_XY, Sy, pY)
    if Sz > 0:
        pZ = section_polygon(base_box, 2, CUT_Z)       # (x, y)
        pZ = widen_polygon(pZ, CUT_XY, Sx, coord=0)
        pZ = widen_polygon(pZ, CUT_XY, Sy, coord=1)
        m = widen_axis(m, 2, CUT_Z, Sz, pZ)
    # XYを原点センタリング (Zは底面位置維持: 上方向にのみ伸びる)
    m.apply_translation([-(W - 1) * MODULE / 2, -(L - 1) * MODULE / 2, 0])

    # 側壁磁石穴の追加 (既存: 各壁のモジュール0・レベル0)
    RX = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])[:3, :3]
    RY = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])[:3, :3]
    OV = 1.0
    half_w = W * MODULE / 2
    half_l = L * MODULE / 2
    cutters = []
    for lvl in range(H):
        z = HOLE_Z0 + lvl * HEIGHT_UNIT
        for sign in (+1, -1):
            # X壁 (x=±half_w): L個のモジュール位置
            xc = sign * (half_w - (HOLE_DEPTH + OV) / 2 + OV)
            for k, y in enumerate(module_centers(L)):
                if k == 0 and lvl == 0:
                    continue  # ストレッチで既存
                cutters.append(hole_cylinder(xc, y, z, RX, OV))
            # Y壁 (y=±half_l): W個のモジュール位置
            yc = sign * (half_l - (HOLE_DEPTH + OV) / 2 + OV)
            for k, x in enumerate(module_centers(W)):
                if k == 0 and lvl == 0:
                    continue
                cutters.append(hole_cylinder(x, yc, z, RY, OV))
    if cutters:
        m = trimesh.boolean.difference(
            [m, trimesh.util.concatenate(cutters)], engine='manifold')

    # 内部仕切り
    nw, nl = sections
    sec_info = ''
    if nw > 1 or nl > 1:
        m, cw, cl = add_dividers(m, W, L, H, nw, nl, divider_t)
        sec_info = (f', 区画 {nw}x{nl} (各内寸 {cw:.2f} x {cl:.2f} mm)')
    m.merge_vertices()
    m.process()
    validate_result(m, f'トレイ {W}x{L}x{H}')
    if verbose:
        n_side = 2 * (W + L) * H
        print(f'  トレイ: {W * MODULE:.1f} x {L * MODULE:.1f} x '
              f'{BASE_H + (H - 1) * HEIGHT_UNIT:.3f} mm, '
              f'側壁穴 {n_side} / 底穴 2 / ポスト 2'
              f'{sec_info}, 検査OK (watertight/単一ボディ/体積正)')
    return m


# ============================================================
# 蓋生成 (パラメトリック・ハニカム)
# ============================================================
def build_lid(W, L, verbose=True):
    sx, sy = W * MODULE, L * MODULE

    def rounded_rect(hx, hy, r):
        return shapely_box(-hx, -hy, hx, hy).buffer(-r).buffer(r, quad_segs=48)

    def hexagon(cx, cy):
        R = HEX_FLAT / np.sqrt(3)
        ang = np.radians(90 + np.arange(0, 360, 60))  # pointy-top
        h = Polygon(np.c_[cx + R * np.cos(ang), cy + R * np.sin(ang)])
        return h.buffer(-HEX_CORNER_R, quad_segs=8).buffer(HEX_CORNER_R, quad_segs=8)

    outline = rounded_rect(sx / 2, sy / 2, LID_CORNER_R)
    inner = outline.buffer(-LID_BORDER, quad_segs=48)

    row_h = HEX_PITCH * np.sqrt(3) / 2
    hexes = []
    nj = int(sy / 2 / row_h) + 2
    ni = int(sx / 2 / HEX_PITCH) + 2
    for j in range(-nj, nj + 1):
        y = j * row_h
        xoff = HEX_PITCH / 2 if j % 2 else 0.0
        for i in range(-ni, ni + 1):
            x = i * HEX_PITCH + xoff
            hexes.append(hexagon(x, y))

    # 磁石パッド: トレイ底穴と同じ対角2隅 (角から4.5699インセット)
    mag = [(-sx / 2 + CORNER_INSET, -sy / 2 + CORNER_INSET),
           (sx / 2 - CORNER_INSET, sy / 2 - CORNER_INSET)]
    pads = unary_union([Point(p).buffer(PAD_R, quad_segs=48) for p in mag])

    cutouts = unary_union(hexes).intersection(inner).difference(pads)
    plate = outline.difference(cutouts)
    lid = trimesh.creation.extrude_polygon(plate, height=LID_THICK)
    lid.apply_translation([0, 0, -LID_THICK / 2])

    cutters = []
    for hx, hy in mag:
        c = trimesh.creation.cylinder(radius=R_HOLE, height=HOLE_DEPTH + 1.0,
                                      sections=HOLE_SECTIONS)
        c.apply_translation([hx, hy,
                             LID_THICK / 2 - HOLE_DEPTH + (HOLE_DEPTH + 1.0) / 2])
        cutters.append(c)
    lid = trimesh.boolean.difference(
        [lid, trimesh.util.concatenate(cutters)], engine='manifold')
    lid.merge_vertices()
    lid.process()
    validate_result(lid, f'蓋 {W}x{L}')
    if verbose:
        print(f'  蓋:    {sx:.1f} x {sy:.1f} x {LID_THICK} mm, '
              f'磁石凹み 2, 検査OK')
    return lid


# ============================================================
# メイン
# ============================================================
def main():
    ap = argparse.ArgumentParser(description='Magnet Tray ジェネレータ')
    ap.add_argument('--source', help='元の 1x1x1 3MFファイル')
    ap.add_argument('--box', help='元トレイのSTL (3MFの代わり)')
    ap.add_argument('-W', '--width', type=int, default=2, help='横モジュール数')
    ap.add_argument('-L', '--length', type=int, default=2, help='縦モジュール数')
    ap.add_argument('-H', '--height', type=int, default=1, help='高さ単位数')
    ap.add_argument('-S', '--sections', default='1,1',
                    help='区画数 "幅方向,奥行方向" (例: 2,1) デフォルト 1,1=仕切りなし')
    ap.add_argument('--divider-thickness', type=float, default=WALL_T,
                    help=f'仕切り厚 [mm] (デフォルト {WALL_T})')
    ap.add_argument('--out-dir', default='.', help='出力ディレクトリ')
    ap.add_argument('--no-lid', action='store_true', help='蓋を生成しない')
    args = ap.parse_args()

    if args.width < 1 or args.length < 1 or args.height < 1:
        sys.exit('W, L, H は1以上を指定してください')

    if args.source:
        base_box, _ = load_bambu_3mf(args.source)
    elif args.box:
        base_box = trimesh.load(args.box)
    else:
        sys.exit('--source (3MF) か --box (STL) を指定してください')

    # 元メッシュの前提チェック (寸法・原点・向き・特徴位置)
    try:
        validate_base(base_box)
    except ValueError as e:
        sys.exit(f'元トレイの検証エラー: {e}')

    W, L, H = args.width, args.length, args.height
    try:
        nw, nl = (int(v) for v in args.sections.split(','))
    except Exception:
        sys.exit('--sections は "2,1" のような形式で指定してください')
    if nw < 1 or nl < 1:
        sys.exit('区画数は1以上を指定してください')
    if (nw > 1 and W < 2) or (nl > 1 and L < 2):
        sys.exit('仕切りは該当方向のサイズが2以上の場合のみ指定できます')
    import os
    os.makedirs(args.out_dir, exist_ok=True)

    tag = f'_S{nw}x{nl}' if (nw > 1 or nl > 1) else ''
    print(f'=== {W}x{L}x{H} Magnet Tray 生成{" (区画 "+str(nw)+"x"+str(nl)+")" if tag else ""} ===')
    tray = build_box(base_box, W, L, H, sections=(nw, nl),
                     divider_t=args.divider_thickness)
    p1 = os.path.join(args.out_dir, f'B_{W}x{L}x{H}{tag}.stl')
    tray.export(p1)
    print(f'  -> {p1}')

    if not args.no_lid:
        lid = build_lid(W, L)
        p2 = os.path.join(args.out_dir, f'T_{W}x{L}.stl')
        lid.export(p2)
        print(f'  -> {p2}')


if __name__ == '__main__':
    main()
