"""
tng50_sid_rgccounts_MG.py

TNG50 subhalo에서 R_GC bin별 별 개수를 계산합니다.

★ 핵심: Gaia와 동일하게 M_G 절대등급으로 별 선택 ★
  - Gaia 관측 흉내 완전 제거 (completeness, SNR 판단 없음)
  - Pecaut & Mamajek (2013) 테이블로 질량 → M_G 변환
  - 동일한 M_G 범위로 TNG 별 선택
  - 태양으로부터 거리 컷 (Gaia parallax 범위와 동일한 공간)

선택 조건:
  1. R_GC:    5.0 ~ 8.0 kpc (원통형)
  2. |Z_GC| < 2.0 kpc
  3. d_sun:   d_sun_min ~ d_sun_max [kpc]
  4. M_G:     mg_min ~ mg_max [mag]  (Pecaut & Mamajek 테이블)
  5. surviving 모드: 주계열 전향점 이하만
"""

import argparse
import os
import time

import h5py
import numpy as np
import requests

# ── 우주론 적분 ───────────────────────────────────────────────────────────────
try:
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    _ASTROPY_AVAILABLE = True
except ImportError:
    _ASTROPY_AVAILABLE = False

BASE_URL = "https://www.tng-project.org/api/"

DEFAULT_SIM  = "TNG50-1"
DEFAULT_SNAP = 99
DEFAULT_SID  = 538905
DEFAULT_H    = 0.6774

TNG_H0 = 67.74
TNG_OM = 0.3089

DEFAULT_R_MIN      = 5.0
DEFAULT_R_MAX      = 8.0
DEFAULT_BIN_W      = 0.5
DEFAULT_CUTOUT_DIR = "cutouts"

DEFAULT_R0_KPC  = 8.122
DEFAULT_Z_MAX   = 2.0
DEFAULT_ALIGN_R = 30.0

# 거리 컷 기본값 (Gaia --plx-min/max 와 대응)
DEFAULT_D_SUN_MIN = 0.1    # kpc
DEFAULT_D_SUN_MAX = 3.5    # kpc

# M_G 절대등급 범위 기본값 (Gaia 코드와 반드시 동일하게 설정)
DEFAULT_MG_MIN = -7.5   # mag (고질량)
DEFAULT_MG_MAX = 10.0   # mag (저질량 ~0.55 Msun)

# IMF 설정
DEFAULT_COUNT_MODE = "surviving"
IMF_M_MIN = 0.08
IMF_M_MAX = 120.0

# ── Pecaut & Mamajek (2013) G밴드 질량-절대등급 테이블 ─────────────────────
_MASS_MG_TABLE = np.array([
    [0.080,  16.00],  # M8 V
    [0.100,  14.80],  # M7 V
    [0.120,  14.00],  # M6 V
    [0.150,  13.00],  # M5 V
    [0.200,  11.80],  # M4 V
    [0.260,  10.90],  # M3 V
    [0.350,  10.00],  # M2 V
    [0.450,   9.10],  # M1 V
    [0.550,   8.20],  # M0 V
    [0.650,   7.30],  # K7 V
    [0.750,   6.60],  # K5 V
    [0.850,   6.00],  # K4 V
    [0.900,   5.70],  # K3 V
    [1.000,   5.10],  # K1 V
    [1.050,   4.85],  # G8 V
    [1.100,   4.65],  # G5 V
    [1.150,   4.40],  # G2 V
    [1.200,   4.15],  # G0 V
    [1.400,   3.40],  # F8 V
    [1.600,   2.80],  # F5 V
    [1.800,   2.20],  # F2 V
    [2.000,   1.65],  # F0 V
    [2.500,   0.60],  # A7 V
    [3.000,  -0.20],  # A2 V
    [3.500,  -0.80],  # B9 V
    [5.000,  -2.00],  # B5 V
    [8.000,  -3.00],  # B2 V
    [15.00,  -4.50],  # B0 V
    [25.00,  -5.50],  # O9 V
    [40.00,  -6.20],  # O6 V
    [80.00,  -7.00],  # O3 V
    [120.0,  -7.50],  # O2 V
])
_TABLE_MASS = _MASS_MG_TABLE[:, 0]
_TABLE_MG   = _MASS_MG_TABLE[:, 1]


def mg_to_mass(mg_arr):
    """
    G밴드 절대등급 → 질량(Msun) 역보간.
    _TABLE_MG는 내림차순(밝은 쪽=큰 질량)이므로 반전 후 보간.
    """
    mg_arr   = np.asarray(mg_arr, dtype=float)
    mg_rev   = _TABLE_MG[::-1]     # 오름차순
    mass_rev = _TABLE_MASS[::-1]
    return np.interp(mg_arr, mg_rev, mass_rev,
                     left=mass_rev[0], right=mass_rev[-1])


def mass_to_mg(mass_arr):
    """질량(Msun) → G밴드 절대등급 보간."""
    return np.interp(np.asarray(mass_arr, dtype=float),
                     _TABLE_MASS, _TABLE_MG,
                     left=_TABLE_MG[0], right=_TABLE_MG[-1])


# ── 우주론 나이 계산 (버그 1 수정) ────────────────────────────────────────────
def _build_age_interpolator():
    if _ASTROPY_AVAILABLE:
        cosmo    = FlatLambdaCDM(H0=TNG_H0, Om0=TNG_OM)
        t_now    = cosmo.age(0.0).to(u.Gyr).value
        a_grid   = np.linspace(0.01, 1.0, 2000)
        z_grid   = 1.0 / a_grid - 1.0
        t_grid   = cosmo.age(z_grid).to(u.Gyr).value
        age_grid = t_now - t_grid
        return a_grid, age_grid, True
    else:
        print("[WARN] astropy 없음. 근사 공식 사용 (오차 ~5%). pip install astropy 권장.")
        return None, None, False


_A_GRID, _AGE_GRID, _USE_ASTROPY = _build_age_interpolator()


def scale_factor_to_stellar_age_gyr(aform_arr):
    aform_arr = np.asarray(aform_arr, dtype=float)
    if _USE_ASTROPY:
        return np.interp(np.clip(aform_arr, _A_GRID[0], _A_GRID[-1]), _A_GRID, _AGE_GRID)
    else:
        H0_inv = 977.8 / TNG_H0
        om = TNG_OM
        ol = 1.0 - om
        def _t(a):
            x = np.sqrt(ol / om) * a ** 1.5
            return (2.0 / (3.0 * np.sqrt(ol))) * H0_inv * np.log(x + np.sqrt(x**2+1))
        return np.maximum(_t(1.0) - _t(np.maximum(aform_arr, 1e-4)), 0.0)


def turnoff_mass(age_gyr):
    if age_gyr <= 0:
        return IMF_M_MAX
    return min(IMF_M_MAX, (10.0 / age_gyr) ** 0.4)


def parse_args():
    parser = argparse.ArgumentParser(
        description="TNG50 R_GC bin counts — M_G 절대등급 기반 (Gaia와 동일 조건)"
    )
    parser.add_argument("--sim",  default=DEFAULT_SIM)
    parser.add_argument("--snap", type=int,   default=DEFAULT_SNAP)
    parser.add_argument("--sid",  type=int,   default=DEFAULT_SID)
    parser.add_argument("--h",    type=float, default=DEFAULT_H)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--r-min",  type=float, default=DEFAULT_R_MIN)
    parser.add_argument("--r-max",  type=float, default=DEFAULT_R_MAX)
    parser.add_argument("--bin-w",  type=float, default=DEFAULT_BIN_W)
    parser.add_argument("--mode", choices=["cylindrical", "spherical"], default="cylindrical")
    parser.add_argument("--count-mode", choices=["surviving", "formed"], default=DEFAULT_COUNT_MODE)
    parser.add_argument("--include-wind", action="store_true")
    parser.add_argument("--cutout",      default=None)
    parser.add_argument("--out",         default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--r0",      type=float, default=DEFAULT_R0_KPC)
    parser.add_argument("--z-max",   type=float, default=DEFAULT_Z_MAX)
    parser.add_argument("--align-r", type=float, default=DEFAULT_ALIGN_R)
    parser.add_argument("--d-sun-min", type=float, default=DEFAULT_D_SUN_MIN,
                        help="태양으로부터 최소 거리 [kpc]")
    parser.add_argument("--d-sun-max", type=float, default=DEFAULT_D_SUN_MAX,
                        help="태양으로부터 최대 거리 [kpc]")
    # ★ M_G 범위 (Gaia 코드와 반드시 동일하게 설정)
    parser.add_argument("--mg-min", type=float, default=DEFAULT_MG_MIN,
                        help="M_G 하한 [mag] (밝은 쪽, 고질량). Gaia --mg-min 과 동일하게.")
    parser.add_argument("--mg-max", type=float, default=DEFAULT_MG_MAX,
                        help="M_G 상한 [mag] (어두운 쪽, 저질량). Gaia --mg-max 와 동일하게.")
    return parser.parse_args()


def get_session(api_key=None):
    api_key = os.environ.get("TNG_API_KEY") or api_key
    if not api_key:
        raise RuntimeError("TNG API key not set. 환경변수 TNG_API_KEY 또는 --api-key 사용.")
    sess = requests.Session()
    sess.headers.update({"api-key": api_key})
    return sess


def get_json(session, url, params=None, timeout=(10, 60), max_retries=8, backoff=2.0):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.HTTPError) as exc:
            last_err = exc
            retryable = True
            if isinstance(exc, requests.exceptions.HTTPError):
                code = exc.response.status_code if exc.response is not None else None
                if code is not None and (code < 500 and code != 429):
                    retryable = False
            if not retryable:
                raise
            wait = min(backoff ** attempt, 60.0)
            print(f"[WARN] get_json attempt {attempt}/{max_retries}, retry in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"get_json failed: {last_err}")


def download_cutout(session, sim, snap, sid, save_as, timeout=(10, 120),
                    max_retries=8, backoff=2.0):
    if os.path.exists(save_as) and os.path.getsize(save_as) > 0:
        return save_as
    if os.path.exists(save_as):
        os.remove(save_as)
    url    = f"{BASE_URL}{sim}/snapshots/{snap}/subhalos/{sid}/cutout.hdf5"
    params = {"stars": "Coordinates,GFM_StellarFormationTime,GFM_InitialMass,Masses"}
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp_head = session.get(url, params=params, timeout=timeout,
                                    allow_redirects=False)
            if resp_head.status_code in (301, 302, 303, 307, 308):
                redirect_url = resp_head.headers.get("Location")
                if not redirect_url:
                    raise RuntimeError("Redirect has no Location header.")
                print(f"[INFO] Redirected: {redirect_url[:80]}...")
                resp = requests.get(redirect_url, timeout=timeout, stream=True)
                resp.raise_for_status()
            else:
                resp_head.raise_for_status()
                resp = resp_head
                resp.raw.decode_content = True
            with open(save_as, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            return save_as
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.HTTPError) as exc:
            last_err = exc
            retryable = True
            if isinstance(exc, requests.exceptions.HTTPError):
                code = exc.response.status_code if exc.response is not None else None
                if code is not None and (code < 500 and code != 429):
                    retryable = False
            if not retryable:
                raise
            wait = min(backoff ** attempt, 60.0)
            print(f"[WARN] download_cutout attempt {attempt}/{max_retries}, "
                  f"retry in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"download_cutout failed: {last_err}")


def fetch_subhalo_center_kpc(session, sim, snap, sid, h):
    url  = f"{BASE_URL}{sim}/snapshots/{snap}/subhalos/{sid}"
    data = get_json(session, url)
    if all(k in data for k in ("pos_x", "pos_y", "pos_z")):
        center = np.array([data["pos_x"], data["pos_y"], data["pos_z"]], dtype=float)
    elif "pos" in data and len(data["pos"]) == 3:
        center = np.array(data["pos"], dtype=float)
    else:
        raise KeyError(f"Subhalo position not found. Keys: {sorted(data.keys())}")
    return center / h


def load_star_data(cutout_path, h, include_wind=False):
    with h5py.File(cutout_path, "r") as f:
        if "PartType4" not in f:
            raise KeyError("PartType4 not found.")
        pt     = f["PartType4"]
        coords = pt["Coordinates"][:] / h
        aform  = pt["GFM_StellarFormationTime"][:]
        m_init = pt["GFM_InitialMass"][:] * 1e10 / h
        m_now  = pt["Masses"][:] * 1e10 / h if "Masses" in pt else None
        if not include_wind:
            mask   = aform > 0
            coords = coords[mask]
            aform  = aform[mask]
            m_init = m_init[mask]
            if m_now is not None:
                m_now = m_now[mask]
    return coords, aform, m_init, m_now


def compute_disk_frame(coords, align_r):
    r    = np.sqrt((coords**2).sum(axis=1))
    mask = r < align_r
    if np.sum(mask) < 1000:
        print(f"[WARN] align_r={align_r} kpc 내 별 {np.sum(mask)}개 < 1000. 단위행렬 반환.")
        return np.eye(3)
    c = coords[mask] - np.mean(coords[mask], axis=0, keepdims=True)
    _, evecs = np.linalg.eigh(np.cov(c.T))
    z_hat = evecs[:, 0]
    x_hat = evecs[:, 2]
    if z_hat[2] < 0:
        z_hat = -z_hat
    y_hat  = np.cross(z_hat, x_hat)
    norm_y = np.linalg.norm(y_hat)
    if norm_y < 1e-10:
        y_hat = np.array([0., 1., 0.]) if abs(z_hat[1]) < 0.9 else np.array([1., 0., 0.])
        y_hat -= y_hat.dot(z_hat) * z_hat
        y_hat /= np.linalg.norm(y_hat)
    else:
        y_hat /= norm_y
    x_hat = np.cross(y_hat, z_hat)
    x_hat /= np.linalg.norm(x_hat)
    z_hat /= np.linalg.norm(z_hat)
    return np.vstack([x_hat, y_hat, z_hat])


def chabrier_imf(m):
    m   = np.asarray(m)
    imf = np.zeros_like(m, dtype=float)
    sigma, mc = 0.69, 0.079
    low  = m <= 1.0
    high = m > 1.0
    if np.any(low):
        imf[low] = (1/m[low]) * np.exp(
            -(np.log10(m[low]) - np.log10(mc))**2 / (2*sigma**2))
    if np.any(high):
        A = (1/1.0) * np.exp(-(np.log10(1.0) - np.log10(mc))**2 / (2*sigma**2))
        imf[high] = A * m[high]**-2.3
    return imf


def _prepare_imf_cdf(m_min=IMF_M_MIN, m_max=IMF_M_MAX, n=4000):
    m_grid = np.logspace(np.log10(m_min), np.log10(m_max), n)
    phi    = chabrier_imf(m_grid)
    n_cum  = np.cumsum((phi[1:] + phi[:-1]) * (m_grid[1:] - m_grid[:-1]) / 2.0)
    m_cum  = np.cumsum(
        ((phi[1:]*m_grid[1:]) + (phi[:-1]*m_grid[:-1])) *
        (m_grid[1:] - m_grid[:-1]) / 2.0
    )
    return m_grid, np.concatenate([[0.], n_cum]), np.concatenate([[0.], m_cum])


def compute_star_weights_MG(m_init, m_now_particle, aform, count_mode, mg_min, mg_max):
    """
    M_G 범위에 해당하는 별만 IMF 가중치로 카운트.

    Pecaut & Mamajek 테이블로 M_G → 질량 역산:
      m_lo = mg_to_mass(mg_max)  (어두운 등급 = 저질량)
      m_hi = mg_to_mass(mg_min)  (밝은 등급 = 고질량)

    방법 C: f_retain으로 현재 질량 기준 역산.
    surviving 모드: turnoff 질량으로 상한 추가.
    """
    m_grid, n_cum, m_cum = _prepare_imf_cdf()
    m_total = m_cum[-1]

    # M_G → 질량 역산 (어두울수록 저질량이므로 mg_max → m_lo)
    m_lo_mg = float(mg_to_mass(mg_max))   # M_G 상한(어두운 쪽) → 질량 하한
    m_hi_mg = float(mg_to_mass(mg_min))   # M_G 하한(밝은 쪽)  → 질량 상한
    m_lo_mg = np.clip(m_lo_mg, IMF_M_MIN, IMF_M_MAX)
    m_hi_mg = np.clip(m_hi_mg, IMF_M_MIN, IMF_M_MAX)

    # 방법 C: f_retain으로 현재 질량 기준 역산
    if m_now_particle is not None:
        f_retain = np.clip(m_now_particle / np.maximum(m_init, 1e-30), 1e-6, 1.0)
    else:
        f_retain = np.ones(len(m_init), dtype=float)

    m_lo = np.minimum(m_lo_mg / f_retain, IMF_M_MAX)
    m_hi = np.minimum(m_hi_mg / f_retain, IMF_M_MAX)

    if count_mode == "surviving":
        age_gyr = scale_factor_to_stellar_age_gyr(aform)
        m_to    = np.vectorize(turnoff_mass)(age_gyr)
        m_hi    = np.minimum(m_hi, m_to)

    n_lo    = np.interp(m_lo, m_grid, n_cum)
    n_hi    = np.interp(m_hi, m_grid, n_cum)
    n_range = np.maximum(n_hi - n_lo, 0.0)
    n_range = np.where(m_lo < m_hi, n_range, 0.0)

    return m_init * (n_range / m_total)


def compute_counts(radius_kpc, weights, r_min, r_max, bin_w):
    edges  = np.arange(r_min, r_max + bin_w, bin_w)
    counts, _ = np.histogram(radius_kpc, bins=edges, weights=weights)
    return edges, counts


def main():
    args = parse_args()

    if args.d_sun_min >= args.d_sun_max:
        raise ValueError("--d-sun-min must be < --d-sun-max")
    if args.mg_min >= args.mg_max:
        raise ValueError("--mg-min must be < --mg-max")

    # M_G 범위에 대응하는 질량 범위 출력 (확인용)
    m_lo_ref = mg_to_mass(args.mg_max)
    m_hi_ref = mg_to_mass(args.mg_min)
    print(
        f"[INFO] M_G 범위: [{args.mg_min:.2f}, {args.mg_max:.2f}) mag\n"
        f"[INFO] 대응 질량: [{m_lo_ref:.3f}, {m_hi_ref:.1f}) Msun "
        f"(Pecaut & Mamajek 2013)\n"
        f"[INFO] 거리 컷: {args.d_sun_min:.3f} ~ {args.d_sun_max:.3f} kpc"
    )

    out_path = args.out or (
        f"tng50_sid{args.sid}_r{args.r_min:g}to{args.r_max:g}kpc"
        f"_d{args.d_sun_min:.2f}to{args.d_sun_max:.2f}kpc"
        f"_MG{args.mg_min:.1f}to{args.mg_max:.1f}_starcounts.txt"
    )

    if args.cutout:
        cutout_path = args.cutout
    else:
        base_dir   = os.path.dirname(os.path.abspath(__file__))
        cutout_dir = os.path.join(base_dir, DEFAULT_CUTOUT_DIR)
        os.makedirs(cutout_dir, exist_ok=True)
        cutout_path = os.path.join(cutout_dir, f"sub{args.sid}_stars.hdf5")

    session    = get_session(args.api_key)
    center_kpc = fetch_subhalo_center_kpc(session, args.sim, args.snap, args.sid, args.h)

    if not os.path.exists(cutout_path):
        if args.no_download:
            raise FileNotFoundError(f"Cutout not found: {cutout_path}")
        print(f"[INFO] Downloading cutout for SID {args.sid}...")
        download_cutout(session, args.sim, args.snap, args.sid, cutout_path)

    coords_kpc, aform, m_init, m_now = load_star_data(
        cutout_path, args.h, include_wind=args.include_wind
    )

    rel      = coords_kpc - center_kpc
    basis    = compute_disk_frame(rel, args.align_r)
    rel_disk = rel @ basis.T

    sun   = np.array([args.r0, 0.0, 0.0], dtype=float)
    d_sun = np.sqrt(((rel_disk - sun)**2).sum(axis=1))

    if args.mode == "cylindrical":
        r_gc = np.sqrt(rel_disk[:, 0]**2 + rel_disk[:, 1]**2)
    else:
        r_gc = np.sqrt((rel_disk**2).sum(axis=1))

    # 선택 마스크
    z_mask = np.abs(rel_disk[:, 2]) < args.z_max
    d_mask = (d_sun >= args.d_sun_min) & (d_sun < args.d_sun_max)
    sel    = z_mask & d_mask

    print(
        f"[INFO] 전체: {len(r_gc):,}  |  |Z|<{args.z_max}kpc: {z_mask.sum():,}  |  "
        f"d_sun [{args.d_sun_min:.2f},{args.d_sun_max:.2f})kpc: {d_mask.sum():,}  |  "
        f"통합: {sel.sum():,}"
    )

    # M_G 기반 IMF 가중치 계산
    weights = compute_star_weights_MG(
        m_init, m_now, aform,
        args.count_mode, args.mg_min, args.mg_max,
    )

    r_gc_sel    = r_gc[sel]
    weights_sel = weights[sel]

    edges, counts = compute_counts(r_gc_sel, weights_sel, args.r_min, args.r_max, args.bin_w)

    n_total = float(weights_sel.sum())
    n_in    = float(counts.sum())
    f_str   = ("f_retain from Masses (방법 C)" if m_now is not None else "f_retain=1 fallback")
    age_str = ("astropy FlatLambdaCDM" if _USE_ASTROPY else "analytic approx")

    header = (
        "# TNG50 star counts — M_G 절대등급 기반 (Gaia와 동일 조건)\n"
        f"# SIM={args.sim} SNAP={args.snap} SubfindID={args.sid}\n"
        f"# Center (kpc) = [{center_kpc[0]:.6f}, {center_kpc[1]:.6f}, {center_kpc[2]:.6f}]\n"
        f"# M_G 범위: [{args.mg_min:.2f}, {args.mg_max:.2f}) mag\n"
        f"#   대응 질량: [{m_lo_ref:.3f}, {m_hi_ref:.1f}) Msun (Pecaut & Mamajek 2013)\n"
        f"# 거리 컷: d_sun [{args.d_sun_min:.3f}, {args.d_sun_max:.3f}) kpc\n"
        f"#   Gaia parallax 대응: [{1/args.d_sun_max:.4f}, {1/args.d_sun_min:.4f}) mas\n"
        f"# R_GC: [{args.r_min}, {args.r_max}) kpc, bin={args.bin_w} kpc, mode={args.mode}\n"
        f"# |Z_GC| < {args.z_max} kpc\n"
        f"# Count mode: {args.count_mode}, {f_str}\n"
        f"# Stellar age: {age_str}\n"
        f"# IMF: Chabrier(2003) [{IMF_M_MIN},{IMF_M_MAX}] Msun\n"
        f"# N_selected={sel.sum()}, N_weighted={n_total:.3e}, N_in_RGC={n_in:.3e}\n"
        "# Columns: R_low_kpc  R_high_kpc  N_star\n"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        for i in range(len(edges)-1):
            f.write(f"{edges[i]:.3f}\t{edges[i+1]:.3f}\t{counts[i]:.6e}\n")

    print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()
