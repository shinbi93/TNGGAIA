"""
gaia_rgccounts_MG.py

Gaia DR3에서 R_GC bin별 별 개수를 계산합니다.

★ 핵심: mass_flame 대신 G밴드 절대등급(M_G)으로 별 선택 ★
  - gaiadr3.astrophysical_parameters JOIN 완전 제거
  - 모든 Gaia 별이 대상 (mass_flame 완결성 문제 해결)
  - 절대등급: M_G = G - (5*log10(d_kpc) + 10) - A_G
    소광: a_g_gspphot 컬럼 사용 (없으면 0으로 처리)
  - TNG 코드의 Pecaut & Mamajek 테이블과 동일한 M_G 범위 선택

선택 조건:
  1. parallax 범위: 1/d_sun_max ~ 1/d_sun_min [mas]
  2. R_GC:          5.0 ~ 8.0 kpc
  3. |Z_GC| < 2.0 kpc
  4. M_G 범위:      mg_min ~ mg_max [mag]
  5. 품질 컷:       parallax_over_error >= 5, ruwe < 1.4

Lutz-Kelker 편향:
  parallax_over_error >= 5 컷으로 ~2% 수준으로 억제.
"""

import argparse
import os
import time

import numpy as np

# ── 설정 ──────────────────────────────────────────────────────────────────────
_OUT_TXT_TEMPLATE  = "gaia_dr3_Rgc_counts_MG{suffix}.txt"
_FAIL_LOG_TEMPLATE = "gaia_rgccounts_MG_failed{suffix}.txt"

TAP_URLS = ["https://gea.esac.esa.int/tap-server/tap"]

GAIA_TABLE = "gaiadr3.gaia_source"

R0_KPC       = 8.122
R_MIN, R_MAX = 5.0, 8.0
BIN_W        = 0.5

# 기본 거리 컷 (TNG --d-sun-min/max 와 대응)
DEFAULT_PLX_MIN = 1.0 / 3.5   # mas (d_sun_max=3.5 kpc)
DEFAULT_PLX_MAX = 1.0 / 0.1   # mas (d_sun_min=0.1 kpc)

# M_G 절대등급 기본 범위
# Pecaut & Mamajek (2013) 기준:
#   M_G = 5.1  → ~1.0 Msun (K1 V, 태양 근방)
#   M_G = 16.0 → ~0.08 Msun (M8 V, 하한)
#   M_G = -7.5 → ~120 Msun (O2 V, 상한)
DEFAULT_MG_MIN = -7.5   # mag (밝은 쪽 = 고질량)
DEFAULT_MG_MAX = 10.0   # mag (어두운 쪽 = 저질량, ~0.55 Msun)

# 품질 컷
PARALLAX_OVER_ERROR_MIN = 5.0
RUWE_MAX                = 1.4
VIS_PERIODS_MIN         = 8

# 디스크 높이 컷
Z_MAX_KPC = 2.0

# 서버 안정성
MAX_RETRIES = 5
BACKOFF_SEC = 6.0

# parallax 슬라이스
PARALLAX_N_BINS    = 20
MAX_SPLIT_DEPTH    = 5
MIN_PARALLAX_WIDTH = 1.0e-4


class NonRetryableQueryError(RuntimeError):
    pass


class TimeoutSplitSignal(RuntimeError):
    pass


def _check_tap_available():
    try:
        from astroquery.utils.tap import TapPlus  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "astroquery 필요: pip install astroquery"
        ) from exc


def format_tap_error(err):
    parts = [f"{type(err).__name__}: {err}"]
    resp  = getattr(err, "response", None)
    if resp is not None:
        code = getattr(resp, "status_code", None)
        if code is not None:
            parts.append(f"status={code}")
        text = getattr(resp, "text", None)
        if text:
            snippet = " ".join(str(text).split())
            parts.append(f"body={snippet[:220]}{'...' if len(snippet)>220 else ''}")
    return " | ".join(parts)


def is_retryable_error(err):
    resp = getattr(err, "response", None)
    code = getattr(resp, "status_code", None) if resp is not None else None
    msg  = str(err).lower()
    if any(s in msg for s in ("cannot parse query", "cannot execute query",
                               "does not exist", "not acceptable")):
        return False
    if code is not None:
        return (code >= 500) or (code == 429)
    return True


def is_timeout_error(err):
    resp = getattr(err, "response", None)
    code = getattr(resp, "status_code", None) if resp is not None else None
    msg  = str(err).lower()
    return (code == 408 or "timeout" in msg or "timed out" in msg
            or "connection reset" in msg or "errno 54" in msg)


def build_count_query(p_lo, p_hi, mg_min, mg_max, snr_min):
    """
    Gaia DR3 TAP ADQL 쿼리.

    mass_flame JOIN 없이 G밴드 절대등급(M_G)으로 별 선택.
    M_G = phot_g_mean_mag - (5*LOG10(1/parallax) + 10) - COALESCE(a_g_gspphot, 0)

    소광 a_g_gspphot이 없는 별은 0으로 처리.
    """
    deg2rad = "PI()/180.0"

    query = f"""
    SELECT
      rgc_bin,
      COUNT(*) AS n_star
    FROM (
      SELECT
        FLOOR( (rgc - {R_MIN}) / {BIN_W} ) AS rgc_bin
      FROM (
        SELECT
          SQRT(
            {R0_KPC}*{R0_KPC} +
            POWER((1.0/gs.parallax) * COS(gs.b*{deg2rad}), 2) -
            2.0*{R0_KPC}*(1.0/gs.parallax)*COS(gs.b*{deg2rad})*COS(gs.l*{deg2rad})
          ) AS rgc,
          (1.0/gs.parallax) * SIN(gs.b*{deg2rad}) AS zgc,
          gs.phot_g_mean_mag
            - (5.0*LOG10(1.0/gs.parallax) + 10.0)
            - COALESCE(gs.a_g_gspphot, 0.0) AS abs_mag_g
        FROM {GAIA_TABLE} AS gs
        WHERE gs.parallax IS NOT NULL
          AND gs.parallax > 0
          AND gs.parallax_error > 0
          AND gs.parallax >= {p_lo:.8f} AND gs.parallax < {p_hi:.8f}
          AND gs.parallax_over_error >= {float(snr_min)}
          AND gs.ruwe < {float(RUWE_MAX)}
          AND gs.visibility_periods_used >= {int(VIS_PERIODS_MIN)}
          AND gs.phot_g_mean_mag IS NOT NULL
      ) AS t0
      WHERE rgc >= {R_MIN} AND rgc < {R_MAX}
        AND ABS(zgc) < {float(Z_MAX_KPC)}
        AND abs_mag_g >= {float(mg_min)}
        AND abs_mag_g < {float(mg_max)}
    ) AS t1
    GROUP BY rgc_bin
    ORDER BY rgc_bin
    """
    return query


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gaia DR3 R_GC bin counts — M_G 절대등급 기반 (mass_flame 없음)"
    )
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--max-slices",    type=int,   default=None)
    parser.add_argument("--parallax-n-bins", type=int, default=PARALLAX_N_BINS)
    parser.add_argument(
        "--plx-min", type=float, default=DEFAULT_PLX_MIN,
        help="parallax 하한 [mas] = 1/d_sun_max"
    )
    parser.add_argument(
        "--plx-max", type=float, default=DEFAULT_PLX_MAX,
        help="parallax 상한 [mas] = 1/d_sun_min"
    )
    parser.add_argument(
        "--mg-min", type=float, default=DEFAULT_MG_MIN,
        help="G밴드 절대등급 하한 [mag] (밝은 쪽, 고질량)"
    )
    parser.add_argument(
        "--mg-max", type=float, default=DEFAULT_MG_MAX,
        help="G밴드 절대등급 상한 [mag] (어두운 쪽, 저질량)"
    )
    parser.add_argument(
        "--snr-min", type=float, default=PARALLAX_OVER_ERROR_MIN,
        help="parallax_over_error 임계값 (기본 5.0)"
    )
    return parser.parse_args()


def _run_query_once(tap_urls, query, label, max_retries, backoff_sec):
    from astroquery.utils.tap import TapPlus

    taps = [TapPlus(url=u) for u in tap_urls]
    res  = None

    for attempt in range(1, max_retries + 1):
        try:
            last_err      = None
            mirror_errors = []
            saw_retryable = False
            saw_timeout   = False

            for tap_idx, tap in enumerate(taps):
                tap_label = tap_urls[tap_idx]
                try:
                    job = tap.launch_job(query, dump_to_file=False)
                    res = job.get_results()
                    break
                except Exception as e:
                    last_err = e
                    if is_timeout_error(e): saw_timeout = True
                    saw_retryable = saw_retryable or is_retryable_error(e)
                    mirror_errors.append(f"{tap_label} [sync]: {format_tap_error(e)}")
                    try:
                        job = tap.launch_job_async(query, dump_to_file=False)
                        res = job.get_results()
                        break
                    except Exception as e2:
                        last_err = e2
                        if is_timeout_error(e2): saw_timeout = True
                        saw_retryable = saw_retryable or is_retryable_error(e2)
                        mirror_errors.append(f"{tap_label} [async]: {format_tap_error(e2)}")

            if res is not None:
                return res

            if saw_timeout:
                raise TimeoutSplitSignal(
                    "timeout | " + " || ".join(mirror_errors)
                ) from last_err

            if mirror_errors:
                msg = "TAP failed | " + " || ".join(mirror_errors)
                if saw_retryable:
                    raise RuntimeError(msg) from last_err
                raise NonRetryableQueryError(msg) from last_err
            raise last_err or RuntimeError("TAP query failed")

        except (NonRetryableQueryError, TimeoutSplitSignal):
            raise
        except Exception as e:
            print(f"[WARN] {label} 시도 {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                sleep_s = min(backoff_sec * (2 ** (attempt - 1)), 120.0)
                print(f"[INFO] {sleep_s:.1f}s 대기...")
                time.sleep(sleep_s)
            else:
                raise

    raise RuntimeError("Unexpected retry loop exit")


def _rows_to_counts(rows, nb):
    counts = np.zeros(nb, dtype=np.int64)
    for b, n in rows:
        if 0 <= b < nb:
            counts[b] += n
    return counts


def _run_interval(tap_urls, p_lo, p_hi, mg_min, mg_max, snr_min, nb, depth=0):
    label = f"parallax=[{p_lo:.6f},{p_hi:.6f}) depth={depth}"
    query = build_count_query(p_lo, p_hi, mg_min, mg_max, snr_min)

    def _split(reason):
        width = p_hi - p_lo
        if (depth < MAX_SPLIT_DEPTH) and (width > MIN_PARALLAX_WIDTH):
            p_mid = float(np.sqrt(p_lo * p_hi))
            print(f"[INFO] {label} → {reason}. 분할: "
                  f"[{p_lo:.6f},{p_mid:.6f}) + [{p_mid:.6f},{p_hi:.6f})")
            c1, f1 = _run_interval(tap_urls, p_lo, p_mid, mg_min, mg_max, snr_min, nb, depth+1)
            c2, f2 = _run_interval(tap_urls, p_mid, p_hi, mg_min, mg_max, snr_min, nb, depth+1)
            return c1+c2, f1+f2
        print(f"[ERROR] {label} 최종 실패 ({reason}): 분할 불가")
        return np.zeros(nb, dtype=np.int64), [(p_lo, p_hi, depth, reason)]

    try:
        res = _run_query_once(tap_urls, query, label, MAX_RETRIES, BACKOFF_SEC)
        colnames_lower = {c.lower(): c for c in res.colnames}
        bin_key = colnames_lower.get("rgc_bin")
        n_key   = colnames_lower.get("n_star")
        if bin_key is None or n_key is None:
            raise KeyError(f"Expected rgc_bin, n_star; got {res.colnames}")
        rows = [(int(row[bin_key]), int(row[n_key])) for row in res]
        return _rows_to_counts(rows, nb), []
    except TimeoutSplitSignal:
        return _split("408 timeout → 즉시 분할")
    except NonRetryableQueryError as e:
        if is_timeout_error(e):
            return _split("timeout → 즉시 분할")
        print(f"[WARN] {label} 비재시도 오류: {e}")
        return np.zeros(nb, dtype=np.int64), [(p_lo, p_hi, depth, str(e))]
    except Exception as e:
        if is_timeout_error(e):
            return _split("timeout → 즉시 분할")
        width = p_hi - p_lo
        if (depth < MAX_SPLIT_DEPTH) and (width > MIN_PARALLAX_WIDTH):
            p_mid = float(np.sqrt(p_lo * p_hi))
            print(f"[WARN] {label} 실패. 분할: "
                  f"[{p_lo:.6f},{p_mid:.6f}) + [{p_mid:.6f},{p_hi:.6f})")
            c1, f1 = _run_interval(tap_urls, p_lo, p_mid, mg_min, mg_max, snr_min, nb, depth+1)
            c2, f2 = _run_interval(tap_urls, p_mid, p_hi, mg_min, mg_max, snr_min, nb, depth+1)
            return c1+c2, f1+f2
        print(f"[ERROR] {label} 최종 실패: {e}")
        return np.zeros(nb, dtype=np.int64), [(p_lo, p_hi, depth, str(e))]


def main():
    args = parse_args()

    if args.plx_min >= args.plx_max:
        raise ValueError("--plx-min must be < --plx-max")
    if args.mg_min >= args.mg_max:
        raise ValueError("--mg-min must be < --mg-max")

    d_max = 1.0 / args.plx_min
    d_min = 1.0 / args.plx_max

    print(
        f"[INFO] 거리 범위: {d_min:.3f} ~ {d_max:.3f} kpc\n"
        f"[INFO] parallax: {args.plx_min:.4f} ~ {args.plx_max:.4f} mas\n"
        f"[INFO] M_G 범위: {args.mg_min:.2f} ~ {args.mg_max:.2f} mag\n"
        f"[INFO] SNR_min={args.snr_min:.1f}, "
        f"Lutz-Kelker 편향 ≈ {0.5*(1/args.snr_min)**2*100:.1f}%"
    )

    suffix   = f"_d{d_min:.2f}to{d_max:.2f}kpc_MG{args.mg_min:.1f}to{args.mg_max:.1f}"
    out_txt  = _OUT_TXT_TEMPLATE.format(suffix=suffix)
    fail_log = _FAIL_LOG_TEMPLATE.format(suffix=suffix)

    edges        = np.arange(R_MIN, R_MAX + BIN_W, BIN_W)
    nb           = len(edges) - 1
    counts_total = np.zeros(nb, dtype=np.int64)

    p_edges = np.geomspace(args.plx_min, args.plx_max, args.parallax_n_bins + 1)
    slices  = [(float(p_edges[i]), float(p_edges[i+1]))
               for i in range(len(p_edges)-1)]
    if args.max_slices:
        slices = slices[:args.max_slices]

    total = len(slices)
    print(f"[INFO] 총 {total}개 parallax 슬라이스 | 출력: {out_txt}")

    if args.dry_run:
        for i, (p_lo, p_hi) in enumerate(slices, 1):
            q = build_count_query(p_lo, p_hi, args.mg_min, args.mg_max, args.snr_min)
            print(f"[{i}/{total}] parallax=[{p_lo:.6f},{p_hi:.6f})")
            print(q.strip())
        return

    _check_tap_available()

    failed = []
    for i, (p_lo, p_hi) in enumerate(slices, 1):
        print(f"[{i}/{total}] 실행: parallax=[{p_lo:.6f},{p_hi:.6f})")
        c, f = _run_interval(
            TAP_URLS, p_lo, p_hi,
            args.mg_min, args.mg_max, args.snr_min, nb
        )
        counts_total += c
        failed.extend(f)

    if failed:
        with open(fail_log, "w", encoding="utf-8") as f:
            for p_lo, p_hi, depth, msg in failed:
                f.write(f"{p_lo:.8f}\t{p_hi:.8f}\t{depth}\t"
                        f"{' '.join(str(msg).split())}\n")
        print(f"[WARN] 실패 구간: {fail_log} (n={len(failed)})")

    header = (
        "# Gaia DR3: R_GC bin counts — M_G 절대등급 기반\n"
        "# mass_flame JOIN 없음 (완결성 문제 해결)\n"
        f"# M_G = phot_g_mean_mag - (5*log10(d_kpc)+10) - COALESCE(a_g_gspphot, 0)\n"
        f"# M_G 범위: [{args.mg_min:.2f}, {args.mg_max:.2f}) mag\n"
        f"# 거리 범위: [{d_min:.3f}, {d_max:.3f}) kpc\n"
        f"#   (parallax [{args.plx_min:.4f}, {args.plx_max:.4f}) mas)\n"
        f"# RGC = sqrt(R0^2 + (d cos b)^2 - 2 R0 d cos b cos l)\n"
        f"# ZGC = d sin b; |ZGC| < {Z_MAX_KPC} kpc\n"
        f"# 품질 컷: parallax_over_error>={args.snr_min}, ruwe<{RUWE_MAX}, "
        f"visibility_periods_used>={VIS_PERIODS_MIN}\n"
        f"# R0={R0_KPC} kpc\n"
        f"# Lutz-Kelker 편향 ≈ {0.5*(1/args.snr_min)**2*100:.1f}%\n"
        "# R_low_kpc  R_high_kpc  N_star\n"
    )

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(header)
        for i in range(nb):
            f.write(f"{edges[i]:.3f}\t{edges[i+1]:.3f}\t{counts_total[i]}\n")

    print(f"[DONE] 저장: {out_txt} (실패 구간: {len(failed)})")


if __name__ == "__main__":
    main()
