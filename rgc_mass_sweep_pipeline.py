"""
rgc_mass_sweep_pipeline.py

Gaia → TNG → ratio/profile 파이프라인.
M_G 절대등급 범위를 스윕합니다.

★ 핵심 변경사항 ★
  - mass_flame 완전 제거
  - M_G 절대등급 기반으로 Gaia/TNG 동일 조건 비교
  - Gaia: M_G 범위 + 소광 보정 (a_g_gspphot)
  - TNG: Pecaut & Mamajek 테이블로 동일 M_G 범위 → 질량 역산
  - --mg-min 스윕 (--mg-max 고정)
"""

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


GAIA_OUTPUT_TEMPLATE = "gaia_dr3_Rgc_counts_MG{suffix}.txt"
GAIA_SCRIPT    = "gaia_rgccounts_fix.py"
TNG_SCRIPT     = "tng50_sid_rgccounts_gaialike_low_M.py"
COMPARE_SCRIPT = "compare_rgc_counts.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gaia → TNG → profile 파이프라인 (M_G 기반, 스윕)"
    )
    # M_G 스윕 설정
    parser.add_argument("--mg-min-start", type=float, default=-2.0,
                        help="M_G 하한 스윕 시작값 [mag] (밝은 쪽)")
    parser.add_argument("--mg-min-end",   type=float, default=8.0,
                        help="M_G 하한 스윕 종료값 [mag]")
    parser.add_argument("--mg-min-step",  type=float, default=0.5,
                        help="M_G 하한 스윕 간격 [mag]")
    parser.add_argument("--mg-max",       type=float, default=10.0,
                        help="M_G 상한 고정값 [mag] (어두운 쪽, 저질량)")

    parser.add_argument("--out-dir", default="rgc_sweep_MG_outputs")

    # TNG 설정
    parser.add_argument("--sim",  default="TNG50-1")
    parser.add_argument("--snap", type=int,   default=99)
    parser.add_argument("--sid",  type=int,   default=538905)
    parser.add_argument("--h",    type=float, default=0.6774)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--cutout",      default=None)
    parser.add_argument("--no-download", action="store_true")

    # 거리 컷 (Gaia와 TNG 동시 적용)
    parser.add_argument("--d-sun-min", type=float, default=0.1,
                        help="태양으로부터 최소 거리 [kpc]")
    parser.add_argument("--d-sun-max", type=float, default=3.5,
                        help="태양으로부터 최대 거리 [kpc]")

    parser.add_argument("--ratio", choices=["tng_over_gaia","gaia_over_tng"],
                        default="tng_over_gaia")
    parser.add_argument("--gaia-parallax-n-bins", type=int, default=20)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run",       action="store_true")
    return parser.parse_args()


def _mg_values(start, end, step):
    if step <= 0:
        raise ValueError("--mg-min-step must be positive")
    n    = int(round((end - start) / step)) + 1
    vals = [round(start + i * step, 10) for i in range(n)]
    vals = [v for v in vals if v <= end + 1e-9]
    return vals


def _mg_tag(value):
    sign = "m" if value < 0 else "p"
    return f"MG{sign}{abs(value):.1f}".replace(".", "p")


def _run_cmd(cmd, dry_run, cwd=None):
    label = f"(cwd={cwd}) " if cwd else ""
    print(f"[RUN] {label}{shlex.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _read_total_count(path):
    total = 0.0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) >= 3:
                total += float(parts[2])
    return total


def _summary_header():
    return (
        "# d_sun_min  d_sun_max  mg_min  mg_max  "
        "N_gaia  N_tng  ratio  gaia_file  tng_file  ratio_file\n"
    )


def _load_summary_rows(path):
    rows = {}
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 10:
                continue
            key = (float(parts[0]), float(parts[1]), round(float(parts[2]), 10))
            rows[key] = tuple(parts)
    return rows


def _write_summary_atomic(path, rows_by_key):
    tmp  = path.with_suffix(path.suffix + ".tmp")
    rows = [rows_by_key[k] for k in sorted(rows_by_key)]
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(_summary_header())
        for r in rows:
            f.write("\t".join(r) + "\n")
    tmp.replace(path)


def _upsert_summary_row(path, rows_by_key,
                        d_min, d_max, mg_min, mg_max,
                        n_gaia, n_tng, ratio,
                        gaia_file, tng_file, ratio_file):
    key = (d_min, d_max, round(mg_min, 10))
    rows_by_key[key] = (
        f"{d_min:.3f}", f"{d_max:.3f}",
        f"{mg_min:.2f}", f"{mg_max:.2f}",
        f"{n_gaia:.6e}", f"{n_tng:.6e}",
        f"{ratio:.6e}",
        gaia_file, tng_file, ratio_file,
    )
    _write_summary_atomic(path, rows_by_key)


def main():
    args = parse_args()

    if args.d_sun_min >= args.d_sun_max:
        raise ValueError("--d-sun-min must be < --d-sun-max")
    if args.mg_min_start >= args.mg_max:
        raise ValueError("--mg-min-start must be < --mg-max")

    plx_min = 1.0 / args.d_sun_max
    plx_max = 1.0 / args.d_sun_min

    print(
        f"[INFO] 거리 컷: {args.d_sun_min:.3f} ~ {args.d_sun_max:.3f} kpc\n"
        f"[INFO] Gaia parallax: {plx_min:.4f} ~ {plx_max:.4f} mas\n"
        f"[INFO] M_G 스윕: {args.mg_min_start:.1f} ~ {args.mg_min_end:.1f} "
        f"(step={args.mg_min_step:.1f}), 상한 고정={args.mg_max:.1f}"
    )

    base_dir       = Path(__file__).resolve().parent
    gaia_script    = base_dir / GAIA_SCRIPT
    tng_script     = base_dir / TNG_SCRIPT
    compare_script = base_dir / COMPARE_SCRIPT

    for p in [gaia_script, tng_script, compare_script]:
        if not p.exists():
            raise FileNotFoundError(f"Required script not found: {p}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mg_values    = _mg_values(args.mg_min_start, args.mg_min_end, args.mg_min_step)
    dtag         = f"_d{args.d_sun_min:.2f}to{args.d_sun_max:.2f}kpc"
    summary_path = out_dir / f"mg_sweep_summary{dtag}.txt"
    summary_rows = {} if args.dry_run else _load_summary_rows(summary_path)

    if summary_rows:
        print(f"[INFO] 기존 summary 로드: {len(summary_rows)}행")

    # Gaia 출력 파일명 패턴
    d_min = args.d_sun_min
    d_max = args.d_sun_max

    for idx, mg_min in enumerate(mg_values, 1):
        if mg_min >= args.mg_max:
            print(f"[WARN] skip mg_min={mg_min:.2f}: must be < mg_max({args.mg_max:.2f})")
            continue

        tag       = f"{_mg_tag(mg_min)}_to_{_mg_tag(args.mg_max)}{dtag}"
        gaia_out  = out_dir / f"gaia_counts_{tag}.txt"
        tng_out   = out_dir / f"tng_counts_sid{args.sid}_{tag}.txt"
        ratio_out = out_dir / f"ratio_sid{args.sid}_{tag}.txt"

        print(f"[{idx}/{len(mg_values)}] M_G_min={mg_min:.2f} ~ {args.mg_max:.2f} mag")

        row_key = (d_min, d_max, round(mg_min, 10))
        if (
            not args.dry_run
            and row_key in summary_rows
            and gaia_out.exists() and tng_out.exists() and ratio_out.exists()
        ):
            print("[INFO] checkpoint 존재, 스킵")
            continue

        if args.skip_existing and gaia_out.exists() and tng_out.exists() and ratio_out.exists():
            print("[INFO] outputs 존재, 스킵")
        else:
            # ── Gaia 실행 ──────────────────────────────────────────────────────
            gaia_suffix = (
                f"_d{d_min:.2f}to{d_max:.2f}kpc"
                f"_MG{mg_min:.1f}to{args.mg_max:.1f}"
            )
            gaia_default_name = GAIA_OUTPUT_TEMPLATE.format(suffix=gaia_suffix)

            gaia_cmd = [
                sys.executable, str(gaia_script),
                "--plx-min",          f"{plx_min:.6f}",
                "--plx-max",          f"{plx_max:.6f}",
                "--mg-min",           f"{mg_min:.2f}",
                "--mg-max",           f"{args.mg_max:.2f}",
                "--parallax-n-bins",  str(args.gaia_parallax_n_bins),
            ]
            _run_cmd(gaia_cmd, dry_run=args.dry_run, cwd=base_dir)

            if not args.dry_run:
                candidates = [base_dir / gaia_default_name, Path.cwd() / gaia_default_name]
                gaia_default_out = next((p for p in candidates if p.exists()), None)
                if gaia_default_out is None:
                    raise FileNotFoundError(
                        "Gaia output not found. Checked: "
                        + ", ".join(str(p) for p in candidates)
                    )
                shutil.copy2(gaia_default_out, gaia_out)

            # ── TNG 실행 ──────────────────────────────────────────────────────
            tng_cmd = [
                sys.executable, str(tng_script),
                "--sim",       args.sim,
                "--snap",      str(args.snap),
                "--sid",       str(args.sid),
                "--h",         str(args.h),
                "--out",       str(tng_out),
                "--d-sun-min", f"{d_min:.4f}",
                "--d-sun-max", f"{d_max:.4f}",
                "--mg-min",    f"{mg_min:.2f}",
                "--mg-max",    f"{args.mg_max:.2f}",
            ]
            if args.api_key:
                tng_cmd.extend(["--api-key", args.api_key])
            if args.cutout:
                tng_cmd.extend(["--cutout", args.cutout])
            if args.no_download:
                tng_cmd.append("--no-download")
            _run_cmd(tng_cmd, dry_run=args.dry_run, cwd=base_dir)

            # ── 비교 ──────────────────────────────────────────────────────────
            compare_cmd = [
                sys.executable, str(compare_script),
                "--gaia",  str(gaia_out),
                "--tng",   str(tng_out),
                "--out",   str(ratio_out),
                "--ratio", args.ratio,
            ]
            _run_cmd(compare_cmd, dry_run=args.dry_run, cwd=base_dir)

        if args.dry_run:
            continue

        gaia_total = _read_total_count(gaia_out)
        tng_total  = _read_total_count(tng_out)
        if args.ratio == "tng_over_gaia":
            total_ratio = tng_total / gaia_total if gaia_total != 0 else float("nan")
        else:
            total_ratio = gaia_total / tng_total if tng_total != 0 else float("nan")

        _upsert_summary_row(
            summary_path, summary_rows,
            d_min, d_max, mg_min, args.mg_max,
            gaia_total, tng_total, total_ratio,
            gaia_out.name, tng_out.name, ratio_out.name,
        )
        print(f"[INFO] checkpoint: mg_min={mg_min:.2f} ({len(summary_rows)}/{len(mg_values)} rows)")

    if args.dry_run:
        print(f"[DONE] dry-run 완료. summary 예정: {summary_path}")
        return

    print(f"[DONE] summary: {summary_path}")
    print(f"[DONE] output dir: {out_dir}")


if __name__ == "__main__":
    main()
