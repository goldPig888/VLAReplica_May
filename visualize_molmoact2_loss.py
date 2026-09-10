#!/usr/bin/env python3
"""Generate a self-contained HTML convergence report from a MolmoAct2 log."""

from __future__ import annotations

import argparse
import html
import re
import statistics
import webbrowser
from pathlib import Path


LOSS_RE = re.compile(
    r"\[step=(\d+)/(\d+)[^\n]*\]\s*\n\s*train/action_flow_loss="
    r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)",
    re.IGNORECASE,
)


def parse_losses(path: Path) -> tuple[list[tuple[int, float]], int]:
    text = path.read_text(errors="replace")
    by_step: dict[int, float] = {}
    totals: list[int] = []
    for step, total, loss in LOSS_RE.findall(text):
        by_step[int(step)] = float(loss)
        totals.append(int(total))
    if not by_step:
        raise ValueError(
            f"No '[step=N/TOTAL]' followed by 'train/action_flow_loss=...' records found in {path}"
        )
    points = sorted(by_step.items())
    return points, max(totals, default=points[-1][0])


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[max(0, min(index, len(ordered) - 1))]


def rolling_mean(points: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    result: list[tuple[int, float]] = []
    running_sum = 0.0
    queue: list[float] = []
    for step, value in points:
        queue.append(value)
        running_sum += value
        if len(queue) > window:
            running_sum -= queue.pop(0)
        result.append((step, running_sum / len(queue)))
    return result


def linear_slope(points: list[tuple[int, float]]) -> float:
    if len(points) < 2:
        return 0.0
    x_mean = mean([float(x) for x, _ in points])
    y_mean = mean([y for _, y in points])
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator


def summarize_bins(points: list[tuple[int, float]], total: int, bin_steps: int) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for start in range(0, total, bin_steps):
        end = min(total, start + bin_steps)
        values = [loss for step, loss in points if start < step <= end]
        if not values:
            continue
        rows.append(
            {
                "start": start + 1,
                "end": end,
                "count": len(values),
                "mean": mean(values),
                "median": statistics.median(values),
                "q10": percentile(values, 0.10),
                "q90": percentile(values, 0.90),
            }
        )
    return rows


def svg_chart(
    raw: list[tuple[int, float]], smoothed: list[tuple[int, float]], total: int
) -> str:
    width, height = 1120, 500
    left, right, top, bottom = 76, 28, 28, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    losses = [value for _, value in raw]
    y_max = max(percentile(losses, 0.99), max(value for _, value in smoothed), 1e-6) * 1.08

    def xy(step: int, value: float) -> tuple[float, float]:
        x = left + plot_w * step / max(total, 1)
        clipped = min(max(value, 0.0), y_max)
        y = top + plot_h * (1.0 - clipped / y_max)
        return x, y

    raw_path = " ".join(
        ("M" if index == 0 else "L") + f"{x:.2f},{y:.2f}"
        for index, (step, value) in enumerate(raw)
        for x, y in [xy(step, value)]
    )
    smooth_path = " ".join(
        ("M" if index == 0 else "L") + f"{x:.2f},{y:.2f}"
        for index, (step, value) in enumerate(smoothed)
        for x, y in [xy(step, value)]
    )

    grid: list[str] = []
    for index in range(6):
        value = y_max * (5 - index) / 5
        y = top + plot_h * index / 5
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>'
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{value:.3f}</text>'
        )
    for index in range(6):
        step = round(total * index / 5)
        x = left + plot_w * index / 5
        grid.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="grid"/>'
            f'<text x="{x:.1f}" y="{top + plot_h + 28}" text-anchor="middle">{step:,}</text>'
        )

    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Action flow loss by training step">
      <style>
        .grid {{ stroke: #d7deea; stroke-width: 1; }}
        text {{ fill: #536179; font: 13px system-ui, sans-serif; }}
      </style>
      {''.join(grid)}
      <path d="{raw_path}" fill="none" stroke="#9db8df" stroke-width="1" opacity="0.38"/>
      <path d="{smooth_path}" fill="none" stroke="#1769e0" stroke-width="3"/>
      <text x="{left + plot_w / 2:.1f}" y="{height - 8}" text-anchor="middle">Optimizer step</text>
      <text x="18" y="{top + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2:.1f})">Action-flow loss</text>
    </svg>
    """


def build_report(path: Path, points: list[tuple[int, float]], total: int, window: int, bin_steps: int) -> str:
    values = [value for _, value in points]
    smooth = rolling_mean(points, window)
    tail_count = max(20, len(points) // 8)
    prior = points[-2 * tail_count : -tail_count]
    final = points[-tail_count:]
    prior_mean = mean([value for _, value in prior]) if prior else values[0]
    final_mean = mean([value for _, value in final])
    final_median = statistics.median(value for _, value in final)
    change = 100.0 * (final_mean - prior_mean) / prior_mean if prior_mean else 0.0
    slope_per_1k = linear_slope(final) * 1000
    relative_slope = 100.0 * slope_per_1k / final_mean if final_mean else 0.0

    if abs(relative_slope) <= 5:
        verdict = "Near plateau"
        verdict_detail = "The fitted trend over the final 12.5% changes by no more than 5% of the final mean per 1K steps."
        verdict_class = "plateau"
    elif relative_slope < 0:
        verdict = "Still decreasing"
        verdict_detail = "The final fitted trend is still downward; use held-out validation or robot success before extending training."
        verdict_class = "improving"
    else:
        verdict = "Not improving"
        verdict_detail = "The final fitted trend is flat-to-upward; more training is not supported by training loss alone."
        verdict_class = "warning"

    rows = summarize_bins(points, total, bin_steps)
    table_rows = "".join(
        "<tr>"
        f"<td>{int(row['start']):,}–{int(row['end']):,}</td>"
        f"<td>{int(row['count']):,}</td>"
        f"<td>{row['mean']:.6f}</td>"
        f"<td>{row['median']:.6f}</td>"
        f"<td>{row['q10']:.6f}</td>"
        f"<td>{row['q90']:.6f}</td>"
        "</tr>"
        for row in rows
    )
    initial_count = min(100, len(values))
    initial_mean = mean(values[:initial_count])
    reduction = 100.0 * (initial_mean - final_mean) / initial_mean if initial_mean else 0.0
    interval = statistics.median(
        [b[0] - a[0] for a, b in zip(points, points[1:])]
    ) if len(points) > 1 else 0

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MolmoAct2 loss convergence</title>
  <style>
    :root {{ color-scheme: light dark; --bg:#f3f6fb; --card:#fff; --ink:#172033; --muted:#637089; --line:#dbe2ec; --blue:#1769e0; }}
    @media (prefers-color-scheme: dark) {{ :root {{ --bg:#101520; --card:#192131; --ink:#eef4ff; --muted:#aab7ca; --line:#344055; --blue:#6ca5ff; }} }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,sans-serif; }}
    main {{ max-width:1200px; margin:auto; padding:32px 20px 64px; }} h1 {{ margin:0 0 4px; }} .subtitle {{ color:var(--muted); margin:0 0 24px; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(175px,1fr)); gap:12px; }}
    .card,.panel {{ background:var(--card); border:1px solid var(--line); border-radius:14px; box-shadow:0 4px 18px #0000000b; }}
    .card {{ padding:16px; }} .label {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em; }}
    .value {{ font-size:25px; font-weight:720; margin-top:4px; }} .panel {{ margin-top:16px; padding:20px; overflow:auto; }}
    .verdict {{ border-left:5px solid var(--blue); }} .verdict.plateau {{ border-left-color:#e89b19; }} .verdict.warning {{ border-left-color:#db4b4b; }}
    svg {{ width:100%; min-width:720px; height:auto; }} .legend {{ display:flex; gap:20px; color:var(--muted); font-size:13px; }}
    .swatch {{ display:inline-block; width:24px; height:3px; vertical-align:middle; margin-right:7px; background:#1769e0; }} .swatch.raw {{ background:#9db8df; opacity:.7; }}
    table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:9px 12px; text-align:right; border-bottom:1px solid var(--line); }} th:first-child,td:first-child {{ text-align:left; }} th {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
    code {{ background:#00000012; padding:2px 5px; border-radius:5px; }}
  </style>
</head>
<body><main>
  <h1>MolmoAct2 loss convergence</h1>
  <p class="subtitle">Source: {html.escape(str(path))} · generated from {len(points):,} measurements</p>
  <section class="cards">
    <div class="card"><div class="label">Completed step</div><div class="value">{points[-1][0]:,}/{total:,}</div></div>
    <div class="card"><div class="label">Final loss</div><div class="value">{values[-1]:.4f}</div></div>
    <div class="card"><div class="label">Final-window mean</div><div class="value">{final_mean:.4f}</div></div>
    <div class="card"><div class="label">Initial → final reduction</div><div class="value">{reduction:.1f}%</div></div>
  </section>
  <section class="panel verdict {verdict_class}"><h2>{verdict}</h2><p>{verdict_detail}</p>
    <p>Final-window median: <strong>{final_median:.6f}</strong>; change versus preceding window: <strong>{change:+.1f}%</strong>; fitted slope: <strong>{slope_per_1k:+.6f}</strong> loss per 1K steps.</p>
    <p>Training loss cannot measure robot task success or held-out generalization. Check validation loss and physical benchmark success before training longer.</p>
  </section>
  <section class="panel"><h2>Loss curve</h2><div class="legend"><span><i class="swatch raw"></i>Raw logged loss</span><span><i class="swatch"></i>Trailing mean ({window:,} measurements ≈ {window * interval:,.0f} steps)</span></div>
    {svg_chart(points, smooth, total)}
  </section>
  <section class="panel"><h2>Training-stage summary</h2><table><thead><tr><th>Steps</th><th>Samples</th><th>Mean</th><th>Median</th><th>10th pct.</th><th>90th pct.</th></tr></thead><tbody>{table_rows}</tbody></table></section>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Full training log or extracted loss-lines text file.")
    parser.add_argument("--output", type=Path, default=Path("outputs/molmoact2-loss-report.html"))
    parser.add_argument("--window", type=int, default=100, help="Trailing-average window in measurements.")
    parser.add_argument("--bin-steps", type=int, default=5000, help="Step width used in the summary table.")
    parser.add_argument("--open", action="store_true", help="Open the generated report in the default browser.")
    args = parser.parse_args()
    if args.window < 1 or args.bin_steps < 1:
        parser.error("--window and --bin-steps must be positive")

    points, total = parse_losses(args.log)
    report = build_report(args.log, points, total, args.window, args.bin_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Parsed {len(points):,} loss measurements through step {points[-1][0]:,}/{total:,}.")
    print(f"Report written to: {args.output.resolve()}")
    if args.open:
        webbrowser.open(args.output.resolve().as_uri())


if __name__ == "__main__":
    main()
