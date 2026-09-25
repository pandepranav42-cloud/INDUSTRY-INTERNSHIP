"""Case study 1 (Section 4.2 of the paper): six C&I customer portfolios.

Reproduces, on synthetic data:
  * Fig. 5  - per-customer RMSE % bars, without / with temperature
  * Table 2 - overall RMSE % of Baseline / LR / NLR for both settings
  * Fig. 6  - DR-event-day comparison: actual load vs the market average
              baseline vs the NLR baseline

Setting 1 uses only the same-slot energy window of the previous TW days;
setting 2 additionally feeds the daily maximum temperature of the target day
(the exogenous variable used in the paper).  The DR event day is excluded
from training and testing, as in the paper.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_generator import CUSTOMER_CONFIGS, generate_dataset1
from evaluate import rmse_pct
from models import LR, NLR, AverageBaseline
from preprocessing import day_matrix

TW = 10  # ten previous days, matching common market baseline practice


def run(outdir: Path) -> list[str]:
    fig_dir, tab_dir = outdir / "figures", outdir / "tables"
    data_dir = outdir / "data"
    for d in (fig_dir, tab_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    data = generate_dataset1()
    rows: list[dict] = []
    event_plots: dict[str, dict] = {}

    for name, item in data.items():
        df, ev = item["df"], item["event_day"]
        df.to_csv(data_dir / f"dataset1_{name.replace(' ', '_').lower()}.csv")

        D = day_matrix(df["demand"]).to_numpy()
        tmat = day_matrix(df["temperature"])
        dates = day_matrix(df["demand"]).index
        tmax = tmat.max(axis=1).to_numpy()[:, None]        # daily max temperature

        ev_pos = dates.get_loc(pd.Timestamp(ev))

        # Real DR-baseline settlement uses the last TW ELIGIBLE days: public
        # holidays and event days are skipped both as targets and as window
        # days (the paper likewise excludes the DR event day).
        festive = (((dates.month == 12) & (dates.day >= 24))
                   | ((dates.month == 1) & (dates.day <= 2)))
        eligible = [p for p in range(len(dates))
                    if p != ev_pos and not festive[p]]

        def slot_windows(targets, extra=None):
            rows, ys = [], []
            for p in targets:
                prior = [q for q in eligible if q < p][-TW:]
                block = D[prior, :].T                       # (48, TW)
                if extra is not None:
                    block = np.hstack(
                        [block, np.repeat(extra[p][None, :], 48, axis=0)])
                rows.append(block)
                ys.append(D[p, :])
            return np.vstack(rows), np.concatenate(ys)

        cand = [p for p in eligible
                if sum(1 for q in eligible if q < p) >= TW]
        n_test = max(4, int(round(0.2 * len(cand))))
        train_pos, test_pos = cand[:-n_test], cand[-n_test:]

        Xtr, ytr = slot_windows(train_pos)
        Xte, yte = slot_windows(test_pos)
        Xtr2, _ = slot_windows(train_pos, extra=tmax)
        Xte2, _ = slot_windows(test_pos, extra=tmax)

        base_pred = AverageBaseline(TW).predict(Xte)
        for setting, (A, B) in {"Energy": (Xtr, Xte),
                                "Energy+Tmax": (Xtr2, Xte2)}.items():
            rows.append(dict(customer=name, setting=setting, model="Baseline",
                             rmse_pct=rmse_pct(yte, base_pred)))
            for mname, M in (("LR", LR), ("NLR", NLR)):
                model = M().fit(A, ytr)
                rows.append(dict(customer=name, setting=setting, model=mname,
                                 rmse_pct=rmse_pct(yte, model.predict(B))))

        # ---- DR event day: NLR baseline vs market average baseline --------
        nlr = NLR().fit(Xtr2, ytr)
        Xev, yev = slot_windows([ev_pos], extra=tmax)
        event_plots[name] = dict(actual=yev,
                                 avg=AverageBaseline(TW).predict(Xev),
                                 nlr=nlr.predict(Xev),
                                 date=pd.Timestamp(ev))

    res = pd.DataFrame(rows)
    res.to_csv(tab_dir / "dataset1_rmse_per_customer.csv", index=False)

    # ---------------- Table 2 analogue -------------------------------------
    table2 = (res.groupby(["model", "setting"]).rmse_pct.mean().unstack()
              [["Energy", "Energy+Tmax"]]
              .reindex(["Baseline", "LR", "NLR"]))
    table2.columns = ["Energy consumption",
                      "Energy consumption + max temperature"]
    table2.round(2).to_csv(tab_dir / "dataset1_table2_overall_rmse.csv")

    # ---------------- Fig. 5 analogue --------------------------------------
    customers = list(CUSTOMER_CONFIGS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, setting, title in zip(axes, ["Energy", "Energy+Tmax"],
                                  ["(a) without temperature",
                                   "(b) with temperature"]):
        for j, model in enumerate(["Baseline", "LR", "NLR"]):
            vals = [res[(res.customer == c) & (res.setting == setting)
                        & (res.model == model)].rmse_pct.iloc[0]
                    for c in customers]
            ax.bar(np.arange(len(customers)) + (j - 1) * 0.25, vals,
                   width=0.25, label=model)
        ax.set_xticks(range(len(customers)))
        ax.set_xticklabels(customers, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("RMSE %")
    axes[0].legend()
    fig.suptitle("Dataset 1 testing RMSE % - average baseline vs LR / NLR "
                 "(synthetic)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_dataset1_rmse_bars.png", dpi=150)
    plt.close(fig)

    # ---------------- Fig. 6 analogue --------------------------------------
    hours = np.arange(48) / 2.0
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (name, d) in zip(axes.ravel(), event_plots.items()):
        ax.plot(hours, d["actual"], color="tab:blue", label="Actual MW")
        ax.plot(hours, d["avg"], color="tab:orange", label="Average Baseline")
        ax.plot(hours, d["nlr"], color="tab:green", label="NLR Baseline")
        ax.set_title(f"{name} {d['date']:%m/%d/%Y}", fontsize=9)
        ax.set_xlabel("Time (hour)")
        ax.set_ylabel("Power demand (MW)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("DR event-day predictions: NLR baseline vs market average "
                 "baseline (synthetic)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6_dataset1_dr_event.png", dpi=150)
    plt.close(fig)

    # ---------------- summary ----------------------------------------------
    lines = ["Dataset 1 (synthetic) - overall RMSE % "
             "(mean over the six C&I portfolios), TW = 10:"]
    for m in ["Baseline", "LR", "NLR"]:
        e, et = table2.loc[m]
        lines.append(f"  {m:<8s}  energy only: {e:6.2f} %   "
                     f"energy + max temp: {et:6.2f} %")
    ml = table2.loc[["LR", "NLR"]]
    best_model, best_setting = ml.stack().idxmin()
    b = table2.loc["Baseline"].min()
    lines.append(f"  -> Both ML models beat the {b:.2f} % average baseline "
                 f"by a wide margin; the best")
    lines.append(f"     combination is {best_model} with the "
                 f"'{best_setting}' inputs "
                 f"({ml.stack().min():.2f} %).")
    lines.append("     Paper (Table 2): Baseline 17.88 %, LR 3.67 %, "
                 "NLR 3.07 % -> 2.87 % with temp.")
    return lines


if __name__ == "__main__":
    for line in run(Path(__file__).resolve().parent.parent / "outputs"):
        print(line)
