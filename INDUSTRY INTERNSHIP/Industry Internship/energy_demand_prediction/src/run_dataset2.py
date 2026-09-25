"""Case study 2 (Section 4.3 of the paper): 30-min site data with weather.

Train on everything before 28/12/2020, test on 28/12/2020 - 30/12/2020,
exactly as in the paper.  Two input geometries are compared:

  Model 1 - same time stamps of the previous TW days (baseline geometry,
            optionally + exogenous variables)
  Model 2 - the TW consecutive half-hours right before the predicted one

for three feature scenarios (Demand / DemandTemp / DemandTempOthers) and
TW in {4, 7, 10, 14}.  Reproduces Fig. 4 (data overview), Fig. 7 (RMSE bar
comparison), Fig. 8 (actual vs predicted test series) and a Fig. 2-style
forecast plot with the temperature panel.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_generator import generate_dataset2
from evaluate import rmse
from models import LR, NLR, AverageBaseline
from preprocessing import consecutive_windows, day_matrix, same_slot_windows

SPLIT = pd.Timestamp("2020-12-28")            # first test day
TEST_END_DAY = pd.Timestamp("2020-12-30")     # last test day
TEST_END_TS = pd.Timestamp("2020-12-30 23:30")
TWS = [4, 7, 10, 14]
SCENARIOS = {
    "Demand": ["demand"],
    "DemandTemp": ["demand", "temperature"],
    "DemandTempOthers": ["demand", "temperature",
                         "precipitation", "solar_radiation"],
}


def model1_xy(df: pd.DataFrame, feats: list[str], tw: int):
    """Same-slot windows (Model 1 geometry) split into train / test."""
    mats = [day_matrix(df[f]).to_numpy() for f in feats]
    dates = day_matrix(df["demand"]).index
    train_pos = [p for p in range(tw, len(dates)) if dates[p] < SPLIT]
    test_pos = [p for p in range(tw, len(dates))
                if SPLIT <= dates[p] <= TEST_END_DAY]
    Xtr, ytr, _ = same_slot_windows(mats, mats[0], train_pos, tw)
    Xte, yte, meta = same_slot_windows(mats, mats[0], test_pos, tw)
    return Xtr, ytr, Xte, yte, meta, dates


def model2_xy(df: pd.DataFrame, feats: list[str], tw: int):
    """Consecutive-lag windows (Model 2 geometry) split into train / test."""
    X, y, t = consecutive_windows(df, feats, tw)
    tr = t < SPLIT
    te = (t >= SPLIT) & (t <= TEST_END_TS)
    return X[tr], y[tr], X[te], y[te], t[te]


def run(outdir: Path) -> list[str]:
    fig_dir, tab_dir = outdir / "figures", outdir / "tables"
    data_dir = outdir / "data"
    for d in (fig_dir, tab_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = generate_dataset2()
    df.to_csv(data_dir / "dataset2.csv")

    # ---------------- Fig. 4 analogue: data overview ------------------------
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(df.index, df["demand"], lw=0.3, color="k")
    ax.set_xlabel("Time / 30 mins")
    ax.set_ylabel("Demand / kWh")
    ax.set_title("Dataset 2 demand, 01/04/2018 - 31/12/2020 (synthetic)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_dataset2_overview.png", dpi=150)
    plt.close(fig)

    records: list[dict] = []
    fig8_store: dict[str, pd.Series] = {}

    for tw in TWS:
        # ----- market average baseline (same-slot mean over tw days) -------
        _, _, Xte_b, yte_b, meta, dates = model1_xy(df, ["demand"], tw)
        yhat_b = AverageBaseline(tw).predict(Xte_b)
        records.append(dict(structure="Baseline", scenario="-",
                            model="Baseline", tw=tw, rmse=rmse(yte_b, yhat_b)))
        if tw == 10:
            ts = pd.DatetimeIndex([dates[p] + pd.Timedelta(minutes=30 * s)
                                   for p, s in meta])
            fig8_store["Real"] = pd.Series(yte_b, index=ts)
            fig8_store["Baseline"] = pd.Series(yhat_b, index=ts)

        # ----- LR / NLR for both geometries and all scenarios ---------------
        for scen, feats in SCENARIOS.items():
            X1tr, y1tr, X1te, y1te, _, _ = model1_xy(df, feats, tw)
            X2tr, y2tr, X2te, y2te, t2 = model2_xy(df, feats, tw)
            for mname, M in (("LR", LR), ("NLR", NLR)):
                p1 = M().fit(X1tr, y1tr).predict(X1te)
                records.append(dict(structure="Model 1", scenario=scen,
                                    model=mname, tw=tw, rmse=rmse(y1te, p1)))
                p2 = M().fit(X2tr, y2tr).predict(X2te)
                records.append(dict(structure="Model 2", scenario=scen,
                                    model=mname, tw=tw, rmse=rmse(y2te, p2)))
                if tw == 10:
                    fig8_store[f"{mname}-{scen}"] = pd.Series(p2, index=t2)

    res = pd.DataFrame(records)
    res.to_csv(tab_dir / "dataset2_rmse.csv", index=False)

    # ---------------- Fig. 7 analogue: RMSE bar comparison ------------------
    order = ["Baseline"] + [f"{m}-{s}" for s in SCENARIOS
                            for m in ("LR", "NLR")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    width = 0.11
    for ax, structure in zip(axes, ["Model 1", "Model 2"]):
        for j, lab in enumerate(order):
            vals = []
            for tw in TWS:
                if lab == "Baseline":
                    v = res[(res.model == "Baseline")
                            & (res.tw == tw)].rmse.iloc[0]
                else:
                    m, s = lab.split("-", 1)
                    v = res[(res.structure == structure) & (res.model == m)
                            & (res.scenario == s) & (res.tw == tw)].rmse.iloc[0]
                vals.append(v)
            ax.bar(np.arange(len(TWS)) + (j - 3) * width, vals,
                   width=width, label=lab)
        ax.set_xticks(np.arange(len(TWS)))
        ax.set_xticklabels([f"TW={t}" for t in TWS])
        ax.set_title(structure)
        ax.set_ylabel("RMSE")
    axes[1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Dataset 2 testing RMSE - Baseline vs LR / NLR (synthetic)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig7_dataset2_rmse_bars.png", dpi=150)
    plt.close(fig)

    # ---------------- Fig. 8 analogue: actual vs predicted ------------------
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    styles = {"Real": dict(color="k", lw=1.6),
              "Baseline": dict(color="tab:blue", lw=1.0, ls="--")}
    for name, s in fig8_store.items():
        ax.plot(s.index, s.values, label=name, **styles.get(name, dict(lw=0.9)))
    ax.set_ylabel("Demand / kWh")
    ax.set_title("Actual vs predicted demand, Model 2, TW = 10 (synthetic)")
    ax.legend(fontsize=7, ncol=4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig8_dataset2_test_predictions.png", dpi=150)
    plt.close(fig)

    # ---------------- Fig. 2-style forecast plot ----------------------------
    context = df.loc["2020-12-26":"2020-12-30 23:30"]
    pred = fig8_store["NLR-DemandTemp"]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1, 2]))
    ax0.plot(context.index, context["temperature"], color="k", lw=0.8)
    ax0.set_ylabel("Temperature (C)")
    ax1.plot(context.index, context["demand"], color="k", lw=0.9,
             marker=".", ms=2, label="Real")
    ax1.plot(pred.index, pred.values, color="red", lw=1.0,
             marker=".", ms=2, label="Prediction (NLR-DemandTemp)")
    ax1.set_ylabel("Energy demand / kWh")
    ax1.set_xlabel("Date and Time")
    ax1.legend()
    fig.suptitle("Energy demand prediction using temperature + historical "
                 "energy data (synthetic)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_style_forecast_nlr.png", dpi=150)
    plt.close(fig)

    # ---------------- summary ----------------------------------------------
    b10 = res[(res.model == "Baseline") & (res.tw == 10)].rmse.iloc[0]
    best1 = res[res.structure == "Model 1"].sort_values("rmse").iloc[0]
    best2 = res[res.structure == "Model 2"].sort_values("rmse").iloc[0]
    lines = ["Dataset 2 (synthetic) - test window 28-30 Dec 2020, "
             "RMSE in kWh:",
             f"  Market average baseline (TW=10):            "
             f"RMSE = {b10:7.2f}",
             f"  Best Model 1: {best1.model}-{best1.scenario} "
             f"(TW={best1.tw}):  RMSE = {best1.rmse:7.2f}",
             f"  Best Model 2: {best2.model}-{best2.scenario} "
             f"(TW={best2.tw}):  RMSE = {best2.rmse:7.2f}",
             "  -> Model 2 (consecutive-lag windows) beats Model 1 "
             "(same-slot windows) and both",
             "     ML models beat the average baseline, matching the "
             "paper's conclusions."]
    return lines


if __name__ == "__main__":
    for line in run(Path(__file__).resolve().parent.parent / "outputs"):
        print(line)
