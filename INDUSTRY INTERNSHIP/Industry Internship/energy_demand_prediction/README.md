# Machine Learning Based Energy Demand Prediction

A complete, runnable implementation of:

> A. Kamoona, H. Song, K. Keshavarzian, K. Levy, M. Jalili, R. Wilkinson,
> X. Yu, B. McGrath, L. Meegahapola, **"Machine learning based energy demand
> prediction"**, *Energy Reports* 9 (2023) 171-176.
> https://doi.org/10.1016/j.egyr.2023.09.151
> (RMIT University School of Engineering + AGL Energy; presented at ICSREE
> 2023, Nice; funded by C4NET; open access, CC BY-NC-ND.)

## 1. What the paper is about

Electricity demand prediction matters for consumers, distribution network
service providers and system operators, and it is complicated by auxiliary
factors such as ambient temperature. The paper's angle is deliberately
industrial: instead of chasing the highest possible accuracy with heavy deep
networks, it asks for a **simple, efficient, interpretable** model that beats
the **predefined baseline** already used in practice — specifically the
demand response (DR) baseline of Australia's national electricity market,
which is essentially an average of the same historical time stamps over
recent days. If an interpretable model forecasts closer to the actual load
profile than that baseline, the baseline can be replaced, and DR events can
be settled more fairly.

The problem is formulated as follows (Section 2 of the paper). Given a
demand time series x = {x1, ..., xT}, a time window TW and a prediction step
s, the number of training samples is n = T - TW - s + 1, and a predictor
f(x, TW) -> y_pred maps historical windows to predictions. When
weather-related information W is available the process becomes
f(x, W, TW) -> y_pred.

The proposed framework (Fig. 3 of the paper) has three steps: first,
concatenation of the historical energy data with the ambient temperature
data; second, pre-processing that splits the series into windows of size TW
to form (X, Y) batches; third, training the ML model that is then used in
the testing phase. A good DR baseline is defined by three characteristics —
accuracy, simplicity and integrity — and the proposed models satisfy them:
they are more accurate than the market baseline, they are not black boxes
(every coefficient is readable), and participants cannot easily manipulate
their consumption to mislead them.

Two interpretable predictors are proposed (Section 3), plus the market
baseline they are compared against:

* **Average baseline** — the mean of the same half-hour over the previous
  TW days (e.g. the 10:00 readings of the last ten days predict the next
  10:00 reading). This is the existing market practice.
* **Linear regression, LR (Eq. 1)** — y_hat = a1·x1 + ... + an·xn + b,
  where x1..xn is the input window (demand, optionally weather).
* **Non-linear regression, NLR (Eq. 2)** — a second-order polynomial,
  y_hat = a1·x1² + ... + an·xn² + b1·x1 + ... + bn·xn + c, squared plus
  linear terms with no cross terms, meant to capture the classically
  curved load-temperature relationship.

Accuracy is measured with the root mean square error (Eq. 3),
RMSE = sqrt( Σ(yi − ŷi)² / n ), where smaller is better.

**Case study 1 (Dataset 1).** Thirty-minute power consumption for six
Commercial & Industrial customer portfolios — Chemical Plant, Telecom,
Telecom VIC, University, Water Utility 2 and Water Utility 5 — each with
its own availability window (Table 1) and one actual DR event day that is
excluded from training and testing. Two settings are tested: energy
consumption only, and energy consumption plus the daily maximum
temperature. In the paper, LR and NLR beat the baseline for five of the six
customers, and overall (their Table 2) the baseline scores 17.88 % RMSE
versus 3.67 % for LR and 3.07 % for NLR, with NLR improving to 2.87 % when
temperature is added. The best model (NLR) is then compared with the
average baseline on the known DR event day (their Fig. 6), where it tracks
the counterfactual consumption far more closely.

**Case study 2 (Dataset 2).** A single site sampled every 30 minutes from
01/04/2018 to 31/12/2020 with exogenous weather variables (temperature,
humidity, pressure, precipitation). Training uses everything before
28/12/2020 and testing uses 28-30/12/2020. Two input geometries are
compared: **Model 1** uses the same time stamps of the previous TW days
(the baseline's geometry, optionally with exogenous variables), while
**Model 2** uses the TW consecutive half-hours immediately before the
predicted one (e.g. 7:00-9:30 to predict 10:00). Three feature scenarios
(Demand, DemandTemp, DemandTempOthers) are swept over TW in {4, 7, 10, 14}.
The paper finds that Model 2 clearly beats Model 1 — respecting the
continuity and dependency of the time series pays off — and that
NLR-DemandTemp with TW = 10 is the best predictor overall, with all ML
predictors producing values much closer to the actual load than the
baseline (their Figs. 7 and 8).

**Conclusions and future work.** Simple, interpretable ML models can
replace the market's average baseline with higher accuracy, and
weather-related factors such as temperature help further. Planned future
work is to use the tariff structure to translate the accuracy gain into a
monetary benefit. The authors declare no conflict of interest, and state
that they do not have permission to share the data.

## 2. What this repository contains

Because the original datasets are proprietary, `src/data_generator.py`
builds **synthetic stand-ins with the same structure**: Dataset 2 as a
30-minute series over the exact 2018-2020 range with temperature, humidity,
pressure, precipitation and solar radiation, and Dataset 1 as six customer
portfolios with the exact availability windows of Table 1, each containing
one injected DR event day. The demand includes daily and weekly shape,
slow level drift, a public-holiday dip, and a day-level cooling response to
the daily maximum temperature — the same exogenous variable the paper feeds
its models.

```
energy_demand_prediction/
├── main.py                  # run everything: python main.py
├── requirements.txt
├── README.md
├── src/
│   ├── data_generator.py    # synthetic Dataset 1 & Dataset 2
│   ├── preprocessing.py     # Model 1 / Model 2 windowing (Sec. 2, Fig. 3)
│   ├── models.py            # AverageBaseline, LR (Eq. 1), NLR (Eq. 2)
│   ├── evaluate.py          # RMSE (Eq. 3) and RMSE %
│   ├── run_dataset1.py      # case study 1 -> Fig. 5, Table 2, Fig. 6
│   └── run_dataset2.py      # case study 2 -> Fig. 4, Fig. 7, Fig. 8
└── outputs/                 # created by main.py
    ├── figures/             # paper-style figures (PNG)
    ├── tables/              # RMSE tables (CSV)
    ├── data/                # the generated datasets (CSV)
    └── summary.txt          # plain-text results summary
```

## 3. How to run

```
pip install -r requirements.txt
python main.py
```

The whole pipeline takes well under a minute on a laptop. Individual case
studies can be run directly with `python src/run_dataset1.py` or
`python src/run_dataset2.py`.

## 4. Results on the synthetic data

Case study 1, overall RMSE % averaged over the six portfolios (TW = 10,
DR event day and public holidays excluded, windows drawn from the last ten
eligible days as in real baseline settlement):

| Model    | Energy only | Energy + max temperature |
|----------|-------------|--------------------------|
| Baseline | 8.22 %      | 8.22 %                   |
| LR       | 7.46 %      | 6.05 %                   |
| NLR      | 7.82 %      | 6.66 %                   |

Paper values for comparison (their Table 2): Baseline 17.88 %, LR 3.67 %
(unchanged with temperature), NLR 3.07 % improving to 2.87 % with
temperature. The qualitative story reproduces: both interpretable models
beat the market baseline, and adding the daily maximum temperature improves
them further, most dramatically for the weather-sensitive University
portfolio. On the DR event day (fig6), the NLR baseline tracks the
counterfactual consumption visibly better than the average baseline, which
is the paper's central practical claim.

Case study 2, test window 28-30 Dec 2020, RMSE in kWh:

| Predictor                          | RMSE  |
|------------------------------------|-------|
| Market average baseline (TW = 10)  | 76.9  |
| Best Model 1 (NLR-Demand, TW = 7)  | 49.3  |
| Best Model 2 (NLR-DemandTemp, TW = 14) | 27.8 |

As in the paper, Model 2's consecutive-lag windows beat Model 1's
same-slot windows by a wide margin, every LR/NLR predictor beats the
baseline, and the best overall predictor is NLR with demand plus
temperature (the paper's best was NLR-DemandTemp at TW = 10; ours peaks at
TW = 14, a negligible difference on a three-day test window).

## 5. Honest deviations and lessons learned

Exact numbers differ from the paper because the data is synthetic; only the
qualitative rankings are expected to reproduce, and they do. Two findings
from the reproduction are worth knowing. First, on the very short Dataset 1
records, plain least-squares NLR pays a small variance tax for its doubled
feature count and lands within about 0.6 percentage points of LR rather
than beating it; on the much longer Dataset 2 the paper's ranking (NLR
best) reappears. Second, window hygiene matters enormously: if the
same-slot windows are allowed to contain holiday or event days, the squared
terms of Eq. 2 amplify those anomalous inputs and NLR degrades sharply —
which is precisely why real DR settlement rules select the last TW
*eligible* days, and this implementation does the same.

## 6. Using your own (real) data

For case study 2, replace `generate_dataset2()` with a `pd.read_csv` that
returns a DataFrame indexed by a 30-minute DatetimeIndex with a `demand`
column and any of `temperature`, `precipitation`, `solar_radiation` (edit
`SCENARIOS` in `run_dataset2.py` to match your columns, and adjust `SPLIT`
and the test-end constants to your dates). For case study 1, make
`generate_dataset1()` return `{customer: {"df": DataFrame(demand,
temperature), "event_day": Timestamp}}` for your own portfolios; everything
downstream is unchanged.

## 7. Reference

Kamoona, A., Song, H., Keshavarzian, K., Levy, K., Jalili, M., Wilkinson,
R., Yu, X., McGrath, B., Meegahapola, L. (2023). Machine learning based
energy demand prediction. *Energy Reports*, 9, 171-176.
https://doi.org/10.1016/j.egyr.2023.09.151
