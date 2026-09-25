"""Synthetic stand-ins for the two proprietary datasets used in

    A. Kamoona et al., "Machine learning based energy demand prediction",
    Energy Reports 9 (2023) 171-176. doi:10.1016/j.egyr.2023.09.151

The authors state that they "do not have permission to share data", so this
module generates statistically similar data with the same structure:

* Dataset 2 - one site sampled every 30 minutes from 01/04/2018 to
  31/12/2020, with exogenous weather variables (temperature, humidity,
  pressure, precipitation, solar radiation).  See Section 4.1 / Fig. 4.
* Dataset 1 - six Commercial & Industrial (C&I) customer portfolios, each
  with its own availability window (Table 1) and one demand-response (DR)
  event day (Fig. 6).

Everything is deterministic given a seed, so runs are reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SLOTS_PER_DAY = 48  # 30-minute resolution


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _ar1(n: int, rho: float, sigma: float, rng) -> np.ndarray:
    """AR(1) process - used for synoptic weather swings and load-level drift."""
    x = np.empty(n)
    x[0] = 0.0
    eps = rng.normal(0.0, sigma, n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    return x


def _temperature(index: pd.DatetimeIndex, rng, mean: float = 15.5,
                 seasonal: float = 9.0, diurnal: float = 4.5,
                 syn_rho: float = 0.995, syn_sigma: float = 0.35) -> np.ndarray:
    """Melbourne-like ambient temperature (southern hemisphere seasons)."""
    doy = index.dayofyear.to_numpy()
    hod = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    seas = seasonal * np.cos(2.0 * np.pi * (doy - 20) / 365.25)   # peak ~20 Jan
    diur = diurnal * np.sin(2.0 * np.pi * (hod - 9.5) / 24.0)     # peak ~15:30
    synoptic = _ar1(len(index), syn_rho, syn_sigma, rng)          # weather fronts
    return mean + seas + diur + synoptic + rng.normal(0, 0.5, len(index))


# ---------------------------------------------------------------------------
# Dataset 2  (single site + rich weather covariates)
# ---------------------------------------------------------------------------
def generate_dataset2(start: str = "2018-04-01", end: str = "2020-12-31",
                      seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, f"{end} 23:30", freq="30min")
    n = len(index)
    n_days = n // SLOTS_PER_DAY
    hod = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    dow = index.dayofweek.to_numpy()
    doy = index.dayofyear.to_numpy()

    temp = _temperature(index, rng)

    # --- weather covariates ------------------------------------------------
    rain_day = rng.random(n_days) < 0.28
    rain = np.zeros(n)
    for d in np.where(rain_day)[0]:
        slots = rng.choice(SLOTS_PER_DAY, size=rng.integers(3, 12), replace=False)
        rain[d * SLOTS_PER_DAY + slots] = rng.gamma(2.0, 1.2, len(slots))
    clear_sky = np.clip(np.sin(np.pi * (hod - 6.0) / 13.0), 0, None)  # 06h-19h
    season_sun = 0.75 + 0.25 * np.cos(2 * np.pi * (doy - 20) / 365.25)
    solar = 950 * clear_sky * season_sun * (1 - 0.75 * (rain > 0))
    solar = np.clip(solar + rng.normal(0, 15, n), 0, None)
    humidity = np.clip(62 - 1.1 * (temp - 15) + 18 * (rain > 0)
                       + rng.normal(0, 5, n), 20, 100)
    pressure = 1014 + _ar1(n, 0.997, 0.15, rng)

    # --- demand ------------------------------------------------------------
    profile = (0.62
               + 0.22 * np.exp(-0.5 * ((hod - 8.5) / 2.0) ** 2)    # morning peak
               + 0.34 * np.exp(-0.5 * ((hod - 18.5) / 2.6) ** 2)   # evening peak
               - 0.10 * np.exp(-0.5 * ((hod - 3.0) / 2.2) ** 2))   # night trough
    daytype = np.where(dow == 5, 0.86, np.where(dow == 6, 0.80, 1.0))
    cooling = 0.016 * np.clip(temp - 22.0, 0, None) ** 1.6
    heating = 0.011 * np.clip(12.0 - temp, 0, None) ** 1.6
    level = np.repeat(1.0 + 0.10 * _ar1(n_days, 0.90, 0.5, rng), SLOTS_PER_DAY)[:n]

    demand = 640.0 * profile * daytype * level * (1.0 + cooling + heating)
    demand += rng.normal(0, 16, n)
    # occasional short dips, for visual similarity with Fig. 4
    k = rng.choice(n, size=int(n * 0.002), replace=False)
    demand[k] *= rng.uniform(0.3, 0.7, len(k))

    return pd.DataFrame({"demand": demand, "temperature": temp,
                         "humidity": humidity, "pressure": pressure,
                         "precipitation": rain, "solar_radiation": solar},
                        index=index)


# ---------------------------------------------------------------------------
# Dataset 1  (six C&I customer portfolios, Table 1 + Fig. 6 of the paper)
# ---------------------------------------------------------------------------
# event = (start hour, end hour, fractional load drop during the DR event)
# temp_sens = day-level cooling sensitivity to the DAILY MAX temperature -
# this is exactly the exogenous variable the paper feeds to the models.
CUSTOMER_CONFIGS: dict[str, dict] = {
    "Chemical Plant": dict(start="2020-11-10", end="2021-01-14", base=11.0,
                           shape="flat", weekend=0.98, temp_sens=0.10,
                           noise=0.010, level_amp=0.05,
                           event_day="2021-01-14", event=(13.0, 20.0, 0.88)),
    "Telecom": dict(start="2020-11-10", end="2021-06-10", base=19.5,
                    shape="flat", weekend=1.00, temp_sens=0.04,
                    noise=0.006, level_amp=0.03,
                    event_day="2021-06-10", event=(14.0, 20.0, 0.68)),
    "Telecom VIC": dict(start="2021-01-01", end="2021-05-25", base=14.3,
                        shape="flat", weekend=1.00, temp_sens=0.05,
                        noise=0.007, level_amp=0.035,
                        event_day="2021-05-20", event=(15.0, 20.0, 0.42)),
    "University": dict(start="2019-12-22", end="2020-01-31", base=2.3,
                       shape="campus", weekend=0.55, temp_sens=0.45,
                       noise=0.030, level_amp=0.09,
                       event_day="2020-01-31", event=(11.0, 17.0, 0.38)),
    "Water Utility 2": dict(start="2019-12-22", end="2020-01-31", base=7.8,
                            shape="water", weekend=0.92, temp_sens=0.10,
                            noise=0.035, level_amp=0.06,
                            event_day="2020-01-31", event=(13.0, 19.0, 0.55)),
    "Water Utility 5": dict(start="2020-01-01", end="2020-01-31", base=0.32,
                            shape="water", weekend=0.90, temp_sens=0.10,
                            noise=0.050, level_amp=0.08,
                            event_day="2020-01-31", event=(12.0, 18.0, 0.85)),
}


def _shape(hod: np.ndarray, kind: str) -> np.ndarray:
    if kind == "flat":        # continuous industrial process / data centre
        return 0.96 + 0.04 * np.sin(2 * np.pi * (hod - 10) / 24)
    if kind == "campus":      # occupancy-driven office / university load
        return 0.35 + 0.65 * np.exp(-0.5 * ((hod - 13.0) / 3.4) ** 2)
    if kind == "water":       # pumping peaks morning / evening
        return (0.62 + 0.24 * np.exp(-0.5 * ((hod - 7.5) / 1.8) ** 2)
                + 0.20 * np.exp(-0.5 * ((hod - 18.5) / 2.2) ** 2))
    raise ValueError(kind)


def generate_dataset1(seed: int = 11) -> dict[str, dict]:
    """Returns {customer: {"df": DataFrame(demand, temperature),
                           "event_day": Timestamp}}  (demand in MW)."""
    out: dict[str, dict] = {}
    for i, (name, cfg) in enumerate(CUSTOMER_CONFIGS.items()):
        r = np.random.default_rng(seed + 100 * i)
        index = pd.date_range(cfg["start"], f"{cfg['end']} 23:30", freq="30min")
        n = len(index)
        n_days = n // SLOTS_PER_DAY
        hod = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
        dow = index.dayofweek.to_numpy()
        dts = index.normalize()

        temp = _temperature(index, r, mean=16.0, syn_rho=0.99, syn_sigma=0.22)
        # Day-level thermal load driven by the DAILY MAX temperature (the
        # exogenous variable of the paper's setting 2).  The response is the
        # classic U-shaped quadratic around a comfort point - precisely the
        # relation that motivates the second-order NLR model of Eq. (2).
        tmax_day = temp.reshape(n_days, SLOTS_PER_DAY).max(axis=1)
        dev = np.clip(tmax_day - 23.0, 0, None) / 10.0   # cooling above 23 C
        thermal = np.repeat(1.0 + cfg["temp_sens"] * dev ** 2,
                            SLOTS_PER_DAY)[:n]
        daytype = np.where(dow >= 5, cfg["weekend"], 1.0)
        # festive-season dip: a non-stationarity that breaks the naive
        # average baseline, exactly the situation the paper targets
        festive = np.where(((dts.month == 12) & (dts.day >= 24))
                           | ((dts.month == 1) & (dts.day <= 2)), 0.78, 1.0)
        level = 1.0 + cfg["level_amp"] * np.repeat(
            _ar1(n_days, 0.75, 0.6, r), SLOTS_PER_DAY)[:n]

        demand = (cfg["base"] * _shape(hod, cfg["shape"]) * daytype
                  * festive * level * thermal)
        demand *= 1.0 + r.normal(0, cfg["noise"], n)

        # ----- inject the DR event ----------------------------------------
        ev = pd.Timestamp(cfg["event_day"])
        h0, h1, drop = cfg["event"]
        on_event = (dts == ev) & (hod >= h0) & (hod < h1)
        demand = demand.copy()
        demand[on_event] *= (1.0 - drop)

        out[name] = dict(df=pd.DataFrame({"demand": demand,
                                          "temperature": temp}, index=index),
                         event_day=ev)
    return out
