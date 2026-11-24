# GA-SLNet

**Gap-Adaptive Switching Liquid Network for Ship Trajectory Prediction under AIS Communication Blackouts**

面向 AIS 通信中断的间隙自适应切换液态网络船舶轨迹预测

---

## Motivation

Ship trajectory prediction in real deployment is not a clean sequence-modelling
problem. AIS reports arrive through satellite passes and shore stations, so the
observation stream is riddled with **blackouts** — irregular, sometimes
tens-of-minutes-long gaps where no report arrives at all. Most trajectory
predictors assume a fixed sampling interval and either interpolate the hole away
or mask it out. Both choices silently discard the one fact that matters at
inference time: *how long the ship has been unobserved.*

GA-SLNet treats the blackout length as a first-class input.

## Method

Three ingredients, in the order they matter:

**1. Time-bearing inputs.** Instead of normalising timestamps away, each report
carries displacement `ΔE = v·Δt` and `log(1+Δt)`, plus a clock channel giving
the report's position within the window. The interval stays multiplicative.

**2. Liquid experts with blackout propagation.** `K` liquid cells (a
continuous-time update `h ← decay·h + (1-decay)·f(x,h)` with a learned,
input-conditioned time constant `τ`) run in parallel. After the last observed
report, every expert is advanced across the blackout `g` with its last input
held constant:

```
h ← f + (h − f)·exp(−g / τ(x_last, h, φ(g), a))
```

Stale memory therefore decays provably onto the learned prior as `g` grows,
rather than being carried forward at a fixed rate.

**3. Gap-adaptive switching.** A gate over `[x_last, h, φ(g)]` produces mixture
weights `π`, so which expert dominates is conditioned on the gap length itself.
Short blackout → motion continuation; long blackout → behavioural prior.

Output is a heteroscedastic mixture: per-mode, per-horizon `(μ, σ)` in km, so
calibration is measurable (ECE / CRPS / NLL) rather than assumed.

## Baselines

All baselines are re-implemented in this repo and share **the same
time-conditioned mixture decoder**, so differences come from the encoder alone:

| Model | Family |
|---|---|
| LSTM / GRU | recurrent |
| Seq2Seq | encoder–decoder |
| Transformer | attention |
| CfC | closed-form continuous-time |
| LTC | liquid time-constant |
| Neural ODE | continuous-depth |
| Constant-velocity | non-learned reference |

## Repository layout

```
.
├── config.py              # single source of truth: data, model, training, eval
├── models2.py             # GA-SLNet (+ all baselines, shared decoder)
├── train2.py              # training loop, evaluation, efficiency benchmark
├── metrics2.py            # ADE/FDE/NLL/CRPS/ECE, mixture losses
├── run2.py                # experiment orchestration (quick/main/ablation/seeds/cross)
├── viz2.py                # result figures
├── scripts/
│   └── download_data.py   # fetch the public NOAA AIS dumps
└── requirements.txt
```

Files named `*2.py` are the current pipeline; the unsuffixed `train.py`,
`models.py`, `evaluate.py`, `metrics.py`, `viz.py`, `interpret.py`,
`run_experiments.py` are the first-generation implementation, kept for
provenance.

## Data

Source: **NOAA MarineCadastre AIS**, US coastal waters, 2020.
<https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2020/>

The raw dumps are not redistributed here — one day is several hundred MB.
Fetch them yourself:

```bash
python scripts/download_data.py
```

Study dates: `2020-01-01`, `2020-07-01` (train), `2020-12-01` (test).

Regions are defined in `config.py`:

| Key | Bounding box (lat, lon) | Role |
|---|---|---|
| `Gulf` | 24.0–31.5 N, 98.0–80.5 W | train |
| `Atlantic` | 24.0–45.0 N, 82.0–64.0 W | train |
| `Pacific` | 32.0–49.0 N, 126.0–116.0 W | cross-region test |

Trip extraction filters to `SOG ∈ [0.6, 40] kn`, requires ≥ 40 reports, thins
sub-20 s duplicates, and splits a new trip on any silence > 60 min.

## Quick start

```bash
pip install -r requirements.txt

python scripts/download_data.py
python scripts/preprocess.py          # -> data/processed/{train2,eval2}.npz

python run2.py --stage quick          # GA-SLNet + 3 baselines, gaps {0,15,30}
python run2.py --stage main           # all models
python run2.py --stage ablation       # component ablations
python run2.py --stage seeds          # 3 seeds, 3 models
python run2.py --stage cross          # Gulf -> Pacific transfer
python run2.py --stage eval           # full gap sweep -> output/results2.json
```

Individual model:

```bash
python train2.py --model gaslnet2 --tag mine
python train2.py --model gaslnet2 --no-switch     # ablation: single expert
python train2.py --model gaslnet2 --no-prop       # ablation: no propagation
```

Stages are idempotent: an existing `output/<tag>.pt` is skipped.

## Evaluation protocol

Blackouts are **injected at evaluation time**, not baked into training data:
after the observation window, `g ∈ {0, 5, 15, 30}` minutes of reports are
deleted before the model sees the sequence. The model is then asked to predict
horizons `[5, 10, 15, 20, 30]` minutes past the last observed point.

Metrics: `ADE`, `FDE`, `MedADE`, `P90ADE` (metres), `NLL`, `CRPS`, `ECE`,
plus parameter count and single-sample latency.

Model selection uses the **mean ADE over the validation gap set**
`{0, 15, 30}`, so the chosen checkpoint is the one that is robust across gap
lengths, not merely the best at `g = 0`.

Gap augmentation (`GAP_AUG_P = 0.25`, up to 30 min) is applied during training
as a curriculum, and ablation `gaslnet2_nodt` isolates its contribution from the
time-bearing input design.

## License

MIT — see [LICENSE](LICENSE).

NOAA AIS data is public domain, subject to NOAA's terms of use.
