# Bike Comparison Method

## Overview

The comparison page answers the question: **which bike is faster (or more efficient) for the same rider, on the same segments, under comparable conditions?**

It uses an XGBoost counterfactual model — train on one bike, predict on the other — so the comparison controls for grade, season, fitness trend, and segment type rather than just averaging raw numbers.

---

## Two modes

### Speed mode (default)
The model learns: *given power, grade, and conditions → what speed did Bike A produce?*  
Then we apply it to Bike B's efforts and ask: **"how fast would Bike A have gone here?"**

- Residual = `actual_B_speed − predicted_A_speed`
- **Positive** → Bike B was faster than Bike A's model expected

### Watt mode
The model learns: *given speed, grade, and conditions → how many watts did Bike A require?*  
Then we apply it to Bike B's efforts and ask: **"how many watts would Bike A have needed here?"**

- Residual = `predicted_A_watts − actual_B_watts`  
  (note: reversed sign vs speed mode, so that positive still means Bike B is better)
- **Positive** → Bike A would need more watts → Bike B is more efficient

---

## Step-by-step walkthrough

### Step 1 — Train model on Bike A
An XGBoost model is trained on Bike A's segment efforts only.  
It learns Bike A's speed-from-power (or watt-from-speed) relationship, controlling for grade, season, and fitness.

### Step 2 — Apply model to Bike B's efforts (A→B)
For every Bike B effort on a matched segment, we use Bike A's model to predict what would have happened if Bike A had done that effort.

**Speed mode question:** "How fast would Bike A have gone here?"  
**Watt mode question:** "How many watts would Bike A have needed to go Bike B's speed?"

Points above the diagonal on the scatter plot mean Bike B came out ahead.

### Step 3 — Summarise the A→B residuals
The mean residual is the average gap across all matched efforts.

**Speed mode:** positive = Bike B was faster on average  
**Watt mode:** positive = Bike A would have needed more watts = Bike B is more efficient

The info bubble phrases this from the model's perspective (Bike A) to stay consistent with the question asked in Step 2.

### Step 4 — Reverse: train on Bike B, apply to Bike A (B→A)
The same process is repeated with the bikes swapped.  
Bike B's model is applied to Bike A's efforts.

**Watt mode question (Step 4):** "How many watts would Bike B have needed to go Bike A's speed?"  
- Positive → Bike B needs more watts → Bike A is more efficient

If both directions agree (one bike consistently comes out ahead), that's strong evidence. If they disagree, there's too much variability or confounding to draw a firm conclusion.

### Step 5 — Aggregate
Both directions are combined into a single estimate, weighted by effort count:

```
combined = (fwd_mean × n_fwd − rev_mean × n_rev) / (n_fwd + n_rev)
```

Bootstrap confidence intervals (30 iterations) are computed by pairing forward and reverse residuals per iteration, giving a symmetric estimate with empirical uncertainty bounds.

**Positive combined** → Bike B is faster / more efficient overall.

---

## Residual sign conventions (quick reference)

| Step | Residual formula | Positive means |
|------|-----------------|----------------|
| Speed A→B | `actual_B_speed − predicted_A_speed` | Bike B faster |
| Speed B→A | `actual_A_speed − predicted_B_speed` | Bike A faster |
| Watt A→B | `predicted_A_watts − actual_B_watts` | Bike B more efficient |
| Watt B→A | `predicted_B_watts − actual_A_watts` | Bike A more efficient |

The aggregate chart negates the B→A direction so that "positive = Bike B better" in all three bars.

---

## Key assumptions

- Efforts on the same named segment are treated as comparable conditions.
- The XGBoost model controls for grade, season (day-of-year Fourier terms), and fitness trend (cumulative effort count), but not weather.
- A time-based 80/20 split is used for model validation in Steps 1–3; the bootstrap in Step 5 retrains on all efforts for maximum coverage.
- Watt mode assumes the recorded average power is accurate and comparable across bikes (same power meter or well-calibrated).
