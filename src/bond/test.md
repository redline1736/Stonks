First, let’s address the elephant in the room: **Your model’s accuracy dropped to 46.8%, and it heavily underperformed the naive "always guess up" baseline (54.1%).** 

Do not panic. You just experienced a **Regime Shift**, which is the hardest thing in quantitative finance. I will explain exactly *why* the model failed here, but first, let's do a deep dive into the two new Yield Curve features you asked about, because XGBoost actually loved them (the 10Y-5Y spread was the **#2 most important feature** in the last fold!).

Here is why those specific yield curve features are so powerful for predicting TNX.

---

### 📉 Deep Dive: Why Yield Curve Features Work

The Yield Curve is essentially the bond market's "crystal ball." It prices in future economic growth, inflation, and Federal Reserve policy.

#### 1. `Yield_Curve_10Y_3M` (TNX - IRX)
*   **What it is:** The difference between the 10-year yield and the 13-week T-bill yield. The 13-week T-bill (IRX) is essentially a direct proxy for the **Federal Funds Rate** (what the Fed controls today).
*   **Why it works:** This is the most important recession indicator in the world. 
    *   **Normal Market:** 10Y is higher than 3M. Banks borrow short and lend long. The economy is healthy.
    *   **Inverted Market (Negative Spread):** 3M is higher than 10Y. This happens when the Fed hikes rates aggressively. The bond market is screaming, *"The Fed is going to cause a recession, and they will have to cut rates drastically in the future."*
*   **How XGBoost uses it:** If the model sees a deeply negative spread (e.g., -1.5%), and then it sees the spread start to "un-invert" (move toward 0), it knows the Fed is about to start cutting rates. When the Fed cuts, the 10Y yield (TNX) plummets. The model learns to associate these specific curve movements with massive TNX drops.

#### 2. `Yield_Curve_10Y_5Y` (TNX - FVX)
*   **What it is:** The spread between the 10-year and the 5-year Treasury.
*   **Why it works:** The 5-year sits in the "belly" of the curve and is hyper-sensitive to Fed policy expectations. 
    *   This spread tells you about **Term Premium** (the extra yield investors demand to hold a bond for 10 years instead of 5).
    *   If the 10Y-5Y spread is *steepening* (getting larger), it means inflation expectations are rising, or the government is issuing too much long-term debt. Both push TNX up.
    *   If it is *flattening* (getting smaller), it means the market is worried about long-term growth, and they are fleeing into the safety of the 10Y, pushing TNX down.
*   **How XGBoost uses it:** Tree models are excellent at finding **thresholds**. XGBoost likely found a rule like: *"If the 10Y-5Y spread drops below -0.20 (deep inversion) AND the VIX is rising, it's a massive flight to safety; predict TNX goes down."*

---

### 🚨 Why Did the Model Fail? (The 46.8% Accuracy)

There are three reasons the model crashed below the baseline. This is a classic masterclass in ML for finance:

**1. The 60% Probability Threshold was too aggressive.**
Look at your last fold classification report:
*   Class 0 (Down) Recall: 0.79 (It predicted Down almost every time).
*   Class 1 (Up) Recall: 0.24 (It almost never predicted Up).
Because you forced the model to only predict when it was `> 60%` sure, it became incredibly conservative. It only saw "Up" when the signals were overwhelming. But the market was trending Up (Naive baseline was 56%). So by abstaining from guessing Up, it missed all the upside.

**2. Tree models cannot extrapolate (The Regime Shift Problem).**
*   In 2018–2021 (Folds 1-3), the yield curve was **positive** (e.g., +2.0%).
*   In 2022–2024 (Folds 4-6), the yield curve was deeply **inverted** (e.g., -1.5%).
*   XGBoost is a tree. It draws boxes. It sees `Yield_Curve_10Y_3M = +2.0` in training. In testing, it sees `-1.5`. It has *never seen this number before* and doesn't know what to do. It fails completely.

**3. You are predicting a 5-day move using a macro feature.**
The yield curve is a slow-moving macro indicator. It predicts what will happen over the next **6 to 12 months**. Trying to use the yield curve to predict a **5-day** move is like using a calendar to figure out what time it is. The signal gets lost in the daily noise.

---

### 🛠️ How to Fix the Model

To make this model actually work and beat the baseline, you need to adjust the strategy. 

**Fix 1: Lower the Probability Threshold**
Change `predictions = (probs > 0.60).astype(int)` to `predictions = (probs > 0.52).astype(int)`. 
*   If the model thinks there is a 52% chance TNX goes up, and the market is in an uptrend, you want to take that trade. 60% is too high for bonds.

**Fix 2: Add a "Regime" Feature**
We need to tell the tree model that an inverted curve is a different world. Add this feature:
```python
# Feature: Is the yield curve inverted? (1 = Yes, 0 = No)
features['Is_Inverted'] = (features['Yield_Curve_10Y_3M'] < 0).astype(int)
```
*Now, the model can split the data: "When Inverted == 1, do X. When Inverted == 0, do Y."* This solves the extrapolation problem.

**Fix 3: Change the Target to a Regression**
Instead of predicting "Up or Down" (which is noisy), predict the **actual percentage change**. Use `XGBRegressor` instead of `XGBClassifier`. Train it to predict: *"What will the 5-day return of TNX be?"* If the model outputs `+0.05`, you buy bonds (yield goes up). If it outputs `-0.05`, you sell bonds. 

**Fix 4: Use a Shorter Target Horizon**
If you are using daily data (VIX, MOVE, SPY), you should predict a **1-day or 2-day** forward move. The yield curve and VIX react too fast for a 5-day horizon. Change `shift(-5)` to `shift(-1)`.

### 🧠 The Big Lesson Here
You didn't do anything wrong. You just learned that **financial markets are non-stationary**. The relationship between the yield curve and TNX in 2020 is completely different from 2023. 

Try lowering the threshold to `0.52`, adding the `Is_Inverted` feature, and see if the accuracy jumps back above the baseline!