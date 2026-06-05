# NIFTY Options Implied Volatility Surface Reconstruction

## Problem Statement
Predict missing Implied Volatility (IV) values across the NIFTY50 options chain to reconstruct a complete, arbitrage-free volatility surface.

## Methodology & Architecture
Instead of relying on pure machine learning, which is blind to financial physics, this solution implements a **Quantitative Hybrid Pipeline**:

1. **Mathematical Base (Cubic Splines):** - Fitted a natural cubic spline to the option chain for every single 5-minute timestamp. 
   - Utilized `bc_type='natural'` to ensure linear extrapolation at the boundaries, preventing extreme arbitrage hallucinations in deep OTM/ITM options.
2. **Residual Calculation:** - Calculated the error (`IV_Residual`) between the theoretical spline curve and the actual market IV.
3. **Machine Learning Micro-Correction Engine:** - Engineered cross-sectional spatial lags (Spatial_IV_Below, Spatial_IV_Above) and time-based memory lags (Previous_IV).
   - Trained an XGBoost Regressor to predict the residual errors and find localized micro-inefficiencies in the surface.
4. **Final Reconstruction:** - Recombined the mathematical baseline with the ML-predicted residuals.

## Validation Strategy
- Implemented a strict **Chronological Time-Series Validation** split to explicitly prevent look-ahead bias and simulate real-world market data arrival. 

## Key Results
- **Pure Spatial XGBoost Model:** 0.000288 (Public Leaderboard)
- **Hybrid Spline + ML Model:** 0.000179 (Public Leaderboard) - *Final Champion Model*
