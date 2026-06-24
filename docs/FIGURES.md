# Figures

All figures are 300 dpi, width >= 2244 px (Elsevier full-page spec).

| File | Size (px) | Caption |
|------|-----------|--------|
| `Figure_1.png` | 2244x1102 | Fig. 1. Overview of the proposed causal regime-shift augmentation. Causal features (local-shift and optional time-since-changepoint) augment the LSTM input; evaluation reports MAE, its critical/normal decomposition, and rho. |
| `Figure_2.png` | 2244x1380 | Fig. 2. Distribution of detected changepoints across the 168,987 clean series. Series with three or more changepoints (right of the dashed line) form the experimental population. |
| `Figure_3.png` | 2244x1234 | Fig. 3. A representative series with three PELT-detected changepoints (dashed lines) and their critical neighbourhoods (shaded, w = 1 quarter), the regions over which MAE_cp is computed. |
| `Figure_4.png` | 2244x1424 | Fig. 4. Effect of the weighting factor lambda. The ratio rho falls (right axis), yet MAE_cp rises while MAE_normal rises faster, revealing that rho improves only through denominator inflation, not better changepoint accuracy. |
| `Figure_5.png` | 2244x1380 | Fig. 5. Changepoint-region error (MAE_cp) by method on 5,000 series. The proposed model (LSTM + local-shift + tau) attains the lowest error precisely at changepoints. |
| `Figure_6.png` | 2244x1424 | Fig. 6. Distribution of absolute error at changepoints for ARIMA, the LSTM baseline, and the proposed method (outliers suppressed). The proposed method shifts the entire distribution toward lower error. |
| `Figure_7.png` | 2244x1436 | Fig. 7. Changepoint error (MAE_cp) of the baseline versus the proposed augmentation across forecast horizons. The benefit emerges at longer horizons, where changepoints fall within the forecast window. |
| `Figure_8.png` | 2244x1371 | Fig. 8. Changepoint error (MAE_cp) of the three mechanisms, in-domain and under cross-domain transfer (FFSU to MCOU). The ranking and the benefit are preserved on the unseen domain. |
| `Figure_9.png` | 2244x1442 | Fig. 9. Changepoint error (MAE_cp) versus model size. The proposed lightweight mechanisms (~5K parameters) approach the accuracy of attention models (~68K parameters), recovering about two-thirds of the baseline-to-iTransformer improvement at one-fourteenth of the parameter count. |
| `Figure_10.png` | 2244x1424 | Fig. 10. The causal local-shift signal reduces changepoint error for both an LSTM and an iTransformer backbone, indicating that the benefit transfers across the two architectures tested. |
