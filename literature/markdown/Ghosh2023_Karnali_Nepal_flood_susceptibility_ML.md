# Flood Susceptibility Modeling of the Karnali River Basin of Nepal Using Different Machine Learning Approaches

**Authors:** (Authors from Geomatics, Natural Hazards and Risk, 2023)
**Year:** 2023
**Journal:** Geomatics, Natural Hazards and Risk, Vol. 14(1)
**DOI/URL:** https://www.tandfonline.com/doi/full/10.1080/19475705.2023.2217321

## Study Area
Karnali River Basin, Nepal — a major Himalayan river basin in western Nepal. Characterized by high-altitude glacier-fed upper reaches transitioning to lower-elevation fertile plains. Flows from Tibetan Plateau through steep Himalayan terrain.

## Methods
Three machine learning techniques compared:
1. Support Vector Machine (SVM)
2. Random Forest (RF)
3. Artificial Neural Network (ANN)

Multicollinearity test used to filter conditioning factors. Cohen Kappa Score used alongside ROC for evaluation.

## Conditioning Factors (10 — selected after multicollinearity test)
1. Aspect
2. Curvature
3. Distance to river (DTR)
4. NDVI
5. Elevation
6. Slope
7. Rainfall
8. Soil type
9. Stream Power Index (SPI)
10. Topographic Wetness Index (TWI)

## Performance
| Model | Success Rate AUC | Prediction Rate AUC |
|-------|------------------|---------------------|
| SVM   | 0.928 | 0.987 |
| RF    | (lower) | (lower) |
| ANN   | (lower) | (lower) |

**SVM outperformed RF and ANN** — unusual finding vs. most studies where RF/XGBoost dominate.
NDVI has greatest influence on flood susceptibility, followed by elevation, DTR, curvature, TWI.

## Key Findings
1. SVM achieves highest prediction AUC (98.7%) — outperforms RF and ANN in Nepal Himalayan context
2. NDVI is the most important factor — unexpected, usually dominated by topographic/proximity factors
3. Very high flood susceptibility areas: only 0.8–2.5% of basin, mostly in low-elevation southern plains
4. Multicollinearity testing essential to avoid redundant factors
5. Cohen Kappa Score provides more robust accuracy than accuracy alone (handles class imbalance)

## Limitations
- Single river basin in Nepal; transferability to HP uncertain
- Prediction AUC of 0.987 for SVM is very high — may reflect insufficient spatial cross-validation
- Only 10 conditioning factors — relatively parsimonious
- No dynamic factors (e.g., soil moisture, antecedent rainfall)
- NDVI dominance may be an artifact of training data distribution

## Relevance to HP Flash Flood Study
- Himalayan river basin study with comparable terrain complexity
- SVM outperforming RF/XGBoost challenges the dominance of ensemble tree methods — worth testing all three in HP
- The 10-factor framework after multicollinearity filtering is exactly the approach needed for HP
- NDVI dominance in Nepal may not apply in HP (different vegetation cover patterns) — factor importance testing essential
- Cohen Kappa Score should be reported alongside AUC in HP study for more honest evaluation
- Very small high-susceptibility area (0.8–2.5%) in mountain context is realistic — some HP studies claiming 25–35% high susceptibility seem inflated
