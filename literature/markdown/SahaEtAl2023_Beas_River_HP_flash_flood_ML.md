# Spatial Flash Flood Modeling in the Beas River Basin of Himachal Pradesh, India, Using GIS-Based Machine Learning Algorithms

**Authors:** Saha et al. (full author list in book chapter)
**Year:** 2023/2024
**Journal:** Book chapter in Springer volume (Natural Hazards in the Himalayan Region; ISBN 978-981-99-7707-9)
**DOI/URL:** https://link.springer.com/chapter/10.1007/978-981-99-7707-9_8

## Study Area
Beas River Basin, Himachal Pradesh, India — one of the five major river systems in HP. Includes the Kullu Valley, Manali area, and downstream Bhuntar region. Upper Beas is sourced from Rohtang Pass (~4,000 m) and flows through steep Himalayan terrain.

## Methods
- Multivariate Adaptive Regression Splines (MARS)
- Random Forest (RF)
- Ensemble model: MARS-RF (combined)
- GIS integration of conditioning factors
- 70/30 train-test split

## Conditioning Factors
(From search results; full list in chapter):
- Elevation
- Slope
- Aspect
- Rainfall
- Drainage density
- Distance to river
- TWI (Topographic Wetness Index)
- Curvature
- LULC
- NDVI
- Geology/lithology

## Performance
| Model | AUC |
|-------|-----|
| MARS  | 0.828 |
| RF    | 0.856 |
| MARS-RF ensemble | 0.880 |

- 11.81% of total area classified as extremely susceptible to flash floods
- Ensemble (MARS-RF) outperforms both standalone models

## Key Findings
1. MARS-RF ensemble provides best prediction performance (AUC = 0.88)
2. 11.81% of Beas basin extremely susceptible — concentrated in upper valley
3. River reaches with highest flash flood susceptibility: Bahang–Manali (Beas), Kullu–Bhuntar (Beas), Manikaran–Kheer-Ganga (Parvati)
4. Northern and eastern basin sections most susceptible due to steep terrain and rainfall
5. Ensemble approaches consistently outperform standalone models

## Limitations
- Book chapter (not journal paper) — lower visibility and peer review rigor
- AUC of 0.88 is good but the specific spatial cross-validation method unclear
- Limited to Beas basin; does not cover Satluj, Chenab, or Ravi basins in HP
- Static conditioning factors — no dynamic seasonal variables
- Relatively modest AUC of 0.828–0.880 vs. some studies claiming 0.99+

## Relevance to HP Flash Flood Study
This is the single most directly relevant ML-based study for HP:
- Beas basin is the most important flash flood zone in HP and likely priority study area
- Kullu–Bhuntar and Manikaran–Parvati are confirmed high-susceptibility zones (consistent with disaster records)
- MARS-RF ensemble is a relatively underused combination — could be updated with XGBoost or deep learning
- AUC of 0.88 for ensemble represents the current baseline for HP — study must improve on this
- The 11.81% extremely susceptible area statistic is a key benchmark
- Book chapter format means there is an opportunity for a more rigorous peer-reviewed journal paper on HP
