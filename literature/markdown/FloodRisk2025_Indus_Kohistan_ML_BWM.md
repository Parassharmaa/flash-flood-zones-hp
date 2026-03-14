# Flood Risk Modelling by the Synergistic Approach of Machine Learning and Best-Worst Method in Indus Kohistan, Western Himalaya

**Authors:** (Authors from Geomatics, Natural Hazards and Risk)
**Year:** 2025
**Journal:** Geomatics, Natural Hazards and Risk, Vol. 16(1)
**DOI/URL:** https://www.tandfonline.com/doi/full/10.1080/19475705.2025.2469766

## Study Area
Indus Kohistan, Pakistan — a mountainous region between Hindu Kush and Himalayan ranges, separated by the Indus River. Coordinates: 35°13′ to 35°8′N, 73°17′ to 73°20′E. Geologically and climatologically similar to parts of Himachal Pradesh.

## Methods
**Hazard assessment:**
- Random Forest (RF)
- XGBoost (Extreme Gradient Boosting)

**Vulnerability assessment:**
- Best-Worst Method (BWM) — multi-criteria decision analysis

**Integration:** Flood hazard map (from ML) × Vulnerability map (from BWM) = Flood Risk Map

Supporting data: 415 flood points, 13 flood hazard factors, 6 flood vulnerability indicators.

## Conditioning Factors

**Hazard factors (13):**
- Distance to streams (most important: relative importance 0.43)
- Elevation (second: 0.19)
- Slope, TWI, SPI, TRI
- Drainage density
- LULC, NDVI
- Rainfall
- Lithology, soil type, curvature

**Vulnerability indicators (6):**
- Population density (40.2% weight)
- Household density (24.7% weight)
- Distance to healthcare
- Agricultural land
- Infrastructure density
- Economic value

## Performance
| Model | AUC |
|-------|-----|
| RF    | 0.99 |
| XGBoost | 0.98 |

RF selected for final hazard map. Note: very high AUC warrants scrutiny of spatial cross-validation.

## Key Findings
1. **Distance to streams** is by far the dominant hazard factor (importance 0.43)
2. **Elevation** is second most important (0.19)
3. RF slightly outperforms XGBoost despite both achieving near-perfect AUC
4. Population density (40.2%) and household density (24.7%) dominate vulnerability
5. Integrated risk map reveals highest risk in valley floors near streams with high population density
6. BWM provides systematic expert-based vulnerability weighting

## Limitations
- Very high AUC (0.98–0.99) — spatial cross-validation method not fully described
- Western Himalayan Pakistan setting; some transferability to HP but different monsoon dynamics
- BWM still involves expert judgment for vulnerability weights
- 415 flood points may not cover full spatial diversity of the basin

## Relevance to HP Flash Flood Study
This is the best methodological template for an integrated hazard + vulnerability → risk framework for HP:
- The ML-hazard + MCDA-vulnerability = risk framework is exactly what HP study needs to produce actionable risk maps
- Distance to streams dominant importance (0.43) confirms it must be a primary factor in HP model
- Elevation importance (0.19) confirms HP's altitudinal gradient is key
- BWM for vulnerability is more systematic than simple AHP — better suited for HP's complex socioeconomic landscape
- Integration with population/infrastructure exposure is needed for disaster risk reduction policy in HP
- Published in target journal (Geomatics, Natural Hazards and Risk) — exactly the publication target for HP work
