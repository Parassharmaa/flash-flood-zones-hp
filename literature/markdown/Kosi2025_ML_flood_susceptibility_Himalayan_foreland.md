# Machine Learning Model Optimization for Flood Susceptibility Zonation over the Kosi Megafan, Himalayan Foreland Basin, India

**Authors:** (Authors from Scientific Reports, 2025)
**Year:** 2025
**Journal:** Scientific Reports (Nature Publishing Group)
**DOI/URL:** https://www.nature.com/articles/s41598-025-07403-w

## Study Area
Kosi Megafan, Himalayan foreland basin, India. One of the most flood-prone alluvial fans in the world, formed by the Kosi River (north Bihar). High sediment load and dynamic channel migration contribute to annual devastating floods.

## Methods
Five advanced machine learning algorithms evaluated and optimized:
1. **Random Subspace** (RS)
2. **J48** decision tree
3. **Maximum Entropy (MaxEnt)**
4. **Artificial Neural Network (ANN-MLP)**
5. **Biogeography-Based Optimization (BBO)**

Data: 19 conditioning factors from ALOS PALSAR DEM, Sentinel-2A, Landsat 5 TM, ENVISAT-1 ASAR, ancillary sources.

## Conditioning Factors (19 variables — most comprehensive in literature)
1. NDVI
2. Altitude
3. Distance to road
4. Rainfall
5. Distance to river
6. Slope
7. TWI
8. SPI
9. TRI
10. Curvature (plan + profile)
11. Aspect
12. Geomorphology
13. Lithology
14. LULC
15. Soil type
16. Drainage density
17. Flood frequency index (from historical inundation)
18. Population density
19. Sediment transport index (STI)

## Performance
| Model | AUC (training) | AUC (validation) |
|-------|----------------|------------------|
| ANN-MLP | 0.995 | 0.992 |
| MaxEnt | High | High |
| Others | Lower | Lower |

ANN-MLP and MaxEnt provide best tools for identifying high-risk areas.

## Key Findings
1. **NDVI** is the most important variable — followed by altitude, distance to road, rainfall, distance to river
2. ANN-MLP achieves nearly perfect AUC (0.992 validation) — likely reflects data characteristics of alluvial plain
3. MaxEnt also performs excellently — unusual finding (typically used for species distribution modeling)
4. 19-factor dataset is among the most comprehensive in the literature
5. ALOS PALSAR DEM (12.5m) provides better resolution than SRTM for flood susceptibility derivation
6. High sediment load and channel migration create non-stationary flood patterns

## Limitations
- Kosi Megafan is an alluvial plain — very different from HP's steep Himalayan terrain
- Very high AUC (0.99) likely due to spatial autocorrelation in flat alluvial settings
- 19 factors increase model complexity and overfitting risk
- MaxEnt's use for flood susceptibility (borrowed from species distribution modeling) is methodologically questionable
- NDVI dominance may reflect vegetation patterns in alluvial plain, not terrain control

## Relevance to HP Flash Flood Study
- ALOS PALSAR at 12.5m resolution is available and superior to SRTM 30m for HP — should be primary DEM
- The 19-factor comprehensive set is useful as a reference checklist; HP study should test a comparable set
- MaxEnt borrowed from ecological modeling is an interesting methodological cross-domain approach — worth testing for HP flash flood given MaxEnt's success in species distribution (used in companion thesis topic)
- NDVI dominance in flat alluvial setting vs. slope/elevation dominance in mountains reflects context-dependency of factor importance
- STI (Sediment Transport Index) is rarely included — worth adding for HP where sediment transport is high
- Very high AUC in alluvial plain setting should NOT be expected in HP's rugged terrain where flood patterns are far more complex
