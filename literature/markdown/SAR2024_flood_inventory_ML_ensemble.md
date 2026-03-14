# SAR-Driven Flood Inventory and Multi-Factor Ensemble Susceptibility Modelling Using Machine Learning Frameworks

**Authors:** (Authors from Natural Hazards and Risk, 2024)
**Year:** 2024
**Journal:** Geomatics, Natural Hazards and Risk
**DOI/URL:** https://www.tandfonline.com/doi/full/10.1080/19475705.2024.2409202

## Study Area
Southwest coast of India (related paper). SAR inventory-based approach tested in a coastal/deltaic setting with monsoon-driven flooding.

## Methods
1. **Multi-year flood inventory** generated from Sentinel-1 SAR imagery (2018–2023) using Google Earth Engine
2. Water pixels extracted using backscatter thresholding
3. Repeated occurrences (≥3 times) classified as flood-prone (reduces incidental errors)
4. ML models applied: RF, XGBoost, CNN
5. SHAP analysis for feature interpretability

## Conditioning Factors
- Topographic: DEM, slope, TWI, SPI, TRI
- Hydrological: drainage density, distance to river
- Environmental: LULC, NDVI, soil type
- Precipitation: derived from multi-temporal data
- SAR-derived flood frequency index as additional factor

## Performance
RF and XGBoost achieve high AUC values (>0.90 in comparable applications). Specific values not fully accessible.

## Key Findings
1. **Multi-temporal SAR** (2018–2023, 6 years) provides more accurate flood inventory than single-event optical mapping
2. Repeated ponding criterion (≥3 occurrences) reduces false positives from temporary water bodies
3. SAR-derived flood frequency can itself serve as a conditioning factor
4. GEE cloud platform enables processing of 6-year SAR archive without local computation
5. ML models trained on SAR-based inventory outperform those trained on single-event optical data
6. SHAP values confirm distance to river and elevation as top predictors

## Limitations
- SAR is less sensitive to shallow, vegetated flooding (urban floods, flooded croplands)
- 6-year SAR archive may not capture low-frequency extreme events (50-year, 100-year return periods)
- Cloud and wind effects on SAR backscatter can create false flood detections
- Multi-temporal approach requires consistent SAR coverage — gaps exist in early Sentinel-1 archive

## Relevance to HP Flash Flood Study
Critical methodological contribution for HP:
- HP receives heavy monsoon rainfall → cloud cover makes optical imagery unreliable during flood events → **SAR is essential**
- Multi-temporal Sentinel-1 analysis (2018–2023 = 6 years of HP monsoon events) can build flood inventory
- The ≥3 occurrences criterion reduces noise from temporary water bodies common in HP rice paddies and glacier melt pools
- GEE makes this computationally feasible without large local infrastructure
- SAR-based inventory for 2018–2023 HP events includes the catastrophic 2023 season — high-quality recent data
- This approach is novel for HP specifically — no published SAR-based flood inventory for HP exists
- Combining SAR inventory with traditional topographic/hydrological conditioning factors is a methodological advancement
