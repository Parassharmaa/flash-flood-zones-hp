# Flash Flood Susceptibility Modelling Using Soft Computing-Based Approaches: From Bibliometric to Meta-Data Analysis and Future Research Directions

**Authors:** (Multiple authors; full list in MDPI Water journal)
**Year:** 2024
**Journal:** Water (MDPI), Vol. 16(1): 173
**DOI/URL:** https://www.mdpi.com/2073-4441/16/1/173

## Study Area
Global review covering 305 documents from Web of Science core collection (15-year period, ~2007–2022).

## Methods Reviewed
Bibliometric analysis (CiteSpace visualization of intellectual networks) combined with meta-data analysis across five research subfields:
1. Assessment scale
2. Assessment unit
3. Assessment index (conditioning factors)
4. Assessment model
5. Model assessment method (validation)

## Conditioning Factors (Most Commonly Used)
Across global literature, most frequently cited:
1. **Slope** (topographic)
2. **Elevation**
3. **Distance from river**
4. TWI
5. SPI
6. Drainage density
7. Rainfall
8. LULC / land cover
9. NDVI
10. Lithology / geology
11. Curvature (plan, profile)
12. Soil type

## Performance Summary Across Literature
- Hybrid models were the **most frequently used** prediction model type
- AUC range: typically 0.80–0.99 depending on model and study area
- Machine learning methods (RF, XGBoost, SVM) consistently outperform statistical bivariate methods

## Key Findings from Meta-Analysis
**Publication trends:**
- Articles from 2020 onward: 54.4% of all publications
- 2016–2019: 34.1%
- Pre-2016: 11.5%
- Flash flood susceptibility assessment is a rapidly accelerating research field

**Dominant methodology:**
- Hybrid ensemble models = most frequently used
- GIS, machine learning, statistical models, AHP are central research focuses

**Conditioning factor consensus:**
- Slope, elevation, distance from river = most commonly used trio
- LULC is used frequently but ranked as **least important** factor in some studies
- DEM is the most important single factor when evaluated with SHAP in some studies

**Future directions identified:**
1. Resolution of input data (higher-resolution DEMs needed)
2. Size and representativeness of training samples
3. Spatial cross-validation (geography-aware train/test splits)
4. Dynamic factors (seasonal soil moisture, antecedent rainfall)
5. Uncertainty quantification

## Limitations of the Field (Identified by Authors)
- Dependency on data-driven algorithms that fail to capture physical flood processes
- Spatial autocorrelation in train/test splits inflates reported AUC
- Training sample size and selection methodology rarely standardized
- Most studies in East Asia, Middle East, Europe — Indian Himalaya underrepresented
- Dynamic/temporal conditioning factors largely absent

## Relevance to HP Flash Flood Study
This is the most authoritative global review of the field:
- Indian Himalaya is explicitly underrepresented in flash flood susceptibility literature — HP study fills a genuine gap
- Hybrid ensemble models (RF + XGBoost or MARS + RF) are the expected standard
- Slope + elevation + distance from river must be the core factor set for any HP study
- Spatial cross-validation is a key methodological gap in existing literature — HP work should use geographic k-fold CV
- SHAP analysis should be included for interpretability
- Higher-resolution DEM (e.g., ALOS PALSAR 12.5m vs. SRTM 30m) is worth testing as a novel contribution
- Dynamic conditioning factors (soil moisture, antecedent precipitation index) are a frontier — could be a novel contribution for HP
