# Research Gaps and Novel Directions: Flash Flood Susceptibility Mapping in Himachal Pradesh

**Compiled:** March 2026
**Based on:** Systematic review of 22+ papers, bibliometric meta-analysis, and regional context analysis

---

## 1. What Is Missing in the Literature

### Gap 1: No Comprehensive Peer-Reviewed ML Study for Himachal Pradesh
**What exists:**
- One book chapter (Springer, 2023) on Beas basin using MARS + RF (AUC = 0.88)
- One preprint CNN U-Net study (Research Square, 2025; not peer-reviewed)
- GIS morphometric study (2017, ScienceDirect)
- Descriptive/news-media analyses of 2023 disaster

**What is missing:**
- A comprehensive peer-reviewed journal paper covering ALL major river basins of HP (Satluj, Beas, Chenab, Ravi, Yamuna) with ML methods
- Any study covering Lahaul-Spiti, Kinnaur, upper Chamba (the most climatologically distinct and GLOF-prone zones)
- Any HP study with proper spatial cross-validation
- Any HP study that explicitly separates monsoon-triggered from GLOF-triggered flood susceptibility

**Novelty opportunity:** A state-wide or multi-basin HP flash flood susceptibility map using current ML methods (RF + XGBoost + stacking ensemble) with spatial cross-validation and SHAP interpretability would be publishable in NHESS, Remote Sensing, or Geomatics Natural Hazards and Risk.

---

### Gap 2: Absence of GLOF-Integrated Flash Flood Susceptibility Maps
**What exists:**
- GLOF susceptibility maps for specific basins (Chenab, Parbati, Spiti) — but these are standalone studies
- Flash flood susceptibility maps — but these ignore GLOF contribution
- No study integrates GLOF risk + monsoon-triggered flash flood susceptibility

**What is missing:**
- A unified flash flood susceptibility framework that accounts for BOTH monsoon-triggered AND GLOF-triggered floods
- For HP: the GLOF contribution cannot be ignored (75% glacial lake area increase 1990–2022 in Chenab basin alone)

**Novelty opportunity:** Including glacial lake proximity, lake area, and glacier retreat rate as conditioning factors in a flash flood susceptibility model is genuinely novel. No published study does this for HP.

---

### Gap 3: Multi-Trigger / Mechanism-Stratified Susceptibility Mapping
**What exists:**
- Single-mechanism models (usually monsoon rainfall as trigger)
- Multi-hazard studies (separate maps for landslide, flood, avalanche — not integrated triggers)

**What is missing:**
- A model that accounts for different flood-triggering mechanisms producing different spatial patterns
- Cloudburst-specific vs. monsoon-general susceptibility distinction
- The compounding effect of antecedent soil moisture + snowmelt + rainfall (identified as key in July 2023 HP analysis)

**Novelty opportunity:** Including antecedent soil moisture (e.g., 30-day prior precipitation index) as a dynamic conditioning factor, and testing whether a mechanism-stratified model outperforms a unified model.

---

### Gap 4: Lack of SAR-Based Flood Inventories for HP
**What exists:**
- HiFlo-DAT historical database (Kullu only, 128 events, documentary records)
- Isolated SAR flood mapping studies for 2023 event
- No multi-year systematic SAR flood inventory for HP

**What is missing:**
- A multi-temporal Sentinel-1 SAR flood inventory for HP covering 2016–2024 (full Sentinel-1 archive)
- Standardized flood point database combining SAR-detected events + historical records
- Training dataset quality is the primary bottleneck for ML performance in HP

**Novelty opportunity:** Constructing the first SAR-based, multi-year flood inventory for HP (or Beas basin as study area) using Google Earth Engine, then using it for ML model training. This inventory creation alone would be publishable as a data paper.

---

### Gap 5: Inadequate Validation — No Spatial Cross-Validation
**What exists:**
- Most studies use random 70/30 or 80/20 train-test splits
- AUC values of 0.85–0.99 widely reported but likely inflated

**What is missing:**
- Geographic k-fold cross-validation (spatial block CV, leave-one-watershed-out)
- Temporal validation (train on pre-2020 floods, test on 2023 floods)
- Uncertainty quantification (confidence intervals on susceptibility predictions)

**Novelty opportunity:** Using spatial block cross-validation for HP study would directly address the most criticized methodological gap in the field. A paper explicitly addressing spatial autocorrelation bias in AUC reporting would be methodologically impactful.

---

### Gap 6: Sparse Data / Ungauged Basin Problem Not Addressed for HP
**What exists:**
- Transfer learning studies for flood forecasting in ungauged basins (urban flood, streamflow)
- HP gauge network documented as sparse in multiple papers

**What is missing:**
- A systematic comparison of rainfall data sources (CHIRPS vs. GPM-IMERG vs. IMD gridded) for HP flash flood susceptibility modeling
- Assessment of model performance degradation as a function of data quality/availability
- Transfer learning approach: train on well-gauged basins (e.g., Beas) and transfer to ungauged basins (Spiti, upper Chenab)

**Novelty opportunity:** Explicit comparison of three satellite/reanalysis rainfall products for HP and their impact on ML model accuracy, with spatial validation. The Dixit (2026) NHESS paper showed GPM-IMERG performs poorly in nearby Uttarakhand — testing this for HP is novel.

---

### Gap 7: No Temporal or Future Projection Analysis for HP
**What exists:**
- Climate-change-based future flood susceptibility studies for South Asia using CMIP6/SSP scenarios
- HP climate trends documented (increasing extreme rainfall events)

**What is missing:**
- Future flash flood susceptibility maps for HP under SSP2-4.5 and SSP5-8.5 scenarios
- Combination of future LULC change (urbanization + deforestation trends) + climate change for HP
- Quantification of which HP sub-basins face greatest increase in flash flood risk by 2050/2070

**Novelty opportunity:** A future-projection component to the HP susceptibility map using CMIP6 projections. This would make the study policy-relevant and suitable for higher-impact journals.

---

### Gap 8: Infrastructure and Tourism Vulnerability Not Mapped
**What exists:**
- HiFlo-DAT shows roads and bridges as primary impact receptors in Kullu
- General flood vulnerability studies for India

**What is missing:**
- Flash flood risk maps specifically for critical HP infrastructure: Manali-Leh highway, Chandigarh-Manali highway, hydroelectric projects, tourist facilities
- Tourism exposure layer: HP receives millions of tourists in summer peak — flash flood risk during tourism season
- Integration of population displacement potential

**Novelty opportunity:** Adding an infrastructure/tourism exposure layer to flash flood susceptibility → risk map. HP is India's top tourist destination and this framing would attract policy attention and journal interest.

---

## 2. Top 5 Most Promising Novel Directions for HP Study

### Direction 1: Integrated Multi-Trigger Flash Flood Susceptibility Mapping with SAR Inventory [HIGH NOVELTY]
**Concept:** Construct a multi-year SAR-based flood inventory (Sentinel-1, 2018–2024) for the Beas and/or Satluj basin using GEE. Use this inventory to train ensemble ML models (RF + XGBoost + stacking). Include both standard conditioning factors AND GLOF-specific factors (glacial lake proximity, glacier retreat area). Apply spatial block cross-validation. Produce the first comprehensive, multi-trigger flash flood susceptibility map for HP.

**Expected contribution:**
1. First SAR-based flood inventory for HP (data contribution)
2. First comprehensive peer-reviewed ML susceptibility map for HP beyond Beas basin
3. GLOF-integrated susceptibility is genuinely novel globally
4. Spatial cross-validation addresses the #1 methodological gap

**Target journal:** NHESS or Remote Sensing (Q1, well within scope)
**Feasibility:** High — all data freely available; GEE for SAR; Python for ML

---

### Direction 2: Multi-Basin Comparative Study with Spatial Cross-Validation [SOLID NOVELTY]
**Concept:** Map flash flood susceptibility across all five major HP river basins (Satluj, Beas, Chenab, Ravi, Yamuna) using identical methodology. Compare performance across basins. Use watershed-based spatial cross-validation. Identify which conditioning factors dominate in which basin (Himalayan vs. sub-Himalayan zones show different patterns).

**Expected contribution:**
1. First state-wide HP flash flood susceptibility study
2. First use of spatial block cross-validation in HP context
3. Basin-to-basin comparison reveals which terrain types drive flash flood risk
4. Direct policy relevance for HP SDMA

**Target journal:** Geomatics, Natural Hazards and Risk (exact target — existing HP papers published here)
**Feasibility:** High — GIS data freely available; main challenge is flood inventory

---

### Direction 3: Antecedent Soil Moisture + Snowmelt as Dynamic Conditioning Factors [METHODOLOGICAL NOVELTY]
**Concept:** Most susceptibility models use static conditioning factors only. For HP, the 2023 flood analysis showed that antecedent soil moisture near-saturation + snowmelt contribution were critical compounding factors. Test whether adding dynamic seasonal factors (30-day prior precipitation index from CHIRPS, or MODIS-based snow cover) improves model accuracy over static-only models.

**Expected contribution:**
1. First study to explicitly test static vs. dynamic conditioning factor sets for Himalayan flash floods
2. Quantifies the added value of soil moisture and snowmelt information
3. Addresses the most commonly stated limitation of existing susceptibility models
4. Relevant to operational early warning systems

**Target journal:** Journal of Hydrology, Natural Hazards, or NHESS
**Feasibility:** Medium — CHIRPS and MODIS snow cover freely available; model comparison framework clear

---

### Direction 4: Explainable AI (XAI) + Infrastructure Risk Mapping [APPLIED NOVELTY]
**Concept:** Produce a flash flood susceptibility map for HP, then overlay with infrastructure exposure data (roads, bridges, hydroelectric projects, tourist facilities). Use SHAP to produce a "factor importance map" showing WHICH conditioning factors drive susceptibility in each micro-region. Produce a district-level risk briefing for HP SDMA and tourism department.

**Expected contribution:**
1. SHAP spatial analysis shows geographic variation in factor importance — novel visualization approach
2. Infrastructure risk quantification directly addresses HiFlo-DAT finding (roads + bridges = primary impact)
3. Tourism sector risk layer addresses a completely unmapped vulnerability
4. High policy relevance for Himachal Pradesh government (SDMA, PWD, Tourism Board)

**Target journal:** Remote Sensing of Environment, NHESS, or International Journal of Disaster Risk Reduction
**Feasibility:** High — SHAP is standard Python library; road/bridge data from OpenStreetMap; tourist facility data available

---

### Direction 5: Multi-Hazard Flash Flood + Landslide Susceptibility Integration [COMPOUND HAZARD NOVELTY]
**Concept:** HP's flash floods frequently co-occur with landslides (landslide-dammed outburst floods, flood-triggered slope failures). Produce simultaneous flash flood AND landslide susceptibility maps. Identify compound hazard zones where both flash flood and landslide susceptibility are high. Assess cascade risk pathways.

**Expected contribution:**
1. Multi-hazard approach addresses a recognized gap in Himalayan hazard literature
2. Cascade pathways are increasingly recognized as critical for HP (Chamoli 2021, Sikkim 2023)
3. Compound hazard index is a novel output for HP
4. Most relevant for emergency planning and land use management

**Target journal:** Landslides (Springer), Natural Hazards, NHESS
**Feasibility:** Medium — requires both flood AND landslide inventory; LiDAR data useful but not available for HP; ALOS PALSAR DEM adequate

---

## 3. Methodological Recommendations for HP Study

### Must-Have Elements (to meet journal standards)
1. **Conditioning factors:** Minimum 10–14 including slope, elevation, TWI, SPI, TRI, drainage density, distance to river, rainfall, LULC, NDVI, soil, lithology, curvature
2. **ML models:** At least RF + XGBoost + one more (SVM, LightGBM, or stacking ensemble)
3. **Validation:** AUC-ROC + Kappa + F1 + Precision + Recall; must use spatial cross-validation (not just random split)
4. **SHAP analysis:** For feature importance and interpretability
5. **Multicollinearity test:** VIF or Pearson correlation before factor selection
6. **Flood inventory:** Minimum 100 flood points; prefer SAR-derived + historical records
7. **Non-flood points:** Random selection from buffer zone (≥5x flood buffer from any flood point)

### Should-Have Elements (for novelty and higher impact)
1. **SAR-based flood inventory** from Sentinel-1 GEE (2018–2024)
2. **Antecedent soil moisture** as dynamic factor
3. **Glacial lake proximity** for high-altitude zones
4. **Spatial block cross-validation** using watershed boundaries
5. **Uncertainty quantification** (bootstrapping or Monte Carlo)
6. **Infrastructure vulnerability layer**

### Nice-to-Have Elements (for top-tier journals)
1. **Future projection** (CMIP6 precipitation change × current susceptibility model)
2. **Transfer learning** across basins
3. **Comparison of DEM resolutions** (ALOS PALSAR 12.5m vs. GLO-30 30m vs. SRTM 30m)
4. **Comparison of rainfall data sources** (CHIRPS vs. GPM-IMERG vs. IMD gridded)
5. **Community-based validation** (ground truthing by local knowledge)

---

## 4. What Makes HP Uniquely Interesting (Justification for Research)

1. **Multiple co-occurring flood mechanisms** (monsoon + GLOF + cloudburst + snowmelt) — not found together elsewhere at this scale
2. **Rapid climate change signal:** Lake area +75% (1990–2022); extreme precipitation events increasing
3. **Tourism and infrastructure exposure:** Millions of tourists, major highways, Himachal's 27 hydroelectric projects
4. **Recent catastrophic events:** 2023 season (404 deaths, Rs. 9,905 crore) provides rich training data
5. **Complete absence of comprehensive ML susceptibility maps** — clear publication gap
6. **HiFlo-DAT database available** for Kullu — ready training data
7. **State-level planning relevance:** HP SDMA directly uses hazard maps for disaster preparedness

---

## 5. Suggested Study Design Summary

**Title options:**
- "Flash Flood Susceptibility Mapping in Himachal Pradesh Using Ensemble Machine Learning and SAR-Based Flood Inventory: Accounting for Multi-Trigger Mechanisms and Spatial Cross-Validation"
- "Multi-Basin Flash Flood Susceptibility Assessment of the Western Himalaya Using Stacking Ensemble Machine Learning with Spatial Block Cross-Validation"
- "Integrating GLOF Risk and Monsoon-Triggered Flash Flood Susceptibility for Comprehensive Hazard Mapping in Himachal Pradesh, India"

**Recommended study area:** Beas + Satluj basins (covers Kullu, Mandi, Shimla, Kinnaur — highest risk districts)
**Or:** Full HP state (novel contribution)

**Recommended ML framework:**
- Base models: RF, XGBoost, LightGBM
- Stacking meta-learner: Logistic Regression
- Evaluation: Spatial block CV (watershed-based) + held-out 2023 event validation
- Interpretability: SHAP global + local analysis

**Recommended conditioning factors (15):**
1. Elevation (ALOS PALSAR DEM)
2. Slope
3. Aspect
4. Plan curvature
5. Profile curvature
6. TWI
7. SPI
8. TRI
9. Drainage density
10. Distance to river
11. Rainfall (GPM-IMERG mean annual + extreme events)
12. LULC (ESA WorldCover)
13. NDVI (Sentinel-2)
14. Lithology (GSI)
15. Soil type (SoilGrids)

**Optional novel factors:**
16. Antecedent precipitation index (30-day prior)
17. Snow cover (MODIS)
18. Distance to glacial lakes (ICIMOD)
19. Road density
