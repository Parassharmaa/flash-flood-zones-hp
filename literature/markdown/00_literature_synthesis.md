# Literature Synthesis: Flash Flood Susceptibility/Hazard/Risk Mapping

**Compiled:** March 2026
**Purpose:** Comprehensive review for HP flash flood zone mapping research paper
**Target journals:** NHESS, Remote Sensing, Geomatics Natural Hazards and Risk, Natural Hazards

---

## 1. Overview of the Field

Flash flood susceptibility mapping (FFSM) is a rapidly growing research domain. A 2024 bibliometric analysis of 305 papers (Web of Science, 2007–2022) found that 54.4% of all publications appeared after 2020, indicating explosive recent growth. The field spans five subfields: assessment scale, assessment unit, conditioning factors, prediction models, and validation methodology.

Flash floods are distinct from riverine floods in their rapid onset (minutes to a few hours), high velocity, and strong connection to local terrain. The Himalayan context is especially challenging due to: (1) extreme topographic relief, (2) multiple flood-triggering mechanisms (monsoon, cloudbursts, GLOFs, snowmelt), (3) sparse gauge networks, and (4) rapid climate change.

---

## 2. Methods: What Dominates the Literature

### 2.1 Statistical Bivariate Methods (Older, Still Used)
- **Frequency Ratio (FR):** Calculates the ratio of flood occurrence to total area for each class of each conditioning factor. Simple, transparent, widely used in India and South Asia.
- **Weights of Evidence (WoE):** Bayesian probabilistic approach; requires flood inventory.
- **Shannon's Entropy Index (SEI):** Information-theoretic measure of uncertainty.
- **Statistical Index (SI):** Similar to WoE.

**Typical AUC:** 0.72–0.90. These methods are transparent and physically interpretable but assume independence between conditioning factors (which is rarely true).

### 2.2 Multi-Criteria Decision Analysis (MCDA)
- **Analytic Hierarchy Process (AHP):** Expert-based pairwise comparison. Most widely used MCDA method in South Asian studies.
- **Best-Worst Method (BWM):** More consistent than AHP; requires fewer comparisons.
- **Fuzzy-AHP:** Handles uncertainty in expert judgments.

**Weakness:** Subjective expert weighting; no AUC validation in most papers. Cannot learn complex nonlinear relationships. Often used in combination with FR as hybrid AHP-FR.

### 2.3 Machine Learning Methods (Current Standard)
**Performance ranking (by reported AUC, 2021–2025):**
1. Stacking ensembles (RF + XGBoost + CatBoost → meta-LR): AUC 0.95–0.99
2. XGBoost / LightGBM / CatBoost: AUC 0.88–0.99
3. Random Forest (RF): AUC 0.85–0.98
4. Support Vector Machine (SVM): AUC 0.83–0.99
5. Artificial Neural Network (ANN/MLP): AUC 0.82–0.995
6. Logistic Regression (LR): AUC 0.70–0.88
7. Naïve Bayes (NB): AUC 0.70–0.82

**Key finding:** The extreme range in AUC (0.70–0.99) within individual model types reflects primarily differences in spatial cross-validation rigor, not model quality. Studies without geographic k-fold cross-validation systematically overestimate AUC by 5–15%.

### 2.4 Deep Learning Methods (Emerging)
- **CNN (Convolutional Neural Network):** Best for spatial susceptibility mapping; captures local terrain patterns through convolution. CNN + U-Net architecture achieves pixel-wise flood probability prediction (AUC ~0.99 claimed; requires scrutiny).
- **LSTM (Long Short-Term Memory):** Best for temporal flood forecasting (stream gauge prediction); requires dense temporal data not available in HP.
- **ConvLSTM:** Hybrid combining CNN spatial feature extraction with LSTM temporal modeling; ~26% improvement over LSTM alone for flash flood prediction.
- **Deep Neural Network (DNN):** Achieves AUC 0.91–0.94 in susceptibility mapping; interpretable via SHAP.

**Key limitation of DL:** Requires large training datasets; generalization to new geographies poor; deterministic outputs (no uncertainty); computationally expensive.

### 2.5 Hybrid / Ensemble Approaches (Best Performance)
- **MARS-RF ensemble** (Beas basin HP): AUC 0.88 vs. 0.856 (RF alone) vs. 0.828 (MARS alone)
- **Stacking RF-XGB-CB-LR:** Consistently best in 2024 literature
- **Nature-inspired optimization (PSO, GA, HHO) + ML:** Feature selection + model optimization combined
- **AHP-FR or AHP-ML:** MCDA weighting combined with data-driven models
- **FuzzyAHP-RF / FuzzyAHP-XGB:** Handles uncertainty + learns nonlinearity

---

## 3. Conditioning Factors: What Is Used and What Matters

### 3.1 Most Commonly Used Factors (Global Consensus)
Based on the 2024 bibliometric review and survey of 50+ individual papers:

| Rank | Factor | Category | Notes |
|------|--------|----------|-------|
| 1 | Distance to river/stream | Hydrological proximity | Consistently most important |
| 2 | Elevation (DEM) | Topographic | Low elevation = high susceptibility |
| 3 | Slope | Topographic | Steep = fast runoff; flat = accumulation |
| 4 | TWI (Topographic Wetness Index) | Hydrological terrain | Flow accumulation proxy |
| 5 | Rainfall (annual or event) | Climatic | Key trigger |
| 6 | NDVI | Vegetation | Runoff reduction; varies by context |
| 7 | LULC / Land use | Land cover | Impervious surface increases runoff |
| 8 | Drainage density | Hydrological | High density = faster response |
| 9 | SPI (Stream Power Index) | Erosion potential | Channel incision potential |
| 10 | Lithology / Geology | Subsurface | Infiltration capacity |
| 11 | Soil type | Subsurface | Permeability, hydraulic conductivity |
| 12 | Curvature (plan, profile) | Topographic | Convergence/divergence of flow |
| 13 | TRI (Terrain Ruggedness Index) | Topographic roughness | Roughness proxy |
| 14 | Aspect | Topographic | Solar exposure → evapotranspiration |

### 3.2 Less Common but Potentially Important Factors
- **Antecedent Soil Moisture Index (AMI):** Dynamic factor capturing pre-event saturation; critical for HP where monsoon events accumulate over weeks
- **Snowmelt/glacier contribution:** Critical for high-altitude HP (Lahaul-Spiti, upper Kinnaur)
- **Distance to glacial lakes:** GLOF-specific factor for high-altitude HP
- **Sediment Transport Index (STI):** Rarely included; relevant for HP's high-sediment rivers
- **Road density / distance to road:** Infrastructure-based factor; relevant for HP's road-dependent economy
- **Local Convexity Factor (LCF):** Novel factor from 2025 CNN-U-Net paper; captures local flow convergence in complex terrain
- **Population density:** Relevant for risk mapping (not susceptibility), but sometimes used as vulnerability proxy
- **Aerosol optical depth:** Novel atmospheric precursor factor (Nagamani 2024); not yet standard

### 3.3 Factor Importance Variability by Context
- **Flat alluvial plains (Kosi megafan, Gangetic basin):** NDVI, distance to river dominate
- **Mountainous terrain (Himalayan, Alpine):** Elevation, slope, TWI dominate
- **Arid regions (MENA, Iran):** Rainfall variability, drainage morphology dominate
- **HP specifically:** Elevation, slope, distance to river, TWI, rainfall expected to dominate; but snowmelt and GLOF introduce additional factors not relevant elsewhere

### 3.4 Multicollinearity in Conditioning Factors
Standard practice: remove factors with VIF > 5–10 or Pearson correlation > ±0.8. Commonly correlated pairs:
- Elevation ↔ Slope (moderate correlation)
- TWI ↔ Drainage density (high correlation)
- SPI ↔ Slope (moderate)
After multicollinearity filtering, most studies retain 8–14 factors.

---

## 4. Study Areas: Global Coverage and Gaps

### 4.1 Well-Studied Regions
- China (Jiangxi, Jilin, Yunnan): Extensive ML-based studies
- Iran (various provinces): Bivariate statistical + ML
- Middle East (Morocco, Algeria, Jordan, Turkey): Arid flash flood focus
- Europe (Romania, Balkans): FR, WoE, ML methods
- Bangladesh, Nepal: Monsoon-driven flood susceptibility
- India: Bihar/Gangetic basin, Assam/Brahmaputra, Kerala, West Bengal

### 4.2 Underrepresented Regions
- **Indian Himalaya (Himachal Pradesh, Uttarakhand, J&K):** Sparse
- Western Himalayas generally: Few peer-reviewed ML studies
- Northeast India (Meghalaya, Nagaland): Limited
- High-altitude regions globally: DEM resolution issues limit studies
- Transboundary Himalayan basins: No comprehensive multi-country study

### 4.3 HP-Specific Studies Found (Chronological)
| Study | Year | Methods | Basin | AUC |
|-------|------|---------|-------|-----|
| Flash flood vulnerability zonation (ScienceDirect) | 2017 | GIS morphometric | Unnamed HP watershed | N/A |
| Identification of flash-flood-prone river reaches in Beas (Nat. Hazards) | 2020 | GIS multi-criteria | Beas | N/A |
| Spatial Flash Flood Modeling, Beas basin (Springer book chapter) | 2023 | MARS, RF, ensemble | Beas | 0.88 |
| Geospatial insights HP flash flood vulnerability (Coordinates) | 2024 | GEE + Sentinel-1 | Kullu/Mandi/Shimla | N/A |
| Geospatial/statistical assessment 2023 monsoon HP (Tandfonline) | 2025 | Spatial stats, remote sensing | All HP | N/A |
| HiFlo-DAT flood database Kullu (Int. J. Disaster Risk Reduction) | 2025 | Historical database | Kullu/Beas | N/A |
| CNN-U-Net HP+Uttarakhand (Research Square preprint) | 2025 | CNN U-Net | HP+Uttarakhand | 0.99 |

**Critical gap:** Only ONE peer-reviewed ML susceptibility mapping paper for HP (Beas basin, book chapter, 2023). The full state of HP with all river basins has never been mapped with machine learning methods in a peer-reviewed journal. This is the primary justification for the proposed study.

---

## 5. Himalayan Context: Specific Challenges and Findings

### 5.1 Multiple Flash Flood Triggers in HP
Unlike most studied regions where rainfall is the primary trigger, HP flash floods are triggered by:
1. **Monsoon rainfall (June–September):** Primary cause; 60–80% of annual precipitation
2. **Cloudbursts:** Localized extreme rainfall (>100 mm/hour); Mandi district had most cloudburst incidents in 2023
3. **Glacial Lake Outburst Floods (GLOFs):** Upper Spiti, Lahaul, upper Kinnaur; rapidly increasing with warming (75% lake area increase 1990–2022 in Chenab basin)
4. **Snowmelt:** Upper catchments; compounding with monsoon rainfall inflates flood peaks (July 2023 analysis)
5. **Landslide-dammed lake outbursts:** Common in HP's steep terrain

**Implication for ML modeling:** A single ML model trained on monsoon-triggered flood inventory may not capture GLOF-triggered or cloudburst-triggered events. Sub-region or mechanism-stratified modeling may be necessary.

### 5.2 Cascade / Compound Hazards
HP experiences compound events where multiple hazards co-occur:
- Rainfall → landslide → landslide dam → dam outburst flood
- Earthquake → glacial destabilization → GLOF → flash flood (Sikkim 2023)
- Monsoon saturation → steep slope failure → debris flow → channel blockage → flash flood

Studies mapping single hazards in isolation miss these cascades. Multi-hazard analysis integrating landslide + flash flood susceptibility is a frontier approach (Kargil-Ladakh multi-hazard study used FR for flash flood + landslide + avalanche simultaneously).

### 5.3 Data Challenges in HP
1. **Sparse gauge network:** Critical limitation. Rain gauges concentrated in valley towns, absent in upper catchments.
2. **Satellite rainfall underperformance:** GPM-IMERG and ERA5 showed correlation r = 0.117–0.173 with ground truth in nearby Uttarakhand watershed (Dixit 2026). Implies satellite rainfall data is unreliable in HP's complex terrain.
3. **Cloud cover during monsoon:** Optical satellite imagery (Landsat, Sentinel-2) cloud-contaminated during peak flood season → must use SAR (Sentinel-1).
4. **DEM resolution:** SRTM 30m available; ALOS PALSAR 12.5m superior for deriving terrain parameters; Copernicus GLO-30 (1 arc-second) now freely available.
5. **Historical flood inventory:** Only HiFlo-DAT has systematic HP records (128 events, 1846–2020 for Kullu). Other districts essentially undocumented at sub-event level.

### 5.4 Flash Flood Patterns in HP
- **Most flood-prone districts:** Kullu, Chamba, Kinnaur, Mandi, Kangra, Shimla
- **Most critical river reaches:** Bahang–Manali (Beas), Kullu–Bhuntar (Beas), Manikaran–Kheer-Ganga (Parvati)
- **Seasonal peak:** July–August monsoon; secondary risk March–April (snowmelt + pre-monsoon rain)
- **Worst events:** 2000 Satluj flood (135 deaths, Rs. 1,466 crore), 2023 monsoon (404 deaths, Rs. 9,905 crore)

---

## 6. Validation: How Models Are Assessed

### 6.1 Standard Validation Methods
1. **AUC-ROC:** Most universal; plots sensitivity vs. 1-specificity; area = discrimination ability
2. **Kappa coefficient:** Agreement beyond chance; handles class imbalance
3. **Accuracy, Precision, Recall, F1-Score:** Classification metrics
4. **Seed Cell Area Index (SCAI):** Ratio of percentage area to percentage of flood inventory
5. **Success rate curve vs. Prediction rate curve:** Training performance vs. hold-out performance
6. **Precision-Recall Curve (PRC):** More informative than ROC when classes are imbalanced

### 6.2 Critical Validation Problem: Spatial Autocorrelation
**The most serious methodological flaw in the literature:** Most studies use random 70/30 or 80/20 train-test splits without accounting for spatial autocorrelation. When flood and non-flood points are spatially clustered, nearby points in test set are highly correlated with training set → AUC is artificially inflated by 5–15%.

**Best practice:** Geographic k-fold cross-validation (e.g., leave-one-watershed-out, spatial block cross-validation). Very few published HP studies use this.

### 6.3 Overconfident AUC Values
Multiple studies reporting AUC > 0.98 (e.g., some Japan and China studies with 0.99) should be treated with skepticism. In truly complex terrain with imperfect data, AUC of 0.80–0.90 is realistic. AUC of 0.98+ often reflects:
- Spatial autocorrelation between train and test
- Training and test points in same clusters
- Unrealistically balanced class ratios

**HP benchmark:** The only ML paper for Beas basin reported AUC = 0.88 (ensemble). This is a realistic and honest target.

---

## 7. Key Data Sources Available for HP Study

| Data Type | Source | Resolution | Availability |
|-----------|--------|------------|--------------|
| DEM | ALOS PALSAR (JAXA) | 12.5 m | Free |
| DEM | Copernicus GLO-30 | 30 m | Free |
| DEM | SRTM | 30 m | Free |
| Flood inventory | Sentinel-1 SAR (GEE) | 10 m | Free (2018–present) |
| Flood inventory | HiFlo-DAT | — | Published database |
| Flood inventory | HP SDMA records | — | Government |
| Rainfall | GPM-IMERG | 0.1° (~10 km) | Free |
| Rainfall | CHIRPS | 0.05° (~5 km) | Free |
| Rainfall | IMD gridded | 0.25° | Available |
| LULC | ESA WorldCover 2020/2021 | 10 m | Free |
| LULC | ISRO LISS-based maps | 30 m | Available |
| Soil | HWSD / SoilGrids | 250 m / 250 m | Free |
| Geology | GSI lithology maps | Variable | Available |
| Glacial lakes | ICIMOD glacial lake inventory | 30 m | Published |
| Vegetation | Sentinel-2 NDVI (GEE) | 10 m | Free |

---

## 8. Summary Table of Reviewed Papers

| Author(s) | Year | Journal | Study Area | Methods | AUC | HP Relevance |
|-----------|------|---------|------------|---------|-----|--------------|
| Chen et al. | 2023 | Front. Earth Sci. | Jiangxi, China | MLP, LR, SVM, RF | 0.973–0.975 | Medium |
| Bentivoglio et al. | 2022 | HESS | Global review (58 papers) | DL review | — | Very High |
| Singha et al. | 2024 | Env. Sci. Poll. Res. | West Bengal, India | RF, XGB, 10 ML models | 0.839–0.847 | High |
| Wahba et al. | 2024 | Heliyon | Ibaraki, Japan | RF regression | 0.9956 | Medium |
| Majeed et al. | 2023 | Front. Environ. Sci. | Jhelum, Pakistan | AHP + FR | N/A | Medium |
| Oddo et al. | 2024 | Front. Water | Maryland, USA | ConvLSTM | 26% improvement | Medium |
| Kumar, V. | 2022 | J. Geogr. Nat. Dis. | HP, India | Descriptive | N/A | Very High |
| Rachit et al. | 2025 | Research Square (preprint) | HP + Uttarakhand | CNN U-Net | ~0.99 | Very High |
| Nagamani et al. | 2024 | Sci. Reports | Uttarakhand, India | Atmospheric analysis | N/A | High |
| Shekhar & Negi | 2024 | Coordinates | HP (5 districts) | GEE + SAR | N/A | Very High |
| HiFlo-DAT | 2025 | IJDRR | Kullu, HP | Historical DB | N/A | Very High |
| Dixit et al. | 2026 | NHESS | Uttarakhand | EWS design | N/A | Very High |
| Saha et al. | 2023 | Springer book | Beas, HP | MARS, RF | 0.88 | Very High |
| Beas flood study | 2020 | Nat. Hazards | Beas, HP | GIS multi-criteria | N/A | Very High |
| Chenab GLOF | 2024 | GNH&R | Chenab, HP | AHP | N/A | Very High |
| Indus Kohistan | 2025 | GNH&R | W. Himalaya, Pakistan | RF, XGBoost + BWM | 0.99 | High |
| Karnali Nepal | 2023 | GNH&R | Karnali, Nepal | SVM, RF, ANN | 0.987 | High |
| Abedi et al. | 2021 | Geocarto Int. | Iran (likely) | XGBoost, RF, BRT | 0.85–0.95 | High |
| Stacking ensemble | 2024 | J. Geogr. Sci. | China | RF-XGB-CB stacking | High | High |
| SAR flood inventory | 2024 | GNH&R | SW India | RF, XGB, CNN + SAR | >0.90 | Very High |
| Kosi megafan | 2025 | Sci. Reports | Bihar, India | RS, J48, MaxEnt, ANN | 0.992–0.995 | Medium |
| July 2023 HP floods | 2024 | Nat. Hazards | HP | Hydrometeorological | N/A | Very High |
| Upper Beas morphometric | 2018 | RS Appl. Soc. Env. | Upper Beas, HP | Geospatial morphometric | N/A | Very High |
| Assam hybrid ML | 2022 | Remote Sensing | Assam/Brahmaputra | RF, SVM, GB ensemble | High | Medium |
| Multi-hazard Kargil-Ladakh | 2022 | Env. Earth Sci. | Trans-Himalaya | Frequency Ratio | N/A | High |
| Flash flood drivers India | 2025 | npj Nat. Hazards | India (national) | Statistical analysis | N/A | Very High |
| SHAP/AutoML Yarlung Tsangpo | 2024 | Sci. Reports | Tibet-Himalaya | H2O AutoML + SHAP | High | High |
| Transfer learning flood | 2021 | J. Hydrology | Multiple catchments | CNN transfer learning | 10–25% gain | Medium |
| Non-flood sample uncertainty | 2025 | Remote Sensing | Not specified | OCSVM + FR | High | High |
| Future flood projection | 2025 | Sci. Reports | Complex terrain | ML + CMIP6 + LULC | High | High |
