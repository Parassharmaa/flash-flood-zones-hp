# Interpretable Flash Flood Susceptibility Mapping in Yarlung Tsangpo River Basin Using H2O Auto-ML

**Authors:** (Authors from Scientific Reports, 2024)
**Year:** 2024
**Journal:** Scientific Reports (Nature Publishing Group)
**DOI/URL:** https://www.nature.com/articles/s41598-024-84655-y

## Study Area
Yarlung Tsangpo River Basin (upper Brahmaputra), Tibetan Plateau / Himalayan context. A high-altitude, glacier-fed river system with history of catastrophic flash floods and GLOFs.

## Methods
- H2O AutoML platform (automated machine learning — automates model selection and hyperparameter tuning)
- Multiple ML models tested automatically: RF, XGBoost, GBM, DL, GLM, stacked ensembles
- SHAP (SHapley Additive exPlanations) tree interpretation for feature importance
- Both global and local SHAP analysis performed

## Conditioning Factors
Standard topographic and hydrological set:
- Elevation, slope, aspect, curvature
- TWI, SPI, TRI, drainage density, distance to river
- Rainfall, LULC, NDVI, soil, lithology

## Performance
Stacked ensemble from H2O AutoML achieves high AUC (specific values not fully accessible but reported as excellent). AutoML approach automates model comparison and hyperparameter tuning.

## Key Findings
1. **H2O AutoML** automatically selects best model and hyperparameters — reduces bias in manual model selection
2. SHAP analysis reveals **distance to streams** is most influential globally, followed by TWI and elevation
3. Local SHAP values show **spatial variation** in factor importance — same factor matters differently in different parts of the basin
4. AutoML stacked ensemble consistently outperforms manually tuned single models
5. Global SHAP identifies top 5 factors: distance to stream, TWI, elevation, SPI, precipitation
6. SHAP interaction plots reveal non-linear factor relationships

## Limitations
- Yarlung Tsangpo basin is on the Tibetan Plateau — much higher elevation and drier than HP
- AutoML black-box selection reduces methodological transparency
- SHAP local values require careful interpretation to avoid over-reading
- H2O platform requires R or Java backend — less accessible than Python sklearn ecosystem

## Relevance to HP Flash Flood Study
- SHAP analysis is now essentially mandatory for publication in top journals — HP study must include it
- H2O AutoML provides automated model selection — could be used in HP to avoid cherry-picking best model
- Distance to stream → TWI → elevation ranking is highly consistent across different Himalayan-adjacent contexts
- Local SHAP spatial variation (same factor differently important in different zones) is an important finding to test in HP (where Lahaul-Spiti and Kullu Valley have very different conditioning)
- Published in Scientific Reports with Tibetan-Himalayan context → directly analogous to HP
- SHAP interaction plots (between elevation and rainfall, for example) could reveal HP-specific insights about how altitude modulates rainfall effects on flood susceptibility
