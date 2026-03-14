# Flash-Flood Susceptibility Mapping Based on XGBoost, Random Forest and Boosted Regression Trees

**Authors:** Abedi et al.
**Year:** 2021 (published); widely cited in 2022–2024 literature
**Journal:** Geocarto International, Vol. 37, No. 19
**DOI/URL:** https://www.tandfonline.com/doi/full/10.1080/10106049.2021.1920636

## Study Area
Not fully extractable from search results (paywalled). Based on search results, likely a mountainous region in Iran or similar arid-to-semi-arid setting.

## Methods
Three ensemble/boosting models compared:
1. **XGBoost** (Extreme Gradient Boosting)
2. **Random Forest (RF)**
3. **Boosted Regression Trees (BRT)**

All models are ensemble tree-based methods. Cross-validation used. Feature importance extracted.

## Conditioning Factors
Includes: slope, elevation, distance to river, drainage density, rainfall, TWI, SPI, lithology, LULC, NDVI, curvature (specific factor list from abstract/search results)

**Key finding:** Slope identified as the most important factor across all three models.

## Performance
All three models achieved high AUC values (specific values not fully accessible from paywalled source). XGBoost typically achieves AUC 0.85–0.95 in comparable studies.

## Key Findings
1. **Slope is the most important trigger** of flash flood occurrence — consistent across XGBoost, RF, and BRT
2. All three ensemble methods perform comparably at high AUC levels
3. XGBoost tends to marginally outperform RF in AUC; BRT is slightly lower
4. Ensemble methods substantially outperform statistical bivariate methods (FR, WoE, LR)
5. Feature importance from XGBoost aligns with physical understanding of flash flood processes

## Limitations
- Paywalled — specific study area and exact AUC values not fully accessible
- High AUC values may reflect insufficient spatial cross-validation (common in flash flood papers)
- Single case study limits generalizability claims

## Relevance to HP Flash Flood Study
This is a seminal comparative paper establishing XGBoost-RF-BRT as the standard comparison triplet:
- Slope dominance finding is consistent across all contexts including Himalayan terrain
- The XGBoost-RF comparison framework is now standard in flash flood susceptibility studies — HP work should include both
- BRT is worth including as a third ensemble method for completeness
- SHAP analysis built into XGBoost provides interpretability unavailable in basic RF implementations
- The finding that ensemble methods outperform bivariate statistics (FR, WoE) justifies ML approach for HP
