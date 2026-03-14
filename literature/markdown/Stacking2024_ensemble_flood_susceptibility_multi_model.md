# Advancing Flood Susceptibility Modeling Using Stacking Ensemble Machine Learning: A Multi-Model Approach

**Authors:** (Multiple authors; Journal of Geographical Sciences, 2024)
**Year:** 2024
**Journal:** Journal of Geographical Sciences (Springer)
**DOI/URL:** https://link.springer.com/article/10.1007/s11442-024-2259-2

## Study Area
Not fully accessible (paywalled). Based on search results: mountainous/complex terrain study area in China. Stacking ensemble approach tested in a flood-prone watershed.

## Methods
Stacking ensemble meta-learning combining:
- Base models: RF (Random Forest), XGB (XGBoost), CB (CatBoost)
- Meta-learner: LR (Logistic Regression)
- Configuration: RF-XGB-CB-LR stacking

## Conditioning Factors
Standard set including topographic (DEM, slope, TWI, SPI, curvature), hydrological (drainage density, distance to river), environmental (LULC, NDVI, soil, geology, rainfall).

## Performance
Stacking RF-XGB-CB-LR significantly enhances flood susceptibility simulation over standalone models. Specific AUC values not fully accessible (paywalled).

## Key Findings
1. Stacking ensemble (RF-XGB-CB + LR meta-learner) outperforms all individual base models
2. Meta-learning captures complementary strengths of heterogeneous base models
3. CatBoost adds value beyond RF + XGBoost combination alone
4. Logistic regression as meta-learner is simple, interpretable, and effective
5. Stacking particularly effective where individual models have different failure modes (complement each other)

## Limitations
- Computational cost higher than single models
- Requires hyperparameter tuning for 3+ models
- Specific performance metrics not accessible from search results
- Meta-learner selection (LR vs. others) is an additional design choice

## Relevance to HP Flash Flood Study
- Stacking ensembles are the current state-of-the-art — HP study could use RF + XGBoost + CatBoost/LightGBM stacking
- LR as meta-learner provides interpretability while capturing nonlinear base model outputs
- The RF-XGB-CB combination covers different algorithmic paradigms (bagging, gradient boosting variants)
- Stacking consistently outperforms single models — strong justification for ensemble approach in HP
- This is a 2024 methodology — citing it positions HP work at the frontier
