# Forecasting of Flash Flood Susceptibility Mapping Using Random Forest Regression Model and Geographic Information Systems

**Authors:** Mohamed Wahba, Radwa Essam, Mustafa El-Rawy, Nassir Al-Arifi, Fathy Abdalla, Wael M. Elsadek
**Year:** 2024
**Journal:** Heliyon, Vol. 10(13): e33982
**DOI/URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11282991/

## Study Area
Ibaraki Prefecture, Japan (~6,100 km²; eastern Honshu coast; 2.87 million residents). Hilly to flat coastal prefecture with significant seasonal precipitation and flash flood history.

## Methods
- Random Forest regression model
- 70% training / 30% validation split
- ROC curve analysis for model evaluation
- Feature importance extracted from RF model

## Conditioning Factors (11 variables)
1. Elevation
2. Slope
3. Aspect
4. Distance to stream
5. Distance to river
6. Distance to road
7. Land cover
8. Topographic Wetness Index (TWI)
9. Stream Power Index (SPI)
10. Plan curvature
11. Profile curvature

## Performance
- **AUC: 0.9956 (99.56%)** — extremely high
- R² = 0.885
- MAE = 0.137
- MSE = 0.038

Note: AUC of ~0.99 may reflect overfitting or spatial autocorrelation between training and test points.

## Key Findings
- Approximately two-thirds of the prefecture shows low-to-very low flood susceptibility
- One-fifth exhibits high-to-very high susceptibility
- Northwestern areas lower risk; southern regions greater vulnerability
- Proximity factors (distance to river/stream) and topographic factors (TWI, SPI) dominate
- RF regression (not classification) approach provides continuous probability surface vs. discrete classes

## Limitations
- Authors note model "might exhibit slight variations when applied to different case studies" — limited transferability
- Larger datasets of flooded/non-flooded points would enhance accuracy
- Nearly perfect AUC (0.9956) is suspiciously high — likely due to lack of spatial cross-validation
- Japan setting has excellent topographic and historical flood data — may not transfer to data-scarce regions
- Only 11 conditioning factors — rainfall as a static variable, not dynamic

## Relevance to HP Flash Flood Study
- Very high AUC from RF using basic topographic and proximity factors is common in literature but requires skepticism
- The 11-factor setup is minimal and replicable with freely available data — suitable for HP data-scarce context
- The suspiciously high AUC (~0.99) demonstrates the need for proper spatial cross-validation in HP work
- Combining RF regression (probability surface) with classification thresholds is worth exploring
- Distance to stream/river and TWI consistently emerge as top predictors — align with other studies
