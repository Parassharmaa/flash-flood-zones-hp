# Assessment of Flash Flood Vulnerability Zonation Through Geospatial Technique in High Altitude Himalayan Watershed, Himachal Pradesh India

**Authors:** (Authors from Remote Sensing Applications: Society and Environment, 2018)
**Year:** 2018
**Journal:** Remote Sensing Applications: Society and Environment (ScienceDirect)
**DOI/URL:** https://www.sciencedirect.com/science/article/abs/pii/S2352938517301507

## Study Area
Upper Beas Valley, Kullu District, Himachal Pradesh, India — high altitude Himalayan watershed. 20 sub-watersheds analyzed. This is the most upstream portion of the Beas basin, from glaciated headwaters to Kullu town.

## Methods
- GIS-based geospatial technology
- Morphometric analysis using Landsat 8 + ASTER DEM
- Geological and geomorphological data integration
- Sub-watershed prioritization for flash flood risk
- Parameters relevant to flash floods analyzed: drainage characteristics, slope, wetness index

## Conditioning Factors (Morphometric Parameters)
- Drainage characteristics (stream order, stream number, stream length)
- Slope gradient
- Topographic wetness index (TWI)
- Relief ratio
- Drainage density
- Bifurcation ratio
- Elongation ratio, circularity ratio
- Geological/geomorphological parameters

## Performance
No AUC/ROC — geomorphic analysis and ranking approach. Validation against known flood-prone reaches.

## Key Findings
1. **47% of total area** (555 km²) in "very high priority" and "very highly susceptible" to flash floods
2. **More than 90%** of total area found "highly vulnerable" to flash floods in some classification
3. Morphometric analysis effectively prioritizes sub-watersheds without ML
4. Drainage characteristics are the primary morphometric drivers of flash flood risk
5. High-altitude headwaters have high runoff generation capacity due to thin soils and steep slopes

## Limitations
- Morphometric approach cannot incorporate dynamic factors (rainfall events, antecedent moisture)
- No quantitative AUC validation
- Pre-ML approach; cannot capture nonlinear factor interactions
- Landsat 8 + ASTER DEM resolution may be insufficient for precise sub-watershed delineation in high-gradient terrain
- 90%+ vulnerability claim seems inflated — likely artifact of conservative morphometric criteria

## Relevance to HP Flash Flood Study
This is the earliest peer-reviewed susceptibility study for HP's upper Beas valley:
- Provides baseline pre-ML assessment for comparison with modern ML methods
- The 20 sub-watersheds analysis framework is reproducible and extensible to other HP basins
- Morphometric parameters (drainage density, elongation ratio, bifurcation ratio) should be tested as conditioning factors in ML models
- The "47% extremely susceptible" finding is substantially higher than the ML study result (11.81% from Saha et al. 2023) — illustrates methodological sensitivity
- Kullu District coverage aligns with HiFlo-DAT database — could combine for integrated study
- Published in 2018 → no ML components → a 2024/2025 ML update would be a clear methodological advancement
