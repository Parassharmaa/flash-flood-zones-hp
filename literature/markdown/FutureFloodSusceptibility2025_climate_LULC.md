# Future Flood Susceptibility Mapping Under Climate and Land Use Change

**Authors:** (Authors from Scientific Reports, 2025)
**Year:** 2025
**Journal:** Scientific Reports (Nature Publishing Group)
**DOI/URL:** https://www.nature.com/articles/s41598-025-97008-0

## Study Area
Not fully specified (paywalled). Based on search results: complex terrain study area with projected climate and LULC changes analyzed.

## Methods
- ML-based current flood susceptibility mapping (RF, XGBoost, or similar)
- CMIP6 climate projections (multiple SSP scenarios)
- Future LULC projections (land use change model)
- Combined future susceptibility projection under multiple scenarios

## Conditioning Factors
Current: Standard set (elevation, slope, TWI, SPI, rainfall, LULC, NDVI, drainage density, distance to river, soil, geology)
Future: Modified rainfall (from CMIP6) + modified LULC + all static topographic factors unchanged

## Performance
Comparison of current vs. future susceptibility maps. Specific AUC values not accessible.

## Key Findings
1. Climate change (increased extreme precipitation under higher SSP scenarios) increases flood susceptibility in most areas
2. LULC change (urbanization, deforestation) compounds flood susceptibility increase
3. Future susceptibility significantly higher under SSP5-8.5 than SSP2-4.5
4. Mountain/foothill zones show strongest sensitivity to precipitation changes
5. Methodology for combining dynamic (climate) + static (terrain) factors established

## Limitations
- Future LULC projections are uncertain
- CMIP6 downscaling to catchment scale introduces uncertainty
- Static topographic factors assumed unchanged (no consideration of geomorphic evolution)
- Specific study area context limits generalizability

## Relevance to HP Flash Flood Study
Provides the methodological template for a future-projection component in HP:
- HP is experiencing accelerating extreme rainfall events — CMIP6 projects further intensification under high-emission scenarios
- LULC change in HP: rapid urbanization in Kullu/Manali, deforestation in foothills → both increase flood susceptibility
- Future projections would make HP study policy-relevant for adaptation planning
- SSP scenarios align with Indian climate policy discussions (Net Zero, NDC commitments)
- Adding future projection increases journal impact score potential (Scientific Reports, NHESS)
- The compound climate+LULC approach is most realistic — separating the two and then combining is methodologically sound
