# Prediction of Flash Flood Susceptibility Using Integrating Analytic Hierarchy Process (AHP) and Frequency Ratio (FR) Algorithms

**Authors:** Muhammad Majeed, Linlin Lu, Muhammad Mushahid Anwar, Aqil Tariq, Shujing Qin, Mohamed E. El-Hefnawy, Mohamed El-Sharnouby, Qingting Li, Abdulrahman Alasmari
**Year:** 2023
**Journal:** Frontiers in Environmental Science, Volume 10
**DOI/URL:** https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2022.1037547/full

## Study Area
District Jhelum, Punjab, Pakistan — a semi-mountainous region bordered by Rawalpindi, Sargodha, Gujrat, Chakwal, and Azad Kashmir. Characterized by rolling terrain with seasonal monsoon flash floods.

## Methods
- Combined Analytic Hierarchy Process (AHP) + Frequency Ratio (FR) algorithms
- GIS overlay analysis integrating all conditioning factors
- Expert-based weight assignment in AHP
- FR-based statistical weight derivation from historical flood inventory

## Conditioning Factors (8 parameters)
1. Digital Elevation Model (DEM)
2. Slope
3. Distance from rivers
4. Drainage density
5. Land use / Land cover (LULC)
6. Geology
7. Soil resistivity
8. Rainfall deviation

## Performance
- 4% of total area (86.25 km²) classified as high-risk flood zone
- Qualitative validation against known flood-prone settlements (Potha, Samothi, Chaklana)
- No quantitative AUC reported — methodological limitation

## Key Findings
- AHP-FR integration outperforms standalone methods in delineating flood zones
- Rainfall deviation and distance from rivers are the strongest predictors
- Slope and drainage density jointly determine flood speed and extent
- Geology and soil type control infiltration and surface runoff generation
- 4% high-risk classification corresponds to valley floors and riparian zones

## Limitations
- "First of its kind" in Jhelum District — no baseline comparison available
- AHP involves subjective expert weighting — susceptible to expert bias
- No AUC/ROC validation reported — cannot assess quantitative model accuracy
- Static conditioning factors — no dynamic or seasonal variables
- FR requires historical flood inventory — may bias toward recently surveyed areas

## Relevance to HP Flash Flood Study
- AHP-FR is one of the most commonly applied MCDA approaches in South Asian flood studies
- The 8-factor framework (DEM, slope, drainage density, distance to river, LULC, geology, soil, rainfall) is a minimum viable set for any HP study
- The lack of quantitative validation (no AUC) is a common weakness in MCDA papers — HP study should provide quantitative metrics
- Semi-mountainous terrain in Jhelum is somewhat analogous to Himalayan foothills of HP
- Rainfall deviation as a conditioning factor (vs. mean annual rainfall) is a nuanced choice worth adopting for HP
