# Spatial Analysis of Flood Hazard Zoning Map Using Novel Hybrid Machine Learning Technique in Assam, India

**Authors:** (Authors from Remote Sensing MDPI, 2022)
**Year:** 2022
**Journal:** Remote Sensing (MDPI), Vol. 14(24): 6229
**DOI/URL:** https://www.mdpi.com/2072-4292/14/24/6229

## Study Area
Assam, India (Northeast India) — the Brahmaputra River valley and floodplain. Assam is among the most flood-prone states in India, annually experiencing severe Brahmaputra flooding. Characterized by active tectonics, high seismicity, and extreme monsoon rainfall.

## Methods
Novel hybrid machine learning technique combining:
- Random Forest (RF)
- Support Vector Machine (SVM)
- Gradient Boosting
- Naïve Bayes
- Decision Tree
- 22 flood-influencing factors analyzed

## Conditioning Factors (22 variables — one of the largest factor sets in literature)
Topographic: elevation, slope, aspect, curvature, TRI, TWI, SPI
Hydrological: drainage density, distance to river, stream power
Vegetation: NDVI, LULC
Geological: lithology, soil type
Climatic: rainfall, temperature
Anthropogenic: distance to road, distance to settlements
Additional: geomorphology, land degradation indices

## Performance
Novel hybrid approach achieves high AUC. RF, SVM, Gradient Boosting as stand-alone models also compared. Specific AUC values not accessible (paywalled).

## Key Findings
1. Hybrid ensemble approach outperforms all stand-alone models
2. Assam's flood susceptibility strongly driven by proximity to Brahmaputra and tributaries
3. 22-factor model provides comprehensive assessment but risks multicollinearity
4. Distance to river and elevation are dominant factors
5. Approximately one-third of Brahmaputra region is moderately-to-highly flood-prone

## Limitations
- 22 factors may include redundant/collinear pairs — need VIF testing
- Brahmaputra alluvial setting differs substantially from HP's mountain setting
- High-resolution imagery expensive/limited for a whole-state analysis
- Assam's flat floodplain dynamics differ from HP's confined mountain valleys

## Relevance to HP Flash Flood Study
- The 22-factor framework provides a checklist for HP — though HP terrain will likely show different dominance patterns
- Hybrid ensemble methodology from Assam is directly transferable to HP
- Demonstrates feasibility of state-wide flood susceptibility mapping (as HP study aims to do)
- Northeast India context shows that remote sensing + ML for Indian Himalayan flood susceptibility is well-established
- One-third high susceptibility in alluvial Assam vs. expected lower proportion in HP mountains (rivers constrained to narrow valleys)
- Published in Remote Sensing (MDPI) — a top target journal for HP study
