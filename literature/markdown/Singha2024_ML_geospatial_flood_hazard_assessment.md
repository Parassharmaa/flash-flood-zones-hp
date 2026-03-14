# Integrating Machine Learning and Geospatial Data Analysis for Comprehensive Flood Hazard Assessment

**Authors:** Chiranjit Singha, Vikas Kumar Rana, Quoc Bao Pham, Duc C. Nguyen, Ewa Łupikasza
**Year:** 2024
**Journal:** Environmental Science and Pollution Research International, Vol. 31(35): 48497–48522
**DOI/URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11297827/

## Study Area
Arambag region, Hooghly district, West Bengal, India (1,044.44 km²). Six blocks with elevation ranging 1–53 m — low-relief, alluvial plain in the Gangetic basin. Flood-prone due to proximity to rivers and monsoon rainfall.

## Methods
- Sentinel-1 SAR analysis combined with Global Flood Database for flood inventory creation
- Ten machine learning algorithms: RF, AdaBoost, rFerns, XGB (XGBoost), DeepBoost, GBM, SDA, BAM, monmlp, MARS
- Feature selection via nature-inspired algorithms: PSO (particle swarm optimization), GA (genetic algorithm), GSO (glowworm swarm optimization), HHO (Harris Hawks optimization), GWO (grey wolf optimizer)
- Boruta and SHAP analyses for feature importance and model interpretability

## Conditioning Factors (15 total)
- Topography: elevation, slope, aspect, curvature, TRI, TWI
- Terrain: geomorphology, lithology
- Land cover: LULC, NDVI
- Hydroclimatic: precipitation, distance to river
- Soil: soil type
- Anthropogenic: distance to road, gMIS (global monthly flood inundation seasonality)

## Performance
| Model | Best AUC |
|-------|----------|
| RF    | 0.847 (resampling factor 5) |
| AdaBoost | 0.839 (resampling factor 10) |
| Overall accuracy range | 74.5–77.8% |

Only 63.3% of variance explained by all conditioning factors — highlights unexplained variance problem.

## Key Findings
- Elevation, precipitation, and distance to rivers are the most crucial factors
- Southern blocks show highest susceptibility (17.2–18.6% highly vulnerable)
- 15.27% of building footprints at high/very high risk; 16.85% of cropland at very high risk
- Nature-inspired feature selection improves model interpretability and accuracy
- SHAP analysis confirms distance to river > precipitation > elevation in importance ranking
- Durbin-Watson test found spatial autocorrelation in OLS residuals (D-W = 0.421) — warns of spatial bias

## Limitations
- Autocorrelation in residuals (D-W = 0.421) — spatial dependence not fully addressed
- Only 63.3% variance explained — key conditioning factors may be missing
- rFerns model showed inconsistency across resampling factors — unstable
- Limited to single region; model transferability unclear
- Low-relief setting (1–53 m elevation) contrasts strongly with mountainous HP terrain

## Relevance to HP Flash Flood Study
- Nature-inspired feature selection (PSO, GA, HHO) is worth exploring for HP study
- SHAP interpretability combined with multiple ML algorithms is a methodological strength to emulate
- The explicit spatial autocorrelation test (Moran's I / Durbin-Watson) should be included in HP analysis
- Low accuracy (74–78%) despite sophisticated methods shows challenge of flood susceptibility in complex terrain
- Distance to river and precipitation consistently dominate — confirms these as core factors for HP
