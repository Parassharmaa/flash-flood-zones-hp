# Modeling Rules of Regional Flash Flood Susceptibility Prediction Using Different Machine Learning Models

**Authors:** Yuguo Chen, Xinyi Zhang, Kejun Yang, Shiyi Zeng, Anyu Hong
**Year:** 2023
**Journal:** Frontiers in Earth Science (Geohazards and Georisks)
**DOI/URL:** https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2023.1117004/full

## Study Area
Longnan County, Jiangxi Province, China (1,640.55 km²; coordinates 114°23′–114°59′ E, 24°29′–25°01′ N). A hilly sub-tropical region with dense stream networks and history of flash flooding.

## Methods
Four machine learning models compared:
- Multilayer Perceptron (MLP)
- Logistic Regression (LR)
- Support Vector Machine (SVM)
- Random Forest (RF)

70/30 train-test split; ROC curve and susceptibility index distribution used for model evaluation.

## Conditioning Factors
14 environmental variables:
- Elevation, slope, aspect, gully density, highway density
- Rainfall, NDVI, MNDWI, lithology, population density
- Curvature, drainage density, TWI, SPI

## Performance
| Model | AUC |
|-------|-----|
| MLP   | 0.973 |
| RF    | 0.975 |
| SVM   | 0.964 |
| LR    | 0.882 |

RF achieved highest overall performance. LR was weakest but still acceptable.

## Key Findings
- All ML models outperformed logistic regression significantly
- Elevation, gully density, and population density are the most influential factors
- Model prediction accuracy varies from district to district — spatial heterogeneity matters
- Hybrid spatial-ML approaches better capture local variation

## Limitations
- Study area limited to single county in China; generalizability unknown
- No spatial cross-validation (risk of spatial autocorrelation inflating AUC)
- Population density as a driver is unusual for susceptibility (vs. risk) mapping
- No uncertainty quantification

## Relevance to HP Flash Flood Study
This paper establishes the typical ML comparison framework. The finding that prediction accuracy "varies from district to district" is especially relevant to HP's highly varied topography. The 14-factor setup including gully density and population density provides a useful baseline for conditioning factor selection. The district-level variation finding supports sub-regional analysis for HP.
