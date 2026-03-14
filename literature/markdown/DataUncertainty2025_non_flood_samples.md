# Data Uncertainty of Flood Susceptibility Using Non-Flood Samples

**Authors:** (Authors from Remote Sensing MDPI, 2025)
**Year:** 2025
**Journal:** Remote Sensing (MDPI), Vol. 17(3): 375
**DOI/URL:** https://www.mdpi.com/2072-4292/17/3/375

## Study Area
Not specified in search results (likely China or comparable data-rich setting). Study focuses on the methodological question of how non-flood sample selection affects model accuracy.

## Methods
Comparison of multiple non-flood (negative sample) selection strategies:
1. Random sampling with buffer zone
2. Spatial range from frequency ratio model + random sampling
3. One-class Support Vector Machine (OCSVM) for non-flood detection
4. Various combinations

## Conditioning Factors
Standard set used as testing ground for the methodological comparison.

## Performance
OCSVM-based and FR-model-guided non-flood datasets achieve higher accuracy than random buffer sampling. Specific AUC values not fully accessible.

## Key Findings
1. Non-flood sample selection methodology significantly affects model accuracy — a frequently overlooked source of uncertainty
2. **Random buffer sampling** (most common approach) is suboptimal — introduces both false positives (areas that could flood but haven't been documented) and false negatives
3. **FR model-guided spatial range + OCSVM** produces most accurate non-flood datasets
4. Feature importance analysis can guide where to place negative samples
5. The dependent variable quality (flood/non-flood binary) is as important as conditioning factor quality

## Limitations
- Specific AUC values not accessible
- Single study area limits generalizability
- OCSVM adds complexity to an already complex workflow

## Relevance to HP Flash Flood Study
Addresses a critical methodological gap in HP study design:
- HP's flood inventory is incomplete — HiFlo-DAT covers only Kullu district, and SAR will miss shallow/short-duration floods
- **False negatives in flood inventory:** Undocumented flood locations may be selected as "non-flood" points — corrupting model training
- For HP: should use buffer distance of at least 1 km from any documented flood point when selecting non-flood points (or use FR-guided selection)
- The ratio of flood to non-flood points (typically 1:1 or 1:5) must be explicitly documented and tested
- OCSVM for non-flood detection could be used as a preprocessing step for HP SAR-based inventory
- Citing this paper in the methods section shows awareness of this commonly overlooked source of bias
