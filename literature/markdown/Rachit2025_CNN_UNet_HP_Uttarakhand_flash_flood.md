# Rapid Flood Susceptibility Mapping in the Indian Himalayan Region Using CNN-U-Net Segmentation: Insights from the 2025 Monsoon Events

**Authors:** Rachit, Vaibhav Tripathi, Mohit Prakash Mohanty, Ashish Pandey, Anil Kumar Gupta
**Year:** 2025 (preprint under revision)
**Journal:** Research Square (preprint; under revision for peer-reviewed journal)
**DOI/URL:** https://www.researchsquare.com/article/rs-7715126/v1

## Study Area
Northwestern Indian Himalayan Region — specifically Himachal Pradesh and Uttarakhand states. Validated against June–August 2025 monsoon flood events.

## Methods
- Convolutional Neural Network with **U-Net architecture** employing semantic segmentation
- Pixel-wise flood probability prediction (dense prediction, not point classification)
- 14 hydro-geomorphological predictors as multi-channel input
- Trained on historical flood inventory; validated on 2025 monsoon events
- Compared against Sentinel-1 SAR-based inundation mapping

## Conditioning Factors (14 variables)
1. Altitude
2. Slope
3. Aspect
4. Plan curvature
5. Topographic Wetness Index (TWI)
6. Topographic Roughness Index (TRI)
7. Local Convexity Factor (LCF) — **novel factor**
8. Drainage density
9. Distance from river
10. Lithology
11. Rainfall
12. Land use / Land cover (LULC)
13. NDVI
14. Soil moisture

## Performance
| Metric | Value |
|--------|-------|
| Overall Accuracy | 97.12% |
| AUC-ROC | ~0.99 |
| Critical Success Index | 68.44% |
| RMSE | ~0.145 |

## Key Findings
- CNN U-Net outperformed Sentinel-1 SAR in detecting **localized** flash floods (not just large riverine inundation)
- NDVI, TWI, and altitude are the most influential predictors
- **Novel Local Convexity Factor (LCF)** enhanced delineation accuracy in rugged terrain
- Model successfully captured both riverine and upland flood hotspots
- 2025 monsoon validation confirms temporal generalizability

## Limitations
- Preprint status — not yet peer-reviewed
- Near-perfect AUC (0.99) may indicate data leakage or lack of spatial cross-validation
- Critical Success Index (CSI) of 68.44% reveals missed detections — model not perfect for extreme events
- Dense model requires significant computational resources vs. simpler ML approaches
- Ground truth for flash flood events in HP/Uttarakhand is sparse

## Relevance to HP Flash Flood Study
This is directly the most relevant recent study — HP is the study area. Critical findings:
- CNN U-Net is state-of-the-art for pixel-wise flood susceptibility in HP terrain
- The novel LCF factor is worth testing for HP — captures local concavities that accumulate water
- Soil moisture as a conditioning factor is critical for HP's pre-monsoon vs. peak-monsoon distinction
- The 14-factor framework is a near-complete set for HP
- High CSI (68%) shows room for improvement — feature engineering and better inventory could help
- Preprint status means this work can still be cited but a peer-reviewed version may differ
