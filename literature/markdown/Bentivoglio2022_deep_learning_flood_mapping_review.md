# Deep Learning Methods for Flood Mapping: A Review of Existing Applications and Future Research Directions

**Authors:** Roberto Bentivoglio, Elvin Isufi, Sebastian Nicolaas Jonkman, Riccardo Taormina
**Year:** 2022
**Journal:** Hydrology and Earth System Sciences (HESS), Volume 26, Issue 16
**DOI/URL:** https://hess.copernicus.org/articles/26/4345/2022/

## Study Area
Review scope: global. Covered 58 papers across local, regional, and national scales. Applications examined across river floods, flash floods, and urban floods. No supra-national-scale DL studies identified.

## Methods Reviewed
Three main deep learning architectures:
1. **Multi-Layer Perceptrons (MLPs):** Fully connected networks, often with optimization (genetic algorithms)
2. **Convolutional Neural Networks (CNNs):** Superior for spatial flood analysis; dominant from 2018–2019 onward; uses translational equivariance
3. **Recurrent Neural Networks (RNNs):** LSTM and GRU variants for temporal sequences

Three application categories:
- **Inundation mapping:** observed flood extent (mostly CNN-based)
- **Susceptibility mapping:** qualitative hazard (mostly MLP/RF)
- **Hazard mapping:** quantitative depth/velocity prediction

## Conditioning Factors
Varies by application. For susceptibility studies: topographic derivatives (DEM, slope, TWI, SPI), LULC, soil, geology, rainfall, proximity to river. For inundation mapping: SAR imagery, DEMs, hydrodynamic model outputs.

## Performance
Deep learning generally outperforms traditional statistical methods in accuracy. CNN-based models achieve best spatial performance. LSTM models better at temporal streamflow prediction. No single benchmark AUC reported across all 58 papers.

## Key Findings
1. CNN applications surged 2018–2019 and became the primary approach for spatial flood analysis
2. Deep learning dramatically increases computation speed vs. numerical simulations
3. Models struggle to generalize to unseen geographies (spatial transferability problem)
4. All reviewed models produce **deterministic** outputs — no probabilistic uncertainty
5. River floods most studied; **flash floods** and coastal floods significantly underrepresented
6. No operational real-time flood warning systems identified using DL
7. Scale gaps: no continental-scale DL flood studies found

## Research Gaps Identified
1. **Graph neural networks and neural operators** — needed for generalization across heterogeneous spatial domains
2. **Physics-informed deep learning** — preserving governing hydraulic equations for more reliable surrogate modeling
3. **Probabilistic frameworks** — Bayesian NNs, deep Gaussian processes for uncertainty quantification
4. **Multi-scale integration** — combining local, regional, national assessments
5. **Real-time warning systems** — leveraging speed advantage for operational EWS
6. **Flood risk mapping** — extending beyond susceptibility and hazard characterization

## Limitations
- Review scope limited to English-language papers
- Rapid field evolution means some approaches may already be outdated
- Performance comparison difficult due to inconsistent reporting across studies

## Relevance to HP Flash Flood Study
This is the most authoritative review of DL methods for flood mapping. Key takeaways for HP work:
- CNN/U-Net architectures are state-of-the-art for spatial susceptibility; LSTM for temporal prediction
- Uncertainty quantification is a major gap — reporting confidence intervals would be novel
- Physics-informed approaches (combining hydraulic models with ML) are a frontier
- Flash floods specifically are underrepresented — a HP flash flood DL study fills a real gap
- Real-time operational systems are essentially absent for Himalayan contexts
