# Improving Urban Flood Susceptibility Mapping Using Transfer Learning

**Authors:** (Authors from Journal of Hydrology, 2021)
**Year:** 2021
**Journal:** Journal of Hydrology
**DOI/URL:** https://www.sciencedirect.com/science/article/abs/pii/S0022169421008271

## Study Area
Multiple urban catchments (for transfer learning experiment). Pre-training in data-rich catchment; transfer to data-sparse catchments.

## Methods
- Convolutional Neural Network (CNN) pre-trained on flood susceptibility in one catchment
- Transfer learning: weights transferred and fine-tuned on new target catchments
- Tested across data-rich and data-sparse scenarios

## Conditioning Factors
Standard urban flood factors: DEM, slope, drainage density, impervious surface, rainfall, distance to river, NDVI.

## Performance
Transfer learning improves model performance in transferred catchments by **10–25%** across different data-rich and data-sparse scenarios. Most improvement seen in data-sparse target catchments.

## Key Findings
1. CNN models transfer well when source and target catchments share similar physiographic characteristics
2. 10–25% performance improvement with transfer learning vs. training from scratch in new catchments
3. Data-sparse catchments benefit most from transfer learning (largest relative improvement)
4. Pre-training on data-rich catchments provides learned feature representations applicable elsewhere
5. Domain adaptation (fine-tuning) needed when source and target are physiographically different

## Limitations
- Urban setting — may not transfer directly to rural Himalayan flash floods
- Requires sufficient labeled data in target catchment for fine-tuning
- Computational cost of CNN training
- Performance improvement varies significantly by catchment pair

## Relevance to HP Flash Flood Study
Directly addresses the sparse data problem in HP:
- Beas basin (data-rich: HiFlo-DAT + 2023 SAR events) could be the source domain
- Chenab, Ravi, or Yamuna basins (data-sparse) could be target domains for transfer
- 10–25% improvement in data-sparse settings is substantial — justifies the transfer learning approach
- The strategy: build best model for Beas basin (most documented), then transfer to other HP basins
- This transfer learning framework would be novel for the Himalayan context and publishable
- Pre-training on global flood datasets (from well-studied regions like Europe, China) and transferring to HP could overcome the HP data scarcity problem
