# Multi-Hazard Susceptibility Mapping for Disaster Risk Reduction in Kargil-Ladakh Region of Trans-Himalayan India

**Authors:** (Authors from Environmental Earth Sciences, 2022)
**Year:** 2022
**Journal:** Environmental Earth Sciences (Springer)
**DOI/URL:** https://link.springer.com/article/10.1007/s12665-022-10729-7

## Study Area
Kargil-Ladakh Region, Trans-Himalayan India — a high-altitude cold-arid region with glaciers, limited monsoon rainfall, and diverse natural hazard exposure. Administratively a part of the Union Territory of Ladakh.

## Methods
- Geo-spatial tools (GIS and remote sensing)
- Frequency Ratio (FR) method applied to all three hazards
- Multi-hazard index created by spatial overlay of individual hazard maps
- Hazard inventories compiled for each hazard type

## Three Hazards Mapped Simultaneously
1. Flash floods
2. Landslides
3. Snow avalanches

## Conditioning Factors
**Flash flood specific:**
- Elevation, slope, aspect
- Distance to river, drainage density
- TWI, rainfall
- LULC, lithology

**Shared factors tested for all three hazards** — frequency ratio computed for each factor-hazard combination.

## Performance
No AUC reported — frequency ratio with qualitative validation against known hazard zones.

## Key Findings
1. Flash floods in Kargil-Ladakh are driven by localized precipitation events despite low overall annual rainfall
2. Multi-hazard approach reveals compound risk zones where flash flood + landslide susceptibility overlap
3. Spatial overlay of three hazard maps identifies "triple-hazard" zones (highest disaster risk)
4. Trans-Himalayan region has distinct hazard patterns vs. Monsoon Himalaya — drier climate shifts flood drivers
5. FR method suitable for data-scarce, ungauged settings like Kargil-Ladakh

## Limitations
- FR method cannot capture nonlinear factor interactions
- No AUC/ROC validation
- High-altitude cold-desert setting is different from HP's monsoon-dominated climate
- Sparse inventory for all three hazards limits model training
- Single method (FR) — no ML comparison

## Relevance to HP Flash Flood Study
- Multi-hazard approach (flash flood + landslide) for Himalayan India is demonstrated here — HP study could adopt this framework
- HP's Lahaul-Spiti district has similar semi-arid, high-altitude characteristics to Kargil-Ladakh — their FR methodology could be adapted
- Compound hazard zone identification (triple overlap) is a novel output that HP study could produce
- FR-based multi-hazard mapping is suitable as a comparison baseline against ML methods in HP
- Snow avalanche + flash flood co-occurrence in high-altitude HP zones is documented here — supports including snow-related variables
