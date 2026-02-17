# Remote Sensing of Surface Water Dynamics — Elephant Butte Reservoir
## Sentinel-2 (2019–2025)

## Study Area

![Elephant Butte Map](outputs/figures/locator_elephant_butte_nm.png)

Elephant Butte Reservoir is a major Rio Grande reservoir in southern New Mexico. This project develops a reproducible remote sensing workflow to monitor seasonal and interannual surface water dynamics using Sentinel-2 satellite imagery.

Such workflows can support reservoir monitoring, drought assessment, and climate resilience planning by providing consistent, satellite-derived surface water indicators.

## Project Goal

Quantify seasonal and interannual surface water extent using optical satellite imagery and the Normalized Difference Water Index (NDWI).

This project emphasizes:

- Remote sensing engineering
- Spatiotemporal analysis
- Transparent, interpretable methods
- Reproducible open-source workflows
- Separation between analysis and visualization pipelines

## Core Questions

- How does surface water extent vary seasonally?
- How stable are interannual comparisons when using late-season metrics?
- What is gained by separating quantitative metrics from visual context imagery?

## Visual Context (True Color RGB)

September 2019 vs September 2025

| 2019 | 2025 |
| --- | --- |
| ![Sentinel-2 RGB September 2019](outputs/images/s2_rgb_2019-09.png) | ![Sentinel-2 RGB September 2025](outputs/images/s2_rgb_2025-09.png) |

True-color median composites were generated in Google Earth Engine using Sentinel-2 L2A surface reflectance. Visualization parameters were explicitly controlled to ensure consistent brightness across years.

## Late-Season Surface Water Metric

![Late-season average surface water area](outputs/figures/lateseason_avg_surfacearea.jpg)

To reduce intra-seasonal variability, surface water area was summarized using an Aug–Oct average for each year.

This late-season metric:

- Reduces short-term inflow noise
- Captures end-of-season reservoir conditions
- Provides a more stable basis for interannual comparison

## Seasonal Animation (September)

![September animation](outputs/maps/story_sept_2019_2025_2000ms.gif)

Monthly median composites were used to generate consistent September frames across years.

## Methods Overview

### Data Sources

- Sentinel-2 L2A (Surface Reflectance)
- Accessed via Microsoft Planetary Computer (STAC + ODC)
- RGB composites generated in Google Earth Engine for presentation

### Processing Workflow

1. STAC-based imagery search and loading  
2. QA60 cloud masking  
3. Monthly median compositing  
4. NDWI computation (NDWI = (Green - NIR) / (Green + NIR))  
5. Threshold-based binary water masking  
6. Pixel-area aggregation (10 m resolution)  
7. Late-season averaging (Aug–Oct)  

## Engineering Design

- Modular `src/` architecture
- Explicit band selection (no hidden defaults)
- Dask-aware computation
- Materialized reference masks to avoid redundant reads
- Separation of analysis and visualization pipelines
- Reproducible export workflows

## Repository Structure





