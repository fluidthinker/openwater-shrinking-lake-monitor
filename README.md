# Remote Sensing of Surface Water Dynamics — Elephant Butte Reservoir
## Sentinel-2 (2019–2025)

## Study Area
<img src="outputs/figures/locator_elephant_butte_nm.png" alt="Elephant Butte Map" width="420">



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

<table>
  <tr>
    <th>September 2019</th>
    <th>September 2025</th>
  </tr>
  <tr>
    <td><img src="outputs/images/s2_rgb_2019-09.png" alt="Sentinel-2 RGB September 2019" width="420"></td>
    <td><img src="outputs/images/s2_rgb_2025-09.png" alt="Sentinel-2 RGB September 2025" width="420"></td>
  </tr>
</table>


True-color median composites were generated in Google Earth Engine using Sentinel-2 L2A surface reflectance. Visualization parameters were explicitly controlled to ensure consistent brightness across years.


# TESTING

<table>
  <tr>
    <th>Late-Season Surface Water Area (Aug–Oct)</th>
    <th>September Surface Water Mask (2019–2025)</th>
  </tr>
  <tr>
    <td valign="top">
      <img src="outputs/figures/late_season_avg_surfacearea_2019_2025.jpg"
           alt="Late-season average surface water area"
           width="480">
    </td>
    <td valign="top">
      <img src="outputs/maps/story_sept_2019_2025_2000ms.gif"
           alt="September binary water mask animation"
           width="480">
    </td>
  </tr>
</table>




# END OF TESTING






## Late-Season Surface Water Metric
<img src="outputs/figures/late_season_avg_surfacearea_2019_2025.jpg" alt="Late-season average surface water area" width="650">



To reduce intra-seasonal variability, surface water area was summarized using an Aug–Oct average for each year.

This late-season metric:

- Reduces short-term inflow noise
- Captures end-of-season reservoir conditions
- Provides a more stable basis for interannual comparison

## Seasonal Animation (September)
<img src="outputs/maps/story_sept_2019_2025_2000ms.gif" alt="September animation" width="500">



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





