# Data Card: Gia Lai Geospatial Deforestation Dataset

**Dataset Name:** GiaLai-ForestRisk-Dataset-2026  
**Lead Data Engineer:** Mahdi Fattahi (`Borino88`)  
**License:** CC BY 4.0

---

## 1. Dataset Scope & Geographic Coverage
* **Region:** K'Bang and Mang Yang districts, Gia Lai Province, Vietnam (Central Highlands / Tây Nguyên).
* **Bounding Box:** Lat `13.8°N to 14.6°N`, Lon `108.1°E to 108.8°E`.
* **Temporal Span:** 2018–2025 multi-year annual composites.

## 2. Data Sources & Integration
1. **Multi-Spectral Imagery:** Sentinel-2 Level-2A surface reflectance composites (European Space Agency / Copernicus).
2. **Topographical Features:** SRTM 30m Digital Elevation Model (NASA / USGS) aggregated to slope angle and elevation profiles.
3. **Forest Loss Reference Labels:** Hansen Global Forest Change (GFC) v1.10 annual loss layers aggregated to 1 km grid cells as binary labels (`0: Stable`, `1: Deforested/High Risk`).
4. **Infrastructure Proximity:** OpenStreetMap (OSM) road networks and settlement boundaries used to calculate Euclidean distance to nearest road (`dist_road_m`) and settlement (`dist_settlement_m`).

## 3. Preprocessing & Quality Control
* **Cloud Masking:** SCL (Scene Classification Layer) band masking applied to remove cloud, cirrus, and shadow pixels before annual NDVI median composite generation.
* **Spatial Normalization:** All layers reprojected to UTM Zone 48N (EPSG:32648) and resampled to a standardized 1,000m x 1,000m raster grid using bilinear interpolation.
* **Privacy & Security Verification:** Confirmed zero inclusion of private household coordinates, indigenous settlement registries, or proprietary commercial plantation boundaries. All data in this repository is derived from publicly accessible remote sensing platforms.
