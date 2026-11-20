# Sources and licences

This collection contains material from the four datasets below, each under the Creative Commons Attribution 4.0 International licence
(https://creativecommons.org/licenses/by/4.0/legalcode). The sources do not endorse LemeBase or this collection.

- **Mancino Market** (GitHub, data folder data/v1_2025-11-01). Maintainer: Dr. Jose Mendoza. URL: https://github.com/jrmst102/mancino. Licence: Creative Commons CC BY 4.0 ("share and adapt with attribution"). The source states the data is fully synthetic.
- **Retail Transactions and Stocks Data** (Mendeley Data, version 1). Author: Jimmy Smith. DOI: https://doi.org/10.17632/27x8mjm8k4.1. Licence: CC BY 4.0.
- **VN2 Inventory Planning Challenge** (Zenodo). Author: Vandeput, Nicolas (SupChains). DOI: https://doi.org/10.5281/zenodo.23265416. Licence of the data: CC BY 4.0 (the source's code is MIT and is not included).
- **Retail Transactions and Stocks Data** (Zenodo). Author: Smith, Jimmy. DOI: https://doi.org/10.5281/zenodo.19784493. Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).

## Changes made by LemeBase

- Converted to Parquet; numbers and dates unchanged.
- Columns renamed to shared plain names; one weekly table reshaped from one column per week to one row per week.
- Text categories (product hierarchy, supplier, sales channel, store type, sales and stock status, unit sizes, weekdays) replaced by codes.
- Dropped: product names and descriptions, brands, store names and addresses, customer tables, derived tables, challenge cost columns.
- Renamed and regrouped into Set A to Set D for a forecasting hackathon.
