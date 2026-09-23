# Third-party notices

Original experiment code is MIT licensed. This does not relicense scientific data or third-party code.

## LiCSAR / Sentinel-1

LiCSAR contains modified Copernicus Sentinel data, analysed by the Centre for the Observation and Modelling of Earthquakes, Volcanoes and Tectonics (COMET). LiCSAR uses JASMIN, the UK's collaborative data analysis environment. Acquisition years and source-product URLs are recorded for every crop in `data/manifest.json` and the per-crop JSON records.

LiCSAR: Lazecky et al. (2020), doi:10.3390/rs12152430. LiCSBAS: Morishita et al. (2020), doi:10.3390/rs12030424.

CEDA's LiCSAR archive license record specifies the Open Government Licence v3: https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/ . Copernicus Sentinel terms also apply to the derived information. Archive catalogue: https://catalogue.ceda.ac.uk/uuid/52cda2e0e6c04272ae15ac836c1e8493 . Archive license statement: https://dap.ceda.ac.uk/neodc/comet/data/licsar_products/00README_catalogue_and_licence.txt .

## Copernicus Emergency Management Service

Copernicus Emergency Management Service (© 2023 European Union), [EMSR648] Kahramanmaras (AOI04): Grading and Monitoring Products. Source: https://mapping.emergency.copernicus.eu/activations/EMSR648/ . The retained original vector records preserve their producer attribution and source identifiers. Citation and dissemination guidance: https://mapping.emergency.copernicus.eu/about/citation-guidelines/ . These are optical photo-interpretations; they are not field structural inspections. The initial product uses building blocks, while the monitoring product uses building points.

## Published recurrent architecture

`src/vendor/rnn_model.py` is from Oliver Stephenson and Eric Zhan's `dpm-rnn-public` repository, commit bc4247fd7126eac66dd07c50149a62be72a316ec. Its MIT license is retained at `src/vendor/LICENSE`.

Source: https://github.com/olliestephenson/dpm-rnn-public . Method: Stephenson et al. (2022), doi:10.1109/TGRS.2021.3084209. Our runner changes training scale, device handling, normalization, and evaluation for this dataset. This is an architecture-based comparison, not a reproduction of the original paper's reported experiment.
