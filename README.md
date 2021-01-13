# Articulatory Data Processor

This repository includes a procedure for processing data features either from articulation and acoustics. This step precedes the procedure described (here)[https://github.com/jaekookang/Articulatory-Data-Extractor]. This is part of unpublished work of Jaekoo Kang's dissertation. Note that data files used here are not uploaded here. You may need to request the use of data if necessary, but it is not guaranteed.

## Status
- [ ] **1. Preprocessing** (in progress)
- [ ] 2. Forward modeling
- [ ] 3. UCM analysis

- Overview:
    - Outlier removal
    - Excluding short tokens (<step_size for formant tracking)
    - Visualizations for articulatory and acoustic data (`data_plots`)
    - Saving the result file (.csv) under `data_processed`

- Note:
    - Data files will not be provided in this repo. You can follow steps described in (link1)[https://github.com/jaekookang/Articulatory-Data-Extractor] and (link2)[https://github.com/jaekookang/Python-EMA-Viewer] to generate data files.

## Requirements
To use this procedure, you have to meet the following requirements:
- EMA data collected using AG501 or NDI WAVE system
- Simultaneous acoustic recording
- Data compatible to MVIEW (developed by Mark Tiede @Haskins labs) in Matlab (Check out the python conversion procedure: (link)[https://github.com/jaekookang/Python-EMA-Viewer])
- data files (`data/*.pkl` and palate files) from (https://github.com/jaekookang/Articulatory-Data-Extractor)[https://github.com/jaekookang/Articulatory-Data-Extractor]


## Procedure
- (1) `01_preprocessing.ipynb`


- (2) `02_check_preprocessing.ipynb`


- You can check your saved data file under `data_processed`


## Reference
- Python-Ema-Viewer: (https://github.com/jaekookang/Python-EMA-Viewer)[https://github.com/jaekookang/Python-EMA-Viewer]
- Articulatory-Data-Extractor: (https://github.com/jaekookang/Articulatory-Data-Extractor)[https://github.com/jaekookang/Articulatory-Data-Extractor]
