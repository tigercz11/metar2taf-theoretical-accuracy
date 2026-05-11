# TAF L3 Maximum Theoretical Accuracy Analysis

Python tool developed as part of a bachelor thesis at the University of Defence for evaluating the theoretical maximum achievable accuracy of TAF forecasts under strict ICAO Annex 3 / L3 regulatory constraints.

## Purpose

The project simulates an "Ideal Forecaster" that has perfect knowledge of future METAR observations but is still constrained by aviation meteorological regulations defining when weather changes may legally appear in TAF forecasts.

The software quantifies:

- theoretical forecast accuracy
- reduction in change detectability caused by regulation
- Brier Score
- confusion matrices
- False Alarm Ratio (FAR)

## Analysed meteorological parameters

- Wind speed (SKNT)
- Visibility (VSBY)
- Significant weather phenomena (WX)

## Methodology

The program compares:

1. Physical model  
   Detects every physical weather change.

2. Regulatory model (L3 model)  
   Applies ICAO Annex 3 / L3 threshold filtering and grouping logic.

The difference between both models defines the metric:

- Reduction (%)

## Data source

The program expects METAR datasets in CSV format containing:

- station
- valid
- sknt
- vsby
- gust
- wxcodes

Historical aviation weather observations were obtained from:

Iowa Environmental Mesonet (IEM)

## Installation

Python 3.10+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Place the METAR CSV dataset in the project folder.

Expected filename:

```text
Germany_metars.csv
```

Run:

```bash
python tempoaccuracy.py
```

## Output

The program generates:

- console statistical output
- confusion matrices
- summary metrics
- CSV summary table:

```text
taf_precision_summary.csv
```



University of Defence, Brno, Czech Republic, 2026.

## Author

Adam Vlachovský
