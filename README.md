# Analýza maximální teoretické přesnosti TAF dle předpisu L3

Python nástroj vyvinutý v rámci bakalářské práce na Univerzitě obrany pro vyhodnocení maximální teoreticky dosažitelné přesnosti předpovědí TAF při striktním dodržení regulačních omezení ICAO Annex 3 / L3.

## Účel

Projekt simuluje „Ideálního předpovídatele", který má dokonalou znalost budoucích pozorování METAR, ale zároveň je omezen leteckými meteorologickými předpisy definujícími, kdy smí být změny počasí legálně uvedeny v předpovědích TAF.

Software kvantifikuje:
- teoretickou přesnost předpovědi
- snížení detekovatelnosti změn způsobené předpisem
- Brierovo skóre
- matice záměn
- četnost falešných poplachů (FAR)

## Analyzované meteorologické parametry

- Rychlost větru (SKNT)
- Dohlednost (VSBY)
- Význačné meteorologické jevy (WX)

## Metodika

Program porovnává:

1. **Fyzikální model**
   Detekuje každou fyzikální změnu počasí.
2. **Regulační model (L3 model)**
   Aplikuje prahové filtrování a seskupovací logiku dle ICAO Annex 3 / L3.

Rozdíl mezi oběma modely definuje metriku:
- Snížení přesnosti (%)

## Zdroj dat

Program očekává datové sady METAR ve formátu CSV obsahující sloupce:
- station
- valid
- sknt
- vsby
- gust
- wxcodes

Historická letecká meteorologická pozorování byla získána z:
Iowa Environmental Mesonet (IEM)

## Instalace

Doporučená verze: Python 3.10+.

Instalace závislostí:
```bash
pip install -r requirements.txt
```

## Použití

Umístěte datovou sadu METAR ve formátu CSV do složky projektu.

Očekávaný název souboru:
```text
Germany_metars.csv
```

Spuštění:
```bash
python tempoaccuracy.py
```

## Výstup

Program generuje:
- statistický výstup do konzole
- matice záměn
- souhrnné metriky
- souhrnnou CSV tabulku:
```text
taf_precision_summary.csv
```

Univerzita obrany, Brno, Česká republika, 2026.

## Autor

Adam Vlachovský
