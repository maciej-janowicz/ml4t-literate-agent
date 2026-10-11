---
name: data-curator
description: >-
  Specjalistyczna umiejętność kurateli danych rynkowych i finansowych przy użyciu
  biblioteki Polars w projektach ilościowych (ML for Trading). Używaj tej umiejętności,
  gdy zachodzi potrzeba pozyskania danych rynkowych (OHLCV, tick, order book), walidacji
  szeregu czasowego pod kątem braków i anomalii, wektorowego obliczania stóp zwrotu
  oraz generowania audytowalnych statystyk opisowych i miar ryzyka (VaR, Sharpe, MDD).
---

# Umiejętność: Data Curator (ML4T & Polars)

Umiejętność **data-curator** odpowiada za bezkompromisową jakość, powtarzalność i audytowalność danych wejściowych w badaniach ilościowych (Quantitative Finance). Działa w oparciu o silnik **Polars** (Apache Arrow / Rust), eliminując wąskie gardła pamięciowe i zapewniając determinizm obliczeń.

---

## Zakres odpowiedzialności

1. **Akwizycja danych giełdowych**:
   - Ingestia notowań z API rynkowych (`yfinance`, AlphaVantage, CCXT) lub plików lokalnych (`Parquet`, `Arrow`, `CSV`).
   - Generowanie deterministycznych serii syntetycznych (GBM ze skokami Student-t) w trybie offline/CI.

2. **Walidacja i integralność**:
   - Wykrywanie braków danych (`null`, `NaN`) i nieciągłości w kalendarzu tradingowym.
   - Identyfikacja anomalii cenowych (split adjustment, dividendo-centryczne korekty).
   - Weryfikacja spójności typów (daty jako `pl.Date` lub `pl.Datetime`, ceny jako `pl.Float64`).

3. **Inżynieria cech finansowych**:
   - Wektorowe obliczanie stóp zwrotu logarytmicznych: $r_t = \ln(P_t / P_{t-1})$.
   - Arytmetyczne stopy zwrotu i stopy skumulowane.
   - Wskaźniki statystyczne: średnia roczna, zmienność zannualizowana ($\sqrt{252}$), skośność, kurtoza (stylized fat tails), Sharpe Ratio, Maximum Drawdown (MDD) oraz Historical Value at Risk (VaR 95%).

4. **Eksport i artefakty**:
   - Zapis wyselekcjonowanych zbiorów do formatu kolumnowego `outputs/curated_market_data.parquet`.
   - Zapis stabelaryzowanych statystyk opisowych (`outputs/summary_stats.csv`, `outputs/summary_stats.parquet`, `outputs/summary_stats.md`) na potrzeby manuskryptu naukowego.

---

## Procedura krok po kroku

### Krok 1: Weryfikacja konfiguracji
Sprawdź plik `config.yaml` w głównym katalogu projektu:
- Zdefiniowane symbole aktywów (`universe.symbols`).
- Horyzont czasowy (`data.start_date` do `data.end_date`).
- Parametry analityczne (`trading_days_per_year: 252`, `risk_free_rate: 0.0425`).

### Krok 2: Uruchomienie potoku kuracji
Wykonaj skrypt kuratora w środowisku projektu:
```bash
python3 src/data_curator.py --config config.yaml
```
W trybie demonstracyjnym lub offline:
```bash
python3 src/data_curator.py --config config.yaml --synthetic
```

### Krok 3: Weryfikacja jakościowa artefaktów
Upewnij się, że w folderze `outputs/` powstały:
- `summary_stats.csv` - tabela do dalszego przetwarzania
- `summary_stats.parquet` - binarna reprezentacja Arrow o ścisłym schemacie
- `summary_stats.md` - wstępnie sformatowana tabela pod Quarto

### Krok 4: Raportowanie do zespołu badawczego
Przekaż wyliczone parametry (np. podwyższona kurtoza akcji technologicznych, asymetria ujemna indeksów) do agenta `literate-writer`.

---

## Standardy implementacyjne Polars

- **Lazy vs Eager**: W przypadku zbiorów przekraczających pamięć RAM stosuj `pl.scan_parquet()` i `pl.LazyFrame`.
- **Wielowątkowość**: Unikaj pętli `for` po wierszach; operuj na wyrażeniach `pl.col(...)` i operacjach okienkowych `.over("symbol")`.
- **Determinizm**: Wszelkie operacje stochastyczne muszą posiadać zafiksowane ziarno (`np.random.seed(42)`).
