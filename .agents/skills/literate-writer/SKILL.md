---
name: literate-writer
description: >-
  Specjalistyczna umiejętność tworzenia i edycji publikacji naukowych oraz raportów
  badawczych w formacie Quarto (.qmd) zgodnie z zasadami Literate Programming (Knuth 1984).
  Używaj tej umiejętności do łączenia rygorystycznej narracji teoretycznej, wzorów matematycznych
  w LaTeX, komórek kodu źródłowego (Python / Polars) oraz dynamicznie powiązanych tabel
  i wykresów w modułowej strukturze manuskryptu naukowej publikacji ML4T.
---

# Umiejętność: Literate Writer (Quarto & Literate Programming)

Umiejętność **literate-writer** wciela w życie zasady Programowania Literackiego (ang. *Literate Programming*) sformułowane przez Donalda Knutha. W kontekście badań nad uczeniem maszynowym w finansach (ML4T), łączy ona w jednym spójnym dokumencie:
1. Uzasadnienie teoretyczne i hipotezy badawcze.
2. Sformalizowany aparat matematyczny (LaTeX).
3. Rzeczywisty kod obliczeniowy generujący wyniki z surowych danych.
4. Interpretację empiryczną bezpośrednio pod wygenerowanymi tabelami i rycinami.

---

## Architektura Manuskryptu

Pliki publikacji zorganizowane są w katalogu `manuscript/`:
```text
manuscript/
├── _quarto.yml             # Globalna konfiguracja renderowania (PDF, HTML, Typst)
├── index.qmd               # Główny punkt wejścia manuskryptu (wstęp, spis treści)
├── references.bib          # Baza bibliograficzna BibTeX
└── sections/               # Poszczególne rozdziały / sekcje artykułu
    ├── 01_data_overview.qmd
    ├── 02_feature_engineering.qmd
    └── ...
```

---

## Wytyczne tworzenia sekcji Quarto (`.qmd`)

1. **Struktura komórek kodu (Code Chunks)**:
   Każda komórka kodu generująca dane wyjściowe musi zawierać deskryptory metadanych Quarto:
   ```markdown
   ```{python}
   #| label: tbl-summary-stats
   #| tbl-cap: "Descriptive Statistics and Risk Metrics for the Asset Universe (2020–2024)"
   #| echo: false
   #| warning: false
   ```
   ```

2. **Odsyłacze krzyżowe (Cross-referencing)**:
   - Tabele muszą być referowane w tekście za pomocą `@tbl-<label>` (np. `Jak przedstawiono w @tbl-summary-stats...`).
   - Równania matematyczne oznaczaj jako `@eq-<label>` z etykietami `\label{eq-...}`.
   - Źródła literaturowe cytuj za pomocą `@<bibtex-key>` (np. `@knuth1984literate`, `@cont2001empirical`).

3. **Matematyka finansowa**:
   Zawsze precyzuj aparat definicyjny przed prezentacją liczb. Przykładowo, zdefiniuj logarytmiczne stopy zwrotu:
   $$r_{i,t} = \ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right)$$
   oraz roczną zmienność:
   $$\sigma_{i,\text{ann}} = \sqrt{252} \cdot \sigma_{i,\text{daily}}$$

4. **Integracja z Data Curator**:
   Dokument Quarto nie powinien ponownie pobierać danych z sieci, lecz odczytywać przetworzone, audytowalne artefakty z `outputs/summary_stats.parquet` lub `outputs/summary_stats.csv`, dbając o pełną powtarzalność.

---

## Procedura weryfikacji i kompilacji

1. Upewnij się, że artefakty w `outputs/` zostały zaktualizowane przez `data-curator`.
2. Zweryfikuj kompilację sekcji:
   ```bash
   quarto preview manuscript/sections/01_data_overview.qmd
   ```
   lub pełnego dokumentu:
   ```bash
   quarto render manuscript
   ```
3. Sprawdź, czy wygenerowane tabele i wykresy nie zawierają niespójności z interpretacją słowną w tekście.
