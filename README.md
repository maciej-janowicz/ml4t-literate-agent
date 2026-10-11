# ML4T Literate Agent

> Autonomous Quantitative Finance & Machine Learning for Trading Framework powered by Polars, Quarto, and Literate Programming.

---

## 🏛️ Struktura Projektu

```text
ml4t-literate-agent/
├── .agents/
│   └── skills/
│       ├── data-curator/          # Umiejętność akwizycji, walidacji i kuracji danych (Polars)
│       └── literate-writer/       # Umiejętność redagowania manuskryptu naukowego (Quarto)
├── config.yaml                    # Deklaratywna konfiguracja parametrów uniwersum i analiz
├── pyproject.toml                 # Standardowa konfiguracja projektu PEP 621 i zależności
├── manuscript/
│   ├── _quarto.yml                # Konfiguracja renderowania Quarto (HTML, PDF, Typst)
│   ├── index.qmd                  # Główny dokument artykułu naukowego
│   ├── references.bib             # Baza bibliograficzna BibTeX
│   └── sections/
│       └── 01_data_overview.qmd   # Sekcja 01: Przegląd danych i statystyki opisowe
├── src/
│   ├── __init__.py
│   └── data_curator.py            # Moduł kuratora danych oparty na bibliotece Polars
└── outputs/
    └── system_init.log            # Szczegółowy dziennik inicjalizacji systemu
```

---

## 🚀 Szybki start

### 1. Instalacja środowiska

Zalecane utworzenie wirtualnego środowiska:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Uruchomienie Data Curator

Aby pobrać notowania rynkowe i wyliczyć statystyki opisowe:

```bash
python3 src/data_curator.py --config config.yaml
```

W trybie demonstracyjnym / offline z deterministycznym generatorem cen (GBM ze skokami t-Studenta):

```bash
python3 src/data_curator.py --config config.yaml --synthetic
```

### 3. Kompilacja manuskryptu Quarto

```bash
quarto preview manuscript/index.qmd
# lub renderowanie do publikacji HTML / PDF:
quarto render manuscript
```

---

## 📚 Paradygmat Literate Programming

Projekt łączy narrację matematyczno-finansową, kod źródłowy w języku Python (Polars) oraz audytowalne tabele wynikowe w jeden spójny dokument naukowy, eliminując ryzyko błędów replikowalności.
