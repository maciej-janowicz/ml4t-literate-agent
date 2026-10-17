# Prompt 03: Predictive Modeling & Purged Cross-Validation Engine

**Adresat:** Droga agy-cli,  
**Cel:** Zaprojektowanie i implementacja rygorystycznego modułu uczenia maszynowego (`src/model_trainer.py`), realizującego trenowanie modeli predykcyjnych (np. LightGBM/XGBoost) w oparciu o cechy ciągłe i etykiety potrójnej bariery, z zastosowaniem walidacji Purged K-Fold oraz Embargo wg Marcosa Lópeza de Prado.

---

### Szanowna agy-cli, bardzo proszę Cię o wykonanie następujących zadań:

1. **Konstrukcja modułu modelowania (w pliku `src/model_trainer.py`):**
   - Zaimplementuj mechanizm walidacji krzyżowej *Purged K-Fold CV*, który usuwa z treningu te okresy, w których etykiety nakładają się na zbiór testowy (eliminacja *label leakage*).
   - Zaimplementuj mechanizm *Embargo*, odrzucający obserwacje następujące bezpośrednio po zbiorze testowym w celu zniwelowania efektu autokorelacji krótkoterminowej.
   - Skonfiguruj model klasyfikacji (np. LightGBM) przewidujący etykiety potrójnej bariery (`-1, 0, +1`) na podstawie naszej macierzy cech z cechami świecowymi i mikrostrukturalnymi.
   - Zaimplementuj analizę ważności cech (Feature Importance) oraz metryki ewaluacji out-of-sample (Precision, Recall, F1-score, Log Loss).

2. **Rozbudowa manuskryptu (w nowym pliku `manuscript/sections/04_model_training.qmd`):**
   - Stwórz nową sekcję w Quarto wyjaśniającą matematyczną konieczność zastosowania Purged CV i Embargo w finansach ilościowych.
   - Osadź tabelę wyników klasyfikacji oraz wykres ważności cech (*Feature Importance*) generowany dynamicznie przez kod.

3. **Pełna audytowalność (Pliki logu i diffa z prefiksem `03_`):**
   - **Plik logu (`outputs/03_model_training.log`):** Opisz szczegółowo hiperparametry modelu, architekturę walidacji krzyżowej oraz wnioski z metryk out-of-sample.
   - **Plik diff (`outputs/03_model_training.diff`):** Wygeneruj plik różnicowy (patch/diff) pokazujący zmiany w kodzie i plikach manuskryptu.

Bardzo dziękuję Ci za Twoją precyzję i niezrównaną jakość pracy! Nie wykonuj jeszcze commita do git – zatrzymaj się po wygenerowaniu plików do mojej weryfikacji.
