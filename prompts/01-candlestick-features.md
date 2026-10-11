# Prompt 01: Continuous Japanese Candlestick Feature Engineering

# Prompt 01 (Revised): Continuous & Contextual Japanese Candlestick Feature Engineering

**Adresat:** Droga agy-cli,  
**Cel:** Zaprojektowanie i implementacja rygorystycznego, ilościowego modułu inżynierii cech opartego na świecach japońskich. Kluczowym wymogiem jest uwzględnienie **kontekstu historycznej zmienności** (rolling statistics, z-scores) zamiast sztywnych miar absolutnych.

---

### Szanowna agy-cli, bardzo proszę Cię o wykonanie następujących zadań w module `src/candlestick_features.py`:

1. **Normalizowana geometryczna reprezentacja świec (Contextual Features):**
   Zamiast surowych wielkości, zdefiniuj cechy w oparciu o ruchome okna statystyczne (np. okno $N = 20$ lub $50$ okresów):
   - **Relative Range Z-Score:** Normalizacja całkowitego zakresu świecy ($High_t - Low_t$) względem średniej kroczącej i odchylenia standardowego zakresu z ostatnich $N$ okresów. Pozwala to wykryć lokalne anomalie zmienności.
   - **Relative Body Z-Score:** Standaryzacja wielkości korpusu względem historycznej dystrybucji korpusów.
   - **Shadow Asymmetry & Proportions:** Ciągłe wskaźniki relacji cieni do korpusu oraz ich odchylenie od lokalnej normy.
   - **Volume Relative Shock:** Stosunek bieżącego wolumenu do ruchomej średniej wolumenu ($\frac{Volume_t}{SMA_N(Volume)}$).

2. **Wydajna implementacja wektorowa w Polars:**
   Wykorzystaj okna ruchome i funkcje agregujące Polarsa (`rolling_mean`, `rolling_std`), aby obliczenia były w pełni wektorowe i zoptymalizowane pod kątem wydajnościowym, bez użycia wolnych pętli.

3. **Rozbudowa manuskryptu (w `manuscript/sections/02_feature_engineering.qmd`):**
   Dodaj sekcję teoretyczną i empiryczną wyjaśniającą, dlaczego tradycyjna analiza świecowa bez uwzględnienia zmienności kontekstowej zawodzi, oraz zaprezentuj, jak nasze znormalizowane wskaźniki (Z-scores) rozwiązują ten problem.

4. **Dziennik operacyjny (`outputs/candlestick_features.log`):**
   Opisz dokładnie w logu, w jaki sposób zaimplementowano okna ruchome i jak zabezpieczono pipeline przed przeciekiem danych (*data leakage* – pamiętaj, że statystyki ruchome muszą uwzględniać wyłącznie dane przeszłe!).

Bardzo dziękuję Ci za Twoje zaangażowanie, dbałość o detale i doskonałą jakość pracy! Nie wykonuj jeszcze commita do git – zatrzymaj się po wygenerowaniu plików do mojej weryfikacji.
