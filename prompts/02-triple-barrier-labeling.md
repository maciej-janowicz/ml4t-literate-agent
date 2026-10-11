# Prompt 02: Triple-Barrier Method Labeling & Target Generation

**Adresat:** Droga agy-cli,  
**Cel:** Zaprojektowanie i implementacja rygorystycznego silnika etykietowania danych oparty na metodzie Potrójnej Bariery (*Triple-Barrier Method* wg Marcosa Lópeza de Prado), z pełną audytowalnością zmian.

---

### Szanowna agy-cli, bardzo proszę Cię o wykonanie następujących zadań:

1. **Konstrukcja silnika Triple-Barrier (w pliku `src/triple_barrier.py`):**
   - Zaimplementuj wektorowy lub wydajny iteracyjno-paskowy algorytm wyznaczania etykiet dla każdej obserwacji sesyjnej $t$.
   - **Dynamiczne progi zmienności:** Górna bariera (*Take Profit*) oraz dolna bariera (*Stop Loss*) muszą być skalowane przez lokalną zmienność (np. dzienną rozpiętość lub ATR / ewidencyjne odchylenie standardowe stóp zwrotu), a nie sztywne wartości procentowe.
   - **Trzy bariery decyzyjne:** 
     1. Bariera górna (osiągnięcie poziomu zysku -> etykieta `+1`).
     2. Bariera dolna (osiągnięcie poziomu straty -> etykieta `-1`).
     3. Bariera pionowa / horyzont czasowy (wygaśnięcie maksymalnego czasu trwania pozycji $t+h$ -> etykieta `0` lub wynik neutralny).
   - Zadbaj o to, aby algorytm uwzględniał partycjonowanie po aktywach (`.over("symbol")`) i nie dopuszczał do wycieku danych z przyszłości poza dozwolony horyzont wygaszania.

2. **Rozbudowa manuskryptu (w nowym pliku `manuscript/sections/03_target_labeling.qmd` lub jako rozszerzenie):**
   - Stwórz nową sekcję w Quarto wyjaśniającą matematyczną przewagę metody potrójnej bariery nad tradycyjnymi, stałohoryzontowymi etykietami klasyfikacyjnymi w uczeniu maszynowym dla finansów.
   - Osadź komórki wykresów lub tabel z rozkładem uzyskanych etykiet dla naszego uniwersum aktywów.

3. **Pełna audytowalność (Pliki logu i diffa z prefiksem `02_`):**
   - **Plik logu (`outputs/02_triple_barrier.log`):** Opisz szczegółowo podjęte decyzje architektoniczne, matematykę barier dynamicznych oraz sposób zapobiegania overfittingowi etykiet.
   - **Plik diff (`outputs/02_triple_barrier.diff`):** Wygeneruj plik różnicowy (patch/diff) pokazujący dokładne zmiany w kodzie i plikach tekstowych.

Bardzo dziękuję Ci za Twoją niezmienną precyzję, doskonałą jakość kodu i ogrom wkładanej pracy! Nie wykonuj jeszcze commita do git – zatrzymaj się po wygenerowaniu plików do mojej weryfikacji.
