# Symulator Tomografii CT

## Opis projektu
Projekt to aplikacja Streamlit do symulacji tomografii komputerowej (CT).
Aplikacja pozwala:
- wybrac obraz wejsciowy (probki lub upload),
- wybrac geometrie wiazki (stozkowa lub rownolegla),
- uruchomic transformacje Radona i rekonstrukcje odwrotna,
- porownac RMSE i wykonac eksperymenty parametryczne,
- wyeksportowac finalny obraz jako plik DICOM.

## Struktura projektu

### Glowny plik aplikacji
- `ct.py`
  - glowny interfejs Streamlit,
  - obsluga kontrolek i `session_state`,
  - podglad sinogramu, rekonstrukcji i krokow posrednich,
  - zakladka eksperymentow i eksport DICOM.

### Pakiet logiki aplikacji (`ct_app/`)
- `ct_app/app_config.py`
  - stale konfiguracyjne i etykiety UI,
  - nazwy geometrii i eksperymentow.

- `ct_app/image_utils.py`
  - preprocessing obrazu (`preprocess_image`),
  - filtrowanie sinogramu (`filter_sinogram`),
  - RMSE (`calculate_rmse`),
  - wykrywanie i listowanie obrazow probkowych.

- `ct_app/reconstruction.py`
  - glowny silnik numeryczny (Numba),
  - generowanie geometrii promieni,
  - transformacja Radona,
  - backprojection i rekonstrukcja odwrotna.

- `ct_app/simulation_data.py`
  - warstwa posrednia miedzy UI i silnikiem obliczen,
  - ladowanie obrazu,
  - uruchamianie pelnej symulacji,
  - przygotowanie podgladow i snapshotow,
  - zapis wyniku do `session_state`.

- `ct_app/experiment_data.py`
  - logika eksperymentow RMSE,
  - serie obliczen dla zmian:
    - liczby detektorow,
    - liczby skanow,
    - rozpietosci geometrii.

- `ct_app/dicom_utils.py`
  - tworzenie obiektu DICOM z obrazu rekonstrukcji i metadanych.

- `ct_app/__init__.py`
  - inicjalizacja pakietu i eksport modulow.

### Dodatkowe katalogi
- `tomograf-obrazy/`
  - obrazy probkowe wykorzystywane przez aplikacje i skrypty.

- `scripts/generate_report_data.py`
  - skrypt CLI do generowania danych raportowych (CSV + JSON).

## Wymagania
Python 3.10+ oraz biblioteki:
- streamlit
- numpy
- scikit-image
- pydicom
- numba
- matplotlib

## Uruchomienie aplikacji
```bash
pip install -r requirements.txt
streamlit run ct.py
```

## Generowanie danych do raportu
Pelny przebieg:
```bash
python3 scripts/generate_report_data.py
```

Tryb kompatybilnosci (zachowuje pelne zakresy parametrow):
```bash
python3 scripts/generate_report_data.py --quick
```

Tylko jedna geometria:
```bash
python3 scripts/generate_report_data.py --geometry "Równoległa"
```

Inny obraz probkowy:
```bash
python3 scripts/generate_report_data.py --image Shepp_logan.jpg
```

Wlasny obraz spoza katalogu probek:
```bash
python3 scripts/generate_report_data.py --image-path /sciezka/do/obrazu.png
```

Porownanie RMSE z filtrem i bez filtra dla wybranych obrazow:
```bash
python3 scripts/generate_report_data.py --filter-compare-images Shepp_logan.jpg CT_ScoutView.jpg
```

Pominiecie sekcji porownania filtra:
```bash
python3 scripts/generate_report_data.py --skip-filter-comparison
```

## Co generuje skrypt raportowy
Skrypt tworzy katalog uruchomienia:
`report_data/YYYYMMDD_HHMMSS/`

W srodku znajduja sie:
- `report_data.csv` - zbiorcza tabela RMSE,
- `rmse_<geometria>_<eksperyment>.csv` - pliki per eksperyment,
- `filter_comparison.csv` - RMSE z filtrem i bez filtra,
- `summary.json` - podsumowanie metryk (best/min/max/mean RMSE + porownanie filtra).

## Przeplyw danych
1. `ct.py` zbiera parametry od uzytkownika.
2. `ct_app/simulation_data.py` laduje i normalizuje obraz.
3. `ct_app/reconstruction.py` liczy sinogram i rekonstrukcje.
4. `ct_app/image_utils.py` filtruje i liczy RMSE.
5. `ct.py` pokazuje wyniki, a `ct_app/dicom_utils.py` tworzy DICOM.
