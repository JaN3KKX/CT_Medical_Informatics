# Symulator Tomografii CT

## Opis projektu
Projekt to aplikacja Streamlit do symulacji tomografu komputerowego (CT).
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

## Przeplyw danych
1. `ct.py` zbiera parametry od uzytkownika.
2. `ct_app/simulation_data.py` laduje i normalizuje obraz.
3. `ct_app/reconstruction.py` liczy sinogram i rekonstrukcje.
4. `ct_app/image_utils.py` filtruje i liczy RMSE.
5. `ct.py` pokazuje wyniki, a `ct_app/dicom_utils.py` tworzy DICOM.
