import io as bytes_io
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import streamlit as st

from ct_app.app_config import (
    EXPERIMENT_X_LABELS,
    GEOMETRY_FAN,
    GEOMETRY_PARALLEL,
    RECON_ALGO_VERSION,
    SOURCE_BUILTIN,
    SOURCE_UPLOAD,
)
from ct_app.dicom_utils import create_dicom
from ct_app.experiment_data import get_experiment_options, run_experiment
from ct_app.image_utils import calculate_rmse, list_sample_images, resolve_sample_dir
from ct_app.simulation_data import (
    build_preview_frames,
    build_result_signature,
    build_snapshot_frames,
    has_matching_result,
    load_input_image,
    run_simulation,
    save_simulation_result,
)


st.set_page_config(layout="wide", page_title="Symulator Tomografu CT 5.0")
st.title("Symulator Tomografu Komputerowego (Geometria Wachlarzowa / Równoległa)")

tab_simulation, tab_experiments = st.tabs(["Symulacja", "Eksperymenty (RMSE)"])

with tab_simulation:
    st.sidebar.header("Parametry Symulacji")

    sample_dir = resolve_sample_dir()
    sample_images = list_sample_images(sample_dir)

    st.sidebar.subheader("1. Źródło obrazu")
    source_mode = st.sidebar.radio(
        "Wybierz źródło:",
        [SOURCE_BUILTIN, SOURCE_UPLOAD],
        index=0,
        horizontal=True,
    )

    uploaded_file = None
    selected_sample = None
    if source_mode == SOURCE_BUILTIN:
        if sample_images:
            default_sample_idx = sample_images.index("CT_ScoutView.jpg") if "CT_ScoutView.jpg" in sample_images else 0
            selected_sample = st.sidebar.selectbox(
                "Wybierz obraz przykładowy",
                sample_images,
                index=default_sample_idx,
                accept_new_options=False,
            )
        else:
            st.sidebar.warning(f"Folder z przykładami nie istnieje lub jest pusty: {sample_dir}")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Wgraj obraz wejściowy",
            type=["jpg", "png", "jpeg", "bmp", "tif", "tiff", "dcm"],
        )

    st.sidebar.divider()
    st.sidebar.subheader("2. Geometria wiązki")
    beam_geometry = st.sidebar.radio(
        "Wybierz geometrię:",
        [GEOMETRY_FAN, GEOMETRY_PARALLEL],
        index=0,
        horizontal=True,
    )

    st.sidebar.divider()
    st.sidebar.subheader("3. Parametry skanowania")
    scan_steps = st.sidebar.slider("Liczba skanów (krok kąta)", 90, 720, 180, 90)
    detector_count = st.sidebar.slider("Liczba detektorów", 90, 720, 180, 90)

    if beam_geometry == GEOMETRY_FAN:
        fan_span_deg = st.sidebar.slider("Rozpiętość wachlarza (stopnie)", 45, 270, 180, 45)
        parallel_span_pct = 100
    else:
        parallel_span_pct = st.sidebar.slider(
            "Rozpiętość detektorów równoległych (% przekątnej)",
            50,
            200,
            100,
            10,
        )
        fan_span_deg = 180

    fan_span_rad = np.radians(fan_span_deg)
    parallel_span_scale = parallel_span_pct / 100.0

    st.sidebar.divider()
    st.sidebar.subheader("4. Przetwarzanie")
    use_filter = st.sidebar.checkbox("Użyj filtrowania (splot)", value=True)
    run_simulation_clicked = st.sidebar.button("Uruchom symulację", use_container_width=True)

    input_image, input_identifier, input_label, input_study_date, load_error = load_input_image(
        source_mode,
        selected_sample,
        uploaded_file,
        sample_dir,
    )

    if load_error is not None:
        st.error(f"Nie udało się wczytać obrazu: {load_error}")

    if input_image is not None:
        height, width = input_image.shape
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        radius = np.sqrt(center_x**2 + center_y**2)

        current_signature = build_result_signature(
            RECON_ALGO_VERSION,
            input_identifier,
            beam_geometry,
            scan_steps,
            detector_count,
            fan_span_deg,
            parallel_span_pct,
            use_filter,
        )

        has_current_result = has_matching_result(st.session_state, current_signature)
        computed_input_identifier = st.session_state.get("computed_input_identifier")
        image_changed_since_last_compute = input_identifier != computed_input_identifier

        auto_recompute = (not image_changed_since_last_compute) and (not has_current_result)
        should_run_simulation = run_simulation_clicked or auto_recompute

        input_col, sinogram_col, recon_col = st.columns(3)
        input_col.image(input_image, caption=f"Obraz wejściowy: {input_label}", width="stretch", clamp=True)

        if should_run_simulation:
            spinner_message = "Obliczanie (Radon i odwrotna Radon)..."
            if auto_recompute and not run_simulation_clicked:
                spinner_message = "Parametry zmienione - automatyczne przeliczanie..."

            with st.spinner(spinner_message):
                sim_result = run_simulation(
                    input_image,
                    beam_geometry,
                    scan_steps,
                    detector_count,
                    fan_span_rad,
                    parallel_span_scale,
                    use_filter,
                )
                save_simulation_result(st.session_state, sim_result, current_signature)

                st.session_state["computed_input_identifier"] = input_identifier

                has_current_result = True

                if run_simulation_clicked:
                    st.success("Obliczenia zakończone.")

        if has_current_result:
            max_steps = st.session_state["reconstruction_history"].shape[0]
            step_idx = st.slider("Podgląd iteracji", 1, max_steps, max_steps) - 1

            current_sin, current_rec = build_preview_frames(
                st.session_state["sinogram_data"],
                st.session_state["reconstruction_history"],
                st.session_state.get("hit_count_map"),
                step_idx,
            )

            sinogram_col.image(current_sin, caption=f"Sinogram (Iteracja {step_idx + 1})", width="stretch", clamp=True)
            recon_col.image(current_rec, caption=f"Rekonstrukcja (Iteracja {step_idx + 1})", width="stretch", clamp=True)

            snapshot_count = st.slider("Liczba kroków pośrednich rekonstrukcji", 3, 8, 4)
            snapshot_indices, snapshot_images = build_snapshot_frames(
                st.session_state["reconstruction_history"],
                st.session_state.get("hit_count_map"),
                snapshot_count,
            )

            snapshot_cols = st.columns(snapshot_count)
            for col, idx, snapshot in zip(snapshot_cols, snapshot_indices, snapshot_images):
                col.image(snapshot, caption=f"Krok {int(idx) + 1}", width="stretch", clamp=True)

            st.write(f"**RMSE (Bieżący krok):** {calculate_rmse(input_image, current_rec):.4f}")

            st.divider()
            st.subheader("Eksport do DICOM")

            auto_study_date = input_study_date or datetime.now().strftime("%Y%m%d")
            if st.session_state.get("study_date_source_identifier") != input_identifier:
                st.session_state["study_date_input"] = auto_study_date
                st.session_state["study_date_source_identifier"] = input_identifier

            patient_name = st.text_input("Imię i nazwisko pacjenta", "Jan Kowalski")
            patient_id = st.text_input("ID pacjenta", "1234567890")
            study_date = st.text_input("Data badania (YYYYMMDD)", key="study_date_input")
            scan_comments = st.text_input(
                "Komentarz do badania",
                "Badanie tomograficzne.",
            )

            dicom_file = create_dicom(
                st.session_state["final_reconstruction"],
                patient_name,
                patient_id,
                scan_comments,
                study_date,
            )

            dicom_buffer = bytes_io.BytesIO()
            pydicom.filewriter.dcmwrite(dicom_buffer, dicom_file, write_like_original=False)

            st.download_button(
                label="Pobierz plik DICOM",
                data=dicom_buffer.getvalue(),
                file_name="rekonstrukcja.dcm",
                mime="application/dicom",
            )
        else:
            if image_changed_since_last_compute:
                st.info("Zmieniono obraz wejściowy. Kliknij 'Uruchom symulację', aby policzyć wyniki dla nowego obrazu.")
            else:
                st.info("Nie udało się przeliczyć wyników automatycznie. Kliknij 'Uruchom symulację'.")

with tab_experiments:
    st.header("Moduł Analizy Statystycznej (Wykresy RMSE)")
    st.write("W tej sekcji analizujesz zmianę RMSE względem wybranych parametrów skanowania.")

    if input_image is None:
        st.warning("Wybierz obraz przykładowy albo wgraj własny obraz w zakładce Symulacja.")
    else:
        experiment_options = get_experiment_options(beam_geometry)
        experiment_type = st.selectbox("Wybierz eksperyment:", experiment_options)

        if st.button("Uruchom eksperyment"):
            with st.spinner("Uruchamianie serii obliczeń... To może chwilę potrwać."):
                parameter_range, rmse_values = run_experiment(
                    input_image,
                    beam_geometry,
                    experiment_type,
                    radius,
                    width,
                    height,
                    fan_span_rad,
                    parallel_span_scale,
                )

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(parameter_range, rmse_values, marker="o", linestyle="-", color="b")
            ax.set_title("Wpływ parametru na RMSE")
            ax.set_xlabel(EXPERIMENT_X_LABELS.get(experiment_type, "Parametr"))
            ax.set_ylabel("Wartość RMSE")
            ax.grid(True)
            st.pyplot(fig)

            st.success("Eksperyment zakończony pomyślnie.")