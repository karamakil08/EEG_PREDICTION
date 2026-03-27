import numpy as np
import pandas as pd
from pathlib import Path
from scipy.fft import fft, fftfreq

INPUT_DIR = Path("YOUR FILE DIRECTORY")
OUTPUT_FILE = "FINAL_DATASET_WINDOWED_FULL.csv"
WINDOW_SECONDS = 5
SFREQ = 500

def parse_metadata(filename):
    clean_name = filename.lower()
    if "healthy" in clean_name:
        label = 0
        prefix = "H"
    else:
        label = 1
        prefix = "MS"

    try:
        number_part = filename.split('_')[0]
        subject_id = f"{prefix}_{number_part}"
    except:
        subject_id = filename
    return subject_id, label

def get_band_power(data, fs, band_range):
    n = len(data)
    fft_vals = fft(data)
    psd = (np.abs(fft_vals) ** 2) / (n * fs)
    freqs = fftfreq(n, 1 / fs)

    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    psd = psd[pos_mask]

    f_min, f_max = band_range
    idx = np.logical_and(freqs >= f_min, freqs <= f_max)

    resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1
    return np.sum(psd[idx]) * resolution

def main():
    if not INPUT_DIR.exists():
        print(f"Directory {INPUT_DIR} not found.")
        return

    files = sorted(INPUT_DIR.glob("*.csv"))
    files = [f for f in files if "features" not in f.name and "FINAL" not in f.name]

    if not files:
        print("No input files found.")
        return

    all_rows = []

    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            sub_id, label = parse_metadata(file_path.name)

            samples_per_window = int(WINDOW_SECONDS * SFREQ)
            total_windows = len(df) // samples_per_window
            eeg_cols = [c for c in df.columns if c not in ['Time', 'Timestamp', 'Unnamed: 0']]

            print(f"Processing {sub_id}... {total_windows} windows.")

            for i in range(total_windows):
                start = i * samples_per_window
                end = start + samples_per_window
                window_df = df.iloc[start:end]

                row = {
                    "Subject_ID": sub_id,
                    "Label": label,
                    "Window_Index": i
                }

                for channel in eeg_cols:
                    data = window_df[channel].values

                    d = get_band_power(data, SFREQ, [0.5, 4])
                    t = get_band_power(data, SFREQ, [4, 8])
                    a = get_band_power(data, SFREQ, [8, 16])
                    b = get_band_power(data, SFREQ, [16, 31])
                    g = get_band_power(data, SFREQ, [31, 50])

                    row[f"{channel}_abs_delta"] = d
                    row[f"{channel}_abs_theta"] = t
                    row[f"{channel}_abs_alpha"] = a
                    row[f"{channel}_abs_beta"] = b
                    row[f"{channel}_abs_gamma"] = g

                    total_power = d + t + a + b + g
                    if total_power > 0:
                        row[f"{channel}_rel_delta"] = d / total_power
                        row[f"{channel}_rel_theta"] = t / total_power
                        row[f"{channel}_rel_alpha"] = a / total_power
                        row[f"{channel}_rel_beta"] = b / total_power
                        row[f"{channel}_rel_gamma"] = g / total_power

                    if b > 0:
                        row[f"{channel}_alpha_beta"] = a / b
                        row[f"{channel}_beta_gamma"] = b / g if g > 0 else 0
                        row[f"{channel}_beta_theta"] = b / t if t > 0 else 0
                        row[f"{channel}_beta_delta"] = b / d if d > 0 else 0

                    if g > 0:
                        row[f"{channel}_alpha_gamma"] = a / g
                        row[f"{channel}_theta_gamma"] = t / g
                        row[f"{channel}_delta_gamma"] = d / g

                    if d > 0:
                        row[f"{channel}_alpha_delta"] = a / d
                        row[f"{channel}_theta_delta"] = t / d

                    if t > 0:
                        row[f"{channel}_alpha_theta"] = a / t

                all_rows.append(row)

        except Exception as e:
            print(f"Error with {file_path.name}: {e}")

    if all_rows:
        final_df = pd.DataFrame(all_rows)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Success! Saved {len(final_df)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()