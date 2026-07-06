"""SEED ICA-cleaned DE+RJSD preprocessing with 4 s windows and a 1 s hop."""

from preprocess_seed_de_rjsd_ica import main


if __name__ == "__main__":
    main(default_window_seconds=4.0, default_hop_seconds=1.0)
