from __future__ import annotations

import subprocess

from src.run_pipeline import step_fetch_components, step_weights


def main() -> None:
    # Ensure weights exist/refresh
    step_weights("data/weights/qqq_weights.csv")
    # Fetch latest components (1h, last 30 days, top 10)
    step_fetch_components(
        "data/weights/qqq_weights.csv", interval="1h", days=30, max_symbols=10
    )
    # Predict index WSS and signal
    subprocess.run(
        [
            "python",
            "-m",
            "src.predict_index",
            "--weights",
            "data/weights/qqq_weights.csv",
            "--interval",
            "1h",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
