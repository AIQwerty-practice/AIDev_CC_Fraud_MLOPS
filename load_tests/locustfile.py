"""Locust load test for the credit-card fraud FastAPI service.

Start the full application first, then run from the repository root:

    locust -f load_tests/locustfile.py --host http://127.0.0.1:8000

Open http://127.0.0.1:8089 and choose the user count and spawn rate.
The CSV fixture can be overridden with the LOCUST_SAMPLE_CSV environment variable.
"""

import csv
import io
import os
from pathlib import Path

from locust import HttpUser, between, task


DEFAULT_HOST = os.getenv("LOCUST_HOST", "http://127.0.0.1:8000")
DEFAULT_SAMPLE = (
    Path(__file__).resolve().parent.parent
    / "backend"
    / "data"
    / "sample_test.csv"
)
SAMPLE_PATH = Path(os.getenv("LOCUST_SAMPLE_CSV", str(DEFAULT_SAMPLE))).expanduser().resolve()
MAX_ROWS = int(os.getenv("LOCUST_SAMPLE_ROWS", "25"))


def load_sample_csv(path: Path, max_rows: int) -> tuple[bytes, int]:
    """Read a small, valid prediction batch once when Locust starts."""
    if max_rows < 1:
        raise ValueError("LOCUST_SAMPLE_ROWS must be at least 1")
    if not path.is_file():
        raise FileNotFoundError(
            f"Locust sample CSV not found: {path}. "
            "Set LOCUST_SAMPLE_CSV to a compatible CSV file."
        )

    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Locust sample CSV is empty: {path}") from exc
        rows = []
        for row in reader:
            if row:
                rows.append(row)
            if len(rows) >= max_rows:
                break

    if not rows:
        raise ValueError(f"Locust sample CSV contains no data rows: {path}")

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8"), len(rows)


SAMPLE_CSV, SAMPLE_ROW_COUNT = load_sample_csv(SAMPLE_PATH, MAX_ROWS)


class FraudApiUser(HttpUser):
    """Represent one simulated client of the fraud prediction API."""

    host = DEFAULT_HOST
    wait_time = between(1, 3)

    def on_start(self):
        """Confirm that the API is ready before this user sends predictions."""
        with self.client.get("/health", name="GET /health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"API readiness check failed with HTTP {response.status_code}")
            elif response.text.strip() != "OK":
                response.failure(f"Unexpected health response: {response.text[:200]}")

    @task(6)
    def predict_transactions(self):
        """Upload a representative transaction batch and validate the response."""
        files = {"file": (SAMPLE_PATH.name, SAMPLE_CSV, "text/csv")}
        with self.client.post(
            "/predict",
            files=files,
            name="POST /predict (CSV batch)",
            catch_response=True,
            timeout=120,
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"Prediction failed: HTTP {response.status_code} - {response.text[:300]}"
                )
                return
            try:
                payload = response.json()
            except ValueError:
                response.failure("Prediction response was not valid JSON")
                return

            predictions = payload.get("predictions") if isinstance(payload, dict) else None
            if not isinstance(predictions, list):
                response.failure("Prediction response did not contain a predictions list")
            elif len(predictions) != SAMPLE_ROW_COUNT:
                response.failure(
                    f"Expected {SAMPLE_ROW_COUNT} predictions, received {len(predictions)}"
                )
            elif any("prediction" not in item for item in predictions):
                response.failure("At least one prediction item is missing the prediction field")
            else:
                response.success()

    @task(2)
    def health_check(self):
        """Measure the lightweight health endpoint independently."""
        self.client.get("/health", name="GET /health")

    @task(1)
    def visit_api_home(self):
        """Exercise the API landing page."""
        self.client.get("/", name="GET /")
