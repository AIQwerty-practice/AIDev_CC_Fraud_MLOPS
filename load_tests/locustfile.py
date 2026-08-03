"""Locust load test for the credit-card fraud FastAPI service.

Start the full application first, then run from the repository root:

    locust -f load_tests/locustfile.py --host http://127.0.0.1:8000

Open http://127.0.0.1:8089/dataset to upload a CSV, then return to the main
Locust page to choose the user count and spawn rate. Environment-variable
configuration remains available for headless and automated runs.
"""

import csv
import io
import os
from threading import Lock
from pathlib import Path

from flask import render_template_string, request
from locust import HttpUser, between, events, task


DEFAULT_HOST = os.getenv("LOCUST_HOST", "http://127.0.0.1:8000")
DEFAULT_SAMPLE = (
    Path(__file__).resolve().parent.parent
    / "backend"
    / "data"
    / "sample_test.csv"
)
SAMPLE_PATH = Path(os.getenv("LOCUST_SAMPLE_CSV", str(DEFAULT_SAMPLE))).expanduser().resolve()
MAX_ROWS = int(os.getenv("LOCUST_SAMPLE_ROWS", "25"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_BATCH_ROWS = 5000
REQUIRED_COLUMNS = [f"V{number}" for number in range(1, 29)] + ["Amount"]
OPTIONAL_COLUMNS = ["Class"]
DATASET_LOCK = Lock()


def prepare_csv(raw_csv: bytes, max_rows: int, source_name: str) -> tuple[bytes, int]:
    """Validate and trim a labeled or unlabeled fraud prediction CSV."""
    if max_rows < 1:
        raise ValueError("Rows per request must be at least 1")
    if max_rows > MAX_BATCH_ROWS:
        raise ValueError(f"Rows per request cannot exceed {MAX_BATCH_ROWS}")
    if not raw_csv:
        raise ValueError("The uploaded CSV is empty")

    try:
        decoded = raw_csv.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The CSV must use UTF-8 encoding") from exc

    reader = csv.reader(io.StringIO(decoded, newline=""))
    try:
        header = [column.strip() for column in next(reader)]
    except StopIteration as exc:
        raise ValueError(f"CSV is empty: {source_name}") from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    unexpected = [column for column in header if column not in REQUIRED_COLUMNS + OPTIONAL_COLUMNS]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected columns: {', '.join(unexpected)}")
    if len(header) != len(set(header)):
        raise ValueError("The CSV contains duplicate column names")

    rows = []
    for line_number, row in enumerate(reader, start=2):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != len(header):
            raise ValueError(f"Row {line_number} has {len(row)} values; expected {len(header)}")
        record = dict(zip(header, row))
        try:
            for column in REQUIRED_COLUMNS:
                float(record[column])
        except ValueError as exc:
            raise ValueError(f"Row {line_number} contains a non-numeric feature value") from exc
        if "Class" in record:
            try:
                class_value = float(record["Class"])
            except ValueError as exc:
                raise ValueError(f"Row {line_number} contains a non-numeric Class value") from exc
            if class_value not in (0.0, 1.0):
                raise ValueError(f"Row {line_number} Class must be 0 or 1")
        rows.append(row)
        if len(rows) >= max_rows:
            break

    if not rows:
        raise ValueError(f"CSV contains no data rows: {source_name}")

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8"), len(rows)


def load_sample_csv(path: Path, max_rows: int) -> tuple[bytes, int]:
    """Read and validate a prediction batch from disk."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Locust sample CSV not found: {path}. "
            "Set LOCUST_SAMPLE_CSV or upload a dataset in the Locust web UI."
        )
    return prepare_csv(path.read_bytes(), max_rows, path.name)


DEFAULT_CSV, DEFAULT_ROW_COUNT = load_sample_csv(SAMPLE_PATH, MAX_ROWS)
ACTIVE_DATASET = {
    "name": SAMPLE_PATH.name,
    "csv": DEFAULT_CSV,
    "rows": DEFAULT_ROW_COUNT,
    "source": "Default file" if SAMPLE_PATH == DEFAULT_SAMPLE.resolve() else "Environment variable",
}


def active_dataset_snapshot() -> dict:
    """Return a consistent active-dataset snapshot for one Locust request."""
    with DATASET_LOCK:
        return ACTIVE_DATASET.copy()


def replace_active_dataset(name: str, csv_bytes: bytes, rows: int, source: str) -> None:
    """Atomically replace the dataset used by subsequent prediction tasks."""
    with DATASET_LOCK:
        ACTIVE_DATASET.update(name=name, csv=csv_bytes, rows=rows, source=source)


DATASET_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Locust Dataset Manager</title>
  <style>
    :root { --bg:#07111f; --panel:#0c1c36; --cyan:#22d3ee; --blue:#3b82f6; --violet:#7c3aed; --text:#e5f2ff; --muted:#9fb3c8; }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; font-family:Inter,Segoe UI,sans-serif; color:var(--text); background:radial-gradient(circle at 18% 0%,rgba(34,211,238,.14),transparent 30%),radial-gradient(circle at 82% 10%,rgba(124,58,237,.15),transparent 32%),linear-gradient(135deg,#050914,#081020 52%,#03101d); }
    main { max-width:880px; margin:0 auto; padding:40px 22px; }
    .hero,.card { border:1px solid rgba(45,211,255,.25); background:linear-gradient(145deg,rgba(12,28,54,.92),rgba(5,15,31,.84)); box-shadow:0 18px 55px rgba(0,0,0,.28),0 0 30px rgba(34,211,238,.12); }
    .hero { border-radius:20px; padding:28px; position:relative; overflow:hidden; }
    .hero h1 { margin:6px 0; font-size:clamp(2rem,5vw,3rem); }
    .kicker { color:var(--cyan); text-transform:uppercase; letter-spacing:.12em; font-size:.76rem; font-weight:800; }
    p { color:var(--muted); line-height:1.6; }
    .card { border-radius:18px; padding:24px; margin-top:18px; }
    .status { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:18px 0; }
    .pill { border:1px solid rgba(45,211,255,.18); border-radius:14px; padding:14px; background:rgba(3,10,24,.52); overflow-wrap:anywhere; }
    .label { color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; }
    .value { margin-top:5px; font-weight:800; }
    label { display:block; margin:14px 0 7px; color:#d8eaff; font-weight:700; }
    input[type=file],input[type=number] { width:100%; padding:12px; color:var(--text); background:rgba(3,10,24,.7); border:1px solid rgba(45,211,255,.22); border-radius:11px; }
    button,.button { display:inline-block; margin-top:18px; padding:11px 18px; border:1px solid rgba(45,211,255,.4); border-radius:11px; color:white; text-decoration:none; font-weight:750; cursor:pointer; background:linear-gradient(135deg,var(--blue),var(--violet)); box-shadow:0 0 22px rgba(37,99,235,.25); }
    button.secondary { margin-left:8px; background:rgba(12,28,54,.8); }
    .message { border-radius:12px; padding:12px 14px; margin-top:16px; }
    .success { color:#a7f3d0; border:1px solid rgba(16,185,129,.35); background:rgba(6,78,59,.23); }
    .error { color:#fecdd3; border:1px solid rgba(244,63,94,.4); background:rgba(127,29,29,.25); }
    code { color:var(--cyan); }
    @media(max-width:650px){ .status{grid-template-columns:1fr} }
  </style>
</head>
<body><main>
  <section class="hero">
    <div class="kicker">FraudGuard AI · Locust</div>
    <h1>Dataset Manager</h1>
    <p>Upload the labeled or unlabeled transaction CSV that subsequent prediction requests should use.</p>
  </section>
  <section class="card">
    <h2>Active dataset</h2>
    <div class="status">
      <div class="pill"><div class="label">File</div><div class="value">{{ dataset.name }}</div></div>
      <div class="pill"><div class="label">Rows/request</div><div class="value">{{ dataset.rows }}</div></div>
      <div class="pill"><div class="label">Source</div><div class="value">{{ dataset.source }}</div></div>
    </div>
    {% if message %}<div class="message {{ message_type }}">{{ message }}</div>{% endif %}
    <form method="post" enctype="multipart/form-data">
      <input type="hidden" name="action" value="upload">
      <label for="dataset">Labeled or unlabeled CSV</label>
      <input id="dataset" name="dataset" type="file" accept=".csv,text/csv" required>
      <label for="rows">Rows included in each prediction request</label>
      <input id="rows" name="rows" type="number" min="1" max="5000" value="{{ dataset.rows }}" required>
      <button type="submit">Upload and activate</button>
    </form>
    <form method="post" style="display:inline">
      <input type="hidden" name="action" value="reset">
      <button class="secondary" type="submit">Restore default dataset</button>
    </form>
    <a class="button" href="/">Return to Locust</a>
    <p>Changes apply to subsequent requests. For predictable results, stop an active test before switching datasets.</p>
  </section>
</main></body></html>
"""


@events.init.add_listener
def add_dataset_page(environment, **_kwargs):
    """Add an upload page to the Locust Flask web interface."""
    if not environment.web_ui:
        return
    web_ui = environment.web_ui

    @web_ui.app.route("/dataset", methods=["GET", "POST"])
    @web_ui.auth_required_if_enabled
    def dataset_page():
        message = ""
        message_type = "success"
        if request.method == "POST":
            action = request.form.get("action", "upload")
            try:
                if action == "reset":
                    replace_active_dataset(
                        SAMPLE_PATH.name,
                        DEFAULT_CSV,
                        DEFAULT_ROW_COUNT,
                        "Default file" if SAMPLE_PATH == DEFAULT_SAMPLE.resolve() else "Environment variable",
                    )
                    message = "The startup dataset has been restored."
                else:
                    upload = request.files.get("dataset")
                    if upload is None or not upload.filename:
                        raise ValueError("Choose a CSV file to upload")
                    if not upload.filename.lower().endswith(".csv"):
                        raise ValueError("The uploaded file must have a .csv extension")
                    if request.content_length and request.content_length > MAX_UPLOAD_BYTES:
                        raise ValueError("The uploaded file is larger than the 10 MB limit")
                    try:
                        requested_rows = int(request.form.get("rows", str(MAX_ROWS)))
                    except ValueError as exc:
                        raise ValueError("Rows per request must be a whole number") from exc
                    uploaded_bytes = upload.read(MAX_UPLOAD_BYTES + 1)
                    if len(uploaded_bytes) > MAX_UPLOAD_BYTES:
                        raise ValueError("The uploaded file is larger than the 10 MB limit")
                    csv_bytes, row_count = prepare_csv(uploaded_bytes, requested_rows, upload.filename)
                    replace_active_dataset(upload.filename, csv_bytes, row_count, "Web upload")
                    message = f"{upload.filename} is active with {row_count} rows per request."
            except ValueError as exc:
                message = str(exc)
                message_type = "error"
        return render_template_string(
            DATASET_PAGE,
            dataset=active_dataset_snapshot(),
            message=message,
            message_type=message_type,
        )


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
        dataset = active_dataset_snapshot()
        files = {"file": (dataset["name"], dataset["csv"], "text/csv")}
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
            elif len(predictions) != dataset["rows"]:
                response.failure(
                    f"Expected {dataset['rows']} predictions, received {len(predictions)}"
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
