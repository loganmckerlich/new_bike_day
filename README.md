# new-bike-day

Analyze Strava cycling data to compare ride performance between bikes.

## Project Structure

```text
new-bike-day/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── data/.gitkeep
├── notebooks/01_eda.ipynb
├── src/
│   ├── auth.py
│   ├── fetch.py
│   ├── weather.py
│   └── ingest.py
├── app/streamlit_app.py
└── tests/
```

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set:

   - `STRAVA_CLIENT_ID`
   - `STRAVA_CLIENT_SECRET`
   - `STRAVA_REFRESH_TOKEN`

## Strava API Credentials

1. Create an app at https://www.strava.com/settings/api.
2. Save your app client ID and client secret in `.env`.
3. Generate a refresh token from Strava OAuth and place it in `.env`.
4. Do not commit `.env` or hardcode credentials.

## Ingestion

Run the ingestion script to refresh a token, fetch activities and streams, enrich weather, and write to a CSV file:

```bash
python src/ingest.py --data-path data/activities.csv
```

Optional arguments:

- `--max-activities <int>`: limit number of fetched activities.

The ingestion is idempotent for activities: already-saved activities are skipped.

## Streamlit

After ingesting data, start the app:

```bash
streamlit run app/streamlit_app.py
```
