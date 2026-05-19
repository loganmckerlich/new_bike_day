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
│   └── fetch.py
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
    - `STRAVA_REDIRECT_URI` (example: `http://localhost:8501`)

## Strava API Credentials

1. Create an app at https://www.strava.com/settings/api.
2. Save your app client ID and client secret in `.env`.
3. Set your Strava app callback URL to match `STRAVA_REDIRECT_URI`.
4. Do not commit `.env` or hardcode credentials.

## Streamlit

Start the app:

```bash
streamlit run app/streamlit_app.py
```

At app startup:

1. Enter your Strava client ID/secret and redirect URI (or preload from `.env`).
2. Click **Sign in with Strava SSO**.
3. Authorize and return with a `code`.
4. Click **Process Data** to fetch, process, and analyze data in memory.

Data is not persisted and will reset with app/session restarts.
