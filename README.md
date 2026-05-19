# new-bike-day

Analyze Strava cycling data to compare ride performance between bikes.

## Project Structure

```text
new-bike-day/
├── .streamlit/
│   └── secrets.example.toml
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

3. Create Streamlit secrets directory and file:

   ```bash
   mkdir -p .streamlit
   cp .streamlit/secrets.example.toml .streamlit/secrets.toml
   ```

4. Edit `.streamlit/secrets.toml` and set:

    - `STRAVA_CLIENT_ID`
    - `STRAVA_CLIENT_SECRET`
    - `STRAVA_REDIRECT_URI` (example: `http://localhost:8501/`)
    - `STRAVA_ACCESS_TOKEN` (optional, used by **Reload Activities**)

## Strava API Credentials

1. Create an app at https://www.strava.com/settings/api.
2. Save your app client ID and client secret in `.streamlit/secrets.toml`.
3. Set your Strava app callback URL to match `STRAVA_REDIRECT_URI`.
4. Do not commit `.streamlit/secrets.toml` (it is automatically gitignored by Streamlit).

## Streamlit

Start the app:

```bash
streamlit run app/streamlit_app.py
```

At app startup:

1. Configure `.streamlit/secrets.toml` with your Strava client ID, client secret, and redirect URI.
2. Click **Sign in with Strava SSO**.
3. Authorize Strava access and return to the app.
4. The app automatically exchanges the returned code and loads activity data in memory.

Data is not persisted and will reset with app/session restarts.
