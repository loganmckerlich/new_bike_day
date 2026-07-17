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
├── data/
├── notebooks/01_eda.ipynb
├── src/
│   ├── auth.py            ← OAuth + token refresh
│   ├── causal_inference.py← Doubly robust causal effect estimation
│   ├── database.py        ← Supabase persistence layer
│   ├── dev_data.py        ← Dev-mode static data
│   ├── fetch.py           ← Strava API helpers
│   ├── weather.py         ← Weather enrichment (stubbed values for now)
│   └── webhook.py         ← Webhook server + subscription CLI
├── app/streamlit_app.py
├── app/pages/causal_analysis.py
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
    - `STRAVA_ACCESS_TOKEN` (optional)
    - `STRAVA_WEBHOOK_VERIFY_TOKEN` (secret string you choose; required for webhooks)

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
4. On the first sign-in the app fetches all data from Strava and stores it in Supabase.
5. On subsequent page loads the app serves data directly from the Supabase cache — **no Strava API calls are made**.
6. Use **Get newer data** or **Get older data** to incrementally ingest 30-day windows from Strava.

## Data caching strategy

The app follows a **webhook-driven, static-cache** model:

- **Single initial load**: The Strava API is called once when the Supabase cache is empty.
- **Webhook-triggered updates**: When Strava notifies you that an activity has been created, updated, or deleted, the webhook server re-ingests data from the API and refreshes the cache.
- **All other reads**: Served directly from Supabase — no network calls.
- **Segment geometry**: Polylines, elevation, and altitude streams are cached permanently in Supabase; they are fetched at most once per segment.

## Webhook server

The webhook server must have a publicly accessible HTTPS URL so that Strava can
reach it.  For local development you can use [ngrok](https://ngrok.com/).

### 1 — Choose a verify token

Pick any secret string and set it in two places:

```bash
# In the webhook server environment:
export STRAVA_WEBHOOK_VERIFY_TOKEN="my-secret-verify-token"

# In .streamlit/secrets.toml (so the Streamlit app can display it):
STRAVA_WEBHOOK_VERIFY_TOKEN = "my-secret-verify-token"
```

### 2 — Start the webhook server

```bash
STRAVA_CLIENT_ID=<id> \
STRAVA_CLIENT_SECRET=<secret> \
STRAVA_WEBHOOK_VERIFY_TOKEN=<token> \
python -m src.webhook serve --host 0.0.0.0 --port 8502
```

The server listens on `GET /webhook` (Strava verification) and `POST /webhook`
(incoming events).

### 3 — Register the subscription with Strava

```bash
STRAVA_CLIENT_ID=<id> \
STRAVA_CLIENT_SECRET=<secret> \
python -m src.webhook subscribe \
  --callback-url https://yourhost.example.com/webhook \
  --verify-token my-secret-verify-token
```

### 4 — View or remove subscriptions

```bash
# List active subscriptions
STRAVA_CLIENT_ID=<id> STRAVA_CLIENT_SECRET=<secret> python -m src.webhook view

# Delete a subscription
STRAVA_CLIENT_ID=<id> STRAVA_CLIENT_SECRET=<secret> \
python -m src.webhook unsubscribe --subscription-id 12345
```

> **Note**: You only need **one** webhook subscription per Strava app; Strava
> will deliver events for every athlete who has authorized your app.
