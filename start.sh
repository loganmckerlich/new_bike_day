#!/bin/bash

# ── Read secrets.toml ──────────────────────────────────────────────────────────
SECRETS_FILE=".streamlit/secrets.toml"

if [ ! -f "$SECRETS_FILE" ]; then
  echo "❌  $SECRETS_FILE not found. Run: cp .streamlit/secrets.example.toml .streamlit/secrets.toml"
  exit 1
fi

read_secret() {
  grep "^$1" "$SECRETS_FILE" | sed 's/.*= *"\(.*\)"/\1/'
}

STRAVA_CLIENT_ID=$(read_secret "STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET=$(read_secret "STRAVA_CLIENT_SECRET")
STRAVA_WEBHOOK_VERIFY_TOKEN=$(read_secret "STRAVA_WEBHOOK_VERIFY_TOKEN")

if [ -z "$STRAVA_CLIENT_ID" ] || [ -z "$STRAVA_CLIENT_SECRET" ] || [ -z "$STRAVA_WEBHOOK_VERIFY_TOKEN" ]; then
  echo "❌  Missing one or more required secrets in $SECRETS_FILE"
  echo "    Required: STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, STRAVA_WEBHOOK_VERIFY_TOKEN"
  exit 1
fi

echo "✅  Secrets loaded"

# ── Start ngrok ────────────────────────────────────────────────────────────────
echo "🚇  Starting ngrok on port 8502..."
ngrok http 8502 --log=stdout > /tmp/ngrok.log 2>&1 &
NGROK_PID=$!

# Wait for ngrok to come up and grab the public URL
sleep 3
NGROK_URL=$(curl -s http://127.0.0.1:4040/api/tunnels | python3 -c "
import sys, json
tunnels = json.load(sys.stdin).get('tunnels', [])
for t in tunnels:
    if t.get('proto') == 'https':
        print(t['public_url'])
        break
")

if [ -z "$NGROK_URL" ]; then
  echo "❌  Could not get ngrok URL. Is ngrok installed and running?"
  kill $NGROK_PID 2>/dev/null
  exit 1
fi

echo "✅  ngrok tunnel: $NGROK_URL"

# ── Start webhook server ───────────────────────────────────────────────────────
echo "🔗  Starting webhook server on port 8502..."
STRAVA_CLIENT_ID=$STRAVA_CLIENT_ID \
STRAVA_CLIENT_SECRET=$STRAVA_CLIENT_SECRET \
STRAVA_WEBHOOK_VERIFY_TOKEN=$STRAVA_WEBHOOK_VERIFY_TOKEN \
python -m src.webhook serve --host 0.0.0.0 --port 8502 &
WEBHOOK_PID=$!

sleep 2
echo "✅  Webhook server started"

# ── Register webhook with Strava ───────────────────────────────────────────────
echo "📡  Registering webhook with Strava..."
STRAVA_CLIENT_ID=$STRAVA_CLIENT_ID \
STRAVA_CLIENT_SECRET=$STRAVA_CLIENT_SECRET \
python -m src.webhook subscribe \
  --callback-url "$NGROK_URL/webhook" \
  --verify-token "$STRAVA_WEBHOOK_VERIFY_TOKEN"

echo "✅  Webhook registered"

# ── Start Streamlit ────────────────────────────────────────────────────────────
echo "🌐  Starting Streamlit on port 8501..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  App:     http://localhost:8501"
echo "  Webhook: $NGROK_URL/webhook"
echo "  ngrok:   http://127.0.0.1:4040"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

streamlit run app/streamlit_main.py

# ── Cleanup on exit ────────────────────────────────────────────────────────────
echo "🛑  Shutting down..."
kill $NGROK_PID $WEBHOOK_PID 2>/dev/null