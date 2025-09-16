import json
from pathlib import Path
import pandas as pd
import requests
from dash import Dash, dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px

from src.config import PROC_DIR, REPORTS_DIR

API_URL = "http://127.0.0.1:8000"

def latest_processed_path() -> Path:
    files = sorted(PROC_DIR.glob("features_*.parquet"))
    if not files:
        raise FileNotFoundError("No processed features found. Run `make features` first.")
    return files[-1]

def latest_val_predictions_path() -> Path | None:
    p = REPORTS_DIR / "val_predictions.csv"
    return p if p.exists() else None

# Feature order must match FastAPI
FEATURES = [
    "lag_1","lag_2","lag_3","lag_5","lag_10",
    "roll_ret_mean_5","roll_ret_mean_10","roll_ret_mean_20",
    "roll_ret_std_5","roll_ret_std_10","roll_ret_std_20",
]

def load_latest_feature_row() -> dict:
    df = pd.read_parquet(latest_processed_path()).sort_values("date")
    row = df.iloc[-1]  # most recent
    return {k: float(row[k]) for k in FEATURES}

def predict_from_api(features: dict) -> float:
    payload = {"features": features}
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()["prediction_next_day_close"]

def build_true_vs_pred_figure():
    csv_path = latest_val_predictions_path()
    if not csv_path:
        return None
    df = pd.read_csv(csv_path, parse_dates=["date"])
    fig = px.line(df, x="date", y=["true", "pred"], title="Validation: true vs pred (next-day close)")
    return fig

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # for deployment later if needed

app.layout = dbc.Container(
    [
        html.H2("Agri Forecast — Level 0 UI"),
        html.Hr(),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("1) Load latest features"),
                    dbc.CardBody([
                        dbc.Button("Load from data/processed", id="btn-load", color="primary"),
                        html.Div(id="load-status", className="mt-2 text-muted"),
                        html.Div(id="features-json", style={"display": "none"}),  # hidden store
                    ])
                ], className="mb-3"),
                dbc.Card([
                    dbc.CardHeader("2) Predict next-day close"),
                    dbc.CardBody([
                        dbc.Button("Predict", id="btn-predict", color="success"),
                        html.Div(id="predict-output", className="mt-3 fw-bold"),
                    ])
                ]),
            ], md=4),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Recent validation (true vs pred)"),
                    dbc.CardBody([
                        dcc.Graph(id="val-graph")
                    ])
                ])
            ], md=8)
        ]),

        html.Hr(),
        html.Small("Tip: Run `make ingest → make features → make train → make evaluate → make serve` before the UI."),
    ],
    fluid=True,
)

@app.callback(
    Output("features-json", "children"),
    Output("load-status", "children"),
    Input("btn-load", "n_clicks"),
    prevent_initial_call=True,
)
def on_load_features(n):
    try:
        feats = load_latest_feature_row()
        return json.dumps(feats), dbc.Alert("Loaded latest feature row from processed parquet.", color="info")
    except Exception as e:
        return "", dbc.Alert(f"Failed to load features: {e}", color="danger")

@app.callback(
    Output("predict-output", "children"),
    Input("btn-predict", "n_clicks"),
    State("features-json", "children"),
    prevent_initial_call=True,
)
def on_predict(n, features_json):
    if not features_json:
        return dbc.Alert("Load features first.", color="warning")
    try:
        feats = json.loads(features_json)
        yhat = predict_from_api(feats)
        return dbc.Alert(f"Predicted next-day close: {yhat:,.2f}", color="success")
    except requests.exceptions.RequestException:
        return dbc.Alert("Could not reach FastAPI at http://127.0.0.1:8000. Is `make serve` running?", color="danger")
    except Exception as e:
        return dbc.Alert(f"Prediction failed: {e}", color="danger")

@app.callback(
    Output("val-graph", "figure"),
    Input("btn-load", "n_clicks"),
    prevent_initial_call=False,
)
def refresh_graph(_n=None):
    fig = build_true_vs_pred_figure()
    if fig is None:
        # empty placeholder
        return px.scatter(title="Run `make evaluate` to generate validation plot")
    return fig

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
