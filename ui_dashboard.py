import os
import json
import pandas as pd
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output

# Configuration
FEATURE_VECTOR_FILE = "output_folder/feature_vectors.csv"
FATIGUE_ALERTS_FILE = "output_folder/fatigue_alerts.json"


def load_feature_data():
    """
    Loads typing speed and pause interval data from the feature vector CSV file.
    """
    if os.path.exists(FEATURE_VECTOR_FILE):
        df = pd.read_csv(FEATURE_VECTOR_FILE)
        return df
    return None


def load_fatigue_alerts():
    """
    Loads fatigue detection alerts from a JSON file.
    """
    if os.path.exists(FATIGUE_ALERTS_FILE):
        with open(FATIGUE_ALERTS_FILE, "r") as f:
            return json.load(f)
    return {"fatigue_detected": False, "alerts": {}, "recommendations": []}


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container(
    [
        html.H1("Typing Fatigue Monitoring Dashboard", className="text-center mt-4"),
        
        # Fatigue Detection Status
        dbc.Alert(id="fatigue-status", className="text-center mt-2"),
        
        # Row for statistics
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="typing-speed-graph"), width=6),
                dbc.Col(dcc.Graph(id="pause-intervals-graph"), width=6),
            ]
        ),

        # Recommendations Section
        html.H4("Fatigue Recommendations", className="mt-3"),
        html.Ul(id="fatigue-recommendations"),
        
        # Auto-refresh every 10 seconds
        dcc.Interval(id="interval-update", interval=10 * 1000, n_intervals=0),
    ],
    fluid=True,
)


@app.callback(
    [
        Output("typing-speed-graph", "figure"),
        Output("pause-intervals-graph", "figure"),
        Output("fatigue-status", "children"),
        Output("fatigue-status", "color"),
        Output("fatigue-recommendations", "children"),
    ],
    Input("interval-update", "n_intervals"),
)
def update_dashboard(n):
    """
    Updates dashboard components with real-time fatigue data.
    """
    df = load_feature_data()
    fatigue_data = load_fatigue_alerts()

    if df is None:
        return dash.no_update, dash.no_update, "No Data Available", "secondary", []

    # Typing Speed Graph
    fig_speed = px.line(df, y="mean_pause", title="Typing Speed Over Time")
    fig_speed.update_xaxes(title="Sessions")
    fig_speed.update_yaxes(title="Typing Speed (KPS)")

    # Pause Intervals Graph
    fig_pause = px.box(df, y="pause_variance", title="Pause Variability")
    fig_pause.update_xaxes(title="Sessions")
    fig_pause.update_yaxes(title="Pause Variance")

    # Fatigue Status
    if fatigue_data["fatigue_detected"]:
        fatigue_status = "Fatigue Detected - Consider Taking a Break!"
        status_color = "danger"
    else:
        fatigue_status = "No Fatigue Detected - Keep Going!"
        status_color = "success"

    # Fatigue Recommendations
    recommendations = [html.Li(rec) for rec in fatigue_data.get("recommendations", [])]

    return fig_speed, fig_pause, fatigue_status, status_color, recommendations


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
