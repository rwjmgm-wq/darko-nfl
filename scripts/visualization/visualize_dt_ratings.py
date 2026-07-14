"""
DT (Interior Defensive Line) Composite Rating Game-by-Game Visualizer

Interactive dashboard for viewing DT composite ratings with:
- Game-by-game trajectory tracking
- EWMA smoothed ratings with uncertainty bands
- Future performance predictions (1, 2, 3 years)
- Trend analysis and regression to mean

Metrics: QB hits, pressures, impact plays, EPA against | Predictive power: r=0.637

Matches QB LEAF visualizer style for consistency across positions.

Usage:
    python visualize_dt_ratings.py

    Then open browser to: http://localhost:8055
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path("data/production")

def load_dt_data():
    """Load DT game-by-game composite ratings data."""
    logger.info("Loading DT game-by-game composite ratings...")

    # Load game-by-game data
    dt_files = sorted(DATA_DIR.glob("dt_composite_game_by_game_*.csv"))
    if not dt_files:
        raise FileNotFoundError("No DT game-by-game file found in data/production")

    dt_data = pd.read_csv(dt_files[-1])
    logger.info(f"  Loaded {len(dt_data):,} game records from {dt_data['player_id'].nunique()} DTs")

    return dt_data

def filter_to_dts(dt_data, min_games=10):
    """Filter to DTs with minimum games for visualization."""
    game_counts = dt_data.groupby('player_id').size()
    qualified_dts = game_counts[game_counts >= min_games].index

    dt_data_filtered = dt_data[dt_data['player_id'].isin(qualified_dts)].copy()

    # Get most recent rating for each DT
    dt_data_sorted = dt_data_filtered.sort_values(['player_id', 'season', 'week'])
    most_recent = dt_data_sorted.groupby('player_id').tail(1).copy()
    most_recent = most_recent.sort_values('smoothed_composite', ascending=False)

    logger.info(f"  Filtered to {len(most_recent)} DTs with {min_games}+ games")

    return dt_data_filtered, most_recent

def calculate_predictions(player_games, years_forward=[1, 2, 3]):
    """
    Calculate future rating predictions based on current trajectory.

    Args:
        player_games: Game-by-game records for player
        years_forward: List of years to predict (e.g., [1, 2, 3])

    Returns:
        DataFrame with predictions
    """
    if len(player_games) == 0:
        return pd.DataFrame()

    # Get current rating and recent trend
    recent_games = player_games.tail(10)
    current_rating = player_games['smoothed_composite'].iloc[-1]
    current_uncertainty = player_games['composite_uncertainty'].iloc[-1]

    # Calculate trend from recent games (linear regression)
    # NOTE: Trends are dampened heavily since short-term noise shouldn't predict long-term
    if len(recent_games) >= 3:
        x = np.arange(len(recent_games))
        y = recent_games['smoothed_composite'].values
        trend_per_game = np.polyfit(x, y, 1)[0]  # Slope per game

        # Dampen trend: only apply 20% of recent trend (trends fade)
        # Cap at ±0.3 per year (prevents extreme extrapolation for z-scores)
        trend_annual = np.clip(trend_per_game * 16 * 0.2, -0.3, 0.3)
    else:
        trend_annual = 0.0

    # DATA-DRIVEN career stage adjustments (based on ACTUAL game data analysis)
    # DT Career Trajectory:
    #   Games 0-48: Improve +0.065/season - Early career development
    #   Games 48-80: Improve +0.100/season - Mid-career peak improvement
    #   Games 80-87: Peak performance window
    #   Games 87+: Decline (assumed -0.15/season based on typical patterns)

    total_games = len(player_games)

    # Generate predictions
    predictions = []
    last_season = player_games['season'].iloc[-1]

    for years in years_forward:
        # Calculate career stage adjustment based on FUTURE games played
        future_games = total_games + (years * 16)

        # Determine career stage improvement/decline rate
        if total_games < 48:
            # Early career - steady improvement phase
            career_adjustment = 0.065 * years
        elif total_games < 80:
            # Mid career - strong improvement to peak
            career_adjustment = 0.100 * years
        else:
            # Post-peak - decline (limited data, using conservative estimate)
            career_adjustment = -0.15 * years

        # If player will cross into new career stage during prediction window, blend rates
        if total_games < 48 and future_games > 48:
            games_in_early = 48 - total_games
            games_in_mid = future_games - 48
            career_adjustment = (games_in_early / 16) * 0.065 + (games_in_mid / 16) * 0.100
        elif total_games < 80 and future_games > 80:
            games_in_mid = 80 - total_games
            games_in_decline = future_games - 80
            career_adjustment = (games_in_mid / 16) * 0.100 + (games_in_decline / 16) * (-0.15)

        # Base prediction: current + career stage adjustment + dampened recent trend
        predicted_rating = current_rating + career_adjustment + (trend_annual * years)

        # Uncertainty increases with time
        uncertainty_multiplier = 1 + (0.4 * years)  # 40% increase per year
        predicted_uncertainty = current_uncertainty * uncertainty_multiplier

        # Add regression to mean (elite DTs tend to decline, poor ones improve)
        # Stronger regression: 20% per year (mean reversion is real in NFL)
        regression_factor = 0.20 * years
        regression_factor = min(regression_factor, 0.5)  # Cap at 50% regression
        mean_rating = 0.0  # Average DT is 0.0 z-score
        predicted_rating = predicted_rating * (1 - regression_factor) + mean_rating * regression_factor

        # Sanity check: Cap predictions at reasonable bounds for z-scores
        # Elite max: 3.0, Poor min: -2.0
        predicted_rating = np.clip(predicted_rating, -2.0, 3.0)

        predictions.append({
            'years_forward': years,
            'predicted_season': last_season + years,
            'predicted_rating': predicted_rating,
            'predicted_uncertainty': predicted_uncertainty,
            'lower_bound': predicted_rating - 1.96 * predicted_uncertainty,
            'upper_bound': predicted_rating + 1.96 * predicted_uncertainty
        })

    return pd.DataFrame(predictions)

def create_player_trajectory_figure(player_games, predictions, player_name):
    """
    Create interactive figure showing historical trajectory and predictions.

    Args:
        player_games: Game-by-game records
        predictions: Future predictions DataFrame
        player_name: Player name for title

    Returns:
        Plotly figure
    """
    # Define color scheme for DTs (dark teal/green)
    PRIMARY_COLOR = "#16a085"  # Teal
    SECONDARY_COLOR = "#52c4a1"  # Light teal

    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=[f"{player_name} - DT Composite Rating Trajectory"]
    )

    # Historical trajectory
    fig.add_trace(
        go.Scatter(
            x=player_games.index,
            y=player_games['smoothed_composite'],
            mode='lines+markers',
            name='Historical Rating',
            line=dict(color=PRIMARY_COLOR, width=2),
            marker=dict(size=6),
            hovertemplate=(
                '<b>Game %{x}</b><br>' +
                'Season: %{customdata[0]}<br>' +
                'Week: %{customdata[1]}<br>' +
                'Rating: %{y:.3f}<br>' +
                '<extra></extra>'
            ),
            customdata=player_games[['season', 'week']].values
        )
    )

    # Uncertainty bands (historical)
    fig.add_trace(
        go.Scatter(
            x=player_games.index,
            y=player_games['smoothed_composite'] + 1.96 * player_games['composite_uncertainty'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=player_games.index,
            y=player_games['smoothed_composite'] - 1.96 * player_games['composite_uncertainty'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(22, 160, 133, 0.2)',
            fill='tonexty',
            name='95% Confidence',
            hoverinfo='skip'
        )
    )

    # Future predictions
    if len(predictions) > 0:
        last_game_idx = len(player_games) - 1
        current_rating = player_games['smoothed_composite'].iloc[-1]

        # Create prediction x-values (extend beyond historical data)
        games_per_year = 16
        pred_x = [last_game_idx] + [last_game_idx + (years * games_per_year)
                                     for years in predictions['years_forward']]
        pred_y = [current_rating] + predictions['predicted_rating'].tolist()

        # Prediction line
        fig.add_trace(
            go.Scatter(
                x=pred_x,
                y=pred_y,
                mode='lines+markers',
                name='Predicted Rating',
                line=dict(color=SECONDARY_COLOR, width=2, dash='dash'),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate=(
                    '<b>Prediction</b><br>' +
                    'Season: %{customdata[0]}<br>' +
                    'Rating: %{y:.3f}<br>' +
                    'Uncertainty: ±%{customdata[1]:.3f}<br>' +
                    '<extra></extra>'
                ),
                customdata=np.column_stack([
                    predictions['predicted_season'],
                    predictions['predicted_uncertainty']
                ])
            )
        )

        # Prediction uncertainty bands
        fig.add_trace(
            go.Scatter(
                x=pred_x,
                y=[current_rating + 1.96 * player_games['composite_uncertainty'].iloc[-1]] +
                  predictions['upper_bound'].tolist(),
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            )
        )

        fig.add_trace(
            go.Scatter(
                x=pred_x,
                y=[current_rating - 1.96 * player_games['composite_uncertainty'].iloc[-1]] +
                  predictions['lower_bound'].tolist(),
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(82, 196, 161, 0.2)',
                fill='tonexty',
                name='Prediction Confidence',
                hoverinfo='skip'
            )
        )

    # Add reference line at 0 (average DT)
    fig.add_hline(y=0, line_dash="dot", line_color="gray",
                  annotation_text="Average DT (0.0)", annotation_position="right")

    # Add elite/poor reference lines
    fig.add_hline(y=1.0, line_dash="dot", line_color="green", opacity=0.5,
                  annotation_text="Elite (+1.0)", annotation_position="right")
    fig.add_hline(y=-1.0, line_dash="dot", line_color="red", opacity=0.5,
                  annotation_text="Below Avg (-1.0)", annotation_position="right")

    # Layout
    fig.update_layout(
        height=600,
        hovermode='x unified',
        xaxis_title="Game Number (Career)",
        yaxis_title="DT Composite Rating (Z-Score)",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="DT Composite Rating Visualizer"
)

# Load data
dt_data = load_dt_data()
dt_data_filtered, dt_summary = filter_to_dts(dt_data, min_games=10)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("DT (Defensive Line) Composite Rating Visualizer", className="text-center mb-4"),
            html.P(
                "Interactive game-by-game visualization of DT composite ratings with predictions | Metrics: QB hits, pressures, impact plays, EPA against | Predictive power: r=0.637",
                className="text-center text-muted mb-4"
            )
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Select DT:", className="fw-bold"),
            dcc.Dropdown(
                id='player-dropdown',
                options=[
                    {'label': f"{row['player_name']} ({row['smoothed_composite']:.3f})",
                     'value': row['player_id']}
                    for _, row in dt_summary.iterrows()
                ],
                value=dt_summary.iloc[0]['player_id'],  # Default to top DT
                clearable=False,
                className="mb-3"
            )
        ], width=6),

        dbc.Col([
            html.Label("Prediction Years:", className="fw-bold"),
            dcc.Checklist(
                id='prediction-years',
                options=[
                    {'label': ' 1 Year', 'value': 1},
                    {'label': ' 2 Years', 'value': 2},
                    {'label': ' 3 Years', 'value': 3}
                ],
                value=[1, 2, 3],
                inline=True,
                className="mb-3"
            )
        ], width=6)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Current Rating", className="mb-0")),
                dbc.CardBody([
                    html.H2(id='current-rating', className="text-center"),
                    html.P(id='rating-interpretation', className="text-center text-muted")
                ])
            ])
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Career Stats", className="mb-0")),
                dbc.CardBody([
                    html.Div(id='career-stats')
                ])
            ])
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Next Season Prediction", className="mb-0")),
                dbc.CardBody([
                    html.Div(id='next-season-pred')
                ])
            ])
        ], width=4)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Recent Performance (Last 10 Games)", className="mb-0")),
                dbc.CardBody([
                    html.Div(id='recent-performance')
                ])
            ])
        ], width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Loading(
                dcc.Graph(id='trajectory-plot', config={'displayModeBar': True})
            )
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P([
                "Data: nflfastR (2020-2024) | ",
                "Metrics: QB hits, pressures, impact plays, EPA against | ",
                "Smoothing: EWMA (span=8) | ",
                "Predictive power: r=0.637"
            ], className="text-center text-muted small")
        ])
    ])
], fluid=True, className="p-4")

@app.callback(
    [Output('current-rating', 'children'),
     Output('current-rating', 'style'),
     Output('rating-interpretation', 'children'),
     Output('career-stats', 'children'),
     Output('next-season-pred', 'children'),
     Output('recent-performance', 'children'),
     Output('trajectory-plot', 'figure')],
    [Input('player-dropdown', 'value'),
     Input('prediction-years', 'value')]
)
def update_visualizations(player_id, prediction_years):
    """Update all visualizations when player or settings change."""

    # Get player data
    player_games = dt_data_filtered[dt_data_filtered['player_id'] == player_id].copy()
    player_games = player_games.sort_values(['season', 'week']).reset_index(drop=True)

    player_name = player_games['player_name'].iloc[0]
    current_rating = player_games['smoothed_composite'].iloc[-1]
    uncertainty = player_games['composite_uncertainty'].iloc[-1]

    # Current rating display
    rating_text = f"{current_rating:+.3f} ± {uncertainty:.3f}"

    # Color based on rating (z-scores)
    if current_rating >= 1.0:
        rating_color = {'color': '#27ae60', 'font-weight': 'bold'}  # Green (elite)
        interpretation = "Elite DT (Top 15%)"
    elif current_rating >= 0.5:
        rating_color = {'color': '#16a085', 'font-weight': 'bold'}  # Teal (above avg)
        interpretation = "Above Average DT"
    elif current_rating >= -0.5:
        rating_color = {'color': '#95a5a6', 'font-weight': 'bold'}  # Gray (average)
        interpretation = "Average DT"
    else:
        rating_color = {'color': '#e74c3c', 'font-weight': 'bold'}  # Red (below avg)
        interpretation = "Below Average DT"

    # Career stats
    total_games = len(player_games)
    seasons = player_games['season'].nunique()

    # DT-specific stats
    total_sacks = player_games['sacks'].sum()
    total_pressures = player_games['pressures'].sum()
    total_tackles = player_games['total_tackles'].sum()

    career_stats_content = html.Div([
        html.P(f"Games: {total_games} ({seasons} seasons)", className="mb-1"),
        html.P(f"Sacks: {total_sacks:.1f} | Pressures: {int(total_pressures)}", className="mb-1"),
        html.P(f"Tackles: {int(total_tackles)} | Position: DL", className="mb-0 small")
    ])

    # Calculate predictions
    if prediction_years and len(prediction_years) > 0:
        predictions = calculate_predictions(player_games, sorted(prediction_years))

        # Next season prediction
        if len(predictions) > 0:
            next_pred = predictions.iloc[0]
            pred_content = html.Div([
                html.H4(f"{next_pred['predicted_rating']:+.3f}",
                       className="text-center mb-1"),
                html.P(f"±{next_pred['predicted_uncertainty']:.3f}",
                      className="text-center text-muted mb-1"),
                html.P(f"({int(next_pred['predicted_season'])} Season)",
                      className="text-center text-muted small mb-0")
            ])
        else:
            pred_content = html.P("No prediction available", className="text-muted")
    else:
        predictions = pd.DataFrame()
        pred_content = html.P("Predictions disabled", className="text-muted")

    # Recent performance (last 10 games)
    recent_games = player_games.tail(10)
    avg_recent_rating = recent_games['smoothed_composite'].mean()

    recent_performance_content = html.Div([
        dbc.Row([
            dbc.Col([
                html.P([
                    html.Strong("Avg Rating: "),
                    f"{avg_recent_rating:+.3f}"
                ], className="mb-1")
            ], width=12)
        ])
    ])

    # Create trajectory figure
    fig = create_player_trajectory_figure(player_games, predictions, player_name)

    return (rating_text, rating_color, interpretation,
            career_stats_content, pred_content, recent_performance_content, fig)

def main():
    """Run the visualization app."""
    logger.info("=" * 80)
    logger.info("DT (Defensive Line) Composite Rating Visualizer")
    logger.info("=" * 80)
    logger.info(f"\nLoaded {len(dt_summary)} DTs with 10+ games")
    logger.info(f"Total game records: {len(dt_data):,}")
    logger.info("\nStarting web server...")
    logger.info("Open browser to: http://localhost:8055")
    logger.info("\nPress Ctrl+C to stop the server")

    app.run(debug=True, host='127.0.0.1', port=8055)

if __name__ == "__main__":
    main()
