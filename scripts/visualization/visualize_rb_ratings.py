"""
RB Composite Rating Game-by-Game Visualizer

Interactive dashboard for viewing RB composite ratings with:
- Game-by-game trajectory tracking
- EWMA smoothed ratings with uncertainty bands
- Future performance predictions (1, 2, 3 years)
- Trend analysis and regression to mean
- Dual-threat RB analysis (rushing + receiving)

Matches QB LEAF visualizer style for consistency across positions.

Usage:
    python visualize_rb_ratings.py

    Then open browser to: http://localhost:8052
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

def load_rb_data():
    """Load RB game-by-game composite ratings data."""
    logger.info("Loading RB game-by-game composite ratings...")

    # Load game-by-game data
    rb_files = sorted(DATA_DIR.glob("rb_composite_game_by_game_*.csv"))
    if not rb_files:
        raise FileNotFoundError("No RB game-by-game file found in data/production")

    rb_data = pd.read_csv(rb_files[-1])
    logger.info(f"  Loaded {len(rb_data):,} game records from {rb_data['player_id'].nunique()} RBs")

    return rb_data

def filter_to_rbs(rb_data, min_games=10):
    """Filter to RBs with minimum games for visualization."""
    game_counts = rb_data.groupby('player_id').size()
    qualified_rbs = game_counts[game_counts >= min_games].index

    rb_data_filtered = rb_data[rb_data['player_id'].isin(qualified_rbs)].copy()

    # Get most recent rating for each RB
    rb_data_sorted = rb_data_filtered.sort_values(['player_id', 'season', 'week'])
    most_recent = rb_data_sorted.groupby('player_id').tail(1).copy()
    most_recent = most_recent.sort_values('smoothed_composite', ascending=False)

    logger.info(f"  Filtered to {len(most_recent)} RBs with {min_games}+ games")

    return rb_data_filtered, most_recent

def classify_rb_role(player_games):
    """
    Classify RB role based on usage patterns.

    Returns:
        String describing RB role (Pure Rusher, Pass-Catching Specialist, Three-Down Back)
    """
    total_rush = player_games['rush_attempts'].sum()
    total_targets = player_games['targets'].sum()
    total_touches = total_rush + total_targets

    if total_touches == 0:
        return "Unknown"

    rush_share = total_rush / total_touches
    target_share = total_targets / total_touches

    if rush_share >= 0.85:
        return "Pure Rusher"
    elif target_share >= 0.40:
        return "Pass-Catching Specialist"
    else:
        return "Three-Down Back"

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
    # RB Career Trajectory (from analyze_career_trajectories.py):
    #   Games 0-32: Improve +0.16/season (R²=0.860) - FASTER improvement than WRs
    #   Games 32-64: Improve +0.10/season (R²=0.425) - Still improving toward peak
    #   Peak at Games 64-71 (rating 0.34) - EARLIER peak than WRs
    #   Games 71+: Decline -0.10/season (89% of players decline) - Physical position

    total_games = len(player_games)

    # Generate predictions
    predictions = []
    last_season = player_games['season'].iloc[-1]

    for years in years_forward:
        # Calculate career stage adjustment based on FUTURE games played
        future_games = total_games + (years * 16)

        # Determine career stage improvement/decline rate
        if total_games < 32:
            # Early career - STRONG improvement phase (faster than WRs)
            career_adjustment = 0.16 * years
        elif total_games < 64:
            # Mid career - continued improvement toward early peak
            career_adjustment = 0.10 * years
        elif total_games < 71:
            # Peak years - maintain
            career_adjustment = 0.0 * years
        else:
            # Post-peak - gradual decline (physical position)
            career_adjustment = -0.10 * years

        # If player will cross into new career stage during prediction window, blend rates
        if total_games < 32 and future_games > 32:
            games_in_early = 32 - total_games
            games_in_mid = future_games - 32
            career_adjustment = (games_in_early / 16) * 0.16 + (games_in_mid / 16) * 0.10
        elif total_games < 64 and future_games > 64:
            games_in_mid = 64 - total_games
            games_in_peak = future_games - 64
            career_adjustment = (games_in_mid / 16) * 0.10 + (games_in_peak / 16) * 0.0
        elif total_games < 71 and future_games > 71:
            games_in_peak = 71 - total_games
            games_in_decline = future_games - 71
            career_adjustment = (games_in_peak / 16) * 0.0 + (games_in_decline / 16) * (-0.10)

        # Base prediction: current + career stage adjustment + dampened recent trend
        predicted_rating = current_rating + career_adjustment + (trend_annual * years)

        # Uncertainty increases with time
        uncertainty_multiplier = 1 + (0.4 * years)  # 40% increase per year
        predicted_uncertainty = current_uncertainty * uncertainty_multiplier

        # Add regression to mean (elite RBs tend to decline, poor RBs improve)
        # Stronger regression for RBs: 25% per year (RB production is less stable)
        regression_factor = 0.25 * years
        regression_factor = min(regression_factor, 0.6)  # Cap at 60% regression
        mean_rating = 0.0  # Average RB is 0.0 z-score
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
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=[f"{player_name} - RB Composite Rating Trajectory"]
    )

    # Historical trajectory
    fig.add_trace(
        go.Scatter(
            x=player_games.index,
            y=player_games['smoothed_composite'],
            mode='lines+markers',
            name='Historical Rating',
            line=dict(color='#9b59b6', width=2),
            marker=dict(size=6),
            hovertemplate=(
                '<b>Game %{x}</b><br>' +
                'Season: %{customdata[0]}<br>' +
                'Week: %{customdata[1]}<br>' +
                'Rating: %{y:.3f}<br>' +
                'Opponent: %{customdata[2]}<br>' +
                'Touches: %{customdata[3]}<br>' +
                'Total Yards: %{customdata[4]}<br>' +
                '<extra></extra>'
            ),
            customdata=player_games[['season', 'week', 'opponent', 'total_touches', 'total_yards']].values
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
            fillcolor='rgba(155, 89, 182, 0.2)',
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
                line=dict(color='#ff7f0e', width=2, dash='dash'),
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
                fillcolor='rgba(255, 127, 14, 0.2)',
                fill='tonexty',
                name='Prediction Confidence',
                hoverinfo='skip'
            )
        )

    # Add reference line at 0 (average RB)
    fig.add_hline(y=0, line_dash="dot", line_color="gray",
                  annotation_text="Average RB (0.0)", annotation_position="right")

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
        yaxis_title="RB Composite Rating (Z-Score)",
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
    title="RB Composite Rating Visualizer"
)

# Load data
rb_data = load_rb_data()
rb_data_filtered, rb_summary = filter_to_rbs(rb_data, min_games=10)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🏃 RB Composite Rating Visualizer", className="text-center mb-4"),
            html.P(
                "Interactive game-by-game visualization of RB composite ratings with predictions",
                className="text-center text-muted mb-4"
            )
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Select RB:", className="fw-bold"),
            dcc.Dropdown(
                id='player-dropdown',
                options=[
                    {'label': f"{row['player_name']} ({row['smoothed_composite']:.3f})",
                     'value': row['player_id']}
                    for _, row in rb_summary.iterrows()
                ],
                value=rb_summary.iloc[0]['player_id'],  # Default to top RB
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
                dbc.CardHeader(html.H5("📊 Current Rating", className="mb-0")),
                dbc.CardBody([
                    html.H2(id='current-rating', className="text-center"),
                    html.P(id='rating-interpretation', className="text-center text-muted")
                ])
            ])
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🏈 Career Stats", className="mb-0")),
                dbc.CardBody([
                    html.Div(id='career-stats')
                ])
            ])
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🎭 RB Role", className="mb-0")),
                dbc.CardBody([
                    html.Div(id='rb-role')
                ])
            ])
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🔮 Next Season Prediction", className="mb-0")),
                dbc.CardBody([
                    html.Div(id='next-season-pred')
                ])
            ])
        ], width=3)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("📈 Recent Performance (Last 10 Games)", className="mb-0")),
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
                "Metrics: targets, total_touches, total_yards, rec_yards, rush_share | ",
                "Smoothing: EWMA (span=5) | ",
                "Multi-year stability: r=0.72 (MOST STABLE POSITION)"
            ], className="text-center text-muted small")
        ])
    ])
], fluid=True, className="p-4")

@app.callback(
    [Output('current-rating', 'children'),
     Output('current-rating', 'style'),
     Output('rating-interpretation', 'children'),
     Output('career-stats', 'children'),
     Output('rb-role', 'children'),
     Output('next-season-pred', 'children'),
     Output('recent-performance', 'children'),
     Output('trajectory-plot', 'figure')],
    [Input('player-dropdown', 'value'),
     Input('prediction-years', 'value')]
)
def update_visualizations(player_id, prediction_years):
    """Update all visualizations when player or settings change."""

    # Get player data
    player_games = rb_data_filtered[rb_data_filtered['player_id'] == player_id].copy()
    player_games = player_games.sort_values(['season', 'week']).reset_index(drop=True)

    player_name = player_games['player_name'].iloc[0]
    current_rating = player_games['smoothed_composite'].iloc[-1]
    uncertainty = player_games['composite_uncertainty'].iloc[-1]

    # Current rating display
    rating_text = f"{current_rating:+.3f} ± {uncertainty:.3f}"

    # Color based on rating (z-scores)
    if current_rating >= 1.0:
        rating_color = {'color': '#2ecc71', 'font-weight': 'bold'}  # Green (elite)
        interpretation = "Elite RB (Top 15%)"
    elif current_rating >= 0.5:
        rating_color = {'color': '#3498db', 'font-weight': 'bold'}  # Blue (above avg)
        interpretation = "Above Average RB"
    elif current_rating >= -0.5:
        rating_color = {'color': '#95a5a6', 'font-weight': 'bold'}  # Gray (average)
        interpretation = "Average RB"
    else:
        rating_color = {'color': '#e74c3c', 'font-weight': 'bold'}  # Red (below avg)
        interpretation = "Below Average RB"

    # Career stats
    total_games = len(player_games)
    total_rush = int(player_games['rush_attempts'].sum())
    total_rush_yds = int(player_games['rush_yards'].sum())
    total_targets = int(player_games['targets'].sum())
    total_rec = int(player_games['receptions'].sum())
    total_rec_yds = int(player_games['rec_yards'].sum())
    total_tds = int(player_games['total_tds'].sum())
    seasons = player_games['season'].nunique()

    career_stats_content = html.Div([
        html.P(f"Games: {total_games} ({seasons} seasons)", className="mb-1"),
        html.P(f"Rush: {total_rush} att, {total_rush_yds:,} yds", className="mb-1 small"),
        html.P(f"Rec: {total_rec}/{total_targets} tgt, {total_rec_yds:,} yds, {total_tds} TD", className="mb-0 small")
    ])

    # RB role classification
    rb_role = classify_rb_role(player_games)
    total_rush = player_games['rush_attempts'].sum()
    total_targets = player_games['targets'].sum()
    total_touches = total_rush + total_targets

    if total_touches > 0:
        rush_pct = (total_rush / total_touches) * 100
        target_pct = (total_targets / total_touches) * 100

        rb_role_content = html.Div([
            html.H5(rb_role, className="text-center mb-2"),
            html.P(f"Rush: {rush_pct:.1f}%", className="mb-1 small text-center"),
            html.P(f"Receiving: {target_pct:.1f}%", className="mb-0 small text-center")
        ])
    else:
        rb_role_content = html.P("Insufficient data", className="text-muted")

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
    avg_recent_touches = recent_games['total_touches'].mean()
    avg_recent_rush = recent_games['rush_attempts'].mean()
    avg_recent_targets = recent_games['targets'].mean()
    avg_recent_yards = recent_games['total_yards'].mean()

    recent_performance_content = html.Div([
        dbc.Row([
            dbc.Col([
                html.P([
                    html.Strong("Avg Rating: "),
                    f"{avg_recent_rating:+.3f}"
                ], className="mb-1")
            ], width=3),
            dbc.Col([
                html.P([
                    html.Strong("Avg Touches: "),
                    f"{avg_recent_touches:.1f}"
                ], className="mb-1")
            ], width=2),
            dbc.Col([
                html.P([
                    html.Strong("Avg Rush: "),
                    f"{avg_recent_rush:.1f}"
                ], className="mb-1")
            ], width=2),
            dbc.Col([
                html.P([
                    html.Strong("Avg Targets: "),
                    f"{avg_recent_targets:.1f}"
                ], className="mb-1")
            ], width=2),
            dbc.Col([
                html.P([
                    html.Strong("Avg Total Yards: "),
                    f"{avg_recent_yards:.1f}"
                ], className="mb-1")
            ], width=3)
        ])
    ])

    # Create trajectory figure
    fig = create_player_trajectory_figure(player_games, predictions, player_name)

    return (rating_text, rating_color, interpretation,
            career_stats_content, rb_role_content, pred_content,
            recent_performance_content, fig)

def main():
    """Run the visualization app."""
    logger.info("=" * 80)
    logger.info("RB Composite Rating Visualizer")
    logger.info("=" * 80)
    logger.info(f"\nLoaded {len(rb_summary)} RBs with 10+ games")
    logger.info(f"Total game records: {len(rb_data):,}")
    logger.info("\nStarting web server...")
    logger.info("Open browser to: http://localhost:8052")
    logger.info("\nPress Ctrl+C to stop the server")

    app.run(debug=True, host='127.0.0.1', port=8052)

if __name__ == "__main__":
    main()
