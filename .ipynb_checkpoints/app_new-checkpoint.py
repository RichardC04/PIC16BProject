import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import datetime
import pandas as pd
import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go

# Import the model functions
from lol_win_predictor import (
    predict_win_rate_from_players,
    evaluate_team_composition,
    load_models,
    load_and_preprocess_data
)

# Define folder paths
models_folder = 'models'
data_folder = 'data'

# Load player statistics
print("Loading player data...")
try:
    # Load the player list with names and positions
    player_list_df = pd.read_csv(os.path.join(data_folder, "players_stats_s14_clean.csv"))
    
    # Load both datasets using the same preprocessing function used during training
    # This ensures consistent data cleaning
    players_df, teams_df = load_and_preprocess_data(
        data_folder=data_folder,
        players_file='players_stats.csv',
        teams_file='teams_stats.csv'
    )
    
    # Filter to only include S14 players
    s14_players = set(player_list_df['Player'].tolist())
    players_df = players_df[(players_df['Player'].isin(s14_players)) & (players_df['Season'] == 'S14')]
    
    # If no players match, use all players but warn
    if len(players_df) == 0:
        print("Warning: No S14 players found in the detailed stats. Using all available players.")
        players_df, _ = load_and_preprocess_data(
            data_folder=data_folder,
            players_file='players_stats.csv',
            teams_file='teams_stats.csv'
        )
    
    print(f"Loaded {len(players_df)} player statistics.")
    
except Exception as e:
    print(f"Error loading player data: {e}")
    import traceback
    traceback.print_exc()
    # Create a sample dataframe for testing
    players_df = pd.DataFrame({
        'Player': ['SamplePlayer1', 'SamplePlayer2', 'SamplePlayer3', 'SamplePlayer4', 'SamplePlayer5'],
        'Position': ['TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT'],
        'Team': ['Team1', 'Team1', 'Team1', 'Team1', 'Team1'],
        'KDA': [3.0, 3.5, 4.0, 4.5, 3.8],
        'Avg kills': [2.5, 2.0, 3.5, 4.0, 1.0],
        'Avg deaths': [2.0, 2.0, 1.8, 1.5, 2.5],
        'Avg assists': [5.0, 7.0, 6.0, 5.0, 12.0],
        'CSM': [8.0, 5.5, 8.5, 9.0, 1.0],
        'GPM': [400, 350, 400, 420, 250],
        'DMG%': [0.22, 0.19, 0.28, 0.30, 0.10],
        'DPM': [450, 400, 500, 550, 200],
        'Avg WPM': [0.6, 0.8, 0.5, 0.4, 1.5],
        'GD@15': [100, 150, 200, 250, 50],
        'CSD@15': [10, 5, 15, 20, 2],
        'XPD@15': [150, 200, 250, 300, 50]
    })
    print("Created sample data with 5 players")


# Load models
print("Loading prediction models...")
try:
    feature_to_win_rate_model, player_to_feature_models, key_features = load_models(models_folder)
    print(f"Models loaded successfully! Key features: {key_features}")
    models_loaded = True
except Exception as e:
    print(f"Error loading models: {e}")
    print("Will use random predictions as fallback.")
    models_loaded = False
    key_features = []

external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500&display=swap"
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Read player data from CSV and rename columns for clarity
players = (
    players_df
    .rename(columns={"Player": "name", "Position": "lane"})
    [["name", "lane"]]
    .to_dict("records")
)

lanes = ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"]

def create_team_layout():
    '''
    Create the layout for a team's player selection dropdowns
    '''
    layout = []
    for lane in lanes:
        # Create dropdown options by filtering players for the current lane
        options = [
            {"label": p["name"], "value": p["name"]}
            for p in players if p["lane"] == lane
        ]
        layout.append(
            dbc.Row([
                dbc.Col(dbc.Label(lane), width=2),
                dbc.Col(
                    dcc.Dropdown(
                        id=f"team-{lane.lower()}",
                        options=options,
                        placeholder=f"Please select {lane} player",
                        className="dropdown-custom",
                        style={
                            'color': 'black',
                            'background-color': '#f0f0f0',
                        }
                    ),
                    width=10
                )
            ], className="mb-3")
        )
    return layout

app.layout = dbc.Container([
    html.H1("League of Legends Team Win Rate Prediction", className="mt-4"),
    
    dbc.Alert(
        "Win Rate Prediction Model Loaded Successfully!" if models_loaded else 
        "Warning: Models not loaded. Using random predictions as fallback.",
        color="success" if models_loaded else "warning",
        className="mt-2 mb-4"
    ),

    dcc.Store(id="store-templates", data=[]),
    dcc.Store(id="store-history", data=[]),
    dcc.Store(id="store-analysis", data={}),
    
    #save template
    dbc.Row([
        dbc.Col([
            dbc.Label("Template Name："),
            dbc.Input(id="template-name-input", placeholder="Input Template Name", type="text")
        ], width=4),
        dbc.Col(
            dbc.Button("Save Template", id="save-template-btn", color="info", className="mt-4"),
            width=2
        )
    ], className="mb-3"),
    
    #load template
    dbc.Row([
        dbc.Col([
            dbc.Label("Load Template："),
            dcc.Dropdown(id="load-template-dropdown", placeholder="Select the Template to Load", className="dropdown-custom")
        ], width=4),
        dbc.Col(
            dbc.Button("Load Template", id="load-template-btn", color="secondary", className="mt-4"),
            width=2
        )
    ], className="mb-3"),

    #select player
    dbc.Row([
        dbc.Col([
            html.H2("Team Composition"),
            *create_team_layout()
        ], width=6)
    ]),

    #winrate
    dbc.Button("Predict Win Rate", id="calc-btn", color="primary", className="mt-3"),
    html.Div(id="result", className="mt-3", style={"fontSize": "20px"}),
    
    # Team analysis visualizations
    html.H3("Team Analysis", className="mt-4"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="team-analysis-graph")
        ], width=6),
        dbc.Col([
            html.Div(id="team-strengths-weaknesses", className="mt-3")
        ], width=6),
    ]),
    
    # Role impact comparison
    html.H3("Role Impact Analysis", className="mt-4"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="role-impact-graph")
        ], width=6),
        dbc.Col([
            html.Div(id="role-recommendations", className="mt-3")
        ], width=6),
    ]),
    
    # Suggested Improvements
    html.H3("Suggested Improvements", className="mt-4"),
    html.Div(id="suggested-improvements", className="mt-3"),

    #history
    html.H2("History", className="mt-4"),
    html.Div(id="history-div")
], fluid=True, style={
    "fontFamily": "'Roboto Mono', monospace",
    "backgroundColor": "black",
    "color": "white"
})


@app.callback(
    Output("store-templates", "data"),
    Input("save-template-btn", "n_clicks"),
    [
        State("template-name-input", "value"),
        State("team-top", "value"),
        State("team-jungle", "value"),
        State("team-mid", "value"),
        State("team-adc", "value"),
        State("team-support", "value"),
        State("store-templates", "data")
    ],
    prevent_initial_call=True
)
def save_template(n_clicks, template_name, top, jungle, mid, adc, support, store_templates):
    '''
    Save the current team composition as a template.
    If no template name is provided, return the current templates unchanged.
    Otherwise, save the selected players for the team.
    '''
    if not template_name:
        return store_templates
    team = [top, jungle, mid, adc, support]
    new_template = {"name": template_name, "team": team}
    store_templates.append(new_template)
    return store_templates

@app.callback(
    Output("load-template-dropdown", "options"),
    Input("store-templates", "data")
)
def update_template_dropdown_options(store_templates):
    '''
    Update the template dropdown options based on the saved templates.
    '''
    options = [{"label": t["name"], "value": t["name"]} for t in store_templates]
    return options

@app.callback(
    [
        Output("team-top", "value"),
        Output("team-jungle", "value"),
        Output("team-mid", "value"),
        Output("team-adc", "value"),
        Output("team-support", "value")
    ],
    Input("load-template-btn", "n_clicks"),
    [State("load-template-dropdown", "value"),
     State("store-templates", "data")],
    prevent_initial_call=True
)
def load_template(n_clicks, selected_template_name, store_templates):
    '''
    Load a saved template to update the team player selections.
    '''
    if not selected_template_name:
        return [None] * 5
    for t in store_templates:
        if t["name"] == selected_template_name:
            team = t["team"]
            return team
    return [None] * 5

def get_player_stats(player_name, lane):
    """
    Get statistics for a specific player in a specific role.
    Returns a dictionary of player statistics.
    """
    if not player_name:
        return None
    
    player_data = players_df[(players_df['Player'] == player_name) & (players_df['Position'] == lane)]
    
    if player_data.empty:
        # Return default stats if player not found
        return {
            'KDA': 3.0,
            'Avg kills': 2.5,
            'Avg deaths': 2.0,
            'Avg assists': 5.0,
            'CSM': 8.0,
            'GPM': 380,
            'DMG%': 0.22,
            'DPM': 450,
            'Avg WPM': 0.6,
            'GD@15': 50,
            'CSD@15': 5,
            'XPD@15': 100
        }
    
    # Get the first row of player data and convert to dict
    stats = player_data.iloc[0].to_dict()
    
    # Create a dictionary with the required stats
    player_stats = {
        'KDA': stats.get('KDA', 3.0),
        'Avg kills': stats.get('Avg kills', 2.5),
        'Avg deaths': stats.get('Avg deaths', 2.0),
        'Avg assists': stats.get('Avg assists', 5.0),
        'CSM': stats.get('CSM', 8.0),
        'GPM': stats.get('GPM', 380),
        'DMG%': stats.get('DMG%', 0.22),
        'DPM': stats.get('DPM', 450),
        'Avg WPM': stats.get('Avg WPM', 0.6),
        'GD@15': stats.get('GD@15', 50),
        'CSD@15': stats.get('CSD@15', 5),
        'XPD@15': stats.get('XPD@15', 100)
    }
    
    return player_stats

def create_team_composition(top, jungle, mid, adc, support):
    """
    Create a team composition dictionary from the selected players.
    """
    team_comp = {
        'TOP': get_player_stats(top, 'TOP'),
        'JUNGLE': get_player_stats(jungle, 'JUNGLE'),
        'MID': get_player_stats(mid, 'MID'),
        'ADC': get_player_stats(adc, 'ADC'),
        'SUPPORT': get_player_stats(support, 'SUPPORT')
    }
    return team_comp

@app.callback(
    [Output("result", "children"),
     Output("store-history", "data"),
     Output("store-analysis", "data"),
     Output("team-analysis-graph", "figure"),
     Output("role-impact-graph", "figure"),
     Output("team-strengths-weaknesses", "children"),
     Output("role-recommendations", "children"),
     Output("suggested-improvements", "children")],
    Input("calc-btn", "n_clicks"),
    [
        State("team-top", "value"),
        State("team-jungle", "value"),
        State("team-mid", "value"),
        State("team-adc", "value"),
        State("team-support", "value"),
        State("store-history", "data")
    ],
    prevent_initial_call=True
)
def calculate_win_rate(n_clicks, top, jungle, mid, adc, support, store_history):
    """
    Calculate the win rate for the team based on the selected players.
    Validates that all positions are filled.
    Updates and returns the result message and the history of predictions.
    """
    team_players = [top, jungle, mid, adc, support]
    
    # Validate players for all lanes
    if None in team_players:
        return ("Please select players for all roles of the team.", 
                store_history, {}, {}, {}, 
                "No data available", "No data available", "No data available")
    
    # Check for repeated players
    if len(set(team_players)) < 5:
        return ("Please do not select the same player multiple times.", 
                store_history, {}, {}, {}, 
                "No data available", "No data available", "No data available")
    
    # Create team composition
    team_comp = create_team_composition(top, jungle, mid, adc, support)
    
    # Use model for prediction if available, otherwise use random
    if models_loaded:
        try:
            # Predict win rate using the model
            win_rate, team_analysis = evaluate_team_composition(
                team_comp, player_to_feature_models, feature_to_win_rate_model, key_features
            )
            
            # Convert win rate to percentage
            win_rate_percentage = win_rate * 100
            
            # Store analysis data
            analysis_data = {'team': team_analysis}
            
            # Create analysis visualizations
            team_fig = create_team_analysis_chart(team_analysis, "Team Analysis")
            
            # Create role impact visualization
            role_impact_fig = create_role_impact_chart(team_analysis)
            
            # Create strengths and weaknesses component
            strengths_weaknesses = create_strengths_weaknesses_component(team_analysis)
            
            # Create role recommendations component
            role_recommendations = create_role_recommendations_component(team_analysis)
            
            # Create suggested improvements component
            suggested_improvements = create_suggested_improvements_component(team_analysis, team_players)
            
        except Exception as e:
            print(f"Error in model prediction: {e}")
            # Fallback to random prediction
            win_rate_percentage = np.random.randint(40, 60)
            analysis_data = {}
            team_fig = {}
            role_impact_fig = {}
            strengths_weaknesses = "Error analyzing team strengths and weaknesses."
            role_recommendations = "Error creating role recommendations."
            suggested_improvements = "Error generating suggested improvements."
    else:
        # Use random prediction as fallback
        win_rate_percentage = np.random.randint(40, 60)
        analysis_data = {}
        team_fig = {}
        role_impact_fig = {}
        strengths_weaknesses = "Model not loaded. Cannot analyze team strengths and weaknesses."
        role_recommendations = "Model not loaded. Cannot provide role recommendations."
        suggested_improvements = "Model not loaded. Cannot suggest improvements."
    
    # Round to 2 decimal places
    win_rate_percentage = round(win_rate_percentage, 2)
    
    # Store history
    store_history.append({
        "team": team_players,
        "win_rate": win_rate_percentage,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Create result text
    result_text = f"Predicted Win Rate: {win_rate_percentage}%"
    
    return (result_text, store_history, analysis_data, team_fig, role_impact_fig, 
            strengths_weaknesses, role_recommendations, suggested_improvements)

def create_team_analysis_chart(analysis, title):
    """
    Create a radar chart visualization for team analysis.
    """
    if not analysis or 'predicted_features' not in analysis:
        # Return empty figure if no analysis data
        return {}
    
    features = analysis['predicted_features']
    feature_names = list(features.keys())
    feature_values = list(features.values())
    
    # Normalize feature values to 0-1 scale for radar chart
    min_vals = {
        'K:D': 0.5, 'GPM': 1600, 'GDM': -200, 
        'Game duration in seconds': 2000, 'Kills / game': 7,
        'DPM': 1700, 'CSM': 28, 'VWPM': 0.5
    }
    
    max_vals = {
        'K:D': 1.5, 'GPM': 2000, 'GDM': 200, 
        'Game duration in seconds': 2600, 'Kills / game': 15,
        'DPM': 2200, 'CSM': 32, 'VWPM': 1.0
    }
    
    normalized_values = []
    for i, val in enumerate(feature_values):
        feature_name = feature_names[i]
        min_val = min_vals.get(feature_name, min(0, val))
        max_val = max_vals.get(feature_name, max(1, val * 1.5))
        
        # Special case: for game duration, lower is better
        if feature_name == 'Game duration in seconds':
            norm_val = 1 - ((val - min_val) / (max_val - min_val) if max_val > min_val else 0.5)
        else:
            norm_val = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        
        normalized_values.append(max(0, min(1, norm_val)))
    
    # Complete the loop by adding the first value again
    feature_names.append(feature_names[0])
    normalized_values.append(normalized_values[0])
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=feature_names,
        fill='toself',
        name="Team Stats",
        line=dict(color='lightgreen'),
        fillcolor='rgba(100, 255, 180, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=title,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_role_impact_chart(analysis):
    """
    Create a bar chart showing impact of each role.
    """
    if not analysis or 'role_scores' not in analysis:
        # Return empty figure if no analysis data
        return {}
    
    role_scores = analysis['role_scores']
    
    # Create a DataFrame for role impact data
    roles = list(role_scores.keys())
    scores = list(role_scores.values())
    
    # Add an impact category
    impact_categories = []
    for score in scores:
        if score > 15:
            impact_categories.append("High Impact")
        elif score > 10:
            impact_categories.append("Medium Impact")
        else:
            impact_categories.append("Low Impact")
    
    data = pd.DataFrame({
        'Role': roles,
        'Impact Score': scores,
        'Impact Category': impact_categories
    })
    
    # Create bar chart
    fig = px.bar(
        data,
        x='Role',
        y='Impact Score',
        color='Impact Category',
        title='Role Impact Analysis',
        color_discrete_map={
            "High Impact": "lightgreen",
            "Medium Impact": "khaki",
            "Low Impact": "lightcoral"
        }
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_strengths_weaknesses_component(analysis):
    """
    Create a component to display team strengths and weaknesses.
    """
    if not analysis or 'strengths' not in analysis or 'weaknesses' not in analysis:
        return html.Div("Not enough data to analyze team strengths and weaknesses.")
    
    strengths = analysis['strengths']
    weaknesses = analysis['weaknesses']
    
    component = html.Div([
        html.H4("Team Strengths", className="text-success"),
        html.Ul([html.Li(strength) for strength in strengths]) if strengths else html.P("No significant strengths detected."),
        
        html.H4("Team Weaknesses", className="text-danger mt-4"),
        html.Ul([html.Li(weakness) for weakness in weaknesses]) if weaknesses else html.P("No significant weaknesses detected."),
    ])
    
    return component

def create_role_recommendations_component(analysis):
    """
    Create a component to display role-specific recommendations.
    """
    if not analysis or 'recommendations' not in analysis:
        return html.Div("Not enough data to provide role-specific recommendations.")
    
    recommendations = analysis['recommendations']
    
    # Filter for role-specific recommendations
    role_recs = []
    for rec in recommendations:
        for role in ['TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT']:
            if role in rec:
                role_recs.append(rec)
    
    component = html.Div([
        html.H4("Role Recommendations"),
        html.Ul([html.Li(rec) for rec in role_recs]) if role_recs else html.P("No role-specific recommendations available.")
    ])
    
    return component

def create_suggested_improvements_component(analysis, players):
    """
    Create a component with suggested improvements for the team.
    """
    if not analysis or 'role_scores' not in analysis:
        return html.Div("Not enough data to suggest improvements.")
    
    role_scores = analysis['role_scores']
    win_rate = analysis['win_rate']
    
    # Find the weakest role(s)
    roles = list(role_scores.keys())
    scores = list(role_scores.values())
    min_score = min(scores)
    weakest_roles = [roles[i] for i, score in enumerate(scores) if score == min_score]
    
    # Generate suggested improvements
    suggestions = []
    
    # Overall win rate assessment
    if win_rate < 0.4:
        suggestions.append(html.Li("This team composition has a low predicted win rate. Consider significant changes to player selection."))
    elif win_rate < 0.5:
        suggestions.append(html.Li("This team has potential but needs improvement to achieve a win rate above 50%."))
    else:
        suggestions.append(html.Li("This team composition has good potential for success. Focus on consistency and execution."))
    
    # Role-specific suggestions
    for role in weakest_roles:
        suggestions.append(html.Li([
            f"Consider replacing or focusing on improving the {role} player (",
            html.Strong(players[lanes.index(role)]),
            ") as this role has the lowest impact score."
        ]))
    
    # Team composition suggestions based on analysis
    if 'strengths' in analysis and len(analysis['strengths']) > 0:
        suggestions.append(html.Li(f"Build your team strategy around your key strength: {analysis['strengths'][0]}"))
    
    if 'weaknesses' in analysis and len(analysis['weaknesses']) > 0:
        suggestions.append(html.Li(f"Address your primary weakness: {analysis['weaknesses'][0]}"))
    
    # Additional general suggestions
    suggestions.append(html.Li("Improve team synergy by ensuring complementary champion selections that align with player strengths."))
    suggestions.append(html.Li("Focus on early game coordination and objective control to build gold advantages."))
    
    component = html.Div([
        html.H4("Suggested Team Improvements"),
        html.Ul(suggestions)
    ])
    
    return component

# history record
@app.callback(
    Output("history-div", "children"),
    Input("store-history", "data")
)
def update_history_div(store_history):
    '''
    Update the history display area with past win rate predictions.
    '''
    if not store_history:
        return "No history"
    display_list = []
    for idx, record in enumerate(store_history, start=1):
        team_str = " / ".join(str(x) for x in record["team"])
        text = (f"#{idx} [{record['timestamp']}]\n"
                f"Team: {team_str}\n"
                f"Result: Win rate = {record['win_rate']}%")
        display_list.append(
            dbc.Card(
                dbc.CardBody(html.Pre(text)),
                className="mb-2"
            )
        )
    return display_list

if __name__ == "__main__":
    app.run(debug=True)