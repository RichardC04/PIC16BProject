import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import random
import datetime
import pandas as pd

external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500&display=swap"
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

players_df = pd.read_csv("players_stats_s14_clean.csv")

# Read player data from CSV and rename columns for clarity
players = (
    players_df
    .rename(columns={"Player": "name", "Position": "lane"})
    [["name", "lane"]]
    .to_dict("records")
)


lanes = ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"]

def create_team_layout(team_id):
    '''
    Create the layout for a team's player selection dropdowns
    '''
    layout = []
    for lane in lanes:# Create dropdown options by filtering players for the current lane
        options = [
            {"label": p["name"], "value": p["name"]}
            for p in players if p["lane"] == lane
        ]
        layout.append(
            dbc.Row([
                dbc.Col(dbc.Label(lane), width=2),
                dbc.Col(
                    dcc.Dropdown(
                        id=f"{team_id}-{lane.lower()}",
                        options=options,
                        placeholder=f"Please select {lane} player",
                        className="dropdown-custom"
                    ),
                    width=10
                )
            ], className="mb-3")
        )
    return layout

app.layout = dbc.Container([
    html.H1("League of Legends Free Team Formation Win Rate Analysis", className="mt-4"),

    dcc.Store(id="store-templates", data=[]),
    dcc.Store(id="store-history", data=[]),
    #save template
    dbc.Row([
        dbc.Col([
            dbc.Label("Template Name："),
            dbc.Input(id="template-name-input", placeholder="Imput Template Name", type="text")
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
            html.H2("Team 1"),
            *create_team_layout("team1")
        ], width=6),
        dbc.Col([
            html.H2("Team 2"),
            *create_team_layout("team2")
        ], width=6)
    ]),

    #winrate
    dbc.Button("Win Rate", id="calc-btn", color="primary", className="mt-3"),
    html.Div(id="result", className="mt-3", style={"fontSize": "20px"}),

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
        State("team1-top", "value"),
        State("team1-jungle", "value"),
        State("team1-mid", "value"),
        State("team1-adc", "value"),
        State("team1-support", "value"),
        State("team2-top", "value"),
        State("team2-jungle", "value"),
        State("team2-mid", "value"),
        State("team2-adc", "value"),
        State("team2-support", "value"),
        State("store-templates", "data")
    ],
    prevent_initial_call=True
)
def save_template(n_clicks, template_name, t1_top, t1_jgl, t1_mid, t1_adc, t1_sup,
                  t2_top, t2_jgl, t2_mid, t2_adc, t2_sup, store_templates):
    '''
    Save the current team composition as a template.
    If no template name is provided, return the current templates unchanged.
    Otherwise, save the selected players for both teams.
    '''
    if not template_name:
        return store_templates
    team1 = [t1_top, t1_jgl, t1_mid, t1_adc, t1_sup]
    team2 = [t2_top, t2_jgl, t2_mid, t2_adc, t2_sup]
    new_template = {"name": template_name, "team1": team1, "team2": team2}
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
        Output("team1-top", "value"),
        Output("team1-jungle", "value"),
        Output("team1-mid", "value"),
        Output("team1-adc", "value"),
        Output("team1-support", "value"),
        Output("team2-top", "value"),
        Output("team2-jungle", "value"),
        Output("team2-mid", "value"),
        Output("team2-adc", "value"),
        Output("team2-support", "value")
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
        return [None] * 10
    for t in store_templates:
        if t["name"] == selected_template_name:
            team1 = t["team1"]
            team2 = t["team2"]
            return team1 + team2
    return [None] * 10

@app.callback(
    [Output("result", "children"),
     Output("store-history", "data")],
    Input("calc-btn", "n_clicks"),
    [
        State("team1-top", "value"),
        State("team1-jungle", "value"),
        State("team1-mid", "value"),
        State("team1-adc", "value"),
        State("team1-support", "value"),
        State("team2-top", "value"),
        State("team2-jungle", "value"),
        State("team2-mid", "value"),
        State("team2-adc", "value"),
        State("team2-support", "value"),
        State("store-history", "data")
    ],
    prevent_initial_call=True
)
def calculate_win_rate(n_clicks, t1_top, t1_jgl, t1_mid, t1_adc, t1_sup,
                       t2_top, t2_jgl, t2_mid, t2_adc, t2_sup, store_history):
    """
    Calculate the win rate for both teams based on the selected players.
    Validates that all positions are filled and that no duplicate players are selected.
    Updates and returns the result message and the history of predictions.
    """
    team1 = [t1_top, t1_jgl, t1_mid, t1_adc, t1_sup]
    team2 = [t2_top, t2_jgl, t2_mid, t2_adc, t2_sup]
    # players for all lanes
    if None in team1 or None in team2:
        return ("Please select players for all branches of both teams.", store_history)
    # repeated player
    if len(set(team1 + team2)) < 10:
        return ("Please do not select the repeated player", store_history)
    win_rate_team1 = random.randint(0, 100)
    win_rate_team2 = 100 - win_rate_team1
    store_history.append({
        "team1": team1,
        "team2": team2,
        "win_rate_team1": win_rate_team1,
        "win_rate_team2": win_rate_team2,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    result_text = f"Team 1 win rate：{win_rate_team1}% | Team 2 win rate：{win_rate_team2}%"
    return (result_text, store_history)

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
        team1_str = " / ".join(str(x) for x in record["team1"])
        team2_str = " / ".join(str(x) for x in record["team2"])
        text = (f"#{idx} [{record['timestamp']}]\n"
                f"Team 1: {team1_str}\n"
                f"Team 2: {team2_str}\n"
                f"Result: Team 1 win rate={record['win_rate_team1']}%, Team 2 win rate={record['win_rate_team2']}%")
        display_list.append(
            dbc.Card(
                dbc.CardBody(html.Pre(text)),
                className="mb-2"
            )
        )
    return display_list

if __name__ == "__main__":
    app.run_server(debug=True)
