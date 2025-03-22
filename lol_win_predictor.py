import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Data Loading and Preprocessing

def load_and_preprocess_data(data_folder='data', players_file='players_stats.csv', teams_file='teams_stats.csv'):
    """
    Load and preprocess the player and team data from the data folder
    
    Args:
        data_folder: Folder containing the data files
        players_file: Name of the player statistics CSV file
        teams_file: Name of the team statistics CSV file
        
    Returns:
        df_players: Preprocessed player statistics DataFrame
        df_teams: Preprocessed team statistics DataFrame
    """
    
    # Create full paths to the data files
    players_path = os.path.join(data_folder, players_file)
    teams_path = os.path.join(data_folder, teams_file)
    rosters_path = os.path.join(data_folder, 'rosters.csv')
    
    print(f"Loading data from {players_path} and {teams_path}...")
    
    # Check if files exist
    if not os.path.exists(players_path):
        raise FileNotFoundError(f"Player stats file not found at {players_path}")
    if not os.path.exists(teams_path):
        raise FileNotFoundError(f"Team stats file not found at {teams_path}")
    
    # Load datasets
    df_players = pd.read_csv(players_path)
    df_teams = pd.read_csv(teams_path)
    
    print("Preprocessing data...")
    # Check if Team column exists
    if 'Team' not in df_players.columns:
        print("Warning: 'Team' column not found in player data.")
        # If we have a rosters.csv file that contains Team info, try to use it
        if os.path.exists(rosters_path):
            print(f"Found rosters file at {rosters_path}. Trying to merge team information...")
            rosters = pd.read_csv(rosters_path)
            # Merge player data with rosters to get team information
            df_players = pd.merge(df_players, rosters[['Player', 'Season', 'Team']], 
                                 on=['Player', 'Season'], how='left')
        else:
            print("No rosters file found. Creating dummy 'Team' column from player names.")
            # Create a dummy Team column as a fallback
            df_players['Team'] = "Team_" + df_players['Player'].str.split().str[0]
    
    # Clean percentage columns in players data
    percentage_columns = ["Win rate", "KP%", "DMG%", "FB %", "FB Victim"]
    for col in percentage_columns:
        if col in df_players.columns:
            df_players[col] = df_players[col].astype(str).str.rstrip('%').astype(float) / 100
    
    # Clean object columns like KDA and Solo Kills
    object_columns = ["KDA", "Solo Kills"]
    for col in object_columns:
        if col in df_players.columns:
            df_players[col] = df_players[col].replace('-', 0).astype(float)
    
    # Clean percentage columns in teams data
    team_percentage_cols = ["Win rate", "FOS%", "VGPG", "ATAKHAN%", "PPG"]
    for col in team_percentage_cols:
        if col in df_teams.columns:
            df_teams[col] = df_teams[col].astype(str).str.replace('%', '').replace('-', '0').astype(float) / 100
    
    # Parse game duration
    def parse_duration(s):
        if isinstance(s, str) and ":" in s:
            mins, secs = s.split(":")
            return int(mins) * 60 + int(secs)
        return float(s) if s and s != '-' else 0
    
    if "Game duration" in df_teams.columns:
        df_teams["Game duration in seconds"] = df_teams["Game duration"].apply(parse_duration)
    
    # Filter to only include seasons S6 and S7 as shown in the provided code
    df_teams = df_teams[df_teams['Season'].isin(['S6', 'S7'])]
    
    # Drop non-essential columns or columns with too many missing values
    if "Country" in df_players.columns:
        df_players = df_players.drop(columns=["Country"])
    
    if "Region" in df_teams.columns:
        df_teams = df_teams.drop(columns=["Region", "FOS%", "VGPG", "ATAKHAN%", "PPG"])
    
    # Fill missing values
    df_players = df_players.fillna(0)
    df_teams = df_teams.fillna(0)
    
    return df_players, df_teams


def aggregate_players_by_team(df_players):
    """
    Aggregate player statistics by team and season, weighted by games played
    
    Args:
        df_players: Preprocessed player statistics DataFrame
        
    Returns:
        df_aggregated: DataFrame with player statistics aggregated by team and season
    """
    # Check if 'Team' column exists
    if 'Team' not in df_players.columns:
        print("Warning: 'Team' column not found in player data. Please ensure players are associated with teams.")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Group players by team and season
    player_roles = df_players['Position'].unique()
    
    # Create a dictionary to store aggregated team data
    team_data = {}
    
    # Process each player
    for _, player in df_players.iterrows():
        if pd.isna(player['Team']):
            continue
            
        team_key = (player['Team'], player['Season'])
        
        if team_key not in team_data:
            team_data[team_key] = {
                'Team': player['Team'],
                'Season': player['Season']
            }
            # Initialize stats for each role
            for role in player_roles:
                for stat in ['KDA', 'Avg kills', 'Avg deaths', 'Avg assists', 'CSM', 'GPM', 
                             'DMG%', 'DPM', 'Avg WPM', 'GD@15', 'CSD@15', 'XPD@15']:
                    if stat in df_players.columns:
                        team_data[team_key][f'{role}_{stat}'] = 0
                        team_data[team_key][f'{role}_Games'] = 0
        
        # Get the player's role
        role = player['Position']
        games = player['Games'] if not pd.isna(player['Games']) and player['Games'] > 0 else 1
        
        # Update games count for this role
        team_data[team_key][f'{role}_Games'] += games
        
        # Update all stats for this role with weighted values
        for stat in ['KDA', 'Avg kills', 'Avg deaths', 'Avg assists', 'CSM', 'GPM', 
                     'DMG%', 'DPM', 'Avg WPM', 'GD@15', 'CSD@15', 'XPD@15']:
            if stat in df_players.columns and not pd.isna(player[stat]):
                team_data[team_key][f'{role}_{stat}'] += player[stat] * games
    
    # Calculate weighted averages
    for team_key in team_data:
        for role in player_roles:
            role_games = team_data[team_key][f'{role}_Games']
            if role_games > 0:
                for stat in ['KDA', 'Avg kills', 'Avg deaths', 'Avg assists', 'CSM', 'GPM', 
                             'DMG%', 'DPM', 'Avg WPM', 'GD@15', 'CSD@15', 'XPD@15']:
                    if f'{role}_{stat}' in team_data[team_key]:
                        team_data[team_key][f'{role}_{stat}'] /= role_games
    
    # Convert to DataFrame
    df_aggregated = pd.DataFrame(list(team_data.values()))
    return df_aggregated


def identify_key_features(df_teams, n_features=10):
    """
    Identify the most important team features for predicting win rate
    
    Args:
        df_teams: Preprocessed team statistics DataFrame
        n_features: Number of top features to select
        
    Returns:
        selected_features: List of selected feature names
    """
    # Prepare features and target
    X = df_teams.drop(columns=['Team', 'Season', 'Win rate'])
    y = df_teams['Win rate']
    
    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Use SelectKBest for feature selection
    selector = SelectKBest(score_func=f_regression, k=n_features)
    selector.fit(X, y)
    
    # Get selected feature names
    feature_indices = selector.get_support(indices=True)
    selected_features = X.columns[feature_indices].tolist()
    
    # Print feature scores
    scores = selector.scores_
    feature_scores = dict(zip(X.columns, scores))
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("Top features for predicting win rate:")
    for feature, score in sorted_features[:n_features]:
        print(f"{feature}: {score:.4f}")
    
    return selected_features


def train_win_rate_model(df_teams, key_features):
    """
    Train models to predict win rate from key team features
    
    Args:
        df_teams: Preprocessed team statistics DataFrame
        key_features: List of feature names to use for prediction
        
    Returns:
        Dictionary of trained models and their performance metrics
    """
    # Prepare features and target
    X = df_teams[key_features]
    y = df_teams['Win rate']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_r2 = r2_score(y_test, lr_pred)
    
    print("\n1. Linear Regression Results:")
    print(f"RMSE: {lr_rmse:.4f}")
    print(f"R²: {lr_r2:.4f}")
    
    # 2. Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)
    
    print("\n2. Random Forest Results:")
    print(f"RMSE: {rf_rmse:.4f}")
    print(f"R²: {rf_r2:.4f}")
    
    # Feature importance from Random Forest
    feature_importances = dict(zip(key_features, rf_model.feature_importances_))
    sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    
    print("\nRandom Forest Feature Importance:")
    for feature, importance in sorted_importances:
        print(f"{feature}: {importance:.4f}")
    
    # 3. Neural Network
    nn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # Early stopping and model checkpoint
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )
    
    history = nn_model.fit(
        X_train_scaled, y_train,
        epochs=200,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    nn_pred = nn_model.predict(X_test_scaled).flatten()
    nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))
    nn_r2 = r2_score(y_test, nn_pred)
    
    print("\n3. Neural Network Results:")
    print(f"RMSE: {nn_rmse:.4f}")
    print(f"R²: {nn_r2:.4f}")
    
    # Find the best model
    best_r2 = max(lr_r2, rf_r2, nn_r2)
    if best_r2 == lr_r2:
        best_model = lr_model
        model_name = "Linear Regression"
    elif best_r2 == rf_r2:
        best_model = rf_model
        model_name = "Random Forest"
    else:
        best_model = nn_model
        model_name = "Neural Network"
    
    print(f"\nBest model: {model_name} with R² = {best_r2:.4f}")
    
    return {
        'linear_regression': {'model': lr_model, 'scaler': scaler, 'rmse': lr_rmse, 'r2': lr_r2},
        'random_forest': {'model': rf_model, 'scaler': scaler, 'rmse': rf_rmse, 'r2': rf_r2},
        'neural_network': {'model': nn_model, 'scaler': scaler, 'rmse': nn_rmse, 'r2': nn_r2},
        'best_model': {'model': best_model, 'scaler': scaler, 'name': model_name, 'r2': best_r2}
    }


def train_player_to_team_features(df_aggregated, df_teams, key_features):
    """
    Train models to predict key team features from player statistics
    
    Args:
        df_aggregated: DataFrame with player statistics aggregated by team
        df_teams: Preprocessed team statistics DataFrame
        key_features: List of key team features to predict
        
    Returns:
        Dictionary of trained models for each key feature
    """
    # Check if required columns exist
    if df_aggregated.empty or 'Team' not in df_aggregated.columns or 'Season' not in df_aggregated.columns:
        print("Error: Aggregated player data is missing required columns 'Team' or 'Season'")
        return {}
        
    # Merge aggregated player data with team data
    try:
        merged_df = pd.merge(
            df_aggregated, 
            df_teams[['Team', 'Season'] + key_features], 
            on=['Team', 'Season']
        )
    except KeyError as e:
        print(f"Error during merge: {e}")
        print("Columns in df_aggregated:", df_aggregated.columns.tolist())
        print("Columns in df_teams:", df_teams.columns.tolist())
        return {}
    
    # Check if merge produced any rows
    if merged_df.empty:
        print("Error: No matching teams found between player data and team data")
        return {}
    
    # Prepare player features (exclude Team, Season and the key_features we're trying to predict)
    player_features = [col for col in df_aggregated.columns 
                       if col not in ['Team', 'Season'] + key_features]
    
    # Dictionary to store models for each key feature
    models = {}
    
    # Train a model for each key feature
    for feature in key_features:
        print(f"\n=== Training models to predict {feature} ===")
        
        # Prepare data
        X = merged_df[player_features]
        y = merged_df[feature]
        
        # Handle any NaN values
        X = X.fillna(0)
        y = y.fillna(0)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        lr_r2 = r2_score(y_test, lr_pred)
        
        print(f"1. Linear Regression - RMSE: {lr_rmse:.4f}, R²: {lr_r2:.4f}")
        
        # 2. Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_r2 = r2_score(y_test, rf_pred)
        
        print(f"2. Random Forest - RMSE: {rf_rmse:.4f}, R²: {rf_r2:.4f}")
        
        # 3. Neural Network
        nn_model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        nn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        )
        
        history = nn_model.fit(
            X_train_scaled, y_train,
            epochs=150,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        nn_pred = nn_model.predict(X_test_scaled).flatten()
        nn_rmse = np.sqrt(mean_squared_error(y_test, nn_pred))
        nn_r2 = r2_score(y_test, nn_pred)
        
        print(f"3. Neural Network - RMSE: {nn_rmse:.4f}, R²: {nn_r2:.4f}")
        
        # Find the best model
        best_r2 = max(lr_r2, rf_r2, nn_r2)
        if best_r2 == lr_r2:
            best_model = lr_model
            model_name = "Linear Regression"
        elif best_r2 == rf_r2:
            best_model = rf_model
            model_name = "Random Forest"
        else:
            best_model = nn_model
            model_name = "Neural Network"
        
        print(f"Best model for {feature}: {model_name} with R² = {best_r2:.4f}")
        
        models[feature] = {
            'linear_regression': {'model': lr_model, 'scaler': scaler, 'rmse': lr_rmse, 'r2': lr_r2},
            'random_forest': {'model': rf_model, 'scaler': scaler, 'rmse': rf_rmse, 'r2': rf_r2},
            'neural_network': {'model': nn_model, 'scaler': scaler, 'rmse': nn_rmse, 'r2': nn_r2},
            'best_model': {'model': best_model, 'scaler': scaler, 'name': model_name, 'r2': best_r2},
            'feature_columns': player_features
        }
    
    return models


def predict_win_rate_from_players(player_stats, player_to_feature_models, feature_to_win_rate_model, key_features):
    """
    Predicts team win rate from player statistics
    
    Args:
        player_stats: Dictionary with player stats for each role (TOP, JUNGLE, MID, ADC, SUPPORT)
        player_to_feature_models: Models to predict team features from player stats
        feature_to_win_rate_model: Model to predict win rate from team features
        key_features: List of key team features used for prediction
        
    Returns:
        Predicted win rate (0-1)
    """
    # Prepare input data from player stats
    player_data = {}
    
    # Process each role's stats
    for role, stats in player_stats.items():
        for stat_name, value in stats.items():
            feature_name = f"{role}_{stat_name}"
            player_data[feature_name] = value
    
    # Create DataFrame with the right columns
    required_columns = player_to_feature_models[key_features[0]]['feature_columns']
    player_df = pd.DataFrame({col: [player_data.get(col, 0)] for col in required_columns})
    
    # Predict each team feature
    predicted_features = {}
    for feature in key_features:
        # Get the best model for this feature
        model_info = player_to_feature_models[feature]['best_model']
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Scale the input
        player_scaled = scaler.transform(player_df)
        
        # Predict the feature
        if model_info['name'] == 'Neural Network':
            feature_value = model.predict(player_scaled).flatten()[0]
        else:
            feature_value = model.predict(player_scaled)[0]
            
        predicted_features[feature] = feature_value
    
    # Create DataFrame for the feature to win rate model
    feature_df = pd.DataFrame([predicted_features])
    
    # Predict win rate using the best model
    win_rate_model_info = feature_to_win_rate_model['best_model']
    win_rate_model = win_rate_model_info['model']
    win_rate_scaler = win_rate_model_info['scaler']
    
    # Scale the features
    feature_scaled = win_rate_scaler.transform(feature_df)
    
    # Predict win rate
    if win_rate_model_info['name'] == 'Neural Network':
        win_rate = win_rate_model.predict(feature_scaled).flatten()[0]
    else:
        win_rate = win_rate_model.predict(feature_scaled)[0]
    
    return win_rate, predicted_features


def evaluate_team_composition(player_stats, player_to_feature_models, feature_to_win_rate_model, key_features):
    """
    Evaluate a team composition and provide analysis
    
    Args:
        player_stats: Dictionary with player stats for each role
        player_to_feature_models: Models to predict team features from player stats
        feature_to_win_rate_model: Model to predict win rate from team features
        key_features: List of key team features used for prediction
        
    Returns:
        win_rate: Predicted win rate
        analysis: Dictionary with analysis results
    """
    win_rate, predicted_features = predict_win_rate_from_players(
        player_stats, 
        player_to_feature_models, 
        feature_to_win_rate_model, 
        key_features
    )
    
    # Format win rate for display
    win_rate_percent = win_rate * 100
    
    # Calculate role impact scores
    role_scores = {}
    for role, stats in player_stats.items():
        kda = stats.get('KDA', 0)
        dmg_percent = stats.get('DMG%', 0)
        gold_diff = stats.get('GD@15', 0)
        
        role_score = kda * 0.4 + dmg_percent * 30 + (gold_diff / 500) * 0.3
        role_scores[role] = role_score
    
    # Analyze team strengths based on predicted features
    # Define some thresholds for strong/weak values
    feature_thresholds = {
        'K:D': {'strong': 1.2, 'weak': 0.8},
        'GPM': {'strong': 1800, 'weak': 1700},
        'GDM': {'strong': 100, 'weak': -100},
        'Game duration in seconds': {'strong': 2100, 'weak': 2400},  # Shorter games are generally better
        'Kills / game': {'strong': 12, 'weak': 8},
        'DPM': {'strong': 2000, 'weak': 1800}
    }
    
    strengths = []
    weaknesses = []
    
    for feature, value in predicted_features.items():
        if feature in feature_thresholds:
            thresholds = feature_thresholds[feature]
            
            # For game duration, lower is better
            if feature == 'Game duration in seconds':
                if value < thresholds['strong']:
                    strengths.append(f"{feature}: {value:.2f} (Good - Fast Games)")
                elif value > thresholds['weak']:
                    weaknesses.append(f"{feature}: {value:.2f} (Slow Games)")
            else:
                if value > thresholds['strong']:
                    strengths.append(f"{feature}: {value:.2f}")
                elif value < thresholds['weak']:
                    weaknesses.append(f"{feature}: {value:.2f}")
    
    # Generate recommendations
    recommendations = []
    
    if win_rate < 0.4:
        recommendations.append("This team composition has a low predicted win rate.")
        recommendations.append("Consider adjusting player roles or selecting players with better synergy.")
    elif win_rate < 0.5:
        recommendations.append("This team has potential but needs improvement.")
        recommendations.append("Focus on early game performance to increase gold and experience advantages.")
    else:
        recommendations.append("This team composition has good potential for success.")
        recommendations.append("Maintain consistent performance to achieve the predicted win rate.")
    
    # Role-specific recommendations
    for role, stats in player_stats.items():
        kda = stats.get('KDA', 0)
        dmg_percent = stats.get('DMG%', 0)
        
        if role == 'TOP' and stats.get('GD@15', 0) < 0:
            recommendations.append(f"The {role} player should focus on lane dominance and early-game CSing.")
        elif role == 'JUNGLE' and stats.get('Avg WPM', 0) < 0.6:
            recommendations.append(f"The {role} player should improve vision control for better map awareness.")
        elif role == 'MID' and dmg_percent < 0.25:
            recommendations.append(f"The {role} player should aim for more damage output in team fights.")
        elif role == 'ADC' and kda < 3.5:
            recommendations.append(f"The {role} player should focus on positioning to improve survival in fights.")
        elif role == 'SUPPORT' and stats.get('Avg assists', 0) < 8:
            recommendations.append(f"The {role} player should improve team fight participation for more assists.")
    
    analysis = {
        'win_rate': win_rate,
        'win_rate_percent': win_rate_percent,
        'predicted_features': predicted_features,
        'role_scores': role_scores,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'recommendations': recommendations
    }
    
    return win_rate, analysis


def save_models(feature_to_win_rate_model, player_to_feature_models, key_features, folder='models', filename='lol_models.pkl'):
    """
    Save trained models to a file in a dedicated folder
    
    Args:
        feature_to_win_rate_model: Dictionary of trained models for win rate prediction
        player_to_feature_models: Dictionary of trained models for team feature prediction
        key_features: List of key features used in the models
        folder: Folder path to save models
        filename: Output filename
        
    Returns:
        None
    """
    import os
    
    # Create models folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    
    model_data = {
        'feature_to_win_rate_model': feature_to_win_rate_model,
        'player_to_feature_models': player_to_feature_models,
        'key_features': key_features
    }
    
    def sanitize_filename(name):
        """Sanitize a string to be used as a filename"""
        # Replace spaces, slashes and other problematic characters
        return name.replace(" ", "_").replace("/", "_").replace(":", "_").replace("\\", "_").replace("%", "pct")
    
    # For neural network models, we need to save them separately
    for model_type, model_dict in feature_to_win_rate_model.items():
        if model_type == 'neural_network' and 'model' in model_dict and model_dict['model'] is not None:
            model = model_dict['model']
            model_dict['model'] = None  # Temporarily set to None for serialization
            model_path = os.path.join(folder, f'win_rate_{model_type}_model.keras')
            model.save(model_path)
            model_dict['model_path'] = model_path
    
    for feature, feature_models in player_to_feature_models.items():
        for model_type, model_dict in feature_models.items():
            if model_type == 'neural_network' and 'model' in model_dict and model_dict['model'] is not None:
                model = model_dict['model']
                model_dict['model'] = None  # Temporarily set to None for serialization
                model_path = os.path.join(folder, f'{sanitize_filename(feature)}_{model_type}_model.keras')
                model.save(model_path)
                model_dict['model_path'] = model_path
    
    # Save the model data
    full_path = os.path.join(folder, filename)
    with open(full_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Models saved to {full_path}")


def load_models(folder='models', filename='lol_models.pkl'):
    """
    Load trained models from a file in a dedicated folder
    
    Args:
        folder: Folder path where models are stored
        filename: Input filename
        
    Returns:
        feature_to_win_rate_model: Dictionary of loaded models for win rate prediction
        player_to_feature_models: Dictionary of loaded models for team feature prediction
        key_features: List of key features used in the models
    """
    import os
    
    full_path = os.path.join(folder, filename)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file not found at {full_path}")
    
    with open(full_path, 'rb') as f:
        model_data = pickle.load(f)
    
    feature_to_win_rate_model = model_data['feature_to_win_rate_model']
    player_to_feature_models = model_data['player_to_feature_models']
    key_features = model_data['key_features']
    
    # Load neural network models
    for model_type, model_dict in feature_to_win_rate_model.items():
        if model_type == 'neural_network' and 'model_path' in model_dict:
            try:
                model_dict['model'] = load_model(model_dict['model_path'])
            except Exception as e:
                print(f"Warning: Could not load model from {model_dict['model_path']}: {e}")
                model_dict['model'] = None
    
    for feature, feature_models in player_to_feature_models.items():
        for model_type, model_dict in feature_models.items():
            if model_type == 'neural_network' and 'model_path' in model_dict:
                try:
                    model_dict['model'] = load_model(model_dict['model_path'])
                except Exception as e:
                    print(f"Warning: Could not load model from {model_dict['model_path']}: {e}")
                    model_dict['model'] = None
    
    print(f"Models loaded from {full_path}")
    return feature_to_win_rate_model, player_to_feature_models, key_features


def train_models_pipeline(players_file='players_stats.csv', teams_file='teams_stats.csv', n_features=8):
    """
    Run the full model training pipeline
    
    Args:
        players_file: Path to the player statistics CSV file
        teams_file: Path to the team statistics CSV file
        n_features: Number of key features to use
        
    Returns:
        df_players: Preprocessed player DataFrame
        df_teams: Preprocessed team DataFrame
        df_aggregated: Aggregated player statistics
        key_features: List of key features
        feature_to_win_rate_model: Dictionary of trained models for win rate prediction
        player_to_feature_models: Dictionary of trained models for team feature prediction
    """
    print("Loading and preprocessing data...")
    df_players, df_teams = load_and_preprocess_data(players_file, teams_file)
    
    print("\nAggregating player statistics by team...")
    df_aggregated = aggregate_players_by_team(df_players)
    
    print("\nIdentifying key team features for win rate prediction...")
    key_features = identify_key_features(df_teams, n_features=n_features)
    
    print("\nTraining win rate prediction model using key team features...")
    feature_to_win_rate_model = train_win_rate_model(df_teams, key_features)
    
    print("\nTraining models to predict key team features from player statistics...")
    player_to_feature_models = train_player_to_team_features(df_aggregated, df_teams, key_features)
    
    # Save the trained models
    save_models(feature_to_win_rate_model, player_to_feature_models, key_features)
    
    return df_players, df_teams, df_aggregated, key_features, feature_to_win_rate_model, player_to_feature_models


def print_team_analysis(analysis):
    """
    Print a formatted analysis of a team composition
    
    Args:
        analysis: Dictionary with analysis results from evaluate_team_composition
        
    Returns:
        None
    """
    print("\n----- Team Composition Analysis -----")
    print(f"Predicted Win Rate: {analysis['win_rate_percent']:.2f}%")
    
    print("\nPredicted Team Stats:")
    for feature, value in analysis['predicted_features'].items():
        print(f"- {feature}: {value:.4f}")
    
    print("\nRole Impact Analysis:")
    for role, score in analysis['role_scores'].items():
        print(f"- {role}: Impact Score = {score:.2f}")
    
    if analysis['strengths']:
        print("\nTeam Strengths:")
        for strength in analysis['strengths']:
            print(f"- {strength}")
    else:
        print("\nNo significant strengths detected.")
    
    if analysis['weaknesses']:
        print("\nTeam Weaknesses:")
        for weakness in analysis['weaknesses']:
            print(f"- {weakness}")
    else:
        print("\nNo significant weaknesses detected.")
    
    print("\nRecommendations:")
    for recommendation in analysis['recommendations']:
        print(f"- {recommendation}")


def generate_random_player_stats():
    """
    Generate random player statistics for testing
    
    Returns:
        Dictionary with random player stats for each role
    """
    roles = ['TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT']
    
    # Define reasonable ranges for different stats
    stat_ranges = {
        'KDA': (1.5, 5.0),
        'Avg kills': {
            'TOP': (1.0, 3.0),
            'JUNGLE': (1.5, 4.0),
            'MID': (2.0, 5.0),
            'ADC': (2.5, 6.0),
            'SUPPORT': (0.5, 2.0)
        },
        'Avg deaths': (1.0, 4.0),
        'Avg assists': {
            'TOP': (3.0, 7.0),
            'JUNGLE': (5.0, 10.0),
            'MID': (4.0, 8.0),
            'ADC': (3.0, 7.0),
            'SUPPORT': (7.0, 15.0)
        },
        'CSM': {
            'TOP': (6.0, 9.0),
            'JUNGLE': (3.0, 6.0),
            'MID': (7.0, 10.0),
            'ADC': (8.0, 11.0),
            'SUPPORT': (0.5, 2.0)
        },
        'GPM': {
            'TOP': (300, 420),
            'JUNGLE': (280, 400),
            'MID': (320, 450),
            'ADC': (350, 480),
            'SUPPORT': (200, 300)
        },
        'DMG%': {
            'TOP': (0.15, 0.25),
            'JUNGLE': (0.10, 0.20),
            'MID': (0.25, 0.35),
            'ADC': (0.25, 0.40),
            'SUPPORT': (0.05, 0.15)
        },
        'DPM': {
            'TOP': (350, 550),
            'JUNGLE': (300, 500),
            'MID': (400, 650),
            'ADC': (450, 700),
            'SUPPORT': (150, 300)
        },
        'Avg WPM': {
            'TOP': (0.4, 0.7),
            'JUNGLE': (0.6, 1.0),
            'MID': (0.4, 0.7),
            'ADC': (0.3, 0.6),
            'SUPPORT': (0.8, 1.6)
        },
        'GD@15': {
            'TOP': (-300, 300),
            'JUNGLE': (-200, 300),
            'MID': (-300, 400),
            'ADC': (-200, 400),
            'SUPPORT': (-150, 150)
        },
        'CSD@15': {
            'TOP': (-15, 15),
            'JUNGLE': (-10, 15),
            'MID': (-15, 20),
            'ADC': (-15, 25),
            'SUPPORT': (-5, 5)
        },
        'XPD@15': {
            'TOP': (-400, 400),
            'JUNGLE': (-300, 300),
            'MID': (-400, 500),
            'ADC': (-350, 350),
            'SUPPORT': (-300, 200)
        }
    }
    
    player_stats = {}
    
    for role in roles:
        stats = {}
        
        for stat, range_info in stat_ranges.items():
            if isinstance(range_info, tuple):
                # Same range for all roles
                min_val, max_val = range_info
            else:
                # Role-specific range
                min_val, max_val = range_info[role]
            
            # Generate random value within range
            stats[stat] = min_val + np.random.random() * (max_val - min_val)
        
        player_stats[role] = stats
    
    return player_stats


def create_team_variants(base_team, n_variants=5):
    """
    Create variations of a team composition for comparison
    
    Args:
        base_team: Dictionary with base player stats for each role
        n_variants: Number of variants to create
        
    Returns:
        List of team variants
    """
    variants = [base_team]  # Include the original team
    
    for i in range(n_variants):
        variant = {}
        
        # Choose 1-3 roles to modify
        n_roles_to_modify = np.random.randint(1, 4)
        roles_to_modify = np.random.choice(list(base_team.keys()), size=n_roles_to_modify, replace=False)
        
        # Copy the base team
        for role in base_team:
            variant[role] = base_team[role].copy()
        
        # Modify the selected roles
        for role in roles_to_modify:
            # Choose 2-5 stats to modify
            stats = list(base_team[role].keys())
            n_stats_to_modify = np.random.randint(2, min(6, len(stats)))
            stats_to_modify = np.random.choice(stats, size=n_stats_to_modify, replace=False)
            
            for stat in stats_to_modify:
                # Modify by ±10-30%
                change_percent = np.random.uniform(0.1, 0.3) * (1 if np.random.random() > 0.5 else -1)
                variant[role][stat] = base_team[role][stat] * (1 + change_percent)
        
        variants.append(variant)
    
    return variants


def plot_team_comparison(team_analyses, title="Team Variants Comparison"):
    """
    Plot a comparison of team variants
    
    Args:
        team_analyses: List of analysis results from evaluate_team_composition
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    win_rates = [analysis['win_rate'] for analysis in team_analyses]
    labels = [f"Team {i+1}" for i in range(len(team_analyses))]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, win_rates, color='skyblue')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax.set_xlabel('Team Variant')
    ax.set_ylabel('Predicted Win Rate')
    ax.set_title(title)
    ax.set_ylim(0, max(win_rates) * 1.2)  # Add some space for labels
    
    plt.tight_layout()
    return fig


def plot_role_impact(analyses, title="Role Impact Analysis"):
    """
    Plot a comparison of role impact across team variants
    
    Args:
        analyses: List of analysis results from evaluate_team_composition
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    roles = list(analyses[0]['role_scores'].keys())
    n_teams = len(analyses)
    
    # Prepare data
    data = {}
    for role in roles:
        data[role] = [analysis['role_scores'][role] for analysis in analyses]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    width = 0.15  # width of the bars
    x = np.arange(n_teams)  # the label locations
    
    # Plot bars for each role
    bars = []
    for i, role in enumerate(roles):
        bars.append(ax.bar(x + (i - 2) * width, data[role], width, label=role))
    
    # Add labels and legend
    ax.set_xlabel('Team Variant')
    ax.set_ylabel('Impact Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Team {i+1}' for i in range(n_teams)])
    ax.legend()
    
    plt.tight_layout()
    return fig


def create_optimal_team(initial_team, player_to_feature_models, feature_to_win_rate_model, key_features, n_iterations=100):
    """
    Create an optimized team composition through iterative improvement
    
    Args:
        initial_team: Dictionary with initial player stats for each role
        player_to_feature_models: Models to predict team features from player stats
        feature_to_win_rate_model: Model to predict win rate from team features
        key_features: List of key team features
        n_iterations: Number of optimization iterations
        
    Returns:
        Optimized team composition
    """
    best_team = initial_team
    best_win_rate, _ = predict_win_rate_from_players(
        best_team, player_to_feature_models, feature_to_win_rate_model, key_features
    )
    
    improvement_log = [(0, best_win_rate)]
    
    for iteration in range(n_iterations):
        # Create a variant
        variant = {}
        for role in best_team:
            variant[role] = best_team[role].copy()
        
        # Modify 1-3 roles
        n_roles_to_modify = np.random.randint(1, 4)
        roles_to_modify = np.random.choice(list(best_team.keys()), size=n_roles_to_modify, replace=False)
        
        for role in roles_to_modify:
            # Choose 2-4 stats to modify
            stats = list(best_team[role].keys())
            n_stats_to_modify = np.random.randint(2, min(5, len(stats)))
            stats_to_modify = np.random.choice(stats, size=n_stats_to_modify, replace=False)
            
            for stat in stats_to_modify:
                # Modify by ±5-20%
                change_percent = np.random.uniform(0.05, 0.2) * (1 if np.random.random() > 0.5 else -1)
                variant[role][stat] = best_team[role][stat] * (1 + change_percent)
        
        # Evaluate the variant
        variant_win_rate, _ = predict_win_rate_from_players(
            variant, player_to_feature_models, feature_to_win_rate_model, key_features
        )
        
        # Update if better
        if variant_win_rate > best_win_rate:
            best_team = variant
            best_win_rate = variant_win_rate
            improvement_log.append((iteration + 1, best_win_rate))
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: Improved win rate to {best_win_rate:.4f}")
    
    return best_team, improvement_log


def plot_optimization_progress(improvement_log, title="Team Optimization Progress"):
    """
    Plot the progress of team optimization
    
    Args:
        improvement_log: List of (iteration, win_rate) tuples
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    iterations, win_rates = zip(*improvement_log)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, win_rates, marker='o')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Win Rate')
    ax.set_title(title)
    ax.grid(True)
    
    plt.tight_layout()
    return fig
    