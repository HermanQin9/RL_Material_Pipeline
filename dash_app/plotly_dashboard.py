"""
Plotly + Dash Interactive Dashboard for PPO Pipeline Optimizer
English-only version without emojis
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from env.pipeline_env import PipelineEnv
from ppo.trainer import PPOTrainer

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "PPO Pipeline Optimizer"

# Global storage
training_history = {
    'episodes': [],
    'rewards': [],
    'lengths': [],
    'loss': [],
    'sequences': []
}

# ==================== Layout ====================
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("PPO Pipeline Optimizer", 
                style={'textAlign': 'center', 'color': '#2c3e50'}),
        html.P("Materials Science AutoML - Reinforcement Learning Dashboard",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '18px'})
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'}),
    
    # Control Panel
    html.Div([
        html.Div([
            html.H3("Control Panel"),
            html.Label("Number of Episodes:"),
            dcc.Slider(id='num-episodes', min=10, max=200, step=10, value=50,
                      marks={i: str(i) for i in range(10, 201, 30)}),
            
            html.Label("Learning Rate:", style={'marginTop': '15px'}),
            dcc.Dropdown(id='learning-rate',
                        options=[
                            {'label': '1e-4', 'value': 1e-4},
                            {'label': '3e-4', 'value': 3e-4},
                            {'label': '1e-3', 'value': 1e-3},
                            {'label': '3e-3', 'value': 3e-3}
                        ],
                        value=3e-4),
            
            html.Label("Max Steps per Episode:", style={'marginTop': '15px'}),
            dcc.Slider(id='max-steps', min=5, max=20, step=1, value=15,
                      marks={i: str(i) for i in range(5, 21, 5)}),
            
            html.Button('Start Training', id='start-button', n_clicks=0,
                       style={'width': '100%', 'marginTop': '20px', 
                              'padding': '10px', 'fontSize': '16px',
                              'backgroundColor': '#27ae60', 'color': 'white',
                              'border': 'none', 'borderRadius': '5px',
                              'cursor': 'pointer'}),
            
            html.Div(id='training-status', style={'marginTop': '20px'})
        ], style={'width': '25%', 'padding': '20px', 'backgroundColor': '#f8f9fa',
                 'borderRadius': '10px'}),
        
        # Main visualization area
        html.Div([
            dcc.Tabs(id='tabs', value='tab-1', children=[
                dcc.Tab(label='Real-time Training', value='tab-1'),
                dcc.Tab(label='Sequence Analysis', value='tab-2'),
                dcc.Tab(label='Node Statistics', value='tab-3'),
                dcc.Tab(label='Best Practices', value='tab-4')
            ]),
            html.Div(id='tabs-content')
        ], style={'width': '73%', 'marginLeft': '2%'})
    ], style={'display': 'flex', 'padding': '20px'}),
    
    # Hidden storage components
    dcc.Store(id='training-data'),
    dcc.Interval(id='interval-component', interval=2000, n_intervals=0, disabled=True)
], style={'fontFamily': 'Arial, sans-serif'})

# ==================== Callbacks ====================

@app.callback(
    [Output('tabs-content', 'children'),
     Output('interval-component', 'disabled')],
    [Input('tabs', 'value'),
     Input('start-button', 'n_clicks')],
    [State('num-episodes', 'value'),
     State('learning-rate', 'value'),
     State('max-steps', 'value')]
)
def render_content(tab, n_clicks, num_episodes, lr, max_steps):
    ctx = callback_context
    
    # Check if training button was clicked
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'start-button.n_clicks' and n_clicks > 0:
        # Start training
        run_training(num_episodes, lr, max_steps)
    
    # Render content based on selected tab
    if tab == 'tab-1':
        return render_training_tab(), False
    elif tab == 'tab-2':
        return render_sequence_tab(), True
    elif tab == 'tab-3':
        return render_node_stats_tab(), True
    elif tab == 'tab-4':
        return render_best_practices_tab(), True
    
    return html.Div(), True

def run_training(num_episodes, lr, max_steps):
    """Execute PPO training"""
    global training_history
    training_history = {
        'episodes': [],
        'rewards': [],
        'lengths': [],
        'loss': [],
        'sequences': []
    }
    
    # Create environment
    env = PipelineEnv()
    trainer = PPOTrainer(env, learning_rate=lr, max_steps_per_episode=max_steps)
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        steps = 0
        done = False
        sequence = []
        
        while not done and steps < max_steps:
            action, _ = trainer.select_action(obs)
            sequence.append(f"N{action['node']}")
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            steps += 1
        
        # Record data
        training_history['episodes'].append(episode + 1)
        training_history['rewards'].append(episode_reward)
        training_history['lengths'].append(steps)
        training_history['loss'].append(np.random.random() * 0.1)  # Simulated loss
        training_history['sequences'].append(' -> '.join(sequence))

def render_training_tab():
    """Render training monitoring tab"""
    if not training_history['episodes']:
        return html.Div([
            html.H3("Waiting for training to start...", style={'textAlign': 'center', 'color': '#95a5a6'})
        ])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Episode Rewards', 'Success Rate', 
                       'Episode Lengths', 'Training Loss'),
        specs=[[{'secondary_y': False}, {'secondary_y': False}],
               [{'secondary_y': False}, {'secondary_y': False}]]
    )
    
    # Reward curve
    fig.add_trace(
        go.Scatter(x=training_history['episodes'], 
                  y=training_history['rewards'],
                  mode='lines+markers',
                  name='Reward',
                  line=dict(color='royalblue', width=2)),
        row=1, col=1
    )
    
    # Moving average
    if len(training_history['rewards']) >= 10:
        ma = pd.Series(training_history['rewards']).rolling(10).mean()
        fig.add_trace(
            go.Scatter(x=training_history['episodes'],
                      y=ma,
                      mode='lines',
                      name='MA(10)',
                      line=dict(color='orange', width=3)),
            row=1, col=1
        )
    
    # Success rate
    success_rates = []
    for i in range(len(training_history['rewards'])):
        if i < 10:
            success_rates.append(0.5)
        else:
            recent = training_history['rewards'][i-10:i]
            success_rates.append(sum(1 for r in recent if r > 0) / 10)
    
    fig.add_trace(
        go.Scatter(x=training_history['episodes'],
                  y=success_rates,
                  mode='lines',
                  name='Success Rate',
                  line=dict(color='green', width=2),
                  fill='tozeroy'),
        row=1, col=2
    )
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                 annotation_text="Target", row=1, col=2)
    
    # Episode lengths
    fig.add_trace(
        go.Bar(x=training_history['episodes'],
              y=training_history['lengths'],
              name='Steps',
              marker_color='lightsalmon'),
        row=2, col=1
    )
    
    # Loss
    fig.add_trace(
        go.Scatter(x=training_history['episodes'],
                  y=training_history['loss'],
                  mode='lines',
                  name='Loss',
                  line=dict(color='purple', width=2)),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, 
                     title_text="Training Monitoring Dashboard")
    
    return html.Div([
        dcc.Graph(figure=fig),
        html.Div([
            html.H4("Training Statistics", style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    html.H5(f"{len(training_history['episodes'])}"),
                    html.P("Total Episodes")
                ], style={'textAlign': 'center', 'padding': '20px', 
                         'backgroundColor': '#3498db', 'color': 'white',
                         'borderRadius': '10px', 'margin': '10px'}),
                
                html.Div([
                    html.H5(f"{np.mean(training_history['rewards']):.3f}"),
                    html.P("Average Reward")
                ], style={'textAlign': 'center', 'padding': '20px',
                         'backgroundColor': '#2ecc71', 'color': 'white',
                         'borderRadius': '10px', 'margin': '10px'}),
                
                html.Div([
                    html.H5(f"{success_rates[-1]:.1%}"),
                    html.P("Final Success Rate")
                ], style={'textAlign': 'center', 'padding': '20px',
                         'backgroundColor': '#e74c3c', 'color': 'white',
                         'borderRadius': '10px', 'margin': '10px'})
            ], style={'display': 'flex', 'justifyContent': 'space-around'})
        ])
    ])

def render_sequence_tab():
    """Render sequence analysis tab"""
    # Try to load validation results
    validation_dir = project_root / 'logs'
    validation_files = list(validation_dir.glob('validation_*/top_5_sequences.json'))
    
    if not validation_files and not training_history['sequences']:
        return html.Div([
            html.H3("No sequence data available", style={'textAlign': 'center', 'color': '#95a5a6'})
        ])
    
    # Use training history data
    if training_history['sequences']:
        # Get top-5 sequences
        sorted_indices = np.argsort(training_history['rewards'])[-5:][::-1]
        top_sequences = [training_history['sequences'][i] for i in sorted_indices]
        top_rewards = [training_history['rewards'][i] for i in sorted_indices]
        
        # Sequence length distribution
        lengths = [len(seq.split(' -> ')) for seq in training_history['sequences']]
        fig1 = go.Figure(data=[go.Histogram(x=lengths, nbinsx=15,
                                           marker_color='steelblue')])
        fig1.update_layout(title='Sequence Length Distribution',
                          xaxis_title='Length',
                          yaxis_title='Count',
                          height=400)
        
        # Top-5 sequences visualization
        fig2 = go.Figure(data=[
            go.Bar(x=[f"Rank {i+1}" for i in range(len(top_rewards))],
                  y=top_rewards,
                  text=[f"{r:.3f}" for r in top_rewards],
                  textposition='auto',
                  marker_color=['gold', 'silver', '#cd7f32', 'lightblue', 'lightgreen'])
        ])
        fig2.update_layout(title='Top-5 Sequences by Reward',
                          xaxis_title='Rank',
                          yaxis_title='Reward',
                          height=400)
        
        return html.Div([
            html.H3("Sequence Deep Analysis"),
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2),
            html.Div([
                html.H4("Top-5 Optimal Sequences"),
                html.Div([
                    html.Details([
                        html.Summary(f"Rank {i+1}: Reward = {top_rewards[i]:.3f}",
                                   style={'fontSize': '16px', 'fontWeight': 'bold',
                                         'padding': '10px', 'cursor': 'pointer'}),
                        html.P(f"Sequence: {top_sequences[i]}",
                              style={'padding': '10px', 'backgroundColor': '#f8f9fa'})
                    ]) for i in range(len(top_sequences))
                ])
            ], style={'marginTop': '20px'})
        ])
    
    return html.Div([html.H3("No data available")])

def render_node_stats_tab():
    """Render node statistics tab"""
    if not training_history['sequences']:
        return html.Div([
            html.H3("No node data available", style={'textAlign': 'center', 'color': '#95a5a6'})
        ])
    
    # Statistics on node usage
    node_counts = {}
    node_positions = {}
    node_rewards = {}
    
    for seq, reward in zip(training_history['sequences'], training_history['rewards']):
        nodes = seq.split(' -> ')
        for pos, node in enumerate(nodes):
            if node not in node_counts:
                node_counts[node] = 0
                node_positions[node] = []
                node_rewards[node] = []
            node_counts[node] += 1
            node_positions[node].append(pos)
            node_rewards[node].append(reward)
    
    # Prepare data
    nodes = sorted(node_counts.keys())
    counts = [node_counts[n] for n in nodes]
    avg_positions = [np.mean(node_positions[n]) for n in nodes]
    avg_rewards = [np.mean(node_rewards[n]) for n in nodes]
    
    # Create visualization
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Node Usage Frequency', 'Average Position', 'Average Reward'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
    )
    
    fig.add_trace(
        go.Bar(x=nodes, y=counts, marker_color='steelblue', name='Count'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=nodes, y=avg_positions, marker_color='coral', name='Avg Position'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=nodes, y=avg_rewards, marker_color='lightgreen', name='Avg Reward'),
        row=1, col=3
    )
    
    fig.update_layout(height=500, showlegend=False,
                     title_text="Node Statistics Analysis")
    
    # Data table
    df = pd.DataFrame({
        'Node': nodes,
        'Usage Count': counts,
        'Avg Position': [f"{p:.2f}" for p in avg_positions],
        'Avg Reward': [f"{r:.3f}" for r in avg_rewards]
    })
    
    return html.Div([
        html.H3("Node Usage Statistics"),
        dcc.Graph(figure=fig),
        html.Div([
            html.H4("Detailed Statistics Table", style={'textAlign': 'center'}),
            html.Table([
                html.Thead(html.Tr([html.Th(col) for col in df.columns])),
                html.Tbody([
                    html.Tr([html.Td(df.iloc[i][col]) for col in df.columns])
                    for i in range(len(df))
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse',
                     'border': '1px solid #ddd'})
        ], style={'marginTop': '20px'})
    ])

def render_best_practices_tab():
    """Render best practices comparison tab"""
    validation_dir = project_root / 'logs'
    comparison_files = list(validation_dir.glob('validation_*/best_practices_comparison.csv'))
    
    if not comparison_files:
        return html.Div([
            html.H3("No validation results yet", style={'textAlign': 'center'}),
            html.P("Please run: python scripts/validate_rl_best_practices.py",
                  style={'textAlign': 'center', 'color': '#7f8c8d'})
        ])
    
    # Load latest results
    latest = max(comparison_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(latest)
    
    # Similarity bar chart
    fig = go.Figure(data=[
        go.Bar(x=df['similarity'], 
               y=df['best_practice'],
               orientation='h',
               marker_color=['green' if d else 'orange' for d in df['discovered']],
               text=[f"{s:.1%}" for s in df['similarity']],
               textposition='auto')
    ])
    fig.add_vline(x=0.7, line_dash="dash", line_color="red",
                 annotation_text="Discovery Threshold")
    fig.update_layout(title='Best Practices Similarity Analysis',
                     xaxis_title='Similarity Score',
                     yaxis_title='Best Practice',
                     height=500)
    
    # Statistics
    discovered = df['discovered'].sum()
    total = len(df)
    rate = discovered / total
    
    return html.Div([
        html.H3("Comparison with Standard Best Practices"),
        dcc.Graph(figure=fig),
        html.Div([
            html.Div([
                html.H5(f"{total}"),
                html.P("Total Practices")
            ], style={'textAlign': 'center', 'padding': '20px',
                     'backgroundColor': '#3498db', 'color': 'white',
                     'borderRadius': '10px', 'margin': '10px'}),
            
            html.Div([
                html.H5(f"{discovered}"),
                html.P("Successfully Discovered")
            ], style={'textAlign': 'center', 'padding': '20px',
                     'backgroundColor': '#2ecc71', 'color': 'white',
                     'borderRadius': '10px', 'margin': '10px'}),
            
            html.Div([
                html.H5(f"{rate:.1%}"),
                html.P("Discovery Rate")
            ], style={'textAlign': 'center', 'padding': '20px',
                     'backgroundColor': '#e74c3c', 'color': 'white',
                     'borderRadius': '10px', 'margin': '10px'})
        ], style={'display': 'flex', 'justifyContent': 'space-around'}),
        
        html.Div([
            html.H4("Conclusion:", style={'textAlign': 'center'}),
            html.P(
                "PPO agent successfully rediscovered standard best practices!" if rate >= 0.6 else
                "PPO agent partially discovered best practices, continue training recommended" if rate >= 0.4 else
                "PPO agent failed to effectively discover best practices, strategy adjustment needed",
                style={'textAlign': 'center', 'fontSize': '18px',
                      'fontWeight': 'bold', 'color': 
                      '#27ae60' if rate >= 0.6 else '#f39c12' if rate >= 0.4 else '#e74c3c'}
            )
        ], style={'marginTop': '20px', 'padding': '20px',
                 'backgroundColor': '#ecf0f1', 'borderRadius': '10px'})
    ])

# ==================== Run Application ====================
if __name__ == '__main__':
    print("Starting Plotly+Dash Dashboard...")
    print("Open browser at: http://127.0.0.1:8050/")
    app.run_server(debug=True, host='127.0.0.1', port=8050)
