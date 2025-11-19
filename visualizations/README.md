# Visualization Results

This folder contains all visualization outputs from PPO training and analysis.

## Current Visualizations

### PPO Training Results

1. **ppo_learning_curves_latest.png**
   - Latest PPO training curves (20 episodes)
   - Shows reward improvement over time
   - Demonstrates 77.4% performance improvement
   - First 10 episodes avg: -10.60
   - Last 10 episodes avg: -2.40

2. **ppo_4k_analysis.png**
   - Large-scale PPO analysis (4K dataset)
   - Extended training visualization
   - Performance benchmarks

## Data Source

All visualizations use **real data**:
- Materials Project API (400 crystalline structures)
- Real PPO training episodes with actual rewards
- True performance metrics (R², MAE)
- Genuine 300 in-dist + 100 out-dist split

## Generating New Visualizations

### PPO Training Curves
```bash
python scripts/train_ppo.py --episodes 20
# Output saved to logs/ and can be copied here
```

### Dashboard Visualizations
```bash
# Interactive dashboard
python dashboard/app.py

# Plotly dashboard
python dash_app/plotly_dashboard.py
```

### Analysis Plots
```bash
# Analyze latest PPO results
python scripts/analysis/plot_latest_ppo.py

# Comprehensive analysis
python scripts/analysis/analyze_ppo_results.py
```

## Visualization Contents

Each training curve typically shows:
- **Episode Rewards**: Reward per episode over time
- **Moving Average**: Smoothed trend line
- **Performance Metrics**: R², MAE for each episode
- **Node Selection Frequency**: Which nodes PPO selects most
- **Method Usage Statistics**: Distribution of methods used

## Notes

- All images are PNG format for easy viewing
- Timestamps included in filenames for version tracking
- Original files preserved in `logs/` directory
- This folder contains curated visualizations for presentations

---

**Last Updated**: November 19, 2025
