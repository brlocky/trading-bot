"""
Real-Time Observation Monitor - Streamlit Dashboard
No disk writes, pure in-memory visualization

Launch: streamlit run streamlit_obs_monitor.py
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from environments.simple_trading_env import SimpleTradingEnv
import time

st.set_page_config(layout="wide", page_title="🔍 Observation Monitor", page_icon="📊")

# Sidebar Configuration
st.sidebar.title("⚙️ Configuration")

DATA_PATH = st.sidebar.text_input("Data Path", "data/binance-BTCUSDT-5m.pkl")
LOOKBACK_WINDOW = st.sidebar.number_input("Lookback Window", value=288, min_value=50, max_value=500)
N_BINS = st.sidebar.number_input("VP Bins", value=50, min_value=20, max_value=200)
DATA_SIZE = st.sidebar.slider("Data Size", 1000, 50000, 10000, step=1000)

# Load data


@st.cache_data
def load_data(path, size):
    df = pd.read_pickle(path)
    return df.iloc[:size].reset_index(drop=True)


try:
    df = load_data(DATA_PATH, DATA_SIZE)
    st.sidebar.success(f"✓ Loaded {len(df):,} rows")
except Exception as e:
    st.sidebar.error(f"❌ Error loading data: {e}")
    st.stop()

# Initialize environment


@st.cache_resource
def create_env(data, lookback, n_bins):
    return SimpleTradingEnv(data, lookback_window=lookback, n_bins=n_bins)


env = create_env(df, LOOKBACK_WINDOW, N_BINS)

# Session state initialization
if 'obs' not in st.session_state:
    st.session_state.obs, st.session_state.info = env.reset()
    st.session_state.step_count = 0
    st.session_state.action_history = []
    st.session_state.reward_history = []
    st.session_state.auto_run = False

# Control panel
st.sidebar.markdown("---")
st.sidebar.title("🎮 Controls")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.obs, st.session_state.info = env.reset()
        st.session_state.step_count = 0
        st.session_state.action_history = []
        st.session_state.reward_history = []
        st.rerun()

with col2:
    if st.button("⏭️ Step", use_container_width=True):
        action = env.action_space.sample()
        st.session_state.obs, reward, done, truncated, info = env.step(action)
        st.session_state.step_count += 1
        st.session_state.action_history.append(action[0] if isinstance(action, (list, np.ndarray)) else action)
        st.session_state.reward_history.append(reward)

        if done or truncated:
            st.session_state.obs, st.session_state.info = env.reset()
            st.sidebar.warning("Episode ended, reset environment")

        st.rerun()

# Manual action selection
st.sidebar.markdown("### 🎯 Manual Action")
action_map = {0: "HOLD", 1: "LONG", 2: "SHORT", 3: "CLOSE"}
selected_action = st.sidebar.selectbox("Select Action", list(action_map.keys()), format_func=lambda x: action_map[x])

if st.sidebar.button("Execute Action", use_container_width=True):
    st.session_state.obs, reward, done, truncated, info = env.step([selected_action])
    st.session_state.step_count += 1
    st.session_state.action_history.append(selected_action)
    st.session_state.reward_history.append(reward)

    if done or truncated:
        st.session_state.obs, st.session_state.info = env.reset()
        st.sidebar.warning("Episode ended, reset environment")

    st.rerun()

# Auto-step controls
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Auto Mode")
auto_speed = st.sidebar.slider("Speed (steps/sec)", 1, 10, 2)
auto_steps = st.sidebar.number_input("Steps to run", 1, 1000, 10)

if st.sidebar.button("▶️ Auto Run", use_container_width=True):
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    for i in range(auto_steps):
        action = env.action_space.sample()
        st.session_state.obs, reward, done, truncated, info = env.step(action)
        st.session_state.step_count += 1
        st.session_state.action_history.append(action[0] if isinstance(action, (list, np.ndarray)) else action)
        st.session_state.reward_history.append(reward)

        if done or truncated:
            st.session_state.obs, st.session_state.info = env.reset()

        progress_bar.progress((i + 1) / auto_steps)
        status_text.text(f"Step {i+1}/{auto_steps}")
        time.sleep(1.0 / auto_speed)

    st.sidebar.success(f"✓ Completed {auto_steps} steps")
    st.rerun()

# Filter controls
st.sidebar.markdown("---")
st.sidebar.title("🎨 Display Options")
selected_groups = st.sidebar.multiselect(
    "Feature Groups to Display",
    list(st.session_state.obs.keys()),
    default=list(st.session_state.obs.keys())
)

colorscale = st.sidebar.selectbox(
    "Color Scale",
    ["RdBu", "Viridis", "Plasma", "Hot", "Cool", "Portland", "Picnic"]
)

show_stats = st.sidebar.checkbox("Show Statistics", value=True)
show_distributions = st.sidebar.checkbox("Show Distributions", value=False)

# Main Title
st.title("🔍 Real-Time Observation Monitor")

# Top metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Step", st.session_state.step_count)
with col2:
    st.metric("Episode Progress", f"{st.session_state.step_count / len(df) * 100:.1f}%")
with col3:
    if len(st.session_state.reward_history) > 0:
        st.metric("Last Reward", f"{st.session_state.reward_history[-1]:.2f}")
    else:
        st.metric("Last Reward", "N/A")
with col4:
    if len(st.session_state.reward_history) > 0:
        st.metric("Cumulative Reward", f"{sum(st.session_state.reward_history):.2f}")
    else:
        st.metric("Cumulative Reward", "0.00")

st.markdown("---")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Heatmaps", "📈 Statistics", "🎯 History"])

with tab1:
    # Display observations as heatmaps
    if not selected_groups:
        st.warning("⚠️ Select at least one feature group to display")
    else:
        for name in selected_groups:
            if name not in st.session_state.obs:
                continue

            data = st.session_state.obs[name]

            st.subheader(f"{name} {data.shape}")

            # Determine zmid based on feature type
            zmid = 0 if name not in ['vp_bins', 'micro_temporal', 'micro_spatial'] else None

            # Create heatmap
            fig = go.Figure(
                go.Heatmap(
                    z=data.T,
                    colorscale=colorscale,
                    zmid=zmid,
                    colorbar=dict(title="Value"),
                    hovertemplate='Time: %{x}<br>Feature: %{y}<br>Value: %{z:.4f}<extra></extra>'
                )
            )

            fig.update_layout(
                height=300,
                xaxis_title="Timestep →",
                yaxis_title="Feature Index ↑",
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show stats if enabled
            if show_stats:
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Mean", f"{data.mean():.4f}")
                with col2:
                    st.metric("Std", f"{data.std():.4f}")
                with col3:
                    st.metric("Min", f"{data.min():.4f}")
                with col4:
                    st.metric("Max", f"{data.max():.4f}")
                with col5:
                    st.metric("Range", f"{data.max() - data.min():.4f}")

            # Show distribution if enabled
            if show_distributions:
                fig_hist = go.Figure(
                    go.Histogram(
                        x=data.flatten(),
                        nbinsx=50,
                        marker_color='lightblue'
                    )
                )
                fig_hist.update_layout(
                    height=200,
                    title=f"{name} - Value Distribution",
                    xaxis_title="Value",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("---")

with tab2:
    # Statistics view
    st.subheader("📊 Feature Statistics Summary")

    stats_data = []
    for name, data in st.session_state.obs.items():
        stats_data.append({
            'Feature Group': name,
            'Shape': str(data.shape),
            'Mean': f"{data.mean():.4f}",
            'Std': f"{data.std():.4f}",
            'Min': f"{data.min():.4f}",
            'Max': f"{data.max():.4f}",
            'NaN Count': int(np.isnan(data).sum()),
            'Inf Count': int(np.isinf(data).sum())
        })

    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True)

    # Correlation matrix (last timestep only)
    st.subheader("🔗 Feature Correlations (Last Timestep)")

    # Flatten all features to 1D for last timestep
    last_step_data = {}
    for name, data in st.session_state.obs.items():
        last_step_data[name] = data[-1, :].flatten()

    # Only show correlation if we have at least 2 groups
    if len(last_step_data) < 2:
        st.info("Need at least 2 feature groups for correlation analysis")
    else:
        # Create combined array for correlation
        max_features = max(len(v) for v in last_step_data.values())
        corr_data = np.zeros((len(last_step_data), max_features))
        labels = []

        for idx, (name, values) in enumerate(last_step_data.items()):
            corr_data[idx, :len(values)] = values
            labels.append(name)

        # Compute correlation between groups (mean correlation)
        group_means = [v.mean() for v in last_step_data.values()]
        group_corr = np.corrcoef(group_means)

        # Ensure group_corr is 2D
        if group_corr.ndim == 0:
            # Single value, convert to 1x1 array
            group_corr = np.array([[group_corr]])
        elif group_corr.ndim == 1:
            # 1D array, reshape to 2D
            group_corr = group_corr.reshape(1, -1)

        # Convert to list of lists for text annotations
        text_annotations = [[f"{val:.2f}" for val in row] for row in group_corr]

        fig_corr = go.Figure(
            go.Heatmap(
                z=group_corr,
                x=labels,
                y=labels,
                colorscale='RdBu',
                zmid=0,
                text=text_annotations,
                texttemplate='%{text}',
                textfont={"size": 10},
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
            )
        )
        fig_corr.update_layout(height=500, title="Group-Level Mean Correlations")
        st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    # Action and reward history
    st.subheader("🎯 Action & Reward History")

    if len(st.session_state.action_history) > 0:
        # Create history dataframe
        history_df = pd.DataFrame({
            'Step': range(len(st.session_state.action_history)),
            'Action': st.session_state.action_history,
            'Action Name': [action_map[a] for a in st.session_state.action_history],
            'Reward': st.session_state.reward_history
        })

        # Action distribution
        col1, col2 = st.columns(2)

        with col1:
            action_counts = history_df['Action'].value_counts().sort_index()
            fig = go.Figure(
                go.Bar(
                    x=[action_map[i] for i in action_counts.index],
                    y=action_counts.values,
                    marker_color=['gray', 'green', 'red', 'blue'][:len(action_counts)],
                    text=action_counts.values,
                    textposition='outside'
                )
            )
            fig.update_layout(title="Action Distribution", height=300, yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(
                go.Scatter(
                    x=history_df['Step'],
                    y=history_df['Reward'].cumsum(),
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='cyan', width=2),
                    fillcolor='rgba(0,255,255,0.2)'
                )
            )
            fig.update_layout(
                title="Cumulative Reward",
                height=300,
                xaxis_title="Step",
                yaxis_title="Cumulative Reward"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Reward over time
        fig_reward = go.Figure(
            go.Scatter(
                x=history_df['Step'],
                y=history_df['Reward'],
                mode='lines+markers',
                line=dict(color='orange', width=1),
                marker=dict(size=4)
            )
        )
        fig_reward.update_layout(
            title="Reward per Step",
            height=300,
            xaxis_title="Step",
            yaxis_title="Reward"
        )
        st.plotly_chart(fig_reward, use_container_width=True)

        # Recent history table
        st.subheader("📋 Recent Steps")
        st.dataframe(
            history_df[['Step', 'Action Name', 'Reward']].tail(20),
            use_container_width=True
        )

        # Summary statistics
        st.subheader("📊 Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Steps", len(history_df))
        with col2:
            st.metric("Avg Reward", f"{history_df['Reward'].mean():.4f}")
        with col3:
            st.metric("Best Reward", f"{history_df['Reward'].max():.4f}")
        with col4:
            st.metric("Worst Reward", f"{history_df['Reward'].min():.4f}")
    else:
        st.info("📝 No history yet. Take some steps to see action and reward tracking!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("- **Step**: Manually advance one step")
st.sidebar.markdown("- **Auto Run**: Continuous monitoring")
st.sidebar.markdown("- **Manual Action**: Test specific actions")
st.sidebar.markdown("- Filter feature groups to focus analysis")
st.sidebar.markdown("- All processing is **in-memory** (no disk writes)")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Current State")
st.sidebar.json({
    "step": st.session_state.step_count,
    "balance": float(st.session_state.info.get('current_balance', 0)),
    "equity": float(st.session_state.info.get('equity', 0)),
    "position": float(st.session_state.info.get('position_size', 0)),
    "actions_taken": len(st.session_state.action_history)
})
