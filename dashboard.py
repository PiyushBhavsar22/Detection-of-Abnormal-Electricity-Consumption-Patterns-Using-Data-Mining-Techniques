import streamlit as st
import requests
import os
import joblib
from datetime import datetime
import pandas as pd
import io
import json
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import numpy as np

# Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 30  # seconds

# Set up the page
st.set_page_config(page_title="Electricity Theft Detector", page_icon="⚡", layout="centered")

# Initialize session state for prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Initialize SHAP explainer cache
if 'shap_explainer' not in st.session_state:
    st.session_state.shap_explainer = None

# ============================================================================
# VISUALIZATION HELPER FUNCTIONS
# ============================================================================

def create_risk_gauge(probability_percent):
    """Create a gauge chart showing theft risk probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability_percent,
        title = {'text': "Theft Risk Probability", 'font': {'size': 20, 'color': '#e2e8f0'}},
        number = {'suffix': "%", 'font': {'size': 40, 'color': '#f8fafc'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
            'bar': {'color': "#1e293b"},
            'bgcolor': "rgba(2, 6, 23, 0.3)",
            'borderwidth': 2,
            'bordercolor': "rgba(148, 163, 184, 0.3)",
            'steps': [
                {'range': [0, 40], 'color': '#10b981'},  # Green - LOW
                {'range': [40, 60], 'color': '#f59e0b'},  # Yellow - MEDIUM
                {'range': [60, 100], 'color': '#ef4444'}  # Red - HIGH
            ],
            'threshold': {
                'line': {'color': "#f8fafc", 'width': 4},
                'thickness': 0.75,
                'value': probability_percent
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#e2e8f0"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_feature_importance_waterfall(model, feature_names, input_data):
    """Create SHAP waterfall chart showing feature contributions"""
    try:
        # Initialize SHAP explainer if not cached
        if st.session_state.shap_explainer is None:
            st.session_state.shap_explainer = shap.TreeExplainer(model)
        
        explainer = st.session_state.shap_explainer
        
        # Ensure input_data is proper numpy array format
        input_array = input_data.values if hasattr(input_data, 'values') else input_data
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(input_array)
        
        # Get SHAP values for the positive class (thief)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]  # Class 1 (thief)
        else:
            shap_vals = shap_values[0]
        
        # Get base value and expected value
        base_value = explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]
        
        # Create dataframe for plotting - ensure we flatten the values properly
        input_values = input_array[0] if len(input_array.shape) > 1 else input_array
        
        feature_data = pd.DataFrame({
            'feature': feature_names,
            'value': input_values,
            'shap': shap_vals
        })
        
        # Sort by absolute SHAP value and take top 5
        feature_data['abs_shap'] = abs(feature_data['shap'])
        feature_data = feature_data.sort_values('abs_shap', ascending=False).head(5)
        
        # Create waterfall chart
        fig = go.Figure()
        
        # Add bars
        colors = ['#ef4444' if x > 0 else '#10b981' for x in feature_data['shap']]
        
        fig.add_trace(go.Bar(
            x=feature_data['shap'],
            y=feature_data['feature'],
            orientation='h',
            marker=dict(color=colors),
            text=[f"{x:+.3f}" for x in feature_data['shap']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>SHAP value: %{x:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={'text': 'Top 5 Feature Contributions', 'font': {'size': 18, 'color': '#e2e8f0'}},
            xaxis_title='SHAP Value (Impact on Prediction)',
            yaxis_title='',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(2, 6, 23, 0.3)',
            font={'color': "#e2e8f0"},
            height=350,
            margin=dict(l=20, r=80, t=50, b=50),
            xaxis=dict(gridcolor='rgba(148, 163, 184, 0.2)', zeroline=True, zerolinecolor='#cbd5e1')
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error generating feature importance: {str(e)}")
        return None

def create_confidence_bar(probabilities):
    """Create horizontal bar showing prediction confidence"""
    if len(probabilities) < 2:
        prob_normal = 1 - probabilities[0]
        prob_thief = probabilities[0]
    else:
        prob_normal = probabilities[0]
        prob_thief = probabilities[1]
    
    fig = go.Figure()
    
    # Add stacked bar
    fig.add_trace(go.Bar(
        y=['Prediction'],
        x=[prob_normal * 100],
        name='Normal User',
        orientation='h',
        marker=dict(color='#10b981'),
        text=f'{prob_normal*100:.1f}%',
        textposition='inside',
        hovertemplate='Normal User: %{x:.2f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        y=['Prediction'],
        x=[prob_thief * 100],
        name='Potential Thief',
        orientation='h',
        marker=dict(color='#ef4444'),
        text=f'{prob_thief*100:.1f}%',
        textposition='inside',
        hovertemplate='Potential Thief: %{x:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        barmode='stack',
        title={'text': 'Model Confidence Distribution', 'font': {'size': 18, 'color': '#e2e8f0'}},
        xaxis=dict(
            title='Confidence (%)',
            range=[0, 100],
            gridcolor='rgba(148, 163, 184, 0.2)'
        ),
        yaxis=dict(showticklabels=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(2, 6, 23, 0.3)',
        font={'color': "#e2e8f0"},
        height=200,
        margin=dict(l=20, r=20, t=50, b=50),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# ============================================================================
# PHASE 2: MODEL PERFORMANCE VISUALIZATION FUNCTIONS
# ============================================================================

def create_confusion_matrix_heatmap(y_true, y_pred):
    """Create interactive confusion matrix heatmap"""
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create labels
    labels = ['Normal User', 'Thief']
    
    # Create annotated heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20, "color": "white"},
        colorscale=[[0, '#1e293b'], [0.5, '#3b82f6'], [1, '#ef4444']],
        showscale=False,
        hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
    ))
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Add annotations for metrics
    metrics_text = f"Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}"
    
    fig.update_layout(
        title={
            'text': f'Confusion Matrix<br><sub>{metrics_text}</sub>',
            'font': {'size': 18, 'color': '#e2e8f0'}
        },
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(2, 6, 23, 0.3)',
        font={'color': "#e2e8f0"},
        height=400,
        margin=dict(l=80, r=20, t=80, b=50),
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    
    return fig, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def create_roc_curve(y_true, y_proba):
    """Create ROC curve with AUC score"""
    from sklearn.metrics import roc_curve, auc
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='#3b82f6', width=3),
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    ))
    
    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='#ef4444', width=2, dash='dash'),
        hovertemplate='Random (AUC = 0.5)<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': f'ROC Curve (AUC = {roc_auc:.3f})',
            'font': {'size': 18, 'color': '#e2e8f0'}
        },
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(2, 6, 23, 0.3)',
        font={'color': "#e2e8f0"},
        height=400,
        margin=dict(l=50, r=20, t=50, b=50),
        xaxis=dict(range=[0, 1], gridcolor='rgba(148, 163, 184, 0.2)'),
        yaxis=dict(range=[0, 1], gridcolor='rgba(148, 163, 184, 0.2)'),
        showlegend=True,
        legend=dict(x=0.6, y=0.1, bgcolor='rgba(2, 6, 23, 0.8)')
    )
    
    return fig, roc_auc

def create_precision_recall_curve(y_true, y_proba):
    """Create Precision-Recall curve"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    fig = go.Figure()
    
    # Add PR curve
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name=f'PR Curve (AP = {avg_precision:.3f})',
        line=dict(color='#10b981', width=3),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)',
        hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
    ))
    
    # Add baseline
    baseline = sum(y_true) / len(y_true)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[baseline, baseline],
        mode='lines',
        name=f'Baseline (AP = {baseline:.3f})',
        line=dict(color='#f59e0b', width=2, dash='dash'),
        hovertemplate='Baseline<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': f'Precision-Recall Curve (AP = {avg_precision:.3f})',
            'font': {'size': 18, 'color': '#e2e8f0'}
        },
        xaxis_title='Recall',
        yaxis_title='Precision',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(2, 6, 23, 0.3)',
        font={'color': "#e2e8f0"},
        height=400,
        margin=dict(l=50, r=20, t=50, b=50),
        xaxis=dict(range=[0, 1], gridcolor='rgba(148, 163, 184, 0.2)'),
        yaxis=dict(range=[0, 1], gridcolor='rgba(148, 163, 184, 0.2)'),
        showlegend=True,
        legend=dict(x=0.6, y=0.9, bgcolor='rgba(2, 6, 23, 0.8)')
    )
    
    return fig, avg_precision

# ============================================================================
# PHASE 3: BATCH & HISTORY ANALYTICS VISUALIZATION FUNCTIONS
# ============================================================================

def create_risk_distribution_pie(risk_counts):
    """Create pie chart showing risk level distribution"""
    labels = ['HIGH', 'MEDIUM', 'LOW']
    values = [risk_counts.get('HIGH', 0), risk_counts.get('MEDIUM', 0), risk_counts.get('LOW', 0)]
    colors = ['#ef4444', '#f59e0b', '#10b981']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors, line=dict(color='#1e293b', width=2)),
        textinfo='label+percent+value',
        textfont=dict(size=14, color='white'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
        hole=0.4  # Donut style
    )])
    
    total = sum(values)
    fig.update_layout(
        title={
            'text': f'Risk Level Distribution<br><sub>Total: {total} predictions</sub>',
            'font': {'size': 18, 'color': '#e2e8f0'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#e2e8f0"},
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(2, 6, 23, 0.8)'
        )
    )
    
    return fig

def create_prediction_timeline(history_data):
    """Create timeline chart from prediction history"""
    if not history_data or len(history_data) == 0:
        return None
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(history_data)
    
    # Limit to last 100 predictions for performance
    df = df.tail(100)
    
    # Parse timestamps and probabilities
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['prob_numeric'] = df['probability'].str.replace('%', '').astype(float)
    
    # Color mapping for risk levels
    color_map = {'HIGH': '#ef4444', 'MEDIUM': '#f59e0b', 'LOW': '#10b981'}
    df['color'] = df['risk_level'].map(color_map)
    
    fig = go.Figure()
    
    # Add scatter plot for each risk level
    for risk in ['HIGH', 'MEDIUM', 'LOW']:
        df_risk = df[df['risk_level'] == risk]
        if len(df_risk) > 0:
            fig.add_trace(go.Scatter(
                x=df_risk['timestamp'],
                y=df_risk['prob_numeric'],
                mode='markers+lines',
                name=risk,
                marker=dict(
                    size=10,
                    color=color_map[risk],
                    line=dict(width=1, color='white')
                ),
                line=dict(color=color_map[risk], width=2),
                hovertemplate='<b>%{text}</b><br>Time: %{x}<br>Probability: %{y:.1f}%<extra></extra>',
                text=[risk] * len(df_risk)
            ))
    
    # Add threshold lines
    fig.add_hline(y=60, line_dash="dash", line_color="#ef4444", 
                  annotation_text="HIGH threshold (60%)", annotation_position="right")
    fig.add_hline(y=40, line_dash="dash", line_color="#f59e0b",
                  annotation_text="MEDIUM threshold (40%)", annotation_position="right")
    
    fig.update_layout(
        title={
            'text': f'Prediction Timeline (Last {len(df)} predictions)',
            'font': {'size': 18, 'color': '#e2e8f0'}
        },
        xaxis_title='Timestamp',
        yaxis_title='Theft Probability (%)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(2, 6, 23, 0.3)',
        font={'color': "#e2e8f0"},
        height=400,
        margin=dict(l=50, r=20, t=50, b=50),
        xaxis=dict(gridcolor='rgba(148, 163, 184, 0.2)'),
        yaxis=dict(range=[0, 105], gridcolor='rgba(148, 163, 184, 0.2)'),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(2, 6, 23, 0.8)'),
        hovermode='closest'
    )
    
    return fig

def create_feature_distribution_violin(batch_data):
    """Create violin plots for feature distributions by risk level"""
    if not batch_data or len(batch_data) == 0:
        return None
    
    df = pd.DataFrame(batch_data)
    
    # Select key features to display
    key_features = ['total_daily_kwh', 'daily_variance', 'peak_to_offpeak_ratio', 'temperatureMax']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f.replace('_', ' ').title() for f in key_features],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    color_map = {'HIGH': '#ef4444', 'MEDIUM': '#f59e0b', 'LOW': '#10b981'}
    
    for idx, feature in enumerate(key_features):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        for risk in ['HIGH', 'MEDIUM', 'LOW']:
            df_risk = df[df['risk_level'] == risk]
            if len(df_risk) > 0 and feature in df_risk.columns:
                # Extract values from inputs dict if needed
                if df_risk[feature].dtype == 'object':
                    try:
                        values = df_risk['inputs'].apply(lambda x: x.get(feature, 0))
                    except:
                        continue
                else:
                    values = df_risk[feature]
                
                fig.add_trace(
                    go.Violin(
                        y=values,
                        name=risk,
                        marker_color=color_map[risk],
                        showlegend=(idx == 0),  # Only show legend once
                        legendgroup=risk,
                        scalegroup=risk,
                        line_color='white',
                        opacity=0.7
                    ),
                    row=row, col=col
                )
    
    fig.update_layout(
        title={
            'text': 'Feature Distributions by Risk Level',
            'font': {'size': 18, 'color': '#e2e8f0'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(2, 6, 23, 0.3)',
        font={'color': "#e2e8f0"},
        height=600,
        margin=dict(l=50, r=20, t=80, b=50),
        showlegend=True,
        legend=dict(x=0.85, y=0.98, bgcolor='rgba(2, 6, 23, 0.8)'),
        violinmode='group'
    )
    
    fig.update_yaxes(gridcolor='rgba(148, 163, 184, 0.2)')
    
    return fig


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(1100px 520px at -5% -10%, rgba(37, 99, 235, 0.22), transparent 62%),
            radial-gradient(900px 420px at 105% -5%, rgba(14, 165, 233, 0.16), transparent 58%),
            linear-gradient(160deg, #020617 0%, #020b1f 42%, #010510 100%);
        color: #e2e8f0;
    }

    .main .block-container {
        position: relative;
        z-index: 1;
        max-width: 980px;
        padding-top: 1.25rem;
    }

    .hero-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.78), rgba(30, 41, 59, 0.38));
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 18px;
        padding: 1.05rem 1.2rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow: 0 18px 48px rgba(2, 6, 23, 0.68), inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }

    .hero-title {
        margin: 0;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0.01em;
        color: #f8fafc;
        text-shadow: 0 8px 30px rgba(56, 189, 248, 0.22);
    }

    .hero-subtitle {
        margin: 0.45rem 0 0;
        text-align: center;
        color: #cbd5e1;
        font-size: 1.02rem;
    }

    .section-label {
        margin: 0 0 0.7rem;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #93c5fd;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.14);
        background: linear-gradient(140deg, rgba(15, 23, 42, 0.52), rgba(2, 6, 23, 0.28));
        backdrop-filter: blur(8px);
    }

    div[data-testid="stSlider"] > div,
    div[data-testid="stCheckbox"] > label {
        background: rgba(2, 6, 23, 0.56);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 0.3rem 0.45rem;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }

    div[data-testid="stNumberInput"] > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }

    div[data-testid="stNumberInput"] input {
        background: rgba(2, 6, 23, 0.56) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }

    div[data-testid="stNumberInput"] button {
        background: rgba(15, 23, 42, 0.72) !important;
        border: 1px solid rgba(148, 163, 184, 0.24) !important;
        color: #e2e8f0 !important;
    }

    label, .stMarkdown p, .stMarkdown li {
        color: #e2e8f0 !important;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.68), rgba(30, 41, 59, 0.44));
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        padding: 0.78rem;
        backdrop-filter: blur(10px);
    }

    .stButton > button {
        border-radius: 12px !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
        background: linear-gradient(135deg, #1d4ed8, #0ea5e9) !important;
        color: #f8fafc !important;
        font-weight: 700 !important;
        box-shadow: 0 10px 24px rgba(14, 116, 215, 0.4);
        transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
    }

    .stButton > button:hover {
        filter: brightness(1.06);
        box-shadow: 0 14px 30px rgba(14, 116, 215, 0.52), 0 0 18px rgba(56, 189, 248, 0.45);
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <h1 class="hero-title">⚡ AI Electricity Theft Detection</h1>
        <p class="hero-subtitle">Analyze consumer behavior in real-time with a clean, production-ready risk dashboard.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# API HEALTH MONITOR
# ============================================================================
st.markdown('<p class="section-label">System Status</p>', unsafe_allow_html=True)

try:
    start_time = time.time()
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    latency_ms = round((time.time() - start_time) * 1000, 2)
    
    if health_response.status_code == 200:
        health_data = health_response.json()
        api_status = "🟢 Online"
        status_color = "#10b981"
    else:
        api_status = "🟡 Degraded"
        status_color = "#f59e0b"
        latency_ms = "N/A"
except requests.exceptions.RequestException:
    api_status = "🔴 Offline"
    status_color = "#ef4444"
    latency_ms = "N/A"

col_status, col_latency, col_time = st.columns(3)
with col_status:
    st.markdown(f"<div style='padding: 0.5rem; text-align: center; background: rgba(2, 6, 23, 0.56); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 12px;'><span style='color: {status_color}; font-weight: 700; font-size: 1.1rem;'>{api_status}</span></div>", unsafe_allow_html=True)
with col_latency:
    st.markdown(f"<div style='padding: 0.5rem; text-align: center; background: rgba(2, 6, 23, 0.56); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 12px;'><span style='color: #93c5fd; font-weight: 600;'>Latency: {latency_ms} ms</span></div>", unsafe_allow_html=True)
with col_time:
    current_time = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"<div style='padding: 0.5rem; text-align: center; background: rgba(2, 6, 23, 0.56); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 12px;'><span style='color: #cbd5e1; font-weight: 600;'>Last Check: {current_time}</span></div>", unsafe_allow_html=True)

st.divider()

# ============================================================================
# MODEL INFO PANEL
# ============================================================================
with st.expander("📊 Model Information & Configuration", expanded=False):
    try:
        # Load model features
        model_features = joblib.load('model_features.pkl')
        model = joblib.load('theft_detection_model.pkl')
        
        # Model metadata
        st.markdown("### 🔬 Model Architecture")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Algorithm", "Random Forest")
        with col_m2:
            st.metric("Trees", f"{model.n_estimators}")
        with col_m3:
            st.metric("Max Features", model.max_features)
        
        # Training info
        st.markdown("### 📅 Training Information")
        model_path = 'theft_detection_model.pkl'
        if os.path.exists(model_path):
            mod_time = os.path.getmtime(model_path)
            trained_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            st.info(f"**Last Trained:** {trained_date}")
        
        st.info(f"**Random State:** {model.random_state} | **Balanced with SMOTE**")
        
        # Feature list
        st.markdown("### 🎯 Required Input Features")
        st.markdown("The model expects exactly **8 features** in the following order:")
        
        feature_descriptions = {
            "total_daily_kwh": "Total electricity consumption per day (kWh)",
            "daily_variance": "Variance in hourly usage patterns",
            "peak_sum": "Sum of peak hour consumption",
            "off_peak_sum": "Sum of off-peak hour consumption",
            "peak_to_offpeak_ratio": "Ratio of peak to off-peak usage",
            "temperatureMax": "Maximum daily temperature (°C)",
            "temp_hr_std": "Standard deviation of hourly temperature",
            "is_holiday": "Holiday flag (0=No, 1=Yes)"
        }
        
        for i, feature in enumerate(model_features, 1):
            description = feature_descriptions.get(feature, "No description available")
            st.markdown(f"**{i}.** `{feature}` — {description}")
        
        # Performance metrics from notebook
        st.markdown("### 📈 Model Performance (Fair Evaluation)")
        st.markdown("""
        Performance metrics from fair train/test split evaluation:
        - **Accuracy:** ~98.21%
        - **Precision:** ~81.82% (few false positives)
        - **Recall:** ~81.82% (detects most thieves)
        - **F1-Score:** ~81.82% (balanced performance)
        
        *Note: Metrics from independent test set with proper SMOTE handling on training data only.*
        """)
        
        # PHASE 2: Add Performance Visualizations
        st.markdown("---")
        st.markdown("### 📊 Performance Visualizations")
        
        # Load test data
        try:
            X_test = pd.read_csv('X_test_sample.csv')
            y_test = pd.read_csv('y_test_sample.csv').values.ravel()
            
            # Generate predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]  # Probability for positive class
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["📉 Confusion Matrix", "📈 ROC Curve", "📊 Precision-Recall"])
            
            with tab1:
                cm_fig, metrics = create_confusion_matrix_heatmap(y_test, y_pred)
                st.plotly_chart(cm_fig, use_container_width=True)
                
                # Display metrics in columns
                col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                with col_p1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                with col_p2:
                    st.metric("Precision", f"{metrics['precision']:.2%}")
                with col_p3:
                    st.metric("Recall", f"{metrics['recall']:.2%}")
                with col_p4:
                    st.metric("F1-Score", f"{metrics['f1']:.2%}")
            
            with tab2:
                roc_fig, auc_score = create_roc_curve(y_test, y_proba)
                st.plotly_chart(roc_fig, use_container_width=True)
                
                st.info(f"""
                **ROC Curve Interpretation:**
                - AUC = {auc_score:.3f} (Area Under Curve)
                - AUC close to 1.0 indicates excellent model performance
                - The curve shows the trade-off between True Positive Rate and False Positive Rate
                """)
            
            with tab3:
                pr_fig, ap_score = create_precision_recall_curve(y_test, y_proba)
                st.plotly_chart(pr_fig, use_container_width=True)
                
                st.info(f"""
                **Precision-Recall Curve Interpretation:**
                - Average Precision = {ap_score:.3f}
                - Shows the trade-off between precision and recall at different thresholds
                - Higher area under curve indicates better model performance
                """)
                
        except FileNotFoundError:
            st.warning("⚠️ Test data files not found. Run `pipeline.py` to generate test samples.")
        except Exception as e:
            st.warning(f"⚠️ Could not load performance visualizations: {str(e)}")
        
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure `model_features.pkl` and `theft_detection_model.pkl` exist.")
    except Exception as e:
        st.error(f"Error loading model info: {str(e)}")

st.divider()

# ============================================================================
# BATCH UPLOAD SECTION
# ============================================================================
st.markdown('<p class="section-label">📤 Batch Processing</p>', unsafe_allow_html=True)

with st.expander("Upload CSV for Bulk Predictions", expanded=False):
    st.markdown("""
    Upload a CSV file with consumer data to process multiple predictions at once. 
    The CSV must contain the following columns:
    - `total_daily_kwh`, `daily_variance`, `peak_sum`, `off_peak_sum`
    - `peak_to_offpeak_ratio`, `temperatureMax`, `temp_hr_std`, `is_holiday`
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Load expected features
            expected_features = joblib.load('model_features.pkl')
            
            # Validate columns
            missing_cols = set(expected_features) - set(df.columns)
            extra_cols = set(df.columns) - set(expected_features)
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            else:
                st.success(f"✅ Schema validated! Found {len(df)} records.")
                
                # Show preview
                st.markdown("#### 📋 Data Preview (first 5 rows)")
                st.dataframe(df.head(), use_container_width=True)
                
                if extra_cols:
                    st.warning(f"⚠️ Extra columns will be ignored: {', '.join(extra_cols)}")
                
                # Process button
                if st.button("🚀 Process Batch Predictions", type="primary", use_container_width=True):
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    total_records = len(df)
                    
                    for idx, row in df.iterrows():
                        # Update progress
                        progress = (idx + 1) / total_records
                        progress_bar.progress(progress)
                        status_text.text(f"Processing record {idx + 1}/{total_records}...")
                        
                        # Prepare payload (matching API structure)
                        payload = {
                            "total_daily_kwh": float(row['total_daily_kwh']),
                            "peak_to_offpeak_ratio": float(row['peak_to_offpeak_ratio']),
                            "daily_variance": float(row['daily_variance']),
                            "temperature_celsius": float(row['temperatureMax']),
                            "additional_features": {
                                "peak_sum": float(row['peak_sum']),
                                "off_peak_sum": float(row['off_peak_sum']),
                                "temp_hr_std": float(row['temp_hr_std']),
                                "is_holiday": int(row['is_holiday'])
                            }
                        }
                        
                        try:
                            response = requests.post(
                                f"{API_URL}/predict_theft",
                                json=payload,
                                timeout=REQUEST_TIMEOUT
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                
                                # Combine input data with prediction
                                result_row = {
                                    **row.to_dict(),
                                    'prediction': result['prediction'],
                                    'thief_probability': result['thief_probability'],
                                    'risk_level': result['risk_level']
                                }
                                results.append(result_row)
                                
                                # Add to history
                                st.session_state.prediction_history.append({
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'source': 'batch',
                                    'inputs': row.to_dict(),
                                    'prediction': result['prediction'],
                                    'probability': result['thief_probability'],
                                    'risk_level': result['risk_level']
                                })
                            else:
                                results.append({
                                    **row.to_dict(),
                                    'prediction': 'ERROR',
                                    'thief_probability': 'N/A',
                                    'risk_level': 'ERROR'
                                })
                        
                        except Exception as e:
                            results.append({
                                **row.to_dict(),
                                'prediction': f'ERROR: {str(e)}',
                                'thief_probability': 'N/A',
                                'risk_level': 'ERROR'
                            })
                    
                    progress_bar.progress(1.0)
                    status_text.text(f"✅ Completed! Processed {total_records} records.")
                    
                    # Display results
                    st.markdown("#### 📊 Batch Prediction Results")
                    results_df = pd.DataFrame(results)
                    
                    # Store results in session state for export
                    st.session_state.batch_results = results_df
                    
                    # Show summary stats
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        high_risk = len(results_df[results_df['risk_level'] == 'HIGH'])
                        st.metric("🚨 High Risk", high_risk)
                    with col_s2:
                        medium_risk = len(results_df[results_df['risk_level'] == 'MEDIUM'])
                        st.metric("⚠️ Medium Risk", medium_risk)
                    with col_s3:
                        low_risk = len(results_df[results_df['risk_level'] == 'LOW'])
                        st.metric("✅ Low Risk", low_risk)
                    
                    # PHASE 3: Add Risk Distribution Pie Chart
                    st.markdown("---")
                    st.markdown("#### 📊 Risk Distribution")
                    risk_counts = {
                        'HIGH': high_risk,
                        'MEDIUM': medium_risk,
                        'LOW': low_risk
                    }
                    pie_fig = create_risk_distribution_pie(risk_counts)
                    st.plotly_chart(pie_fig, use_container_width=True)
                    
                    # PHASE 3: Add Feature Distribution Violin Plots
                    st.markdown("---")
                    st.markdown("#### 🎻 Feature Distributions by Risk Level")
                    violin_fig = create_feature_distribution_violin(results)
                    if violin_fig:
                        st.plotly_chart(violin_fig, use_container_width=True)
                    
                    # Display full results table
                    st.markdown("---")
                    st.markdown("#### 📋 Detailed Results")
                    st.dataframe(results_df, use_container_width=True, height=400)
                    
                    # Export button
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

st.divider()

# ============================================================================
# SINGLE PREDICTION SECTION
# ============================================================================
st.markdown('<p class="section-label">Consumer Inputs</p>', unsafe_allow_html=True)

st.divider()

# Create sliders and inputs using the EXACT model feature names
col1, col2 = st.columns(2)

with col1:
    total_daily_kwh = st.number_input("Total Daily Usage (total_daily_kwh)", min_value=0.0, max_value=500.0, value=15.0)
    daily_variance = st.number_input("Hourly Variance (daily_variance)", min_value=0.0, max_value=100.0, value=2.5)
# Change the max_value from 200.0 to 1000.0
    peak_sum = st.number_input("Peak Usage Sum (peak_sum)", min_value=0.0, max_value=1000.0, value=5.0)
    off_peak_sum = st.number_input("Off-Peak Usage Sum (off_peak_sum)", min_value=0.0, max_value=1000.0, value=10.0)

with col2:
    peak_to_offpeak_ratio = st.slider("Peak/Off-Peak Ratio", min_value=0.0, max_value=20.0, value=1.2)
    temperatureMax = st.slider("Max Daily Temp (temperatureMax)", min_value=-10.0, max_value=50.0, value=18.0)
    temp_hr_std = st.slider("Temp Hourly Std Dev (temp_hr_std)", min_value=0.0, max_value=15.0, value=2.0)
    is_holiday = st.checkbox("Is it a Holiday? (is_holiday)")

st.divider()

if st.button("🔍 Run Fraud Analysis", type="primary", use_container_width=True):
    
    # We map the UI variables directly to the keys the model_features.pkl expects
    payload = {
        "total_daily_kwh": total_daily_kwh,
        "peak_to_offpeak_ratio": peak_to_offpeak_ratio,
        "daily_variance": daily_variance,
        "temperature_celsius": temperatureMax,
        "additional_features": {
            "peak_sum": peak_sum,
            "off_peak_sum": off_peak_sum,
            "temp_hr_std": temp_hr_std,
            "is_holiday": int(is_holiday)
        }
    }

    try:
        with st.spinner('AI is analyzing behavioral patterns...'):
            response = requests.post(f"{API_URL}/predict_theft", json=payload, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                result = response.json()
                risk_level = result.get("risk_level", "UNKNOWN")

                # Display result based on risk level with appropriate styling
                if risk_level == "HIGH":
                    st.error(f"🚨 **{result['prediction']}** - Risk Level: {risk_level}")
                elif risk_level == "MEDIUM":
                    st.warning(f"⚠️ **{result['prediction']}** - Risk Level: {risk_level}")
                else:
                    st.success(f"✅ **{result['prediction']}** - Risk Level: {risk_level}")

                # Display metrics in columns
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(label="Theft Probability", value=result["thief_probability"])
                with col_b:
                    st.metric(label="Risk Level", value=risk_level)

                st.info(f"**Analysis:** {result['message']}")
                
                # ============================================================================
                # PHASE 1 VISUALIZATIONS: Risk Gauge & Confidence Bar
                # ============================================================================
                st.markdown("---")
                st.markdown("### 📊 Visual Analytics")
                
                # Extract probability as float
                prob_str = result["thief_probability"].replace('%', '')
                probability_percent = float(prob_str)
                
                # 1. Risk Gauge Chart
                st.markdown("#### 🎯 Risk Gauge")
                gauge_fig = create_risk_gauge(probability_percent)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # 2. Confidence Distribution Bar
                st.markdown("#### 📈 Confidence Distribution")
                try:
                    model = joblib.load('theft_detection_model.pkl')
                    model_features = joblib.load('model_features.pkl')
                    
                    # Prepare input dataframe
                    input_dict = {
                        'total_daily_kwh': total_daily_kwh,
                        'daily_variance': daily_variance,
                        'peak_sum': peak_sum,
                        'off_peak_sum': off_peak_sum,
                        'peak_to_offpeak_ratio': peak_to_offpeak_ratio,
                        'temperatureMax': temperatureMax,
                        'temp_hr_std': temp_hr_std,
                        'is_holiday': int(is_holiday)
                    }
                    input_df = pd.DataFrame([input_dict])
                    input_df = input_df[model_features]
                    
                    # Get prediction probabilities
                    probabilities = model.predict_proba(input_df)[0]
                    confidence_fig = create_confidence_bar(probabilities)
                    st.plotly_chart(confidence_fig, use_container_width=True)
                    
                except Exception as e:
                    st.warning("Confidence chart temporarily unavailable")
                
                st.markdown("---")
                
                # Add to prediction history
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'single',
                    'inputs': {
                        'total_daily_kwh': total_daily_kwh,
                        'daily_variance': daily_variance,
                        'peak_sum': peak_sum,
                        'off_peak_sum': off_peak_sum,
                        'peak_to_offpeak_ratio': peak_to_offpeak_ratio,
                        'temperatureMax': temperatureMax,
                        'temp_hr_std': temp_hr_std,
                        'is_holiday': int(is_holiday)
                    },
                    'prediction': result['prediction'],
                    'probability': result['thief_probability'],
                    'risk_level': risk_level
                })
            else:
                st.error(f"API Error: {response.text}")
                
    except requests.exceptions.ConnectionError:
        st.error("API Offline. Run 'uvicorn api:app' in your terminal!")
    except requests.exceptions.Timeout:
        st.error(f"Request timed out after {REQUEST_TIMEOUT}s. Please try again.")

# ============================================================================
# PREDICTION HISTORY
# ============================================================================
st.divider()

if st.session_state.prediction_history:
    st.markdown('<p class="section-label">📜 Prediction History</p>', unsafe_allow_html=True)
    
    with st.expander(f"View History ({len(st.session_state.prediction_history)} predictions)", expanded=False):
        # Summary stats
        st.markdown("### 📊 Summary Statistics")
        
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        with col_h1:
            st.metric("Total Predictions", len(history_df))
        with col_h2:
            high_count = len(history_df[history_df['risk_level'] == 'HIGH'])
            st.metric("🚨 High Risk", high_count)
        with col_h3:
            medium_count = len(history_df[history_df['risk_level'] == 'MEDIUM'])
            st.metric("⚠️ Medium Risk", medium_count)
        with col_h4:
            low_count = len(history_df[history_df['risk_level'] == 'LOW'])
            st.metric("✅ Low Risk", low_count)
        
        # PHASE 3: Add Prediction Timeline
        st.markdown("---")
        st.markdown("### 📈 Prediction Timeline")
        timeline_fig = create_prediction_timeline(st.session_state.prediction_history)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
        else:
            st.info("Timeline will appear after making predictions.")
        
        # Display history table
        st.markdown("---")
        st.markdown("### 📋 Full History")
        
        # Create simplified view for display
        display_data = []
        for record in st.session_state.prediction_history:
            display_data.append({
                'Timestamp': record['timestamp'],
                'Source': record['source'].upper(),
                'Prediction': record['prediction'],
                'Probability': record['probability'],
                'Risk Level': record['risk_level']
            })
        
        display_df = pd.DataFrame(display_data)
        st.dataframe(display_df, use_container_width=True, height=300)
        
        # Export history
        st.markdown("### 💾 Export Options")
        
        col_e1, col_e2 = st.columns(2)
        
        with col_e1:
            # CSV export
            csv_buffer = io.StringIO()
            history_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download History (CSV)",
                data=csv_data,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_e2:
            # JSON export
            import json
            json_data = json.dumps(st.session_state.prediction_history, indent=2)
            
            st.download_button(
                label="📥 Download History (JSON)",
                data=json_data,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Clear history button
        if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()
