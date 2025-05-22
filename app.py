import streamlit as st
import tensorflow as tf
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="COVID-19 Detection App",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .covid-positive {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #c62828;
    }
    .covid-negative {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Constants
BASE_PATH = Path(".")
METRICS_PATH = BASE_PATH / "metrics"
MODELS_PATH = BASE_PATH / "model"
DEFAULT_THRESHOLD = 0.592  # Your specified threshold for 90% recall

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model_path = MODELS_PATH / "covid19_model.keras"
        if model_path.exists():
            return tf.keras.models.load_model(model_path)
        else:
            st.error(f"Model not found at {model_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_metrics():
    """Load all saved metrics"""
    metrics = {}
    
    # Load dataset info
    try:
        with open(METRICS_PATH / "dataset_info.json", "r") as f:
            metrics["dataset_info"] = json.load(f)
    except FileNotFoundError:
        st.warning("Dataset info not found")
        metrics["dataset_info"] = {}
    
    # Load optimal threshold
    try:
        with open(METRICS_PATH / "optimal_threshold.json", "r") as f:
            metrics["optimal_threshold"] = json.load(f)
    except FileNotFoundError:
        st.warning("Optimal threshold not found, using default")
        metrics["optimal_threshold"] = {"optimal_threshold": DEFAULT_THRESHOLD}
    
    # Load performance scores
    try:
        with open(METRICS_PATH / "performance_scores.json", "r") as f:
            metrics["performance_scores"] = json.load(f)
    except FileNotFoundError:
        metrics["performance_scores"] = {}
    
    # Load training history
    try:
        metrics["training_history"] = pd.read_csv(METRICS_PATH / "training_history.csv")
    except FileNotFoundError:
        metrics["training_history"] = pd.DataFrame()
    
    # Load threshold scan
    try:
        metrics["threshold_scan"] = pd.read_csv(METRICS_PATH / "threshold_scan.csv")
    except FileNotFoundError:
        metrics["threshold_scan"] = pd.DataFrame()
    
    # Load test results
    try:
        metrics["test_results"] = pd.read_csv(METRICS_PATH / "test_results.csv")
    except FileNotFoundError:
        metrics["test_results"] = pd.DataFrame()
    
    # Load normalization stats
    try:
        with open(METRICS_PATH / "normalization_stats.json", "r") as f:
            metrics["normalization_stats"] = json.load(f)
    except FileNotFoundError:
        metrics["normalization_stats"] = {}
    
    return metrics

def preprocess_image(image, target_size):
    """Preprocess uploaded image for prediction"""
    # Convert to grayscale and resize
    if image.mode != 'L':
        image = image.convert('L')
    
    image = image.resize(target_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(image)
    img_array = img_array.astype("float32") / 255.0
    img_array = img_array.reshape(1, target_size[0], target_size[1], 1)
    
    return img_array

def make_prediction(model, image_array, threshold):
    """Make prediction with custom threshold"""
    prediction_prob =1 - model.predict(image_array)[0][0]
    prediction_label = "COVID-19" if prediction_prob >= threshold else "Normal"
    confidence = prediction_prob if prediction_prob >= threshold else 1 - prediction_prob
    
    return prediction_prob, prediction_label, confidence

def plot_training_history(history_df):
    """Plot training history"""
    if history_df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training & Validation Loss', 'Training & Validation Accuracy',
                       'Loss Comparison', 'Accuracy Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(x=list(range(len(history_df))), y=history_df['loss'], 
                   name='Training Loss', line=dict(color='red')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(history_df))), y=history_df['val_loss'], 
                   name='Validation Loss', line=dict(color='orange')),
        row=1, col=1
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(x=list(range(len(history_df))), y=history_df['accuracy'], 
                   name='Training Accuracy', line=dict(color='blue')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(history_df))), y=history_df['val_accuracy'], 
                   name='Validation Accuracy', line=dict(color='green')),
        row=1, col=2
    )
    
    # Combined plots
    for col in ['loss', 'val_loss']:
        fig.add_trace(
            go.Scatter(x=list(range(len(history_df))), y=history_df[col], 
                       name=col, showlegend=False),
            row=2, col=1
        )
    
    for col in ['accuracy', 'val_accuracy']:
        fig.add_trace(
            go.Scatter(x=list(range(len(history_df))), y=history_df[col], 
                       name=col, showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(height=600, title_text="Model Training History")
    return fig

def plot_threshold_analysis(threshold_df, current_threshold):
    """Plot threshold analysis"""
    if threshold_df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=threshold_df['threshold'],
        y=threshold_df['accuracy'],
        name='Accuracy',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=threshold_df['threshold'],
        y=threshold_df['f1_score'],
        name='F1 Score',
        line=dict(color='red')
    ))
    
    # Add current threshold line
    fig.add_vline(
        x=current_threshold,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Current Threshold: {current_threshold:.3f}"
    )
    
    fig.update_layout(
        title="Threshold Analysis",
        xaxis_title="Threshold",
        yaxis_title="Score",
        hovermode='x unified'
    )
    
    return fig

# Main app
def main():
    st.markdown('<h1 class="main-header">🫁 COVID-19 Detection App</h1>', unsafe_allow_html=True)
    
    # Load model and metrics
    model = load_model()
    metrics = load_metrics()
    
    if model is None:
        st.error("Could not load the model. Please ensure the model file exists.")
        return
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Threshold selection
    optimal_threshold = metrics.get("optimal_threshold", {}).get("optimal_threshold", DEFAULT_THRESHOLD)
    threshold = st.sidebar.slider(
        "Prediction Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(optimal_threshold),
        step=0.001,
        help=f"Default threshold for 90% recall: {DEFAULT_THRESHOLD}"
    )
    
    # Model info
    dataset_info = metrics.get("dataset_info", {})
    if dataset_info:
        st.sidebar.subheader("📊 Dataset Info")
        st.sidebar.write(f"**Training samples:** {dataset_info.get('train_size', 'N/A')}")
        st.sidebar.write(f"**Test samples:** {dataset_info.get('test_size', 'N/A')}")
        st.sidebar.write(f"**Image size:** {dataset_info.get('resize_to', {}).get('width', 'N/A')}x{dataset_info.get('resize_to', {}).get('height', 'N/A')}")
    
    # Performance metrics
    perf_scores = metrics.get("performance_scores", {})
    if perf_scores:
        st.sidebar.subheader("📈 Model Performance")
        st.sidebar.write(f"**Accuracy:** {perf_scores.get('accuracy', 'N/A'):.3f}")
        st.sidebar.write(f"**F1 Score:** {perf_scores.get('f1_score', 'N/A'):.3f}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Model Analytics", "📋 Test Results"])
    
    with tab1:
        st.header("Upload Image for Prediction")
        
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image to get COVID-19 prediction"
        )
        
        if uploaded_file is not None:
            # Display image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                # Get image size for preprocessing
                target_size = (
                    dataset_info.get('resize_to', {}).get('width', 102),
                    dataset_info.get('resize_to', {}).get('height', 102)
                )
                
                # Preprocess and predict
                processed_image = preprocess_image(image, target_size)
                prediction_prob, prediction_label, confidence = make_prediction(
                    model, processed_image, threshold
                )
                
                # Display prediction
                st.subheader("Prediction Results")
                
                # Prediction box
                box_class = "covid-positive" if prediction_label == "COVID-19" else "covid-negative"
                st.markdown(
                    f'<div class="prediction-box {box_class}">Prediction: {prediction_label}</div>',
                    unsafe_allow_html=True
                )
                
                # Metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Probability", f"{prediction_prob:.3f}")
                with col_b:
                    st.metric("Confidence", f"{confidence:.1%}")
                
                # Probability bar
                st.subheader("Probability Distribution")
                prob_data = pd.DataFrame({
                    'Class': ['Normal', 'COVID-19'],
                    'Probability': [1-prediction_prob, prediction_prob]
                })
                
                fig_bar = px.bar(
                    prob_data, x='Class', y='Probability',
                    color='Class',
                    color_discrete_map={'Normal': 'green', 'COVID-19': 'red'}
                )
                fig_bar.add_hline(y=threshold, line_dash="dash", 
                                 annotation_text=f"Threshold: {threshold:.3f}")
                st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.header("Model Analytics")
        
        # Training history
        if not metrics["training_history"].empty:
            st.subheader("Training History")
            history_fig = plot_training_history(metrics["training_history"])
            if history_fig:
                st.plotly_chart(history_fig, use_container_width=True)
        
        # Threshold analysis
        if not metrics["threshold_scan"].empty:
            st.subheader("Threshold Analysis")
            threshold_fig = plot_threshold_analysis(metrics["threshold_scan"], threshold)
            if threshold_fig:
                st.plotly_chart(threshold_fig, use_container_width=True)
            
            # Find optimal thresholds
            thresh_df = metrics["threshold_scan"]
            max_acc_idx = thresh_df['accuracy'].idxmax()
            max_f1_idx = thresh_df['f1_score'].idxmax()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Best Accuracy Threshold",
                    f"{thresh_df.loc[max_acc_idx, 'threshold']:.3f}",
                    f"Accuracy: {thresh_df.loc[max_acc_idx, 'accuracy']:.3f}"
                )
            with col2:
                st.metric(
                    "Best F1 Threshold",
                    f"{thresh_df.loc[max_f1_idx, 'threshold']:.3f}",
                    f"F1: {thresh_df.loc[max_f1_idx, 'f1_score']:.3f}"
                )
            with col3:
                st.metric(
                    "Current Threshold",
                    f"{threshold:.3f}",
                    "90% Recall Target"
                )
    
    with tab3:
        st.header("Test Results Analysis")
        
        if not metrics["test_results"].empty:
            test_df = metrics["test_results"]
            
            # Summary statistics
            st.subheader("Test Set Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", len(test_df))
            with col2:
                covid_count = len(test_df[test_df['true_label'] == 'COVID-19'])
                st.metric("COVID-19 Cases", covid_count)
            with col3:
                normal_count = len(test_df[test_df['true_label'] == 'Normal'])
                st.metric("Normal Cases", normal_count)
            with col4:
                # Calculate accuracy with current threshold
                pred_labels_current = (test_df['predicted_prob'] >= threshold).astype(int)
                true_labels_binary = (test_df['true_label'] == 'COVID-19').astype(int)
                current_accuracy = (pred_labels_current == true_labels_binary).mean()
                st.metric(f"Accuracy @ {threshold:.3f}", f"{current_accuracy:.3f}")
            
            # Prediction distribution
            st.subheader("Prediction Probability Distribution")
            fig_hist = px.histogram(
                test_df, x='predicted_prob', color='true_label',
                nbins=50, title="Distribution of Prediction Probabilities"
            )
            fig_hist.add_vline(x=threshold, line_dash="dash", 
                             annotation_text=f"Threshold: {threshold:.3f}")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Detailed results table
            st.subheader("Detailed Test Results")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                filter_class = st.selectbox("Filter by True Label", 
                                          ["All", "COVID-19", "Normal"])
            with col2:
                show_misclassified = st.checkbox("Show only misclassified")
            
            # Apply filters
            filtered_df = test_df.copy()
            if filter_class != "All":
                filtered_df = filtered_df[filtered_df['true_label'] == filter_class]
            
            if show_misclassified:
                # Add predicted labels with current threshold
                filtered_df['predicted_label_current'] = np.where(
                    filtered_df['predicted_prob'] >= threshold, 'COVID-19', 'Normal'
                )
                filtered_df = filtered_df[
                    filtered_df['true_label'] != filtered_df['predicted_label_current']
                ]
            
            # Display table
            st.dataframe(
                filtered_df[['filepath', 'true_label', 'predicted_prob']],
                use_container_width=True
            )
        else:
            st.info("No test results available. Run the training script first.")

if __name__ == "__main__":
    main()