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
        color: #000000;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        color: #000000;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
        color: #000000;
    }
    .covid-positive {
        background-color: #ffebee;
        color: #000000;
        border: 2px solid #000000;
    }
    .covid-negative {
        background-color: #e8f5e8;
        color: #000000;
        border: 2px solid #000000;
    }
    .uncertain {
        background-color: #fff3e0;
        color: #000000;
        border: 2px solid #000000;
    }
    .confidence-warning {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #000000;
        margin: 1rem 0;
        color: #000000;
    }
    .recommendation-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #000000;
        margin: 1rem 0;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Constants
BASE_PATH = Path(".")
METRICS_PATH = BASE_PATH / "metrics"
MODELS_PATH = BASE_PATH / "model"
DEFAULT_THRESHOLD = 0.592  # Your specified threshold for 90% recall

# Confidence thresholds
COVID_CONFIDENCE_THRESHOLD = 0.90  # 90% confidence for COVID prediction
NORMAL_CONFIDENCE_THRESHOLD = 0.80  # 80% confidence for Normal prediction

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

def make_enhanced_prediction(model, image_array, threshold, covid_conf_thresh=0.90, normal_conf_thresh=0.80):
    """Make prediction with confidence-based classification"""
    # Get raw prediction probability
    prediction_prob = 1 - model.predict(image_array)[0][0]
    
    # Determine prediction based on threshold
    basic_prediction = "COVID-19" if prediction_prob >= threshold else "Normal"
    
    # Apply confidence thresholding
    if basic_prediction == "COVID-19":
        # For COVID prediction, need high confidence
        if prediction_prob >= covid_conf_thresh:
            final_prediction = "COVID-19"
            confidence = prediction_prob
            certainty_level = "High Confidence"
        else:
            final_prediction = "Uncertain - Consult Specialist"
            confidence = prediction_prob
            certainty_level = "Low Confidence"
    else:
        # For Normal prediction, need moderate confidence
        normal_confidence = 1 - prediction_prob
        if normal_confidence >= normal_conf_thresh:
            final_prediction = "Normal"
            confidence = normal_confidence
            certainty_level = "High Confidence"
        else:
            final_prediction = "Uncertain - Consult Specialist"
            confidence = max(prediction_prob, normal_confidence)
            certainty_level = "Low Confidence"
    
    return {
        'prediction_prob': prediction_prob,
        'final_prediction': final_prediction,
        'basic_prediction': basic_prediction,
        'confidence': confidence,
        'certainty_level': certainty_level,
        'covid_confidence': prediction_prob,
        'normal_confidence': 1 - prediction_prob
    }

def get_clinical_recommendation(prediction_result):
    """Generate clinical recommendations based on prediction"""
    pred = prediction_result['final_prediction']
    confidence = prediction_result['confidence']
    covid_prob = prediction_result['covid_confidence']
    
    if pred == "COVID-19":
        return {
            'title': '🔴 HIGH PRIORITY ACTION REQUIRED',
            'message': f"""
            **Immediate Recommendations:**
            - Isolate immediately and follow local health guidelines
            - Contact healthcare provider for RT-PCR confirmation
            - Monitor symptoms closely
            - Inform close contacts about potential exposure
            
            **Model Confidence:** {confidence:.1%} (High confidence COVID-19 detection)
            """
        }
    elif pred == "Normal":
        return {
            'title': '✅ LOW RISK INDICATION',
            'message': f"""
            **Recommendations:**
            - Continue regular health monitoring
            - Follow standard prevention measures
            - Consult healthcare provider if symptoms develop or worsen
            - Consider retesting if exposure risk is high
            
            **Model Confidence:** {confidence:.1%} (High confidence normal scan)
            """
        }
    else:  # Uncertain
        return {
            'title': '⚠️ INCONCLUSIVE RESULT - SPECIALIST CONSULTATION REQUIRED',
            'message': f"""
            **Critical Next Steps:**
            - **Consult a healthcare specialist immediately**
            - Request professional radiological interpretation
            - Consider RT-PCR testing regardless of symptoms
            - Do not rely solely on this AI screening
            
            **Why uncertain?**
            - COVID probability: {covid_prob:.1%} (needs ≥90% for confident COVID diagnosis)
            - Normal probability: {1-covid_prob:.1%} (needs ≥80% for confident normal diagnosis)
            - The model cannot make a reliable determination
            """
        }

def plot_confidence_visualization(prediction_result, covid_thresh, normal_thresh):
    """Create confidence visualization"""
    covid_prob = prediction_result['covid_confidence']
    normal_prob = prediction_result['normal_confidence']
    
    # Create confidence bar chart
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=['Normal', 'COVID-19'],
        y=[normal_prob, covid_prob],
        marker_color=['green' if normal_prob >= normal_thresh else 'orange',
                     'red' if covid_prob >= covid_thresh else 'orange'],
        text=[f'{normal_prob:.1%}', f'{covid_prob:.1%}'],
        textposition='auto',
    ))
    
    # Add confidence threshold lines
    fig.add_hline(y=normal_thresh, line_dash="dash", line_color="green", 
                  annotation_text=f"Normal Confidence Threshold: {normal_thresh:.0%}")
    fig.add_hline(y=covid_thresh, line_dash="dash", line_color="red",
                  annotation_text=f"COVID Confidence Threshold: {covid_thresh:.0%}")
    
    fig.update_layout(
        title="Prediction Confidence Analysis",
        yaxis_title="Confidence Level",
        xaxis_title="Predicted Class",
        yaxis=dict(range=[0, 1], tickformat='.0%'),
        showlegend=False
    )
    
    return fig

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
        "Base Prediction Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(optimal_threshold),
        step=0.001,
        help=f"Default threshold for 90% recall: {DEFAULT_THRESHOLD}"
    )
    
    # Confidence threshold controls
    st.sidebar.subheader("🎯 Confidence Thresholds")
    covid_conf_thresh = st.sidebar.slider(
        "COVID-19 Confidence Threshold",
        min_value=0.50,
        max_value=0.99,
        value=COVID_CONFIDENCE_THRESHOLD,
        step=0.01,
        help="Minimum confidence required to predict COVID-19"
    )
    
    normal_conf_thresh = st.sidebar.slider(
        "Normal Confidence Threshold", 
        min_value=0.50,
        max_value=0.95,
        value=NORMAL_CONFIDENCE_THRESHOLD,
        step=0.01,
        help="Minimum confidence required to predict Normal"
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
        
        # Important disclaimer
        st.markdown("""
        <div class="confidence-warning">
        <strong>⚠️ Medical Disclaimer:</strong> This is an AI screening tool and should NOT replace professional medical diagnosis. 
        Always consult healthcare professionals for medical decisions.
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image to get COVID-19 screening"
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
                prediction_result = make_enhanced_prediction(
                    model, processed_image, threshold, covid_conf_thresh, normal_conf_thresh
                )
                
                # Display prediction
                st.subheader("AI Screening Results")
                
                # Prediction box with appropriate styling
                final_pred = prediction_result['final_prediction']
                if final_pred == "COVID-19":
                    box_class = "covid-positive"
                elif final_pred == "Normal":
                    box_class = "covid-negative"
                else:
                    box_class = "uncertain"
                
                st.markdown(
                    f'<div class="prediction-box {box_class}">{final_pred}</div>',
                    unsafe_allow_html=True
                )
                
                # Metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("COVID Probability", f"{prediction_result['covid_confidence']:.1%}")
                with col_b:
                    st.metric("Normal Probability", f"{prediction_result['normal_confidence']:.1%}")
                with col_c:
                    st.metric("Certainty Level", prediction_result['certainty_level'])
        
            # Full-width sections
            if uploaded_file is not None:
                # Confidence visualization
                st.subheader("Confidence Analysis")
                confidence_fig = plot_confidence_visualization(
                    prediction_result, covid_conf_thresh, normal_conf_thresh
                )
                st.plotly_chart(confidence_fig, use_container_width=True)
                
                # Clinical recommendations
                recommendation = get_clinical_recommendation(prediction_result)
                st.markdown(f"""
                <div class="recommendation-box">
                <h4>{recommendation['title']}</h4>
                {recommendation['message']}
                </div>
                """, unsafe_allow_html=True)
                
                # Technical details (collapsible)
                with st.expander("🔧 Technical Details"):
                    st.write(f"**Base prediction threshold:** {threshold:.3f}")
                    st.write(f"**COVID confidence threshold:** {covid_conf_thresh:.1%}")
                    st.write(f"**Normal confidence threshold:** {normal_conf_thresh:.1%}")
                    st.write(f"**Raw COVID probability:** {prediction_result['prediction_prob']:.3f}")
                    st.write(f"**Basic prediction (before confidence check):** {prediction_result['basic_prediction']}")
    
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
            
            # Enhanced prediction analysis with confidence thresholds
            st.subheader("Enhanced Prediction Analysis")
            
            # Apply enhanced prediction logic to test set
            enhanced_predictions = []
            for _, row in test_df.iterrows():
                pred_prob = row['predicted_prob']
                basic_pred = "COVID-19" if pred_prob >= threshold else "Normal"
                
                if basic_pred == "COVID-19" and pred_prob >= covid_conf_thresh:
                    enhanced_pred = "COVID-19"
                elif basic_pred == "Normal" and (1 - pred_prob) >= normal_conf_thresh:
                    enhanced_pred = "Normal"
                else:
                    enhanced_pred = "Uncertain"
                
                enhanced_predictions.append(enhanced_pred)
            
            test_df_enhanced = test_df.copy()
            test_df_enhanced['enhanced_prediction'] = enhanced_predictions
            
            # Summary of enhanced predictions
            pred_counts = test_df_enhanced['enhanced_prediction'].value_counts()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                covid_pred = pred_counts.get('COVID-19', 0)
                st.metric("Confident COVID Predictions", covid_pred, 
                         f"{covid_pred/len(test_df_enhanced):.1%} of total")
            with col2:
                normal_pred = pred_counts.get('Normal', 0)
                st.metric("Confident Normal Predictions", normal_pred,
                         f"{normal_pred/len(test_df_enhanced):.1%} of total")
            with col3:
                uncertain_pred = pred_counts.get('Uncertain', 0)
                st.metric("Uncertain Cases", uncertain_pred,
                         f"{uncertain_pred/len(test_df_enhanced):.1%} of total")
            
            # Prediction distribution
            st.subheader("Prediction Probability Distribution")
            fig_hist = px.histogram(
                test_df, x='predicted_prob', color='true_label',
                nbins=50, title="Distribution of Prediction Probabilities"
            )
            fig_hist.add_vline(x=threshold, line_dash="dash", 
                             annotation_text=f"Base Threshold: {threshold:.3f}")
            fig_hist.add_vline(x=covid_conf_thresh, line_dash="dot", line_color="red",
                             annotation_text=f"COVID Confidence: {covid_conf_thresh:.1%}")
            fig_hist.add_vline(x=1-normal_conf_thresh, line_dash="dot", line_color="green",
                             annotation_text=f"Normal Confidence: {normal_conf_thresh:.1%}")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Detailed results table
            st.subheader("Enhanced Test Results")
            
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_class = st.selectbox("Filter by True Label", 
                                          ["All", "COVID-19", "Normal"])
            with col2:
                filter_prediction = st.selectbox("Filter by Enhanced Prediction",
                                               ["All", "COVID-19", "Normal", "Uncertain"])
            with col3:
                show_misclassified = st.checkbox("Show only misclassified")
            
            # Apply filters
            filtered_df = test_df_enhanced.copy()
            if filter_class != "All":
                filtered_df = filtered_df[filtered_df['true_label'] == filter_class]
            
            if filter_prediction != "All":
                filtered_df = filtered_df[filtered_df['enhanced_prediction'] == filter_prediction]
            
            if show_misclassified:
                # Only show cases where enhanced prediction doesn't match true label
                # (excluding uncertain cases as they're intentionally ambiguous)
                misclassified = (
                    (filtered_df['enhanced_prediction'] != 'Uncertain') &
                    (filtered_df['true_label'] != filtered_df['enhanced_prediction'])
                )
                filtered_df = filtered_df[misclassified]
            
            # Display table
            display_columns = ['filepath', 'true_label', 'predicted_prob', 'enhanced_prediction']
            st.dataframe(
                filtered_df[display_columns],
                use_container_width=True
            )
        else:
            st.info("No test results available. Run the training script first.")

if __name__ == "__main__":
    main()