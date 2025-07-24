import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="AutoML Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class AutoMLPipeline:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.task_type = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def detect_task_type(self, y):
        """Detect if it's classification or regression"""
        if y.dtype == 'object' or len(y.unique()) <= 20:
            return 'classification'
        else:
            return 'regression'
    
    def preprocess_data(self, df, target_column):
        """Preprocess the dataset"""
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle missing values
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
        y = y.fillna(y.mode().iloc[0] if y.dtype == 'object' else y.mean())
        
        # Encode categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
        # Encode target if classification
        if self.task_type == 'classification' and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoders['target'] = le
        
        return X, y
    
    def get_models(self, task_type):
        """Get appropriate models based on task type"""
        if task_type == 'classification':
            return {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42, probability=True),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'KNN': KNeighborsClassifier(),
                'Naive Bayes': GaussianNB()
            }
        else:
            return {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression(),
                'SVM': SVR(),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'KNN': KNeighborsRegressor()
            }
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and evaluate performance"""
        models = self.get_models(self.task_type)
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            if self.task_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'predictions': y_pred
                }
                
                # Track best model
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = model
                    
            else:
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                self.results[name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2_score': r2,
                    'predictions': y_pred
                }
                
                # Track best model (higher R² is better)
                if r2 > self.best_score:
                    self.best_score = r2
                    self.best_model = model

def create_download_link(obj, filename):
    """Create a download link for the model"""
    buffer = io.BytesIO()
    pickle.dump(obj, buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
    """Create confusion matrix plot"""
    cm = confusion_matrix(y_test, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f'Predicted {i}' for i in range(len(cm))],
        y=[f'Actual {i}' for i in range(len(cm))],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=600,
        height=500
    )
    
    return fig

def plot_roc_curve(y_test, y_pred_proba, title="ROC Curve"):
    """Create ROC curve plot"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.2f})',
        line=dict(color='darkorange', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='navy', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600,
        height=500
    )
    
    return fig

def plot_feature_importance(model, feature_names, title="Feature Importance"):
    """Create feature importance plot"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        fig = go.Figure(data=[
            go.Bar(
                x=[feature_names[i] for i in indices[:10]],
                y=importance[indices[:10]],
                marker_color='skyblue'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title='Features',
            yaxis_title='Importance',
            width=800,
            height=500
        )
        
        return fig
    else:
        return None

def main():
    st.markdown('<h1 class="main-header">🤖 AutoML Studio</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Upload your dataset and let AI find the best machine learning model for you!</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'automl' not in st.session_state:
        st.session_state.automl = AutoMLPipeline()
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # File upload
    st.sidebar.subheader("📁 Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            st.sidebar.success(f"Dataset loaded successfully! Shape: {df.shape}")
            
            # Display dataset info
            st.subheader("📊 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'<div class="metric-card"><h3>{df.shape[0]}</h3><p>Rows</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h3>{df.shape[1]}</h3><p>Columns</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><h3>{df.isnull().sum().sum()}</h3><p>Missing Values</p></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card"><h3>{df.select_dtypes(include=[np.number]).shape[1]}</h3><p>Numeric Columns</p></div>', unsafe_allow_html=True)
            
            # Display first few rows
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
            # Target column selection
            st.sidebar.subheader("🎯 Target Selection")
            target_column = st.sidebar.selectbox("Select target column", df.columns)
            
            # Task type detection
            if target_column:
                st.session_state.automl.task_type = st.session_state.automl.detect_task_type(df[target_column])
                st.sidebar.info(f"Task type: {st.session_state.automl.task_type.title()}")
            
            # Model training parameters
            st.sidebar.subheader("🔧 Training Parameters")
            test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2)
            random_state = st.sidebar.number_input("Random state", 0, 100, 42)
            
            # Train models button
            if st.sidebar.button("🚀 Train Models", type="primary"):
                if target_column:
                    with st.spinner("Training models... This may take a few minutes."):
                        try:
                            # Preprocess data
                            X, y = st.session_state.automl.preprocess_data(df, target_column)
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=random_state
                            )
                            
                            # Scale features
                            X_train_scaled = st.session_state.automl.scaler.fit_transform(X_train)
                            X_test_scaled = st.session_state.automl.scaler.transform(X_test)
                            
                            # Train models
                            st.session_state.automl.train_models(X_train_scaled, X_test_scaled, y_train, y_test)
                            
                            # Store test data for visualization
                            st.session_state.X_test = X_test_scaled
                            st.session_state.y_test = y_test
                            st.session_state.feature_names = X.columns.tolist()
                            
                            st.session_state.models_trained = True
                            st.sidebar.success("Models trained successfully!")
                            
                        except Exception as e:
                            st.sidebar.error(f"Error training models: {str(e)}")
                else:
                    st.sidebar.error("Please select a target column")
            
        except Exception as e:
            st.sidebar.error(f"Error loading dataset: {str(e)}")
    
    # Main content
    if uploaded_file is None:
        st.markdown("""
        <div class="upload-section">
            <h2>🎯 Get Started</h2>
            <p>Upload your CSV dataset to begin automated machine learning</p>
            <ul style="text-align: left; display: inline-block;">
                <li>📈 Automatic model selection and training</li>
                <li>📊 Performance metrics and comparisons</li>
                <li>🎨 Interactive visualizations</li>
                <li>💾 Download trained models</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif st.session_state.models_trained:
        # Display results
        st.header("🏆 Model Performance Results")
        
        # Create results dataframe
        results_data = []
        for name, result in st.session_state.automl.results.items():
            if st.session_state.automl.task_type == 'classification':
                results_data.append({
                    'Model': name,
                    'Accuracy': f"{result['accuracy']:.4f}",
                    'F1 Score': f"{result['f1_score']:.4f}",
                    'Precision': f"{result['precision']:.4f}",
                    'Recall': f"{result['recall']:.4f}"
                })
            else:
                results_data.append({
                    'Model': name,
                    'R² Score': f"{result['r2_score']:.4f}",
                    'MSE': f"{result['mse']:.4f}",
                    'MAE': f"{result['mae']:.4f}"
                })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
        
        # Best model info
        best_model_name = max(st.session_state.automl.results.keys(), 
                            key=lambda x: st.session_state.automl.results[x]['accuracy' if st.session_state.automl.task_type == 'classification' else 'r2_score'])
        
        st.success(f"🥇 Best Model: {best_model_name}")
        
        # Download best model
        if st.button("💾 Download Best Model"):
            model_data = {
                'model': st.session_state.automl.best_model,
                'scaler': st.session_state.automl.scaler,
                'label_encoders': st.session_state.automl.label_encoders,
                'feature_names': st.session_state.feature_names,
                'task_type': st.session_state.automl.task_type
            }
            
            st.markdown(create_download_link(model_data, f"best_model_{best_model_name.lower().replace(' ', '_')}.pkl"), 
                       unsafe_allow_html=True)
        
        # Visualizations
        st.header("📊 Model Visualizations")
        
        # Model comparison chart
        st.subheader("📈 Model Comparison")
        metric_key = 'accuracy' if st.session_state.automl.task_type == 'classification' else 'r2_score'
        
        models = list(st.session_state.automl.results.keys())
        scores = [st.session_state.automl.results[model][metric_key] for model in models]
        
        fig = go.Figure(data=[
            go.Bar(x=models, y=scores, marker_color='lightblue')
        ])
        
        fig.update_layout(
            title=f"Model Comparison - {metric_key.title()}",
            xaxis_title="Models",
            yaxis_title=metric_key.title(),
            width=800,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification specific visualizations
        if st.session_state.automl.task_type == 'classification':
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Confusion Matrix")
                best_predictions = st.session_state.automl.results[best_model_name]['predictions']
                cm_fig = plot_confusion_matrix(st.session_state.y_test, best_predictions, 
                                             f"Confusion Matrix - {best_model_name}")
                st.plotly_chart(cm_fig, use_container_width=True)
            
            with col2:
                st.subheader("📈 ROC Curve")
                try:
                    if hasattr(st.session_state.automl.best_model, 'predict_proba'):
                        y_pred_proba = st.session_state.automl.best_model.predict_proba(st.session_state.X_test)[:, 1]
                        roc_fig = plot_roc_curve(st.session_state.y_test, y_pred_proba, 
                                               f"ROC Curve - {best_model_name}")
                        st.plotly_chart(roc_fig, use_container_width=True)
                    else:
                        st.info("ROC curve not available for this model type")
                except:
                    st.info("ROC curve not available for multi-class classification")
        
        # Feature importance
        st.subheader("⭐ Feature Importance")
        importance_fig = plot_feature_importance(st.session_state.automl.best_model, 
                                               st.session_state.feature_names,
                                               f"Feature Importance - {best_model_name}")
        
        if importance_fig:
            st.plotly_chart(importance_fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type")
        
        # Model predictions preview
        st.subheader("🔍 Predictions Preview")
        predictions_df = pd.DataFrame({
            'Actual': st.session_state.y_test,
            'Predicted': st.session_state.automl.results[best_model_name]['predictions']
        })
        
        if st.session_state.automl.task_type == 'regression':
            predictions_df['Residual'] = predictions_df['Actual'] - predictions_df['Predicted']
        
        st.dataframe(predictions_df.head(10))
        
        # Download predictions
        if st.button("📥 Download Predictions"):
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
