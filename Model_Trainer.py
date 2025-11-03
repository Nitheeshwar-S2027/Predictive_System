import pandas as pd
import numpy as np
import os
import joblib
import warnings
import time
# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")
import lightgbm as lgb
# Suppress LightGBM internal warnings during basic operation
lgb.basic._log_warning = lambda *args, **kwargs: None
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, mean_absolute_error
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor, RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor


import os

# ====================================================================
# --- CRITICAL FIXES for macOS/M-series Segmentation Faults (SIGSEGV) ---
# Forces NumPy, MKL, OpenBLAS, etc., to use single-threaded operations 
# to prevent conflicts with Python's multithreading and PyTorch.
# Must be set BEFORE heavy libraries like pandas/numpy/torch are imported.
# ====================================================================
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
# ====================================================================

# The rest of your imports should follow here:
# import pandas as pd
# import numpy as np
# ...

# --- Global Configuration and Setup ---
DATA_LIMIT = None # Limit data loading for fast execution/testing
EPOCHS = 25
N_JOBS = -1 # Use all available cores for ML models
os.makedirs("models", exist_ok=True) # Create directory to save trained models

# Configuration mapping datasets to their specific settings
DATA_CONFIG = {
    # Engine Data: Failure Classification Task
    "engine": {
        "path": "engine_data.csv",
        "task": "classification",
        "target": "Machine failure",
        "categorical_features": ["Product ID", "Type"],
        "original_numerical_features": ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"],
        "final_numerical_features": ['Temp_Diff', 'Power_Proxy', 'Overstrain_Proxy'], # Engineered features
    },
    # EV Data: Remaining Useful Life (RUL) Regression Task
    "ev": {
        "path": "ev_data.csv",
        "task": "regression",
        "target": "RUL",
        "categorical_features": [],
        "original_numerical_features": ["Cycle_Index", "Discharge Time (s)", "Decrement 3.6-3.4V (s)", "Max. Voltage Dischar. (V)",
                       "Min. Voltage Charg. (V)", "Time at 4.15V (s)", "Time constant current (s)", "Charging time (s)"],
        "final_numerical_features": ["cycle_index", "discharge time (s)", "decrement 3.6-3.4v (s)", "max. voltage dischar. (v)",
                   "min. voltage charg. (v)", "time at 4.15v (s)", "time constant current (s)", "charging time (s)"],
    }
}

# --- Deep Learning Model Definitions (PyTorch) ---

class TransformerEncoderBlock(nn.Module):
    """
    A minimal one-layer Transformer Encoder block using nn.TransformerEncoderLayer.
    Used as the final layer before the dense output network in hybrid models.
    """
    def __init__(self, d_model, nhead, dim_feedforward):
        super(TransformerEncoderBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
    
    def forward(self, x):
        return self.transformer_encoder(x)

class HybridDNN(nn.Module):
    """
    Hybrid Deep Neural Network: Dense Embedding -> Transformer -> Dense Output.
    """
    def __init__(self, input_size, output_size=1):
        super(HybridDNN, self).__init__()
        # Embed the input features into a latent space (32 dimensions)
        self.embedding = nn.Linear(input_size, 32)
        # Apply the single-layer Transformer encoder
        self.transformer = TransformerEncoderBlock(d_model=32, nhead=1, dim_feedforward=64)
        # Final dense layers for prediction
        self.net = nn.Sequential(nn.Linear(32 * 1, 64), nn.ReLU(), nn.Linear(64, output_size))
        
    def forward(self, x):
        x = self.embedding(x)
        # Unsqueeze to add sequence dimension (Batch, 1, Feature_dim) for the transformer
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        return self.net(x)

class HybridCNN(nn.Module):
    """
    Hybrid Convolutional Neural Network: 1D CNN -> Transformer -> Dense Output.
    """
    def __init__(self, input_size, output_size=1):
        super(HybridCNN, self).__init__()
        # 1D Convolutional layers for local feature extraction
        self.conv = nn.Sequential(nn.Conv1d(1, 32, kernel_size=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.transformer = TransformerEncoderBlock(d_model=32, nhead=1, dim_feedforward=64)
        self.net = nn.Sequential(nn.Linear(32 * 1, 64), nn.ReLU(), nn.Linear(64, output_size))
        
    def forward(self, x):
        # Reshape for Conv1D: (Batch, 1, input_size)
        x = x.unsqueeze(1)
        x = self.conv(x)
        # Reshape for Transformer: (Batch, 1, 32)
        x = x.squeeze(-1)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        return self.net(x)

class HybridGRU(nn.Module):
    """
    Hybrid Gated Recurrent Unit Network: GRU -> Transformer -> Dense Output.
    """
    def __init__(self, input_size, output_size=1):
        super(HybridGRU, self).__init__()
        # GRU layer, processes feature vector as a sequence of single-dimension inputs
        self.gru = nn.GRU(input_size=1, hidden_size=32, batch_first=True)
        self.transformer = TransformerEncoderBlock(d_model=32, nhead=1, dim_feedforward=64)
        self.net = nn.Sequential(nn.Linear(32 * 1, 64), nn.ReLU(), nn.Linear(64, output_size))
        
    def forward(self, x):
        # Reshape for GRU: (Batch, Seq_len=input_size, Feature_dim=1)
        x = x.unsqueeze(2)
        # Get the final hidden state (h_n)
        _, h_n = self.gru(x)
        # Reshape final hidden state for transformer
        x = h_n[-1].unsqueeze(1) 
        x = self.transformer(x)
        x = x.flatten(start_dim=1)
        return self.net(x)

print("\nSelect model type to TRAIN for Engine (Failure Classification) and EV (RUL Regression):")
print("1. Hybrid DNN (Deep Learning) \n2. Hybrid CNN (Deep Learning) \n3. Hybrid GRU (Deep Learning) \n4. XGBoost (ML - Boosting) \n5. LightGBM (ML - Boosting) \n6. CatBoost (ML - Boosting) \n7. Stacking Ensemble (ML - Stacking) ")
choice = input("Enter choice (1-7): ").strip()

# --- PyTorch Utility Functions ---

def pytorch_predict_wrapper(model, X_np, task_type):
    """Handles inference for PyTorch models and formats output based on task type."""
    model.eval() # Set model to evaluation mode
    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    with torch.no_grad():
        output = model(X_tensor)
        
        if task_type == "classification":
            # Apply sigmoid to get probabilities for class 1
            probs = torch.sigmoid(output).squeeze(1).numpy()
            # Format output as (p_class_0, p_class_1)
            p_class_1 = probs
            p_class_0 = 1 - probs
            return np.column_stack((p_class_0, p_class_1))
        else:
            # Regression prediction returns raw output values
            return output.squeeze(1).numpy()

def train_pytorch_model(model, X_test_tensor, train_loader_local, task_type):
    """
    Runs the main PyTorch training loop (forward pass, loss, backward pass, optimizer step).
    """
    # Use BCEWithLogitsLoss for classification (handles sigmoid internally), MSELoss for regression
    criterion = nn.BCEWithLogitsLoss() if task_type == "classification" else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader_local:
            optimizer.zero_grad() 
            output = model(X_batch) 
            loss = criterion(output, y_batch) 
            loss.backward() 
            optimizer.step() 
            
    end_time = time.time()
    training_time = end_time - start_time
    
    # Predict on the test set and get class predictions
    y_pred_probs_or_values = pytorch_predict_wrapper(model, X_test_tensor.numpy(), task_type)
    
    if task_type == "classification":
        # Convert probabilities (from class 1 column) to class labels (0 or 1)
        y_pred = (y_pred_probs_or_values[:, 1] > 0.5).astype(int)
    else:
        y_pred = y_pred_probs_or_values
        
    return model, y_pred, training_time


# --- Main Training and Evaluation Loop ---
for vehicle_type, config in DATA_CONFIG.items():
    task_type = config["task"]
    print(f"\n=======================================================")
    print(f"🚀 Starting training for {vehicle_type.upper()} ({task_type.upper()} Task)")
    print(f"=======================================================")
    
    # 1. Data Loading and Initial Cleaning
    try:
        data = pd.read_csv(config["path"], nrows=DATA_LIMIT)
    except FileNotFoundError:
        print(f"🚨 CRITICAL ERROR: File not found at {config['path']}. Skipping.")
        continue
        
    # Standardize data cleaning: ensure columns are numeric where possible, handling array-like strings
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].astype(str).str.replace('[\[\]]', '', regex=True)
        data[col] = pd.to_numeric(data[col], errors='ignore')
        
    for col in data.select_dtypes(include='object').columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        
    data = data.fillna(0) # Simple fill missing values with zero
    
    # 2. Feature Engineering (Specific to 'engine' dataset)
    if vehicle_type == "engine":
        required_eng_cols = config["original_numerical_features"]
        
        # Ensure numerical columns are correctly typed before calculation
        for col in required_eng_cols:
             data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
             
        # Create new features based on domain knowledge
        data['Temp_Diff'] = data['Process temperature [K]'] - data['Air temperature [K]']
        data['Power_Proxy'] = data['Torque [Nm]'] * data['Rotational speed [rpm]']
        data['Overstrain_Proxy'] = data['Tool wear [min]'] * data['Torque [Nm]']
        
        numerical_features = config["final_numerical_features"]
        categorical_features = config["categorical_features"]
        
        # Ensure categorical columns are string type for preprocessor
        for cat_col in categorical_features:
            if cat_col in data.columns:
                data[cat_col] = data[cat_col].astype(str)
                
        # Filter the DataFrame to only include relevant features and target
        data_filtered = data[numerical_features + categorical_features + [config["target"]]]
        data = data_filtered
        
    else:
        numerical_features = config["final_numerical_features"]
        categorical_features = config["categorical_features"]

    # 3. Data Preprocessing Pipeline (Scaling & Encoding)
    if categorical_features:
        # Preprocessor for numerical (StandardScaler) and categorical (OneHotEncoder) features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
    else:
        # Preprocessor only for numerical features (no categorical features in 'ev')
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features)
            ])
            
    X = data.drop(config["target"], axis=1)
    y = data[config["target"]]
    
    # Apply preprocessing (scaling and encoding)
    X_scaled = preprocessor.fit_transform(X)
    
    # ✅ Save preprocessor for consistent scaling during prediction
    joblib.dump(preprocessor, os.path.join("models", f"{vehicle_type}_preprocessor.pkl"))

    # Ensure X_scaled is a numpy array (important for SMOTE and PyTorch/ML models)
    if not isinstance(X_scaled, np.ndarray):
        X_scaled = X_scaled.toarray()
    X_scaled = X_scaled.astype(np.float64)

    # 4. Handle Class Imbalance (SMOTE for Classification)
    if task_type == "classification":
        # Apply SMOTE to balance the classes
        k_val = 1 if len(y.unique()) < 3 or (y.value_counts().min() < 2) else 2 
        smote = SMOTE(random_state=None, k_neighbors=k_val)
        X_scaled, y = smote.fit_resample(X_scaled, y)
        
    # 5. Split Data and Convert to PyTorch Tensors
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_scaled, y, test_size=0.3, random_state=None)
    
    # Convert numpy arrays to PyTorch tensors for DL models
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    # Target tensor must be (N, 1) for BCEWithLogitsLoss
    y_train_tensor = torch.tensor(y_train_np.values, dtype=torch.float32).unsqueeze(1)
    
    # Create DataLoader for batch processing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
    
    INPUT_SIZE = X_train_np.shape[1]
    OUTPUT_SIZE = 1 
    training_time = 0
    model = None
    y_pred = None

    # 6. Model Training and Saving
    
    # --- Deep Learning Model Training (Choice 1-3) ---
    if choice in ["1", "2", "3"]:
        if choice == "1":
            model = HybridDNN(INPUT_SIZE, OUTPUT_SIZE)
            base_name = "hybrid_dnn_model.pt"
        elif choice == "2":
            model = HybridCNN(INPUT_SIZE, OUTPUT_SIZE)
            base_name = "hybrid_cnn_model.pt"
        elif choice == "3":
            model = HybridGRU(INPUT_SIZE, OUTPUT_SIZE)
            base_name = "hybrid_gru_model.pt"
            
        # Train the selected PyTorch model
        model, y_pred, training_time = train_pytorch_model(model, X_test_tensor, train_loader, task_type)
        
        # Save PyTorch model state dict
        torch.save(model.state_dict(), os.path.join("models", f"{vehicle_type}_{base_name}"))
        print(f"✅ Deep Learning model saved as {vehicle_type}_{base_name}")
        
    # --- Machine Learning Model Training (Choice 4-7) ---
    elif choice in ["4", "5", "6", "7"]:
        # Select appropriate ML model classes based on task type (Classification or Regression)
        if task_type == "classification":
            XGBM = XGBClassifier
            StackM = StackingClassifier
            RFM = RandomForestClassifier
            GBM = GradientBoostingClassifier
            model_kwargs = {'n_jobs': N_JOBS, 'random_state': 42}
        else:
            XGBM = XGBRegressor
            StackM = StackingRegressor
            RFM = RandomForestRegressor
            GBM = GradientBoostingRegressor
            model_kwargs = {'n_jobs': N_JOBS, 'random_state': 42}
            
        if choice == "4":
            model = XGBM(eval_metric='logloss', use_label_encoder=False, **model_kwargs)
            base_name = "xgboost_model.pkl"
        elif choice == "5":
            # LightGBM configuration with silent verbose setting
            model_kwargs_silent = model_kwargs.copy()
            model_kwargs_silent['verbose'] = -1 
            model = LGBMClassifier(**model_kwargs_silent) if task_type == "classification" else LGBMRegressor(**model_kwargs_silent)
            base_name = "lightgbm_model.pkl"
        elif choice == "6":
            # CatBoost configured to be silent during training
            model = CatBoostClassifier(verbose=0, random_state=42) if task_type == "classification" else CatBoostRegressor(verbose=0, random_state=42)
            base_name = "catboost_model.pkl"
        elif choice == "7":
            # Stacking Ensemble definition
            base_models = [
                ('rf', RFM(n_estimators=50, n_jobs=N_JOBS, random_state=42)),
                ('gb', GBM(n_estimators=50, random_state=42))
            ]
            meta_model = XGBM(eval_metric='logloss', use_label_encoder=False, **model_kwargs)
            model = StackM(estimators=base_models, final_estimator=meta_model, n_jobs=N_JOBS)
            base_name = "stacking_ensemble.pkl"
            
        start_time = time.time()
        model.fit(X_train_np, y_train_np)
        end_time = time.time()
        training_time = end_time - start_time
        
        y_pred = model.predict(X_test_np)
        
        # Save ML models using joblib
        joblib.dump(model, os.path.join("models", f"{vehicle_type}_{base_name}"))
        print(f"✅ {model.__class__.__name__} model saved as {vehicle_type}_{base_name}")
        
    else:
        print("❌ Invalid choice! Skipping training for this type.")
        continue
        
    # 7. Model Evaluation and Reporting
    if y_pred is None:
        print("❌ Evaluation skipped: Model training failed or y_pred was not generated.")
        continue
        
    print(f"\n⏱️ Training Time: {training_time:.4f} seconds")
    y_test_np_flat = y_test_np.values.flatten()
    
    if task_type == "classification":
        # Convert prediction to integers (0 or 1)
        report_pred = np.round(y_pred).astype(int) 
        overall_accuracy = accuracy_score(y_test_np_flat, report_pred)
        print(f"🎯 **Overall Accuracy:** {overall_accuracy:.4f}")
        print(f"\n📊 Model Evaluation Report for {vehicle_type.upper()} ({task_type.upper()}):")
        # Print detailed classification report
        print(classification_report(y_test_np_flat.astype(int), report_pred)) 
    else:
        # Calculate standard regression metrics
        mse = mean_squared_error(y_test_np_flat, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_np_flat, y_pred)
        r2 = r2_score(y_test_np_flat, y_pred)
        
        print(f"📈 **R-squared (R2):** {r2:.4f}")
        print(f"\n📊 Model Evaluation Report for {vehicle_type.upper()} ({task_type.upper()}):")
        
        # Display regression metrics in a formatted table
        report = pd.DataFrame({
            'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'R-squared (R2)'],
            'Value': [f"{mae:.4f}", f"{mse:.4f}", f"{rmse:.4f}", f"{r2:.4f}"]
        })
        print(report.to_string(index=False))
        print("\nNote: MAE represents the average error in RUL cycles.")

print("\n🎉 Training complete for all selected models on both datasets.")
