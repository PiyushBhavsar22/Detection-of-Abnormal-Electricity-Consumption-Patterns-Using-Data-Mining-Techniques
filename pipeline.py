import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
import os
import glob
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific warnings only
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CONTAMINATION_RATE = 0.05
RANDOM_STATE = 42
TEST_SIZE = 0.2

def run_pipeline():
    
    logger.info("STARTING ADVANCED ML PIPELINE")
    
    # Create output directory if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # PHASE 1: LOAD ALL DATASETS

    logger.info("--- PHASE 1: LOADING DATA ---")

    block_folder = os.path.join(DATA_DIR, "hhblock_dataset")
    block_files = glob.glob(os.path.join(block_folder, "**", "*.csv"), recursive=True)

    logger.info(f"Searching inside -> {block_folder}")
    logger.info(f"Found {len(block_files)} CSV files")

    if not block_files:
        raise FileNotFoundError(f"No CSV files found in '{block_folder}'. Check folder name!")

    logger.info(f"Loading {len(block_files)} block files...")
    df_list = []
    for i, file in enumerate(block_files):
        df_list.append(pd.read_csv(file))
        if (i + 1) % 10 == 0:
            logger.info(f"  Loaded {i + 1}/{len(block_files)} files...")
    df = pd.concat(df_list, ignore_index=True)
    del df_list  # Free memory
    logger.info(f"Loaded {len(df):,} rows of smart meter data.")

    df_house = pd.read_csv(os.path.join(DATA_DIR, "informations_households.csv"), encoding='latin-1')
    df_weather_daily = pd.read_csv(os.path.join(DATA_DIR, "weather_daily_darksky.csv"), encoding='latin-1')
    df_holidays = pd.read_csv(os.path.join(DATA_DIR, "uk_bank_holidays.csv"), encoding='latin-1')
    df_weather_hourly = pd.read_csv(os.path.join(DATA_DIR, "weather_hourly_darksky.csv"), encoding='latin-1')
    df_acorn = pd.read_csv(os.path.join(DATA_DIR, "acorn_details.csv"), encoding='latin-1')

    # PHASE 2: ADVANCED PREPROCESSING & MERGING
    
    logger.info("--- PHASE 2: PREPROCESSING & MERGING ---")
    
    df.columns = df.columns.str.strip()
    df_house.columns = df_house.columns.str.strip()
    
    df_weather_hourly['day'] = pd.to_datetime(df_weather_hourly['time']).dt.normalize()
    hourly_agg = df_weather_hourly.groupby('day').agg(
        temp_hr_std=('temperature', 'std'), 
        humidity_hr_mean=('humidity', 'mean')
    ).reset_index()
    
    df_weather_daily['day'] = pd.to_datetime(df_weather_daily['time']).dt.normalize()
    df_weather_master = pd.merge(df_weather_daily, hourly_agg, on='day', how='left')
    
    df_acorn['feature_name'] = df_acorn['MAIN CATEGORIES'].astype(str) + "_" + df_acorn['REFERENCE'].astype(str)
    df_acorn = df_acorn.drop_duplicates(subset=['feature_name'])
    
    df_acorn_t = df_acorn.set_index('feature_name').drop(columns=['MAIN CATEGORIES', 'CATEGORIES', 'REFERENCE']).T
    df_acorn_t.index.name = 'Acorn'
    df_acorn_t = df_acorn_t.reset_index()
    
    for col in df_acorn_t.columns:
        if col != 'Acorn':
            df_acorn_t[col] = pd.to_numeric(df_acorn_t[col], errors='coerce').fillna(0)
    
    df_house_enriched = pd.merge(df_house, df_acorn_t, on='Acorn', how='left')
    
    cols_to_encode = [col for col in ['Acorn', 'stdorToU'] if col in df_house_enriched.columns]
    df_house_encoded = pd.get_dummies(df_house_enriched, columns=cols_to_encode, drop_first=True)

    holiday_time_col = 'Bank holidays' if 'Bank holidays' in df_holidays.columns else df_holidays.columns[0]
    df_holidays['day'] = pd.to_datetime(df_holidays[holiday_time_col], errors='coerce').dt.normalize()
    df_holidays['is_holiday'] = 1

    if 'tpep_pickup_datetime' in df.columns:
        date_col = 'tpep_pickup_datetime'
    elif 'day' in df.columns:
        date_col = 'day'
    elif 'tstp' in df.columns:
        date_col = 'tstp'
    else:
        raise ValueError(f"No valid date column found. Available columns: {df.columns.tolist()}")

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df['day_formatted'] = df[date_col].dt.normalize()

    if 'energy_kwh' in df.columns:
        df['energy_kwh'] = pd.to_numeric(df['energy_kwh'], errors='coerce').fillna(0)
    elif 'energy(kWh/hh)' in df.columns:
        df['energy_kwh'] = pd.to_numeric(df['energy(kWh/hh)'], errors='coerce').fillna(0)
    else:
        # Check for half-hourly columns (hh_0 to hh_47)
        hh_cols = [col for col in df.columns if col.startswith('hh_')]
        if hh_cols:
            logger.info(f"Found {len(hh_cols)} half-hourly columns. Reshaping data...")
            # Melt wide format to long format
            id_cols = ['LCLid', date_col, 'day_formatted']
            df = df.melt(id_vars=id_cols, value_vars=hh_cols, var_name='hh_slot', value_name='energy_kwh')
            df['energy_kwh'] = pd.to_numeric(df['energy_kwh'], errors='coerce').fillna(0)
            # Extract hour from hh_slot (hh_0 to hh_47 -> 0 to 23, each hour has 2 slots)
            df['hh_num'] = df['hh_slot'].str.replace('hh_', '').astype(int)
            df['hour'] = df['hh_num'] // 2  # Convert half-hour slot to hour
            logger.info(f"Reshaped data to {len(df):,} rows")
        else:
            raise ValueError(f"No energy column found. Available columns: {df.columns.tolist()}")

    # PHASE 3: FEATURE ENGINEERING
    
    logger.info("--- PHASE 3: FEATURE ENGINEERING ---")
    
    # Calculate hour if not already computed from half-hourly data
    if 'hour' not in df.columns:
        df['hour'] = df[date_col].dt.hour
    df['is_peak'] = df['hour'].apply(lambda x: 1 if 17 <= x <= 21 else 0)

    # Calculate daily aggregates safely (avoiding lambda with outer scope reference)
    daily_basic = df.groupby(['LCLid', 'day_formatted']).agg(
        total_daily_kwh=('energy_kwh', 'sum'),
        daily_variance=('energy_kwh', 'var')
    ).reset_index()

    # Calculate peak and off-peak sums separately
    peak_data = df[df['is_peak'] == 1].groupby(['LCLid', 'day_formatted'])['energy_kwh'].sum().reset_index(name='peak_sum')
    off_peak_data = df[df['is_peak'] == 0].groupby(['LCLid', 'day_formatted'])['energy_kwh'].sum().reset_index(name='off_peak_sum')

    # Merge all daily features
    daily_features = daily_basic.merge(peak_data, on=['LCLid', 'day_formatted'], how='left')
    daily_features = daily_features.merge(off_peak_data, on=['LCLid', 'day_formatted'], how='left')
    daily_features['peak_sum'] = daily_features['peak_sum'].fillna(0)
    daily_features['off_peak_sum'] = daily_features['off_peak_sum'].fillna(0)

    daily_features.rename(columns={'day_formatted': 'day'}, inplace=True)

    daily_features['daily_variance'] = daily_features['daily_variance'].fillna(0)
    daily_features['peak_to_offpeak_ratio'] = daily_features['peak_sum'] / (daily_features['off_peak_sum'] + 0.001)

    daily_features = pd.merge(daily_features, df_weather_master[['day', 'temperatureMax', 'temp_hr_std', 'cloudCover']], on='day', how='left')
    daily_features = pd.merge(daily_features, df_holidays[['day', 'is_holiday']], on='day', how='left')
    
    daily_features['is_holiday'] = daily_features['is_holiday'].fillna(0).astype(int)
    daily_features.fillna(0, inplace=True) 

    # PHASE 4: UNSUPERVISED ANOMALY DETECTION
    
    logger.info("--- PHASE 4: UNSUPERVISED LEARNING (Isolation Forest) ---")
    ml_features = ['total_daily_kwh', 'daily_variance', 'peak_sum', 'off_peak_sum',
                   'peak_to_offpeak_ratio', 'temperatureMax', 'temp_hr_std', 'is_holiday']

    X_unsupervised = daily_features[ml_features]

    iso_forest = IsolationForest(contamination=CONTAMINATION_RATE, random_state=RANDOM_STATE)
    daily_features['anomaly_score'] = iso_forest.fit_predict(X_unsupervised)
    daily_features['is_anomaly'] = daily_features['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

    # PHASE 5: SUPERVISED TRAINING SETUP
    
    logger.info("--- PHASE 5: SUPERVISED TRAINING SETUP ---")
    
    # Group data by user
    agg_dict = {col: 'mean' for col in ml_features}
    agg_dict['is_anomaly'] = 'sum' 
    user_profiles = daily_features.groupby('LCLid').agg(agg_dict).reset_index()

    # Merge demographics, fill missing with 0
    user_profiles = pd.merge(user_profiles, df_house_encoded, on='LCLid', how='left')
    user_profiles.fillna(0, inplace=True) 

    # Label the top 5% anomalies as thieves
    anomaly_threshold = user_profiles['is_anomaly'].quantile(0.95)
    user_profiles['is_confirmed_thief'] = (user_profiles['is_anomaly'] >= anomaly_threshold).astype(int)

    # STRICTLY isolate the 8 behavioral features so the AI doesn't get confused by demographics
    X_supervised = user_profiles[ml_features]
    y_supervised = user_profiles['is_confirmed_thief']

    X_supervised.rename(columns=str, inplace=True)
    X_supervised.columns = pd.Index([str(c) for c in X_supervised.columns])

    # PHASE 6: SMOTE, EVALUATION & EXPORT

    logger.info("--- PHASE 6: RIGOROUS EVALUATION & SMOTE ---")

    # 1. Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_supervised, y_supervised, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_supervised
    )

    # 2. Export the untouched 20% test data for the Jupyter Notebook visualizations
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test_sample.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test_sample.csv"), index=False)
    logger.info("Exported unseen test data to CSV for Jupyter Notebook visualizations.")

    # 3. Apply SMOTE strictly to the Training Data to fix Class Imbalance
    logger.info(f"Original Training Class Balance: \n{y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    logger.info(f"New SMOTE Training Class Balance: \n{y_train_smote.value_counts().to_dict()}")

    # 4. Define the models for the head-to-head showdown
    models = {
        "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Gradient Boosting (Complex)": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Random Forest (Proposed Model)": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    }

    # 5. Train on SMOTE data and Evaluate on UNTOUCHED Test data
    for name, model in models.items():
        logger.info(f"[ Evaluating: {name} ]")
        model.fit(X_train_smote, y_train_smote)
        y_pred = model.predict(X_test)

        logger.info(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        logger.info("-" * 50)

    # 6. Final API Deployment Prep: Apply SMOTE to ALL data, then train
    logger.info("Retraining Proposed Model on 100% of SMOTE data for API deployment...")
    X_full_smote, y_full_smote = smote.fit_resample(X_supervised, y_supervised)

    final_model = models["Random Forest (Proposed Model)"]
    final_model.fit(X_full_smote, y_full_smote)

    # Save the final AI brain
    joblib.dump(final_model, os.path.join(BASE_DIR, "theft_detection_model.pkl"))
    joblib.dump(list(X_supervised.columns), os.path.join(BASE_DIR, "model_features.pkl"))

    logger.info("SUCCESS: SMOTE Pipeline Complete! Model saved for API.")

if __name__ == "__main__":
    run_pipeline()