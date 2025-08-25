import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import math
import io
import os 
import json 
class DistanceWeightedSelfAttention(Layer):
    def __init__(self, d_model, lambda_decay=0.5, **kwargs):
        super(DistanceWeightedSelfAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.lambda_decay = lambda_decay 
        self.query_dense = Dense(d_model, use_bias=False, name='query_dense')
        self.key_dense = Dense(d_model, use_bias=False, name='key_dense')
        self.value_dense = Dense(d_model, use_bias=False, name='value_dense')

    def build(self, input_shape):
        super(DistanceWeightedSelfAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        normalized_sizes = inputs[:, :, 0] 
        normalized_heights = inputs[:, :, 1] 
        q = self.query_dense(tf.expand_dims(normalized_heights, -1)) 
        k = self.key_dense(tf.expand_dims(normalized_heights, -1))  
        v = self.value_dense(tf.expand_dims(normalized_heights, -1))   
        matmul_qk = tf.matmul(q, k, transpose_b=True) 
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        size_distances = tf.abs(tf.expand_dims(normalized_sizes, -1) - tf.expand_dims(normalized_sizes, -2))
        distance_penalty = -self.lambda_decay * size_distances
        modified_attention_logits = scaled_attention_logits + distance_penalty
        if mask is not None:
            attention_mask = tf.cast(tf.not_equal(normalized_heights, 0), tf.float32) 
            query_mask = tf.expand_dims(attention_mask, axis=-1) 
            key_mask = tf.expand_dims(attention_mask, axis=1)    
            modified_attention_logits = modified_attention_logits + (1.0 - key_mask) * -1e9
            modified_attention_logits = modified_attention_logits + (1.0 - query_mask) * -1e9

        attention_weights = tf.nn.softmax(modified_attention_logits, axis=-1)
        if mask is not None:
            attention_weights = attention_weights * key_mask 
        output = tf.matmul(attention_weights, v) 
        return output 

    def get_config(self):
        config = super(DistanceWeightedSelfAttention, self).get_config()
        config.update({
            'd_model': self.d_model,
            'lambda_decay': self.lambda_decay,
        })
        return config

def parse_sample_file(sample_file_name):
    true_num_people = 0
    mixing_ratios = []
    m_val = 0.0
    ip_val = 0.0
    q_val = 0.0
    sec_val = 0.0
    sample_prefix = ''
    parts = sample_file_name.split('-')
    
    if len(parts) > 2:
        person_ids_str = parts[2]
        if person_ids_str:
            true_num_people = len(person_ids_str.split('_'))
    
    if len(parts) > 3:
        ratios_str = parts[3]
        try:
            raw_ratios = [int(r) for r in ratios_str.split(';')]
            total_ratio_sum = sum(raw_ratios)
            if total_ratio_sum > 0:
                mixing_ratios = [r / total_ratio_sum for r in raw_ratios]
            else:
                mixing_ratios = [0.0] * len(raw_ratios) 
        except ValueError:
            mixing_ratios = [1.0 / true_num_people] * true_num_people if true_num_people > 0 else []
        if len(mixing_ratios) != true_num_people:
            mixing_ratios = [1.0 / true_num_people] * true_num_people if true_num_people > 0 else []
        
        mixing_ratios.sort(reverse=True) 
    else:
        if true_num_people > 0:
            mixing_ratios = [1.0 / true_num_people] * true_num_people
        else:
            mixing_ratios = [] 
    m_val_match = re.search(r'M(\d+\.?\d*)(?:e|S)?', sample_file_name)
    if m_val_match:
        m_val = float(m_val_match.group(1))

    ip_val_match = re.search(r'(\d+\.?\d*)IP', sample_file_name)
    if ip_val_match:
        ip_val = float(ip_val_match.group(1))

    q_val_match = re.search(r'Q(\d+\.?\d*)', sample_file_name)
    if q_val_match:
        q_val = float(q_val_match.group(1))

    sec_val_match = re.search(r'(\d+\.?\d*)sec', sample_file_name)
    if sec_val_match:
        sec_val = float(sec_val_match.group(1))

    prefix_match = re.match(r'([A-Z]\d{2}_RD\d{2}-\d{4})', sample_file_name)
    if prefix_match:
        sample_prefix = prefix_match.group(1)

    return true_num_people, mixing_ratios, m_val, ip_val, q_val, sec_val, sample_prefix


def preprocess_data(df):
    print("\nDEBUG: 开始对 'Allele' 列进行数据清洗 (将非数字值转换为 NaN)...")
    allele_columns = [col for col in df.columns if 'Allele' in col]

    if not allele_columns:
        print("DEBUG: 未找到任何以 'Allele' 命名的列。")
    else:
        print(f"DEBUG: 识别到 {len(allele_columns)} 个 'Allele' 相关列。")
        for col in allele_columns:
            if df[col].dtype == 'object':
                non_numeric_mask = df[col].astype(str).str.contains(r'[^0-9\.]', na=False)
                if non_numeric_mask.any():
                    print(f"DEBUG: 列 '{col}' 包含非数字值 (例如 'OL')，正在转换为数值类型...")
            df[col] = pd.to_numeric(df[col], errors='coerce')
        print("DEBUG: 'Allele' 列数据清洗完成。")
    df[['true_num_people', 'true_mixing_ratios', 'M_Value', 'IP_Value', 'Q_Value', 'Sec_Value', 'Sample_Prefix']] = df['Sample File'].apply(lambda x: pd.Series(parse_sample_file(x)))

    print("\nDEBUG: 检查 'true_num_people' 列的数据类型和值...")
    print(f"DEBUG: 'true_num_people' 转换前的数据类型: {df['true_num_people'].dtype}")
    df['true_num_people'] = pd.to_numeric(df['true_num_people'], errors='coerce')
    df['true_num_people'] = df['true_num_people'].fillna(0).astype(int) 
    print(f"DEBUG: 'true_num_people' 转换后填充后的数据类型: {df['true_num_people'].dtype}")
    print(f"DEBUG: 'true_num_people' 列值计数 (转换后):")
    print(df['true_num_people'].value_counts())
    print("DEBUG: 'true_num_people' 列前5行 (转换后):")
    print(df[['Sample File', 'true_num_people']].head())

    MAX_PEOPLE = df['true_num_people'].max()
    if MAX_PEOPLE == 0:
        MAX_PEOPLE = 1

    markers = df['Marker'].unique()
    
    max_alleles = 0
    for col in df.columns:
        match_space = re.match(r'Allele (\d+)', col)
        match_no_space = re.match(r'Allele(\d+)', col)
        if match_space:
            num = int(match_space.group(1))
            if num > max_alleles:
                max_alleles = num
        elif match_no_space:
            num = int(match_no_space.group(1))
            if num > max_alleles:
                max_alleles = num

    processed_data_list = []
    for index, row in df.iterrows():
        allele_height_pairs = []
        for i in range(1, max_alleles + 1):
            allele_col_space = f'Allele {i}'
            size_col_space = f'Size {i}'
            height_col_space = f'Height {i}'
            
            allele_col_no_space = f'Allele{i}'
            size_col_no_space = f'Size{i}'
            height_col_no_space = f'Height{i}'

            allele = None
            size = None
            height = None

            if allele_col_space in row and size_col_space in row and height_col_space in row:
                allele = row.get(allele_col_space)
                size = row.get(size_col_space)
                height = row.get(height_col_space)
            elif allele_col_no_space in row and size_col_no_space in row and height_col_no_space in row:
                allele = row.get(allele_col_no_space)
                size = row.get(size_col_no_space)
                height = row.get(height_col_no_space)

            if pd.notna(allele) and pd.notna(size) and pd.notna(height):
                try:
                    allele_height_pairs.append({'Allele': float(allele), 'Size': float(size), 'Height': float(height)})
                except ValueError:
                    continue

        allele_height_pairs.sort(key=lambda x: x['Size'])

        processed_data_list.append({
            'Sample File': row['Sample File'],
            'Marker': row['Marker'],
            'Dye': row['Dye'],
            'Allele_Height_Pairs': allele_height_pairs,
            'true_num_people': row['true_num_people'], 
            'true_mixing_ratios': row['true_mixing_ratios'],
            'M_Value': row['M_Value'],
            'IP_Value': row['IP_Value'],
            'Q_Value': row['Q_Value'],
            'Sec_Value': row['Sec_Value'],
            'Sample_Prefix': row['Sample_Prefix']
        })

    processed_df = pd.DataFrame(processed_data_list)
    print("DEBUG: processed_df columns after initial processing:", processed_df.columns)
    print("DEBUG: processed_df['true_num_people'] value_counts():")
    print(processed_df['true_num_people'].value_counts())
    print("DEBUG: processed_df[['Sample File', 'true_num_people']] head():")
    print(processed_df[['Sample File', 'true_num_people']].head())
    print("DEBUG: processed_df[['Sample File', 'true_num_people']] tail():")
    print(processed_df[['Sample File', 'true_num_people']].tail())


    all_sizes = []
    all_heights = []
    for pairs_list in processed_df['Allele_Height_Pairs']:
        for pair in pairs_list:
            all_sizes.append(pair['Size'])
            all_heights.append(pair['Height'])

    size_scaler = MinMaxScaler()
    height_scaler = MinMaxScaler()
    if all_sizes:
        size_scaler.fit(np.array(all_sizes).reshape(-1, 1))
    if all_heights:
        height_scaler.fit(np.array(all_heights).reshape(-1, 1))

    max_sequence_length = processed_df['Allele_Height_Pairs'].apply(len).max()
    if max_sequence_length == 0:
        max_sequence_length = 1

    sequence_inputs = []
    aux_features_list = []
    for index, row in processed_df.iterrows():
        current_heights = [p['Height'] for p in row['Allele_Height_Pairs']]
        height_threshold = np.percentile(current_heights, 10) if current_heights else 0
        height_threshold = max(height_threshold, 50)

        effective_peaks = [p for p in row['Allele_Height_Pairs'] if p['Height'] >= height_threshold]
        if not effective_peaks and row['Allele_Height_Pairs']:
            effective_peaks = row['Allele_Height_Pairs'] 
        elif not effective_peaks:
            effective_peaks = [{'Allele': 0.0, 'Size': 0.0, 'Height': 0.0}] 

        below_threshold_peaks = [p for p in row['Allele_Height_Pairs'] if p['Height'] < height_threshold]

        sequence_input_data = []
        for p in effective_peaks:
            norm_size = size_scaler.transform([[p['Size']]])[0][0] if all_sizes else 0.0
            norm_height = height_scaler.transform([[p['Height']]])[0][0] if all_heights else 0.0
            sequence_input_data.append([norm_size, norm_height])

        padded_sequence = sequence_input_data + [[0.0, 0.0]] * (max_sequence_length - len(sequence_input_data))
        padded_sequence = np.array(padded_sequence, dtype=np.float32)
        sequence_inputs.append(padded_sequence)

        aux_features = {}
        aux_features['num_effective_alleles'] = len(effective_peaks)
        aux_features['total_height_effective'] = sum(p['Height'] for p in effective_peaks)
        aux_features['mean_height_effective'] = np.mean([p['Height'] for p in effective_peaks]) if effective_peaks else 0.0
        aux_features['median_height_effective'] = np.median([p['Height'] for p in effective_peaks]) if effective_peaks else 0.0
        aux_features['max_height_effective'] = np.max([p['Height'] for p in effective_peaks]) if effective_peaks else 0.0
        aux_features['min_height_effective'] = np.min([p['Height'] for p in effective_peaks]) if effective_peaks else 0.0
        aux_features['height_std_dev_effective'] = np.std([p['Height'] for p in effective_peaks]) if effective_peaks else 0.0
        aux_features['min_to_max_height_ratio_effective'] = (aux_features['min_height_effective'] / aux_features['max_height_effective']) if aux_features['max_height_effective'] > 0 else 0.0

        aux_features['num_peaks_below_threshold'] = len(below_threshold_peaks)
        aux_features['total_height_below_threshold'] = sum(p['Height'] for p in below_threshold_peaks)
        aux_features['ratio_above_to_below_peaks'] = (aux_features['num_effective_alleles'] / (aux_features['num_peaks_below_threshold'] + 1e-6))

        aux_features['M_Value'] = row['M_Value']
        aux_features['IP_Value'] = row['IP_Value']
        aux_features['Q_Value'] = row['Q_Value']
        aux_features['Sec_Value'] = row['Sec_Value']
        aux_features['Sample_Prefix'] = row['Sample_Prefix']
        aux_features_list.append(aux_features)

    features_df = processed_df[['Sample File', 'Marker', 'true_num_people', 'true_mixing_ratios']].copy()
    features_df['sequence_input'] = sequence_inputs
    features_df['aux_features'] = aux_features_list
    print("DEBUG: features_df columns after adding sequence and aux_features:", features_df.columns)
    print("DEBUG: features_df['true_num_people'] value_counts() after adding sequence and aux_features:")
    print(features_df['true_num_people'].value_counts())
    print("DEBUG: features_df[['Sample File', 'true_num_people']] head() after adding sequence and aux_features:")
    print(features_df[['Sample File', 'true_num_people']].head())
    print("DEBUG: features_df[['Sample File', 'true_num_people']] tail() after adding sequence and aux_features:")
    print(features_df[['Sample File', 'true_num_people']].tail())

    unique_prefixes = features_df['aux_features'].apply(lambda x: x['Sample_Prefix']).unique()
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(unique_prefixes.reshape(-1, 1)) 
    
    prefix_encoded = ohe.transform(features_df['aux_features'].apply(lambda x: [x['Sample_Prefix']]).tolist())
    prefix_df = pd.DataFrame(prefix_encoded, columns=ohe.get_feature_names_out(['Sample_Prefix']), index=features_df.index)
    
    aux_features_df = pd.json_normalize(features_df['aux_features'])
    aux_features_df = aux_features_df.drop(columns=['Sample_Prefix'])
    aux_features_df = pd.concat([aux_features_df, prefix_df], axis=1) 

    numerical_aux_cols = aux_features_df.columns.drop(prefix_df.columns)
    aux_scaler = MinMaxScaler()
    if not numerical_aux_cols.empty:
        aux_features_df[numerical_aux_cols] = aux_scaler.fit_transform(aux_features_df[numerical_aux_cols])
    
    features_df['aux_features_processed'] = aux_features_df.apply(lambda x: x.values.astype(np.float32), axis=1)

    print("DEBUG: features_df columns after adding aux_features_processed:", features_df.columns)
    print("DEBUG: features_df['true_num_people'] value_counts() after adding aux_features_processed:")
    print(features_df['true_num_people'].value_counts())
    print("DEBUG: features_df[['Sample File', 'true_num_people']] head() after adding aux_features_processed:")
    print(features_df[['Sample File', 'true_num_people']].head())
    print("DEBUG: features_df[['Sample File', 'true_num_people']] tail() after adding aux_features_processed:")
    print(features_df[['Sample File', 'true_num_people']].tail())

    return features_df, max_sequence_length, markers, MAX_PEOPLE

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    elif isinstance(obj, (np.float32, np.float64)): 
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)): 
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist() 
    else:
        return obj


def train_people_models(features_df, max_sequence_length, markers):
    """
    Trains individual models for each marker to predict the number of people (via allele count)
    and saves them along with evaluation metrics on the training set.
    """
    print("DEBUG: train_people_models received features_df columns:", features_df.columns)
    print("DEBUG: train_people_models received features_df['true_num_people'] value_counts():")
    print(features_df['true_num_people'].value_counts())

    trained_people_models = {}
    people_model_train_metrics = {} 
    models_dir = "trained_people_models"
    os.makedirs(models_dir, exist_ok=True)
    for marker in markers:
        sanitized_marker = marker.replace(' ', '_').replace('.', '')
        
        marker_data_for_training = features_df[features_df['Marker'] == marker]
        
        if marker_data_for_training.empty:
            print(f"Skipping marker {marker}: No training data for this marker for people prediction.")
            continue
        X_train_seq = np.stack(marker_data_for_training['sequence_input'].values)
        X_train_aux = np.stack(marker_data_for_training['aux_features_processed'].values)
        y_train_allele_count = (marker_data_for_training['true_num_people'] * 2).values.astype(np.float32)
        print(f"DEBUG: Marker {marker} - y_train_allele_count (first 10 values): {y_train_allele_count[:10]}")
        print(f"DEBUG: Marker {marker} - y_train_allele_count dtype: {y_train_allele_count.dtype}")
        print(f"DEBUG: Marker {marker} - y_train_allele_count shape: {y_train_allele_count.shape}")

        if X_train_seq.shape[1] == 0:
            print(f"Skipping marker {marker}: Sequence input dimension is 0. Check data processing.")
            continue
        peak_input = Input(shape=(max_sequence_length, 2), name=f'{sanitized_marker}_peak_input')
        attention_output = DistanceWeightedSelfAttention(d_model=256, lambda_decay=0.5)(peak_input) 
        peak_features = tf.keras.layers.GlobalAveragePooling1D()(attention_output)
        peak_features = Dense(512, activation='relu')(peak_features)
        peak_features = Dropout(0.2)(peak_features)
        peak_features = Dense(256, activation='relu')(peak_features) 
        peak_features = Dropout(0.2)(peak_features)
        peak_features = Dense(128, activation='relu')(peak_features)

        aux_input = Input(shape=(X_train_aux.shape[1],), name=f'{sanitized_marker}_aux_input')
        aux_features_processed = Dense(512, activation='relu')(aux_input)
        aux_features_processed = Dropout(0.2)(aux_features_processed)
        aux_features_processed = Dense(256, activation='relu')(aux_features_processed)
        aux_features_processed = Dropout(0.2)(aux_features_processed)
        aux_features_processed = Dense(128, activation='relu')(aux_features_processed)

        merged_features = Concatenate()([peak_features, aux_features_processed])

        x = Dense(256, activation='relu')(merged_features)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x) 
        x = Dropout(0.1)(x)
        shared_features = Dense(64, activation='relu')(x)

        allele_count_output = Dense(1, activation='linear', name=f'{sanitized_marker}_allele_count_output')(shared_features)
        model = Model(inputs=[peak_input, aux_input], outputs=[allele_count_output])
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss={f'{sanitized_marker}_allele_count_output': 'mae'})
        early_stopping = EarlyStopping(monitor='loss', patience=150, restore_best_weights=True)
        print(f"\n--- Training People Model for Marker: {marker} ---")
        history = model.fit(
            {f'{sanitized_marker}_peak_input': X_train_seq, f'{sanitized_marker}_aux_input': X_train_aux},
            y_train_allele_count, 
            epochs=1000,
            batch_size=32,
            callbacks=[early_stopping], 
            verbose=0
        )
        
        trained_people_models[marker] = model
        y_pred_allele_count_train = model.predict(
            {f'{sanitized_marker}_peak_input': X_train_seq, f'{sanitized_marker}_aux_input': X_train_aux}, verbose=0
        ).flatten()
        
        y_pred_allele_count_train_rounded = np.round(y_pred_allele_count_train)
        y_pred_allele_count_train_rounded = np.maximum(2, y_pred_allele_count_train_rounded) 

        mae_ac = mean_absolute_error(y_train_allele_count, y_pred_allele_count_train)
        rmse_ac = np.sqrt(mean_squared_error(y_train_allele_count, y_pred_allele_count_train))
        accuracy_ac = accuracy_score(y_train_allele_count, y_pred_allele_count_train_rounded) 

        people_model_train_metrics[marker] = {
            'mae_allele_count': mae_ac,
            'rmse_allele_count': rmse_ac,
            'accuracy_allele_count': accuracy_ac
        }
        print(f"Finished training People Model for Marker: {marker}. Training Metrics (Allele Count):")
        print(f"  MAE: {mae_ac:.4f}, RMSE: {rmse_ac:.4f}, Accuracy: {accuracy_ac:.4f}")

        model_save_path = os.path.join(models_dir, f'people_model_{sanitized_marker}.keras')
        model.save(model_save_path)
        print(f"People Model for marker {marker} saved to {model_save_path}")

    serializable_metrics = convert_numpy_types(people_model_train_metrics)
    metrics_save_path = os.path.join(models_dir, 'people_model_train_metrics.json')
    with open(metrics_save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    print(f"People Model training metrics saved to {metrics_save_path}")
    return trained_people_models, people_model_train_metrics, models_dir

def predict_people_and_save(features_df, markers, models_dir_people):
    print("DEBUG: predict_people_and_save received features_df columns:", features_df.columns)
    print("DEBUG: predict_people_and_save received features_df['true_num_people'] value_counts():")
    print(features_df['true_num_people'].value_counts())
    trained_people_models = {}
    people_model_train_metrics = {}
    metrics_load_path = os.path.join(models_dir_people, 'people_model_train_metrics.json')
    if os.path.exists(metrics_load_path):
        with open(metrics_load_path, 'r') as f:
            people_model_train_metrics = json.load(f)
        print(f"Loaded people model training metrics from {metrics_load_path}")
        print("DEBUG: Loaded people_model_train_metrics content:", people_model_train_metrics)
    else:
        print(f"Error: People model metrics file not found at {metrics_load_path}. Cannot perform weighted ensemble or per-marker evaluation.")
        return pd.DataFrame(), "Prediction failed: Metrics file not found."

    for marker in markers:
        sanitized_marker = marker.replace(' ', '_').replace('.', '')
        model_load_path = os.path.join(models_dir_people, f'people_model_{sanitized_marker}.keras')
        if os.path.exists(model_load_path):
            try:
                model = load_model(model_load_path, custom_objects={
                    'DistanceWeightedSelfAttention': DistanceWeightedSelfAttention
                })
                trained_people_models[marker] = model
                print(f"Loaded people model for marker: {marker}")
            except Exception as e:
                print(f"Error loading people model for marker {marker} from {model_load_path}: {e}")
        else:
            print(f"Warning: People model file not found for marker {marker} at {model_load_path}. Skipping this marker.")

    print("DEBUG: Loaded trained_people_models keys:", trained_people_models.keys())

    if not trained_people_models:
        return pd.DataFrame(), "No people models were loaded. Prediction cannot be performed."

    per_marker_allele_count_eval_results = []
    print("\n--- Evaluating Allele Count for Each Marker Model on the Entire Dataset ---")
    for marker in markers:
        marker_data = features_df[features_df['Marker'] == marker]
        if marker_data.empty or marker not in trained_people_models:
            print(f"Skipping evaluation for marker {marker}: No data or model not loaded.")
            continue
        
        model = trained_people_models[marker]
        sanitized_marker = marker.replace(' ', '_').replace('.', '')

        X_seq_marker = np.stack(marker_data['sequence_input'].values)
        X_aux_marker = np.stack(marker_data['aux_features_processed'].values)
        y_true_allele_count_marker = (marker_data['true_num_people'] * 2).values.astype(np.float32)

        if X_seq_marker.size == 0 or X_aux_marker.size == 0:
            print(f"Warning: Empty input data for marker {marker}. Skipping allele count evaluation.")
            continue

        y_pred_allele_count_marker = model.predict(
            {f'{sanitized_marker}_peak_input': X_seq_marker, f'{sanitized_marker}_aux_input': X_aux_marker}, verbose=0
        ).flatten()
        y_pred_allele_count_marker_rounded = np.round(y_pred_allele_count_marker)
        y_pred_allele_count_marker_rounded = np.maximum(2, y_pred_allele_count_marker_rounded)

        mae_ac = mean_absolute_error(y_true_allele_count_marker, y_pred_allele_count_marker)
        rmse_ac = np.sqrt(mean_squared_error(y_true_allele_count_marker, y_pred_allele_count_marker))
        accuracy_ac = accuracy_score(y_true_allele_count_marker, y_pred_allele_count_marker_rounded)

        per_marker_allele_count_eval_results.append({
            'Marker': marker,
            'MAE_Allele_Count': mae_ac,
            'RMSE_Allele_Count': rmse_ac,
            'Accuracy_Allele_Count': accuracy_ac
        })
        print(f"  Marker {marker} - Allele Count Metrics: MAE={mae_ac:.4f}, RMSE={rmse_ac:.4f}, Accuracy={accuracy_ac:.4f}")

    people_prediction_results = []
    unique_sample_files_full = features_df['Sample File'].unique()

    print("\n--- DEBUG: Verifying true_num_people from features_df before processing each sample (in predict_people_and_save) ---")
    for sample_file_debug in unique_sample_files_full:
        sample_df_debug = features_df[features_df['Sample File'] == sample_file_debug]
        if not sample_df_debug.empty:
            debug_true_num_people = sample_df_debug['true_num_people'].iloc[0]
            print(f"DEBUG: Sample: {sample_file_debug}, true_num_people (from features_df): {debug_true_num_people}")


    print("\n--- Predicting Number of People and Saving to CSV ---")
    for sample_file in unique_sample_files_full:
        sample_df = features_df[features_df['Sample File'] == sample_file]
        print(f"DEBUG: Processing sample file: {sample_file}")
        print(f"DEBUG: sample_df columns for {sample_file}: {sample_df.columns}")

        current_sample_marker_predictions_allele_count = []
        if 'true_num_people' in sample_df.columns and not sample_df.empty:
            true_label_num_people = sample_df['true_num_people'].iloc[0]
            print(f"DEBUG: Retrieved true_num_people for {sample_file}: {true_label_num_people}")
        else:
            true_label_num_people = -1 
            print(f"ERROR: 'true_num_people' column missing or sample_df is empty for {sample_file}. Setting to -1.")


        for index, row in sample_df.iterrows():
            if 'Marker' not in row.index:
                print(f"ERROR: 'Marker' key not found in row for sample {sample_file}, index {index}. Skipping this row.")
                continue

            marker = row['Marker']
            sanitized_marker = marker.replace(' ', '_').replace('.', '')
            
            if marker in trained_people_models:
                model = trained_people_models[marker]
                seq_input = np.expand_dims(row['sequence_input'], axis=0)
                aux_input = np.expand_dims(row['aux_features_processed'], axis=0)
                
                if seq_input.size > 0 and aux_input.size > 0:
                    pred_allele_count = model.predict(
                        {f'{sanitized_marker}_peak_input': seq_input, f'{sanitized_marker}_aux_input': aux_input}, verbose=0
                    )
                    marker_mae_for_weight = people_model_train_metrics.get(marker, {}).get('mae_allele_count', float('inf')) 

                    current_sample_marker_predictions_allele_count.append({
                        'marker': marker, 
                        'prediction': pred_allele_count[0][0], 
                        'mae': marker_mae_for_weight
                    })
                else:
                    print(f"Warning: Empty input for {sample_file} marker {marker}. Skipping allele count prediction for this marker.")
            else:
                print(f"Warning: No trained model found for marker {marker}. Skipping prediction for this marker in sample {sample_file}.")
        
        final_pred_allele_count = 0
        final_pred_num_people = 0

        if current_sample_marker_predictions_allele_count:
            raw_weights_ac = [1.0 / (item['mae'] + 1e-6) for item in current_sample_marker_predictions_allele_count]
            total_raw_weight_ac = sum(raw_weights_ac)

            if total_raw_weight_ac > 0:
                normalized_weights_ac = [w / total_raw_weight_ac for w in raw_weights_ac]
                weighted_sum_ac = sum(item['prediction'] * norm_w for item, norm_w in zip(current_sample_marker_predictions_allele_count, normalized_weights_ac))
                
                final_pred_allele_count = round(weighted_sum_ac)
                final_pred_allele_count = max(2, final_pred_allele_count)
                final_pred_num_people = math.ceil(final_pred_allele_count / 2)
            else:
                if current_sample_marker_predictions_allele_count:
                    avg_pred_ac = np.mean([item['prediction'] for item in current_sample_marker_predictions_allele_count])
                    final_pred_allele_count = round(avg_pred_ac)
                    final_pred_allele_count = max(2, final_pred_allele_count)
                    final_pred_num_people = math.ceil(final_pred_allele_count / 2)
        print(f"DEBUG: Appending for {sample_file}: True People = {true_label_num_people}, Predicted People = {final_pred_num_people}")


        people_prediction_results.append({
            'Sample File': sample_file,
            'True Number of People': true_label_num_people,
            'Predicted Number of People': final_pred_num_people,
            'Predicted Allele Count': final_pred_allele_count, 
            'Marker Allele Predictions': [{'marker': p['marker'], 'predicted_allele_count': p['prediction']} for p in current_sample_marker_predictions_allele_count] 
        })

    people_df = pd.DataFrame(people_prediction_results)
    print("\nDEBUG: people_df head after creation:")
    print(people_df.head())
    print("\nDEBUG: people_df tail after creation:")
    print(people_df.tail())

    output_csv_path = "predicted_num_people.csv"
    people_df.to_csv(output_csv_path, index=False)
    print(f"\nPredicted number of people and allele count saved to {output_csv_path}")
    true_labels_num_people_overall = people_df['True Number of People'].tolist()
    final_predictions_num_people_overall = people_df['Predicted Number of People'].tolist()

    mae_num_people = mean_absolute_error(true_labels_num_people_overall, final_predictions_num_people_overall)
    rmse_num_people = np.sqrt(mean_squared_error(true_labels_num_people_overall, final_predictions_num_people_overall))
    accuracy_num_people = accuracy_score(true_labels_num_people_overall, final_predictions_num_people_overall)
    
    try:
        conf_mat_num_people = confusion_matrix(true_labels_num_people_overall, final_predictions_num_people_overall)
    except ValueError as e:
        conf_mat_num_people = f"无法生成混淆矩阵 (人数): {e}. 可能是因为预测或真实标签为空，或类别不一致。"

    sample_prediction_output_details = "\n#### 每个样本的真实值和预测值 (包含每个标记的等位基因数预测):\n"
    for sample_res in people_prediction_results:
        sample_prediction_output_details += f"- 样本文件: {sample_res['Sample File']}\n"
        sample_prediction_output_details += f"  真实等位基因数 (总计): {sample_res['True Number of People'] * 2}, 预测等位基因数 (集成): {sample_res['Predicted Allele Count']}\n"
        sample_prediction_output_details += f"  真实人数: {sample_res['True Number of People']}, 预测人数: {sample_res['Predicted Number of People']}\n"
        if sample_res['Marker Allele Predictions']:
            sample_prediction_output_details += "  各标记等位基因数预测:\n"
            for marker_pred in sample_res['Marker Allele Predictions']:
                sample_prediction_output_details += f"    - {marker_pred['marker']}: {marker_pred['predicted_allele_count']:.2f}\n"
        else:
            sample_prediction_output_details += "  无此样本的标记等位基因数预测。\n"


    per_marker_summary = "\n### 阶段一：等位基因数预测 - 每个标记模型的评估结果 (在整个数据集上):\n"
    if per_marker_allele_count_eval_results:
        for res in per_marker_allele_count_eval_results:
            per_marker_summary += f"- **标记: {res['Marker']}**\n"
            per_marker_summary += f"  MAE (等位基因数): {res['MAE_Allele_Count']:.4f}\n"
            per_marker_summary += f"  RMSE (等位基因数): {res['RMSE_Allele_Count']:.4f}\n"
            per_marker_summary += f"  准确率 (等位基因数): {res['Accuracy_Allele_Count']:.4f}\n"
    else:
        per_marker_summary += "  无每个标记的等位基因数评估结果。\n"


    people_results_summary = f"""
    {per_marker_summary}

   
{conf_mat_num_people}
    ```
    {sample_prediction_output_details}
    """
    return people_df, people_results_summary


if __name__ == "__main__":
    csv_file_path = "附件1：不同人数的STR图谱数据noOL.csv" 
    print(f"--- 正在处理文件: {csv_file_path} ---")
    
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"错误: 文件 '{csv_file_path}' 未找到。请确保文件路径和文件名正确。")
        csv_file_path = "附件2：不同混合比例的STR图谱数据noOL - 副本.csv.txt"
        print(f"尝试加载另一个文件: {csv_file_path}")
        try:
            df = pd.read_csv(csv_file_path)
            print(f"成功加载文件: {csv_file_path}")
        except FileNotFoundError:
            print(f"错误: 文件 '{csv_file_path}' 也未找到。请检查所有提供的文件路径和文件名。")
            exit()
        except Exception as e:
            print(f"加载文件时发生错误: {e}")
            exit()
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        exit()

    print("DEBUG: Initial DataFrame columns after reading CSV:", df.columns)

    features_df, max_sequence_length, markers, MAX_PEOPLE = preprocess_data(df.copy())
    trained_people_models, people_model_train_metrics, people_models_dir = train_people_models(features_df.copy(), max_sequence_length, markers)
    predicted_people_df, people_eval_summary = predict_people_and_save(features_df.copy(), markers, people_models_dir)
    print(people_eval_summary)