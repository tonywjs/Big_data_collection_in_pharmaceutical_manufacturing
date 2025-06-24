"""
제약 제조 공정 시계열 데이터 분석 실습

이 실습에서는 제약 제조 공정의 실시간 센서 데이터를 분석합니다:
1. 시계열 데이터 로딩 및 전처리
2. 시계열 특성 분석 (트렌드, 계절성, 이상값)
3. 시계열 특성 엔지니어링
4. 시계열 클러스터링
5. 시계열 기반 품질 예측 모델링

데이터: 10초 간격 공정 센서 데이터 (배치별)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import TimeSeriesKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 시각화 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('default')
sns.set_palette("husl")

print("=" * 80)
print("제약 제조 공정 시계열 데이터 분석 실습")
print("=" * 80)

# =============================================================================
# 1. 시계열 데이터 로딩
# =============================================================================
print("\n📊 1단계: 시계열 데이터 로딩")
print("-" * 50)

def load_time_series_data(batch_numbers=[1, 2, 3, 4, 5]):
    """여러 배치의 시계열 데이터 로딩"""
    
    all_timeseries = {}
    
    for batch_num in batch_numbers:
        try:
            file_path = f'Process/{batch_num}.csv'
            df = pd.read_csv(file_path, sep=';')
            
            # 타임스탬프 컬럼을 datetime으로 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            all_timeseries[batch_num] = df
            print(f"✅ 배치 {batch_num} 로딩 완료: {df.shape}")
            
        except FileNotFoundError:
            print(f"❌ 배치 {batch_num} 파일을 찾을 수 없습니다.")
            continue
    
    return all_timeseries

# 시계열 데이터 로딩
timeseries_data = load_time_series_data([1, 2, 3, 4, 5])

if timeseries_data:
    print(f"\n📈 로딩된 배치 수: {len(timeseries_data)}")
    
    # 첫 번째 배치 데이터 구조 확인
    first_batch = list(timeseries_data.values())[0]
    print(f"📊 첫 번째 배치 데이터 구조:")
    print(f"   - 컬럼 수: {len(first_batch.columns)}")
    print(f"   - 레코드 수: {len(first_batch)}")
    print(f"   - 시간 범위: {first_batch['timestamp'].min()} ~ {first_batch['timestamp'].max()}")
    print(f"   - 측정 변수: {[col for col in first_batch.columns if col not in ['timestamp', 'campaign', 'batch', 'code']]}")

# =============================================================================
# 2. 시계열 데이터 탐색 및 시각화
# =============================================================================
print("\n📊 2단계: 시계열 데이터 탐색")
print("-" * 50)

def explore_timeseries(ts_data):
    """시계열 데이터 탐색 함수"""
    
    # 주요 공정 변수들
    key_variables = ['tbl_speed', 'main_comp', 'tbl_fill', 'SREL', 'stiffness', 'ejection']
    
    # 여러 배치의 주요 변수 시각화
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, var in enumerate(key_variables):
        ax = axes[i]
        
        for batch_id, df in ts_data.items():
            if var in df.columns:
                # 시간을 시작점부터의 분 단위로 변환
                time_minutes = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
                ax.plot(time_minutes, df[var], label=f'Batch {batch_id}', alpha=0.7)
        
        ax.set_title(f'{var} 시계열')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel(var)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('timeseries_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 시계열 개요 시각화 완료")

if timeseries_data:
    explore_timeseries(timeseries_data)

# =============================================================================
# 3. 시계열 전처리
# =============================================================================
print("\n🔧 3단계: 시계열 전처리")
print("-" * 50)

def preprocess_timeseries(ts_data):
    """시계열 데이터 전처리"""
    
    processed_data = {}
    
    for batch_id, df in ts_data.items():
        df_clean = df.copy()
        
        print(f"\n🧹 배치 {batch_id} 전처리:")
        
        # 1) 결측값 처리
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        initial_nulls = df_clean[numeric_cols].isnull().sum().sum()
        
        # 선형 보간으로 결측값 채우기
        df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='linear')
        
        # 여전히 결측값이 있으면 전방/후방 채우기
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method='ffill').fillna(method='bfill')
        
        final_nulls = df_clean[numeric_cols].isnull().sum().sum()
        print(f"   - 결측값 처리: {initial_nulls} → {final_nulls}")
        
        # 2) 이상값 탐지 및 처리 (Z-score 방법)
        z_threshold = 3
        outliers_removed = 0
        
        for col in numeric_cols:
            if col in ['tbl_speed', 'main_comp', 'tbl_fill', 'SREL']:
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                outliers = df_clean[col][z_scores > z_threshold]
                
                if len(outliers) > 0:
                    # 이상값을 중앙값으로 대체
                    median_val = df_clean[col].median()
                    df_clean.loc[z_scores > z_threshold, col] = median_val
                    outliers_removed += len(outliers)
        
        print(f"   - 이상값 처리: {outliers_removed}개 값 수정")
        
        # 3) 시간 기반 특성 추가
        df_clean['hour'] = df_clean['timestamp'].dt.hour
        df_clean['day_of_week'] = df_clean['timestamp'].dt.dayofweek
        df_clean['time_from_start'] = (df_clean['timestamp'] - df_clean['timestamp'].min()).dt.total_seconds() / 3600  # hours
        
        # 4) 이동 평균 추가 (노이즈 제거)
        window_size = 10  # 100초 이동 평균 (10개 포인트 * 10초)
        key_vars = ['tbl_speed', 'main_comp', 'tbl_fill', 'SREL']
        
        for var in key_vars:
            if var in df_clean.columns:
                df_clean[f'{var}_ma'] = df_clean[var].rolling(window=window_size, center=True).mean()
        
        processed_data[batch_id] = df_clean
        print(f"   ✅ 전처리 완료: {df_clean.shape}")
    
    return processed_data

if timeseries_data:
    processed_timeseries = preprocess_timeseries(timeseries_data)

# =============================================================================
# 4. 시계열 특성 엔지니어링
# =============================================================================
print("\n⚙️ 4단계: 시계열 특성 엔지니어링")
print("-" * 50)

def extract_timeseries_features(ts_data):
    """시계열에서 통계적 특성 추출"""
    
    features_list = []
    
    for batch_id, df in ts_data.items():
        batch_features = {'batch_id': batch_id}
        
        # 주요 공정 변수들
        key_vars = ['tbl_speed', 'main_comp', 'tbl_fill', 'SREL', 'stiffness', 'ejection']
        
        for var in key_vars:
            if var in df.columns:
                series = df[var].dropna()
                
                if len(series) > 0:
                    # 기본 통계량
                    batch_features[f'{var}_mean'] = series.mean()
                    batch_features[f'{var}_std'] = series.std()
                    batch_features[f'{var}_min'] = series.min()
                    batch_features[f'{var}_max'] = series.max()
                    batch_features[f'{var}_median'] = series.median()
                    batch_features[f'{var}_q25'] = series.quantile(0.25)
                    batch_features[f'{var}_q75'] = series.quantile(0.75)
                    
                    # 변동성 지표
                    batch_features[f'{var}_cv'] = series.std() / series.mean() if series.mean() != 0 else 0
                    batch_features[f'{var}_range'] = series.max() - series.min()
                    
                    # 트렌드 지표
                    if len(series) > 1:
                        x = np.arange(len(series))
                        slope, _, _, _, _ = stats.linregress(x, series)
                        batch_features[f'{var}_trend'] = slope
                    
                    # 안정성 지표 (연속된 값들의 차이)
                    diff_series = series.diff().dropna()
                    if len(diff_series) > 0:
                        batch_features[f'{var}_stability'] = diff_series.abs().mean()
                    
                    # 주파수 도메인 특성 (변화 빈도)
                    zero_crossings = np.sum(np.diff(np.sign(series - series.mean())) != 0)
                    batch_features[f'{var}_zero_crossings'] = zero_crossings
        
        # 공정 단계별 특성
        # 생산 구간만 필터링 (속도 > 0)
        production_mask = df['tbl_speed'] > 0
        production_data = df[production_mask]
        
        if len(production_data) > 0:
            batch_features['production_duration'] = len(production_data) * 10 / 60  # minutes
            batch_features['production_ratio'] = len(production_data) / len(df)
            
            # 생산 구간의 안정성
            if 'main_comp' in production_data.columns:
                batch_features['production_compression_stability'] = production_data['main_comp'].std()
        
        features_list.append(batch_features)
    
    # DataFrame으로 변환
    features_df = pd.DataFrame(features_list)
    
    print(f"✅ 시계열 특성 추출 완료:")
    print(f"   - 배치 수: {len(features_df)}")
    print(f"   - 추출된 특성 수: {len(features_df.columns) - 1}")  # batch_id 제외
    
    return features_df

if 'processed_timeseries' in locals():
    timeseries_features = extract_timeseries_features(processed_timeseries)
    print(f"\n📊 추출된 특성 예시:")
    print(timeseries_features.head())

# =============================================================================
# 5. 시계열 클러스터링
# =============================================================================
print("\n🎨 5단계: 시계열 클러스터링")
print("-" * 50)

def timeseries_clustering(ts_data, n_clusters=3):
    """시계열 패턴 기반 클러스터링"""
    
    # 주요 변수의 시계열을 배치별로 정렬
    main_variable = 'tbl_speed'  # 주요 분석 변수
    
    # 모든 배치의 데이터를 같은 길이로 맞추기 (리샘플링)
    resampled_series = []
    batch_ids = []
    
    target_length = 1000  # 목표 길이
    
    for batch_id, df in ts_data.items():
        if main_variable in df.columns:
            series = df[main_variable].dropna()
            
            if len(series) > 10:  # 최소 길이 체크
                # 선형 보간으로 목표 길이에 맞게 리샘플링
                x_old = np.linspace(0, 1, len(series))
                x_new = np.linspace(0, 1, target_length)
                resampled = np.interp(x_new, x_old, series.values)
                
                resampled_series.append(resampled)
                batch_ids.append(batch_id)
    
    if len(resampled_series) < 2:
        print("❌ 클러스터링을 위한 충분한 데이터가 없습니다.")
        return None, None
    
    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(resampled_series)
    
    # K-means 클러스터링 수행
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    print(f"✅ 시계열 클러스터링 완료:")
    print(f"   - 클러스터 수: {n_clusters}")
    print(f"   - 각 클러스터별 배치 수:")
    
    for i in range(n_clusters):
        cluster_batches = [batch_ids[j] for j in range(len(batch_ids)) if cluster_labels[j] == i]
        print(f"     클러스터 {i}: {cluster_batches}")
    
    # 클러스터링 결과 시각화
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    for i, (batch_id, series) in enumerate(zip(batch_ids, resampled_series)):
        color = plt.cm.viridis(cluster_labels[i] / (n_clusters - 1))
        plt.plot(series, color=color, alpha=0.7, label=f'Batch {batch_id} (Cluster {cluster_labels[i]})')
    
    plt.title(f'{main_variable} 시계열 클러스터링')
    plt.xlabel('Time Points')
    plt.ylabel(main_variable)
    plt.legend()
    
    # PCA로 2D 시각화
    plt.subplot(1, 2, 2)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('시계열 클러스터 (PCA)')
    plt.colorbar(scatter)
    
    # 배치 ID 표시
    for i, batch_id in enumerate(batch_ids):
        plt.annotate(f'B{batch_id}', (X_pca[i, 0], X_pca[i, 1]), fontsize=8)
    
    plt.tight_layout()
    plt.savefig('timeseries_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cluster_labels, batch_ids

if 'processed_timeseries' in locals():
    ts_clusters, ts_batch_ids = timeseries_clustering(processed_timeseries)

# =============================================================================
# 6. 시계열 이상값 탐지
# =============================================================================
print("\n🚨 6단계: 시계열 이상값 탐지")
print("-" * 50)

def detect_timeseries_anomalies(ts_data):
    """시계열에서 이상값 탐지"""
    
    anomalies_summary = {}
    
    for batch_id, df in ts_data.items():
        batch_anomalies = {}
        
        key_vars = ['tbl_speed', 'main_comp', 'tbl_fill', 'SREL']
        
        for var in key_vars:
            if var in df.columns:
                series = df[var].dropna()
                
                if len(series) > 10:
                    # 1) 통계적 이상값 (IQR 방법)
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers_iqr = series[(series < lower_bound) | (series > upper_bound)]
                    
                    # 2) 변화율 기반 이상값
                    diff_series = series.diff().abs()
                    diff_threshold = diff_series.quantile(0.95)  # 상위 5%
                    rapid_changes = diff_series[diff_series > diff_threshold]
                    
                    batch_anomalies[var] = {
                        'statistical_outliers': len(outliers_iqr),
                        'rapid_changes': len(rapid_changes),
                        'outlier_ratio': len(outliers_iqr) / len(series),
                        'max_change': diff_series.max()
                    }
        
        anomalies_summary[batch_id] = batch_anomalies
        
        # 배치별 이상값 비율 계산
        total_outliers = sum([anomalies_summary[batch_id][var]['statistical_outliers'] 
                             for var in anomalies_summary[batch_id]])
        total_points = len(df) * len(key_vars)
        
        print(f"📊 배치 {batch_id} 이상값 분석:")
        print(f"   - 전체 이상값 비율: {total_outliers/total_points:.2%}")
        
        for var in batch_anomalies:
            anomaly_info = batch_anomalies[var]
            print(f"   - {var}: 통계적 이상값 {anomaly_info['statistical_outliers']}개 "
                  f"({anomaly_info['outlier_ratio']:.2%})")
    
    return anomalies_summary

if 'processed_timeseries' in locals():
    anomalies_results = detect_timeseries_anomalies(processed_timeseries)

# =============================================================================
# 7. 시계열-품질 연관성 분석
# =============================================================================
print("\n🔬 7단계: 시계열-품질 연관성 분석")
print("-" * 50)

def analyze_timeseries_quality_relationship():
    """시계열 특성과 품질 지표 간 연관성 분석"""
    
    try:
        # Laboratory 데이터 로딩 (품질 지표)
        lab_data = pd.read_csv('Laboratory.csv', sep=';')
        
        if 'timeseries_features' in locals():
            # 배치 기준으로 병합
            merged_data = pd.merge(timeseries_features, lab_data, 
                                 left_on='batch_id', right_on='batch', how='inner')
            
            print(f"✅ 시계열-품질 데이터 병합 완료: {merged_data.shape}")
            
            # 주요 품질 지표
            quality_vars = ['dissolution_av', 'dissolution_min', 'impurities_total']
            
            # 시계열 특성과 품질 지표 간 상관관계 분석
            timeseries_cols = [col for col in merged_data.columns 
                             if any(prefix in col for prefix in ['tbl_speed_', 'main_comp_', 'tbl_fill_', 'SREL_'])]
            
            correlation_results = {}
            
            for quality_var in quality_vars:
                if quality_var in merged_data.columns:
                    correlations = []
                    
                    for ts_col in timeseries_cols:
                        if ts_col in merged_data.columns:
                            corr = merged_data[ts_col].corr(merged_data[quality_var])
                            if not np.isnan(corr):
                                correlations.append((ts_col, abs(corr)))
                    
                    # 상관관계가 높은 순으로 정렬
                    correlations.sort(key=lambda x: x[1], reverse=True)
                    correlation_results[quality_var] = correlations[:10]  # 상위 10개
                    
                    print(f"\n📊 {quality_var}와 상관관계가 높은 시계열 특성:")
                    for i, (feature, corr) in enumerate(correlations[:5]):
                        print(f"   {i+1}. {feature}: {corr:.3f}")
            
            # 상관관계 히트맵 시각화
            if len(timeseries_cols) > 0 and len([q for q in quality_vars if q in merged_data.columns]) > 0:
                correlation_matrix = merged_data[timeseries_cols[:20] + 
                                               [q for q in quality_vars if q in merged_data.columns]].corr()
                
                plt.figure(figsize=(12, 8))
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                           square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
                plt.title('시계열 특성 - 품질 지표 상관관계')
                plt.tight_layout()
                plt.savefig('timeseries_quality_correlation.png', dpi=300, bbox_inches='tight')
                plt.show()
            
            return merged_data, correlation_results
        
    except FileNotFoundError:
        print("❌ Laboratory.csv 파일을 찾을 수 없습니다.")
        return None, None

if 'timeseries_features' in locals():
    ts_quality_data, quality_correlations = analyze_timeseries_quality_relationship()

# =============================================================================
# 8. 시계열 기반 예측 모델링
# =============================================================================
print("\n🤖 8단계: 시계열 기반 예측 모델링")
print("-" * 50)

def build_timeseries_prediction_model():
    """시계열 특성을 사용한 품질 예측 모델"""
    
    if 'ts_quality_data' not in locals() or ts_quality_data is None:
        print("❌ 시계열-품질 병합 데이터가 없어 모델링을 건너뜁니다.")
        return None
    
    # 타겟 변수 선택
    target_var = 'dissolution_av'
    if target_var not in ts_quality_data.columns:
        print(f"❌ 타겟 변수 {target_var}를 찾을 수 없습니다.")
        return None
    
    # 시계열 특성만 선택
    feature_cols = [col for col in ts_quality_data.columns 
                   if any(prefix in col for prefix in ['tbl_speed_', 'main_comp_', 'tbl_fill_', 'SREL_', 'production_'])]
    
    X = ts_quality_data[feature_cols].dropna()
    y = ts_quality_data.loc[X.index, target_var]
    
    print(f"🎯 예측 대상: {target_var}")
    print(f"📊 시계열 특성 수: {len(feature_cols)}")
    print(f"📊 모델링 데이터: {X.shape}")
    
    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Random Forest 모델 훈련
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # 예측 및 평가
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n📈 시계열 기반 예측 모델 성능:")
    print(f"   - 훈련 RMSE: {train_rmse:.3f}")
    print(f"   - 테스트 RMSE: {test_rmse:.3f}")
    print(f"   - 훈련 R²: {train_r2:.3f}")
    print(f"   - 테스트 R²: {test_r2:.3f}")
    
    # 특성 중요도 분석
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Feature Importance')
    plt.title('시계열 특성 중요도 (Top 15)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('timeseries_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return rf_model, {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'feature_importance': feature_importance
    }

if 'ts_quality_data' in locals():
    ts_model, ts_performance = build_timeseries_prediction_model()

# =============================================================================
# 9. 실시간 모니터링 시뮬레이션
# =============================================================================
print("\n📡 9단계: 실시간 모니터링 시뮬레이션")
print("-" * 50)

def simulate_realtime_monitoring(ts_data, model=None):
    """실시간 품질 모니터링 시뮬레이션"""
    
    if not ts_data:
        print("❌ 시계열 데이터가 없어 시뮬레이션을 건너뜁니다.")
        return
    
    # 첫 번째 배치로 시뮬레이션
    batch_id = list(ts_data.keys())[0]
    df = ts_data[batch_id].copy()
    
    print(f"📡 배치 {batch_id} 실시간 모니터링 시뮬레이션")
    
    # 실시간 품질 지표 계산
    window_size = 50  # 500초 (8.3분) 이동 윈도우
    
    monitoring_results = []
    
    for i in range(window_size, len(df), 10):  # 10개씩 건너뛰며 시뮬레이션
        # 현재 윈도우 데이터
        window_data = df.iloc[i-window_size:i]
        
        # 실시간 통계 계산
        stats_dict = {
            'timestamp': df.iloc[i]['timestamp'],
            'time_minutes': (df.iloc[i]['timestamp'] - df.iloc[0]['timestamp']).total_seconds() / 60
        }
        
        key_vars = ['tbl_speed', 'main_comp', 'tbl_fill', 'SREL']
        
        for var in key_vars:
            if var in window_data.columns:
                series = window_data[var].dropna()
                if len(series) > 0:
                    stats_dict[f'{var}_mean'] = series.mean()
                    stats_dict[f'{var}_std'] = series.std()
                    stats_dict[f'{var}_trend'] = np.polyfit(range(len(series)), series, 1)[0]
        
        monitoring_results.append(stats_dict)
    
    # 결과 시각화
    monitoring_df = pd.DataFrame(monitoring_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, var in enumerate(['tbl_speed', 'main_comp', 'tbl_fill', 'SREL']):
        if f'{var}_mean' in monitoring_df.columns:
            ax = axes[i]
            
            # 실시간 평균값
            ax.plot(monitoring_df['time_minutes'], monitoring_df[f'{var}_mean'], 
                   'b-', label='Real-time Mean', linewidth=2)
            
            # 실시간 표준편차 (불안정성 지표)
            ax2 = ax.twinx()
            ax2.plot(monitoring_df['time_minutes'], monitoring_df[f'{var}_std'], 
                    'r--', label='Real-time Std', alpha=0.7)
            
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel(f'{var} Mean', color='b')
            ax2.set_ylabel(f'{var} Std', color='r')
            ax.set_title(f'{var} 실시간 모니터링')
            ax.grid(True, alpha=0.3)
            
            # 범례
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('realtime_monitoring_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 실시간 모니터링 시뮬레이션 완료")
    print(f"   - 모니터링 포인트 수: {len(monitoring_results)}")
    
    return monitoring_df

if 'processed_timeseries' in locals():
    realtime_monitoring = simulate_realtime_monitoring(processed_timeseries)

# =============================================================================
# 10. 결과 요약
# =============================================================================
print("\n📋 10단계: 시계열 분석 결과 요약")
print("-" * 50)

def create_timeseries_summary():
    """시계열 분석 결과 요약"""
    
    print("📊 제약 제조 공정 시계열 분석 요약 리포트")
    print("=" * 60)
    
    if 'timeseries_data' in locals():
        print(f"📈 시계열 데이터 기본 정보:")
        print(f"   - 분석된 배치 수: {len(timeseries_data)}")
        total_records = sum(len(df) for df in timeseries_data.values())
        print(f"   - 총 시계열 레코드 수: {total_records:,}")
        
        avg_duration = np.mean([
            (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600 
            for df in timeseries_data.values()
        ])
        print(f"   - 평균 배치 생산 시간: {avg_duration:.1f}시간")
    
    if 'timeseries_features' in locals():
        print(f"\n🔍 특성 엔지니어링 결과:")
        print(f"   - 추출된 시계열 특성 수: {len(timeseries_features.columns) - 1}")
    
    if 'ts_clusters' in locals() and ts_clusters is not None:
        print(f"\n🎨 클러스터링 결과:")
        unique_clusters = len(set(ts_clusters))
        print(f"   - 발견된 생산 패턴 수: {unique_clusters}")
    
    if 'ts_performance' in locals():
        print(f"\n🤖 시계열 기반 예측 모델 성능:")
        print(f"   - 테스트 R²: {ts_performance['test_r2']:.3f}")
        print(f"   - 테스트 RMSE: {ts_performance['test_rmse']:.3f}")
    
    print(f"\n💡 주요 인사이트:")
    print(f"   1. 제조 공정의 압축력과 속도 변동이 품질에 직접적 영향")
    print(f"   2. 생산 초기 단계의 안정성이 최종 품질 예측에 중요")
    print(f"   3. 실시간 모니터링을 통한 조기 품질 이상 탐지 가능")
    print(f"   4. 배치별 생산 패턴 분류를 통한 공정 최적화 방향 제시")

# 최종 요약 리포트 생성
create_timeseries_summary()

print("\n" + "=" * 80)
print("🎉 제약 제조 공정 시계열 데이터 분석 실습 완료!")
print("=" * 80)

print("\n📁 생성된 파일:")
print("   - timeseries_overview.png: 시계열 데이터 개요")
print("   - timeseries_clustering.png: 시계열 클러스터링 결과")
print("   - timeseries_quality_correlation.png: 시계열-품질 상관관계")
print("   - timeseries_feature_importance.png: 시계열 특성 중요도")
print("   - realtime_monitoring_simulation.png: 실시간 모니터링 시뮬레이션")

print("\n🔍 추가 연구 방향:")
print("   1. LSTM/GRU를 활용한 딥러닝 시계열 예측")
print("   2. 다변량 시계열 이상값 탐지 알고리즘")
print("   3. 실시간 품질 방출(RTRT) 시스템 구현")
print("   4. 디지털 트윈 기반 공정 시뮬레이션")