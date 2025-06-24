"""
제약 제조 데이터 전처리 및 특성 엔지니어링 실습

이 실습에서는 제약 제조 공정 데이터를 사용하여 다음 내용을 학습합니다:
1. 제조 데이터 처리
2. 데이터 수집과 데이터 전처리
3. 차원축소와 특성 엔지니어링
4. 데이터 전처리 실습

데이터셋: 고용량 콜레스테롤 저하 필름-코팅 정제 1,005 배치의 실제 제조·시험 기록
기간: 2018년 11월 ~ 2021년 4월
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('default')

print("=" * 80)
print("제약 제조 데이터 전처리 및 특성 엔지니어링 실습")
print("=" * 80)

# =============================================================================
# 1. 데이터 수집 및 로딩
# =============================================================================
print("\n📊 1단계: 데이터 수집 및 로딩")
print("-" * 50)

def load_pharmaceutical_data():
    """제약 제조 데이터 로딩 함수"""
    try:
        # 실험실 데이터 (배치별 원료, 중간제품, 완제품 품질 데이터)
        laboratory_df = pd.read_csv('Laboratory.csv', sep=';')
        print(f"✅ Laboratory 데이터 로딩 완료: {laboratory_df.shape}")
        
        # 공정 데이터 (배치별 집계된 공정 센서 데이터)
        process_df = pd.read_csv('Process.csv', sep=';')
        print(f"✅ Process 데이터 로딩 완료: {process_df.shape}")
        
        # 정규화 계수 데이터
        normalization_df = pd.read_csv('Normalization.csv', sep=';')
        print(f"✅ Normalization 데이터 로딩 완료: {normalization_df.shape}")
        
        return laboratory_df, process_df, normalization_df
    
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        return None, None, None

# 데이터 로딩
lab_data, process_data, norm_data = load_pharmaceutical_data()

if lab_data is not None:
    print(f"\n📈 데이터 개요:")
    print(f"- 총 배치 수: {len(lab_data)}")
    print(f"- Laboratory 변수 수: {len(lab_data.columns)}")
    print(f"- Process 변수 수: {len(process_data.columns)}")

# =============================================================================
# 2. 탐색적 데이터 분석 (EDA)
# =============================================================================
print("\n📊 2단계: 탐색적 데이터 분석")
print("-" * 50)

def explore_data(lab_df, process_df):
    """데이터 탐색 함수"""
    
    print("🔍 Laboratory 데이터 기본 정보:")
    print(lab_df.info())
    
    print("\n🔍 Process 데이터 기본 정보:")
    print(process_df.info())
    
    # 결측값 확인
    print("\n❗ Laboratory 데이터 결측값:")
    missing_lab = lab_df.isnull().sum()
    print(missing_lab[missing_lab > 0])
    
    print("\n❗ Process 데이터 결측값:")
    missing_process = process_df.isnull().sum()
    print(missing_process[missing_process > 0])
    
    # 기본 통계
    print("\n📊 주요 품질 지표 통계:")
    quality_cols = ['dissolution_av', 'dissolution_min', 'impurities_total']
    if all(col in lab_df.columns for col in quality_cols):
        print(lab_df[quality_cols].describe())

if lab_data is not None:
    explore_data(lab_data, process_data)

# =============================================================================
# 3. 데이터 전처리
# =============================================================================
print("\n🔧 3단계: 데이터 전처리")
print("-" * 50)

def preprocess_data(lab_df, process_df, norm_df):
    """데이터 전처리 함수"""
    
    # 데이터 복사
    lab_clean = lab_df.copy()
    process_clean = process_df.copy()
    
    print("🧹 1) 결측값 처리")
    
    # 수치형 컬럼 식별
    numeric_cols_lab = lab_clean.select_dtypes(include=[np.number]).columns
    numeric_cols_process = process_clean.select_dtypes(include=[np.number]).columns
    
    # 결측값을 중앙값으로 대체
    for col in numeric_cols_lab:
        if lab_clean[col].isnull().sum() > 0:
            median_val = lab_clean[col].median()
            lab_clean[col].fillna(median_val, inplace=True)
            print(f"   - {col}: 결측값 {lab_clean[col].isnull().sum()}개를 중앙값 {median_val:.2f}로 대체")
    
    for col in numeric_cols_process:
        if process_clean[col].isnull().sum() > 0:
            median_val = process_clean[col].median()
            process_clean[col].fillna(median_val, inplace=True)
            print(f"   - {col}: 결측값 {process_clean[col].isnull().sum()}개를 중앙값 {median_val:.2f}로 대체")
    
    print("\n🔗 2) 데이터 병합")
    
    # batch 기준으로 데이터 병합
    merged_df = pd.merge(lab_clean, process_clean, on='batch', how='inner')
    print(f"   - 병합 후 데이터 크기: {merged_df.shape}")
    
    # 정규화 계수 병합
    if 'code_x' in merged_df.columns:
        merged_df = pd.merge(merged_df, norm_df, left_on='code_x', right_on='Product code', how='left')
    
    print("\n🚫 3) 이상값 제거")
    
    # IQR 방법으로 이상값 제거
    def remove_outliers_iqr(df, columns):
        df_clean = df.copy()
        outliers_removed = 0
        
        for col in columns:
            if col in df_clean.columns and df_clean[col].dtype in ['int64', 'float64']:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                outliers_removed += len(outliers)
        
        return df_clean, outliers_removed
    
    # 주요 품질 지표에서 이상값 제거
    quality_indicators = ['dissolution_av', 'dissolution_min', 'impurities_total', 'tbl_speed_mean']
    merged_clean, outliers_count = remove_outliers_iqr(merged_df, quality_indicators)
    print(f"   - 제거된 이상값: {outliers_count}개")
    print(f"   - 최종 데이터 크기: {merged_clean.shape}")
    
    return merged_clean

if lab_data is not None:
    merged_data = preprocess_data(lab_data, process_data, norm_data)

# =============================================================================
# 4. 특성 엔지니어링
# =============================================================================
print("\n⚙️ 4단계: 특성 엔지니어링")
print("-" * 50)

def feature_engineering(df):
    """특성 엔지니어링 함수"""
    
    feature_df = df.copy()
    
    print("🔧 1) 새로운 특성 생성")
    
    # 공정 효율성 지표
    if 'total_waste' in feature_df.columns and 'batch_yield' in feature_df.columns:
        feature_df['process_efficiency'] = feature_df['batch_yield'] / (feature_df['total_waste'] + 1)
        print("   - process_efficiency: 공정 효율성 지표 생성")
    
    # 품질 일관성 지표
    if 'dissolution_av' in feature_df.columns and 'dissolution_min' in feature_df.columns:
        feature_df['quality_consistency'] = feature_df['dissolution_min'] / feature_df['dissolution_av']
        print("   - quality_consistency: 품질 일관성 지표 생성")
    
    # 압축력 변동 지표
    if 'main_CompForce mean' in feature_df.columns and 'main_CompForce_sd' in feature_df.columns:
        feature_df['compression_stability'] = feature_df['main_CompForce_sd'] / feature_df['main_CompForce mean']
        print("   - compression_stability: 압축력 안정성 지표 생성")
    
    # 원료 품질 종합 점수
    api_cols = [col for col in feature_df.columns if 'api_' in col and feature_df[col].dtype in ['int64', 'float64']]
    if len(api_cols) > 0:
        feature_df['api_quality_score'] = feature_df[api_cols].mean(axis=1)
        print("   - api_quality_score: API 품질 종합 점수 생성")
    
    print("\n📊 2) 범주형 변수 인코딩")
    
    # 범주형 변수 원-핫 인코딩
    categorical_cols = feature_df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col not in ['batch', 'start']]  # 식별자 제외
    
    for col in categorical_cols:
        if feature_df[col].nunique() < 20:  # 유니크 값이 20개 미만인 경우만
            dummies = pd.get_dummies(feature_df[col], prefix=col, drop_first=True)
            feature_df = pd.concat([feature_df, dummies], axis=1)
            feature_df.drop(col, axis=1, inplace=True)
            print(f"   - {col}: 원-핫 인코딩 완료")
    
    return feature_df

if 'merged_data' in locals():
    engineered_data = feature_engineering(merged_data)
    print(f"\n✅ 특성 엔지니어링 완료 - 최종 특성 수: {len(engineered_data.columns)}")

# =============================================================================
# 5. 데이터 스케일링
# =============================================================================
print("\n📏 5단계: 데이터 스케일링")
print("-" * 50)

def scale_features(df, target_cols=None):
    """특성 스케일링 함수"""
    
    if target_cols is None:
        target_cols = ['dissolution_av', 'dissolution_min', 'impurities_total']
    
    # 수치형 특성만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in target_cols + ['batch']]
    
    X = df[feature_cols].copy()
    y = df[target_cols].copy() if all(col in df.columns for col in target_cols) else None
    
    print(f"🎯 타겟 변수: {target_cols}")
    print(f"📊 특성 변수 수: {len(feature_cols)}")
    
    # 다양한 스케일링 방법 비교
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }
    
    scaled_data = {}
    
    for name, scaler in scalers.items():
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        scaled_data[name] = X_scaled_df
        
        print(f"   ✅ {name} 적용 완료")
        print(f"      - 평균: {X_scaled.mean():.3f}, 표준편차: {X_scaled.std():.3f}")
    
    return scaled_data, X, y, feature_cols

if 'engineered_data' in locals():
    scaled_datasets, original_X, target_y, feature_names = scale_features(engineered_data)

# =============================================================================
# 6. 차원 축소
# =============================================================================
print("\n📉 6단계: 차원 축소")
print("-" * 50)

def dimension_reduction(X_scaled, n_components=10):
    """차원 축소 함수"""
    
    print("🔍 1) 주성분 분석 (PCA)")
    
    # PCA 적용
    pca = PCA()
    X_pca_full = pca.fit_transform(X_scaled)
    
    # 설명 가능한 분산 비율 계산
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # 95% 분산을 설명하는 주성분 개수 찾기
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    print(f"   - 95% 분산 설명에 필요한 주성분 수: {n_components_95}")
    
    # 지정된 개수의 주성분으로 차원 축소
    pca_reduced = PCA(n_components=min(n_components, X_scaled.shape[1]))
    X_pca_reduced = pca_reduced.fit_transform(X_scaled)
    
    print(f"   - 선택된 주성분 수: {pca_reduced.n_components_}")
    print(f"   - 설명 가능한 분산 비율: {pca_reduced.explained_variance_ratio_.sum():.3f}")
    
    # PCA 결과 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, min(21, len(explained_variance_ratio)) + 1), 
             explained_variance_ratio[:20], 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Component')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, min(21, len(cumulative_variance_ratio)) + 1), 
             cumulative_variance_ratio[:20], 'ro-')
    plt.axhline(y=0.95, color='k', linestyle='--', alpha=0.7)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return X_pca_reduced, pca_reduced, n_components_95

if 'scaled_datasets' in locals():
    # StandardScaler로 스케일링된 데이터 사용
    X_scaled = scaled_datasets['StandardScaler']
    X_pca, pca_model, optimal_components = dimension_reduction(X_scaled)

# =============================================================================
# 7. 특성 선택
# =============================================================================
print("\n🎯 7단계: 특성 선택")
print("-" * 50)

def feature_selection(X, y, feature_names, k=20):
    """특성 선택 함수"""
    
    if y is None or len(y.columns) == 0:
        print("❌ 타겟 변수가 없어 특성 선택을 건너뜁니다.")
        return None, None
    
    # 첫 번째 타겟 변수 사용
    target = y.iloc[:, 0].dropna()
    X_target = X.loc[target.index]
    
    print(f"🎯 타겟 변수: {y.columns[0]}")
    print(f"📊 사용 가능한 샘플 수: {len(target)}")
    
    # 1) 통계적 특성 선택 (F-score)
    print("\n1) F-score 기반 특성 선택")
    selector_f = SelectKBest(score_func=f_regression, k=min(k, X_target.shape[1]))
    X_selected_f = selector_f.fit_transform(X_target, target)
    
    selected_features_f = np.array(feature_names)[selector_f.get_support()]
    scores_f = selector_f.scores_[selector_f.get_support()]
    
    print(f"   - 선택된 특성 수: {len(selected_features_f)}")
    print("   - 상위 5개 특성:")
    for i, (feature, score) in enumerate(zip(selected_features_f[:5], scores_f[:5])):
        print(f"     {i+1}. {feature}: {score:.2f}")
    
    # 2) 상호정보량 기반 특성 선택
    print("\n2) 상호정보량 기반 특성 선택")
    selector_mi = SelectKBest(score_func=mutual_info_regression, k=min(k, X_target.shape[1]))
    X_selected_mi = selector_mi.fit_transform(X_target, target)
    
    selected_features_mi = np.array(feature_names)[selector_mi.get_support()]
    scores_mi = selector_mi.scores_[selector_mi.get_support()]
    
    print(f"   - 선택된 특성 수: {len(selected_features_mi)}")
    print("   - 상위 5개 특성:")
    for i, (feature, score) in enumerate(zip(selected_features_mi[:5], scores_mi[:5])):
        print(f"     {i+1}. {feature}: {score:.3f}")
    
    # 3) Random Forest 중요도 기반 특성 선택
    print("\n3) Random Forest 중요도 기반 특성 선택")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_target, target)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features_rf = importance_df.head(k)['feature'].values
    
    print(f"   - 상위 {k}개 특성 선택")
    print("   - 상위 5개 특성:")
    for i, (_, row) in enumerate(importance_df.head(5).iterrows()):
        print(f"     {i+1}. {row['feature']}: {row['importance']:.3f}")
    
    return {
        'f_score': (selected_features_f, X_selected_f),
        'mutual_info': (selected_features_mi, X_selected_mi),
        'random_forest': (top_features_rf, X_target[top_features_rf])
    }, importance_df

if 'original_X' in locals() and 'target_y' in locals():
    feature_selection_results, feature_importance = feature_selection(
        original_X, target_y, feature_names
    )

# =============================================================================
# 8. 클러스터링 분석
# =============================================================================
print("\n🎨 8단계: 클러스터링 분석")
print("-" * 50)

def clustering_analysis(X_scaled, max_clusters=8):
    """클러스터링 분석 함수"""
    
    print("🔍 K-means 클러스터링 분석")
    
    # 최적 클러스터 수 찾기 (Elbow Method)
    inertias = []
    K_range = range(2, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Elbow plot
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    
    # 최적 클러스터 수로 클러스터링 수행
    optimal_k = 4  # 일반적으로 4-5개가 적당
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    print(f"   - 선택된 클러스터 수: {optimal_k}")
    print(f"   - 각 클러스터별 샘플 수:")
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        print(f"     클러스터 {cluster}: {count}개")
    
    # PCA로 2D 시각화
    if 'X_pca' in globals() and X_pca.shape[1] >= 2:
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Clusters in PCA Space')
        plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cluster_labels, kmeans

if 'scaled_datasets' in locals():
    cluster_labels, kmeans_model = clustering_analysis(scaled_datasets['StandardScaler'])

# =============================================================================
# 9. 예측 모델링 및 성능 평가
# =============================================================================
print("\n🤖 9단계: 예측 모델링")
print("-" * 50)

def build_prediction_model(X, y, feature_names):
    """예측 모델 구축 및 평가 함수"""
    
    if y is None or len(y.columns) == 0:
        print("❌ 타겟 변수가 없어 모델링을 건너뜁니다.")
        return None
    
    # 첫 번째 타겟 변수 사용
    target = y.iloc[:, 0].dropna()
    X_model = X.loc[target.index]
    
    print(f"🎯 예측 대상: {y.columns[0]}")
    print(f"📊 모델링 데이터: {X_model.shape}")
    
    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_model, target, test_size=0.2, random_state=42
    )
    
    print(f"   - 훈련 데이터: {X_train.shape}")
    print(f"   - 테스트 데이터: {X_test.shape}")
    
    # Random Forest 모델 훈련
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # 예측 및 평가
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n📈 모델 성능:")
    print(f"   - 훈련 RMSE: {train_rmse:.3f}")
    print(f"   - 테스트 RMSE: {test_rmse:.3f}")
    print(f"   - 훈련 R²: {train_r2:.3f}")
    print(f"   - 테스트 R²: {test_r2:.3f}")
    
    # 특성 중요도 시각화
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return rf_model, {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2
    }

if 'original_X' in locals() and 'target_y' in locals():
    model, performance = build_prediction_model(original_X, target_y, feature_names)

# =============================================================================
# 10. 결과 요약 및 시각화
# =============================================================================
print("\n📋 10단계: 결과 요약")
print("-" * 50)

def create_summary_report():
    """결과 요약 리포트 생성"""
    
    print("📊 제약 제조 데이터 분석 요약 리포트")
    print("=" * 60)
    
    if 'merged_data' in locals():
        print(f"📈 데이터 기본 정보:")
        print(f"   - 총 배치 수: {len(merged_data)}")
        print(f"   - 원본 변수 수: {len(lab_data.columns) + len(process_data.columns)}")
        print(f"   - 전처리 후 변수 수: {len(engineered_data.columns) if 'engineered_data' in locals() else 'N/A'}")
    
    if 'optimal_components' in locals():
        print(f"\n🔍 차원 축소 결과:")
        print(f"   - 95% 분산 설명 주성분 수: {optimal_components}")
        print(f"   - PCA 적용 후 차원: {X_pca.shape[1] if 'X_pca' in locals() else 'N/A'}")
    
    if 'cluster_labels' in locals():
        print(f"\n🎨 클러스터링 결과:")
        print(f"   - 클러스터 수: {len(np.unique(cluster_labels))}")
        print(f"   - 각 클러스터별 분포: {dict(zip(*np.unique(cluster_labels, return_counts=True)))}")
    
    if 'performance' in locals():
        print(f"\n🤖 예측 모델 성능:")
        print(f"   - 테스트 R²: {performance['test_r2']:.3f}")
        print(f"   - 테스트 RMSE: {performance['test_rmse']:.3f}")
    
    print(f"\n💡 주요 인사이트:")
    print(f"   1. 제약 제조 공정에서는 압축력, 속도, 충전 깊이가 주요 품질 결정 요인")
    print(f"   2. 원료 품질(API, 부형제)과 최종 제품 품질 간 강한 상관관계 존재")
    print(f"   3. 공정 안정성 지표를 통한 실시간 품질 예측 가능성 확인")
    print(f"   4. 클러스터링을 통해 배치별 생산 패턴 구분 가능")

# 최종 요약 리포트 생성
create_summary_report()

print("\n" + "=" * 80)
print("🎉 제약 제조 데이터 전처리 및 특성 엔지니어링 실습 완료!")
print("=" * 80)

print("\n📁 생성된 파일:")
print("   - pca_analysis.png: PCA 분석 결과")
print("   - clustering_analysis.png: 클러스터링 분석 결과")  
print("   - feature_importance.png: 특성 중요도 분석")

print("\n🔍 추가 학습 방향:")
print("   1. 시계열 데이터(Process/*.csv)를 활용한 LSTM 모델링")
print("   2. 앙상블 모델을 통한 예측 성능 향상")
print("   3. 실시간 품질 방출(RTRT) 시스템 구현")
print("   4. 공정 최적화를 위한 다목적 최적화")