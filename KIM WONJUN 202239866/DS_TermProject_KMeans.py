import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 데이터 경로
CSV_PATH = "C:/Users/admin/Downloads/archive (3)/healthcare-dataset-stroke-data.csv"

# 1. 데이터 불러오기
df = pd.read_csv(CSV_PATH)

# 2. 전처리 (예: 숫자형 변수만 사용, 결측치 처리)
# 결측치 간단히 제거
df = df.dropna()

# stroke 예측에 쓸 수 있는 숫자형 피처 선택 (예시)
features = ['age', 'avg_glucose_level', 'bmi']

X = df[features]

# 3. 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 클러스터 수 후보
K = range(2, 10)

sse = []
silhouette_scores = []

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)  # SSE
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# 5. 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K, sse, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('SSE')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'ro-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Index')

plt.tight_layout()
plt.show()
