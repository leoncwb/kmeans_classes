import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ===============================
# 1. Dados de treino (apenas normais)
# ===============================
X_train = np.array([
    [1, 1],
    [1.5, 2],
    [3, 4],
    [5, 7],
    [3.5, 5],
    [4.5, 5]
])

# ===============================
# 2. Treinar K-Means
# ===============================
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)

# Coordenadas dos centroides
centroides = kmeans.cluster_centers_
print("Centroides:\n", centroides)

# ===============================
# 3. Calcular distâncias de treino
# ===============================
distances_train = np.min(kmeans.transform(X_train), axis=1)
threshold = distances_train.mean() + 2 * distances_train.std()
print(f"Threshold (limiar) definido: {threshold:.3f}")

# ===============================
# 4. Testar novos pontos
# ===============================
X_test = np.array([
    [1, 1],      # ponto normal
    [4, 4],      # ponto normal
    [10, 10]     # provável anomalia
])

distances_test = np.min(kmeans.transform(X_test), axis=1)

for point, dist in zip(X_test, distances_test):
    status = "Anomalia" if dist > threshold else "Normal"
    print(f"Ponto {point} → {status} (distância: {dist:.2f})")

# Depois de kmeans.fit(X_train)
print("\nLabels no treino (cluster de cada ponto):", kmeans.labels_)
for i, (x, y) in enumerate(X_train, start=1):
    print(f"P{i} = ({x:.2f}, {y:.2f}) -> cluster {kmeans.labels_[i-1]}")

# Distâncias de treino e threshold
distances_train = np.min(kmeans.transform(X_train), axis=1)
threshold = distances_train.mean() + 2 * distances_train.std()
print("\nDistâncias de treino ao centroide mais próximo:", np.round(distances_train, 3))
print(f"Threshold (média + 2σ): {threshold:.3f}")

# Distâncias e classificação dos testes
distances_test = np.min(kmeans.transform(X_test), axis=1)
print("\nDistâncias de teste ao centroide mais próximo:", np.round(distances_test, 3))
for point, dist in zip(X_test, distances_test):
    status = "Anomalia" if dist > threshold else "Normal"
    print(f"Ponto {point} → {status} (distância: {dist:.3f})")

# ===============================
# 5. Visualização
# ===============================
plt.figure(figsize=(6, 6))
# Pontos de treino coloridos por cluster
plt.scatter(X_train[:, 0], X_train[:, 1], c=kmeans.labels_, cmap='viridis', s=100, label="Treino")
# Pontos de teste
plt.scatter(X_test[:, 0], X_test[:, 1], c='red', marker='x', s=150, label="Teste")
# Centroides
plt.scatter(centroides[:, 0], centroides[:, 1], c='black', marker='X', s=200, label="Centroides")

plt.title("K-Means com Detecção de Anomalias")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
