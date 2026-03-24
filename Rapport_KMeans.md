# Rapport Complet sur l'Algorithme K-Means

## 1. Introduction
- **Définition du clustering :** Le clustering est une méthode d'apprentissage automatique non supervisé qui consiste à regrouper un ensemble de données en sous-groupes (clusters) homogènes. Les éléments d'un même cluster partagent des caractéristiques similaires, tandis qu'ils sont distincts de ceux des autres clusters.
- **Présentation de K-Means :** K-Means est l'un des algorithmes de clustering les plus populaires et intuitifs. Son objectif est de partitionner les données en $K$ groupes distincts en minimisant la variance à l'intérieur de chaque cluster.
- **Cas d’utilisation :** Segmentation client en marketing, compression d'images (réduction des couleurs), détection d'anomalies, ou encore regroupement de documents par thématique.

## 2. Principe de l’algorithme
- **Explication mathématique (WCSS) :** L'algorithme cherche à minimiser la somme des carrés des distances (Within-Cluster Sum of Squares - WCSS) entre chaque point de donnée et le centroïde (centre) de son cluster d'appartenance. C'est ce qu'on appelle "l'inertie".
- **Étapes de l’algorithme (Méthode de Lloyd) :**
  1. **Initialisation :** Sélectionner aléatoirement $K$ points comme centroïdes initiaux.
  2. **Assignation :** Associer chaque point de donnée au centroïde le plus proche (généralement via la distance euclidienne).
  3. **Mise à jour :** Recalculer la position de chaque centroïde comme étant la moyenne des points de son cluster.
  4. **Itération :** Répéter les étapes 2 et 3 jusqu'à ce que les centroïdes ne bougent plus (convergence).
- **Hypothèses :** K-Means suppose que les clusters sont convexes (sans creux géométriques) et isotropes (de forme sphérique), avec des variances et des tailles approximativement comparables.

## 3. Préparation des données
Pour ce rapport, nous utiliserons le célèbre jeu de données **Iris**, disponible via `scikit-learn`.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# --- 1. Charger les données Iris ---
iris = load_iris()
X = iris.data
y_true = iris.target  # Vraies étiquettes (pour comparaison future)
feature_names = iris.feature_names

print(f"Dimensions des données originales : {X.shape}")

# --- 2. Normaliser avec StandardScaler ---
# Étape cruciale car K-Means est sensible aux échelles des différentes variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. Réduire en 2D avec PCA pour visualisation ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Dimensions après réduction PCA : {X_pca.shape}")
```

## 4. Implémentation de K-Means
Nous allons entraîner un modèle K-Means pour partitionner les données en 3 clusters ($K=3$), car nous savons qu'il y a 3 espèces de fleurs dans le jeu Iris.

```python
from sklearn.cluster import KMeans

# Entraîner le modèle KMeans en configurant k=3
# "k-means++" optimise l'initialisation des centroïdes pour une meilleure convergence
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# Prédire et récupérer les clusters assignés pour chaque observation
labels_pred = kmeans.labels_

# Afficher les coordonnées des centroïdes (dans l'espace normalisé 4D)
centroids = kmeans.cluster_centers_
print("\nCoordonnées des centroïdes (dans l'espace normalisé) :")
print(centroids)
```

## 5. Évaluation du modèle
Dans un contexte non supervisé, nous évaluons la qualité intrinsèque de nos clusters avec des métriques spécifiques.

```python
from sklearn.metrics import silhouette_score

# 1. Calcul de l'Inertie (WCSS)
# C'est la somme des distances au carré entre chaque point et son centroïde
wcss = kmeans.inertia_

# 2. Calcul du Silhouette Score
# Varie de -1 (mauvais) à 1 (excellent), évalue à quel point un objet est 
# similaire à son propre cluster (cohésion) par rapport aux autres (séparation).
sil_score = silhouette_score(X_scaled, labels_pred)

print(f"\nÉvaluation du Modèle :")
print(f"-> Inertie intra-cluster (WCSS) : {wcss:.2f}")
print(f"-> Silhouette Score : {sil_score:.4f}")
```

**Interprétation des résultats :**
- **Inertie (WCSS) :** Avec une valeur d'environ 139.82, l'inertie indique la dispersion interne des clusters. Elle diminue naturellement quand $K$ augmente.
- **Silhouette Score :** Un score d'environ `0.46` est très correct. Il indique que nos clusters sont globalement bien séparés et denses, confirmant que chaque donnée est relativement bien assignée à son groupe.

## 6. Visualisation
La visualisation permet d'interpréter visuellement l'efficacité de la séparation spatiale, ainsi que de justifier le choix du nombre de clusters $K$.

```python
plt.figure(figsize=(14, 6))

# === Graphique 1 : Méthode Elbow (Le Coude) ===
plt.subplot(1, 2, 1)
wcss_list = []
k_range = range(1, 11)

for i in k_range:
    km = KMeans(n_clusters=i, init='k-means++', random_state=42)
    km.fit(X_scaled)
    wcss_list.append(km.inertia_)

plt.plot(k_range, wcss_list, marker='o', color='b', linestyle='-')
plt.title('Méthode Elbow (Détermination du K optimal)')
plt.xlabel('Nombre de clusters (K)')
plt.ylabel('Inertie (WCSS)')
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.7)

# === Graphique 2 : Clusters en 2D (PCA) et Centroïdes ===
plt.subplot(1, 2, 2)
# Nuage de points des données, coloré selon le cluster prédit
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_pred, cmap='viridis', s=50, alpha=0.8, edgecolor='k')

# Projection des centroïdes 4D vers l'espace 2D de la PCA 
centroids_pca = pca.transform(centroids)
# Affichage des centroïdes avec une croix rouge
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', s=200, marker='X', edgecolor='k', label='Centroïdes')

plt.title('Clustering K-Means du Dataset Iris (réduction PCA)')
plt.xlabel('Composante Principale 1')
plt.ylabel('Composante Principale 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
```

**Description des graphiques :**
1. **Méthode Elbow :** L'inertie chute drastiquement de $K=1$ à $K=2$, puis de $K=2$ à $K=3$, avant de décroître beaucoup plus lentement. Ce "coude" visuel à $K=2$ ou $K=3$ justifie mathématiquement que 3 clusters est un choix de complexité optimal pour ce jeu de données.
2. **Clusters 2D (PCA) :** Le nuage de points révèle qu'un des clusters est parfaitement distinctement isolé des deux autres (à gauche). Les deux autres clusters se touchent légèrement avec une frontière invisible qui les sépare, illustrant parfaitement la formation de groupes sphériques et denses. Les centroïdes rouges sont bien positionnés au cœur de chaque densité de points.

## 7. Analyse des résultats
- **Interprétation des clusters obtenus :** K-Means a réussi à segmenter les données en 3 groupes structurels logiques sans utiliser d'étiquettes réelles. L'un des groupes est très facile à séparer, tandis que les deux autres partagent des caractéristiques plus proches.
- **Comparaison avec les vraies classes (Iris) :** Le cluster gauche parfaitement isolé correspond à la sous-espèce *Iris Setosa*. Les deux clusters centraux correspondent aux espèces *Iris Versicolor* et *Iris Virginica*. Ayant des tailles morphologiques très similaires, il est naturel et mathématiquement justifiable que K-Means confonde légèrement les bordures de ces deux classes.
- **Qualité du clustering :** La séparation obtenue est très fidèle à la réalité biologique. L'algorithme se comporte exactement comme attendu sur un problème convexe classique.

## 8. Conclusion
- **Résumé des performances :** Le modèle a convergé avec succès, séparant le dataset Iris en un partitionnement scientifiquement et visuellement cohérent, avec un score Silhouette décent (~0.459).
- **Avantages de K-Means :** 
  - Simplicité extrême d'implémentation.
  - Vitesse d'exécution rapide ($O(n)$) sur de gros jeux de données.
  - Résultat facilement interprétable.
- **Limites de K-Means :** 
  - Nécessite de spécifier $K$ à l'avance (d'où l'utilité de l'Elbow Method).
  - Très sensible aux outliers (valeurs aberrantes) qui peuvent "tirer" les centroïdes.
  - Ne fonctionne pas bien sur des clusters imbriqués ou non-sphériques (ex: lunes, cercles concentriques).
- **Cas recommandé :** Il est fortement recommandé comme *baseline* (modèle de référence) pour tout problème de clustering, particulièrement adapté quand on suppose que les groupes sont volumineux, denses et globulaires, comme le regroupement de documents (NLP), ou la segmentation de clientèle simple.
