# Compte Rendu — Algorithmes de Clustering

## Exemples avec données réelles (scikit-learn)

> **Fichier source :** `clustering_algorithms.ipynb` → `clustering_examples.py`  
> **Date :** Mars 2026  
> **Datasets :** Iris (150×4), Wine (178×13), Digits (1797×64)

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Préparation des données](#2-préparation-des-données)
3. [K-Means](#3-k-means)
4. [K-Medoids (PAM)](#4-k-medoids-pam)
5. [K-Medians](#5-k-medians)
6. [DBSCAN](#6-dbscan)
7. [HDBSCAN](#7-hdbscan)
8. [OPTICS](#8-optics)
9. [Clustering Hiérarchique Agglomératif (HAC)](#9-clustering-hiérarchique-agglomératif-hac)
10. [Gaussian Mixture Models (GMM)](#10-gaussian-mixture-models-gmm)
11. [Spectral Clustering](#11-spectral-clustering)
12. [Affinity Propagation](#12-affinity-propagation)
13. [Self-Organizing Maps (SOM)](#13-self-organizing-maps-som)
14. [UMAP + K-Means](#14-umap--k-means)
15. [Comparaison globale](#15-comparaison-globale)

---

## 1. Introduction

Le **clustering** (ou classification non supervisée) est une technique d'apprentissage automatique
qui vise à regrouper automatiquement des données similaires sans utiliser d'étiquettes.

### Métriques principales

| Métrique | Description | Interprétation |
|----------|-------------|----------------|
| **Silhouette Score** | Cohésion intra-cluster vs séparation inter-cluster | [-1, 1] → plus proche de 1 = meilleur |
| **WCSS / Inertie** | Somme des distances au centroïde (K-Means) | Minimiser |
| **BIC / AIC** | Critères de sélection de modèle (GMM) | Minimiser |

### Datasets utilisés

| Dataset | Taille | Features | Classes | Usage |
|---------|--------|----------|---------|-------|
| **Iris** | 150 | 4 | 3 | K-Means, K-Medoids, K-Medians, DBSCAN, OPTICS, HAC, Spectral, AP |
| **Wine** | 178 | 13 | 3 | GMM, SOM |
| **Digits** | 1 797 | 64 | 10 | UMAP + K-Means |
| **Moons/Circles** | 300 | 2 | 2 | DBSCAN, Spectral |

---

## 2. Préparation des données

### Principe

Avant d'appliquer tout algorithme de clustering, il est indispensable de :

1. **Normaliser** les données : ramener chaque feature à moyenne 0 et écart-type 1
2. **Réduire la dimension** pour la visualisation (PCA → 2D)

Chaque dataset utilise un objet `PCA` **distinct** pour éviter les conflits de dimensions lors de la projection des centroïdes.

### Code commenté

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets

# --- Chargement des datasets réels ---
iris   = datasets.load_iris()
wine   = datasets.load_wine()
digits = datasets.load_digits()

X_iris,   y_iris   = iris.data,   iris.target   # (150, 4)
X_wine,   y_wine   = wine.data,   wine.target   # (178, 13)
X_digits, y_digits = digits.data, digits.target # (1797, 64)

# --- Normalisation : StandardScaler met chaque feature à N(0,1) ---
scaler     = StandardScaler()
X_iris_s   = scaler.fit_transform(X_iris)   # moyenne=0, ecart-type=1
X_wine_s   = scaler.fit_transform(X_wine)
X_digits_s = scaler.fit_transform(X_digits)

# --- PCA 2D : un objet DISTINCT par dataset ---
# (les datasets n'ont pas le même nb de features : 4, 13, 64)
pca_iris   = PCA(n_components=2)
pca_wine   = PCA(n_components=2)
pca_digits = PCA(n_components=2)

X_iris_2d   = pca_iris.fit_transform(X_iris_s)     # (150, 2)
X_wine_2d   = pca_wine.fit_transform(X_wine_s)     # (178, 2)
X_digits_2d = pca_digits.fit_transform(X_digits_s) # (1797, 2)
```

### Données synthétiques (formes non-convexes)

```python
# Croissants de lune enchevêtrés → DBSCAN / Spectral
X_moons, _ = datasets.make_moons(n_samples=300, noise=0.05, random_state=42)

# Cercles concentriques → DBSCAN / Spectral
X_circles, _ = datasets.make_circles(n_samples=300, noise=0.05,
                                      factor=0.5, random_state=42)
```

---

## 3. K-Means

### Principe théorique

K-Means partitionne les données en **k clusters** en minimisant la **somme des variances intra-cluster** (WCSS — Within-Cluster Sum of Squares) :

```
WCSS = Σ_{i=1}^{k} Σ_{x ∈ Ci} ||x − μi||²
```

**Algorithme (Lloyd) :**

1. Initialiser k centroïdes aléatoirement (méthode `k-means++` par défaut)
2. Assigner chaque point au centroïde le plus proche
3. Recalculer les centroïdes (moyenne des points assignés)
4. Répéter jusqu'à convergence

**Complexité :** O(n × k × t × d) avec t = nb d'itérations, d = dimensions

**Hypothèses :** clusters convexes, isotropes et de taille comparable

| Hyperparamètre | Rôle |
|---------------|------|
| `n_clusters=k` | Nombre de clusters (à spécifier) |
| `n_init=10` | Nombre de réinitialisations aléatoires |
| `random_state` | Reproductibilité |

### Code commenté

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- Entraînement ---
kmeans    = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_km = kmeans.fit_predict(X_iris_s)  # retourne étiquette pour chaque point

# --- Évaluation ---
sil = silhouette_score(X_iris_s, labels_km)  # score de silhouette [-1, 1]
print(f"Silhouette : {sil:.3f}")             # attendu : ~0.46
print(f"WCSS      : {kmeans.inertia_:.2f}")

# --- Méthode Elbow : trouver le k optimal ---
inertias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_iris_s)
    inertias.append(km.inertia_)
# Tracer inertias vs k : le "coude" indique le k optimal (ici k=3)

# --- Projection des centroïdes en 2D ---
centers_2d = pca_iris.transform(kmeans.cluster_centers_)
# pca_iris déjà entraîné sur X_iris_s → projection cohérente
```

### Résultats attendus (Iris)

| Métrique | Valeur |
|----------|--------|
| Silhouette | **~0.46** |
| WCSS (k=3) | ~139 |
| Clusters trouvés | 3 |

> Le coude de la courbe Elbow apparaît clairement à **k=3**, cohérent avec les 3 espèces Iris.

---

## 4. K-Medoids (PAM)

### Principe théorique

K-Medoids est une variante robuste de K-Means où les centroïdes sont **des points réels du dataset** (les *médoïdes*), et non des moyennes. La distance minimisée est la distance L1 (Manhattan) :

```
coût = Σ_i Σ_{x ∈ Ci} d(x, mi)    avec mi ∈ X (médoïde)
```

**Avantage :** Très robuste aux outliers (un outlier ne peut pas être sélectionné comme médoïde s'il éloigne trop les autres).

**Algorithme PAM (Partitioning Around Medoids) :**

1. Initialiser k médoïdes aléatoirement parmi les points
2. Assigner chaque point au médoïde le plus proche
3. Tester chaque échange (médoïde, non-médoïde) → garder si amélioration
4. Répéter jusqu'à stabilité

**Complexité :** O(k × (n−k)²) par itération — plus coûteux que K-Means

| Vs K-Means | K-Medoids |
|-----------|-----------|
| Centroïde = moyenne | Centroïde = point réel |
| Sensible aux outliers | Robuste aux outliers |
| Distance euclidienne | Toute distance (Manhattan, etc.) |

### Code commenté

```python
# Nécessite : pip install scikit-learn-extra
from sklearn_extra.cluster import KMedoids

kmed        = KMedoids(n_clusters=3, random_state=42)
labels_kmed = kmed.fit_predict(X_iris_s)

sil = silhouette_score(X_iris_s, labels_kmed)
print(f"Silhouette : {sil:.3f}")

# kmed.cluster_centers_ contient les médoïdes (points réels du dataset)
medoid_2d = pca_iris.transform(kmed.cluster_centers_)
# Affichage : les médoïdes apparaissent comme des points rouges X sur le plot
```

### Résultats attendus (Iris)

| Métrique | Valeur |
|----------|--------|
| Silhouette | **~0.46** |
| Clusters | 3 |
| Médoïdes | 3 points réels du dataset |

> Résultat très proche de K-Means sur Iris (données propres, peu d'outliers).

---

## 5. K-Medians

### Principe théorique

K-Medians remplace le calcul de la **moyenne** (K-Means) par la **médiane** pour la mise à jour des centres. La distance utilisée est L1 (Manhattan) :

```
centre_j = median({ x | label(x) = j })
```

**Propriété clé :** La médiane est plus robuste que la moyenne face aux valeurs extrêmes (propriété de résistance statistique).

**Implémentation ici :** Manuelle en NumPy (pas dans scikit-learn standard).

### Code commenté

```python
def k_medians(X, k, max_iter=200, random_state=42):
    """
    K-Medians : centres = medianes composante par composante (distance L1).
    
    Args:
        X          : données normalisées (n_samples, n_features)
        k          : nombre de clusters
        max_iter   : limite d'itérations
        random_state : graine aléatoire
    Returns:
        labels  : étiquette de cluster pour chaque point
        centers : coordonnées des k medians
    """
    rng     = np.random.default_rng(random_state)
    centers = X[rng.choice(len(X), k, replace=False)].copy()  # init aléatoire
    labels  = np.zeros(len(X), dtype=int)

    for _ in range(max_iter):
        # ÉTAPE 1 : distance L1 de chaque point à chaque centre
        # dists[j, i] = Σ_d |X[i,d] - centers[j,d]|
        dists      = np.array([np.sum(np.abs(X - c), axis=1) for c in centers])
        new_labels = np.argmin(dists, axis=0)  # assigner au centre le plus proche

        if np.all(new_labels == labels):  # convergence
            break
        labels = new_labels

        # ÉTAPE 2 : mise à jour des centres via la MÉDIANE
        for j in range(k):
            mask = labels == j
            if mask.sum() > 0:
                centers[j] = np.median(X[mask], axis=0)  # mediane par colonne

    return labels, centers

# Application sur Iris normalisé
labels_km, centers = k_medians(X_iris_s, k=3)
sil = silhouette_score(X_iris_s, labels_km)
```

### Résultats attendus (Iris)

| Métrique | Valeur |
|----------|--------|
| Silhouette | **~0.45** |
| Clusters | 3 |

> Résultats similaires à K-Means (Iris est bien structuré), la différence se manifeste sur des données avec outliers.

---

## 6. DBSCAN

### Principe théorique

DBSCAN (*Density-Based Spatial Clustering of Applications with Noise*) identifie les clusters comme des **régions denses** séparées par des régions creuses. Il détecte automatiquement le bruit (outliers).

**Concepts clés :**

- **ε (epsilon)** : rayon de voisinage
- **MinPts** : nombre minimal de points dans un voisinage ε
- **Point core** : a ≥ MinPts voisins dans rayon ε
- **Point border** : dans le voisinage d'un core, mais < MinPts voisins
- **Point bruit** : ni core, ni border → étiqueté -1

**Algorithme :**

1. Sélectionner un point non visité
2. Si core → créer un cluster, propager récursivement à tous les densité-atteignables
3. Sinon → marquer comme bruit (provisoire)
4. Répéter jusqu'à traitement de tous les points

**Avantage majeur :** Détecte des clusters de **forme arbitraire** (spirales, croissants, etc.)

| Hyperparamètre | Rôle | Impact |
|---------------|------|--------|
| `eps` | Rayon de voisinage | Grand → moins de clusters |
| `min_samples` | MinPts | Grand → moins de clusters, plus de bruit |

### Code commenté

```python
from sklearn.cluster import DBSCAN

# --- Sur données synthétiques (Make Moons) ---
dbscan_moons  = DBSCAN(eps=0.2, min_samples=5)
labels_moons  = dbscan_moons.fit_predict(X_moons)
# eps=0.2 : adapté à la densité des lunes
# Points de bruit → label = -1

# --- Sur données réelles (Iris normalisé) ---
dbscan_iris   = DBSCAN(eps=0.5, min_samples=5)
labels_iris_db = dbscan_iris.fit_predict(X_iris_s)
# eps=0.5 : plus grand car espace 4D normalisé

# --- Extraction des métriques ---
n_clusters = len(set(labels_iris_db)) - (1 if -1 in labels_iris_db else 0)
n_noise    = (labels_iris_db == -1).sum()
print(f"Clusters : {n_clusters} | Bruit : {n_noise}")
```

### Résultats attendus

| Dataset | Eps | Clusters | Bruit |
|---------|-----|----------|-------|
| Moons | 0.2 | **2** | 0 |
| Circles | 0.2 | **2** | 0 |
| Iris | 0.5 | **2-3** | ~5 |

> DBSCAN excelle sur Moons et Circles là où K-Means échoue (formes non-convexes).

---

## 7. HDBSCAN

### Principe théorique

HDBSCAN (*Hierarchical DBSCAN*) étend DBSCAN en construisant une **hiérarchie de densité** au lieu d'utiliser un seuil ε fixe. Il extrait ensuite les clusters les plus **persistants** dans cette hiérarchie.

**Distance core mutuellement atteignable :**

```
d_mreach(a,b) = max(d_core(a), d_core(b), d(a,b))
```

**Étapes :**

1. Calculer les distances core pour tous les points
2. Construire un arbre couvrant minimal sur le graphe mreach
3. Condenser la hiérarchie → arbre de clusters
4. Extraire les clusters stables (ceux qui persistent longtemps)

**Avantages :** Robustesse à la densité variable, pas d'epsilon à choisir, probabilités d'appartenance.

| Hyperparamètre | Rôle |
|---------------|------|
| `min_cluster_size` | Taille minimale d'un cluster |
| `min_samples` | MinPts pour la densité core |

### Code commenté

```python
# Nécessite : pip install hdbscan
import hdbscan

hdb        = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
labels_hdb = hdb.fit_predict(X_iris_s)

# hdb.probabilities_ : probabilité [0,1] d'appartenance de chaque point
# 0 = potentiellement bruit, 1 = coeur du cluster
print(f"Clusters : {len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)}")
```

### Résultats attendus (Iris)

| Métrique | Valeur |
|----------|--------|
| Clusters détectés | **2-3** |
| Points bruit | ~5-10 |

---

## 8. OPTICS

### Principe théorique

OPTICS (*Ordering Points To Identify the Clustering Structure*) est une généralisation de DBSCAN qui traite les clusters de **densités variables**. Il produit un **reachability plot** qui révèle la structure hiérarchique.

**Distance d'atteignabilité :**

```
reach_dist(p, q) = max(d_core(q), d(p, q))
```

Les **vallées** dans le reachability plot correspondent aux clusters : plus la vallée est profonde, plus le cluster est dense.

| DBSCAN | OPTICS |
|--------|--------|
| Besoin d'un ε fixe | Pas d'ε (ou ε large) |
| Un seul niveau de densité | Densités variables |
| Résultat direct | Résultat via reachability plot |

### Code commenté

```python
from sklearn.cluster import OPTICS

optics        = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
labels_optics = optics.fit_predict(X_iris_s)
# xi=0.05 : méthode d'extraction des clusters (pente minimale de descente)
# min_cluster_size=0.1 : cluster = au moins 10% des points

# --- Reachability Plot ---
reachability = optics.reachability_[optics.ordering_]
# optics.ordering_ : ordre de visite des points (pas l'ordre original)
# Les vallées dans reachability révèlent les clusters

import matplotlib.pyplot as plt
plt.plot(np.arange(len(X_iris_s)), reachability, 'k-')
plt.ylabel("Distance de portée")
plt.title("Reachability Plot -- OPTICS")
```

### Résultats attendus (Iris)

| Métrique | Valeur |
|----------|--------|
| Clusters | **2-3** |
| Bruit | variable |

> Le reachability plot montre 2-3 vallées distinctes correspondant aux espèces Iris.

---

## 9. Clustering Hiérarchique Agglomératif (HAC)

### Principe théorique

HAC (*Hierarchical Agglomerative Clustering*) construit une arborescence de clusters (**dendrogramme**) en fusionnant successivement les paires les plus proches, du bas vers le haut.

**Algorithme :**

1. Départ : chaque point est son propre cluster
2. Trouver les 2 clusters les plus proches
3. Fusionner → nouveau cluster
4. Répéter jusqu'à n'avoir plus qu'un seul cluster
5. Couper le dendrogramme à la hauteur désirée pour obtenir k clusters

**Méthodes de liaison (linkage) :**

| Méthode | Formule de distance | Comportement |
|---------|---------------------|-------------|
| `ward` | Minimize l'augmentation de variance intra | Clusters équilibrés, convexes |
| `complete` | Max des distances inter-points | Clusters compacts |
| `average` | Moyenne des distances inter-points | Bon compromis |
| `single` | Min des distances inter-points | Sensible au "chaining" |

### Code commenté

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# --- 4 méthodes de liaison comparées ---
for method in ['ward', 'complete', 'average', 'single']:
    hac        = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels_hac = hac.fit_predict(X_iris_s)
    sil        = silhouette_score(X_iris_s, labels_hac)
    print(f"Liaison : {method:8s} | Silhouette : {sil:.3f}")

# --- Dendrogramme (sur sous-échantillon de 50 points) ---
# linkage() vient de scipy : construit la matrice de fusion complète
Z = linkage(X_iris_s[:50], method='ward')
# Z[i] = [cluster_a, cluster_b, distance, nb_points]

fig, ax = plt.subplots()
dendrogram(Z, ax=ax, no_labels=True,
           color_threshold=0.7 * max(Z[:, 2]))
# color_threshold : colore les branches en dessous de ce seuil
```

### Résultats attendus (Iris)

| Liaison | Silhouette |
|---------|-----------|
| `ward` | **~0.48** |
| `complete` | ~0.45 |
| `average` | ~0.47 |
| `single` | ~0.22 |

> La méthode `ward` donne les meilleurs résultats sur Iris. `single` est sensible au chaining.

---

## 10. Gaussian Mixture Models (GMM)

### Principe théorique

GMM modélise les données comme un **mélange de k distributions gaussiennes**. Chaque point appartient à chaque cluster avec une **probabilité** (clustering mou).

**Densité du mélange :**

```
p(x) = Σ_{k=1}^{K} π_k × N(x | μ_k, Σ_k)
```

avec π_k = poids de la composante k, N = gaussienne de moyenne μ_k et covariance Σ_k.

**Algorithme EM (Expectation-Maximization) :**

- **E-step :** Calculer r_{ik} = P(point i ∈ cluster k) pour tous i, k
- **M-step :** Mettre à jour π_k, μ_k, Σ_k en maximisant la vraisemblance
- Répéter jusqu'à convergence

**Types de covariance :**

| Type | Description | Forme des clusters |
|------|-------------|-------------------|
| `full` | Matrice complète par composante | Ellipses quelconques |
| `tied` | Matrice identique pour toutes | Ellipses mêmes orientation |
| `diag` | Matrice diagonale | Ellipses axes-alignées |
| `spherical` | Variance scalaire | Sphères de tailles différentes |

**Sélection du nombre de composantes :**

- **BIC** (Bayesian Information Criterion) : pénalise la complexité
- **AIC** (Akaike) : pénalité moins forte → favorise plus de composantes

### Code commenté

```python
from sklearn.mixture import GaussianMixture

# --- 4 types de covariance comparés (dataset Wine) ---
for cov_type in ['full', 'tied', 'diag', 'spherical']:
    gmm       = GaussianMixture(n_components=3, covariance_type=cov_type,
                                random_state=42)
    label_gmm = gmm.fit_predict(X_wine_s)
    sil       = silhouette_score(X_wine_s, label_gmm)
    bic       = gmm.bic(X_wine_s)    # Bayesian Information Criterion
    aic       = gmm.aic(X_wine_s)    # Akaike Information Criterion
    print(f"{cov_type:10s} | Sil : {sil:.3f} | BIC : {bic:.0f}")

# --- Sélection du k optimal par BIC ---
bics = []
for n in range(2, 11):
    gm = GaussianMixture(n_components=n, random_state=42).fit(X_wine_s)
    bics.append(gm.bic(X_wine_s))
# k optimal = argmin(BIC) → généralement k=3 pour Wine
```

### Résultats attendus (Wine)

| Covariance | Silhouette | BIC |
|-----------|-----------|-----|
| `full` | **~0.28** | le plus bas |
| `tied` | ~0.27 | |
| `diag` | ~0.26 | |
| `spherical` | ~0.15 | le plus haut |

> Le BIC identifie correctement **k=3** pour le dataset Wine (3 types de vins).

---

## 11. Spectral Clustering

### Principe théorique

Spectral Clustering utilise la **structure spectrale** (valeurs propres) du graphe de similarité pour effectuer une réduction de dimension, puis applique K-Means dans l'espace réduit.

**Étapes :**

1. Construire la **matrice de similarité** W (via RBF kernel ou k-NN)
2. Calculer le **Laplacien** L = D − W (D = matrice degré)
3. Extraire les k **vecteurs propres** de plus petites valeurs propres
4. Appliquer **K-Means** sur la représentation spectrale

**Formule kernel RBF :**

```
W_ij = exp(−||xi − xj||² / (2σ²))     avec σ = 1/√(2γ)
```

**Avantage :** Fonctionne sur des clusters **non-convexes** (cercles, spirales).

| Affinité | Usage |
|----------|-------|
| `rbf` | Données denses, bonne connaissance du σ |
| `nearest_neighbors` | Données sparses, plus robuste |

### Code commenté

```python
from sklearn.cluster import SpectralClustering

# --- Sur données circulaires (non-convexes) ---
spec_circles = SpectralClustering(
    n_clusters=2,
    affinity='rbf',        # kernel gaussien : W_ij = exp(-gamma * ||xi-xj||²)
    gamma=1.0,             # paramètre du kernel (= 1/2σ²)
    random_state=42
)
labels_sc_circ = spec_circles.fit_predict(X_circles)
# K-Means échouerait ici, Spectral réussit car il opère dans l'espace spectral

# --- Sur Iris (k-NN graph) ---
spec_iris = SpectralClustering(
    n_clusters=3,
    affinity='nearest_neighbors',  # W_ij = 1 si xj dans les k-NN de xi
    n_neighbors=10,
    random_state=42
)
labels_sc_iris = spec_iris.fit_predict(X_iris_s)
sil = silhouette_score(X_iris_s, labels_sc_iris)
print(f"Silhouette Iris : {sil:.3f}")   # attendu : ~0.45
```

### Résultats attendus

| Dataset | Clusters | Silhouette |
|---------|----------|-----------|
| Circles | **2** ✓ | — |
| Iris | **3** | **~0.45** |

> Spectral Clustering sépare parfaitement les cercles concentriques (K-Means échoue sur cette géométrie).

---

## 12. Affinity Propagation

### Principe théorique

Affinity Propagation détermine automatiquement le **nombre de clusters** en faisant passer des messages entre points. Chaque point peut devenir **exemplaire** (représentant de cluster).

**Deux types de messages :**

- **Responsabilité r(i,k)** : à quel point le point k est qualifié pour être l'exemplaire du point i
- **Disponibilité a(i,k)** : à quel point il est approprié pour le point i de choisir k comme exemplaire

**Algorithme :**

1. Initialiser les affinités s(i,k) = −||xi − xk||²
2. Mettre à jour r(i,k) et a(i,k) en alternance (message passing)
3. Identifier les exemplaires : argmax_{k} r(i,k) + a(i,k)
4. Répéter jusqu'à stabilité

**Paramètre `damping`** : facteur d'amortissement ∈ [0.5, 1) pour éviter les oscillations.

| Avantage | Inconvénient |
|----------|-------------|
| k déterminé auto. | Plus lent (O(n²)) |
| Pas d'init. aléatoire | Peut trouver trop de clusters |
| Exemplaires = vrais points | Sensible au paramètre damping |

### Code commenté

```python
from sklearn.cluster import AffinityPropagation

ap = AffinityPropagation(
    damping=0.9,      # amortissement (0.9 = fort → convergence stable)
    max_iter=500,     # limite d'itérations
    random_state=42
)
labels_ap = ap.fit_predict(X_iris_s)

# Nombre de clusters détectés automatiquement
n_clusters = len(ap.cluster_centers_indices_)
# ap.cluster_centers_indices_ : indices des exemplaires dans X_iris_s
print(f"Clusters auto : {n_clusters}")   # attendu : 7-8 (plus que les 3 réelles)
print(f"Silhouette    : {silhouette_score(X_iris_s, labels_ap):.3f}")

# ap.n_iter_ : nombre d'itérations jusqu'à convergence
print(f"Convergence en {ap.n_iter_} itérations")
```

### Résultats attendus (Iris)

| Métrique | Valeur |
|----------|--------|
| Clusters détectés | **7–9** (tendance à sur-segmenter) |
| Silhouette | **~0.45** |
| Convergence | ~100–200 itérations |

> A.P. tend à créer plus de clusters que prévu (damping=0.9 réduit ce phénomène). Idéal quand on ne connaît pas k.

---

## 13. Self-Organizing Maps (SOM)

### Principe théorique

Les SOM (*cartes auto-organisatrices*) sont un réseau de neurones non supervisé qui projette des données **haute dimension** sur une grille 2D en préservant la topologie.

**Structure :** Grille de neurones NxN, chaque neurone a un vecteur de poids w_i ∈ ℝ^d.

**Algorithme d'apprentissage :**

1. Pour chaque point x, trouver le **BMU** (Best Matching Unit) : neurone dont w_i est le plus proche de x
2. Mettre à jour le BMU et ses voisins :

```
w_i ← w_i + α(t) × h(i, BMU, t) × (x − w_i)
```

avec α = learning rate décroissant, h = fonction de voisinage gaussienne

**U-Matrix :** Visualise les distances entre neurones voisins → frontières = zones creuses (sombres).

### Code commenté

```python
# Nécessite : pip install minisom
from minisom import MiniSom

SOM_SIZE = 7  # grille 7×7 = 49 neurones
som = MiniSom(
    SOM_SIZE, SOM_SIZE,          # dimensions de la grille
    X_wine_s.shape[1],           # dimension des données = 13 (Wine)
    sigma=1.5,                   # rayon de voisinage initial
    learning_rate=0.5,           # taux d'apprentissage initial
    random_seed=42
)
som.random_weights_init(X_wine_s)  # initialisation des poids par des données
som.train_random(X_wine_s, 1000, verbose=False)  # 1000 itérations aléatoires

# BMU pour chaque point : coordonnée (row, col) sur la grille
winners    = np.array([som.winner(x) for x in X_wine_s])  # (178, 2)
labels_som = winners[:, 0] * SOM_SIZE + winners[:, 1]     # -> id unique par cellule

# U-Matrix : distance moyenne entre neurones adjacents
umatrix = som.distance_map()  # forme : (SOM_SIZE, SOM_SIZE)
# Visualisation : zones sombres = frontières entre clusters
```

### Résultats attendus (Wine, grille 7×7)

| Métrique | Valeur |
|----------|--------|
| Cellules activées | ~25–35 (sur 49 possibles) |
| Séparation visible | 3 zones dans la U-Matrix |

> La U-Matrix révèle clairement les **3 régions** correspondant aux 3 types de vin, avec des frontières sombres entre elles.

---

## 14. UMAP + K-Means

### Principe théorique

**UMAP** (*Uniform Manifold Approximation and Projection*) est un algorithme de réduction dimensionnelle non-linéaire basé sur la théorie des catégories et la géométrie riemannienne.

**Idée principale :** Construire une représentation par graphe fuzzy en haute dimension, puis en optimiser une représentation fidèle en basse dimension.

Avantages sur PCA :

- Préserve la **structure locale ET globale**
- Bien plus efficace que t-SNE sur de grands datasets
- La représentation 2D est utilisable pour le clustering

**Pipeline :**

1. UMAP réduit 64 dimensions (Digits) → 2 dimensions
2. K-Means est appliqué sur l'espace UMAP (plus séparable)

| UMAP | t-SNE |
|------|-------|
| Plus rapide | Plus lent |
| Structure globale préservée | Structure locale uniquement |
| Reproductible | Moins reproductible |

### Code commenté

```python
# Nécessite : pip install umap-learn
import umap

# --- Réduction dimensionnelle 64D → 2D ---
reducer = umap.UMAP(
    n_components=2,    # dimension cible
    n_neighbors=15,    # taille du voisinage local (15 = balance local/global)
    min_dist=0.1,      # distance minimale dans l'espace réduit (0.1 = compact)
    random_state=42
)
X_digits_umap = reducer.fit_transform(X_digits_s)  # (1797, 2)

# --- K-Means sur l'espace UMAP ---
# Les 10 chiffres (0-9) forment des clusters bien séparés dans l'espace UMAP
kmeans_umap   = KMeans(n_clusters=10, random_state=42, n_init=10)
labels_umap   = kmeans_umap.fit_predict(X_digits_umap)

# --- Comparaison silhouette espacee original vs UMAP ---
sil_raw  = silhouette_score(X_digits_s,    labels_umap)  # dans espace 64D
sil_umap = silhouette_score(X_digits_umap, labels_umap)  # dans espace 2D
print(f"Silhouette (64D original) : {sil_raw:.3f}")   # attendu : ~0.15
print(f"Silhouette (2D UMAP)      : {sil_umap:.3f}")  # attendu : ~0.65
```

### Résultats attendus (Digits)

| Espace | Silhouette |
|--------|-----------|
| Original (64D) | **~0.15–0.20** (clusters proches) |
| UMAP (2D) | **~0.60–0.70** (clusters bien séparés) |

> UMAP visualise spectaculairement les **10 îlots** correspondant aux chiffres 0-9 — beaucoup plus nets qu'avec PCA.

---

## 15. Comparaison globale

### Tableau récapitulatif (Dataset Iris normalisé)

| Algorithme | k détecté | Bruit | Silhouette | Remarque |
|-----------|-----------|-------|-----------|---------|
| **K-Means** | 3 | 0 | **~0.46** | Référence |
| **K-Medians** | 3 | 0 | **~0.45** | Robuste outliers |
| **HAC Ward** | 3 | 0 | **~0.48** | Meilleur silhouette |
| **GMM full** | 3 | 0 | **~0.28** | Clustering mou |
| **Spectral** | 3 | 0 | **~0.45** | Formes complexes |
| **DBSCAN** | 2–3 | ~5 | **~0.50** | Détection bruit |
| **OPTICS** | 2–3 | variable | ~0.40 | Densités variables |
| **Affinity Prop.** | 7–9 | 0 | ~0.45 | k automatique |

### Synthèse des choix algorithmiques

| Situation | Algorithme recommandé |
|-----------|----------------------|
| k connu, clusters sphériques | **K-Means** |
| k connu, robustesse outliers | **K-Medoids** |
| k connu, clusters gaussiens | **GMM** |
| k connu, forme quelconque | **Spectral Clustering** |
| k inconnu, détection bruit | **DBSCAN / HDBSCAN** |
| k inconnu, densités variables | **OPTICS / HDBSCAN** |
| k inconnu, pas de contrainte | **Affinity Propagation** |
| Haute dimension + visualisation | **UMAP + K-Means** |
| Données catégorielles/mixtes | **K-Medoids** |

### Axes de comparaison

```
                 FORME DES CLUSTERS
                 Non-convexe
                      │
         OPTICS   HDBSCAN  Spectral
              \      │      /
               \     │     /
   K inconnu ───────────────── K connu
               /     │     \
              /      │      \
          AP      DBSCAN   K-Means
                           HAC
                      │
                  Convexe
```

---

## Dépendances et installation

```bash
# Librairies de base (incluses dans scikit-learn)
pip install scikit-learn scipy numpy matplotlib

# Librairies optionnelles (algorithmes 2, 5, 11, 12)
pip install scikit-learn-extra   # K-Medoids
pip install hdbscan              # HDBSCAN
pip install minisom              # Self-Organizing Maps
pip install umap-learn           # UMAP
```

---

## Exécution

```bash
python clustering_examples.py
```

Les graphiques apparaissent séquentiellement. Fermer chaque fenêtre pour passer à l'algorithme suivant.

---

*Généré automatiquement à partir de `clustering_algorithms.ipynb` — Mars 2026*
