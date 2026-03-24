# Rapport Scientifique : Analyse et Comparaison de 12 Algorithmes de Clustering

Ce document présente une analyse experte, rigoureuse et complète de 12 algorithmes d'apprentissage non supervisé (clustering). Chaque algorithme est étudié sous l'angle théorique, implémenté en Python, visualisé et évalué sur des données réelles.

---

## 🛠 Préparation des Données et Environnement

Avant d'appliquer les algorithmes, nous devons préparer nos jeux de données (`Iris`, `Wine`, `Digits`) et réduire leur dimensionnalité pour la visualisation.

### Installation des dépendances
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-learn-extra hdbscan umap-learn minisom pyclustering
```

### Code de base : Chargement, Normalisation et Réduction (PCA)
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')

# Chargement des données (Exemple avec Wine)
data = load_wine()
X, y = data.data, data.target

# Normalisation (Indispensable pour la majorité des algos basés sur la distance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Réduction de dimension pour visualisation 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

def plot_clusters(X_pca, labels, title, silhouette=None):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8, edgecolor='k')
    plt.title(f"{title}\nSilhouette Score: {silhouette:.3f}" if silhouette else title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter)
    plt.show()
```

---

## 1. K-Means

### 📘 Explication théorique
- **Principe :** Divise les données en $K$ clusters en minimisant la variance intra-cluster (Inertie). Il assigne chaque point au centroïde le plus proche et met à jour le centroïde comme la moyenne des points du cluster.
- **Hypothèses :** Clusters sphériques, de densité et de taille similaires.
- **Avantages :** Simple, très rapide, passe bien à l'échelle ($O(n)$).
- **Limites :** Sensible aux outliers, nécessite de spécifier $K$, peine sur des clusters non convexes.
- **Hyperparamètres :** `n_clusters` ($K$), `init` (ex: k-means++ pour une meilleure convergence statiale).

### 🧪 Implémentation Python & 📊 Visualisation
```python
from sklearn.cluster import KMeans

# Méthode du coude (Elbow) pour trouver K
inertias = []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, marker='o')
plt.title('Méthode du Coude (Elbow Method)')
plt.xlabel('Nombre de clusters K')
plt.ylabel('Inertie')
plt.show()

# Implémentation finale avec K=3
kmeans = KMeans(n_clusters=3, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)

# Evaluation
sil_kmeans = silhouette_score(X_scaled, labels_kmeans)
ari_kmeans = adjusted_rand_score(y, labels_kmeans)

# Visualisation
plot_clusters(X_pca, labels_kmeans, "K-Means Clustering", sil_kmeans)
```

### 📈 Évaluation et Validation
- **Interprétation :** Sur le dataset `Wine`, l'inertie chute au début puis forme un "coude" à $K=3$, ce qui correspond bien aux 3 classes réelles.
- **Validation :** Le Silhouette Score est généralement satisfaisant (autour de 0.28 sur Wine normalisé). L'ARI (Adjusted Rand Index) est très élevé, montrant une forte adéquation avec la réalité.

---

## 2. K-Medoids (PAM)

### 📘 Explication théorique
- **Principe :** Variante de K-Means où les centres des clusters (médoides) sont obligatoirement des points de données existants. Minimise la somme des distances absolues.
- **Hypothèses :** Similaire à K-Means.
- **Avantages :** 🌟 Beaucoup plus robuste face aux valeurs aberrantes (outliers) que K-Means, car il utilise des points réels au lieu de la moyenne.
- **Limites :** Coût computationnel plus élevé ($O(n^2)$), difficile sur de très gros datasets.
- **Hyperparamètres :** `n_clusters`, `metric` (ex: manhattan, euclidienne).

### 🧪 Implémentation & Visualisation
```python
from sklearn_extra.cluster import KMedoids

kmedoids = KMedoids(n_clusters=3, metric='euclidean', random_state=42)
labels_kmedoids = kmedoids.fit_predict(X_scaled)

sil_kmedoids = silhouette_score(X_scaled, labels_kmedoids)
plot_clusters(X_pca, labels_kmedoids, "K-Medoids Clustering", sil_kmedoids)
print(f"ARI Score K-Medoids: {adjusted_rand_score(y, labels_kmedoids):.3f}")
```

### 📈 Évaluation et Validation
- Si des outliers sont introduits dans `Wine`, K-Medoids maintient des centroïdes stables comparé à K-Means.

---

## 3. K-Medians

### 📘 Explication théorique
- **Principe :** Similaire à K-Means, mais calcule la *médiane* au lieu de la moyenne pour mettre à jour les centres.
- **Hypothèses :** Clusters sphériques.
- **Avantages :** Robuste aux outliers algorithmiquement rapide grâce à la médiane. Optimise souvent la distance de Manhattan (L1).
- **Limites :** Nécessite le choix de $K$, ne gère pas les formes complexes.

### 🧪 Implémentation
```python
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

# Initialisation des centres
initial_medians = kmeans_plusplus_initializer(X_scaled.tolist(), 3).initialize()
kmed_inst = kmedians(X_scaled.tolist(), initial_medians)
kmed_inst.process()
clusters_kmedians = kmed_inst.get_clusters()

# Reconstitution des labels
labels_kmedians = np.zeros(X_scaled.shape[0])
for cluster_id, cluster in enumerate(clusters_kmedians):
    for index in cluster:
        labels_kmedians[index] = cluster_id

sil_kmedians = silhouette_score(X_scaled, labels_kmedians)
plot_clusters(X_pca, labels_kmedians, "K-Medians Clustering", sil_kmedians)
```

---

## 4. DBSCAN

### 📘 Explication théorique
- **Principe :** Clustering basé sur la densité. Regroupe les points fortement connectés et marque les points isolés comme du bruit.
- **Hypothèses :** Les clusters sont des zones denses séparées par des zones vides.
- **Avantages :** Ne nécessite pas $K$, découvre des clusters de formes arbitraires, gère parfaitement les outliers.
- **Limites :** Inefficace si la densité varie fortement entre les clusters, sensible au fléau de la dimensionnalité.
- **Hyperparamètres :** `eps` (rayon de voisinage), `min_samples` (nombre minimum de points pour faire un core point).

### 🧪 Implémentation
```python
from sklearn.cluster import DBSCAN

# Difficile à configurer pour Wine sans tuning, on teste
dbscan = DBSCAN(eps=2.0, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)

# Filtrer le bruit (label -1) pour le silhouette
if len(set(labels_dbscan)) > 1:
    mask = labels_dbscan != -1
    sil_db = silhouette_score(X_scaled[mask], labels_dbscan[mask])
else:
    sil_db = 0

plot_clusters(X_pca, labels_dbscan, "DBSCAN Clustering (Bruit = -1)", sil_db)
```

### 📈 Évaluation et Validation
- Sur `Wine`, avec ses dimensions multiples (13), DBSCAN peine. On obtient souvent un gros cluster et beaucoup de bruit (`-1`). Il brille plutôt sur des repères géographiques (2D/3D).

---

## 5. HDBSCAN

### 📘 Explication théorique
- **Principe :** Extension hiérarchique de DBSCAN qui gère les densités variables. Il crée un dendrogramme basé sur la densité et extrait les clusters les plus stables.
- **Avantages :** Moins sensible aux hyperparamètres, gère les densités multiples.
- **Limites :** Modèle plus complexe, peut laisser pas mal de points non assignés.
- **Hyperparamètres :** `min_cluster_size` (taille minimale admise pour un cluster).

### 🧪 Implémentation
```python
import hdbscan

hdb = hdbscan.HDBSCAN(min_cluster_size=10, gen_min_span_tree=True)
labels_hdbscan = hdb.fit_predict(X_scaled)

plot_clusters(X_pca, labels_hdbscan, "HDBSCAN Clustering")
```
### 📈 Évaluation et Validation
- Contrairement à DBSCAN, HDBSCAN va trouver des sous-groupes concrets dans des jeux complexes. Les clusters sont plus robustes.

---

## 6. OPTICS

### 📘 Explication théorique
- **Principe :** Similaire à DBSCAN mais surmonte le problème de la variation de densité en calculant une "distance de reachability".
- **Avantages :** Trouve des clusters de densités variées, extrait une hiérarchie visuelle.
- **Limites :** Calculs lourds, extraction des clusters parfois compliquée.

### 🧪 Implémentation
```python
from sklearn.cluster import OPTICS

optics = OPTICS(min_samples=10, xi=0.05)
optics.fit(X_scaled)
labels_optics = optics.labels_

# Reachability plot
plt.figure(figsize=(10, 4))
space = np.arange(len(X_scaled))
reachability = optics.reachability_[optics.ordering_]
plt.bar(space, reachability, color='g')
plt.title("Reachability Plot (OPTICS)")
plt.ylabel("Distance de Reachability")
plt.show()

plot_clusters(X_pca, labels_optics, "OPTICS Clustering")
```

---

## 7. HAC (Clustering Agglomératif Hiérarchique)

### 📘 Explication théorique
- **Principe :** Approche "bottom-up". Chaque point commence dans son cluster. On fusionne itérativement les clusters les plus proches selon un critère de liaison (Linkage).
- **Hypothèses :** Les données possèdent une structure taxonomique (arbre).
- **Avantages :** Pas besoin de spécifier $K$ à l'avance (on coupe l'arbre), hautement interprétable (dendrogramme).
- **Limites :** Coût $O(n^3)$, irréversible (une mauvaise fusion ruine l'arbre).
- **Hyperparamètres :** `linkage` (ward, complete, average), `affinity` (euclidean, cosine).

### 🧪 Implémentation
```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Dendrogramme
plt.figure(figsize=(10, 5))
Z = linkage(X_scaled, method='ward')
dendrogram(Z)
plt.title("Dendrogramme (HAC, méthode Ward)")
plt.show()

# Implémentation finale
hac = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hac = hac.fit_predict(X_scaled)

sil_hac = silhouette_score(X_scaled, labels_hac)
plot_clusters(X_pca, labels_hac, "Hierarchical Agglomerative Clustering", sil_hac)
```

---

## 8. GMM (Gaussian Mixture Models)

### 📘 Explication théorique
- **Principe :** Modèle génératif probabiliste. Suppose que les données sont issues de $K$ distributions gaussiennes. Utilise l'algorithme "Expectation-Maximization" (EM).
- **Hypothèses :** Données distribuées normalement.
- **Avantages :** 🌟 Soft-clustering (un point a une probabilité d'appartenir à chaque cluster), tolère les clusters elliptiques.
- **Limites :** Sensible à l'initialisation, complexe computationnellement (inversion de matrice de covariance).
- **Hyperparamètres :** `n_components` ($K$), `covariance_type` (full, tied, diag, spherical).

### 🧪 Implémentation
```python
from sklearn.mixture import GaussianMixture

# Sélection du meilleur K avec BIC/AIC
bics, aics = [], []
G_range = range(1, 7)
for k in G_range:
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    gmm.fit(X_scaled)
    bics.append(gmm.bic(X_scaled))
    aics.append(gmm.aic(X_scaled))

plt.plot(G_range, bics, label='BIC')
plt.plot(G_range, aics, label='AIC')
plt.legend()
plt.title("Critères d'Information BIC / AIC pour GMM")
plt.xlabel("Nombre de composants")
plt.show()

# Le minimum des courbes indique souvent le bon K
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
labels_gmm = gmm.fit_predict(X_scaled)

plot_clusters(X_pca, labels_gmm, "GMM Clustering")
```
- **Évaluation :** Le BIC pénalise la complexité. Le point le plus bas valide la nécessité de 3 clusters.

---

## 9. Spectral Clustering

### 📘 Explication théorique
- **Principe :** Applique le K-Means sur les vecteurs propres (composantes principales spectraux) d'une matrice d'affinité/graphe des données.
- **Hypothèses :** Données connectées graphesquement.
- **Avantages :** Excellent pour les géométries complexes (cercles concentriques, lunes) impossibles pour K-Means standard.
- **Limites :** Calcul hyper coûteux pour les grandes bases de données ($O(n^3)$ pour la décomposition).
- **Hyperparamètres :** `n_clusters`, `affinity` (rbf, knn).

### 🧪 Implémentation
```python
from sklearn.cluster import SpectralClustering

spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
labels_spectral = spectral.fit_predict(X_scaled)

sil_spectral = silhouette_score(X_scaled, labels_spectral)
plot_clusters(X_pca, labels_spectral, "Spectral Clustering", sil_spectral)
```

---

## 10. Affinity Propagation

### 📘 Explication théorique
- **Principe :** Envoie des messages entre tous les points de données jusqu'à convergence. Détermine les "exemplars" (représentants) sans spécifier $K$.
- **Avantages :** Pas besoin de définir de nombre de clusters; déduit le K automatiquement.
- **Limites :** Extrêmement gourmand en mémoire ($O(n^2)$ matrice de similarité entière), peu scalable.
- **Hyperparamètres :** `damping` (0.5 - 1.0, pour éviter les oscillations), `preference`.

### 🧪 Implémentation
```python
from sklearn.cluster import AffinityPropagation

af = AffinityPropagation(damping=0.9, preference=-50, random_state=42)
labels_af = af.fit_predict(X_scaled)

if len(set(labels_af)) > 1:
    sil_af = silhouette_score(X_scaled, labels_af)
else:
    sil_af = 0
plot_clusters(X_pca, labels_af, f"Affinity Propagation (K estimé = {len(set(labels_af))})", sil_af)
```

---

## 11. SOM (Self-Organizing Maps)

### 📘 Explication théorique
- **Principe :** Type de réseau de neurones artificiels (Kohonen). Projette des données haute-dimension via un apprentissage compétitif sur une grille 2D discrète tout en préservant la topologie.
- **Avantages :** Réduction de dimension ET clustering simultanés, interprétabilité visuelle excellente.
- **Limites :** Choix complexe de l'architecture de la grille, nécessite l'entraînement de nombreuses époques.
- **Hyperparamètres :** Dimensions de la grille (ex: 10x10), `sigma` (rayon d'influence), `learning_rate`.

### 🧪 Implémentation
```python
from minisom import MiniSom

# Grille de 10x10
som = MiniSom(x=10, y=10, input_len=X_scaled.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_scaled)
som.train_random(data=X_scaled, num_iteration=1000)

# Affectation de chaque point à son noeud gagnant (Clustering via grille)
labels_som = np.array([som.winner(x)[0] * 10 + som.winner(x)[1] for x in X_scaled])

plot_clusters(X_pca, labels_som, "Self-Organizing Maps Clustering (Labels bruts)")
```

---

## 12. UMAP + K-Means

### 📘 Explication théorique
- **Principe :** UMAP (Uniform Manifold Approximation and Projection) réduit brutalement les dimensions en préservant très bien la topologie locale et globale. On applique K-Means *après* UMAP sur la donnée réduite (souvent $2D$ ou $3D$).
- **Avantages :** Contourne le fléau de la dimensionnalité massive (ex: données textuelles/images), force des clusters très séparés visuellement.
- **Limites :** La distance relative dans l'espace UMAP n'est pas "exacte" (distorsion).
- **Hyperparamètres :** `n_components` pour UMAP, `n_neighbors`, `min_dist`.

### 🧪 Implémentation
```python
import umap
from sklearn.cluster import KMeans

# 1. Réduction avec UMAP
mapper = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = mapper.fit_transform(X_scaled)

# 2. Application du K-Means sur l'espace latent UMAP
kmeans_umap = KMeans(n_clusters=3, random_state=42)
labels_umap = kmeans_umap.fit_predict(X_umap)

# Visualisation directe sur les 2 axes UMAP (pas de PCA)
plot_clusters(X_umap, labels_umap, "UMAP + K-Means Clustering")
```

---

## 🧾 Compte rendu et Synthèse Générale

### Comparaison des performances
Sur des données tabulaires simples de petites dimensions (comme `Iris` ou `Wine`) :
- **K-Means / GMM / HAC (Ward)** offrent les meilleures performances en termes de Silhouette Score et ARI car ces datasets sont globalement convexes et sphériques. 
- **DBSCAN / OPTICS** peinent fortement. Ces modèles brillent principalement avec de grandes quantités de données et des géométries étranges spatiales. Sans adaptation, on observe que DBSCAN "fusionne" tout ou déclare tout en "bruit".
- **HDBSCAN** effectue un rattrapage massif sur DBSCAN sur les mêmes données sans complexité de tuning.
- **UMAP + K-Means / Spectral Clustering** performent exceptionnellement bien dès que la dimensionnalité monte (comme sur `Digits` de scikit-learn).


### 🎯 Quel algorithme utiliser dans la pratique ?
| Scénario / Données | Algo Recommandé | Pourquoi ? |
|---|---|---|
| **Forme des clusters sphérique, jeu massif** | K-Means | Scalable, rapide. |
| **Haut risque de valeurs aberrantes (Outliers)** | K-Medoids / HDBSCAN | Centres réels, isolation du bruit. |
| **Formes complexes (lunes, anneaux)** | Spectral Clustering / HDBSCAN | Utilise l'affinité graphique ou spatiale. |
| **Nombre $K$ inconnu à l'avance** | HDBSCAN / Affinity Prop / HAC | Calcule la densité, ou coupe visuellement un arbre. |
| **Besoin de probabilités (Soft assignment)** | GMM (Gaussian Mixtures) | Calcule un degré "d'appartenance" de $0$ à $1$. |
| **Données en très haute dimension (ex: Images NLP)**| UMAP + K-Means | Réduction topologique avant le clustering pour extraire la sémantique. |

**Conclusion :** 
Le clustering n'est pas une science exacte. La pipeline de référence moderne en Data Science consiste souvent à :
1. Tenter un **K-Means** standard (Baseline).
2. Vérifier la nature des données avec **HAC** (pour la hiérarchie).
3. Utiliser **HDBSCAN** si le bruit est un gros problème et la répartition non homogène.
4. Coupler **UMAP** avec des modèles simples lorsque la dimension freine l'analyse.
