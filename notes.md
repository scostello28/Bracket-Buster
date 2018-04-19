1. How many games could I predict if I just picked favorites?
2. SOS per year
3. player data df creator
4. cluster players
5. include players who played more than a specific amount per game
6. Cluster player archetypes into team composition clusters
7. Alter to point spread predictor (Classification to regression)


Data:
1. SOS per year (done)
2. player data (done)
  - pulling position data for LDA

Clustering:
1. Dimensionality Reduction:
  - reduce dimensionality with PCA, t-SNE, SV D, and LDA
  - get position for prior for LDA
  - figure out of to better separate
    - which features to keep?
2. Clustering
  - Use K-means, and EM
3. Cluster on data without dimension reduction
  - Cluster again and compare results with higher dimensional clustering
  - Make sick plot!


NMF non matrix factorization - dimensionality reduction and clustering combined
