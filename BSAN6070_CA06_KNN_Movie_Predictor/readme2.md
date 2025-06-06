# Machine Learning Final Exam Study Guide: kNN and Clustering Concepts (Expanded)

## 1. k-Nearest Neighbors (kNN) Algorithm

### 1.1 Core Concepts
- Distance Metrics:
  * Role in kNN: Determines similarity between instances
    - Different metrics may yield different neighbor sets
    - Choice depends on data type (continuous, binary, text)
  * Common metrics: Euclidean, Manhattan, Minkowski, Cosine
    - Euclidean: Straight-line distance, sensitive to scale
    - Cosine: Measures angle between vectors, good for text
    - We used kd_tree when working with binary & continuous data (8-Dimensional). Would not use this for high-dimensional data or for text.
    - Ball Tree is good for high dimensional data or non-Euclidean metrics.
    - - - Text Data (TF-IDF vectors with 1000+ dimensions)
        - Image Embeddings, Genomics data

- Choosing k Value:
  * Bias-variance tradeoff (small k vs large k)
    - Small k: High variance, captures noise (overfitting)
    - Large k: High bias, smoother decision boundaries
  * Computational cost implications
    - Larger k requires more distance calculations
    - Impacts prediction time more than training time
  * Odd vs even k in binary classification
    - Prevents tie votes in binary classification
    - Especially important with balanced classes

### 1.2 Practical Considerations
- Class Imbalance:
  * Majority class dominance in voting
    - Nearest neighbors may ignore minority class
    - Can lead to always predicting majority class
  * Potential solutions: weighted voting, sampling techniques
    - Weighted voting: Closer neighbors get more vote weight
    - Oversampling minority or undersampling majority class

- Feature Scaling:
  * Why necessary: distance-based algorithms are scale-sensitive
    - Features with larger scales dominate distance calculations
    - Example: Age (0-100) vs Income (0-1,000,000)
  * Consequences of not scaling
    - Features with larger ranges dominate neighbor selection
    - May lead to suboptimal performance

- Regression Adaptation:
  * Average/weighted average of neighbors' values
    - Simple average for unweighted version
    - Distance-weighted average gives more influence to closer points
  * Distance-weighted predictions
    - Inverse distance weighting common (1/d)
    - Helps smooth predictions
      
  * When to use KNN Regression for Tabular Data
     - Pros : Simple, no assumptions about data, handles non-linear relationships
     - Cons : Slow for large datasets, requires careful encoding of categoricals, sensitive to irrelevant features.

### 1.3 Advanced Applications
- Recommendation Systems:
  * User/item similarity measurement
    - Users as points in feature space
    - Items purchased/rated as features
  * Collaborative filtering approaches
    - User-user: Find similar users, recommend their liked items
    - Item-item: Find similar items to those user liked

### 1.4 Comparative Analysis
- vs Model-based Algorithms:
  * Lazy vs eager learning
    - Lazy (kNN): No training, memorizes all data
    - Eager (LR/DT): Builds model during training
       - We could not use LR/DT for Text Data (TF-IDF vectors with 1000+ dimensions)
       - DT makes axis-aligned splits but in 1000D data, no word splits the space meaningfully, as a result the tree grows deep and overfits
       - LR assumes linear relationships, but text data is non-linear
       - Distance based methods like cosine similarity or neural networks work better.
  * Interpretability differences
    - Decision trees provide clear rules
    - kNN decisions based on local neighbors
  * Handling of feature relationships
    - Linear regression assumes linear relationships
    - kNN can capture complex, non-linear patterns

### 1.5 Strengths and Limitations
- Advantages:
  * Simple implementation
    - No complex math required
    - Easy to explain conceptually
  * No training phase
    - Immediately ready for predictions
    - Can update with new data without retraining
- Limitations:
  * Computational cost at prediction time
    - Must compute distances to all training points
    - Becomes slow with large datasets
  * Curse of dimensionality (metric dependent)
    - Distance becomes meaningless in high dimensions (hundreds/thousands)
    - All points become equally distant

## 2. Clustering Algorithms

### 2.1 Fundamental Concepts
- Learning Context:
  * Unsupervised learning paradigm
    - No labels provided
    - Discovers inherent structure
  * Discovery of inherent groupings
    - Based solely on feature similarity
    - Quality depends on distance metric choice

### 2.2 Algorithm Comparison
- K-Means vs Hierarchical:
  * Partitioning vs hierarchical approach
    - K-Means: Flat structure, must specify k
    - Hierarchical: Creates dendrogram, multiple k levels
  * Complexity differences
    - K-Means: O(n), faster for large datasets
    - Hierarchical: O(n²), memory intensive

### 2.3 High-Dimensional Challenges
- Curse of dimensionality:
  * Distance concentration problems
    - All pairwise distances become similar
    - Hard to distinguish clusters
  * Dimensionality reduction solutions (PCA, t-SNE)
    - PCA: Linear projection
    - t-SNE: Non-linear, preserves local structure

### 2.4 Cluster Evaluation
- Quality Assessment:
  * Internal metrics (cohesion, separation)
    - Cohesion: Average distance within clusters
    - Separation: Distance between clusters
  * Visual inspection methods
    - 2D/3D plots after dimensionality reduction
    - Heatmaps of distance matrices

- Determining Optimal k:
  * Elbow Method: SSE analysis
    - Plot SSE vs k, look for "elbow" point
    - Tradeoff between error and complexity
  * Silhouette Score: cohesion vs separation
    - Ranges from -1 (bad) to 1 (good)
    - Measures how well points fit their cluster

### 2.5 Real-World Applications
- Business Strategy Example:
  * Retail customer segmentation
    - Group by purchasing behavior/demographics
    - Example clusters: bargain hunters, premium buyers
  * Targeted marketing strategies
    - Custom promotions for each segment
    - Cluster-specific product recommendations

### 2.6 K-Means Characteristics
- Advantages:
  * Scalability to large datasets
    - Linear time complexity
    - Can use mini-batch variants
  * Simple interpretation
    - Centroid represents cluster prototype
    - Easy to explain to non-technical stakeholders
- Limitations:
  * Spherical cluster assumption
    - Assumes clusters are round, equally sized
    - Struggles with elongated or irregular shapes
  * Sensitivity to initialization
    - Random starts may yield different results
    - K-means++ helps with better initialization
