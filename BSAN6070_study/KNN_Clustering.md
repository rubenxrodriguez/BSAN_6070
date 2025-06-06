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
        - Nodes of the tree is a series of nesting hyper-sphere
        -  Text Data (TF-IDF vectors with 1000+ dimensions)
        - Image Embeddings, Genomics data

- Choosing k Value:
  * Bias-variance tradeoff (small k vs large k)
    - Small k: High variance, captures noise (overfitting)
    - Large k: High bias, smoother decision boundaries
    - Bank Fraud
        * k = 1 --> Overfits to noise and has too many false positives
        * k = 100 --> Over-smooths decision boundaries and has too many false negatives (misses subtle fraud patterns)
  * Computational cost implications
    - Larger k requires more distance calculations
    - Impacts prediction time more than training time
  * Odd vs even k in binary classification
    - Prevents tie votes in binary classification
    - Especially important with balanced classes

### 1.2 Practical Considerations
- Data Prep Needs
  * Standardize the data before training the model
  * Scalers
     * StandardScaler() - Mean = 0, sd = 1 per column
     * MinMaxScaler() - Scales to [0,1] per column
     * Normalizer(norm = 'l2') - Each row has ||X|| = 1
- Brute Force
  * Note: For small sample sizes brute force search can be more efficient than a tree-based query
  * Brute force is largely unaffected by the value of k
- Sparsity of Data Structure
  * A dataset is sparse when most feature values are zeros (TF-IDF vectors, one-hot encoded categories)
  * Sparsity doesn't affect Brute Force
     - No assumptions about structure. Easily parallelized.
  * Ball Tree performs better than KD Tree with sparser data structure
     - Trees partition all dimensions, including empty ones (wasted computation)
     - In high-D sparse spaces, all points appear equidistant (curse of dimensionality)
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
    - Use this for
         * non-uniform neighborhoods (Medical diagnosis based on patient vitals) - closer matches (age/weight) matter more than distant ones
         * Avoiding outlier influence (Stock price prediction) - prevent single volatile stock from skewing forecast
      
  * When to use KNN Regression for Tabular Data
     - Pros : Simple, no assumptions about data, handles non-linear relationships
     - Cons : Slow for large datasets, requires careful encoding of categoricals, sensitive to irrelevant features.

### 1.3 Advanced Applications
- Recommendation Systems:
  * User/item similarity measurement
    - Users as points in feature space
    - Items purchased/rated as features
  * Feature Extraction in Computer Vision
  * Inputing missing values and minority ovversampling of imbalanced data in Data Analytics

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
  * We can't or don't "PREDICT" using Clustering !!
  * Discovery of inherent groupings
    - Based solely on feature similarity
    - Quality depends on distance metric choice
  * Major purposes of clustering include:
    - Description of dataset
    - Facilitating improvement in performance of other ML techniques when there are many competing patterns in the data
       * Real estate pricing: Cluster houses into "luxury" "suburban" groups, then train distinct regression models for each cluster.
       * Dimensionality reduction: text data -> TF-IDF vectors -> use cluster IDs as simplified features
  * Variable Rules
     - Variables can be interval, binary, categorical (ordinal, nominal)
     - Nominal categories have no inherent order or numerical meaning. Arbitrary numerical encoding distorts true relationships. One hot introduces sparsity.
     - variables should be standardized/normalized

### 2.2 Algorithm Comparison
- There are multiple clustering algorithms
     * Partitional, Hierarchical, Probabilistic, Grid-based, Sequential
        * It is NEVER clear which algorithm and which parameter settings is the most appropriate --> Experimentation
- Hierarchical
    - Does not require the number of clusters _k_ as an input. 
    - **Approach**: Builds nested clusters (tree-like structure)
        * Single Link: Merges clusters based on _minimum_ distance between points (sensitive to outliers)
        * Complete Link: Merges based on _maximum_ distance (creates compact clusters)
    - Agglomerative
        * Bottom-Up approach that merges pairs of clusters
            * Start with each object in its own cluster
            * At each step, merge the closest pair of clusters
      
        * Centroid Method
            * Merge 2 clusters that have the most similar Centroids
        * Average Linkage
            * Assign each object to its own cluster -> then merge the two most similar objects.
            * Calculate the average distance of the two points merged together, then find the point that is closest to this average distance and merge it to the cluster. Repeat until you have g clusters.
            * Merge 2 clusters such that the average of all distances between pairs of cases is minimized
        * Average Linkage assigns each to its own cluster. Complete linkage is a bit like Density-Based (DBSCAN) in that you leave all objects separate. You make the first cluster by the two closest objects, then you add new objects to this cluster based on their distance to the first cluster. Complete linkage is better than single linkage (chain reaction) where clustering happens when single elements are close to each other. Complete linkage tends to find compact clusters.
        * Ward Method
            * Merge the 2 clusters whose merger gives the minimum increase in the variation

  - Use Cases:
     * Genomics (Group genes with similar expression patterns) --> Average linkage
     * Retail (Segment stores by sales patterns) --> Ward's
     * Document Clustering (Group articles by topic) --> Complete linkage [avoke chain reaction]
  - Divisive
    * Top-Down approach involving binary division of clusters
           * Start with all objects in an all-inclusive cluster
           * At each step, split a cluster until each object is in its own cluster
           * Use Cases:
              - smaller data size (computationally expensive), top-down hierarchy (taxonomies)
              - Has a low outlier sensitivity because it isolates anomolies early
              - don't use if natural groupings are flat (e.g., customer segments)
              - can use for organizing a product catalog for an e-commerce giant (start broad) (centroid linkage), use for voting pattern analysis (wards)
  
- Partitional Clustering
    - **Approach**: Divides data into non-overlapping subsets
        * Square Error Minimization - minimize within-cluster variance
            * k-means : Assigns points to nearest centroid (deterministic)
            * k-means : Assign each object to the cluster whose CENTROID is closest to the object.
                * Let the new CENTROID be the mean of the objects in the cluster
        * Mixture Resolving:
            * Models data as a mix of probability distributions ~ P(cluster A) = 0.7, P(Cluster B) = 0.3
        * Graph Theoretic: Uses graph structures (spectral clustering)
        * Mode Seeking: Finds dense regions (mean-shift clustering)
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
  * Internal metrics (compactness, separation)
    - Cohesion: Average distance within clusters
    - Separation: Distance between clusters **should** be widely spaced
       * distance between **closest** members
       * distance between **most distant** members
       * distance between **centers** of clusters
  * Good clustering depends on similarity measure used by the method and its implementation
     - quality of clustering is measured by its ability to discover some or all the hidden patterns
- Determining Optimal k:
  * Elbow Method: SSE analysis
    - Plot SSE (sum of squared errors) vs k, look for "elbow" point. The point at which adding more clusters does not improve the SSE.
    - Good for quick, visual estimation (but subjective)
    - Use for compact, globular clusters
    - Avoid for no clear elbow (plateau). 
  * Silhouette Score: cohesion vs separation
    - More precise but slower. Use when cluster shapes are complext, noisy data.
    - Ranges from -1 (bad) to 1 (good)
    - Measures how well points fit their cluster

### 2.5 Real-World Applications
- Business Strategy Example:
  * Retail customer segmentation
    - Group by purchasing behavior/demographics
    - Example clusters: bargain hunters, premium buyers
  * Discovery : Find songs similar to songs you like
  * Hierarchy: Find good taxonomy of species from genes
  * Graph Partitioning: Find groups in social networks

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
  * Outliers: very sensitive to the presence of outliers.
  * Sensitivity to initialization
    - Random starts may yield different results
    - K-means++ helps with better initialization
