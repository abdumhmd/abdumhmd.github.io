# FastRP Algorithm Notes

## How FastRP Works - Step by Step

FastRP (Fast Random Projection) creates vector embeddings for graph nodes by using random projections and iterative neighborhood averaging. Here's how it works in simple terms:

### Step 1: Graph Setup and Preprocessing
- Select which vertex types and edge types to include in the computation
- Filter vertices based on connected components (if batching is enabled)
- Calculate degree-based normalization factors for each vertex

### Step 2: Random Initialization
- Each vertex gets a **sparse random vector** as its starting embedding
- Most values in this vector are zero (controlled by `sampling_constant`)
- The few non-zero values are either `√sampling_constant` or `-√sampling_constant`
- This sparsity makes the algorithm computationally efficient

### Step 3: Iterative Neighborhood Aggregation
For each iteration:
1. **Message Passing**: Each vertex sends its current embedding to all its neighbors
2. **Aggregation**: Each vertex sums up all embeddings it receives from neighbors
3. **Normalization**: Divide by degree and L2-normalize to prevent explosion
4. **Weighting**: Apply iteration-specific weights to control influence

### Step 4: Final Embedding Construction
- Combine embeddings from all iterations using weighted averaging
- Store results in vertex attributes or export to files
- The final embedding captures multi-hop neighborhood information

## Key Code Snippets

### Degree-Based Normalization
```gsql
// Calculate normalization factor based on vertex degree
verts =
  SELECT s FROM verts:s -(e_type_set:e)- v_type_set:t
  WHERE t.@include == TRUE
  ACCUM @@m += 1
  POST-ACCUM s.@L = pow(s.outdegree(e_type_set) / @@m, beta);
```

### Random Sparse Initialization
```gsql
// PRNG parameters for sparse random vectors
v1 = sqrt(sampling_constant);
v2 = -v1;
v3 = 0.0;
p1 = 0.5 / sampling_constant;      // Probability for +√k
p2 = p1;                           // Probability for -√k  
p3 = 1 - 1.0 / sampling_constant;  // Probability for 0

// Initialize each dimension randomly
if (mr <= p1) THEN
  t.@embedding_arr += (i -> v1 * s.@L * weight)
ELSE IF (mr <= p1 + p2) THEN
  t.@embedding_arr += (i -> v2 * s.@L * weight)
ELSE
  t.@embedding_arr += (i -> v3 * s.@L * weight)
END
```

### Iterative Message Passing
```gsql
FOREACH depth IN RANGE[0, @@weights.size()-1] DO
  // Send embeddings to neighbors
  verts =
    SELECT s FROM verts:s -(e_type_set)- v_type_set:t
    WHERE t.@include == TRUE
    ACCUM
      t.@embedding_arr += s.@embedding_arr
    POST-ACCUM
      // Normalize by degree and L2 norm
      FLOAT out = max([1.0, t.outdegree(e_type_set)]),
      // ... L2 normalization code ...
      t.@final_embedding_arr += (i -> t.@embedding_arr.get(i) / out / square_sum * @@weights.get(depth))
END;
```

### Heterogeneous Graph Support
```gsql
// Different edge types can affect different embedding regions
FOREACH string_tuple IN embedding_dim_map DO
  temp_dim_key = substr(string_tuple, 0, idx_1);
  temp_length = str_to_int(substr(string_tuple, idx_1+1, idx_2-1-idx_1));
  temp_weight = tg_str_to_float(substr(string_tuple, idx_2+1));
  @@embedding_dim_map += (temp_dim_key -> Dim_Tuple(idx_3, idx_3+temp_length, temp_weight));
END;
```

## Important Hyperparameters

### Core Parameters

| Parameter | Type | Description | Typical Values |
|-----------|------|-------------|----------------|
| `embedding_dimension` | INT | Size of output embedding vectors | 64, 128, 256 |
| `iteration_weights` | STRING | Comma-separated weights for each iteration | "1.0,0.5,0.25" |
| `beta` | FLOAT | Degree normalization factor | -1.0 to 0.0 |
| `sampling_constant` | INT | Controls sparsity of random initialization | 1-3 |

### Advanced Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `default_index` | INT | Starting position for default edge types in embedding | 0 |
| `default_length` | INT | Length of embedding region for default edges | Required |
| `default_weight` | FLOAT | Influence weight for default edge types | 1.0 |
| `embedding_dim_map` | SET<STRING> | Maps edge types to embedding regions | Empty set |
| `random_seed` | INT | Seed for reproducible results | 42 |

### Output and Utility Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `result_attribute` | STRING | Vertex attribute to store embeddings |
| `component_attribute` | STRING | Attribute for connected component batching |
| `batch_number` | INT | Which component batch to process |
| `filepath` | STRING | File path for exporting results |
| `print_results` | BOOL | Whether to print embeddings to console |
| `choose_k` | INT | Number of sample vertices to display |

## Parameter Guidelines

### Beta (β) Values
- **β = -1**: Heavily penalize high-degree vertices
- **β = -0.5**: Moderate penalization (common choice)
- **β = 0**: No degree-based penalization
- **β > 0**: Favor high-degree vertices (less common)

### Sampling Constant
- **1**: Dense random vectors (all values non-zero)
- **2**: Medium sparsity (~50% zeros)
- **3**: High sparsity (~67% zeros) - **most common**

### Iteration Weights
- **Equal weights**: "1.0,1.0,1.0" - all iterations contribute equally
- **Decreasing weights**: "1.0,0.5,0.25" - emphasize early iterations
- **Single iteration**: "1.0" - fastest but less expressive

### Embedding Dimension Map Format
For heterogeneous graphs, specify as: `"<edge_type>,<length>,<weight>,<start_index>"`

Example:
```
["friendship,20,1.0,0", "collaboration,10,0.8,20", "mentorship,5,1.2,30"]
```

This creates specialized embedding regions for different relationship types.

## Comparison between Domain Features and FastRP graph embeddings
Here are some reasons why domain features might fall short.
- Scalability: as graphs grow, computing complex features (especially those involving multi-hop relationships or aggregations over large neighborhoods) becomes computationally expensive and difficult to maintain.
- Coverage: Manual features often focus on direct and easily conceptualized relationships, potentially missing subtle, high-order patterns or combinations of features that are not obvious to human analysts.
- Feature explosion: As we try to capture more nuanced behavior, the number of engineered features can grow rapidly, leading to redundancy and overfitting risks.
But what does a graph embedding bring to the table?
- They automatically learn dense vector representations for each node that capture both local and global graph structure and even indirect relationships. This allows models to detect fraud rings or subtle collusion patterns that manual features might miss.
- Instead of hand-crafting dozens of features, embeddings are learned directly from the data, saving significant time and effort while capturing a broader spectrum of graph information.
- They are a way to capture a lot of information about the graph in an unsupervised manner, without the need for time-consuming feature engineering that some of the other approaches require.

Check this [link](https://www.tigergraph.com/blog/using-graph-machine-learning-to-improve-fraud-detection-rates/)
