import sys
import numpy as np
import pandas as pd
import mykmeanspp # The C extension module you compiled

def convert_to_int(val):
    try:
        return int(float(val))  # Handles "3", "3.0", "03"
    except:
        return None

def convert_to_float(val):
    try:
        return float(val)
    except:
        return None

def is_valid_file(filename):
    return filename.endswith(".txt") or filename.endswith(".csv")

def load_and_join(file1, file2, delimiter=',', has_header=False): #MakeSure??
    """Loads and joins two input files on their first column (IDs) using inner join."""
    header_option = 0 if has_header else None

    # Load files
    df1 = pd.read_csv(file1, delimiter=delimiter, header=header_option)
    df2 = pd.read_csv(file2, delimiter=delimiter, header=header_option)

    # Rename first column to 'key' for merging
    df1 = df1.rename(columns={df1.columns[0]: 'key'})
    df2 = df2.rename(columns={df2.columns[0]: 'key'})

    # Perform inner join on 'key'
    merged = pd.merge(df1, df2, on='key')

    # Sort by 'key'
    merged = merged.sort_values(by='key').reset_index(drop=True)

    return merged

def kmeans_pp_init(points, ids, k):
    """Initialize centroids using K-means++ algorithm."""
    n = points.shape[0]
    centroids = []
    chosen_indices = []

    np.random.seed(1234)
    first_index = np.random.choice(n)
    centroids.append(points[first_index])
    chosen_indices.append(ids[first_index])

    for _ in range(1, k):
        dists = np.array([
            min(np.sum((point - centroid) ** 2) for centroid in centroids)
            for point in points
        ])
        probs = dists / np.sum(dists) #we won't choose the same centroid more than once! cuz dist will be 0 and prob will be 0
        next_idx = np.random.choice(n, p=probs)
        centroids.append(points[next_idx])
        chosen_indices.append(ids[next_idx])

    return np.array(centroids), chosen_indices

def main():

    #reading the cmd line and checking validity of inputs
    args = sys.argv[1:]

    if not (4 <= len(args) <= 5):
        print("An Error Has Occurred")
        sys.exit(1)

    k = convert_to_int(args[0])
    if len(args) == 4:
        max_iter = 300
        eps = convert_to_float(args[1])
        file1 = args[2]
        file2 = args[3]
    else:
        max_iter = convert_to_int(args[1])
        eps = convert_to_float(args[2])
        file1 = args[3]
        file2 = args[4]
        
    if None in [k, max_iter, eps]:
        if k is None:
            print("invalid number of clusters!")

        elif max_iter is None:
            print("invalid maximum iteration!")

        elif eps is None:
            print("invalid epsilon!")
        sys.exit(1)

    if not (1 < k):
        print("invalid number of clusters!")
        sys.exit(1)
    if not (1 < max_iter < 1000):
        print("invalid maximum iteration!")
        sys.exit(1)
    if eps < 0:
        print("invalid epsilon!")
        sys.exit(1)
    if not (is_valid_file(file1) and is_valid_file(file2)):
        print("An Error Has Occurred")
        sys.exit(1)

    # Load and join data
    ids, points = load_and_join(file1, file2) #ids are the first column(dims) in each point

    if k >= len(points):
        print("invalid number of clusters!")
        sys.exit(1)

    # Initialize centroids
    initial_centroids, chosen_ids = kmeans_pp_init(points, ids, k)

    # Run the C extension
    final_centroids = mykmeanspp.fit(
        points.tolist(),
        initial_centroids.tolist(),
        max_iter,
        eps,
        len(points),             # N
        len(points[0])           # dim
    )

    # Output
    print(",".join(map(str, chosen_ids)))
    for centroid in final_centroids:
        print(",".join([f"{coord:.4f}" for coord in centroid]))

if __name__ == "__main__":
    main()