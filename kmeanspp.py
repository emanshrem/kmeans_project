import sys
import numpy as np
import pandas as pd
import mykmeanspp # The C extension module you compiled

def convert_to_int(val):
    try:
        f = float(val)
        if f.is_integer():  # Checks if float is whole number, e.g. 3.0 but not 3.4
            return int(f)
        else:
            return None
    except:
        return None

def convert_to_float(val):
    try:
        f = float(val)
        return f if f >= 0 else None
    except:
        return None



def load_and_join(file1, file2, delimiter=',', has_header=False):
    """
    Loads and joins two input files on their first column (IDs) using inner join.
    Returns (ids, points) where ids is a list of IDs and points is np.array of floats.
    """

    header_option = 0 if has_header else None

    # Load files
    df1 = pd.read_csv(file1, delimiter=delimiter, header=header_option)
    df2 = pd.read_csv(file2, delimiter=delimiter, header=header_option)

    # Rename first column to 'key' for merging
    df1 = df1.rename(columns={df1.columns[0]: 'key'})
    df2 = df2.rename(columns={df2.columns[0]: 'key'})

    # Perform inner join on 'key'
    merged = pd.merge(df1, df2, on='key', how='inner')

    # Sort by 'key'
    merged = merged.sort_values(by='key').reset_index(drop=True)

    # Extract IDs as list
    ids = merged['key'].tolist()

    # Drop the ID column and convert remaining columns to numpy float array
    points = merged.drop(columns=['key']).to_numpy(dtype=float)
    if merged.empty:
        print("An Error Has Occurred!")
        sys.exit(1)

    return ids, points



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
            min(np.sqrt(np.sum((point - centroid) ** 2)) for centroid in centroids)
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
        return 1

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
            print("Invalid number of clusters!")

        elif max_iter is None:
            print("Invalid maximum iteration!")

        elif eps is None:
            print("Invalid epsilon!")
        return 1 

    if not (1 < k):
        print("Invalid number of clusters!")
        return 1
    if not (1 < max_iter < 1000):
        print("Invalid maximum iteration!")
        return 1
    if eps < 0:
        print("Invalid epsilon!")
        return 1

    # Load and join data
    ids, points = load_and_join(file1, file2) #ids are the first column(dims) in each point

    if k >= len(points):
        print("Invalid number of clusters!")
        return 1

    # Initialize centroids
    initial_centroids, chosen_ids = kmeans_pp_init(points, ids, k)

    try:
    # Run the C extension
        final_centroids = mykmeanspp.fit(
            points.tolist(),
            initial_centroids.tolist(),
            max_iter,
            eps,
            len(points),             # N
            len(points[0])           # dim
        )
    except Exception:
        print("An Error Has Occurred!")
        return 1  # or return 1 if inside a function


    # Output
    print(",".join(str(int(x)) for x in chosen_ids))
    for centroid in final_centroids:
        print(",".join([f"{coord:.4f}" for coord in centroid]))
    return 0  # or return 0 if inside a function

if __name__ == "__main__":
    sys.exit(main())