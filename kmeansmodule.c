/* KmeansTrial2 */
#define _GNU_SOURCE  /* Required before stdio.h to use getline, per HW1 instructions */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>  /* for strtok, allowed as per forum */
#include <math.h>

/* Declare getline prototype for ANSI C */
ssize_t getline(char **lineptr, size_t *n, FILE *stream);

/* --- Function declarations omitted for brevity --- */

/* Loads points from stdin dynamically, sets *num_points and *dimension */
double** load_points_from_stdin(int* num_points, int* dimension) {
    char* line = NULL;
    size_t len = 0;
    ssize_t read;
    int capacity;
    double** points;
    int dim_count;
    int i, idx;
    double** temp;
    char* token;

    capacity = 10;  /* Initial capacity, grows dynamically if needed */

    points = malloc(capacity * sizeof(double*));
    if (!points) {
        printf("An Error Has Occurred!");
        exit(1);
    }

    *num_points = 0;
    *dimension = 0;

    while ((read = getline(&line, &len, stdin)) != -1) {
        /* Remove trailing newline if present */
        if (read > 0 && line[read - 1] == '\n') {
            line[read - 1] = '\0';
        }

        /* Count commas to find number of dimensions = commas + 1 */
        dim_count = 1;
        for (i = 0; line[i]; i++) {
            if (line[i] == ',') dim_count++;
        }

        /* On first line, set dimension and allocate accordingly */
        if (*num_points == 0) {
            *dimension = dim_count;
        } else {
            /* Dimension consistency check (optional per HW notes) */
            if (dim_count != *dimension) {
                printf("An Error Has Occurred!");
                free(line);
                for (i = 0; i < *num_points; i++) free(points[i]);
                free(points);
                exit(1);
            }
        }

        /* Resize points array if capacity reached */
        if (*num_points >= capacity) {
            capacity *= 2;
            temp = realloc(points, capacity * sizeof(double*));
            if (!temp) {
                printf("An Error Has Occurred!");
                free(line);
                for (i = 0; i < *num_points; i++) free(points[i]);
                free(points);
                exit(1);
            }
            points = temp;
        }

        /* Allocate space for current point's coordinates */
        points[*num_points] = malloc((*dimension) * sizeof(double));
        if (!points[*num_points]) {
            printf("An Error Has Occurred!");
            free(line);
            for (i = 0; i < *num_points; i++) free(points[i]);
            free(points);
            exit(1);
        }

        /* Parse coordinates using strtok and atof */
        token = strtok(line, ",");
        idx = 0;
        while (token != NULL) {
            points[*num_points][idx++] = atof(token);
            token = strtok(NULL, ",");
        }

        (*num_points)++;  /* Increment number of points loaded */
    }

    free(line);  /* Free buffer allocated by getline */
    return points;  /* Caller is responsible to free points array and all its rows */
}

/* Initialize centroids as a deep copy of the first K points (caller frees) */
double** initialize_centroids(double** points, int K, int d, int N) {
    double **copy;
    int i, j;

    copy = malloc(K * sizeof(double*));
    if (!copy) {
        printf("An Error Has Occurred!");
        /* Free points before exiting */
        for (i = 0; i < N; i++) {
            free(points[i]);
        }
        free(points);
        exit(1);
    }

    for (i = 0; i < K; i++) {
        copy[i] = malloc(d * sizeof(double));
        if (!copy[i]) {
            printf("An Error Has Occurred!");
            /* Free previously allocated before exiting */
            for (j = 0; j < i; j++) free(copy[j]);
            free(copy);
                    /* Free points before exiting */
            for (i = 0; i < N; i++) {
                free(points[i]);
            }
            free(points);
            exit(1);
        }
        for (j = 0; j < d; j++) {
            copy[i][j] = points[i][j];
        }
    }

    return copy;
}

/* Euclidean distance between two points in d-dimensional space */
double dist(double* p, double* q, int d) {
    double sum = 0.0;
    int i;
    double diff;

    for (i = 0; i < d; i++) {
        diff = p[i] - q[i];
        sum += diff * diff;  /* faster than pow(diff, 2) */
    }
    return sqrt(sum);
}

/* Print K centroids with 4 decimals, comma separated (no spaces) */
void print_centroids(double** centroids, int K, int d) {
    int i, j;

    for (i = 0; i < K; i++) {
        for (j = 0; j < d; j++) {
            printf("%.4f", centroids[i][j]);
            if (j < d - 1) printf(",");
        }
        printf("\n");
    }
}

/* Assign each point to the closest centroid, returns allocated int array (caller frees) */
int* assign_clusters(double** points, double** centroids, int K, int d, int N) {
    int* cluster_indices;
    int i, j;
    double min_dist, distance;
    int min_index;

    cluster_indices = malloc(N * sizeof(int));
    if (!cluster_indices) {
        printf("An Error Has Occurred!");
        /* Free points before exiting */
        for (i = 0; i < N; i++) {
            free(points[i]);
        }
        free(points);       
        for (i = 0; i < K; i++) {
            free(centroids[i]);
        }
        free(centroids); 
        exit(1);
    }

    for (i = 0; i < N; i++) {
        min_dist = INFINITY;
        min_index = -1;

        for (j = 0; j < K; j++) {
            distance = dist(points[i], centroids[j], d);
            if (distance < min_dist) {
                min_dist = distance;
                min_index = j;
            }
        }

        cluster_indices[i] = min_index;  /* Assign closest centroid index */
    }

    return cluster_indices;
}

/* Calculate new centroids by averaging assigned points, returns allocated double** (caller frees) */
double** update_centroids(double** points, double** centroids, int* cluster_indices, int K, int N, int d) {
    double** new_centroids;
    int* counts;
    int i, j, cluster;

    new_centroids = malloc(K * sizeof(double*));
    if (!new_centroids) {
        printf("An Error Has Occurred!");
        /* Free points before exiting */
        for (i = 0; i < N; i++) {
            free(points[i]);
        }
        free(points);       
        for (i = 0; i < K; i++) {
            free(centroids[i]);
        }
        free(centroids);
        free(cluster_indices);
        exit(1);
    }

    for (i = 0; i < K; i++) {
        new_centroids[i] = calloc(d, sizeof(double));  /* zero-initialized */
        if (!new_centroids[i]) {
            printf("An Error Has Occurred!");
            for (j = 0; j < i; j++) free(new_centroids[j]);
            free(new_centroids);
                /* Free points before exiting */
            for (i = 0; i < N; i++) {
                free(points[i]);
            }
            free(points);       
            for (i = 0; i < K; i++) {
                free(centroids[i]);
            }
            free(centroids);
            free(cluster_indices);
            exit(1);
        }
    }

    counts = calloc(K, sizeof(int));  /* track number of points in each cluster */
    if (!counts) {
        printf("An Error Has Occurred!");
        for (i = 0; i < K; i++) free(new_centroids[i]);
        free(new_centroids);
        /* Free points before exiting */
        for (i = 0; i < N; i++) {
            free(points[i]);
        }
        free(points);       
        for (i = 0; i < K; i++) {
            free(centroids[i]);
        }
        free(centroids);
        free(cluster_indices);
        exit(1);
    }

    /* Sum coordinates for points assigned to each cluster */
    for (i = 0; i < N; i++) {
        cluster = cluster_indices[i];
        counts[cluster]++;
        for (j = 0; j < d; j++) {
            new_centroids[cluster][j] += points[i][j];
        }
    }

    /* Divide by counts to get average for each centroid coordinate */
    for (i = 0; i < K; i++) {
        if (counts[i] == 0) continue;  /* avoid division by zero if cluster empty */
        for (j = 0; j < d; j++) {
            new_centroids[i][j] /= counts[i];
        }
    }

    free(counts);  /* Free temporary counts array */
    return new_centroids;
}

/* Check if all centroids have moved less than epsilon distance (converged) */
int has_converged(double** old_centroids, double** new_centroids, int K, int d, double epsilon) {
    int i;
    double distance;

    for (i = 0; i < K; i++) {
        distance = dist(old_centroids[i], new_centroids[i], d);
        if (distance > epsilon) {
            return 0;  /* Not yet converged */
        }
    }
    return 1;  /* Converged */
}

/* Helper to free 2D array of points or centroids */
void free_points(double** points, int n) {
    int i;
    for (i = 0; i < n; i++) {
        free(points[i]);
    }
    free(points);
}


int is_valid_integer_string(const char *s) {
    char *endptr;
    double val;
    

    val = strtod(s, &endptr);
    if (*endptr != '\0') {
        return 0;
    }
    return floor(val) == val;
}

double** kmeans(double** points, int N, int d, int K, int iter, double epsilon) {
    double** centroids;
    int* cluster_indices;
    double** new_centroids;
    int i, j;

    if (!(1 < K && K < N)) {
        return NULL;  // invalid number of clusters
    }

    centroids = initialize_centroids(points, K, d, N);
    if (!centroids) return NULL;  // error already handled in helper

    for (i = 0; i < iter; i++) {
        cluster_indices = assign_clusters(points, centroids, K, d, N);
        if (!cluster_indices) {
            free_points(centroids, K);
            return NULL;
        }

        new_centroids = update_centroids(points, centroids, cluster_indices, K, N, d);
        free(cluster_indices);

        if (!new_centroids) {
            free_points(centroids, K);
            return NULL;
        }

        if (has_converged(centroids, new_centroids, K, d, epsilon)) {
            free_points(centroids, K);
            return new_centroids;
        }

        free_points(centroids, K);
        centroids = new_centroids;
    }

    return centroids;
}