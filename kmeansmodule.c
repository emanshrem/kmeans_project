/* KmeansTrial2 */
#define _GNU_SOURCE  /* Required before stdio.h to use getline, per HW1 instructions */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>  /* for strtok, allowed as per forum */
#include <math.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>


/* KMeans algorithm declaration (from your C implementation) */
double** kmeans(double** points, int N, int d, int K, int iter, double epsilon, double** centroids);


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



double** kmeans(double** points, int N, int d, int K, int iter, double epsilon, double** centroids) {
    int* cluster_indices;
    double** new_centroids;
    int i, j;

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

/* Helper to allocate 2D array */
double** allocate_2d_double(int rows, int cols) {
    double **array = malloc(rows * sizeof(double*));
    if (!array) return NULL;
    for (int i = 0; i < rows; i++) {
        array[i] = malloc(cols * sizeof(double));
        if (!array[i]) {
            for (int j = 0; j < i; j++) free(array[j]);
            free(array);
            return NULL;
        }
    }
    return array;
}


/* Python wrapper for kmeans */
static PyObject* fit(PyObject *self, PyObject *args) {
    PyObject *points_obj, *initial_obj;
    int max_iter, N, d;
    double epsilon;

    if (!PyArg_ParseTuple(args, "OOidii", &points_obj, &initial_obj, &max_iter, &epsilon, &N, &d)) {
        return NULL;
    }

    int K = PyList_Size(initial_obj);

    double **points = allocate_2d_double(N, d);
    double **initial = allocate_2d_double(K, d);
    if (!points || !initial) {
        free_points(points, N);
        free_points(initial, K);
        PyErr_SetString(PyExc_MemoryError, "An Error Has Occurred!");
        return NULL;
    }

        // Convert Python points to C array
    for (int i = 0; i < N; i++) {
        PyObject* point = PyList_GetItem(points_obj, i);
        if (!point) {
            free_points(points, N);
            free_points(initial_obj, K);
            PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred!");
            return NULL;
        }
        for (int j = 0; j < d; j++) {
            PyObject* coord = PyList_GetItem(point, j);
            if (!coord) {
                free_points(points, N);
                free_points(initial_obj, K);
                PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred!");
                return NULL;
            }
            points[i][j] = PyFloat_AsDouble(coord);
        }
    }
    // Convert Python initial centroids to C array
    for (int i = 0; i < K; i++) {
        PyObject* centroid = PyList_GetItem(initial_obj, i);
        if (!centroid) {
            free_points(points, N);
            free_points(initial, K);
            PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred!");
            return NULL;
        }
        for (int j = 0; j < d; j++) {
            PyObject* coord = PyList_GetItem(centroid, j);
            if (!coord) {
                free_points(points, N);
                free_points(initial, K);
                PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred!");
                return NULL;
            }
            initial[i][j] = PyFloat_AsDouble(coord);
        }
    }


    double **final_centroids = kmeans(points, N, d, K, max_iter, epsilon, initial);

    if (!final_centroids) {
        free_points(points, N);
        free_points(initial, K);
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred!");
        return NULL;
    }

    PyObject *result = PyList_New(K); //Creates a new Python list with K elements, initialized to NULL.
    for (int i = 0; i < K; i++) { //For each centroid i, create a new Python list of length d.
        PyObject *centroid = PyList_New(d);
        for (int j = 0; j < d; j++) {
            PyList_SetItem(centroid, j, PyFloat_FromDouble(final_centroids[i][j])); //For each coordinate j in centroid i, convert the C double to a Python float.
        }
        PyList_SetItem(result, i, centroid); //Once the full centroid list is built, insert it into the outer list result at index i.
    }

    free_points(points, N);
    free_points(final_centroids, K);

    return result;
}

static PyMethodDef KMeansMethods[] = {
    {"fit",
      (PyCFunction)fit,
      METH_VARARGS,
      "Run the KMeans algorithm."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef kmeansmodule = {
    PyModuleDef_HEAD_INIT,
    "mykmeanspp",
    NULL,
    -1,
    KMeansMethods
};

PyMODINIT_FUNC PyInit_mykmeanspp(void) {
    return PyModule_Create(&kmeansmodule);
}
