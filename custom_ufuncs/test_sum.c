#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"

/*
 * multi_arg_logit.c
 * This is the C code for creating your own
 * NumPy ufunc for a multiple argument, multiple
 * return value ufunc. The places where the
 * ufunc computation is carried out are marked
 * with comments.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 *
 */


static PyMethodDef TestSumMethods[] = {
        {NULL, NULL, 0, NULL}
};

/* The loop definition must precede the PyMODINIT_FUNC. */

static void double_testsum(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1];
    char *out1 = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1];
    npy_intp out1_step = steps[2];

    double tmp;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        if (i == 0) {
            *((double *)out1) = *(double *)in1;
        }
        else {
            tmp = *(double *)(out1 - steps[2]) * *(double *)in2;
            tmp += *(double *)in1;
            *((double *)out1) = tmp;
 
        }
        /*END main ufunc computation*/

        in1 += in1_step;
        in2 += in2_step;
        out1 += out1_step;
    }
}


/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&double_testsum};

/* These are the input and return dtypes of logit.*/

static char types[4] = {NPY_DOUBLE, NPY_DOUBLE,
                        NPY_DOUBLE, NPY_DOUBLE};


static void *data[1] = {NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "npufunc",
    NULL,
    -1,
    TestSumMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_npufunc(void)
{
    PyObject *m, *testsum, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    testsum = PyUFunc_FromFuncAndData(funcs, data, types, 1, 2, 1,
                                    PyUFunc_None, "testsum",
                                    "testsum_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "testsum", testsum);
    Py_DECREF(testsum);

    return m;
}
#else
PyMODINIT_FUNC initnpufunc(void)
{
    PyObject *m, *testsum, *d;


    m = Py_InitModule("npufunc", TestSumMethods);
    if (m == NULL) {
        return;
    }

    import_array();
    import_umath();

    testsum = PyUFunc_FromFuncAndData(funcs, data, types, 1, 2, 2,
                                    PyUFunc_None, "testsum",
                                    "testsum_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "testsum", testsum);
    Py_DECREF(testsum);
}
#endif
