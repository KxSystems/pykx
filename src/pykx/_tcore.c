#define KXVER 3
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <pthread.h>
#include "k.h"


struct QFuture {
    bool done;
    K res;
};

bool is_done(struct QFuture* fut) {
    if (fut->done)
        return true;
    return false;
}

struct QCall {
    struct QFuture* fut;
    int handle;
    bool is_dot;
    const char* query;
    int argc;
    K arg0;
    K arg1;
    K arg2;
    K arg3;
    K arg4;
    K arg5;
    K arg6;
    K arg7;
};

struct QCallNode {
    struct QCall* call;
    struct QCallNode* next;
};

static struct QCallNode* calls_head;
static struct QCallNode* calls_tail;
static void* _q_handle;
static pthread_t q_thread;
static int qinit_rc;
static pthread_mutex_t head_mutex;
static pthread_mutex_t cond_mutex;
static pthread_cond_t cond;
static pthread_mutex_t init_mutex;
static pthread_cond_t init;
static bool kill_thread;


int (*_qinit)(int, char**, char*, char*, char*);

static K (*__ee)(K x);
K _ee(K x) {
    return __ee(x);
}

static K (*__b9)(int x, K k);
K _b9(int x, K k){
    return __b9(x, k);
}

static K (*__d9)(K x);
K _d9(K x) {
    return __d9(x);
}

static int (*__dj)(int date);
int _dj(int date) {
    return __dj(date);
}

static K (*__dl)(void* f, long long n);
K _dl(void* f, long long n) {
    return  __dl(f, n);
}

static K (*__dot)(K x, K y);
K _dot_internal(K x, K y) {
    return __ee(__dot(x, y));
}
K _dot(K x, K y) {
    struct QFuture* fut = malloc(sizeof(struct QFuture));
    fut->done = false;
    fut->res = (K)0;
    struct QCall* call = malloc(sizeof(struct QCall));
    call->fut = fut;
    call->handle = 0;
    call->is_dot = true;
    call->query = NULL;
    call->argc = 2;
    call->arg0 = x;
    call->arg1 = y;
    call->arg2 = NULL;
    call->arg3 = NULL;
    call->arg4 = NULL;
    call->arg5 = NULL;
    call->arg6 = NULL;
    call->arg7 = NULL;
    struct QCallNode* call_node = malloc(sizeof(struct QCallNode));
    call_node->next = NULL;
    call_node->call = call;
    pthread_mutex_lock(&head_mutex);
    if (calls_head == NULL) {
        calls_head = call_node;
        calls_tail = call_node;
    } else {
        calls_tail->next = call_node;
        calls_tail = call_node;
    }
    pthread_mutex_unlock(&head_mutex);
    while (1 == 1) {
        pthread_mutex_lock(&cond_mutex);
        pthread_cond_signal(&cond);
        pthread_mutex_unlock(&cond_mutex);
        if (is_done(fut)) {
            pthread_mutex_lock(&head_mutex);
            free(call_node);
            free(call);
            K res = fut->res;
            free(fut);
            pthread_mutex_unlock(&head_mutex);
            return res;
        }
    }
    return (K)0;
}

static K (*__ja)(K* x, void* y);
K _ja(K* x, void* y) {
    return __ja(x, y);
}

static K (*__jk)(K* x, K y);
K _jk(K* x, K y) {
    return __jk(x, y);
}

static K (*__js)(K* x, char* s);
K _js(K* x, char* s) {
    return __js(x, s);
}

static K (*__jv)(K* x, K y);
K _jv(K* x, K y) {
    return __jv(x, y);
}

static K (*__k)(int handle, const char* s, ...);
K _k_internal(int handle, const char* s, int argc, K arg0, K arg1, K arg2, K arg3, K arg4, K arg5, K arg6, K arg7) {
    switch (argc) {
        case 0: return __ee(__k(handle, s, NULL));
        case 1: return __ee(__k(handle, s, arg0, NULL));
        case 2: return __ee(__k(handle, s, arg0, arg1, NULL));
        case 3: return __ee(__k(handle, s, arg0, arg1, arg2, NULL));
        case 4: return __ee(__k(handle, s, arg0, arg1, arg2, arg3, NULL));
        case 5: return __ee(__k(handle, s, arg0, arg1, arg2, arg3, arg4, NULL));
        case 6: return __ee(__k(handle, s, arg0, arg1, arg2, arg3, arg4, arg5, NULL));
        case 7: return __ee(__k(handle, s, arg0, arg1, arg2, arg3, arg4, arg5, arg6, NULL));
        case 8: return __ee(__k(handle, s, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, NULL));
    }
    return (K)0;
}
K _k(int handle, const char* s, ...) {
    va_list argp;
    va_start(argp, 8);
    K qargs[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    int qargc = 0;
    while (true) {
        qargs[qargc] = va_arg(argp, K);
        if (!qargs[qargc])
            break;
        qargc++;
    }
    va_end(argp);

    struct QFuture* fut = malloc(sizeof(struct QFuture));
    fut->done = false;
    fut->res = (K)0;
    struct QCall* call = malloc(sizeof(struct QCall));
    call->fut = fut;
    call->handle = handle;
    call->is_dot = false;
    call->query = s;
    call->argc = qargc;
    call->arg0 = qargs[0];
    call->arg1 = qargs[1];
    call->arg2 = qargs[2];
    call->arg3 = qargs[3];
    call->arg4 = qargs[4];
    call->arg5 = qargs[5];
    call->arg6 = qargs[6];
    call->arg7 = qargs[7];
    struct QCallNode* call_node = malloc(sizeof(struct QCallNode));
    call_node->next = NULL;
    call_node->call = call;

    pthread_mutex_lock(&head_mutex);
    if (calls_head == NULL) {
        calls_head = call_node;
        calls_tail = call_node;
    } else {
        calls_tail->next = call_node;
        calls_tail = call_node;
    }
    pthread_mutex_unlock(&head_mutex);
    while (1 == 1) {
        pthread_mutex_lock(&cond_mutex);
        pthread_cond_signal(&cond);
        pthread_mutex_unlock(&cond_mutex);
        if (is_done(fut)) {
            pthread_mutex_lock(&head_mutex);
            free(call_node);
            free(call);
            K res = fut->res;
            free(fut);
            pthread_mutex_unlock(&head_mutex);
            return res;
        }
    }
    return (K)0;
}

static K (*__ka)(int t);
K _ka(int t) {
    return __ka(t);
}

static K (*__kb)(int x);
K _kb(int x) {
    return __kb(x);
}

static K (*__kc)(int x);
K _kc(int x) {
    return __kc(x);
}

static void (*__kclose)(int x);
void _kclose(int x) {
    __kclose(x);
}

static K (*__kd)(int x);
K _kd(int x) {
    return __kd(x);
}

static K (*__ke)(double x);
K _ke(double x) {
    return __ke(x);
}

static K (*__kf)(double x);
K _kf(double x) {
    return __kf(x);
}

static K (*__kg)(int x);
K _kg(int x) {
    return __kg(x);
}

static K (*__kh)(int x);
K _kh(int x) {
    return __kh(x);
}

static int (*__khpunc)(char* v, int w, char* x, int y, int z);
int _khpunc(char* v, int w, char* x, int y, int z) {
    return __khpunc(v, w, x, y, z);
}

static K (*__ki)(int x);
K _ki(int x) {
    return __ki(x);
}

static K (*__kj)(long long x);
K _kj(long long x){
    return __kj(x);
}

static K (*__knk)(int n, ...);
K _knk(int n, ...) {
    void** args = (void**)malloc(sizeof(void*) * n);

    va_list argp;
    va_start(argp, n);
    for (int i = 0; i < n; i++) {
        args[i] = va_arg(argp, void*);
    }
    va_end(argp);
    K res = (K)0;
    switch (n) {
        case 1:
            res = __knk(n, args[0]);
            break;
        case 2:
            res = __knk(n, args[0], args[1]);
            break;
        case 3:
            res = __knk(n, args[0], args[1], args[2]);
            break;
        case 4:
            res = __knk(n, args[0], args[1], args[2], args[3]);
            break;
        // TODO: We only use knk(2, ...) internally but there may be a point where we need more.
        default:
            free(args);
            return res;
    }

    free(args);
    return res;
}

static K (*__knt)(long long n, K x);
K _knt(long long n, K x) {
    return __knt(n, x);
}

static K (*__kp)(char* x);
K _kp(char* x) {
    return __kp(x);
}


static K (*__kpn)(char* x, long long n);
K _kpn(char* x, long long n) {
    return __kpn(x, n);
}

static K (*__krr)(const char* s);
K _krr(const char* s) {
    return __krr(s);
}


static K (*__ks)(char* x);
K _ks(char* x) {
    return __ks(x);
}

static K (*__kt)(int x);
K _kt(int x) {
    return __kt(x);
}

static K (*__ktd)(K x);
K _ktd(K x) {
    return __ktd(x);
}

static K (*__ktj)(short _type, long long x);
K _ktj(short _type, long long x) {
    return __ktj(_type, x);
}

static K (*__ktn)(int _type, long long length);
K _ktn(int _type, long long length) {
    return __ktn(_type, length);
}

static K (*__ku)(U x);
K _ku(U x) {
    return __ku(x);
}

static K (*__kz)(double x);
K _kz(double x) {
    return __kz(x);
}

static void (*__m9)();
void _m9() {
    return __m9();
}

static int (*__okx)(K x);
int _okx(K x) {
    return __okx(x);
}

static K (*__orr)(const char* x);
K _orr(const char* x) {
    return __orr(x);
}

static void (*__r0)(K k);
void _r0(K k){
    __r0(k);
}

static K (*__r1)(K k);
K _r1(K k) {
    return __r1(k);
}

static void (*__sd0)(int d);
void _sd0(int d) {
    return __sd0(d);
}

static void (*__sd0x)(int d, int f);
void _sd0x(int d, int f) {
    return __sd0x(d, f);
}

static K (*__sd1)(int d, void* (*f)(int));
K _sd1(int d, void* (*f)(int)) {
    return __sd1(d, f);
}

static char* (*__sn)(char* s, long long n);
char* _sn(char* s, long long n) {
    return __sn(s, n);
}

static char* (*__ss)(char* s);
char* _ss(char* s) {
    return __ss(s);
}

static K (*__sslInfo)(K x);
K _sslInfo(K x) {
    return __sslInfo(x);
}

static K (*__vak)(int x, const char* s, va_list l);
K _vak(int x, const char* s, va_list l) {
    return __vak(x, s, l);
}

static K (*__vaknk)(int x, va_list l);
K _vaknk(int x, va_list l) {
    return __vaknk(x, l);
}

static int (*__ver)();
int _ver() {
    return __ver();
}

static K (*__xD)(K x, K y);
K _xD(K x, K y) {
    return __xD(x, y);
}

static K (*__xT)(K x);
K _xT(K x) {
    return __xT(x);
}

static int (*__ymd)(int year, int month, int day);
int _ymd(int year, int month, int day) {
    return __ymd(year, month, day);
}

int (*_qinit)(int, char**, char*, char*, char*);

struct QInit {
    int argc;
    char** argv;
    char* qhome;
    char* qlic;
    char* qqq;
};

void* q_thread_init(void* _qini) {
    struct QInit* qini = (struct QInit*)_qini;
    qinit_rc = _qinit(qini->argc, qini->argv, qini->qhome, qini->qlic, qini->qqq);
    pthread_mutex_lock(&init_mutex);
    pthread_cond_signal(&init);
    pthread_mutex_unlock(&init_mutex);
    while (1 == 1) {
        pthread_mutex_lock(&cond_mutex);
        while (calls_head == NULL && kill_thread == false) {
            pthread_cond_wait(&cond, &cond_mutex);
        }
        pthread_mutex_unlock(&cond_mutex);
        pthread_mutex_lock(&head_mutex);
        if (kill_thread)
            break;
        if (calls_head != NULL) {
            struct QCall* call = calls_head->call;
            if (call->is_dot) {
                K res = _dot_internal(call->arg0, call->arg1);
                call->fut->res = res;
                call->fut->done = true;
            } else {
                K res = _k_internal(call->handle, call->query, call->argc, call->arg0, call->arg1, call->arg2, call->arg3, call->arg4, call->arg5, call->arg6, call->arg7);
                call->fut->res = res;
                call->fut->done = true;
            }
            calls_head = calls_head->next;
        }
        pthread_mutex_unlock(&head_mutex);
    }
    pthread_exit(0);
    return NULL;
}

void shutdown_thread() {
    pthread_mutex_lock(&head_mutex);
    kill_thread = true;
    pthread_mutex_unlock(&head_mutex);
    pthread_mutex_lock(&cond_mutex);
    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&cond_mutex);
}

int q_init(int argc, char** argv, char* qhome, char* qlic, char* qqq) {
    calls_head = NULL;
    calls_tail = NULL;
    kill_thread = false;
    pthread_mutex_init(&head_mutex, NULL);
    pthread_mutex_init(&cond_mutex, NULL);
    pthread_cond_init(&cond, NULL);
    pthread_mutex_init(&init_mutex, NULL);
    pthread_cond_init(&init, NULL);
    struct QInit* qini = malloc(sizeof(struct QInit));
    qini->argc = argc;
    qini->argv = argv;
    qini->qhome = qhome;
    qini->qlic = qlic;
    qini->qqq = qqq; // TODO: ADD COMMENT
    qinit_rc = -256;
    int rc = pthread_create(&q_thread, NULL, q_thread_init, (void*)qini);

    pthread_mutex_lock(&init_mutex);
    while (qinit_rc == -256) {
        pthread_cond_wait(&init, &init_mutex);
    }
    pthread_mutex_unlock(&init_mutex);
    return qinit_rc;
}

void sym_init(char* libq_path) {
    _q_handle = dlopen(libq_path, RTLD_NOW | RTLD_GLOBAL);

    _qinit = dlsym(_q_handle, "qinit");
    __b9 = dlsym(_q_handle, "b9");
    __d9 = dlsym(_q_handle, "d9");
    __dj = dlsym(_q_handle, "dj");
    __dl = dlsym(_q_handle, "dl");
    __dot = dlsym(_q_handle, "dot");
    __ee = dlsym(_q_handle, "ee");
    __ja = dlsym(_q_handle, "ja");
    __jk = dlsym(_q_handle, "jk");
    __js = dlsym(_q_handle, "js");
    __jv = dlsym(_q_handle, "jv");
    __k = dlsym(_q_handle, "k");
    __ka = dlsym(_q_handle, "ka");
    __kb = dlsym(_q_handle, "kb");
    __kc = dlsym(_q_handle, "kc");
    __kclose = dlsym(_q_handle, "kclose");
    __kd = dlsym(_q_handle, "kd");
    __ke = dlsym(_q_handle, "ke");
    __kf = dlsym(_q_handle, "kf");
    __kg = dlsym(_q_handle, "kg");
    __kh = dlsym(_q_handle, "kh");
    __khpunc = dlsym(_q_handle, "khpunc");
    __ki = dlsym(_q_handle, "ki");
    __kj = dlsym(_q_handle, "kj");
    __knk = dlsym(_q_handle, "knk");
    __knt = dlsym(_q_handle, "knt");
    __kp = dlsym(_q_handle, "kp");
    __kpn = dlsym(_q_handle, "kpn");
    __krr = dlsym(_q_handle, "krr");
    __ks = dlsym(_q_handle, "ks");
    __kt = dlsym(_q_handle, "kt");
    __ktd = dlsym(_q_handle, "ktd");
    __ktj = dlsym(_q_handle, "ktj");
    __ktn = dlsym(_q_handle, "ktn");
    __ku = dlsym(_q_handle, "ku");
    __kz = dlsym(_q_handle, "kz");
    __m9 = dlsym(_q_handle, "m9");
    __okx = dlsym(_q_handle, "okx");
    __orr = dlsym(_q_handle, "orr");
    __r0 = dlsym(_q_handle, "r0");
    __r1 = dlsym(_q_handle, "r1");
    __sd0 = dlsym(_q_handle, "sd0");
    __sd0x = dlsym(_q_handle, "sd0x");
    __sd1 = dlsym(_q_handle, "sd1");
    __sn = dlsym(_q_handle, "sn");
    __ss = dlsym(_q_handle, "ss");
    __sslInfo = dlsym(_q_handle, "sslInfo");
    __vak = dlsym(_q_handle, "vak");
    __vaknk = dlsym(_q_handle, "vaknk");
    __ver = dlsym(_q_handle, "ver");
    __xD = dlsym(_q_handle, "xD");
    __xT = dlsym(_q_handle, "xT");
    __ymd = dlsym(_q_handle, "ymd");
}
