#ifndef KX
#define KX
#define KXVER 3
typedef struct k0{
    signed char m,a,t;
    char u;
    int r;
    union{
        unsigned char g;
        short h;
        int i;
        long long j;
        float e;
        double f;
        char* s;
        struct k0* k;
        struct{
            long long n;
            unsigned char G0[1];
        };
    };
} *K;
typedef struct _U{
    unsigned char g[16];
} U;
#define kU(x) ((U*)kG(x))
#define xU ((U*)xG)
extern K ku(U),knt(long long,K),ktn(int,long long),kpn(char*,long long);
extern int setm(int),ver();
#define DO(n,x)	{long long i=0,_i=(n);for(;i<_i;++i){x;}}
extern int qinit(int, char**, char*, char*, char*);

// vector accessors, e.g. kF(x)[i] for float&datetime
#define kG(x)	((x)->G0)
#define kC(x)	kG(x)
#define kH(x)	((short*)kG(x))
#define kI(x)	((int*)kG(x))
#define kJ(x)	((long long*)kG(x))
#define kE(x)	((float*)kG(x))
#define kF(x)	((double*)kG(x))
#define kS(x)	((char**)kG(x))
#define kK(x)	((K*)kG(x))

//      type bytes qtype     ctype  accessor
#define KB 1  // 1 boolean   char   kG
#define UU 2  // 16 guid     U      kU
#define KG 4  // 1 byte      char   kG
#define KH 5  // 2 short     short  kH
#define KI 6  // 4 int       int    kI
#define KJ 7  // 8 long      long   kJ
#define KE 8  // 4 real      float  kE
#define KF 9  // 8 float     double kF
#define KC 10 // 1 char      char   kC
#define KS 11 // * symbol    char*  kS

#define KP 12 // 8 timestamp long   kJ (nanoseconds from 2000.01.01)
#define KM 13 // 4 month     int    kI (months from 2000.01.01)
#define KD 14 // 4 date      int    kI (days from 2000.01.01)

#define KN 16 // 8 timespan  long   kJ (nanoseconds)
#define KU 17 // 4 minute    int    kI
#define KV 18 // 4 second    int    kI
#define KT 19 // 4 time      int    kI (millisecond)

#define KZ 15 // 8 datetime  double kF (DO NOT USE)

// table,dict
#define XT 98 //   x->k is XD
#define XD 99 //   kK(x)[0] is keys. kK(x)[1] is values.

#ifdef __cplusplus
#include<cstdarg>
extern"C"{
extern void m9();
#else
#include<stdarg.h>
extern void m9(void);
#endif
extern int khpunc(char*,int,char*,int,int),
       khpun(const char*,int,const char*,int),
       khpu(const char*,int,const char*),
       khp(const char*,int),
       okx(K),
       ymd(int,int,int),
       dj(int);
extern void r0(K),
       sd0(int),
       sd0x(int d,int f),
       kclose(int);
extern char* sn(char*,int),
       ss(char*);
extern K ee(K),
       ktj(int,long long),
       ka(int),
       kb(int),
       kg(int),
       kh(int),
       ki(int),
       kj(long long),
       ke(double),
       kf(double),
       kc(int),
       ks(char*),
       kd(int),
       kz(double),
       kt(int),
       sd1(int,K(*)(int)),
       dl(void*f,long long),
       knk(int,...),
       kp(char*),
       ja(K*,void*),
       js(K*,char*),
       jk(K*,K),
       jv(K*k,K),
       k(int,const char*,...),
       xT(K),
       xD(K,K),
       ktd(K),
       r1(K),
       krr(const char*),
       orr(const char*),
       dot(K,K),
       b9(int,K),
       d9(K),
       sslInfo(K x),
       vaknk(int,va_list),
       vak(int,const char*,va_list);
#ifdef __cplusplus
}
#endif

// nulls(n?) and infinities(w?)
#define nh ((int)0xFFFF8000)
#define wh ((int)0x7FFF)
#define ni ((int)0x80000000)
#define wi ((int)0x7FFFFFFF)
#define nj ((long long)0x8000000000000000LL)
#define wj 0x7FFFFFFFFFFFFFFFLL
#if defined(WIN32) || defined(_WIN32)
#define nf (log(-1.0))
#define finite _finite
extern double log(double);
#else
#define nf (0/0.0)
#define closesocket(x) close(x)
#endif

// remove more clutter
/*
#define O printf
#define Z static
#define SW switch
#define CS(n,x)	case n:x;break;
#define CD default

#define ZV Z V
#define ZK Z K
#define ZH Z H
#define ZI Z I
#define ZJ Z J
#define ZE Z E
#define ZF Z F
#define ZC Z C
#define ZS Z S

#define K1(f) K f(K x)
#define K2(f) K f(K x,K y)
#define TX(T,x) (*(T*)((G*)(x)+8))
#define xr x->r
#define xt x->t
#define xu x->u
#define xx xK[0]
#define xy xK[1]
#define xg TX(G,x)
#define xh TX(H,x)
#define xi TX(I,x)
#define xj TX(J,x)
#define xe TX(E,x)
#define xf TX(F,x)
#define xs TX(S,x)
#define xk TX(K,x)
#define xG x->G0
#define xH ((H*)xG)
#define xI ((I*)xG)
#define xJ ((J*)xG)
#define xE ((E*)xG)
#define xF ((F*)xG)
#define xS ((S*)xG)
#define xK ((K*)xG)
#define xC xG
#define xB ((G*)xG)
*/
#define xt x->t
#define P(x,y) {if(x)return(y);}
#define U(x) P(!(x),0)
#endif
