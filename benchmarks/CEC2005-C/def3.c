/* Source file for custom initialization */
/* Hard-coded for every function based on the type and nature of input files */
/* At present hard-coded for D=2, 10, 30 and 50 */

# include <stdio.h>
# include <stdlib.h>
# include <math.h>

# include "global.h"
# include "sub.h"
# include "rand.h"

# ifdef f1
void initialize()
{
    int i, j;
    FILE *fpt;
    fpt = fopen("input_data/sphere_func_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
    }
    fclose(fpt);
    bias[0] = -450.0;
    return;
}
#endif

# ifdef f2
void initialize()
{
    int i, j;
    FILE *fpt;
    fpt = fopen("input_data/schwefel_102_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
    }
    fclose(fpt);
    bias[0] = -450.0;
    return;
}
#endif

# ifdef f3
void initialize()
{
    int i, j;
    FILE *fpt;
    if (nreal==2) fpt = fopen("input_data/elliptic_M_D2.txt","r");
    if (nreal==10) fpt = fopen("input_data/elliptic_M_D10.txt","r");
    if (nreal==30) fpt = fopen("input_data/elliptic_M_D30.txt","r");
    if (nreal==50) fpt = fopen("input_data/elliptic_M_D50.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nreal; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&g[i][j]);
            printf("\n G[%d][%d] = %LE",i+1,j+1,g[i][j]);
        }
    }
    fclose(fpt);
    fpt = fopen("input_data/high_cond_elliptic_rot_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
    }
    fclose(fpt);
    bias[0] = -450.0;
    return;
}
#endif

# ifdef f4
void initialize()
{
    int i, j;
    FILE *fpt;
    fpt = fopen("input_data/schwefel_102_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
    }
    fclose(fpt);
    bias[0] = -450.0;
    return;
}
#endif

# ifdef f5
void initialize()
{
    int i, j;
    int index;
    FILE *fpt;
    char c;
    A = (long double **)malloc(nreal*sizeof(long double));
    for (i=0; i<nreal; i++)
    {
        A[i] = (long double *)malloc(nreal*sizeof(long double));
    }
    B = (long double *)malloc(nreal*sizeof(long double));
    fpt = fopen("input_data/schwefel_206_data.txt","r");
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
        do
        {
            fscanf(fpt,"%c",&c);
        }
        while (c!='\n');
    }
    for (i=0; i<nreal; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&A[i][j]);
            printf("\n A[%d][%d] = %LE",i+1,j+1,A[i][j]);
        }
        do
        {
            fscanf(fpt,"%c",&c);
        }
        while (c!='\n');
    }
    fclose(fpt);
    if (nreal%4==0)
    {
        index = nreal/4;
    }
    else
    {
        index = nreal/4 + 1;
    }
    for (i=0; i<index; i++)
    {
        o[0][i] = -100.0;
    }
    index = (3*nreal)/4 - 1;
    for (i=index; i<nreal; i++)
    {
        o[0][i] = 100.0;
    }
    for (i=0; i<nreal; i++)
    {
        B[i] = 0.0;
        for (j=0; j<nreal; j++)
        {
            B[i] += A[i][j]*o[0][j];
        }
    }
    bias[0] = -310.0;
    return;
}
#endif

# ifdef f6
void initialize()
{
    int i, j;
    FILE *fpt;
    fpt = fopen("input_data/rosenbrock_func_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            o[i][j] -= 1.0;
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
    }
    fclose(fpt);
    bias[0] = 390.0;
    return;
}
#endif

# ifdef f7
void initialize()
{
    int i, j;
    FILE *fpt;
    if (nreal==2)    fpt = fopen("input_data/griewank_M_D2.txt","r");
    if (nreal==10)    fpt = fopen("input_data/griewank_M_D10.txt","r");
    if (nreal==30)    fpt = fopen("input_data/griewank_M_D30.txt","r");
    if (nreal==50)    fpt = fopen("input_data/griewank_M_D50.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nreal; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&g[i][j]);
            printf("\n G[%d][%d] = %LE",i+1,j+1,g[i][j]);
        }
    }
    fclose(fpt);
    fpt = fopen("input_data/griewank_func_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
    }
    fclose(fpt);
    bias[0] = -180.0;
    return;
}
#endif

# ifdef f8
void initialize()
{
    int i, j;
    int index;
    FILE *fpt;
    if (nreal==2)    fpt = fopen("input_data/ackley_M_D2.txt","r");
    if (nreal==10)    fpt = fopen("input_data/ackley_M_D10.txt","r");
    if (nreal==30)    fpt = fopen("input_data/ackley_M_D30.txt","r");
    if (nreal==50)    fpt = fopen("input_data/ackley_M_D50.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nreal; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&g[i][j]);
            printf("\n M[%d][%d] = %LE",i+1,j+1,g[i][j]);
        }
    }
    fclose(fpt);
    fpt = fopen("input_data/ackley_func_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
    }
    fclose(fpt);
    index = nreal/2;
    for (i=1; i<=index; i++)
    {
        o[0][2*i-2] = -32.0;
    }
    bias[0] = -140.0;
    return;
}
#endif

# ifdef f9
void initialize()
{
    int i, j;
    FILE *fpt;
    fpt = fopen("input_data/rastrigin_func_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
    }
    fclose(fpt);
    bias[0] = -330.0;
    return;
}
#endif

# ifdef f10
void initialize()
{
    int i, j;
    FILE *fpt;
    if (nreal==2)    fpt = fopen("input_data/rastrigin_M_D2.txt","r");
    if (nreal==10)    fpt = fopen("input_data/rastrigin_M_D10.txt","r");
    if (nreal==30)    fpt = fopen("input_data/rastrigin_M_D30.txt","r");
    if (nreal==50)    fpt = fopen("input_data/rastrigin_M_D50.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nreal; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&g[i][j]);
            printf("\n M[%d][%d] = %LE",i+1,j+1,g[i][j]);
        }
    }
    fclose(fpt);
    fpt = fopen("input_data/rastrigin_func_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
    }
    fclose(fpt);
    bias[0] = -330.0;
    return;
}
#endif

# ifdef f11
void initialize()
{
    int i, j;
    FILE *fpt;
    if (nreal==2)    fpt = fopen("input_data/weierstrass_M_D2.txt","r");
    if (nreal==10)    fpt = fopen("input_data/weierstrass_M_D10.txt","r");
    if (nreal==30)    fpt = fopen("input_data/weierstrass_M_D30.txt","r");
    if (nreal==50)    fpt = fopen("input_data/weierstrass_M_D50.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nreal; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&g[i][j]);
            printf("\n M[%d][%d] = %LE",i+1,j+1,g[i][j]);
        }
    }
    fclose(fpt);
    fpt = fopen("input_data/weierstrass_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
    }
    fclose(fpt);
    bias[0] = 90.0;
    return;
}
#endif

# ifdef f12
void initialize()
{
    int i, j;
    FILE *fpt;
    char c;
    A = (long double **)malloc(nreal*sizeof(long double));
    B = (long double **)malloc(nreal*sizeof(long double));
    alpha = (long double *)malloc(nreal*sizeof(long double));
    for (i=0; i<nreal; i++)
    {
        A[i] = (long double *)malloc(nreal*sizeof(long double));
        B[i] = (long double *)malloc(nreal*sizeof(long double));
    }
    fpt = fopen("input_data/schwefel_213_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    /* Reading A */
    for (i=0; i<nreal; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&A[i][j]);
            printf("\n A[%d][%d] = %LE",i+1,j+1,A[i][j]);
        }
        do
        {
            fscanf(fpt,"%c",&c);
        }
        while (c!='\n');
    }
    if (i!=100)
    {
        for (i=nreal; i<100; i++)
        {
            do
            {
                fscanf(fpt,"%c",&c);
            }
            while(c!='\n');
        }
    }
    /* Reading B */
    for (i=0; i<nreal; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&B[i][j]);
            printf("\n B[%d][%d] = %LE",i+1,j+1,B[i][j]);
        }
        do
        {
            fscanf(fpt,"%c",&c);
        }
        while (c!='\n');
    }
    if (i!=100)
    {
        for (i=nreal; i<100; i++)
        {
            do
            {
                fscanf(fpt,"%c",&c);
            }
            while(c!='\n');
        }
    }
    /* Reading alpha */
    for (i=0; i<nreal; i++)
    {
        fscanf(fpt,"%Lf",&alpha[i]);
        printf("\n alpha[%d] = %LE",i+1,alpha[i]);
    }
    fclose(fpt);
    bias[0] = -460.0;
    return;
}
#endif

# ifdef f13
void initialize()
{
    int i, j;
    FILE *fpt;
    fpt = fopen("input_data/EF8F2_func_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            o[i][j] -= 1.0;
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
    }
    fclose(fpt);
    bias[0] = -130.0;
    return;
}
#endif

# ifdef f14
void initialize()
{
    int i, j;
    FILE *fpt;
    if (nreal==2)    fpt = fopen("input_data/E_ScafferF6_M_D2.txt","r");
    if (nreal==10)    fpt = fopen("input_data/E_ScafferF6_M_D10.txt","r");
    if (nreal==30)    fpt = fopen("input_data/E_ScafferF6_M_D30.txt","r");
    if (nreal==50)    fpt = fopen("input_data/E_ScafferF6_M_D50.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nreal; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&g[i][j]);
            printf("\n M[%d][%d] = %LE",i+1,j+1,g[i][j]);
        }
    }
    fclose(fpt);
    fpt = fopen("input_data/E_ScafferF6_func_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
    }
    fclose(fpt);
    bias[0] = -300.0;
    return;
}
#endif

# ifdef f15
void initialize()
{
    int i, j;
    FILE *fpt;
    char c;
    fpt = fopen("input_data/hybrid_func1_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
        do
        {
            fscanf(fpt,"%c",&c);
        }
        while (c!='\n');
        printf("\n");
    }
    fclose(fpt);
    lambda[0] = 1.0;
    lambda[1] = 1.0;
    lambda[2] = 10.0;
    lambda[3] = 10.0;
    lambda[4] = 1.0/12.0;
    lambda[5] = 1.0/12.0;
    lambda[6] = 5.0/32.0;
    lambda[7] = 5.0/32.0;
    lambda[8] = 1.0/20.0;
    lambda[9] = 1.0/20.0;
    global_bias = 120.0;
    return;
}
#endif

# ifdef f16
void initialize()
{
    int i, j, k;
    FILE *fpt;
    char c;
    fpt = fopen("input_data/hybrid_func1_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
        do
        {
            fscanf(fpt,"%c",&c);
        }
        while (c!='\n');
        printf("\n");
    }
    fclose(fpt);
    if (nreal==2)    fpt = fopen("input_data/hybrid_func1_M_D2.txt","r");
    if (nreal==10)    fpt = fopen("input_data/hybrid_func1_M_D10.txt","r");
    if (nreal==30)    fpt = fopen("input_data/hybrid_func1_M_D30.txt","r");
    if (nreal==50)    fpt = fopen("input_data/hybrid_func1_M_D50.txt","r");
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            for (k=0; k<nreal; k++)
            {
                fscanf(fpt,"%Lf",&l[i][j][k]);
                printf("\n M[%d][%d][%d] = %LE",i+1,j+1,k+1,l[i][j][k]);
            }
            do
            {
                fscanf(fpt,"%c",&c);
            }
            while (c!='\n');
        }
        printf("\n");
    }
    lambda[0] = 1.0;
    lambda[1] = 1.0;
    lambda[2] = 10.0;
    lambda[3] = 10.0;
    lambda[4] = 1.0/12.0;
    lambda[5] = 1.0/12.0;
    lambda[6] = 5.0/32.0;
    lambda[7] = 5.0/32.0;
    lambda[8] = 1.0/20.0;
    lambda[9] = 1.0/20.0;
    global_bias = 120.0;
    return;
}
#endif

# ifdef f17
void initialize()
{
    int i, j, k;
    FILE *fpt;
    char c;
    fpt = fopen("input_data/hybrid_func1_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
        do
        {
            fscanf(fpt,"%c",&c);
        }
        while (c!='\n');
        printf("\n");
    }
    fclose(fpt);
    if (nreal==2)    fpt = fopen("input_data/hybrid_func1_M_D2.txt","r");
    if (nreal==10)    fpt = fopen("input_data/hybrid_func1_M_D10.txt","r");
    if (nreal==30)    fpt = fopen("input_data/hybrid_func1_M_D30.txt","r");
    if (nreal==50)    fpt = fopen("input_data/hybrid_func1_M_D50.txt","r");
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            for (k=0; k<nreal; k++)
            {
                fscanf(fpt,"%Lf",&l[i][j][k]);
                printf("\n M[%d][%d][%d] = %LE",i+1,j+1,k+1,l[i][j][k]);
            }
            do
            {
                fscanf(fpt,"%c",&c);
            }
            while (c!='\n');
        }
        printf("\n");
    }
    lambda[0] = 1.0;
    lambda[1] = 1.0;
    lambda[2] = 10.0;
    lambda[3] = 10.0;
    lambda[4] = 1.0/12.0;
    lambda[5] = 1.0/12.0;
    lambda[6] = 5.0/32.0;
    lambda[7] = 5.0/32.0;
    lambda[8] = 1.0/20.0;
    lambda[9] = 1.0/20.0;
    global_bias = 120.0;
    return;
}
#endif

# ifdef f18
void initialize()
{
    int i, j, k;
    FILE *fpt;
    char c;
    fpt = fopen("input_data/hybrid_func2_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
        do
        {
            fscanf(fpt,"%c",&c);
        }
        while (c!='\n');
        printf("\n");
    }
    fclose(fpt);
    if (nreal==2)    fpt = fopen("input_data/hybrid_func2_M_D2.txt","r");
    if (nreal==10)    fpt = fopen("input_data/hybrid_func2_M_D10.txt","r");
    if (nreal==30)    fpt = fopen("input_data/hybrid_func2_M_D30.txt","r");
    if (nreal==50)    fpt = fopen("input_data/hybrid_func2_M_D50.txt","r");
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            for (k=0; k<nreal; k++)
            {
                fscanf(fpt,"%Lf",&l[i][j][k]);
                printf("\n M[%d][%d][%d] = %LE",i+1,j+1,k+1,l[i][j][k]);
            }
            do
            {
                fscanf(fpt,"%c",&c);
            }
            while (c!='\n');
        }
        printf("\n");
    }
    for (i=0; i<nreal; i++)
    {
        o[nfunc-1][i] = 0.0;
    }
    sigma[0] = 1.0;
    sigma[1] = 2.0;
    sigma[2] = 1.5;
    sigma[3] = 1.5;
    sigma[4] = 1.0;
    sigma[5] = 1.0;
    sigma[6] = 1.5;
    sigma[7] = 1.5;
    sigma[8] = 2.0;
    sigma[9] = 2.0;
    lambda[0] = 5.0/16.0;
    lambda[1] = 5.0/32.0;
    lambda[2] = 2.0;
    lambda[3] = 1.0;
    lambda[4] = 1.0/10.0;
    lambda[5] = 1.0/20.0;
    lambda[6] = 20.0;
    lambda[7] = 10.0;
    lambda[8] = 1.0/6.0;
    lambda[9] = 1.0/12.0;
    global_bias = 10.0;
    return;
}
#endif

# ifdef f19
void initialize()
{
    int i, j, k;
    FILE *fpt;
    char c;
    fpt = fopen("input_data/hybrid_func2_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
        do
        {
            fscanf(fpt,"%c",&c);
        }
        while (c!='\n');
        printf("\n");
    }
    fclose(fpt);
    if (nreal==2)    fpt = fopen("input_data/hybrid_func2_M_D2.txt","r");
    if (nreal==10)    fpt = fopen("input_data/hybrid_func2_M_D10.txt","r");
    if (nreal==30)    fpt = fopen("input_data/hybrid_func2_M_D30.txt","r");
    if (nreal==50)    fpt = fopen("input_data/hybrid_func2_M_D50.txt","r");
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            for (k=0; k<nreal; k++)
            {
                fscanf(fpt,"%Lf",&l[i][j][k]);
                printf("\n M[%d][%d][%d] = %LE",i+1,j+1,k+1,l[i][j][k]);
            }
            do
            {
                fscanf(fpt,"%c",&c);
            }
            while (c!='\n');
        }
        printf("\n");
    }
    for (i=0; i<nreal; i++)
    {
        o[nfunc-1][i] = 0.0;
    }
    sigma[0] = 0.1;
    sigma[1] = 2.0;
    sigma[2] = 1.5;
    sigma[3] = 1.5;
    sigma[4] = 1.0;
    sigma[5] = 1.0;
    sigma[6] = 1.5;
    sigma[7] = 1.5;
    sigma[8] = 2.0;
    sigma[9] = 2.0;
    lambda[0] = 0.5/32.0;
    lambda[1] = 5.0/32.0;
    lambda[2] = 2.0;
    lambda[3] = 1.0;
    lambda[4] = 1.0/10.0;
    lambda[5] = 1.0/20.0;
    lambda[6] = 20.0;
    lambda[7] = 10.0;
    lambda[8] = 1.0/6.0;
    lambda[9] = 1.0/12.0;
    global_bias = 10.0;
    return;
}
#endif

# ifdef f20
void initialize()
{
    int i, j, k;
    int index;
    FILE *fpt;
    char c;
    fpt = fopen("input_data/hybrid_func2_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
        do
        {
            fscanf(fpt,"%c",&c);
        }
        while (c!='\n');
        printf("\n");
    }
    fclose(fpt);
    index = nreal/2;
    for (i=1; i<=index; i++)
    {
        o[0][2*i-1] = 5.0;
    }
    if (nreal==2)    fpt = fopen("input_data/hybrid_func2_M_D2.txt","r");
    if (nreal==10)    fpt = fopen("input_data/hybrid_func2_M_D10.txt","r");
    if (nreal==30)    fpt = fopen("input_data/hybrid_func2_M_D30.txt","r");
    if (nreal==50)    fpt = fopen("input_data/hybrid_func2_M_D50.txt","r");
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            for (k=0; k<nreal; k++)
            {
                fscanf(fpt,"%Lf",&l[i][j][k]);
                printf("\n M[%d][%d][%d] = %LE",i+1,j+1,k+1,l[i][j][k]);
            }
            do
            {
                fscanf(fpt,"%c",&c);
            }
            while (c!='\n');
        }
        printf("\n");
    }
    for (i=0; i<nreal; i++)
    {
        o[nfunc-1][i] = 0.0;
    }
    sigma[0] = 1.0;
    sigma[1] = 2.0;
    sigma[2] = 1.5;
    sigma[3] = 1.5;
    sigma[4] = 1.0;
    sigma[5] = 1.0;
    sigma[6] = 1.5;
    sigma[7] = 1.5;
    sigma[8] = 2.0;
    sigma[9] = 2.0;
    lambda[0] = 5.0/16.0;
    lambda[1] = 5.0/32.0;
    lambda[2] = 2.0;
    lambda[3] = 1.0;
    lambda[4] = 1.0/10.0;
    lambda[5] = 1.0/20.0;
    lambda[6] = 20.0;
    lambda[7] = 10.0;
    lambda[8] = 1.0/6.0;
    lambda[9] = 1.0/12.0;
    global_bias = 10.0;
    return;
}
#endif

# ifdef f21
void initialize()
{
    int i, j, k;
    FILE *fpt;
    char c;
    fpt = fopen("input_data/hybrid_func3_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
        do
        {
            fscanf(fpt,"%c",&c);
        }
        while (c!='\n');
        printf("\n");
    }
    fclose(fpt);
    if (nreal==2)    fpt = fopen("input_data/hybrid_func3_M_D2.txt","r");
    if (nreal==10)    fpt = fopen("input_data/hybrid_func3_M_D10.txt","r");
    if (nreal==30)    fpt = fopen("input_data/hybrid_func3_M_D30.txt","r");
    if (nreal==50)    fpt = fopen("input_data/hybrid_func3_M_D50.txt","r");
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            for (k=0; k<nreal; k++)
            {
                fscanf(fpt,"%Lf",&l[i][j][k]);
                printf("\n M[%d][%d][%d] = %LE",i+1,j+1,k+1,l[i][j][k]);
            }
            do
            {
                fscanf(fpt,"%c",&c);
            }
            while (c!='\n');
        }
        printf("\n");
    }
    sigma[0] = 1.0;
    sigma[1] = 1.0;
    sigma[2] = 1.0;
    sigma[3] = 1.0;
    sigma[4] = 1.0;
    sigma[5] = 2.0;
    sigma[6] = 2.0;
    sigma[7] = 2.0;
    sigma[8] = 2.0;
    sigma[9] = 2.0;
    lambda[0] = 1.0/4.0;
    lambda[1] = 1.0/20.0;
    lambda[2] = 5.0;
    lambda[3] = 1.0;
    lambda[4] = 5.0;
    lambda[5] = 1.0;
    lambda[6] = 50.0;
    lambda[7] = 10.0;
    lambda[8] = 1.0/8.0;
    lambda[9] = 1.0/40.0;
    global_bias = 360.0;
    return;
}
#endif

# ifdef f22
void initialize()
{
    int i, j, k;
    FILE *fpt;
    char c;
    fpt = fopen("input_data/hybrid_func3_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
        do
        {
            fscanf(fpt,"%c",&c);
        }
        while (c!='\n');
        printf("\n");
    }
    fclose(fpt);
    if (nreal==2)    fpt = fopen("input_data/hybrid_func3_HM_D2.txt","r");
    if (nreal==10)    fpt = fopen("input_data/hybrid_func3_HM_D10.txt","r");
    if (nreal==30)    fpt = fopen("input_data/hybrid_func3_HM_D30.txt","r");
    if (nreal==50)    fpt = fopen("input_data/hybrid_func3_HM_D50.txt","r");
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            for (k=0; k<nreal; k++)
            {
                fscanf(fpt,"%Lf",&l[i][j][k]);
                printf("\n M[%d][%d][%d] = %LE",i+1,j+1,k+1,l[i][j][k]);
            }
            do
            {
                fscanf(fpt,"%c",&c);
            }
            while (c!='\n');
        }
        printf("\n");
    }
    sigma[0] = 1.0;
    sigma[1] = 1.0;
    sigma[2] = 1.0;
    sigma[3] = 1.0;
    sigma[4] = 1.0;
    sigma[5] = 2.0;
    sigma[6] = 2.0;
    sigma[7] = 2.0;
    sigma[8] = 2.0;
    sigma[9] = 2.0;
    lambda[0] = 1.0/4.0;
    lambda[1] = 1.0/20.0;
    lambda[2] = 5.0;
    lambda[3] = 1.0;
    lambda[4] = 5.0;
    lambda[5] = 1.0;
    lambda[6] = 50.0;
    lambda[7] = 10.0;
    lambda[8] = 1.0/8.0;
    lambda[9] = 1.0/40.0;
    global_bias = 360.0;
    return;
}
#endif

# ifdef f23
void initialize()
{
    int i, j, k;
    FILE *fpt;
    char c;
    fpt = fopen("input_data/hybrid_func3_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
        do
        {
            fscanf(fpt,"%c",&c);
        }
        while (c!='\n');
        printf("\n");
    }
    fclose(fpt);
    if (nreal==2)    fpt = fopen("input_data/hybrid_func3_M_D2.txt","r");
    if (nreal==10)    fpt = fopen("input_data/hybrid_func3_M_D10.txt","r");
    if (nreal==30)    fpt = fopen("input_data/hybrid_func3_M_D30.txt","r");
    if (nreal==50)    fpt = fopen("input_data/hybrid_func3_M_D50.txt","r");
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            for (k=0; k<nreal; k++)
            {
                fscanf(fpt,"%Lf",&l[i][j][k]);
                printf("\n M[%d][%d][%d] = %LE",i+1,j+1,k+1,l[i][j][k]);
            }
            do
            {
                fscanf(fpt,"%c",&c);
                /*printf("\n got here \n");*/
            }
            while (c!='\n');
        }
        printf("\n");
    }
    sigma[0] = 1.0;
    sigma[1] = 1.0;
    sigma[2] = 1.0;
    sigma[3] = 1.0;
    sigma[4] = 1.0;
    sigma[5] = 2.0;
    sigma[6] = 2.0;
    sigma[7] = 2.0;
    sigma[8] = 2.0;
    sigma[9] = 2.0;
    lambda[0] = 1.0/4.0;
    lambda[1] = 1.0/20.0;
    lambda[2] = 5.0;
    lambda[3] = 1.0;
    lambda[4] = 5.0;
    lambda[5] = 1.0;
    lambda[6] = 50.0;
    lambda[7] = 10.0;
    lambda[8] = 1.0/8.0;
    lambda[9] = 1.0/40.0;
    global_bias = 360.0;
    return;
}
#endif

# if defined f24 || defined f25
void initialize()
{
    int i, j, k;
    FILE *fpt;
    char c;
    fpt = fopen("input_data/hybrid_func4_data.txt","r");
    if (fpt==NULL)
    {
        fprintf(stderr,"\n Error: Cannot open input file for reading \n");
        exit(0);
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            fscanf(fpt,"%Lf",&o[i][j]);
            printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
        }
        do
        {
            fscanf(fpt,"%c",&c);
        }
        while (c!='\n');
        printf("\n");
    }
    fclose(fpt);
    if (nreal==2)    fpt = fopen("input_data/hybrid_func4_M_D2.txt","r");
    if (nreal==10)    fpt = fopen("input_data/hybrid_func4_M_D10.txt","r");
    if (nreal==30)    fpt = fopen("input_data/hybrid_func4_M_D30.txt","r");
    if (nreal==50)    fpt = fopen("input_data/hybrid_func4_M_D50.txt","r");
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            for (k=0; k<nreal; k++)
            {
                fscanf(fpt,"%Lf",&l[i][j][k]);
                printf("\n M[%d][%d][%d] = %LE",i+1,j+1,k+1,l[i][j][k]);
            }
            do
            {
                fscanf(fpt,"%c",&c);
            }
            while (c!='\n');
        }
        printf("\n");
    }
    for (i=0; i<nfunc; i++)
    {
        sigma[i] = 2.0;
    }
    lambda[0] = 10.0;
    lambda[1] = 1.0/4.0;
    lambda[2] = 1.0;
    lambda[3] = 5.0/32.0;
    lambda[4] = 1.0;
    lambda[5] = 1.0/20.0;
    lambda[6] = 1.0/10.0;
    lambda[7] = 1.0;
    lambda[8] = 1.0/20.0;
    lambda[9] = 1.0/20.0;
    global_bias = 260.0;
    return;
}
#endif
