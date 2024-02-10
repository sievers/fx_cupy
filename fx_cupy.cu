//nvcc -o libfx_cupy.so fx_cupy.cu -shared -Xcompiler -fPIC -lgomp

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cufft.h>
#include <omp.h>
#include <cuComplex.h>


#define KMAX 4

extern "C"
{
void h2d(void *hptr, void *dptr, int nbyte)
{
  if (cudaMemcpy(dptr,hptr,(size_t)nbyte,cudaMemcpyHostToDevice)!=cudaSuccess)
    fprintf(stderr,"Failed copy on h2d with %d bytes.\n",nbyte);
}
}

/*--------------------------------------------------------------------------------*/
template<typename T>
__global__
void conv_cols_gpu(T *in, float *out, float *win, int n, int m, int kk)
//__global__
//void conv_cols_gpu(int8_t *in, float *out, float *win, int n, int m, int kk)

{
  float mywin[KMAX];
  float mydat[KMAX];
  int di=blockDim.x*gridDim.x;
  for (int i=blockDim.x*blockIdx.x+threadIdx.x;i<n;i+=di)
    {
      //copy window
      for (int j=0;j<kk;j++)
	mywin[j]=win[i+j*n];
      //copy first bit of data
      for (int j=0;j<kk-1;j++)
	mydat[j]=in[i+j*n];

      //loop over rows
      for (int j=kk-1;j<m;j++) {
	mydat[kk-1]=in[i+(j)*n];
	float tmp=0;
	for (int k=0;k<kk;k++) 
	  tmp+=mywin[k]*mydat[k];
	//out[i+(j+1-kk)*n]=mydat[0];
	out[i+(j+1-kk)*n]=tmp;
	//out[i+(j+1-kk)*n]=mywin[1];
	for (int k=0;k<kk-1;k++)
	  mydat[k]=mydat[k+1];

	
      }
      
    }
}
/*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*/
template<typename T>
__global__
void conv_cols_complex_gpu(T *in, cuFloatComplex *out, float *win, int n, int m, int kk)
{
  cuFloatComplex mywin[KMAX];
  cuFloatComplex mydat[KMAX];
  int di=blockDim.x*gridDim.x;
  for (int i=blockDim.x*blockIdx.x+threadIdx.x;i<n;i+=di)
    {
      //copy window
      for (int j=0;j<kk;j++)
	mywin[j]=make_cuFloatComplex(win[i+j*n],0);
      //copy first bit of data
      for (int j=0;j<kk-1;j++)  {
	T tmp;
	tmp=in[i+j*n];
	//mydat[j]=toComplex(in[i+j*n]);
	mydat[j]=make_cuFloatComplex(tmp.x,tmp.y);
      }

      //loop over rows
      for (int j=kk-1;j<m;j++) {
	{
	  T tmp;
	  tmp=in[i+(j)*n];
	  mydat[kk-1]=make_cuFloatComplex(tmp.x,tmp.y);
	  //mydat[kk-1]=in[i+(j)*n];
	}
	cuFloatComplex tmp=make_cuFloatComplex(0,0);
	for (int k=0;k<kk;k++) 
	  tmp=cuCaddf(tmp,cuCmulf(mywin[k],mydat[k]));
	out[i+(j+1-kk)*n]=tmp;
	for (int k=0;k<kk-1;k++)
	  mydat[k]=mydat[k+1];
	
      }
      
    }
}
/*--------------------------------------------------------------------------------*/
extern "C"
{
void conv_cols(void *in, float *out, float *win, int n, int m, int kk, int elemsize)
{
  //printf("elemsize is %d\n",elemsize);
  switch(elemsize) {
  case 1:
    conv_cols_gpu<<<128,256>>>((int8_t *)in,out,win,n,m,kk);
    break;
  case 2:
    conv_cols_gpu<<<256,256>>>((int16_t *)in,out,win,n,m,kk);
    break;
  case 4:
    conv_cols_gpu<<<256,256>>>((int *)in,out,win,n,m,kk);
    break;
  case -4:
    conv_cols_gpu<<<256,256>>>((float *)in,out,win,n,m,kk);
    break;
  default:
    fprintf(stderr,"Unhandled element size %d in conv_cols_gpu.\n",elemsize);
    break;
  }
}
}
/*--------------------------------------------------------------------------------*/
extern "C"
{
void conv_cols_complex(void *in, cuFloatComplex *out, float *win, int n, int m, int kk, int elemsize)
{
  //printf("elemsize is %d\n",elemsize);
  switch(elemsize) {
  case -8:
    conv_cols_complex_gpu<<<128,256>>>((cuFloatComplex *)in,out,win,n,m,kk);
    break;
  case -16:
    conv_cols_complex_gpu<<<256,256>>>((cuDoubleComplex *)in,out,win,n,m,kk);
    break;
  default:
    fprintf(stderr,"Unhandled element size %d in conv_cols_complex_gpu.\n",elemsize);
    break;
  }
}
}
/*--------------------------------------------------------------------------------*/

#define BS_CHERK 8

__global__
void cherk_batched_gpu(cuFloatComplex *in, cuFloatComplex *out, int nmat, int n, int k)
{
  int nblock=k/BS_CHERK;
  if ((nblock*BS_CHERK)<n)
    nblock++;
  for (int imat=blockIdx.x;imat<nmat;imat+=gridDim.x)
    {
      __shared__ cuFloatComplex patch[BS_CHERK][BS_CHERK];

      cuFloatComplex tmp=make_cuFloatComplex(0,0);
      for (int i=0;i<nblock;i++) {
	int myind=i*BS_CHERK+threadIdx.x;
	if (myind<k)
	  patch[threadIdx.y][threadIdx.x]=in[threadIdx.y*k+myind+imat*n*k];
	else
	  patch[threadIdx.y][threadIdx.x]=make_cuFloatComplex(0,0);
	__syncthreads();
	for (int j=0;j<BS_CHERK;j++)
	  tmp=cuCaddf(tmp,cuCmulf(patch[threadIdx.x][j],cuConjf(patch[threadIdx.y][j])));
	__syncthreads();
	//tmp=patch[0][0];
	//__syncthreads();
      }
      out[imat*n*n+threadIdx.x*BS_CHERK+threadIdx.y]=tmp;
      //out[imat*n*n+threadIdx.x*BS_CHERK+threadIdx.y]=make_cuFloatComplex(1,0);
      __syncthreads();
      //out[0]=make_cuFloatComplex(1,0);
      
    }
}
/*--------------------------------------------------------------------------------*/
__global__
void cherk_batched_gpu_transpose(cuFloatComplex *in, cuFloatComplex *out, int nmat, int n, int k)
///do a batched cherk, but expect data ordering as per PFB, so (n,k,nmat)
{
  int nblock=k/BS_CHERK;
  if ((nblock*BS_CHERK)<n)
    nblock++;
  for (int imat=blockIdx.x;imat<nmat;imat+=gridDim.x)
    {
      __shared__ cuFloatComplex patch[BS_CHERK][BS_CHERK];

      cuFloatComplex tmp=make_cuFloatComplex(0,0);
      for (int i=0;i<nblock;i++) {
	
	int myind=i*BS_CHERK+threadIdx.x;
	if (myind<k)
	  patch[threadIdx.y][threadIdx.x]=in[threadIdx.y*k+myind+imat*n*k];
	else
	  patch[threadIdx.y][threadIdx.x]=make_cuFloatComplex(0,0);

	__syncthreads();
	for (int j=0;j<BS_CHERK;j++)
	  tmp=cuCaddf(tmp,cuCmulf(patch[threadIdx.x][j],cuConjf(patch[threadIdx.y][j])));
	__syncthreads();
	//tmp=patch[0][0];
	//__syncthreads();
      }
      out[imat*n*n+threadIdx.x*BS_CHERK+threadIdx.y]=tmp;
      //out[imat*n*n+threadIdx.x*BS_CHERK+threadIdx.y]=make_cuFloatComplex(1,0);
      __syncthreads();
      //out[0]=make_cuFloatComplex(1,0);
      
    }
}
/*--------------------------------------------------------------------------------*/

extern "C"
{
void cherk_batched(cuFloatComplex *in, cuFloatComplex *out, int nmat, int n, int k)
{
  //cherk_batched_gpu<<<512,(BS_CHERK,BS_CHERK)>>>(in,out,nmat,n,k);
  dim3 threadsize;
  threadsize.x=BS_CHERK;
  threadsize.y=BS_CHERK;
  threadsize.z=1;
  cherk_batched_gpu<<<256,threadsize>>>(in,out,nmat,n,k);
}
}
/*--------------------------------------------------------------------------------*/
__global__
void apply_pfb_win(int8_t *din, float *dout, float *win, int ntap, int nchan, int npol, int nblock, int nn)
{
  
}

