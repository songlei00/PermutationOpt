#include <string>
#include <sstream>
#include <map>
#include <sys/stat.h>
#include <algorithm>
#include <vector>
#include<iostream>
#include<functional>
#include<math.h>
#include<stdio.h>
#include<cstring>
#include<time.h>

using namespace std;
using std::string;
using std::vector;

struct I_M
{
	int I;
	int M;
};


void MARKPOSITION(int XPERM[], int size);

bool COEX2(int sp1, int sp2);

inline string& lTrim(string &ss)
{
	string::iterator   p=::find_if(ss.begin(),ss.end(),::not1(::ptr_fun(::isspace)));
	ss.erase(ss.begin(),p);
	return  ss;
}

inline string& rTrim(string &ss)
{
	string::reverse_iterator  p=::find_if(ss.rbegin(),ss.rend(),::not1(::ptr_fun(::isspace)));
	ss.erase(p.base(),ss.end());
	return   ss;
}

inline string& trim(string &st)
{
	lTrim(rTrim(st));
	return   st;
}

vector<string> &split(const string &s, char delim, vector<string> &elems);

vector<string> split(const string &s, char delim);

inline bool fileExists(const std::string& filename)
{
	struct stat buf;
	if (stat(filename.c_str(), &buf) != -1)
	{
		return true;
	}
	return false;
}

	template <typename T>
inline void SetVal(T* p,int x, T val)
{
	for(int i=0;i<x;i++)
		p[i]=val;

}

	template <typename T>
inline void SetVal(T** p,int x,int y, T val)
{
	for(int i=0;i<x;i++)
		for(int j=0;j<y;j++)p[i][j]=val;

}

	template <typename T>
inline void SetVal(T*** p,int x,int y,int z, T val)
{
	for(int i=0;i<x;i++)
		for(int j=0;j<y;j++)
			for (int k = 0; k<z; k++)p[i][j][k] = val;
}

	template <typename T>
inline void CopyArray(T* src,int x, T* dest)
{
	memcpy(dest,src,sizeof(T)*x);	
}

	template <typename T>
inline void CopyArray(T** src,int x,int y, T** dest)
{
	T size = sizeof(T);
	for(int i=0;i<x;i++)
		memcpy(dest[i], src[i],size*y);

}

	template <typename T>
inline T MinVal(T** p,int x,int y)
{
	T retval=p[0][0];

	for(int j=0;j<y;j++)
	{
		if(p[0][j]<retval)
			retval=p[0][j];
	}	

	return retval;
}
	template <typename T>
inline int ONEMinval(T *p,int x)
{
	T retval = p[0];
	int min = 0;
	for (int i = 1; i < x; i++)
	{		
		if (p[i] < retval) {
			retval = p[i];
			min = i;
		}
	}

	return min;
}

	template <typename T>
inline float BetterPercent(T *p, int x, double lstpen)
{

	float betterCount = 0;
	for (int i = 0; i < x; i++)
	{
		if (p[i] < lstpen) {
			betterCount++;
		}
	}

	return betterCount/x;
}

	template <typename T>
inline T MAXVal(T* p,int x)
{
	T retval=p[0];
	for(int i=0;i<x;i++)
		if(p[i]>retval)retval=p[i];
	return retval;
}

	template <typename T>
inline T MAXVal(T** p,int x,int y)
{
	T retval=p[0][0];
	for(int i=0;i<x;i++)
		for(int j=0;j<y;j++)
		{
			if(p[i][j]>retval)retval=p[i][j];
		}

	return retval;
}

inline double random(double start, double end)
{
	return start+(end-start)*rand()/(RAND_MAX + 1.0);
}

void EVENTSUM(int flg, double fac);
void REORDER(int perm[], int before[], int a, int order[], int obj[], int b);
bool IORZERO(int evtrow, int sct);

bool RZERO(int evtrow, int sct);

bool ALLRZERO(int evtrow);

void FINDROW(int EVT, int TYP, int& ROW);

void ASORT(double X[], int L[], int N);

void ASORT_DESC(double X[], int L[], int N);

void IASORT(int X[], int L[], int N);

double EPEN0(int EVENTROW, int LEVEL, int JOPT);

double NEGPEN(int EVENTROW, int LEVEL, int JOPT);

void SCTPEN(int HPERM[], int JOPT, double& PENJ);

void JSPAN(int XPERM[]);

void SMOOTH(int HPERM[]);

void SQUEEZE(int HPERM[]);

void SHRINK(int HPERM[]);

bool COEX(int sp1, int sp2, int XPERM[]);

void ROYAL(int XPERM[], double& XPEN);

bool FB4L(int sp1, int sp2, int XPERM[]);

void SEQUEL(int XPERM[], double& XPEN);

void TEASE(int HPERM[]);

void TWOPEN(int HPERM[], double& HPEN);

void GETPEN(int HPERM[], double& HPEN);

void GETPEN_WITH_TWOPEN(int HPERM[], double& HPEN);

void DEMPEN(int HPERM[], double& HPEN);

void DOTOO();

void STEPOUT(int myperm[]);

void TRAJOUT();

void NABRLIMT(int IRANK, int& RANKUP, int& RANKDN);

void SMLNABR(int& IRANK);

void BIGNABR(int& IRANK, int& JRANK);

void SMLNABR2(int& IRANK, int count, int last, int &first, int C);

void BIGNABR2(int& IRANK, int& JRANK, int count, int last, int& first, int C);

void TWONABR(int& IRANK, int& JRANK);

void GONABR(int& IRANK, int& JRANK, int count, int last, int& first, int C);

void RESCTPEN(int RNK, int RNK2, int HPERM[], int JOPT, double& PENJ);

void NEWPEN(int IRNK, int JRNK, int HPERM[], double& HPEN);

void MAXSEQ(int MAXTYPE, int K, int IRNK, int JRNK, int XPERM[], double& XPEN);

void MINSEQ(int MINTYPE, int K, int IRNK, int JRNK, int XPERM[], double& XPEN);

void NEWSEQUEL(int IRNK, int JRNK, int XPERM[], double& XPEN);

void MAXROY(int MAXTYPE, int K, int IRNK, int JRNK, int XPERM[], double& XPEN);

void MINROY(int MINTYPE, int K, int IRNK, int JRNK, int XPERM[], double& XPEN);

void NEWROYAL(int IRNK, int JRNK, int XPERM[], double& XPEN);

void NEWSPAN(int IRNK, int JRNK, int XPERM[],double& XPEN);

void NEW2PEN(int IRNK, int JRNK, int HPERM[], double& HPEN, int BACK);

void NWDMPEN(int IRNK, int JRNK, int HPERM[], double& HPEN);

void GETEVNT(int I, string& xtitle, int flg1, int flg2, int flg3, int flg4, int flg5, int flg6);

double PLACE(int evt, int sec);

void UNBASE();

void CalcEPEN();

