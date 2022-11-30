#include <iostream>
#include <iomanip> 
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <math.h>
#include "COMMOD.h"
#include "Helper.h"
#include "PREPARE.h"
#include "DOINPUT.h"
#include "DOCOEX.h"
#include "DOFB4L.h"
#include "DOPTLORD.h"
#include "DORASC.h"
#include "GETSTART.h"
#include "ANNEAL.h"
#include "Output.h"


#ifdef NEAR

#undef NEAR

#endif
using namespace std;

extern "C"
{
   double getPenalty(int* solnarr, int solnlen);
   int unload();
}


bool *allrzero;
float *epen;
int epenCount;
int EVT_X_LEVEL;
int *IDX_SCT_EPEN;
int *position;

int ** IDX_EVT_SCT_EPEN;

//New Flag for DLL
COMMOD* COMMOD9 = COMMOD::Singleton();
bool isInitialized=false;
bool Unloaded=false;

//unsigned short  *I_M_Arr;
//int i_m_count;
//int *IROWS_3;



//int main(int  argc,char  **argv)
double getPenalty(int* solnarr, int solnlen)
{

	
	int namelen;
	int myid=0, numprocs;	
	int C=1;


	int NUMBER1 = 0, NUMBER2 = 0, NUMBER3 = 0, NUMBER4 = 0;
	int I = 0, J = 0, M = 0, L = 0;
	int ttlco = 0, ranco = 0, maxco = 0;
	float rtlco = 0.0f;
	double topj = 0.0, btmj = 0.0, minc = 0.0;
	string xtitle="";
	int rngj = 0;

        if(Unloaded==true)throw "Error: data has been unloaded.";        
        if(isInitialized==true)goto LABEL_GETPEN; 	
 

	//clock_t start_t,finish_t, finish_t2,finish_t3,finish_t4,finish_t5,finish_t6;
	//double totalTime = 0.0,totalTime2=0.0,totalTime3=0.0,totalTime4=0.0,totalTime5=0.0,totalTime6=0.0,totalTime7=0.0;
	//start_t = clock();

	//COMMOD* COMMOD9 = COMMOD::Singleton();

	COMMOD9->STOPF = 0;
	COMMOD9->NTRIES = 0;
	COMMOD9->MXLVL = 0;
	COMMOD9->MXLBL = 0;
	COMMOD9->WTF = 0;
	COMMOD9->PENF = 0;
	COMMOD9->PEN2F = 0;
	COMMOD9->PENM = 0;
	COMMOD9->PEN2M = 0;
	COMMOD9->NFALA = 0;
	COMMOD9->MXFALA = 0;
	COMMOD9->NCOEX = 0;
	COMMOD9->MXCOEX = 0;




	PREPARE::Singleton()->RunIt(myid);	

	//finish_t2 = clock();
	//totalTime2 = (double)(finish_t2 - start_t) / CLOCKS_PER_SEC;	



	if (COMMOD9->STOPF == 1) goto Label998;
	DOINPUT::Singleton()->RunIt(myid);


	//finish_t3 = clock();
	//totalTime3 = (double)(finish_t3 - finish_t2) / CLOCKS_PER_SEC;	

	allrzero = new bool[COMMOD9->NEVNT];


	{	
		EVT_X_LEVEL = COMMOD9->NEVNT*COMMOD9->MXLVL;		
		position = new int[COMMOD9->NEVNT];		
	}


	COMMOD9->SOL_SIZE = sizeof(int)*COMMOD9->NEVNT;

	for (I = 0; I < COMMOD9->NEVNT; I++)
	{
		allrzero[I] = ::ALLRZERO(I);
	}

	COMMOD9->EXCLSECT = new int[COMMOD9->JEXCL];
	I = -1;
	for (J = 0; J < COMMOD9->NSCT; J++)
	{
		if (COMMOD9->SECTPROP[J][1] > 0)
		{
			I = I + 1;
			COMMOD9->EXCLSECT[I] = J* COMMOD9->NEVNT;
		}
	}
	delete[] COMMOD9->SECTPROP;
	if (COMMOD9->STOPF == 1) goto Label998;

	//**************************************************************
	//   fill up the COXST PTLORD FADLAD arrays to make constraints 
	//   suitable for generating the initial permutation
	if (!COMMOD9->LOADCOEX)
	{

		if(0==myid)
			cout << "  Building Coexistence Matrix . . . " << endl;

		DOCOEX::Singleton()->RunIt(COMMOD9->COXSTF);

		//finish_t4 = clock();
		//totalTime4 = (double)(finish_t4 - finish_t3) / CLOCKS_PER_SEC;
		if (myid == 0) {				
			Output::COEXOUT();
			//clock_t f = clock();				
		}			
		//MPI_Barrier(MPI_COMM_WORLD);

		if (COMMOD9->STOPF == 1) goto Label998;

	}
	else if (COMMOD9->LOADCOEX && (COMMOD9->NSPC > 0))
	{

		::SetVal(COMMOD9->COEXST, COMMOD9->NSPC*COMMOD9->NSPC, (BYTE)0);
		for (I = 0; I < COMMOD9->NSPC; I++)
		{
			COMMOD9->COEXST[I*COMMOD9->NSPC+I] = 3;
		}
		if (::fileExists(COMMOD9->COEXFILE))
		{
			::fstream _file;				
			_file.open(COMMOD9->COEXFILE.c_str(), ios::in);
			string line;				
			while (std::getline(_file, line))
			{
				vector<string> elems = ::split(line, ' ');
				I= atoi(elems[0].c_str());
				J = atoi(elems[1].c_str());
				M= atoi(elems[2].c_str());
				COMMOD9->COEXST[I*COMMOD9->NSPC+J] = (BYTE)M;
				COMMOD9->COEXST[J*COMMOD9->NSPC+I] = (BYTE)M;
			}
			_file.close();
		}

		if (myid == 0) {
			//clock_t f = clock();				
		}

	}


	COMMOD9->NCOEX = 0;

	if (COMMOD9->NSPC > 0)
	{
		for (I = 0; I < COMMOD9->NSPC - 1; I++)
		{
			for (M = I + 1; M < COMMOD9->NSPC; M++)
			{


				if (COMMOD9->COEXST[I*COMMOD9->NSPC+M] >= COMMOD9->COXSTF)
					COMMOD9->NCOEX = COMMOD9->NCOEX + 1;
			}
		}
	}


	switch (COMMOD9->WTF)
	{
		case 0:			
			memset(COMMOD9->NRSTATIC, 1.0f, sizeof(float)*COMMOD9->RSTATICCount);

			if(0==myid)
				cout << "  Setting all weights to 1.00 . . ." << endl;
			break;
		case 1:			
			break;
		case 2:
		case 22:
		case 23:
		case 3:			
			if(0==myid)
				cout << "  Calculating weights from coexistences . . ." << endl;

			int IDX;
			for (I = 0; I < COMMOD9->NEVNT; I++)
			{				
				if ((COMMOD9->IROWS[I][1] == 1) || (COMMOD9->IROWS[I][1] == 2))
				{					
					ttlco = 0;

					for (J = 0; J < COMMOD9->NSPC; J++)
					{

						if (COMMOD9->IROWS[I][2] == J) continue;

						if (COMMOD9->COEXST[(COMMOD9->IROWS[I][2])*COMMOD9->NSPC+J] >= (int)COMMOD9->COXSTF)
						{
							ttlco = ttlco + 1;
						}
					}

					COMMOD9->RANGE[COMMOD9->IROWS[I][2]] = (int)ttlco;


					if (COMMOD9->WTF != 3)
					{						
						rtlco = max((float)ttlco, 1.0f);
						if (COMMOD9->WTF == 22) rtlco = pow(rtlco, 0.5f);
						if (COMMOD9->WTF == 23) rtlco = pow(rtlco, 0.33f);
						cout << "event - weight: " << I << " - " << rtlco << endl;

						for (J = 0; J < COMMOD9->NSCT; J++)
						{			

							IDX = COMMOD9->IDX_SCT_RSTATIC[J] + I * 2;
							COMMOD9->NRSTATIC[IDX] = 1.0f / rtlco;
							IDX = COMMOD9->IDX_SCT_RSTATIC[J] + I * 2 + 1;
							COMMOD9->NRSTATIC[IDX] = 1.0f / rtlco;
						}
					}
				}
			}

			if (COMMOD9->WTF == 3)
			{
				cout << "   % of taxa with reduced weights: " << COMMOD9->cutco << endl;
			}

			COMMOD9->cutco = COMMOD9->cutco *COMMOD9->NSPC / 100;
			if ((COMMOD9->WTF == 3) && (COMMOD9->cutco > 0))
			{				
				ranco = 1;
				maxco = ::MAXVal(COMMOD9->RANGE, COMMOD9->NSCT);

				for (; ranco <= COMMOD9->cutco; )
				{
					for (I = 0; I < COMMOD9->NEVNT; I++)
					{
						if ((COMMOD9->IROWS[I][1] != 1) && (COMMOD9->IROWS[I][1] != 2)) continue;

						if (COMMOD9->RANGE[COMMOD9->IROWS[I][2]] == (int)maxco)
						{						
							if (COMMOD9->LOADNAMES) {
								if ((COMMOD9->CDF != 1) && (COMMOD9->IROWS[I][1] == 1))
								{
									::GETEVNT(I, xtitle, 0, 1, 0, 1, 0, 0);									
								}
							}

							for (J = 0; J < COMMOD9->NSCT; J++)
							{							
								IDX = COMMOD9->IDX_SCT_RSTATIC[J] + I * 2;
								COMMOD9->NRSTATIC[IDX] = COMMOD9->NRSTATIC[IDX] / 10.0f;
								IDX = COMMOD9->IDX_SCT_RSTATIC[J] + I * 2 + 1;
								COMMOD9->NRSTATIC[IDX] = COMMOD9->NRSTATIC[IDX] / 10.0f;
							}

							ranco = ranco + 1;
							COMMOD9->RANGE[COMMOD9->IROWS[I][2]] = 0;
						}
					}

					::MAXVal(COMMOD9->RANGE, COMMOD9->NSCT);
				}

			}

			break;
	}


	if (COMMOD9->NOTHR > 0)
	{
		cout << "  Building the Partial Ordering Matrix . . ." << endl;
		DOPTLORD::Singleton()->RunIt();			
		if (COMMOD9->STOPF == 1) goto Label998;

	}


	if (myid == 0) {
		if (COMMOD9->PENF == 4)
		{
			cout << endl;
			cout << "  Building RASC Matrix . . . " << endl;

			DORASC::Singleton()->RunIt();				
		}
	}	

	//finish_t5 = clock();

	if (!COMMOD9->LOADCEFb4L && (COMMOD9->NSPC > 0))
	{			
		if(0==myid)
			cout << "  Building FAD\\LAD Matrix . . . " << endl;

		DOFB4L::Singleton()->RunIt();

		//totalTime5 = (double)(finish_t5 - finish_t4) / CLOCKS_PER_SEC;


		if (myid == 0) {				
			Output::FADLADOUT();
			//clock_t f = clock();				
		}
		//MPI_Barrier(MPI_COMM_WORLD);

		if (COMMOD9->STOPF == 1) goto Label998;
	}
	else if (COMMOD9->LOADCEFb4L && (COMMOD9->NSPC > 0))
	{

		if(0==myid)
			cout << "  Loading the FAD\\LAD Matrix from file. . . " << endl;

		::SetVal(COMMOD9->FADLAD, COMMOD9->NSPC*COMMOD9->NSPC, (BYTE)0);
		for (I = 0; I < COMMOD9->NSPC; I++)
		{
			COMMOD9->FADLAD[I*COMMOD9->NSPC+I] = 1;
		}
		if (::fileExists(COMMOD9->FB4LFILE))
		{
			::fstream _file;				
			_file.open(COMMOD9->FB4LFILE.c_str(), ios::in);
			string line;
			while (std::getline(_file, line))
			{
				vector<string> elems = ::split(line, ' ');
				I = atoi(elems[0].c_str());
				J = atoi(elems[1].c_str());
				COMMOD9->FADLAD[I*COMMOD9->NSPC+J] = 1;
			}
			_file.close();
		}
		if (myid == 0) {
			//clock_t f = clock();				

		}			

	}

	if (COMMOD9->FB4LF != 0)
	{
		COMMOD9->NFALA = 0;

		for (I = 0; I < COMMOD9->NSPC; I++)
		{
			for (M = 0; M < COMMOD9->NSPC; M++)
			{
				if (I == M) continue;
				if (COMMOD9->FADLAD[I*COMMOD9->NSPC+M] == 1)
					COMMOD9->NFALA = COMMOD9->NFALA + 1;
			}
		}

	}


	while ((COMMOD9->ESTART == -1) && (::ALLRZERO(COMMOD9->XEVNT)))
	{
		COMMOD9->XEVNT = rand() % (COMMOD9->NEVNT);
	}


	{
		string orderFileName = "EXCLUDE.DAT";
		int orderCount = 0;
		int* orderArr = new int[COMMOD9->NSPC];
		if (::fileExists(orderFileName))
		{

			::fstream _file;			
			_file.open(orderFileName.c_str(), ios::in);
			string line;
			while (std::getline(_file, line))
			{
				orderArr[orderCount] = atoi(line.c_str()) - 1;
				orderCount++;
			}
			_file.close();

			for (I = 0; I < orderCount; I++)
			{				
				for (M = 0; M < COMMOD9->NSPC; M++)
				{
					if (orderArr[I] == M) continue;
					COMMOD9->FADLAD[orderArr[I] * COMMOD9->NSPC + M] = 0;
					COMMOD9->FADLAD[M* COMMOD9->NSPC + orderArr[I]] = 0;
					COMMOD9->COEXST[orderArr[I] * COMMOD9->NSPC + M] = 0;
					COMMOD9->COEXST[M* COMMOD9->NSPC + orderArr[I]] = 0;

				}
			}

			//count the number of coexistences proven by observation - minus the exclude
			COMMOD9->NCOEX_EXCLUDE = 0;

			if (COMMOD9->NSPC > 0)
			{
				for (I = 0; I < COMMOD9->NSPC - 1; I++)
				{
					for (M = I + 1; M < COMMOD9->NSPC; M++)
					{


						if (COMMOD9->COEXST[I*COMMOD9->NSPC + M] >= COMMOD9->COXSTF)
							COMMOD9->NCOEX_EXCLUDE = COMMOD9->NCOEX_EXCLUDE + 1;
					}
				}
			}
			
		}
		else
		{
			COMMOD9->NCOEX_EXCLUDE = COMMOD9->NCOEX;
		}

		delete[] orderArr;
	}

	epenCount = 0;
	{
		int h_Level = 0;

		IDX_EVT_SCT_EPEN = new int*[COMMOD9->NSCT];
		for (int i = 0; i < COMMOD9->NSCT; i++)
		{
			IDX_EVT_SCT_EPEN[i] = new int[COMMOD9->NEVNT];
			h_Level = COMMOD9->HLEVEL[i] + 1;

			for (int j = 0; j < COMMOD9->NEVNT; j++)
			{
				if (COMMOD9->ISTATIC_0[i*COMMOD9->NEVNT + j] == -1)
					IDX_EVT_SCT_EPEN[i][j] = -1;
				else
				{
					IDX_EVT_SCT_EPEN[i][j] = epenCount;
					epenCount += h_Level;
				}
			}
		}

		epen = new float[epenCount];			
		memset(epen, 0, sizeof(float)*epenCount);

	}		

	CalcEPEN();

	/*i_m_count = 0;

	{
		int irowi_x_nspc;

		for (I = 0; I < COMMOD9->NEVNT - 1; I++)
		{

			if (COMMOD9->IROWS[I][1] != 1) continue;

			irowi_x_nspc = COMMOD9->IROWS[I][2] * COMMOD9->NSPC;

			if (I == 0)i_m_count++; 
			else i_m_count += 2;   

			for (M = I + 1; M < COMMOD9->NEVNT; M++)
			{
				if (COMMOD9->IROWS[M][1] != 1) continue;

				if (COMMOD9->COEXST[irowi_x_nspc + COMMOD9->IROWS[M][2]] >= COMMOD9->COXSTF) continue;

				i_m_count++;
			}
		}

		I_M_Arr = new unsigned short[i_m_count];

		int num = -1;;

		for (I = 0; I < COMMOD9->NEVNT - 1; I++)
		{				
			if (COMMOD9->IROWS[I][1] != 1) continue;

			irowi_x_nspc = COMMOD9->IROWS[I][2] * COMMOD9->NSPC;

			if (I == 0)
			{
				num++;
				I_M_Arr[num] = I;
			}
			else
			{
				num++;
				I_M_Arr[num] = 0;
				num++;
				I_M_Arr[num] = I;
			}

			for (M = I + 1; M < COMMOD9->NEVNT; M++)
			{
				if (COMMOD9->IROWS[M][1] != 1) continue;

				if (COMMOD9->COEXST[irowi_x_nspc + COMMOD9->IROWS[M][2]] >= COMMOD9->COXSTF) continue;

				num++;
				I_M_Arr[num] = M;
			}
		}

	}

	{
		IROWS_3 = new int[COMMOD9->IRCOUNT];
		for (int i=0;i<COMMOD9->IRCOUNT;i++)
		{
			IROWS_3[i] = COMMOD9->IROWS[i][3];
		}
	}


	COMMOD9->IROWS_3 = new int[COMMOD9->NEVNT];
	for (int i = 0; i < COMMOD9->NEVNT; i++) {
		COMMOD9->IROWS_3[i] = COMMOD9->IROWS[i][3];
	}*/

	
        //GETSTART::Singleton()->RunIt(1);

       isInitialized=true;

        //::SetVal(COMMOD9->INIPERM, COMMOD9->NEVNT, 0);
LABEL_GETPEN:
        for(int ii=0;ii<solnlen;ii++)
        {
           COMMOD9->INIPERM[ii]=solnarr[ii];
		   //cout<<solnarr[ii]<<" ";
        }


	if (COMMOD9->STOPF == 1) goto Label998;

	if (COMMOD9->NTRIES > COMMOD9->NSTOP) COMMOD9->PAUSF = 0;

	if (((COMMOD9->INIGEN >= 2) && (COMMOD9->FIXF > 0) &&
				((COMMOD9->PAUSF == 2) || (COMMOD9->PAUSF == 4))) ||
			((COMMOD9->INIGEN >= 2) && (COMMOD9->PAUSF == 3)))
	{
		GETSTART::Singleton()->RunIt(0);			

	}
	else
	{ }	

	COMMOD9->SPANPEN = 0.0;
	COMMOD9->ASPNPEN = 0.0;

	if(0==myid)
		//cout << "Ready for annealing" << endl;
	ANNEAL::Singleton()->RunIt(numprocs, myid);	
	//finish_t6 = clock();
	//totalTime6 = (double)(finish_t6 - finish_t5) / CLOCKS_PER_SEC;

	COMMOD9->BSTPLUS = COMMOD9->BSTPEN + (COMMOD9->BSTPEN * COMMOD9->NEAR / (double)COMMOD9->NOBS);

	if (COMMOD9->PAUSF == 5) goto Label22;

	COMMOD9->LCLBST = COMMOD9->BSTPEN;	


	for (J = 0; J < COMMOD9->NSCT; J++)
	{

		topj = ::PLACE(COMMOD9->NEVNT - 1, J);

		btmj = ::PLACE(1 - 1, J);	

	

		COMMOD9->VALEVEL[J][COMMOD9->MXLVL] = topj - btmj;

		if (topj > btmj)
		{				
			I = COMMOD9->NEVNT - 1;

			while (COMMOD9->VALEVEL[J][COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + COMMOD9->BSTPERM[I]]] == topj)
			{

				I = I - 1;
			}

			rngj = I;
			COMMOD9->LIMITS[J][1] = COMMOD9->BSTPERM[I];

			L = COMMOD9->NEVNT - 1;
			while (L >= 0)
			{					
				if ((COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+COMMOD9->BSTPERM[L]] >= 0) &&
						(COMMOD9->IROWS[COMMOD9->BSTPERM[L]][1] == 1))
				{
					break;
				}

				L = L - 1;
			}				

			if (L > I) COMMOD9->LIMITS[J][1] = COMMOD9->BSTPERM[L];

			I = 1 - 1;
			while (COMMOD9->VALEVEL[J][COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + COMMOD9->BSTPERM[I]]] == btmj)
			{					
				I = I + 1;
			} 

			rngj = rngj - I;
			COMMOD9->LIMITS[J][0] = COMMOD9->BSTPERM[I];

			L = 0;
			while (L < COMMOD9->NEVNT)
			{					
				if ((COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+COMMOD9->BSTPERM[L]] >= 0) &&
						(COMMOD9->IROWS[COMMOD9->BSTPERM[L]][1] == 2))
				{
					break;
				}

				L = L + 1;
			}


			if (L < I) COMMOD9->LIMITS[J][0] = COMMOD9->BSTPERM[L];

		}

		if ((topj - btmj) == 0.0)
		{
			COMMOD9->VALEVEL[J][COMMOD9->MXLVL + 1] = 0.0;
		}
		else
		{
			COMMOD9->VALEVEL[J][COMMOD9->MXLVL + 1] = (float)(rngj + 2);
		}

	}



	::SetVal(COMMOD9->COVER, COMMOD9->NEVNT, 0);

	for (J = 0; J < COMMOD9->NSCT; J++)
	{			
		for (L = COMMOD9->LIMITS[J][0]; L < COMMOD9->LIMITS[J][1]; L++)
		{
			COMMOD9->COVER[L] = COMMOD9->COVER[L] + 1;
		}
	}


	::EVENTSUM(4, minc);

	::UNBASE();

	::JSPAN(COMMOD9->BSTPERM);


	if (COMMOD9->JSPANF > 0)
	{
		switch (COMMOD9->PENF)
		{
			case -1:
				COMMOD9->RVLPEN = COMMOD9->RVLPEN - COMMOD9->SPANPEN;
				break;
			case 0:
				COMMOD9->INTPEN = COMMOD9->INTPEN - COMMOD9->SPANPEN;
				break;
			case 1:
				COMMOD9->LVLPEN = COMMOD9->LVLPEN - COMMOD9->SPANPEN;
				break;
			case 2:
				COMMOD9->ORDPEN = COMMOD9->ORDPEN - COMMOD9->SPANPEN;
				break;
			case 3:
				COMMOD9->SPTPEN = COMMOD9->SPTPEN - COMMOD9->SPANPEN;
				break;
			case 4:
				COMMOD9->RSCPEN = COMMOD9->RSCPEN - COMMOD9->SPANPEN;
				break;
			case 5:
				COMMOD9->ROYPEN = COMMOD9->ROYPEN - COMMOD9->SPANPEN;
				break;
			case 6:
				if (COMMOD9->FB4LF != 0) COMMOD9->SEQPEN = COMMOD9->SEQPEN - COMMOD9->SPANPEN;
				break;
			case 7:
				COMMOD9->MOMPEN = COMMOD9->MOMPEN - COMMOD9->SPANPEN;
				break;
		}
	}		

Label22:
        /*
	if (COMMOD9->PAUSF != 5)
	{
		if (COMMOD9->CDF != 1)
		{
			if (myid == 0)
				Output::STARTOUT(COMMOD9->BSTPERM);
		}
	}
	else
	{
		if (COMMOD9->CDF != 1) Output::STARTOUT(COMMOD9->LSTPERM);
		::CopyArray(COMMOD9->LSTPERM, COMMOD9->NEVNT, COMMOD9->INIPERM);
		goto Label998;
	}	
        */	

Label998:

	if (myid == 0 &&!COMMOD9->ANNEALONLY)
	{//nothing
	}
/*
	delete[] COMMOD9->SectionMaxPenOrder;		
	delete[] COMMOD9->EXCLSECT;
	delete[]COMMOD9->FADLAD;
	delete[]COMMOD9->COEXST;
	delete[]COMMOD9->ISTATIC_0;
	delete[]COMMOD9->ISTATIC_1;
	delete[]COMMOD9->NRSTATIC;
	delete[]allrzero;
	delete[]position;
	delete[]epen;
	delete[]IDX_SCT_EPEN;	
	//delete[] COMMOD9->IROWS_3;
	{ 
		for (int i = 0; i < COMMOD9->NSCT; i++)
		{
			delete IDX_EVT_SCT_EPEN[i];

		}
		delete IDX_EVT_SCT_EPEN;
	}


	finish_t = clock();
	totalTime = (double)(finish_t - start_t) / CLOCKS_PER_SEC;
	if (myid==0)
	{
		cout << "=> Best Penalty = " <<setprecision(8)<< COMMOD9->BSTPEN << endl;
		cout << "   Time elapsed = " << totalTime << endl << endl;
		//cout << "Press any key to continue..." << endl;			
	}
	
*/
	//if(0==myid)cin.get();

	return COMMOD9->BSTPEN;	

}



int unload()
{        
        delete[] COMMOD9->SectionMaxPenOrder;		
	delete[] COMMOD9->EXCLSECT;
	delete[]COMMOD9->FADLAD;
	delete[]COMMOD9->COEXST;
	delete[]COMMOD9->ISTATIC_0;
	delete[]COMMOD9->ISTATIC_1;
	delete[]COMMOD9->NRSTATIC;
	delete[]allrzero;
	delete[]position;
	delete[]epen;
	delete[]IDX_SCT_EPEN;	
	//delete[] COMMOD9->IROWS_3;
	{ 
		for (int i = 0; i < COMMOD9->NSCT; i++)
		{
			delete IDX_EVT_SCT_EPEN[i];

		}
		delete IDX_EVT_SCT_EPEN;
	}

        Unloaded=true;

    return 0;
}
