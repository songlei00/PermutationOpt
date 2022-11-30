#include <iostream>
#include <fstream>
#include <math.h>
#include "COMMOD.h"
#include "Helper.h"
#include "GETSTART.h"

using namespace std;
using std::string;

GETSTART* GETSTART::g_singleton = NULL;

void GETSTART::RunIt(int OUT) {
	COMMOD* COMMOD9 = COMMOD::Singleton();

	double UNIFRN;
	double* U = new double[COMMOD9->NEVNT];
	int* L = new int[COMMOD9->NEVNT];
	int* L1 = new int[COMMOD9->NEVNT];
	int* X = new int[COMMOD9->NEVNT];

	int* FOUNDS = new int[COMMOD9->NEVNT];

	int* MFIRSTS = new int[COMMOD9->NEVNT - COMMOD9->NSPC - COMMOD9->NMIDS];
	int* MLASTS = new int[COMMOD9->NEVNT - COMMOD9->NSPC - COMMOD9->NMIDS];
	int* MOTHERS = new int[max(1, COMMOD9->NEVNT - COMMOD9->TWOSPC)];

	int ISWAP=0, TEMP=0;
	int SPC=0, TYP=0, RNK=0, ROW=0;
	int NF=0, NMF=0, NML=0, FILLED=0, NMO=0;
	int I=0, K=0, M=0;

	UNIFRN = 0.0;

	::SetVal(U, COMMOD9->NEVNT, 0.0);
	::SetVal(L, COMMOD9->NEVNT, 0);
	::SetVal(L1, COMMOD9->NEVNT, 0);
	::SetVal(X, COMMOD9->NEVNT, 0);

	::SetVal(FOUNDS, COMMOD9->NEVNT, 0);
	::SetVal(MFIRSTS, COMMOD9->NEVNT - COMMOD9->NSPC - COMMOD9->NMIDS, 0);
	::SetVal(MLASTS, COMMOD9->NEVNT - COMMOD9->NSPC - COMMOD9->NMIDS, 0);
	::SetVal(MOTHERS, max(1, COMMOD9->NEVNT - COMMOD9->TWOSPC), 0);


	if ((COMMOD9->INIGEN == 1) || (COMMOD9->INIGEN == 3))
	{

		::SetVal(COMMOD9->INIPERM, COMMOD9->NEVNT, 0);

		FILLED = 0;

		NF = 0;		
		NMO = 0;		
		NMF = 0;		
		NML = 0;

		for (I = 0; I < COMMOD9->IRCOUNT; I++) 
		{
			if (COMMOD9->ISTATIC_0[COMMOD9->JSTART*COMMOD9->NEVNT+I] > -1)
			{
				if (COMMOD9->INIGEN == 1) {
					NF = NF + 1;
					FOUNDS[NF - 1] = I;
					X[NF - 1] = COMMOD9->ISTATIC_0[COMMOD9->JSTART*COMMOD9->NEVNT+I];					
				} else if (COMMOD9->IROWS[I][1] == 1) {
					NMF = NMF + 1;
					MFIRSTS[NMF - 1] = I;
				} else if (COMMOD9->IROWS[I][1] == 2) {
					NML = NML + 1;
					MLASTS[NML - 1] = I;
				} else if ((COMMOD9->IROWS[I][1] > 2)
						|| (COMMOD9->IROWS[I][1] < 1)) {
					NMO = NMO + 1;
					MOTHERS[NMO - 1] = I;
				}

			} else if (COMMOD9->ISTATIC_0[COMMOD9->JSTART*COMMOD9->NEVNT+I] == -1) 
			{

				if (COMMOD9->IROWS[I][1] == 1) {
					NMF = NMF + 1;
					MFIRSTS[NMF - 1] = I;
				}

				else if (COMMOD9->IROWS[I][1] == 2) {
					NML = NML + 1;
					MLASTS[NML - 1] = I;
				}

				else if ((COMMOD9->IROWS[I][1] > 2)
						|| (COMMOD9->IROWS[I][1] < 1)) {
					NMO = NMO + 1;
					MOTHERS[NMO - 1] = I;
				}
			}
		}

		if (NMF > 0) {

			for (I = 0; I < NMF; I++) {
				UNIFRN = ::random(0, 1); 
				U[I] = UNIFRN;
			}

			::ASORT(U, L1, NMF);

			for (K = 0; K < NMF; K++) {				
				COMMOD9->INIPERM[K] = MFIRSTS[L1[K]];				
			}

			FILLED = NMF;
		}

		if (NF > 0) {
			::IASORT(X, L, NF);

			for (K = 0; K < NF; K++) {
				COMMOD9->INIPERM[K + FILLED] = FOUNDS[L[K]];
			}

			FILLED = FILLED + NF;

		}


		if (NMO > 0) {			
			for (I = COMMOD9->NEVNT - (COMMOD9->NSPC * 2) - 1; I >= 0; I--) {

				for (ROW = 0; ROW < COMMOD9->NEVNT - (COMMOD9->NSPC * 2);
						ROW++) {

					if ((COMMOD9->PTLORD[ROW][COMMOD9->NEVNT
								- (COMMOD9->NSPC * 2)] == I)
							&& (COMMOD9->PTLORD[ROW][COMMOD9->NEVNT
								- (COMMOD9->NSPC * 2)]
								> COMMOD9->PTLORD[ROW][COMMOD9->NEVNT
								- (COMMOD9->NSPC * 2) + 1])) {

						FILLED = FILLED + 1;
						COMMOD9->INIPERM[FILLED - 1] = ROW;             
					}
				}
			}

			for (ROW = 0; ROW < COMMOD9->NEVNT - (COMMOD9->NSPC * 2); ROW++) {

				if (COMMOD9->PTLORD[ROW][COMMOD9->NEVNT - (COMMOD9->NSPC * 2)]
						== COMMOD9->PTLORD[ROW][COMMOD9->NEVNT
						- (COMMOD9->NSPC * 2) + 1]) {					
					FILLED = FILLED + 1;
					COMMOD9->INIPERM[FILLED - 1] = ROW; 
				}
			}

			for (I = 0; I < COMMOD9->NEVNT - (COMMOD9->NSPC * 2); I++) {

				for (ROW = 0; ROW < COMMOD9->NEVNT - (COMMOD9->NSPC * 2);
						ROW++) {

					if ((COMMOD9->PTLORD[ROW][COMMOD9->NEVNT
								- (COMMOD9->NSPC * 2) + 1] == I)
							&& (COMMOD9->PTLORD[ROW][COMMOD9->NEVNT
								- (COMMOD9->NSPC * 2) + 1]
								> COMMOD9->PTLORD[ROW][COMMOD9->NEVNT
								- (COMMOD9->NSPC * 2)])) {
						FILLED = FILLED + 1;
						COMMOD9->INIPERM[FILLED - 1] = ROW; 

					}
				}
			}

		}

		if (NML > 0)  
		{

			for (I = 0; I < NML; I++) {
				UNIFRN = ::random(0, 1);
				U[I] = UNIFRN;
			}

			::ASORT(U, L1, NML);

			for (K = 0; K < NML; K++) {
				COMMOD9->INIPERM[K + FILLED] = MLASTS[L1[K]];				
			}

			FILLED = FILLED + NML;
		}

		if (FILLED != COMMOD9->IRCOUNT) {
			cout << "ERROR IN GETSTART!!" << endl;
			cout << "INIPERM WRONG SIZE" << endl;
			cout << "IRCOUNT =   " << COMMOD9->IRCOUNT << endl;
			cout << "FILLED =  " << FILLED << endl;

			COMMOD9->STOPF = 1;
			goto Label8888;
		}

	} else if ((COMMOD9->INIGEN == 2) || (COMMOD9->INIGEN > 3)) {		

		string LOADFILE = "";

		switch (COMMOD9->INIGEN) {
			case 2:
			case 6:			
				LOADFILE = COMMOD9->INITSOL;
				if ((COMMOD9->AUTF == 0) && (OUT == 1)) {
					cout << "  Loading Last Solution from File . . . " << endl;
				}
				break;
			case 4:			
				cout << "  Loading Best Solution from File . . . " << endl;
				break;
			case 5:
				if (COMMOD9->CONTF == 1) {
					cout << "  Recalling Interrupted Run . . . " << endl;				
					::FINDROW(SPC, TYP, ROW);

					if (ROW != -1) {
						COMMOD9->BSTPERM[RNK] = ROW;
					}

					if (COMMOD9->PENF <= 1) {
						::GETPEN_WITH_TWOPEN(COMMOD9->BSTPERM, COMMOD9->BSTPEN);
					} else if (COMMOD9->PENF >= 2) {
						::DEMPEN(COMMOD9->BSTPERM, COMMOD9->BSTPEN);
					}

					::DOTOO();

					if (COMMOD9->AUTF == 0) {
						COMMOD9->BS2PEN = COMMOD9->TOOPEN;
						COMMOD9->SQBST = COMMOD9->SQPEN;
						COMMOD9->SHBST = COMMOD9->SHPEN;
						COMMOD9->TSBST = COMMOD9->TSPEN;
					}			


				} else {
					cout << "  Loading Step Solution from File . . . " << endl;				
				}

				break;
		}


		FILLED = 0;



		::fstream _file;
		_file.open(LOADFILE.c_str(), ios::in);
		string line = "";

		while (std::getline(_file, line)) {
			vector<string> elems = ::split(line, ' ');
			::trim(elems[0]);
			::trim(elems[1]);
			::trim(elems[2]);

			SPC = atoi(elems[0].c_str());
			TYP = atoi(elems[1].c_str());
			RNK = atoi(elems[2].c_str());

			::FINDROW(SPC - 1, TYP, ROW);

			FILLED = FILLED + 1;

			if (ROW != -1) {
				COMMOD9->INIPERM[RNK - 1] = ROW;   
			} else {
				if (COMMOD9->CDF != 1) {
					//nothing
				}
			}
		}

		_file.close();		

		if (FILLED != COMMOD9->NEVNT) {			
			COMMOD9->STOPF = 1;
			goto Label8888;
		}
	}

	if (COMMOD9->COXSTF != 4) {		
		for (I = 0; I < COMMOD9->IRCOUNT - 1; I++) {

			if (COMMOD9->IROWS[COMMOD9->INIPERM[I]][1] == 2) {

				ISWAP = 0;
				M = I + 1;

				while ((M < COMMOD9->IRCOUNT) && (ISWAP == 0))         
				{
					if (COMMOD9->IROWS[COMMOD9->INIPERM[M]][1] == 1)   
					{

						if ((COMMOD9->COEXST[(COMMOD9->IROWS[COMMOD9->INIPERM[I]][2])*COMMOD9->NSPC+(COMMOD9->IROWS[COMMOD9->INIPERM[M]][2])]
									>= COMMOD9->COXSTF)
								|| (COMMOD9->IROWS[COMMOD9->INIPERM[M]][0]
									== COMMOD9->IROWS[COMMOD9->INIPERM[I]][0])) {

							TEMP = COMMOD9->INIPERM[M];
							COMMOD9->INIPERM[M] = COMMOD9->INIPERM[I];
							COMMOD9->INIPERM[I] = TEMP;
							ISWAP = 1;
						} else {
							M = M + 1;
						}
					} else {
						M = M + 1;
					}
				}
			}
		}
	} else {

		for (I = 0; I < COMMOD9->IRCOUNT - 1; I++) {

			if (COMMOD9->IROWS[COMMOD9->INIPERM[I]][1] == 2) {

				ISWAP = 0;
				M = I + 1;

				while ((M < COMMOD9->IRCOUNT) && (ISWAP == 0))          
				{

					if ((COMMOD9->IROWS[COMMOD9->INIPERM[M]][1] == 1)
							&& (COMMOD9->IROWS[COMMOD9->INIPERM[M]][0]
								== COMMOD9->IROWS[COMMOD9->INIPERM[I]][0])) {
						TEMP = COMMOD9->INIPERM[M];
						COMMOD9->INIPERM[M] = COMMOD9->INIPERM[I];
						COMMOD9->INIPERM[I] = TEMP;
						ISWAP = 1;
					} else {
						M = M + 1;
					}
				}                      
			}                       
		}                        

	}                      



Label8888:

	delete[] U;
	delete[] L;
	delete[] L1;
	delete[] X;
	delete[] FOUNDS;
	delete[] MFIRSTS;
	delete[] MLASTS;
	delete[] MOTHERS;

	return;
}
