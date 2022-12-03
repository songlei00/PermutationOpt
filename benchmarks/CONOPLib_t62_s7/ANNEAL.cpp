#include <iostream>
#include <iomanip> 
#include <math.h>
#include "COMMOD.h"
#include "Helper.h"
#include "ANNEAL.h"
#include<omp.h>
#include "Output.h"

using namespace std;
using std::string;

ANNEAL* ANNEAL::g_singleton = NULL;

void ANNEAL::RunIt(int NUMBERPROCESS, int MYID)
{
	int C = 1;

	// if(0==MYID)
	// 	cout <<"process["<< MYID << "] enter anneal." << endl;
	
	double DELTA=0.0;

	double U=0.0, OUTER = 0.0;
	int IRANK=0, JRANK=0, LAST=0, NCTR=0, FIXM=0, NABRM=0;
	int I=0, L=0, X=0, NCT=0;

	int INISCOR=0, ISMSCOR=0, ISQSCOR=0, ISHSCOR=0, ITSSCOR=0, INGSCOR=0;
	int BSTSCOR=0, AUTSCOR=0;
	double PCT=0.0;
	int siz = 0;

	string SCORE;

	int horiz=0, vert=900, div=0;
	bool rndf=true, stepf, aimf, redf, trjf, algf, rlxf, rngf, hilf, s45f, s67f, s89f;
	
	COMMOD* COMMOD9 = COMMOD::Singleton();
	

	COMMOD9->UU = 0.0;	

	if ((COMMOD9->AUTF == 0) && (COMMOD9->CONTF == 0))
	{
		COMMOD9->BSTPEN = 0.0;
		COMMOD9->BS2PEN = 0.0;
	}

	COMMOD9->TOOPEN = 0.0;
	COMMOD9->TO2PEN = 0.0;
	COMMOD9->TSPEN = 0.0;
	COMMOD9->SMPEN = 0.0;
	COMMOD9->SHPEN = 0.0;
	COMMOD9->CRTPEN = 0.0;
	COMMOD9->NEWRANK = 0;
	//NEWRANK is the rank to which the last (BIG|SML) mutation moved an event

	stepf = false;
	if ((COMMOD9->STEPSOL.substr(0, 3) != "OFF") && (COMMOD9->STEPSOL.substr(0, 3) != "off"))
	{
		stepf = true;
	}
	
	aimf = false;
	if ((COMMOD9->SOLVEF == 3) && (COMMOD9->FIXF == 2)) aimf = true;

	rngf = false;
	if ((COMMOD9->RUNGRF == 2) || (COMMOD9->RUNGRF == 5)) rngf = true;

	trjf = false;
	if (((COMMOD9->RUNGRF == 2) || (COMMOD9->RUNGRF == 0)) && (COMMOD9->TRJGRF > 0)) trjf = true;

	redf = false;
	if (trjf)
	{
		if ((COMMOD9->RUNGRF != 0) && (COMMOD9->SOLVEF != 2) && (COMMOD9->SOLVEF != 4) &&
				(COMMOD9->SOLVEF != 6) && (COMMOD9->SOLVEF != 8)) redf = true;
	}

	algf = false;
	if ((COMMOD9->SOLVEF == 2) || (COMMOD9->SOLVEF == 4) || (COMMOD9->SOLVEF == 6) ||
			(COMMOD9->SOLVEF == 8)) algf = true;

	rlxf = false;
	if ((COMMOD9->RUNGRF >= 4) || (COMMOD9->GRIDF == 1)) rlxf = true;

	hilf = false;
	if ((COMMOD9->SOLVEF != 2) && (COMMOD9->TRJGRF >= 3)) hilf = true;

	s45f = false;
	if ((COMMOD9->SOLVEF == 4) || (COMMOD9->SOLVEF == 5)) s45f = true;

	s67f = false;
	if ((COMMOD9->SOLVEF == 6) || (COMMOD9->SOLVEF == 7)) s67f = true;

	s89f = false;
	if ((COMMOD9->SOLVEF == 8) || (COMMOD9->SOLVEF == 9)) s89f = true;
	

	NCTR = 0;
	COMMOD9->CTRF = 0;
	COMMOD9->NBETR = 0;

	if (COMMOD9->CONTF == 1) goto Label111;

	switch (COMMOD9->AUTF)
	{
		case 0:		
			COMMOD9->T = COMMOD9->STARTT;
			break;
		case 1:		
			COMMOD9->Rx = COMMOD9->R;
			break;
		case -1:		
			COMMOD9->Rx = COMMOD9->R;
			if ((COMMOD9->ADAF == 1) && (COMMOD9->NTRIES > 0) && (COMMOD9->LSTRIES > 1))
			{
				L=(rand() % (min(COMMOD9->LSTRIES, 5)+1));			
				COMMOD9->Rx = pow(COMMOD9->R, L);
				COMMOD9->NTRIES = COMMOD9->NTRIES + (L - 1);
			}

			COMMOD9->T = COMMOD9->T * COMMOD9->Rx;

			break;
	}

Label111:
	PCT = COMMOD9->NINNER * COMMOD9->NOUTER;


	int tempPenf = COMMOD9->PENF;
	if (COMMOD9->PENF <= 1)
	{		
		COMMOD9->PENF = 0;
		// cout<<"getPenalty"<<endl;
		::GETPEN_WITH_TWOPEN(COMMOD9->INIPERM, COMMOD9->NXTPEN);
		COMMOD9->PENF = tempPenf;
	}
	else if (COMMOD9->PENF >= 2)
	{
		::DEMPEN(COMMOD9->INIPERM, COMMOD9->NXTPEN);
	}

	//cout<<COMMOD9->NXTPEN<<endl;

	::CopyArray(COMMOD9->INIPERM,COMMOD9->NEVNT,COMMOD9->PERM);
	::CopyArray(COMMOD9->INIPERM,COMMOD9->NEVNT, COMMOD9->LSTPERM);
	::CopyArray(COMMOD9->INIPERM,COMMOD9->NEVNT, COMMOD9->PXLPERM);

	if (COMMOD9->CONTF == 0) ::CopyArray(COMMOD9->INIPERM,COMMOD9->NEVNT,COMMOD9->BSTPERM);

	COMMOD9->PEN = COMMOD9->NXTPEN;
	COMMOD9->INIPEN = COMMOD9->NXTPEN;
	INISCOR = (int)COMMOD9->INIPEN;
	AUTSCOR = (int)COMMOD9->NXTPEN;

	if (COMMOD9->AUTF == 0)
	{
		if (COMMOD9->CONTF == 0) COMMOD9->BSTPEN = COMMOD9->NXTPEN;
	}
	else
	{
		if (COMMOD9->CONTF == 0) COMMOD9->BSTPEN = COMMOD9->AUTPEN;
	}

	BSTSCOR = (int)COMMOD9->BSTPEN;	

	COMMOD9->NGTPEN = COMMOD9->NGHPEN;
	INGSCOR = (int)COMMOD9->NGTPEN;

	if ((COMMOD9->AUTF == 0) && (COMMOD9->CONTF == 0))
	{
		COMMOD9->BS2PEN = COMMOD9->TOOPEN;
		COMMOD9->SQBST = COMMOD9->SQPEN;
		COMMOD9->SHBST = COMMOD9->SHPEN;
		COMMOD9->TSBST = COMMOD9->TSPEN;
	}

	ISQSCOR = (int)COMMOD9->SQPEN;
	ISHSCOR = (int)COMMOD9->SHPEN;
	COMMOD9->TSLST = COMMOD9->TSPEN;
	ITSSCOR = (int)COMMOD9->TSPEN;

	ISMSCOR = (int)COMMOD9->SMPEN;

	if ((COMMOD9->KSM > 0.0) && (COMMOD9->PENF < 2))
	{
		ISMSCOR = (int)(COMMOD9->SMPEN / COMMOD9->KSM);
		INISCOR = (int)(COMMOD9->PEN - COMMOD9->SMPEN);
		BSTSCOR = (int)(COMMOD9->BSTPEN - COMMOD9->SMPEN);
	}

	if (COMMOD9->CTRF == 1)
	{
		NCT = NCT + 1;
		COMMOD9->CTRF = 0;
	}

	if (COMMOD9->GRID2F == 1)
	{
		//nothing		
	}


	::CopyArray(COMMOD9->HSCTSOL, COMMOD9->SCTSOL_SIZE, COMMOD9->SCTSOL);
	::CopyArray(COMMOD9->COLPEN,COMMOD9->NSCT,COMMOD9->SCJPEN);
	LAST = 1;
	COMMOD9->XPOSN = 0;

	if (COMMOD9->FIXF == 3)
	{
		for (COMMOD9->XPOSN = 0; COMMOD9->XPOSN < COMMOD9->NEVNT; COMMOD9->XPOSN++)
		{
			if (COMMOD9->PERM[COMMOD9->XPOSN] == COMMOD9->XEVNT) break;
		}
	}


	COMMOD9->CONTF = 0;

	//*******************************************************************
	//  MAIN LOOPS
	//     The following lines are executed with every trial!!
	//     Analyse a big sequence of neighbors ('permutations').
	//     Compute the penalty and decide whether to accept the new perm
	//     according to the SOLVER option
	//*******************************************************************

													
	

	if ((MYID == 0 && COMMOD9->ANNEALONLY) && (COMMOD9->NOUTER > 0 || COMMOD9->NINNER > 0))
	{
		::GETPEN_WITH_TWOPEN(COMMOD9->BSTPERM, COMMOD9->BSTPEN);	
	}

	if (MYID == 0&&!COMMOD9->ANNEALONLY) {
		COMMOD9->AUTPEN = COMMOD9->BSTPEN;
		COMMOD9->NUDGUP = false;
		COMMOD9->NUDGDN = false;
		COMMOD9->FIXF = FIXM;
		COMMOD9->NABRGEN = NABRM;
		if ((COMMOD9->STEPSOL.substr(0, 3) == "LST") ||
				(COMMOD9->STEPSOL.substr(0, 3) == "lst"))
		{
			if (COMMOD9->CDF != 1) ::STEPOUT(COMMOD9->LSTPERM);
			if (COMMOD9->CDF != 1) ::TRAJOUT();
		}
		LAST = COMMOD9->PENF;
		if ((COMMOD9->PAUSF == 5) && (COMMOD9->NTRIES < COMMOD9->NSTOP - 1))
		{
			COMMOD9->NUDGUP = false;
			COMMOD9->NUDGDN = false;
			goto Label33;
		}
		if (COMMOD9->PENF == 1)
		{
			COMMOD9->PENF = -1;
			::GETPEN_WITH_TWOPEN(COMMOD9->BSTPERM, COMMOD9->RVLPEN);
			if (LAST == -1) ::CopyArray(COMMOD9->COLPEN, COMMOD9->NSCT, COMMOD9->SCJPEN);

			COMMOD9->PENF = 0;
			::GETPEN_WITH_TWOPEN(COMMOD9->BSTPERM, COMMOD9->INTPEN);
			if (LAST == 0) ::CopyArray(COMMOD9->COLPEN, COMMOD9->NSCT, COMMOD9->SCJPEN);

			COMMOD9->PENF = 1;
			::GETPEN_WITH_TWOPEN(COMMOD9->BSTPERM, COMMOD9->LVLPEN);
			if (LAST == 1) ::CopyArray(COMMOD9->COLPEN, COMMOD9->NSCT, COMMOD9->SCJPEN);
		}
		else if (COMMOD9->PENF == 0)
		{
			COMMOD9->PENF = -1;
			CalcEPEN();
			::GETPEN_WITH_TWOPEN(COMMOD9->BSTPERM, COMMOD9->RVLPEN);
			if (LAST == -1) ::CopyArray(COMMOD9->COLPEN, COMMOD9->NSCT, COMMOD9->SCJPEN);

			COMMOD9->PENF = 1;
			CalcEPEN();
			::GETPEN_WITH_TWOPEN(COMMOD9->BSTPERM, COMMOD9->LVLPEN);
			if (LAST == 1) ::CopyArray(COMMOD9->COLPEN, COMMOD9->NSCT, COMMOD9->SCJPEN);

			COMMOD9->PENF = 0;
			CalcEPEN();
			::GETPEN_WITH_TWOPEN(COMMOD9->BSTPERM, COMMOD9->INTPEN);
			if (LAST == 0) ::CopyArray(COMMOD9->COLPEN, COMMOD9->NSCT, COMMOD9->SCJPEN);
		}
		else if (COMMOD9->PENF == -1)
		{
			COMMOD9->PENF = 1;
			::GETPEN_WITH_TWOPEN(COMMOD9->BSTPERM, COMMOD9->LVLPEN);
			if (LAST == 1) ::CopyArray(COMMOD9->COLPEN, COMMOD9->NSCT, COMMOD9->SCJPEN);
			COMMOD9->PENF = 0;
			::GETPEN_WITH_TWOPEN(COMMOD9->BSTPERM, COMMOD9->INTPEN);
			if (LAST == 0) ::CopyArray(COMMOD9->COLPEN, COMMOD9->NSCT, COMMOD9->SCJPEN);
			COMMOD9->PENF = -1;
			::GETPEN_WITH_TWOPEN(COMMOD9->BSTPERM, COMMOD9->RVLPEN);
			if (LAST == -1) ::CopyArray(COMMOD9->COLPEN, COMMOD9->NSCT, COMMOD9->SCJPEN);
		}
		COMMOD9->NGTPEN = COMMOD9->NGHPEN;
		COMMOD9->PENF = 6;
		if (COMMOD9->FB4LF != 0) ::DEMPEN(COMMOD9->BSTPERM, COMMOD9->SEQPEN);
		COMMOD9->PENF = 5;
		::DEMPEN(COMMOD9->BSTPERM, COMMOD9->ROYPEN);
		COMMOD9->PENF = 7;
		::DEMPEN(COMMOD9->BSTPERM, COMMOD9->MOMPEN);
		if (LAST == 7) ::CopyArray(COMMOD9->COLPEN, COMMOD9->NSCT, COMMOD9->SCJPEN);
		COMMOD9->PENF = 4;
		::DEMPEN(COMMOD9->BSTPERM, COMMOD9->RSCPEN);
		if (LAST == 4) ::CopyArray(COMMOD9->COLPEN, COMMOD9->NSCT, COMMOD9->SCJPEN);
		COMMOD9->PENF = 3;
		::DEMPEN(COMMOD9->BSTPERM, COMMOD9->SPTPEN);
		if (LAST == 3) ::CopyArray(COMMOD9->COLPEN, COMMOD9->NSCT, COMMOD9->SCJPEN);
		COMMOD9->PENF = 2;
		::DEMPEN(COMMOD9->BSTPERM, COMMOD9->ORDPEN);
		if (LAST == 2) ::CopyArray(COMMOD9->COLPEN, COMMOD9->NSCT, COMMOD9->SCJPEN);
		if (LAST == 5) ::CopyArray(COMMOD9->COLPEN, COMMOD9->NSCT, COMMOD9->SCJPEN);
		if ((LAST == 6) && (COMMOD9->FB4LF != 0)) ::CopyArray(COMMOD9->COLPEN, COMMOD9->NSCT, COMMOD9->SCJPEN);
		if (LAST > 2)
		{
			COMMOD9->PENF = 0;
			::GETPEN_WITH_TWOPEN(COMMOD9->BSTPERM, COMMOD9->INTPEN);
		}
		COMMOD9->PENF = LAST;
		switch (COMMOD9->PENF)
		{
			case -1://eventual
				COMMOD9->BSTPEN = COMMOD9->RVLPEN;
				break;
			case 0://interval
				COMMOD9->BSTPEN = COMMOD9->INTPEN;
				break;
			case 1://level
				COMMOD9->BSTPEN = COMMOD9->LVLPEN;
				break;
			case 2://ordinal
				COMMOD9->BSTPEN = COMMOD9->ORDPEN;
				break;
			case 3://spatial
				COMMOD9->BSTPEN = COMMOD9->SPTPEN;
				break;
			case 4://rascal/rascer
				COMMOD9->BSTPEN = COMMOD9->RSCPEN;
				break;
			case 5://royal (Coex)
				COMMOD9->BSTPEN = COMMOD9->ROYPEN;
				break;
			case 6://sequel (Fb4L)
				if (COMMOD9->FB4LF != 0) COMMOD9->BSTPEN = COMMOD9->SEQPEN;
				break;
			case 7://momental
				COMMOD9->BSTPEN = COMMOD9->MOMPEN;
				break;
		}

Label33:
		COMMOD9->CTRF = NCTR;
	}

	
}

