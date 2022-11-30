#include<stdio.h>
#include <iomanip>
#include <string>
#include <vector>
#include<iostream>
#include <map>
#include <math.h>
#include "Helper.h"
#include "COMMOD.h"
#include<string.h>
#include<omp.h>

using namespace std;
using std::string;
using std::vector;
extern float *epen;
extern int epenCount;
extern int	*position;
extern int EVT_X_LEVEL;
extern int *IDX_SCT_EPEN;
extern int ** IDX_EVT_SCT_EPEN;
//extern unsigned short *I_M_Arr;
//extern int i_m_count;
//extern int *IROWS_3;

vector<string> &split(const string &s, char delim, vector<string> &elems) {
	stringstream ss(s);
	string item;
	while (getline(ss, item, delim)) {
		if (item != "")elems.push_back(item);
	}
	return elems;
}

void REORDER(int perm[], int before[], int a, int order[], int obj[], int b) {
	int i = 0;
	int j = 0;

	for (i = 0; i < a; i++) {
		perm[order[j]] = obj[j];
	}
}

void MARKPOSITION(int XPERM[], int size)
{

	for (int i = 0; i < size; i++) {
		position[XPERM[i]] = i;
	}
}

bool COEX2(int sp1, int sp2) {

	COMMOD* COMMOD9 = COMMOD::Singleton();

	int lsp1 = COMMOD9->IROWS[sp1][3];
	int	lsp2 = COMMOD9->IROWS[sp2][3];

	if (position[sp1] < position[sp2]) {
		return position[sp2] < position[lsp1];
	}
	else {
		return position[sp1] < position[lsp2];
	}
}

vector<string> split(const string &s, char delim) {
	vector<string> elems;
	split(s, delim, elems);
	return elems;
}

void EVENTSUM(int flg, double fac)
{
	int I, J;
	int Nj, Nmax, Nmin;
	double NEARBEST, mrg;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	NEARBEST = 0.0;
	mrg = 0.0;
	mrg = fac + 1.0;

	if (flg == 4)
	{
		NEARBEST = COMMOD9->VERYBEST * ::max(1.0, mrg);
	}

	for (I = 0; I < COMMOD9->NEVNT; I++)
	{
		Nj = 0;
		Nmin = COMMOD9->NEVNT;
		Nmax = 1;

		if (flg == 4)
		{
			for (J = 0; J < COMMOD9->NEVNT; J++)
			{			
				if (COMMOD9->NXTPEN <= NEARBEST)
				{
					Nmin = ::min(Nmin, J);
					Nmax = ::max(Nmax, J);
				}

			}

			COMMOD9->CULLIST[I][0] = Nmax;
			COMMOD9->CULLIST[I][1] = Nmin;

		}
		else
		{
			for (J = 0; J < COMMOD9->NSCT; J++)
			{
				if (flg == 1)
				{
					if (COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT + I] > -1) Nj = Nj + 1;
				}

				if (flg == 2)
				{
					if ((COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT + I] >= 0) && (COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT + I] < COMMOD9->HLEVEL[J])) Nj = Nj + 1;
				}

				if (flg == 3)
				{
					if ((COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + I] >= 0) && (COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + I]< COMMOD9->HLEVEL[J])) Nj = Nj + 1;
				}
			}

			if (flg == 1) COMMOD9->CULLIST[I][2] = Nj;
			if (flg == 2) COMMOD9->CULLIST[I][3] = Nj;
			if (flg == 3) COMMOD9->CULLIST[I][4] = Nj;
		}

	}
}

bool IORZERO(int evtrow, int sct)
{
	COMMOD* COMMOD9 = COMMOD::Singleton();
	int IDX = COMMOD9->IDX_SCT_RSTATIC[sct] + evtrow * 2;

	bool retVal = false;

	if ((COMMOD9->ISTATIC_0[sct*COMMOD9->NEVNT + evtrow] == -1) ||
			((COMMOD9->NRSTATIC[IDX] == 0.0) &&
			 (COMMOD9->NRSTATIC[IDX + 1] == 0.0)))
	{
		retVal = true;
	}
	return retVal;
}

bool RZERO(int evtrow, int sct)
{
	bool retVal = false;
	COMMOD* COMMOD9 = COMMOD::Singleton();

	int IDX = COMMOD9->IDX_SCT_RSTATIC[sct] + evtrow * 2;	

	if ((COMMOD9->NRSTATIC[IDX] == 0.0) &&
			(COMMOD9->NRSTATIC[IDX + 1] == 0.0))
	{
		retVal = true;
	}
	return retVal;

}

bool ALLRZERO(int evtrow)
{
	bool retVal = true;
	int sct;
	COMMOD* COMMOD9 = COMMOD::Singleton();

	int IDX = 0;
	for (sct = 0; sct < COMMOD9->NSCT; sct++)
	{
		IDX = COMMOD9->IDX_SCT_RSTATIC[sct] + evtrow * 2;
		if ((COMMOD9->NRSTATIC[IDX] > 0.0) && (COMMOD9->NRSTATIC[IDX + 1] > 0.0))
		{		
			return false;
		}

	}
	return retVal;
}

void FINDROW(int EVT, int TYP, int& ROW)
{
	COMMOD* COMMOD9 = COMMOD::Singleton();

	for (ROW = 0; ROW < COMMOD9->NEVNT; ROW++)
	{
		if ((COMMOD9->IROWS[ROW][0] == EVT) && (COMMOD9->IROWS[ROW][1] == TYP)) break;
	}

	if (ROW == COMMOD9->NEVNT) ROW = -1;
}

void ASORT(double X[], int L[], int N)
{
	int I, J;
	int D, TEMP, FLAG;

	::SetVal(L, N, 0);

	D = (int)::pow(2.0L, (int)(log((double)N) / log(2.0)));

	for (I = 0; I < N; I++)
	{
		L[I] = I;
	}

Label15:
	if (D > 0)
	{
		for (I = 0; I < N - D; I++)
		{
			if (X[L[I]] > X[L[I + D]])
			{
				TEMP = L[I + D];
				L[I + D] = L[I];

				if (I <= D)
				{
					L[I] = TEMP;
				}

				else
				{
					FLAG = 1;
					J = I - D;

Label25:
					if (FLAG != 0)
					{

						if (X[TEMP] > X[L[J]])
						{
							L[J + D] = TEMP;
							FLAG = 0;
						}

						else
						{
							L[J + D] = L[J];
							J = J - D;

							if (J < 1)
							{
								FLAG = 0;
								L[J + D] = TEMP;
							}
						}

						goto Label25;
					}
				}
			}
		}

		D = (int)(D / 2);

		goto Label15;
	}
}

void ASORT_DESC(double X[], int L[], int N)
{
	int I, J;
	int D, TEMP, FLAG;

	::SetVal(L, N, 0);

	D = (int)::pow(2.0L, (int)(log((double)N) / log(2.0)));

	for (I = 0; I < N; I++)
	{
		L[I] = I;
	}

Label915:
	if (D > 0)
	{

		for (I = 0; I < N - D; I++)
		{
			if (X[L[I]] < X[L[I + D]])
			{
				TEMP = L[I + D];
				L[I + D] = L[I];

				if (I <= D)
				{
					L[I] = TEMP;
				}

				else
				{
					FLAG = 1;
					J = I - D;

Label925:
					if (FLAG != 0)
					{

						if (X[TEMP] < X[L[J]])
						{
							L[J + D] = TEMP;
							FLAG = 0;
						}

						else
						{
							L[J + D] = L[J];
							J = J - D;

							if (J < 1)
							{
								FLAG = 0;
								L[J + D] = TEMP;
							}
						}

						goto Label925;
					}
				}
			}
		}

		D = (int)(D / 2);

		goto Label915;
	}
}

void IASORT(int X[], int L[], int N)
{
	int I, J;
	int D, TEMP, FLAG;

	::SetVal(L, N, 0);

	D = (int)::pow(2.0L, (int)(log((double)N) / log(2.0))) - 1;

	for (I = 0; I < N; I++)
	{
		L[I] = I;
	}

Label15:
	if (D > 0)
	{

		for (I = 0; I < N - D; I++)
		{
			if (X[L[I]] > X[L[I + D]])
			{
				TEMP = L[I + D];
				L[I + D] = L[I];

				if (I <= D)
				{
					L[I] = TEMP;
				}

				else
				{
					FLAG = 1;
					J = I - D;

Label25:
					if (FLAG != 0)
					{

						if (X[TEMP] > X[L[J]])
						{
							L[J + D] = TEMP;
							FLAG = 0;
						}

						else
						{
							L[J + D] = L[J];
							J = J - D;

							if (J < 1)
							{
								FLAG = 0;
								L[J + D] = TEMP;
							}
						}

						goto Label25;
					}
				}
			}
		}

		D = (int)(D / 2);

		goto Label15;
	}
}

double NEGPEN(int EVENTROW, int LEVEL, int JOPT)
{
	double retVal = 0.0;

	double DIST = 0.0;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	COMMOD9->CTRF = 0;

	if ((COMMOD9->IROWS[EVENTROW][1] > 2) || (COMMOD9->IROWS[EVENTROW][1] < 1) ||
			(((COMMOD9->IROWS[EVENTROW][1] < 3) || (COMMOD9->IROWS[EVENTROW][1] > 0)) &&
			 ((COMMOD9->NEGATIVE[COMMOD9->IROWS[EVENTROW][2]][JOPT] == 2) &&
			  (COMMOD9->NEGATF == 1))))
	{

		DIST = 0.0;
	}
	else
	{

		if (COMMOD9->IROWS[EVENTROW][1] == 1)
		{

			switch (COMMOD9->STKF)
			{
				case 2:

					if (LEVEL < COMMOD9->HLEVEL[JOPT])
					{
						DIST = 0.5;
					}
					else
					{
						DIST = -0.5;
					}
					break;
				case 5:

					if (COMMOD9->PENF == 1)
					{

						DIST = 1.0 - (((double)LEVEL) / (double)(COMMOD9->HLEVEL[JOPT]));
					}
					else if ((COMMOD9->PENF == 0) || (COMMOD9->PENF > 1))
					{
						DIST = 1.0 - (COMMOD9->VALEVEL[JOPT][LEVEL] + 1)
							/ (COMMOD9->VALEVEL[JOPT][COMMOD9->HLEVEL[JOPT]] + 1);
					}
					else if (COMMOD9->PENF == -1)
					{
						DIST = 1.0 - ((double)(COMMOD9->ELEVEL[JOPT][LEVEL] + 1)
								/ (double)(COMMOD9->ELEVEL[JOPT][COMMOD9->HLEVEL[JOPT]] + 1));
					}
					break;
			}

		}
		else if (COMMOD9->IROWS[EVENTROW][1] == 2)
		{

			switch (COMMOD9->STKF)
			{
				case 2:

					if (LEVEL > 1)
					{
						DIST = 0.5;
					}
					else
					{
						DIST = -0.5;
					}

					break;
				case 5:

					if (COMMOD9->PENF == 1)
					{
						DIST = (double)(LEVEL - 1 + 1) / ((double)COMMOD9->HLEVEL[JOPT] + 1);
					}
					else if ((COMMOD9->PENF == 0) || (COMMOD9->PENF > 1))
					{
						DIST = (COMMOD9->VALEVEL[JOPT][LEVEL] - 1.0) / (double)(COMMOD9->HLEVEL[JOPT] + 1);
					}
					else if (COMMOD9->PENF == -1)
					{
						DIST = (double)(COMMOD9->ELEVEL[JOPT][LEVEL] - 1 + 1)
							/ (double)(COMMOD9->ELEVEL[JOPT][COMMOD9->HLEVEL[JOPT]] + 1);
					}
					break;
			}

		}
	}

	retVal = DIST;
	COMMOD9->NGHPEN = COMMOD9->NGHPEN + DIST;

	return retVal;

}

void SCTPEN(int HPERM[], int JOPT, double& PENJ)
{
	COMMOD* COMMOD9 = COMMOD::Singleton();
	
	int I;
	float PENDOWN, PENHERE;
	int H_LEVEL = COMMOD9->HLEVEL[JOPT] + 1;
	float* LPEN = new float[COMMOD9->MXLVL];
	int DOWN, HERE, INOW, ITGT, LPREV, PLACED, THISONE, CTRSECT, CTRHERE, CTRDOWN;
	int IDX_EPEN = 0;

	int HPERM_V = 0;
	int EPEN_IDX_V = 0;
	int currEvent;
	//C***********************************************************************
	//C   PLACE THE OBS EVENTS TO MINIMIZE THE TOTAL PENALTY IN SECTION JOPT
	//C   AND COMPUTE THE TOTAL PENALTY FOR THIS SECTION - PENJ
	//C         ITGT is the level in section JOPT that the current event
	//C             would be placed at if no other events were in the way
	//C         if the event has a datum in JOPT, the level of the datum
	//C             is ITGT
	//C         there are three special cases:
	//C             if the event has no datum in JOPT, ITGT = 0
	//C             if the event is the top of a range that extends
	//C                 up out of JOPT, the level is coded -1, and
	//C                 ITGT is the highest level in the section
	//C                 (HLEVEL(JOPT))
	//C             if the event is the bottom of a range that extends
	//C                 down out of JOPT, the level is coded -2, and
	//C                 ITGT is 1
	//CPMS     INOW - loop counter
	//CPMS  THISONE - temporary INOW
	//CPMS     HERE - temporary LPREV
	//CPMS     DOWN - level below HERE
	//CPMS  PENDOWN - sum of EPEN's
	//C-------------------------------
	//c     initialize:
	COMMOD9->CTRF = 0;
	CTRSECT = 0;
	CTRHERE = 0;
	CTRDOWN = 0;
	//c     initialize more for Intel:
	DOWN = 0;
	HERE = 0;
	INOW = 0;
	ITGT = 0;
	//c     not JOPT  ! intent is INPUT!
	PLACED = 0;
	THISONE = 0;
	I = 0;
	//c
	//c     not HPERM     ! intent is INPUT!

	//CPMS  blank the section penalty:
	PENJ = 0.0;  //OK it is passed for OUTPUT
				 //CPMS  initialize for Intel
	PENDOWN = 0.0;
	PENHERE = 0.0;
	//CPMS  start at base level:
	INOW = 0;//<=HXD
	LPREV = 0;//<=HXD
	THISONE = 0;//<=HXD
	
	//IDX_EPEN = IDX_SCT_EPEN[JOPT];
	
	//CPMS  zero out the array of level penalties:
	::SetVal(LPEN, COMMOD9->MXLVL, (float) 0.0);
	

	int J_X_NEVNT = JOPT * COMMOD9->NEVNT;
	int J_X_NEVNT_CURREVENT;

	//C*********************************************
	//C     start the overall loop
	//C         - finding placements for all events in one section
	//C     one pass through this loop for each event in the data set.
	//C         we start with the event with RANK 1 in HPERM and place the
	//C             events in order of their rank
	//C
	//C         as each event is placed, others are moved, as necessary, so
	//C             that the overall penalty for everything placed thus far
	//C             is minimized
	//C
	//C         first we find the target in this section for the event
	//C
	//C         then we find the placement (for this event and any others
	//C             that must move) that minimizes the overall penalty thus far
	//C
	//C         the penalty for a placement of an event is based on the
	//C             distance from, and position relative to, its target
	//C
	//C         the placement must also consider any additional penalties that
	//C             result from moving other events to accomodate placing this
	//C             one (the order must be retained) and any constraints
	//C----------------------------------------------------------------------

	//float* EVEVTPEN = new float[COMMOD9->NEVNT];
	//::SetVal(EVEVTPEN, COMMOD9->NEVNT, (float) 0.0);


	//Get This Section SubPERM, Added by HXD 2020-08-09
	int* SUBPERM = new int[COMMOD9->NEVNT];
	int SCTEVENTS = 0;
	int FirstEVTID = -1;
	int LastEVTID = 0;	

	for (INOW = 0; INOW < COMMOD9->NEVNT; ++INOW)
	{
		
		currEvent = HPERM[INOW];
		J_X_NEVNT_CURREVENT = J_X_NEVNT + currEvent;
		ITGT = COMMOD9->ISTATIC_0[J_X_NEVNT_CURREVENT];

		/*if (ITGT > -1)
		{
			SUBPERM[SCTEVENTS] = currEvent;
			SCTEVENTS++;

		}*/

		switch (ITGT)
		{
		case -1:
			break;
		default:
			SUBPERM[SCTEVENTS] = currEvent;
			SCTEVENTS++;
			if (-1 == FirstEVTID)
			{
				FirstEVTID = INOW;
			}
			LastEVTID = INOW;
			break;
		}
	}


	for (INOW = 0; INOW < SCTEVENTS; ++INOW)
	{

		currEvent = SUBPERM[INOW];
		J_X_NEVNT_CURREVENT = J_X_NEVNT + currEvent;
		ITGT = COMMOD9->ISTATIC_0[J_X_NEVNT_CURREVENT];

		if (LPREV <= ITGT)
		{

			COMMOD9->HSCTSOL[J_X_NEVNT_CURREVENT] = ITGT;
			LPREV = ITGT;
		}
		else
		{
			//CPMS    -----CASE C. the hard one, one or more events are placed above this
			//C                    event's datum. FIND THE LOCATION WITH THE SMALLEST
			//C            COMBINED PENALTY
			//CPMS   NOT ELSEIF, because it is the only remaining option
			//CPMS       placement of earlier events has already extended to
			//CPMS       horizons higher than target.  IF INOW is an LAD,
			//CPMS       extension penalty arises; if an FAD, contraction penalty arises
			//CPMS       later try to reduce penalty by moving whole group down.
			//C
			//C      Place this event (row HPERM(INOW) in the arrays)
			//C      AT LPREV AND MOVE it down,
			//C      TAKING OTHERS ALONG AS NECESSARY TO PRESERVE HPERM ORDER
			//c         1. calc pen for all at LPREV, including current
			//c         2. calc pen for all if moved to LPREV - 1
			//c         3. if pen moved down is less, decrement LPREV and repeat
			//c
			//c       THISONE and HERE are temp variables for INOW and LPREV
			//c       so they can be decremented without losing the current values.
			//c
			//c       DOWN is the level below HERE
			//c
			//c       LPEN is used to hold the penalty increment at each
			//c       level in the current section (sum of the penalties
			//c       for all events placed at this level
			//c
			//c       Only Case-C placements will cause penalties
			//cpms    IF NEGATF>0 some of the C placements will need EPEN
			//cpms    and others will need NEGPEN -- so we need to check
			//cpms    NEGATF and ISTATIC before any EPEN/NEGPEN call
			//cpms
			//C-------------initial setup
			//C             place current event at LPREV, set some temp variables,
			//C             compute the penalty at LPREV, and set a flag saying the event
			//C             has not been placed yet
			//C----------------------------------------------------------------------

			COMMOD9->HSCTSOL[J_X_NEVNT_CURREVENT] = LPREV;

			//CPMS   add INOW to developing horizon solution
			//CPMS   set up temporary variables to decrement
			THISONE = INOW;
			HERE = LPREV;

			//HACK:Flag Removed by HXD 2020-08-08
			//COMMOD9->CTRF = 0;

			//CPMS    add INOW penalty to temporary level penalty
			//CPMS         PENHERE does not accumulate so it is never emptied
			//CPMS         it is always replaced in one shot
			//CPMS    ####### CHECK FOR NEGATIVE PENALTIES ##################
			//cpms    do common case first!
			//if (COMMOD9->ISTATIC[JOPT][HPERM[INOW]][0] != -1)//<=HXD
			//{
			//	//PENHERE = LPEN[LPREV] + ::EPEN(HPERM[INOW], HERE, JOPT);
			//	//PENHERE = LPEN[LPREV] + epen[JOPT][HPERM[INOW]][HERE];
			//	PENHERE = LPEN[LPREV] + epen[JOPT_X_EVT_X_LEVEL + HPERM[INOW]*COMMOD9->MXLVL+HERE];
			//}
			//else if (COMMOD9->NEGATF > 0)
			//{
			//	PENHERE = LPEN[LPREV] + ::NEGPEN(HPERM[INOW], HERE, JOPT);
			//}
			//if(COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT+ HPERM[INOW]]!=-1)
			//PENHERE = LPEN[LPREV] + epen[IDX_EPEN + currEvent *H_LEVEL + HERE];

			///if (IDX_EVT_SCT_EPEN[JOPT][currEvent] != -1)
			///{
			PENHERE = LPEN[LPREV] + epen[IDX_EVT_SCT_EPEN[JOPT][currEvent] + HERE];
			//EVEVTPEN[currEvent] = epen[IDX_EVT_SCT_EPEN[JOPT][currEvent] + HERE];
		///}

		//CPMS    ##################
		//CPMS    open option to move
			PLACED = 0;

			//HACK:Flag Removed by HXD 2020-08-08
			//if (COMMOD9->CTRF == 1) CTRHERE = 1;

			//C----- BIG LOOP ----------------------------------------------
			//CPMS          repeat until the placement with the smallest penalty
			//CPMS          is found (PLACED = 1)
			//CPMS          or events have moved down to the base (LEVEL 1)
			//C-----------------------------------------------------------------
			//HACK:Flag Modified by HXD 2020-08-08 # (PLACED == 0)
			while ((HERE > 0) && (PLACED == 0))//<=HXD
			{

				//CPMS          set up temporary variable for the next level down:
				DOWN = HERE - 1;
				//CPMS          set up 0 variable to accumulate penalty increments
				//CPMS          for events moving down to DOWN
				PENDOWN = 0.0;

				//CPMS          SMALL LOOP -------------------------------------
				//CPMS               find penalty for moving all at HERE down 1 level
				//CPMS               work down through earlier events as long as they
				//CPMS               exist (>0) and have been placed at or above the
				//CPMS               current level (i.e. reach up and fetch down)
				//C-------------------------------------------------------------------

				if (THISONE >= 0)//HACK: May be 0
				{
					HPERM_V = SUBPERM[THISONE];

					//while ((COMMOD9->HSCTSOL[J_X_NEVNT + HPERM[THISONE]] >= HERE))
					while ((COMMOD9->HSCTSOL[J_X_NEVNT + HPERM_V] >= HERE))
					{
						//CPMS   - work back through all earlier events on HPERM
						//CPMS     at level HERE
						//HACK:Flag Removed by HXD 2020-08-08
						//COMMOD9->CTRF = 0;

						//CPMS  #### CHECK FOR NEGATIVE PENALTIES ##################
						//cpms       do common case first!
						//if (COMMOD9->ISTATIC[JOPT][HPERM[INOW]][0] != -1)//<=HXD
						//{
						//PENDOWN = PENDOWN + ::EPEN(HPERM[THISONE], DOWN, JOPT);
						//PENDOWN = PENDOWN + epen[JOPT][HPERM[THISONE]][DOWN];
						//if (COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[INOW]] != -1)
						//PENDOWN = PENDOWN + epen[IDX_EPEN + tmpEvent * H_LEVEL + DOWN];


						EPEN_IDX_V = IDX_EVT_SCT_EPEN[JOPT][HPERM_V];
						///if (EPEN_IDX_V != -1)
						///{
						PENDOWN = PENDOWN + epen[EPEN_IDX_V + DOWN];

						//EVEVTPEN[HPERM[THISONE]] = epen[IDX_EVT_SCT_EPEN[JOPT][HPERM[THISONE]] + DOWN];							
					///}


					//if (IDX_EVT_SCT_EPEN[JOPT][HPERM[THISONE]] != -1)
					//{
					//	PENDOWN = PENDOWN + epen[IDX_EVT_SCT_EPEN[JOPT][HPERM[THISONE]] + DOWN];

					//	//EVEVTPEN[HPERM[THISONE]] = epen[IDX_EVT_SCT_EPEN[JOPT][HPERM[THISONE]] + DOWN];							
					//}

					/*}
					else if (COMMOD9->NEGATF > 0)
					{
					PENDOWN = PENDOWN + ::NEGPEN(HPERM[THISONE], DOWN, JOPT);
					}*/

					//CPMS    ##################
					//CPMS    - total up penalty at new low level due to events
					//CPMS      that have just been moved down						

						if (THISONE == 0)break;//<=0							

						//CPMS    select earlier event if there is one i.e. THISONE.gt.1
						THISONE = THISONE - 1;
						HPERM_V = SUBPERM[THISONE];


						//HACK:Flag Removed by HXD 2020-08-08
						//if (COMMOD9->CTRF == 1) CTRDOWN = 1;
					}

				}//the THISONE condition


				 //CPMS     see if DOWN is going to be as far as we move:
				 //C        check to see if DOWN is better than HERE (PENDOWN < PENHERE)
				 //C        if it is,
				 //C         check if DOWN is bottom (level 1).
				 //C         if it is,
				 //C          place group at DOWN and update things
				 //C          if not,
				 //C           get ready for the next pass through big loop
				 //C
				 //C        note that the penalty at the new HERE is the penalty for moving
				 //C        all the guys above down to HERE (PENDOWN) plus the penalty
				 //C        for guys that are already there (LPEN(DOWN))
				 //C----------------------------------------------------------------------
				 //CWGK       20 dec 94 -- I duplicated some code below to correct an error
				 //CWGK       it should be made cleaner
				 //C----------------------------------------------------------------------
				 //CPMS       if penalty has improved, prepare to try going down farther
				 //CPMS       move HERE down
				if (PENDOWN <= PENHERE)
				{

					//penalty has improved
					//HACK:Modified by HXD 2020-08-08					
					//HERE = HERE - 1;
					HERE = DOWN;

					PENHERE = PENDOWN + LPEN[DOWN];

					//add extra penalty and earlier penalty for new HERE
					//unless at base; in that case wrap up

					if (HERE == 0)//<=HXD
					{
						//at base of section!  wrap up!!
						for (I = INOW; I >= THISONE + 1; I--)//HACK:DO I=INOW,THISONE+1,-1 IN FORTRAN
						{
							//update horizon array for all moved events
							COMMOD9->HSCTSOL[J_X_NEVNT + SUBPERM[I]] = HERE;
						}

						//set flag for end:
						PLACED = 1;

						//HACK:Commented Modified by HXD 2020-08-08
						/*if (CTRDOWN == 1) CTRSECT = 1;
						CTRDOWN = 0;
						CTRHERE = 0;*/


						//update LPEN at HERE:
						LPEN[HERE] = PENHERE;

						for (I = HERE + 1; I <= LPREV; I++)//HACK:DO I = HERE+1, LPREV IN FORTRAN
						{
							//reset to zero all levels above HERE up to LPREV
							LPEN[I] = 0.0;
						}


						//move LPREV to HERE; no higher events remain
						LPREV = HERE;
						//Helper.Write("PENDOWN <= PENHERE LPREV={0}\n", LPREV);
					}
					else
					{
						//not at base, prepare to move on down
						//HACK:Commented  by HXD 2020-08-08
						/*CTRHERE = CTRDOWN;
						CTRDOWN = 0;*/

						//reset THISONE toINOW (youngest event)
						THISONE = INOW;
					}
				}
				else//PENDOWN>PENHERE
				{
					//CPMS  if penalty is worse (PENDOWN>PENHERE), prepare to stop
					//CPMS     at HERE
					//C     if DOWN is not better than HERE, HERE is best.
					//C        place all that need to move HERE
					//C        set PLACED to 1
					//C        update LPEN(HERE)
					//C        zeroize LPEN for levels above HERE
					//C        update LPREV
					//CPMS       penalty has not improved, don't move on down

					for (I = INOW; I >= THISONE + 1; I--)//HACK: DO I = INOW,THISONE+1,-1 IN FORTRAN
					{
						//update horizon array for all moved events
						COMMOD9->HSCTSOL[J_X_NEVNT + SUBPERM[I]] = HERE;
					}

					//set flag for end:
					PLACED = 1;

					//update LPEN at HERE
					//HACK:Commented by HXD 2020-08-08
					/*if (CTRHERE == 1) CTRSECT = 1;
					CTRHERE = 0;
					CTRDOWN = 0;*/


					LPEN[HERE] = PENHERE;

					//reset to 0 all levels above HERE, up to LPREV
					for (I = HERE + 1; I <= LPREV; I++)//HACK:DO I = HERE+1, LPREV IN FORTRAN
					{
						LPEN[I] = 0.0;
					}

					//update LPREV - move it to HERE, no events go higher
					LPREV = HERE;
				}


				//C-----------------------------------------------------------------
				//C             go back to top of big loop
				//C             if PLACED = 0, this event not placed, repeat the loop
				//C             not if PLACED = 1, this event has been placed -- for now
				//C             go to the overall loop and place the next guy
				//C----------------------------------------------------------

				//HACK:Flag Modified by HXD 2020-08-08 # (PLACED == 0)
				//if (PLACED == 1) break; //moved from DO WHILE (().and.()) condition


			}//while HERE>0
			 //C-----------end of big loop ---------
			 //C           the event has been placed
			 //C------------------------------------

		}

	}//End for(INOW=0;INOW<COMMOD9->NEVNT;INOW++)

	//update HSCTSOL, Added by HXD 2020-08-09
	LPREV = 0;

	for (INOW = FirstEVTID; INOW <= LastEVTID; ++INOW)
	{
		currEvent = HPERM[INOW];
		J_X_NEVNT_CURREVENT = J_X_NEVNT + currEvent;

		ITGT = COMMOD9->ISTATIC_0[J_X_NEVNT_CURREVENT];

		/*if (ITGT == -1)
		{
			COMMOD9->HSCTSOL[J_X_NEVNT_CURREVENT] = LPREV;
		}
		else
		{
			LPREV = COMMOD9->HSCTSOL[J_X_NEVNT_CURREVENT];
		}*/

		switch (ITGT)
		{
		case -1:
			COMMOD9->HSCTSOL[J_X_NEVNT_CURREVENT] = LPREV;
			break;
		default:
			LPREV = COMMOD9->HSCTSOL[J_X_NEVNT_CURREVENT];
			break;
		}
	}


	//Set HSCTSOL below
	if(FirstEVTID>0)
		memset(COMMOD9->HSCTSOL + J_X_NEVNT,0, sizeof(int)*FirstEVTID);

	//Set HSCTSOL above
	if (LastEVTID < COMMOD9->NEVNT)
		memset(COMMOD9->HSCTSOL + J_X_NEVNT+ LastEVTID, HERE, sizeof(int)*(COMMOD9->NEVNT-LastEVTID));

	//C-------------------------------------------------------------
	//C-----all events placed, highest at HERE, now calc PENJ
	//CPMS--add up all level penalties to get section penalty (JPEN)
	//C-------------------------------------------------------------
	for (I = 0; I <= HERE; I++)//<=HXD
	{
		PENJ = PENJ + LPEN[I];

		//Helper.Write("LPEN[{0}]={1}\n",I,LPEN[I]);
	}

	/*
	for (int t = 0; t < COMMOD9->NEVNT; t++)
	{
	string FADLAD = (COMMOD9->ISTATIC[t, 0, 1] == 1 ? "FAD" : (COMMOD9->ISTATIC[t, 0, 1] == 2 ? "LAD" : "--"));
	Helper.Write("IROWS[{0},0]={1}[E] ISTATIC[{0},0,0]={2}[L] {3}\n", t, COMMOD9->IROWS[t, 0], COMMOD9->ISTATIC[t, 0, 0], FADLAD);

	}

	Helper.Write("\n");

	for (int t = 0; t < COMMOD9->NEVNT; t++)
	{
	int newp = COMMOD9->INIPERM[t];
	string FADLAD = (COMMOD9->ISTATIC[newp, 0, 1] == 1 ? "FAD" : (COMMOD9->ISTATIC[newp, 0, 1] == 2 ? "LAD" : "--"));
	Helper.Write("IROWS[{0},0]={1}[E] ISTATIC[{0},0,0]={2}[L] {3}\n", newp, COMMOD9->IROWS[newp, 0], COMMOD9->ISTATIC[newp, 0, 0], FADLAD);

	}

	Helper.Write("\n");

	for (I = 0; I < COMMOD9->NEVNT; I++)//<=HXD
	{
	int newp = COMMOD9->INIPERM[I];
	Helper.Write("HSCTSOL[{0},0]={1}\n", newp, COMMOD9->HSCTSOL[newp, 0]);

	//Helper.Write("LPEN[{0}]={1}\n",I,LPEN[I]);
	}

	Helper.Write("\n");

	*/

	//C-------------------------



	//HACK: Commented by HXD 2020-08-08
	//if (JOPT == 0)
	//{
	//	ofstream evtpenFile;
	//	evtpenFile.open("EVENTPEN.TXT");

	//	

	//	for (int e = 0; e <= HERE; e++)
	//	{
	//		/*if(EVEVTPEN[e]>0)
	//		{*/
	//			//int fadlad = (e % 2) == 0 ? 1 : 2;
	//			//evtpenFile << (e+1)/2 << "\t"<< fadlad<<"\t" << EVEVTPEN[e] << "\r\n";
	//			evtpenFile << e << "\t" << LPEN[e] << "\r\n";
	//			
	//		//}
	//	}

	//	evtpenFile.close();
	//}


	COMMOD9->CTRF = CTRSECT;
	CTRSECT = 0;

	delete[] LPEN;

	delete[] SUBPERM;

}

void JSPAN(int XPERM[])
{
	int topx, btmx, J;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	COMMOD9->SPANPEN = 0.0;

	for (J = 0; J < COMMOD9->JEXCL; J++)
	{

		topx = COMMOD9->NEVNT - 1;

		while (COMMOD9->ISTATIC_0[COMMOD9->EXCLSECT[J] + XPERM[topx]] == -1)
		{
			topx = topx - 1;
		}

		btmx = 0;

		while (COMMOD9->ISTATIC_0[COMMOD9->EXCLSECT[J] + XPERM[btmx]] == -1)
		{
			btmx = btmx + 1;
		}

		if (topx > btmx) COMMOD9->SPANPEN = COMMOD9->SPANPEN + (double)(topx - btmx);

	}

	if (COMMOD9->JEXCL > 0)
	{
		COMMOD9->SPANPEN = COMMOD9->SPANPEN / COMMOD9->JEXCL;
	}

}

void SMOOTH(int HPERM[])
{
	double X1, X2, X3, Y1, Y2, Y3;
	int J, K, L;

	J = 0;
	K = 0;
	L = 0;
	X1 = 0.0;
	X2 = 0.0;
	X3 = 0.0;
	Y1 = 0.0;
	Y2 = 0.0;
	Y3 = 0.0;

	COMMOD* COMMOD9 = COMMOD::Singleton();
	COMMOD9->SMPEN = 0.00;

	for (J = 0; J < COMMOD9->NSCT; J++)
	{

		for (K = J + 1; K < COMMOD9->NSCT; K++)
		{

			for (L = 1; L < COMMOD9->NEVNT - 1; L++)
			{

				X1 = COMMOD9->VALEVEL[J][COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[L - 1]]];
				X3 = COMMOD9->VALEVEL[J][COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[L + 1]]];

				if (X1 == X3) continue;

				X2 = COMMOD9->VALEVEL[J][COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[L]]];
				Y1 = COMMOD9->VALEVEL[K][COMMOD9->HSCTSOL[K*COMMOD9->NEVNT + HPERM[L - 1]]];
				Y3 = COMMOD9->VALEVEL[K][COMMOD9->HSCTSOL[K*COMMOD9->NEVNT + HPERM[L + 1]]];

				if (Y1 == Y3) continue;

				Y2 = COMMOD9->VALEVEL[K][COMMOD9->HSCTSOL[K*COMMOD9->NEVNT + HPERM[L]]];

				COMMOD9->SMPEN = COMMOD9->SMPEN +
					sqrt(pow((X2 - X1), 2) + pow((Y2 - Y1), 2)) +
					sqrt(pow((X3 - X2), 2) + pow((Y3 - Y2), 2)) -
					sqrt(pow((X3 - X1), 2) + pow((Y3 - Y1), 2));

			}
		}
	}

}

void SQUEEZE(int HPERM[])
{
	int I, J, Imax, Imin;
	int Itop, Ibtm;

	Imax = 0;
	Imin = 0;
	Itop = 0;
	Ibtm = 0;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	COMMOD9->SQPEN = 0.0;
	for (J = 0; J < COMMOD9->NSCT; J++)
	{

		for (I = 0; I < COMMOD9->NEVNT; I++)
		{
			if (COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT + HPERM[I]] > -1) break;
		}

		Imin = I;

		for (I = COMMOD9->NEVNT - 1; I >= 0; I--)
		{
			if (COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT + HPERM[I]] > -1) break;
		}

		Imax = I;

		for (I = Imax; I >= 0; I--)
		{
			if (COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[I]] < COMMOD9->HLEVEL[J]) break;
		}

		Itop = I + 1;

		for (I = Imin; I < COMMOD9->NEVNT; I++)
		{
			if (COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[I]] > 0) break;
		}

		Ibtm = I - 1;

		for (I = Imin; I <= Ibtm; I++)
		{
			if (!::IORZERO(HPERM[I], J))
			{
				COMMOD9->SQPEN = COMMOD9->SQPEN + max(0, Ibtm - I);
			}
		}

		for (I = Imax; I >= Itop; I--)
		{
			if (!::IORZERO(HPERM[I], J))
			{
				COMMOD9->SQPEN = COMMOD9->SQPEN + max(0, I - Itop);
			}
		}

	}
}

void SHRINK(int HPERM[])
{
	int I, J;
	int Itop, Ibtm;

	Itop = 0;
	Ibtm = 0;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	COMMOD9->SHPEN = 0.0;

	for (J = 0; J < COMMOD9->NSCT; J++)
	{

		for (I = COMMOD9->NEVNT - 1; I >= 0; I--)
		{
			if (COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[I]]< COMMOD9->HLEVEL[J]) break;
		}

		Itop = I;

		for (I = 0; I < COMMOD9->NEVNT; I++)
		{
			if (COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[I]] > 0) break;
		}

		Ibtm = I;

		COMMOD9->SHPEN = COMMOD9->SHPEN + max(0, Itop - Ibtm);

	}

}

bool COEX(int sp1, int sp2, int XPERM[])
{
	int spX, I, Iprm, N;
	bool retVal = false;

	spX = 0;
	N = 0;
	COMMOD* COMMOD9 = COMMOD::Singleton();

	for (I = 0; I < COMMOD9->NEVNT; I++)
	{
		Iprm = COMMOD9->IROWS[XPERM[I]][0];

		if ((Iprm == sp1) || (Iprm == sp2))
		{
			if (COMMOD9->IROWS[XPERM[I]][1] != 3)
			{
				N = N + 1;

				if (N == 1)
				{

					spX = Iprm;
				}
				else if (N == 2)
				{

					if (Iprm != spX) retVal = true;

					break;
				}
			}
		}
	}

	return retVal;
}

void ROYAL(int XPERM[], double& XPEN)
{
	int PPEN = 0;	
	int CurrFAD,CurrLAD,CurrEvt;

	XPEN = 0.0;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	for (int k = 0;k < COMMOD9->NEVNT;k++)
	{
		CurrEvt = XPERM[k];

		//�ж��Ƿ�ΪFAD����Ϊ�ǳɶԵ������¼���ż��ΪFAD

		if (!(XPERM[k] & 1))//ż��,FAD
		{
			CurrFAD = CurrEvt;
			CurrLAD = CurrFAD + 1;

			for (int j = k + 1;;j++)
			{
				if (XPERM[j] == CurrLAD)break;

				if (!(XPERM[j] & 1))PPEN++;
			}
		}	

	}

	XPEN= PPEN - COMMOD9->NCOEX_EXCLUDE;//�ܵĹ����� - ԭ�������� = ROYAL����������
}

bool FB4L(int sp1, int sp2, int XPERM[])
{
	int I, Iprm;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	bool retVal = false;

	for (I = 0; I < COMMOD9->NEVNT; I++)
	{
		Iprm = COMMOD9->IROWS[XPERM[I]][0];

		if ((Iprm == sp1) &&
				(COMMOD9->IROWS[XPERM[I]][1] == 1))
		{
			retVal = true;
			break;
		}

		if ((Iprm == sp2) && (COMMOD9->IROWS[XPERM[I]][1] == 2)) break;

	}

	return retVal;
}

void SEQUEL(int XPERM[], double& XPEN)
{
	int I, M;

	XPEN = 0.0;
	COMMOD* COMMOD9 = COMMOD::Singleton();

	for (I = 0; I < COMMOD9->NEVNT - 1; I++)
	{

		if (COMMOD9->IROWS[I][1] != 1) continue;

		for (M = 0; M < COMMOD9->NEVNT; M++)
		{
			if (COMMOD9->IROWS[M][1] != 2) continue;

			if (COMMOD9->FADLAD[(COMMOD9->IROWS[I][2])*COMMOD9->NSPC + (COMMOD9->IROWS[M][2])] >= 1) continue;

			if (::FB4L(COMMOD9->IROWS[I][0], COMMOD9->IROWS[M][0], XPERM)) XPEN = XPEN + 1;
		}
	}

}

void TEASE(int HPERM[])
{
	int I, J;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	COMMOD9->TSPEN = 0.0;

	if (COMMOD9->STKF == 7) goto Label7000;
	if (COMMOD9->STKF == 8) goto Label8000;
	if (COMMOD9->STKF == 9) goto Label9000;

	for (I = 0; I < COMMOD9->NEVNT; I++)
	{
		for (J = 0; J < COMMOD9->NSCT; J++)
		{

			if ((COMMOD9->IROWS[HPERM[I]][1] > 2) ||
					(COMMOD9->IROWS[HPERM[I]][1] < 1)) continue;

			switch (COMMOD9->HOMEF)
			{

				case 1:

					if (COMMOD9->NEGATIVE[COMMOD9->IROWS[HPERM[I]][2]][J] == 1) continue;
					break;
				case 2:

					if (COMMOD9->NEGATIVE[COMMOD9->IROWS[HPERM[I]][2]][J] > 0) continue;
					break;
			}

			switch (COMMOD9->STKF)
			{
				case 1:

					if ((COMMOD9->IROWS[HPERM[I]][1] == 1) &&
							(COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[I]] == 0))
					{

						if (COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + COMMOD9->IROWS[HPERM[I]][3]] == COMMOD9->HLEVEL[J])
						{
							COMMOD9->TSPEN = COMMOD9->TSPEN + 1;
						}
					}

					break;
				case 2:
				case 4:

					if (COMMOD9->IROWS[HPERM[I]][1] == 1)
					{
						if (COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[I]] < COMMOD9->HLEVEL[J])
						{
							if (COMMOD9->STKF == 4)
							{
								COMMOD9->TSPEN = COMMOD9->TSPEN + (0.5 * COMMOD9->CULLIST[HPERM[I]][2]);
							}
							else
							{
								COMMOD9->TSPEN = COMMOD9->TSPEN + 0.5;
							}
						}
						else
						{

							if (COMMOD9->STKF == 4)
							{
								COMMOD9->TSPEN = COMMOD9->TSPEN - (0.5 * COMMOD9->CULLIST[HPERM[I]][2]);
							}
							else
							{
								COMMOD9->TSPEN = COMMOD9->TSPEN = 0.5;
							}
						}
					}

					else if (COMMOD9->IROWS[HPERM[I]][1] == 2)
					{
						if (COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[I]] > 0)
						{

							if (COMMOD9->STKF == 4)
							{
								COMMOD9->TSPEN = COMMOD9->TSPEN + (0.5 * COMMOD9->CULLIST[HPERM[I]][2]);
							}
							else
							{
								COMMOD9->TSPEN = COMMOD9->TSPEN + 0.5;
							}
						}
						else
						{

							if (COMMOD9->STKF == 4)
							{
								COMMOD9->TSPEN = COMMOD9->TSPEN - (0.5 * COMMOD9->CULLIST[HPERM[I]][2]);
							}
							else
							{
								COMMOD9->TSPEN = COMMOD9->TSPEN = 0.5;
							}
						}
					}

					break;

				case 3:
				case 6:

					if (COMMOD9->IROWS[HPERM[I]][1] == 1)
					{

						if (COMMOD9->STKF == 3)
						{
							COMMOD9->TSPEN = COMMOD9->TSPEN - (double)(COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[I]] + 1);
						}
						else if (COMMOD9->STKF == 6)
						{
							COMMOD9->TSPEN = COMMOD9->TSPEN - (double)(COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[I]] + 1)
								/ (double)(COMMOD9->HLEVEL[J] + 1);
						}
					}

					else if (COMMOD9->IROWS[HPERM[I]][1] == 2)
					{

						if (COMMOD9->STKF == 3)
						{
							COMMOD9->TSPEN = COMMOD9->TSPEN + (double)(COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[I]] + 1);
						}
						else if (COMMOD9->STKF == 6)
						{
							COMMOD9->TSPEN = COMMOD9->TSPEN + (double)(COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[I]] + 1)
								/ (double)(COMMOD9->HLEVEL[J] + 1);
						}
					}

					break;

				case 5:

					if (COMMOD9->IROWS[HPERM[I]][1] == 1)
					{

						if ((COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[I]] != 0) ||
								(COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[COMMOD9->IROWS[I][3]]] != 0))
						{
							COMMOD9->TSPEN = COMMOD9->TSPEN + (double)(COMMOD9->HLEVEL[J] -
									COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[I]]);
						}
					}
					else if (COMMOD9->IROWS[HPERM[I]][1] == 2)
					{

						if ((COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[I]] != COMMOD9->HLEVEL[J]) ||
								(COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[COMMOD9->IROWS[I][3]]] != COMMOD9->HLEVEL[J]))
						{
							COMMOD9->TSPEN = COMMOD9->TSPEN + (double)(COMMOD9->HSCTSOL[J*COMMOD9->NEVNT + HPERM[I]] - 1 + 1);
						}
					}

					break;
			}

		}
	}

Label7000:
	if (COMMOD9->STKF == 7)
	{

		if (COMMOD9->NSPC == 0) goto Label8000;

		for (I = 0; I < COMMOD9->NEVNT; I++)
		{
			if (COMMOD9->IROWS[HPERM[I]][1] == 1)
			{
				COMMOD9->NEGATIVE[COMMOD9->IROWS[HPERM[I]][2]][COMMOD9->NSCT + 2] = I;
			}
			else if (COMMOD9->IROWS[HPERM[I]][1] == 2)
			{
				COMMOD9->NEGATIVE[COMMOD9->IROWS[HPERM[I]][2]][COMMOD9->NSCT + 3] = I;
			}
		}

		for (I = 0; I < COMMOD9->NSPC; I++)
		{
			COMMOD9->TSPEN = COMMOD9->TSPEN +
				COMMOD9->NEGATIVE[I][COMMOD9->NSCT + 3] - COMMOD9->NEGATIVE[I][COMMOD9->NSCT + 2];
		}
	}

Label8000:
	if (COMMOD9->STKF == 8)
	{

		::ROYAL(HPERM, COMMOD9->TSPEN);
	}

Label9000:
	if (COMMOD9->STKF == 9)
	{

		if (COMMOD9->FB4LF != 0) SEQUEL(HPERM, COMMOD9->TSPEN);
	}

}

void TWOPEN(int HPERM[], double& HPEN)
{
	double XPEN = 0.0;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	if (COMMOD9->PENF <= 1)
	{

		if (COMMOD9->KSM > 0.00)
		{
			::SMOOTH(HPERM);
			HPEN = HPEN + (COMMOD9->SMPEN * COMMOD9->KSM);
		}

		if (COMMOD9->KSQ > 0.00)
		{
			::SQUEEZE(HPERM);
			HPEN = HPEN + (COMMOD9->SQPEN * COMMOD9->KSQ);
		}

		if (COMMOD9->KSH > 0.00)
		{
			::SHRINK(HPERM);
			HPEN = HPEN + (COMMOD9->SHPEN * COMMOD9->KSH);
		}

	}

	if (COMMOD9->STKF == 8)
	{
		::ROYAL(HPERM, XPEN);
		COMMOD9->TSPEN = XPEN;
	}
	else if (COMMOD9->STKF == 9)
	{
		if (COMMOD9->FB4LF != 0) SEQUEL(HPERM, XPEN);
		COMMOD9->TSPEN = XPEN;
	}
	else if ((COMMOD9->KTS > 0.00) || (COMMOD9->STKF == 7))
	{
		::TEASE(HPERM);
	}

	if (COMMOD9->KTS > 0.00) HPEN = HPEN + (COMMOD9->TSPEN * COMMOD9->KTS);

}

void GETPEN(int HPERM[], double& HPEN)
{
	int JOPT, CTRGET;
	double PENJ = 0;

	COMMOD* COMMOD9 = COMMOD::Singleton();
	bool exitFlag=false;
	JOPT = 0;
	COMMOD9->CTRF = 0;
	CTRGET = 0;
	HPEN = 0.0;
	COMMOD9->NGHPEN = 0.0;
	::SetVal(COMMOD9->COLPEN, COMMOD9->NSCT, 0.0);

	for (int i = 0; i < COMMOD9->NSCT; i++)
	{

		JOPT = COMMOD9->SectionMaxPenOrder[i];

		::SCTPEN(HPERM, JOPT, PENJ);

		if (COMMOD9->CTRF == 1)
		{
			CTRGET = 1;
		}

		HPEN = HPEN + PENJ;

		COMMOD9->COLPEN[JOPT] = PENJ;

		if (COMMOD9->CRTPEN > 0.0 && HPEN > COMMOD9->CRTPEN)
		{
			exitFlag = true;
			break;
		}
	}

	if (!exitFlag) {
		if (COMMOD9->JSPANF == 1)
		{
			JSPAN(HPERM);
			HPEN = HPEN + COMMOD9->SPANPEN;
		}

		::TWOPEN(HPERM, HPEN);
	}

	COMMOD9->CTRF = CTRGET;

}

void GETPEN_WITH_TWOPEN(int HPERM[], double& HPEN)
{
	int CTRGET;
	double PENJ = 0;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	COMMOD9->CTRF = 0;
	CTRGET = 0;
	HPEN = 0.0;
	COMMOD9->NGHPEN = 0.0;
	::SetVal(COMMOD9->COLPEN, COMMOD9->NSCT, 0.0);
	
	for (int i = 0; i < COMMOD9->NSCT; i++)
	{
		//cout<<"i: "<<i<<endl;
		::SCTPEN(HPERM, i, PENJ);

		if (COMMOD9->CTRF == 1)
		{
			CTRGET = 1;
		}

		HPEN = HPEN + PENJ;

		COMMOD9->COLPEN[i] = PENJ;

	}
	//cout<<"helper1720"<<endl;
	if (COMMOD9->JSPANF == 1)
	{
		JSPAN(HPERM);
		HPEN = HPEN + COMMOD9->SPANPEN;
	}

	::TWOPEN(HPERM, HPEN);

	COMMOD9->CTRF = CTRGET;

}

void DEMPEN(int HPERM[], double& HPEN)
{
	double PENJ;
	int JOPT, SEP, I, K, statij;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	HPEN = 0.0;
	statij = -1;
	::SetVal(COMMOD9->COLPEN, COMMOD9->NSCT, 0.0);

	if (COMMOD9->PENF <= 3 || COMMOD9->PENF == 7)
	{
		for (JOPT = 0; JOPT < COMMOD9->NSCT; JOPT++)
		{
			PENJ = 0.0;

			for (I = 0; I < COMMOD9->NEVNT - 1; I++)
			{

				if (::IORZERO(HPERM[I], JOPT))continue;
				statij = COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[I]];

				for (K = I + 1; K < COMMOD9->NEVNT; K++)
				{

					if (::IORZERO(HPERM[K], JOPT)) continue;

					SEP = (statij)-(COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[K]]);

					if ((COMMOD9->PENF < 3) && (SEP > 0))
					{

						PENJ = PENJ + 1;

					}
					else if ((COMMOD9->PENF == 3) && (SEP > 0))
					{
						PENJ = PENJ + SEP;

					}
					else if ((COMMOD9->PENF == 7) && (SEP > 0))
					{
						PENJ = PENJ + (double)(K - I) / (double)(COMMOD9->NEVNT);
					}
				}
			}
			HPEN = HPEN + PENJ;
			COMMOD9->COLPEN[JOPT] = PENJ;
		}
	}
	else if (COMMOD9->PENF == 4)
	{
		if (COMMOD9->RASCon) goto Label143;

		for (I = 0; I < COMMOD9->NEVNT - 1; I++)
		{
			for (K = I + 1; K < COMMOD9->NEVNT; K++)
			{
				PENJ = 0.0;
				SEP = 0;

				for (JOPT = 0; JOPT < COMMOD9->NSCT; JOPT++)
				{
					if (::IORZERO(HPERM[I], JOPT)) continue;
					if (::IORZERO(HPERM[K], JOPT)) continue;

					SEP = SEP + 1;

					if (COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[I]] >
							COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[K]])
					{
						PENJ = PENJ + 1;
						COMMOD9->COLPEN[JOPT] = COMMOD9->COLPEN[JOPT] + 1;
					}
				}

				if (SEP > 0) HPEN = HPEN + (PENJ / (double)SEP);
			}
		}

		goto Label144;

Label143:

		for (I = 0; I < COMMOD9->NEVNT - 1; I++)
		{
			for (K = I + 1; K < COMMOD9->NEVNT; K++)
			{

				if (COMMOD9->RASC[HPERM[K]][HPERM[I]] <= 0) continue;

				if (COMMOD9->RASC[HPERM[I]][HPERM[K]] +
						COMMOD9->RASC[HPERM[K]][HPERM[I]] <= 0) continue;

				HPEN = HPEN + (double)(COMMOD9->RASC[HPERM[K]][HPERM[I]]) /
					(double)(COMMOD9->RASC[HPERM[I]][HPERM[K]] +
							COMMOD9->RASC[HPERM[K]][HPERM[I]]);
			}
		}

Label144:
		;
	}
	else if (COMMOD9->PENF == 5)
	{
		::ROYAL(HPERM, HPEN);
	}
	else if (COMMOD9->PENF == 6)
	{

		if (COMMOD9->FB4LF != 0) ::SEQUEL(HPERM, HPEN);
	}

	::JSPAN(HPERM);

	if (COMMOD9->JSPANF == 1) HPEN = HPEN + COMMOD9->SPANPEN;

	::TWOPEN(HPERM, HPEN);

}

void DOTOO()
{
	COMMOD* COMMOD9 = COMMOD::Singleton();

	COMMOD9->TOOPEN = (COMMOD9->SMPEN * COMMOD9->KSM) +
		(COMMOD9->SQPEN * COMMOD9->KSQ) +
		(COMMOD9->SHPEN * COMMOD9->KSH) +
		(COMMOD9->TSPEN * COMMOD9->KTS);

	COMMOD9->TO2PEN = 0.0;
	if (COMMOD9->KTS > 0.0) COMMOD9->TO2PEN = COMMOD9->TO2PEN + COMMOD9->TSPEN;
	if (COMMOD9->KSM > 0.0) COMMOD9->TO2PEN = COMMOD9->TO2PEN + COMMOD9->SMPEN;
	if (COMMOD9->KSQ > 0.0) COMMOD9->TO2PEN = COMMOD9->TO2PEN + COMMOD9->SQPEN;
	if (COMMOD9->KSH > 0.0) COMMOD9->TO2PEN = COMMOD9->TO2PEN + COMMOD9->SHPEN;

}

void STEPOUT(int myperm[]) {};

void TRAJOUT() {};

void NABRLIMT(int IRANK, int& RANKUP, int& RANKDN)
{
	int RANK, RTYPE, INUMB;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	RANK = 0;  
	RANKUP = 0;  
	RANKDN = 0;  
	RTYPE = 0;  
	INUMB = 0;

	INUMB = COMMOD9->IROWS[COMMOD9->PERM[IRANK]][0];

	if (COMMOD9->IROWS[COMMOD9->PERM[IRANK]][1] == 1)
	{

		if (COMMOD9->SOLVEF != 8)
		{
			RANK = 0;
			RANKDN = 0;
			RANKUP = COMMOD9->NEVNT - 1;
		}
		else 
		{

			RANK = IRANK;
			RANKDN = IRANK;
			RANKUP = COMMOD9->NEVNT - 1;
		}

		RTYPE = COMMOD9->IROWS[COMMOD9->PERM[RANK]][1];

		while (!((COMMOD9->IROWS[COMMOD9->PERM[RANK]][0] == INUMB) && (RTYPE != 1)))
		{

			if (RANK < IRANK)
			{

				if (COMMOD9->NOTHR > 0)
				{
					if ((RTYPE > 0))
					{
						if ((RTYPE > 2) || (RTYPE < 1))
						{
							if (COMMOD9->PTLORD[COMMOD9->PERM[IRANK]][COMMOD9->PERM[RANK]] == 2)
							{

								RANKDN = RANK + 1;
							}
						}
					}
				}
			}

			else if (RANK > IRANK)
			{

				if (COMMOD9->NOTHR > 0)
				{
					if ((RTYPE > 2) || (RTYPE < 1))
					{
						if (COMMOD9->PTLORD[COMMOD9->PERM[IRANK]][COMMOD9->PERM[RANK]] == 1)
						{
							break;
						}
					}

				}

				else if (COMMOD9->NSPC > 0)
				{
					if (COMMOD9->COXSTF != 4)
					{
						if (COMMOD9->IROWS[COMMOD9->PERM[RANK]][1] == 2)
						{
							if (COMMOD9->COEXST[(COMMOD9->IROWS[COMMOD9->PERM[IRANK]][2])*COMMOD9->NSPC + (COMMOD9->IROWS[COMMOD9->PERM[RANK]][2])]
									>= COMMOD9->COXSTF)
							{
								break;
							}

							if (COMMOD9->FB4LF != 0)
							{
								if (COMMOD9->FADLAD[(COMMOD9->IROWS[COMMOD9->PERM[IRANK]][2])*COMMOD9->NSPC + (COMMOD9->IROWS[COMMOD9->PERM[RANK]][2])] > 0)
								{
									break;
								}
							}
						}
					}
				}
			}

			RANK = RANK + 1;
			RTYPE = COMMOD9->IROWS[COMMOD9->PERM[RANK]][1];
		}

		RANKUP = RANK - 1;
	}

	else if (COMMOD9->IROWS[COMMOD9->PERM[IRANK]][1] == 11)
	{

		if (COMMOD9->SOLVEF != 8)
		{
			RANK = 0;
			RANKDN = 0;
			RANKUP = COMMOD9->NEVNT - 1;
		}
		else
		{
			RANK = IRANK;
			RANKDN = IRANK;
			RANKUP = COMMOD9->NEVNT - 1;
		}

		RTYPE = COMMOD9->IROWS[COMMOD9->PERM[RANK]][1];

		while (!((COMMOD9->IROWS[COMMOD9->PERM[RANK]][0] == INUMB) && (RTYPE != 11)))
		{
			if (RANK < IRANK)
			{

				if (COMMOD9->NOTHR > 0)
				{
					if (((RTYPE > 2) || (RTYPE < 1)) &&
							(COMMOD9->PTLORD[COMMOD9->PERM[IRANK]][COMMOD9->PERM[RANK]] == 2))
					{

						RANKDN = RANK + 1;
					}
				}
			}
			else if (RANK > IRANK)
			{

				if (COMMOD9->NOTHR > 0)
				{
					if (((RTYPE > 2) || (RTYPE < 1)) &&
							(COMMOD9->PTLORD[COMMOD9->PERM[IRANK]][COMMOD9->PERM[RANK]] == 1))
					{
						break;
					}
				}
			}

			RANK = RANK + 1;
			RTYPE = COMMOD9->IROWS[COMMOD9->PERM[RANK]][1];

		}

		RANKUP = RANK - 1;
	}
	else if ((COMMOD9->IROWS[COMMOD9->PERM[IRANK]][1]) == 2)
	{

		if (COMMOD9->SOLVEF != 8)
		{
			RANK = COMMOD9->NEVNT - 1;
			RANKUP = COMMOD9->NEVNT - 1;
			RANKDN = 0;
		}
		else
		{
			RANK = IRANK;
			RANKUP = IRANK;
			RANKDN = 0;
		}

		RTYPE = COMMOD9->IROWS[COMMOD9->PERM[RANK]][1];

		while (!((COMMOD9->IROWS[COMMOD9->PERM[RANK]][0] == INUMB) && (RTYPE != 2)))
		{
			if (RANK > IRANK)
			{

				if (COMMOD9->NOTHR > 0)
				{
					if ((RTYPE > 2) || (RTYPE < 1))
					{
						if (COMMOD9->PTLORD[COMMOD9->PERM[IRANK]][COMMOD9->PERM[RANK]] == 1)
						{
							RANKUP = RANK - 1;
						}
					}
				}
			}
			else if (RANK < IRANK)
			{

				if (COMMOD9->NOTHR > 0)
				{
					if ((RTYPE > 2) || (RTYPE < 1))
					{
						if (COMMOD9->PTLORD[COMMOD9->PERM[IRANK]][COMMOD9->PERM[RANK]] == 2)
						{
							break;
						}
					}
				}

				if (COMMOD9->NSPC > 0)
				{
					if (COMMOD9->COXSTF != 4)
					{
						if (COMMOD9->IROWS[COMMOD9->PERM[RANK]][1] == 1)
						{
							if (COMMOD9->COEXST[(COMMOD9->IROWS[COMMOD9->PERM[RANK]][2])*COMMOD9->NSPC + (COMMOD9->IROWS[COMMOD9->PERM[IRANK]][2])] >= COMMOD9->COXSTF)
							{
								break;
							}

							if (COMMOD9->FB4LF != 0)
							{
								if (COMMOD9->FADLAD[(COMMOD9->IROWS[COMMOD9->PERM[RANK]][2])*COMMOD9->NSPC + (COMMOD9->IROWS[COMMOD9->PERM[IRANK]][2])] > 0)
								{
									break;
								}
							}
						}
					}
				}
			}

			RANK = RANK - 1;
			RTYPE = COMMOD9->IROWS[COMMOD9->PERM[RANK]][1];

		}

		RANKDN = RANK + 1;
	}
	else if ((COMMOD9->IROWS[COMMOD9->PERM[IRANK]][1]) == 12)
	{

		if (COMMOD9->SOLVEF != 8)
		{
			RANK = COMMOD9->NEVNT - 1;
			RANKUP = COMMOD9->NEVNT - 1;
			RANKDN = 0;
		}
		else
		{
			RANK = IRANK - 1;
			RANKUP = IRANK - 1;
			RANKDN = 0;
		}

		RTYPE = COMMOD9->IROWS[COMMOD9->PERM[RANK]][1];

		while (!((COMMOD9->IROWS[COMMOD9->PERM[RANK]][0] == INUMB) && (RTYPE != 12)))
		{
			if (RANK > IRANK)
			{

				if (COMMOD9->NOTHR > 0)
				{
					if (((RTYPE > 2) || (RTYPE < 1)) &&
							(COMMOD9->PTLORD[COMMOD9->PERM[IRANK]][COMMOD9->PERM[RANK]] == 1))
					{
						RANKUP = RANK - 1;
					}
				}
			}
			else if (RANK < IRANK)
			{

				if (COMMOD9->NOTHR > 0)
				{
					if (((RTYPE > 2) || (RTYPE < 1)) &&
							(COMMOD9->PTLORD[COMMOD9->PERM[IRANK]][COMMOD9->PERM[RANK]] == 2))
					{
						break;
					}
				}
			}

			RANK = RANK - 1;
			RTYPE = COMMOD9->IROWS[COMMOD9->PERM[RANK]][1];

		}

		RANKDN = RANK + 1;

	}
	else if (COMMOD9->IROWS[COMMOD9->PERM[IRANK]][1] == 3)
	{

		for (RANKDN = IRANK - 1; RANKDN >= 1; RANKDN--)
		{
			if (COMMOD9->IROWS[COMMOD9->PERM[RANKDN - 1]][0] == INUMB) break;
		}

		for (RANKUP = IRANK - 1; RANKUP < COMMOD9->NEVNT - 1; RANKUP++)
		{
			if (COMMOD9->IROWS[COMMOD9->PERM[RANKUP + 1]][0] == INUMB) break;
		}

	}
	else if ((COMMOD9->IROWS[COMMOD9->PERM[IRANK]][1] > 3) ||
			(COMMOD9->IROWS[COMMOD9->PERM[IRANK]][1] < 1))
	{

		RANK = 0;
		RANKUP = COMMOD9->NEVNT - 1;
		RANKDN = 0;

		while (RANK <= COMMOD9->NEVNT - 1)
		{
			if (RANK < IRANK)
			{

				if (COMMOD9->PTLORD[COMMOD9->PERM[RANK]][COMMOD9->PERM[IRANK]] == 1)
				{
					RANKDN = RANK + 1;
				}
			}
			else if (RANK > IRANK)
			{

				if (COMMOD9->PTLORD[COMMOD9->PERM[RANK]][COMMOD9->PERM[IRANK]] == 2)
				{
					RANKUP = RANK - 1;
					break;
				}
			}

			RANK = RANK + 1;

		}

	}

}

void SMLNABR(int& IRANK)
{
	int RANKUP, RANKDN;

	RANKUP = 0;
	RANKDN = 0;
	IRANK = 0;

	COMMOD* COMMOD9 = COMMOD::Singleton();

Label1000:
	IRANK = int(random(0, COMMOD9->NEVNT - 1));

	if (COMMOD9->FIXF > 0)
	{
		if (COMMOD9->PERM[IRANK] == COMMOD9->XEVNT) goto Label1000;
	}

	if (COMMOD9->FIXF == 3)
	{
		if (COMMOD9->PERM[IRANK + 1] == COMMOD9->XEVNT) goto Label1000;
	}

	RANKUP = IRANK + 1;
	RANKDN = IRANK;

	::NABRLIMT(IRANK, RANKUP, RANKDN);

	if ((IRANK + 1) > RANKUP) goto Label1000;

	goto Label2000;

	if ((COMMOD9->IROWS[COMMOD9->PERM[IRANK]][1]) ==
			(COMMOD9->IROWS[COMMOD9->PERM[IRANK + 1]][1])) goto Label1000;

	if (COMMOD9->NOTHR == 0) goto Label2000; 

	if (((COMMOD9->IROWS[COMMOD9->PERM[IRANK]][1]) == 2) &&
			((COMMOD9->IROWS[COMMOD9->PERM[IRANK + 1]][1]) == 2))
	{
		if ((COMMOD9->NSPC > 0) && (COMMOD9->COXSTF != 4))
		{
			if (COMMOD9->COEXST[(COMMOD9->IROWS[COMMOD9->PERM[IRANK]][2])*COMMOD9->NSPC + (COMMOD9->IROWS[COMMOD9->PERM[IRANK + 1]][2])]
					>= COMMOD9->COXSTF) goto Label1000;
		}
	}

	if (((COMMOD9->IROWS[COMMOD9->PERM[IRANK + 1]][1] > 2) ||
				(COMMOD9->IROWS[COMMOD9->PERM[IRANK + 1]][1] < 1)) &&
			(COMMOD9->PTLORD[COMMOD9->PERM[IRANK]][COMMOD9->PERM[IRANK + 1]] == 1))
	{
		goto Label1000;
	}

	if (((COMMOD9->IROWS[COMMOD9->PERM[IRANK]][1] > 2) ||
				(COMMOD9->IROWS[COMMOD9->PERM[IRANK]][1] < 1)) &&
			(COMMOD9->PTLORD[COMMOD9->PERM[IRANK + 1]][COMMOD9->PERM[IRANK]] == 2))
	{
		goto Label1000;
	}

Label2000:
	return;

}

void SMLNABR2(int& IRANK, int count, int last, int &first, int C)
{
	int RANKUP, RANKDN;

	RANKUP = 0;
	RANKDN = 0;
	IRANK = 0;

	COMMOD* COMMOD9 = COMMOD::Singleton();

Label1000:
	if (count%C == 0) {
		IRANK = int(random(0, COMMOD9->NEVNT - 1));
		first = IRANK;
	}
	else {
		IRANK = last;
	}

	if (COMMOD9->FIXF > 0)
	{
		if (COMMOD9->PERM[IRANK] == COMMOD9->XEVNT) goto Label1000;
	}

	if (COMMOD9->FIXF == 3)
	{
		if (COMMOD9->PERM[IRANK + 1] == COMMOD9->XEVNT) goto Label1000;
	}

	RANKUP = IRANK + 1;
	RANKDN = IRANK;

	::NABRLIMT(IRANK, RANKUP, RANKDN);

	if ((IRANK + 1) > RANKUP) goto Label1000;

	goto Label2000;

	if ((COMMOD9->IROWS[COMMOD9->PERM[IRANK]][1]) ==
			(COMMOD9->IROWS[COMMOD9->PERM[IRANK + 1]][1])) goto Label1000;

	if (COMMOD9->NOTHR == 0) goto Label2000; 

	if (((COMMOD9->IROWS[COMMOD9->PERM[IRANK]][1]) == 2) &&
			((COMMOD9->IROWS[COMMOD9->PERM[IRANK + 1]][1]) == 2))
	{
		if ((COMMOD9->NSPC > 0) && (COMMOD9->COXSTF != 4))
		{
			if (COMMOD9->COEXST[(COMMOD9->IROWS[COMMOD9->PERM[IRANK]][2])*COMMOD9->NSPC + (COMMOD9->IROWS[COMMOD9->PERM[IRANK + 1]][2])]
					>= COMMOD9->COXSTF) goto Label1000;
		}
	}

	if (((COMMOD9->IROWS[COMMOD9->PERM[IRANK + 1]][1] > 2) ||
				(COMMOD9->IROWS[COMMOD9->PERM[IRANK + 1]][1] < 1)) &&
			(COMMOD9->PTLORD[COMMOD9->PERM[IRANK]][COMMOD9->PERM[IRANK + 1]] == 1))
	{
		goto Label1000;
	}

	if (((COMMOD9->IROWS[COMMOD9->PERM[IRANK]][1] > 2) ||
				(COMMOD9->IROWS[COMMOD9->PERM[IRANK]][1] < 1)) &&
			(COMMOD9->PTLORD[COMMOD9->PERM[IRANK + 1]][COMMOD9->PERM[IRANK]] == 2))
	{
		goto Label1000;
	}

Label2000:
	return;

}

void BIGNABR(int& IRANK, int& JRANK)
{
	int RANKUP, RANKDN, I, K;

	int BigEvent = 0;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	I = 0;
	K = 0;
	RANKUP = COMMOD9->NEVNT - 1; 
	RANKDN = 0;    

Label1000:
	if ((COMMOD9->NUDGER) && (COMMOD9->NUDGUP || COMMOD9->NUDGDN))
	{

		IRANK = COMMOD9->NEWRANK;

	}
	else
	{

		IRANK = int(random(0, COMMOD9->NEVNT));

	}

	if (((COMMOD9->FIXF == 4) || (COMMOD9->FIXF == 5)) && (int(random(0, 9)) != 1))
	{
		for (I = 0; I < COMMOD9->NEVNT; I++)
		{
			if ((COMMOD9->PERM[I] == COMMOD9->XEVNT) ||
					(COMMOD9->PERM[I] == COMMOD9->YEVNT))
			{
				if (int(random(0, 2)) == 1)
				{
					IRANK = I;
				}
				else
				{
					for (K = I + 1; K < COMMOD9->NEVNT; K++)
					{
						if ((COMMOD9->PERM[K] == COMMOD9->XEVNT) ||
								(COMMOD9->PERM[K] == COMMOD9->YEVNT))
						{
							IRANK = K;
							break;
						}
					}
				}

				break;
			}

		}
	}

	if ((COMMOD9->PERM[IRANK] == COMMOD9->XEVNT) &&
			(COMMOD9->FIXF > 0) && (COMMOD9->FIXF < 4)) goto Label1000;

	::NABRLIMT(IRANK, RANKUP, RANKDN);

	if ((COMMOD9->FIXF == 4) && (COMMOD9->PERM[IRANK] == COMMOD9->XEVNT))
	{

		if (COMMOD9->IROWS[COMMOD9->XEVNT][1] == 1) JRANK = RANKUP;
		if (COMMOD9->IROWS[COMMOD9->XEVNT][1] == 2) JRANK = RANKDN;
	}
	else if ((COMMOD9->FIXF == 4) && (COMMOD9->PERM[IRANK] == COMMOD9->YEVNT))
	{

		if (COMMOD9->IROWS[COMMOD9->YEVNT][1] == 1) JRANK = RANKDN;
		if (COMMOD9->IROWS[COMMOD9->YEVNT][1] == 2) JRANK = RANKUP;
	}
	else if (COMMOD9->FIXF == 5)
	{

		if (int(random(0, 2)) == 1)
		{
			JRANK = RANKUP;
		}
		else
		{
			JRANK = RANKDN;
		}
	}
	else
	{
		if ((COMMOD9->NUDGER) && (COMMOD9->NUDGUP || COMMOD9->NUDGDN))
		{

			if (COMMOD9->NUDGUP)
			{

				if (RANKUP - IRANK>0)
					JRANK = IRANK + int(random(0, RANKUP - IRANK));
				else
					JRANK = IRANK;
			}
			else if (COMMOD9->NUDGDN)
			{

				if (IRANK - RANKDN>0)
					JRANK = RANKDN + int(random(0, IRANK - RANKDN));
				else
					JRANK = RANKDN;
			}
		}
		else
		{

			if (RANKUP - RANKDN + 1>0)
				JRANK = RANKDN + int(random(0, RANKUP - RANKDN + 1));
			else
				JRANK = RANKDN;
		}
	}

	if (COMMOD9->FIXF == 3)
	{
		if (COMMOD9->PERM[JRANK] == COMMOD9->XEVNT) goto Label1000;
		if ((IRANK > COMMOD9->XPOSN) && (JRANK < COMMOD9->XPOSN)) goto Label1000;
		if ((IRANK < COMMOD9->XPOSN) && (JRANK > COMMOD9->XPOSN)) goto Label1000;
	}

	if ((IRANK == JRANK) ||
			((COMMOD9->IROWS[COMMOD9->PERM[IRANK]][0]) == (COMMOD9->IROWS[COMMOD9->PERM[JRANK]][0])))
	{
		if (COMMOD9->NUDGER)
		{
			COMMOD9->NUDGUP = false;
			COMMOD9->NUDGDN = false;
		}

		goto Label1000;
	}

	if (COMMOD9->NUDGER)
	{

		if (JRANK > IRANK) COMMOD9->NUDGUP = true;
		if (JRANK < IRANK) COMMOD9->NUDGDN = true;

		COMMOD9->NEWRANK = JRANK;

	}

}

void BIGNABR2(int& IRANK, int& JRANK, int count, int last, int& first, int C)
{
	int RANKUP, RANKDN, I, K;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	I = 0;
	K = 0;
	RANKUP = COMMOD9->NEVNT - 1; 
	RANKDN = 0;    

Label1000:
	if ((COMMOD9->NUDGER) && (COMMOD9->NUDGUP || COMMOD9->NUDGDN))
	{

		IRANK = COMMOD9->NEWRANK;
		cout << "wrong" << endl;
	}
	else
	{
		IRANK = int(random(0, COMMOD9->NEVNT));
		first = IRANK;
	}

	if (((COMMOD9->FIXF == 4) || (COMMOD9->FIXF == 5)) && (int(random(0, 9)) != 1))
	{
		for (I = 0; I < COMMOD9->NEVNT; I++)
		{
			if ((COMMOD9->PERM[I] == COMMOD9->XEVNT) ||
					(COMMOD9->PERM[I] == COMMOD9->YEVNT))
			{
				if (int(random(0, 2)) == 1)
				{
					IRANK = I;
				}
				else
				{
					for (K = I + 1; K < COMMOD9->NEVNT; K++)
					{
						if ((COMMOD9->PERM[K] == COMMOD9->XEVNT) ||
								(COMMOD9->PERM[K] == COMMOD9->YEVNT))
						{
							IRANK = K;
							break;
						}
					}
				}

				break;
			}

		}
	}

	if ((COMMOD9->PERM[IRANK] == COMMOD9->XEVNT) &&
			(COMMOD9->FIXF > 0) && (COMMOD9->FIXF < 4)) goto Label1000;

	::NABRLIMT(IRANK, RANKUP, RANKDN);

	if ((COMMOD9->FIXF == 4) && (COMMOD9->PERM[IRANK] == COMMOD9->XEVNT))
	{

		if (COMMOD9->IROWS[COMMOD9->XEVNT][1] == 1) JRANK = RANKUP;
		if (COMMOD9->IROWS[COMMOD9->XEVNT][1] == 2) JRANK = RANKDN;
	}
	else if ((COMMOD9->FIXF == 4) && (COMMOD9->PERM[IRANK] == COMMOD9->YEVNT))
	{

		if (COMMOD9->IROWS[COMMOD9->YEVNT][1] == 1) JRANK = RANKDN;
		if (COMMOD9->IROWS[COMMOD9->YEVNT][1] == 2) JRANK = RANKUP;
	}
	else if (COMMOD9->FIXF == 5)
	{

		if (int(random(0, 2)) == 1)
		{
			JRANK = RANKUP;
		}
		else
		{
			JRANK = RANKDN;
		}
	}
	else
	{
		if ((COMMOD9->NUDGER) && (COMMOD9->NUDGUP || COMMOD9->NUDGDN))
		{

			if (COMMOD9->NUDGUP)
			{

				if (RANKUP - IRANK>0)
					JRANK = IRANK + int(random(0, RANKUP - IRANK));
				else
					JRANK = IRANK;
			}
			else if (COMMOD9->NUDGDN)
			{

				if (IRANK - RANKDN>0)
					JRANK = RANKDN + int(random(0, IRANK - RANKDN));
				else
					JRANK = RANKDN;
			}
		}
		else
		{

			if (RANKUP - RANKDN + 1>0)
				JRANK = RANKDN + int(random(0, RANKUP - RANKDN + 1));
			else
				JRANK = RANKDN;
		}
	}

	if (COMMOD9->FIXF == 3)
	{
		if (COMMOD9->PERM[JRANK] == COMMOD9->XEVNT) goto Label1000;
		if ((IRANK > COMMOD9->XPOSN) && (JRANK < COMMOD9->XPOSN)) goto Label1000;
		if ((IRANK < COMMOD9->XPOSN) && (JRANK > COMMOD9->XPOSN)) goto Label1000;
	}

	if ((IRANK == JRANK) ||
			((COMMOD9->IROWS[COMMOD9->PERM[IRANK]][0]) == (COMMOD9->IROWS[COMMOD9->PERM[JRANK]][0])))
	{
		if (COMMOD9->NUDGER)
		{
			COMMOD9->NUDGUP = false;
			COMMOD9->NUDGDN = false;
		}

		goto Label1000;
	}

	if (COMMOD9->NUDGER)
	{

		if (JRANK > IRANK) COMMOD9->NUDGUP = true;
		if (JRANK < IRANK) COMMOD9->NUDGDN = true;

		COMMOD9->NEWRANK = JRANK;

	}

}

void TWONABR(int& IRANK, int& JRANK)
{
	int KRANK, RANK, RANKDN, RANKUP;
	int TYPER;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	RANKDN = 0;
	RANKUP = 0;
	IRANK = 0;
	JRANK = 0;
	KRANK = 0;
	RANK = 0;
	TYPER = 0;

Label1000:
	IRANK = int(random(0, COMMOD9->NEVNT));
	if (COMMOD9->FIXF > 0)
	{
		if (COMMOD9->PERM[IRANK] == COMMOD9->XEVNT) goto Label1000;
	}

	RANKUP = COMMOD9->NEVNT - 1;
	RANKDN = 1 - 1;

	::NABRLIMT(IRANK, RANKUP, RANKDN);

	if (RANKUP == RANKDN) goto Label1000;

Label2000:

	JRANK = RANKDN + int(random(0, RANKUP - RANKDN + 1)) - 1;
	if (IRANK == JRANK) goto Label2000;

	if (COMMOD9->FIXF == 3)
	{
		if (COMMOD9->PERM[JRANK] == COMMOD9->XEVNT) goto Label1000;
	}

	if (IRANK == JRANK) goto Label1000;
	if (COMMOD9->FIXF > 0)
	{
		if (COMMOD9->PERM[JRANK] == COMMOD9->XEVNT) goto Label1000;
	}

	RANKUP = max(IRANK, JRANK);
	RANKDN = min(IRANK, JRANK);
	::NABRLIMT(JRANK, RANKUP, RANKDN);
	if (JRANK < IRANK)
	{

		if (IRANK > RANKUP) goto Label1000;
	}
	else if (JRANK > IRANK)
	{

		if (IRANK < RANKDN) goto Label1000;
	}

	goto Label3000;

	if (JRANK < IRANK)
	{

		for (RANK = JRANK; RANK < IRANK; RANK++)
		{
			JRANK = RANK;
			TYPER = COMMOD9->IROWS[COMMOD9->PERM[RANK]][1];
			for (KRANK = RANK + 1; KRANK < IRANK; KRANK++)
			{
				if ((TYPER == 1) || (TYPER == 3))
				{
					if (COMMOD9->IROWS[COMMOD9->PERM[KRANK]][0] ==
							COMMOD9->IROWS[COMMOD9->PERM[RANK]][0])
					{

						JRANK = RANK + 1;
						break;
					}
					else if ((COMMOD9->COXSTF != 4) && (COMMOD9->IROWS[COMMOD9->PERM[KRANK]][1] == 2))
					{
						if (COMMOD9->COEXST[(COMMOD9->IROWS[COMMOD9->PERM[RANK]][2])*COMMOD9->NSPC + (COMMOD9->IROWS[COMMOD9->PERM[KRANK]][2])] >= COMMOD9->COXSTF)
						{

							JRANK = RANK + 1;
							break;
						}
					}
					else if (COMMOD9->NOTHR > 0)
					{
						if ((COMMOD9->IROWS[COMMOD9->PERM[KRANK]][1] > 2) ||
								(COMMOD9->IROWS[COMMOD9->PERM[KRANK]][1] < 1))
						{
							if (COMMOD9->PTLORD[COMMOD9->PERM[RANK]][COMMOD9->PERM[KRANK]] == 1)
							{

								JRANK = RANK + 1;
								break;
							}
						}
					}
				}
				else if ((TYPER > 2) || (TYPER < 1))
				{
					if (COMMOD9->PTLORD[COMMOD9->PERM[KRANK]][COMMOD9->PERM[KRANK]] == 2)
					{

						JRANK = RANK + 1;
						break;
					}
				}
			}

			if (JRANK == RANK)
			{

				break;
			}
		}
	}
	else if (JRANK > IRANK)
	{

		for (RANK = JRANK; RANK > IRANK; RANK--)
		{
			JRANK = RANK;
			TYPER = COMMOD9->IROWS[COMMOD9->PERM[RANK]][1];
			for (KRANK = RANK - 1; KRANK > IRANK; KRANK--)
			{
				if ((TYPER == 2) || (TYPER == 3))
				{
					if (COMMOD9->IROWS[COMMOD9->PERM[KRANK]][0] ==
							COMMOD9->IROWS[COMMOD9->PERM[RANK]][0])
					{

						JRANK = RANK - 1;
						break;
					}
					else if ((COMMOD9->COXSTF != 4) && (COMMOD9->IROWS[COMMOD9->PERM[KRANK]][1] == 1))
					{
						if (COMMOD9->COEXST[(COMMOD9->IROWS[COMMOD9->PERM[RANK]][2])*COMMOD9->NSPC + (COMMOD9->IROWS[COMMOD9->PERM[KRANK]][2])] >= COMMOD9->COXSTF)
						{

							JRANK = RANK - 1;
							break;
						}
					}
					else if (COMMOD9->NOTHR > 0)
					{
						if ((COMMOD9->IROWS[COMMOD9->PERM[KRANK]][1] > 2) ||
								(COMMOD9->IROWS[COMMOD9->PERM[KRANK]][1] < 1))
						{
							if (COMMOD9->PTLORD[COMMOD9->PERM[RANK]][COMMOD9->PERM[KRANK]] == 2)
							{

								JRANK = RANK - 1;
								break;
							}
						}
					}
				}
				else if ((TYPER > 3) || (TYPER < 1))
				{
					if (COMMOD9->PTLORD[COMMOD9->PERM[KRANK]][COMMOD9->PERM[RANK]] == 1)
					{

						JRANK = RANK - 1;
						break;
					}
				}
			}

			if (JRANK == RANK)
			{

				break;
			}

		}

	}

	if (IRANK == JRANK) goto Label1000;

	if (COMMOD9->FIXF == 3)
	{
		if (COMMOD9->PERM[JRANK] == COMMOD9->XEVNT) goto Label1000;
	}

Label3000:

	return;

}

void GONABR(int& IRANK, int& JRANK, int count, int last, int& first, int C)
{
	int RANK = 0, ITEMP = 0;

	IRANK = 0; 
	JRANK = 0; 

	COMMOD* COMMOD9 = COMMOD::Singleton();

	if (COMMOD9->NABRGEN == 2)
	{

		::SMLNABR2(IRANK, count, last, first, C);
		JRANK = IRANK + 1;
		::CopyArray(COMMOD9->PERM, COMMOD9->NEVNT, COMMOD9->LSTPERM);
		ITEMP = COMMOD9->PERM[IRANK];
		COMMOD9->PERM[IRANK] = COMMOD9->PERM[IRANK + 1];
		COMMOD9->PERM[IRANK + 1] = ITEMP;
	}

	if (COMMOD9->NABRGEN == 1)
	{

		::BIGNABR2(IRANK, JRANK, count, last, first, C);

		::CopyArray(COMMOD9->PERM, COMMOD9->NEVNT, COMMOD9->LSTPERM);
		RANK = IRANK;
		ITEMP = COMMOD9->PERM[IRANK];

		if (IRANK < JRANK)
		{
			while (RANK < JRANK)
			{
				COMMOD9->PERM[RANK] = COMMOD9->PERM[RANK + 1];
				RANK = RANK + 1;
			}
		}

		else if (IRANK > JRANK)
		{
			while (RANK > JRANK)
			{
				COMMOD9->PERM[RANK] = COMMOD9->PERM[RANK - 1];
				RANK = RANK - 1;
			}
		}

		COMMOD9->PERM[JRANK] = ITEMP;

	}

	if (COMMOD9->NABRGEN == 3)
	{
		::TWONABR(IRANK, JRANK);

		::CopyArray(COMMOD9->PERM, COMMOD9->NEVNT, COMMOD9->LSTPERM);
		ITEMP = COMMOD9->PERM[IRANK];
		COMMOD9->PERM[IRANK] = COMMOD9->PERM[JRANK];
		COMMOD9->PERM[JRANK] = ITEMP;
	}

}

void RESCTPEN(int RNK, int RNK2, int HPERM[], int JOPT, double& PENJ)
{
	COMMOD* COMMOD9 = COMMOD::Singleton();
	int H_LEVEL = COMMOD9->HLEVEL[JOPT] + 1;
	double* LPEN = new double[COMMOD9->MXLVL];
	double PENDOWN, PENHERE;
	int DOWN, HERE, INOW, ITGT, LPREV, PLACED, THISONE;

	int I, RR;

	int JOPT_X_EVT_X_LEVEL_ADD_DOWN = 0;
	int IDX_EPEN = IDX_SCT_EPEN[JOPT];
	int JOPT_X_NEVENTS = JOPT*COMMOD9->NEVNT;

	int LSTEvtID, HEvtID, temEvtID;

	LSTEvtID = 0;
	HEvtID = 0;
	I = 0;
	RR = 0;
	ITGT = 0;
	HERE = 0;
	DOWN = 0;
	THISONE = 0;
	PLACED = 0;
	PENJ = 0.0;

	::SetVal(LPEN, COMMOD9->MXLVL, 0.0);
	PENDOWN = 0.0;
	PENHERE = 0.0;

	INOW = 0;
	LPREV = 0;

	if (RNK > 0)
	{

		for (INOW = 0; INOW <= RNK; INOW++)
		{
			LSTEvtID = COMMOD9->LSTPERM[INOW];

			LPREV = COMMOD9->SCTSOL[JOPT_X_NEVENTS + LSTEvtID];

			COMMOD9->HSCTSOL[JOPT_X_NEVENTS + LSTEvtID] = LPREV;

			if (IDX_EVT_SCT_EPEN[JOPT][LSTEvtID] != -1)
				LPEN[LPREV] = LPEN[LPREV] + epen[IDX_EVT_SCT_EPEN[JOPT][LSTEvtID] + LPREV];

		}

		HERE = LPREV;
	}
	else
	{
		RNK = -1;
	}

	for (INOW = RNK + 1; INOW < COMMOD9->NEVNT; INOW++)
	{

		HEvtID = HPERM[INOW];

		ITGT = COMMOD9->ISTATIC_0[JOPT_X_NEVENTS + HEvtID];

		if (ITGT == -1)
		{

			COMMOD9->HSCTSOL[JOPT_X_NEVENTS + HEvtID] = LPREV;

		}
		else if (LPREV <= ITGT)
		{

			COMMOD9->HSCTSOL[JOPT_X_NEVENTS + HEvtID] = ITGT;

			LPREV = ITGT;

		}
		else
		{

			COMMOD9->HSCTSOL[JOPT_X_NEVENTS + HEvtID] = LPREV;

			THISONE = INOW;
			HERE = LPREV;

			if (IDX_EVT_SCT_EPEN[JOPT][HEvtID] != -1)
				PENHERE = LPEN[LPREV] + epen[IDX_EVT_SCT_EPEN[JOPT][HEvtID] + HERE];

			PLACED = 0;

			while ((PLACED == 0) && (HERE >= 1))
			{

				DOWN = HERE - 1;

				PENDOWN = 0.0;

				temEvtID = HPERM[THISONE];
				while ((THISONE >= 0) && (COMMOD9->HSCTSOL[JOPT_X_NEVENTS + temEvtID] >= HERE))
				{

					if (IDX_EVT_SCT_EPEN[JOPT][temEvtID] != -1)
						PENDOWN = PENDOWN + epen[IDX_EVT_SCT_EPEN[JOPT][temEvtID] + DOWN];

					THISONE--;
					temEvtID = HPERM[THISONE];

				}

				if (PENDOWN <= PENHERE)
				{

					HERE = HERE - 1;
					PENHERE = PENDOWN + LPEN[DOWN];

					if (HERE == 0)
					{

						for (I = INOW; I >= THISONE + 1; I--)
						{
							COMMOD9->HSCTSOL[JOPT_X_NEVENTS + HPERM[I]] = HERE;

						}

						PLACED = 1;

						LPEN[HERE] = PENHERE;

						for (I = HERE + 1; I <= LPREV; I++)
						{

							LPEN[I] = 0;

						}

						LPREV = HERE;
					}
					else
					{

						THISONE = INOW;
					}
				}
				else
				{

					for (I = INOW; I >= THISONE + 1; I--)
					{

						COMMOD9->HSCTSOL[JOPT_X_NEVENTS + HPERM[I]] = HERE;
					}

					PLACED = 1;

					LPEN[HERE] = PENHERE;

					for (I = HERE + 1; I <= LPREV; I++)
					{
						LPEN[I] = 0;
					}

					LPREV = HERE;

				}

			}

		}

		if ((INOW > RNK2) && (COMMOD9->HSCTSOL[JOPT_X_NEVENTS + HEvtID] ==
					COMMOD9->SCTSOL[JOPT_X_NEVENTS + COMMOD9->LSTPERM[INOW]]))
		{

			for (RNK2 = INOW + 1; RNK2 < COMMOD9->NEVNT; RNK2++)
			{
				temEvtID = HPERM[RNK2];

				HERE = COMMOD9->SCTSOL[JOPT_X_NEVENTS + COMMOD9->LSTPERM[RNK2]];

				COMMOD9->HSCTSOL[JOPT_X_NEVENTS + temEvtID] = HERE;

				if (IDX_EVT_SCT_EPEN[JOPT][temEvtID] != -1)
					LPEN[HERE] = LPEN[HERE] + epen[IDX_EVT_SCT_EPEN[JOPT][temEvtID] + HERE];

			}

			break;
		}

	}

	for (I = 0; I <= HERE; I++)
	{
		PENJ = PENJ + LPEN[I];
	}

	delete[] LPEN;
}

void RESCTPEN_NEW(int RNK, int RNK2, int HPERM[], int JOPT, double& PENJ)
{
	COMMOD* COMMOD9 = COMMOD::Singleton();
	int H_LEVEL = COMMOD9->HLEVEL[JOPT] + 1;
	double* LPEN = new double[COMMOD9->MXLVL];
	double PENDOWN, PENHERE;
	int DOWN, HERE, INOW, ITGT, LPREV, PLACED, THISONE;

	int I, RR;

	int JOPT_X_EVT_X_LEVEL_ADD_DOWN = 0;
	int IDX_EPEN = IDX_SCT_EPEN[JOPT];
	int JOPT_X_NEVENTS = JOPT*COMMOD9->NEVNT;

	int LSTEvtID, HEvtID, temEvtID;

	LSTEvtID = 0;
	HEvtID = 0;
	I = 0;
	RR = 0;
	ITGT = 0;
	HERE = 0;
	DOWN = 0;
	THISONE = 0;
	PLACED = 0;
	PENJ = 0.0;

	::SetVal(LPEN, COMMOD9->MXLVL, 0.0);
	PENDOWN = 0.0;
	PENHERE = 0.0;

	INOW = 0;
	LPREV = 0;

	if (RNK > 0)
	{

		for (INOW = 0; INOW <= RNK; INOW++)
		{
			LSTEvtID = COMMOD9->LSTPERM[INOW];

			LPREV = COMMOD9->SCTSOL[JOPT_X_NEVENTS + LSTEvtID];

			COMMOD9->HSCTSOL[JOPT_X_NEVENTS + LSTEvtID] = LPREV;

			if (IDX_EVT_SCT_EPEN[JOPT][LSTEvtID] != -1)
				LPEN[LPREV] = LPEN[LPREV] + epen[IDX_EVT_SCT_EPEN[JOPT][LSTEvtID] + LPREV];

		}

		HERE = LPREV;
	}
	else
	{
		RNK = -1;
	}

	for (INOW = RNK + 1; INOW < COMMOD9->NEVNT; INOW++)
	{

		HEvtID = HPERM[INOW];

		ITGT = COMMOD9->ISTATIC_0[JOPT_X_NEVENTS + HEvtID];

		if (ITGT == -1)
		{

			COMMOD9->HSCTSOL[JOPT_X_NEVENTS + HEvtID] = LPREV;

		}
		else if (LPREV <= ITGT)
		{

			COMMOD9->HSCTSOL[JOPT_X_NEVENTS + HEvtID] = ITGT;

			LPREV = ITGT;

		}
		else
		{

			COMMOD9->HSCTSOL[JOPT_X_NEVENTS + HEvtID] = LPREV;

			THISONE = INOW;
			HERE = LPREV;

			if (IDX_EVT_SCT_EPEN[JOPT][HEvtID] != -1)
				PENHERE = LPEN[LPREV] + epen[IDX_EVT_SCT_EPEN[JOPT][HEvtID] + HERE];

			PLACED = 0;

			while ((PLACED == 0) && (HERE >= 1))
			{

				DOWN = HERE - 1;

				PENDOWN = 0.0;

				temEvtID = HPERM[THISONE];
				while ((THISONE >= 0) && (COMMOD9->HSCTSOL[JOPT_X_NEVENTS + temEvtID] >= HERE))
				{

					if (IDX_EVT_SCT_EPEN[JOPT][temEvtID] != -1)
						PENDOWN = PENDOWN + epen[IDX_EVT_SCT_EPEN[JOPT][temEvtID] + DOWN];

					THISONE--;
					temEvtID = HPERM[THISONE];

				}

				if (PENDOWN <= PENHERE)
				{

					HERE = HERE - 1;
					PENHERE = PENDOWN + LPEN[DOWN];

					if (HERE == 0)
					{

						for (I = INOW; I >= THISONE + 1; I--)
						{
							COMMOD9->HSCTSOL[JOPT_X_NEVENTS + HPERM[I]] = HERE;

						}

						PLACED = 1;

						LPEN[HERE] = PENHERE;

						for (I = HERE + 1; I <= LPREV; I++)
						{

							LPEN[I] = 0;

						}

						LPREV = HERE;
					}
					else
					{

						THISONE = INOW;
					}
				}
				else
				{

					for (I = INOW; I >= THISONE + 1; I--)
					{

						COMMOD9->HSCTSOL[JOPT_X_NEVENTS + HPERM[I]] = HERE;
					}

					PLACED = 1;

					LPEN[HERE] = PENHERE;

					for (I = HERE + 1; I <= LPREV; I++)
					{
						LPEN[I] = 0;
					}

					LPREV = HERE;

				}

			}

		}

		if ((INOW > RNK2) && (COMMOD9->HSCTSOL[JOPT_X_NEVENTS + HEvtID] ==
					COMMOD9->SCTSOL[JOPT_X_NEVENTS + COMMOD9->LSTPERM[INOW]]))
		{

			for (RNK2 = INOW + 1; RNK2 < COMMOD9->NEVNT; RNK2++)
			{
				temEvtID = HPERM[RNK2];

				HERE = COMMOD9->SCTSOL[JOPT_X_NEVENTS + COMMOD9->LSTPERM[RNK2]];

				COMMOD9->HSCTSOL[JOPT_X_NEVENTS + temEvtID] = HERE;

				if (IDX_EVT_SCT_EPEN[JOPT][temEvtID] != -1)
					LPEN[HERE] = LPEN[HERE] + epen[IDX_EVT_SCT_EPEN[JOPT][temEvtID] + HERE];

			}

			break;
		}

	}

	for (I = 0; I <= HERE; I++)
	{
		PENJ = PENJ + LPEN[I];
	}

	delete[] LPEN;
}

void NEWPEN(int IRNK, int JRNK, int HPERM[], double& HPEN)
{

	double PENJ = 0.0;

	int CTRGET;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	COMMOD9->CTRF = 0;
	CTRGET = 0;

	HPEN = 0.0;
	COMMOD9->NGHPEN = 0.0;

	for (int i = 0; i < COMMOD9->NSCT; i++)
	{

		int JOPT = COMMOD9->SectionMaxPenOrder[i];

		int JOPT_X_NEVENTS = JOPT*COMMOD9->NEVNT;
		unsigned short* pHSCTSOL = COMMOD9->HSCTSOL + JOPT_X_NEVENTS, *pSCTSOL = COMMOD9->SCTSOL + JOPT_X_NEVENTS;

		double LocalPENJ = 0.0;
		int TRNK = 0;

		if ((COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + COMMOD9->LSTPERM[IRNK]] != -1)
				|| (COMMOD9->NEGATF != 0))
		{

			TRNK = min(IRNK, JRNK);
			int minRankLVL = COMMOD9->SCTSOL[JOPT*COMMOD9->NEVNT + COMMOD9->LSTPERM[TRNK]] - 1;

			if (TRNK > 0)
			{
				while (COMMOD9->SCTSOL[JOPT*COMMOD9->NEVNT + COMMOD9->LSTPERM[TRNK]]
						>= minRankLVL)
				{
					if (TRNK == 0) break;

					TRNK = TRNK - 1;
				}
			}

			::RESCTPEN(TRNK, max(IRNK, JRNK), HPERM, JOPT, LocalPENJ);

			COMMOD9->HSCTSOL_STATE[JOPT] = 1;

		}
		else if ((COMMOD9->NABRGEN < 3))
		{

			if (COMMOD9->HSCTSOL_STATE[JOPT] == 1)
			{
				memcpy(pHSCTSOL, pSCTSOL, COMMOD9->SOL_SIZE);

				COMMOD9->HSCTSOL_STATE[JOPT] = 0;
			}
			else if (COMMOD9->HSCTSOL_LASTPOSITION[JOPT] != -1)
			{
				COMMOD9->HSCTSOL[COMMOD9->HSCTSOL_LASTPOSITION[JOPT]] = COMMOD9->HSCTSOL_LASTVALUE[JOPT];
			}

			COMMOD9->HSCTSOL_LASTPOSITION[JOPT] = JOPT_X_NEVENTS + COMMOD9->LSTPERM[IRNK];
			COMMOD9->HSCTSOL_LASTVALUE[JOPT] = COMMOD9->HSCTSOL[COMMOD9->HSCTSOL_LASTPOSITION[JOPT]];

			COMMOD9->HSCTSOL[JOPT_X_NEVENTS + COMMOD9->LSTPERM[IRNK]] =
				COMMOD9->HSCTSOL[JOPT_X_NEVENTS + COMMOD9->LSTPERM[JRNK]];

			LocalPENJ = COMMOD9->SCJPEN[JOPT];
		}
		else if ((COMMOD9->NABRGEN == 3) &&
				(COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + COMMOD9->LSTPERM[JRNK]] == -1))
		{

			memcpy(COMMOD9->HSCTSOL + JOPT*COMMOD9->NEVNT, COMMOD9->SCTSOL + JOPT*COMMOD9->NEVNT, COMMOD9->SOL_SIZE);

			TRNK = COMMOD9->HSCTSOL[JOPT*COMMOD9->NEVNT + COMMOD9->LSTPERM[IRNK]];
			COMMOD9->HSCTSOL[JOPT*COMMOD9->NEVNT + COMMOD9->LSTPERM[IRNK]] = COMMOD9->HSCTSOL[JOPT*COMMOD9->NEVNT + COMMOD9->LSTPERM[JRNK]];
			COMMOD9->HSCTSOL[JOPT*COMMOD9->NEVNT + COMMOD9->LSTPERM[JRNK]] = TRNK;

			LocalPENJ = COMMOD9->SCJPEN[JOPT];

		}
		else
		{

			TRNK = min(IRNK, JRNK);

			if (TRNK > 0)
			{
				while (COMMOD9->SCTSOL[JOPT*COMMOD9->NEVNT + COMMOD9->LSTPERM[TRNK]]
						>= (COMMOD9->SCTSOL[JOPT*COMMOD9->NEVNT + COMMOD9->LSTPERM[min(IRNK, JRNK)]] - 1))
				{
					if (TRNK == 0) break;

					TRNK = TRNK - 1;
				}
			}

			::RESCTPEN(TRNK, max(IRNK, JRNK), HPERM, JOPT, LocalPENJ);

		}

		if (COMMOD9->CTRF == 1) CTRGET = 1;

		PENJ = PENJ + LocalPENJ;

		if (PENJ >= COMMOD9->CRTPEN)
		{
			break;

		}

		COMMOD9->COLPEN[JOPT] = LocalPENJ;

	}

	HPEN = PENJ;

	if (COMMOD9->JSPANF == 1)
	{
		::JSPAN(HPERM);
		HPEN = HPEN + COMMOD9->SPANPEN;
	}

	if (COMMOD9->PEN2F > 0) { ::NEW2PEN(IRNK, JRNK, HPERM, HPEN, 0); }

	COMMOD9->CTRF = CTRGET;

}
void MAXSEQ(int MAXTYPE, int K, int IRNK, int JRNK, int XPERM[], double& XPEN)
{

	COMMOD* COMMOD9 = COMMOD::Singleton();

	if ((COMMOD9->IROWS[XPERM[K]][1] > 2) ||
			(COMMOD9->IROWS[XPERM[K]][1] < 1)) goto Label999;

	if (COMMOD9->COEXST[(COMMOD9->IROWS[XPERM[max(IRNK, JRNK)]][2])*COMMOD9->NSPC + (COMMOD9->IROWS[XPERM[K]][2])] < COMMOD9->COXSTF)
	{

		if (MAXTYPE == 1)
		{
			if (COMMOD9->IROWS[XPERM[K]][1] == 2) XPEN = XPEN - 1.00;
		}
		else if (MAXTYPE == 2)
		{
			if (COMMOD9->IROWS[XPERM[K]][1] == 1) XPEN = XPEN + 1.00;
		}
	}

Label999:
	return;
}

void MINSEQ(int MINTYPE, int K, int IRNK, int JRNK, int XPERM[], double& XPEN)
{

	COMMOD* COMMOD9 = COMMOD::Singleton();

	if ((COMMOD9->IROWS[XPERM[K]][1] > 2) ||
			(COMMOD9->IROWS[XPERM[K]][1] < 1)) goto Label888;

	if (COMMOD9->COEXST[(COMMOD9->IROWS[XPERM[min(IRNK, JRNK)]][2])*COMMOD9->NSPC + (COMMOD9->IROWS[XPERM[K]][2])] < COMMOD9->COXSTF)
	{

		if (MINTYPE == 1)
		{
			if (COMMOD9->IROWS[XPERM[K]][1] == 2) XPEN = XPEN + 1.00;
		}
		else if (MINTYPE == 2)
		{
			if (COMMOD9->IROWS[XPERM[K]][1] == 1) XPEN = XPEN - 1.00;
		}
	}

Label888:
	return;

}

void NEWSEQUEL(int IRNK, int JRNK, int XPERM[], double& XPEN)
{
	int K;
	int MAXTYPE, MINTYPE;

	K = 0;
	MAXTYPE = 0;
	MINTYPE = 0;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	MAXTYPE = COMMOD9->IROWS[XPERM[max(IRNK, JRNK)]][1];
	MINTYPE = COMMOD9->IROWS[XPERM[min(IRNK, JRNK)]][1];

	if ((COMMOD9->NABRGEN == 1) || (COMMOD9->NABRGEN == 2))
	{

		if ((COMMOD9->IROWS[XPERM[JRNK]][1] > 2) ||
				(COMMOD9->IROWS[XPERM[JRNK]][1] < 1)) goto Label9999;

		if (IRNK < JRNK)
		{

			for (K = min(IRNK, JRNK); K < max(IRNK, JRNK) - 1; K++)
			{
				::MAXSEQ(MAXTYPE, K, IRNK, JRNK, XPERM, XPEN);
			}

		}
		else if (IRNK > JRNK)
		{

			for (K = min(IRNK, JRNK) + 1; K < max(IRNK, JRNK); K++)
			{
				::MINSEQ(MINTYPE, K, IRNK, JRNK, XPERM, XPEN);
			}
		}

	}
	else if (COMMOD9->NABRGEN == 3)
	{

		if ((MAXTYPE == 2) || (MAXTYPE == 1))
		{
			for (K = min(IRNK, JRNK); K < max(IRNK, JRNK) - 1; K++)
			{

				::MAXSEQ(MAXTYPE, K, IRNK, JRNK, XPERM, XPEN);
			}
		}

		if ((MINTYPE == 2) || (MINTYPE == 1))
		{
			for (K = min(IRNK, JRNK) + 1; K < max(IRNK, JRNK) - 1; K++)
			{

				::MINSEQ(MINTYPE, K, IRNK, JRNK, XPERM, XPEN);
			}
		}
	}

Label9999:
	return;

}

void MAXROY(int MAXTYPE, int K, int IRNK, int JRNK, int XPERM[], double& XPEN)
{

	COMMOD* COMMOD9 = COMMOD::Singleton();

	if ((COMMOD9->IROWS[XPERM[K]][1] > 2) ||
			(COMMOD9->IROWS[XPERM[K]][1] < 1)) goto Label999;

	if (COMMOD9->COEXST[(COMMOD9->IROWS[XPERM[max(IRNK, JRNK)]][2])*COMMOD9->NSPC + (COMMOD9->IROWS[XPERM[K]][2])] < COMMOD9->COXSTF)
	{

		if (MAXTYPE == 1)
		{
			if (COMMOD9->IROWS[XPERM[K]][1] == 2) XPEN = XPEN - 1.00;
		}
		else if (MAXTYPE == 2)
		{
			if (COMMOD9->IROWS[XPERM[K]][1] == 1) XPEN = XPEN + 1.00;
		}
	}

Label999:
	return;

}

void MINROY(int MINTYPE, int K, int IRNK, int JRNK, int XPERM[], double& XPEN)
{

	COMMOD* COMMOD9 = COMMOD::Singleton();

	if ((COMMOD9->IROWS[XPERM[K]][1] > 2) ||
			(COMMOD9->IROWS[XPERM[K]][1] < 1)) goto Label888;

	if (COMMOD9->COEXST[(COMMOD9->IROWS[XPERM[min(IRNK, JRNK)]][2])*COMMOD9->NSPC + (COMMOD9->IROWS[XPERM[K]][2])] < COMMOD9->COXSTF)
	{

		if (MINTYPE == 1)
		{
			if (COMMOD9->IROWS[XPERM[K]][1] == 2) XPEN = XPEN + 1.00;
		}
		else if (MINTYPE == 2)
		{
			if (COMMOD9->IROWS[XPERM[K]][1] == 1) XPEN = XPEN - 1.00;
		}
	}

Label888:
	return;

}

void NEWROYAL(int IRNK, int JRNK, int XPERM[], double& XPEN)
{
	int K;
	int MAXTYPE, MINTYPE;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	K = 0;
	MAXTYPE = 0;
	MINTYPE = 0;
	MAXTYPE = COMMOD9->IROWS[XPERM[max(IRNK, JRNK)]][1];
	MINTYPE = COMMOD9->IROWS[XPERM[min(IRNK, JRNK)]][1];

	if ((COMMOD9->NABRGEN == 1) || (COMMOD9->NABRGEN == 2))
	{

		if ((COMMOD9->IROWS[XPERM[JRNK]][1] > 2) ||
				(COMMOD9->IROWS[XPERM[JRNK]][1] < 1)) goto Label9999;

		if (IRNK < JRNK)
		{

			for (K = min(IRNK, JRNK); K < max(IRNK, JRNK) - 1; K++)
			{
				::MAXROY(MAXTYPE, K, IRNK, JRNK, XPERM, XPEN);
			}
		}
		else if (IRNK > JRNK)
		{

			for (K = min(IRNK, JRNK) + 1; K < max(IRNK, JRNK); K++)
			{
				::MINROY(MINTYPE, K, IRNK, JRNK, XPERM, XPEN);

			}
		}
	}
	else if (COMMOD9->NABRGEN == 3)
	{

		if ((MAXTYPE == 2) || (MAXTYPE == 1))
		{
			for (K = min(IRNK, JRNK); K < max(IRNK, JRNK) - 1; K++)
			{

				::MAXROY(MAXTYPE, K, IRNK, JRNK, XPERM, XPEN);
			}
		}

		if ((MINTYPE == 2) || (MINTYPE == 1))
		{
			for (K = min(IRNK, JRNK) + 1; K < max(IRNK, JRNK) - 1; K++)
			{

				::MINROY(MINTYPE, K, IRNK, JRNK, XPERM, XPEN);
			}
		}
	}

Label9999:
	return;

}

void NEWSPAN(int IRNK, int JRNK, int XPERM[], double& XPEN)
{
	int I, jump;

	I = 0;
	jump = 0;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	jump = max(IRNK, JRNK) - min(IRNK, JRNK);

	if ((COMMOD9->NABRGEN == 1) || (COMMOD9->NABRGEN == 2))
	{

		if (IRNK < JRNK)
		{

			if (COMMOD9->IROWS[XPERM[JRNK]][1] == 1)
			{
				XPEN = XPEN - jump;
			}
			else if (COMMOD9->IROWS[XPERM[JRNK]][1] == 2)
			{

				XPEN = XPEN + jump;
			}

			for (I = IRNK; I < JRNK - 1; I++)
			{

				if (COMMOD9->IROWS[XPERM[I]][1] == 1)
				{
					XPEN = XPEN + 1;
				}
				else if (COMMOD9->IROWS[XPERM[I]][1] == 2)
				{

					XPEN = XPEN - 1;
				}
			}
		}
		else if (IRNK > JRNK)
		{

			if (COMMOD9->IROWS[XPERM[JRNK]][1] == 1)
			{
				XPEN = XPEN + jump;
			}
			else if (COMMOD9->IROWS[XPERM[JRNK]][1] == 2)
			{

				XPEN = XPEN - jump;
			}

			for (I = JRNK + 1; I < IRNK; I++)
			{

				if (COMMOD9->IROWS[XPERM[I]][1] == 1)
				{
					XPEN = XPEN - 1;
				}
				else if (COMMOD9->IROWS[XPERM[I]][1] == 2)
				{

					XPEN = XPEN + 1;
				}
			}
		}
	}
	else if (COMMOD9->NABRGEN == 3)
	{

		if (COMMOD9->IROWS[XPERM[max(IRNK, JRNK)]][1] == 1)
		{
			XPEN = XPEN - jump;
		}
		else if (COMMOD9->IROWS[XPERM[max(IRNK, JRNK)]][1] == 2)
		{
			XPEN = XPEN + jump;
		}

		if (COMMOD9->IROWS[XPERM[min(IRNK, JRNK)]][1] == 1)
		{
			XPEN = XPEN + jump;
		}
		else if (COMMOD9->IROWS[XPERM[min(IRNK, JRNK)]][1] == 2)
		{
			XPEN = XPEN - jump;
		}
	}

}

void NEW2PEN(int IRNK, int JRNK, int HPERM[], double& HPEN, int BACK)
{

	COMMOD* COMMOD9 = COMMOD::Singleton();

	if (BACK == 1) HPEN = HPEN - COMMOD9->TOOPEN;

	if (COMMOD9->KSM > 0.0)
	{
		::SMOOTH(HPERM);

		HPEN = HPEN + (COMMOD9->SMPEN * COMMOD9->KSM);
	}

	if ((COMMOD9->KSQ > 0.0) || (COMMOD9->SOLVEF == 4) || (COMMOD9->SOLVEF == 5))
	{
		::SQUEEZE(HPERM);
		HPEN = HPEN + (COMMOD9->SQPEN * COMMOD9->KSQ);
	}

	if ((COMMOD9->KSM > 0.0) || (COMMOD9->SOLVEF == 6) || (COMMOD9->SOLVEF == 7))
	{
		::SHRINK(HPERM);
		HPEN = HPEN + (COMMOD9->SHPEN * COMMOD9->KSH);
	}

	if ((COMMOD9->KTS > 0.0) || (COMMOD9->SOLVEF == 8) || (COMMOD9->SOLVEF == 9))
	{
		if (COMMOD9->STKF == 9)
		{

			if (COMMOD9->FB4LF != 0)
			{
				::NEWSEQUEL(IRNK, JRNK, HPERM, COMMOD9->TSPEN);
			}

		}
		else if (COMMOD9->STKF == 8)
		{

			::NEWROYAL(IRNK, JRNK, HPERM, COMMOD9->TSPEN);		
		}
		else if (COMMOD9->STKF == 7)
		{

			::NEWSPAN(IRNK, JRNK, HPERM, COMMOD9->TSPEN);
		}
		else
		{
			::TEASE(HPERM);
		}

		HPEN = HPEN + (COMMOD9->TSPEN * COMMOD9->KTS);

	}

}

void NWDMPEN(int IRNK, int JRNK, int HPERM[], double& HPEN)
{
	int JOPT;
	double PENJ;
	int SEP, K, statij;

	K = 0;
	SEP = 0;
	JOPT = 0;
	statij = 0;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	HPEN = COMMOD9->PEN;

	if (COMMOD9->JSPANF == 1) HPEN = HPEN - COMMOD9->ASPNPEN;

	if ((COMMOD9->PENF == 5) || (COMMOD9->PENF == 6)) goto Label5000;

	if ((COMMOD9->NABRGEN == 1) || (COMMOD9->NABRGEN == 2))
	{
		if (COMMOD9->PENF != 4)
		{

			for (JOPT = 0; JOPT < COMMOD9->NSCT; JOPT++)
			{
				PENJ = 0.0;

				if (::IORZERO(HPERM[JRNK], JOPT)) goto Label2000;

				statij = COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[JRNK]];

				if (IRNK < JRNK)
				{

					for (K = min(IRNK, JRNK); K < max(IRNK, JRNK) - 1; K++)
					{

						if (::IORZERO(HPERM[K], JOPT)) goto Label1000;

						SEP = statij - COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[K]];
						if ((COMMOD9->PENF == 2) && (SEP > 0))
						{
							PENJ = PENJ - 1;
						}
						else if ((COMMOD9->PENF == 2) && (SEP < 0))
						{
							PENJ = PENJ + 1;
						}
						else if ((COMMOD9->PENF == 7) && (SEP > 0))
						{
							PENJ = PENJ - (double)(JRNK - K) / (double)COMMOD9->NEVNT;
						}
						else if ((COMMOD9->PENF == 7) && (SEP < 0))
						{
							PENJ = PENJ + (double)(JRNK - K) / (double)COMMOD9->NEVNT;
						}
						else if (COMMOD9->PENF == 3)
						{
							PENJ = PENJ - SEP;
						}

					}

Label1000:
					HPEN = HPEN + PENJ;

					COMMOD9->COLPEN[JOPT] = COMMOD9->COLPEN[JOPT] + PENJ;
				}
				else if (IRNK > JRNK)
				{

					for (K = min(IRNK, JRNK) + 1; K < max(IRNK, JRNK); K++)
					{
						if (::IORZERO(HPERM[K], JOPT)) goto Label1500;

						SEP = COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[K]] - statij;

						if ((COMMOD9->PENF == 2) && (SEP > 0))
						{
							PENJ = PENJ - 1;
						}
						else if ((COMMOD9->PENF == 2) && (SEP < 0))
						{
							PENJ = PENJ + 1;
						}
						else if (COMMOD9->PENF == 3)
						{
							PENJ = PENJ - SEP;
						}
						else if ((COMMOD9->PENF == 7) && (SEP > 0))
						{
							PENJ = PENJ - (double)(K - JRNK) / (double)COMMOD9->NEVNT;
						}
						else if ((COMMOD9->PENF == 7) && (SEP < 0))
						{
							PENJ = PENJ + (double)(K - JRNK) / (double)COMMOD9->NEVNT;
						}

					}

Label1500:
					HPEN = HPEN + PENJ;
					COMMOD9->COLPEN[JOPT] = COMMOD9->COLPEN[JOPT] + PENJ;
				}

Label2000:
				;
			}
		}
		else if (COMMOD9->PENF == 4)
		{

			if (IRNK < JRNK)
			{

				if (COMMOD9->RASCon) goto Label2100;

				for (K = min(IRNK, JRNK); K < max(IRNK, JRNK) - 1; K++)
				{
					PENJ = 0.0;
					SEP = 0;

					for (JOPT = 0; JOPT < COMMOD9->NSCT; JOPT++)
					{

						if ((::IORZERO(HPERM[JRNK], JOPT))) continue;

						SEP = SEP + 1;

						if (COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[JRNK]] >
								COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[K]])
						{
							PENJ = PENJ - 1;
							COMMOD9->COLPEN[JOPT] = COMMOD9->COLPEN[JOPT] - 1;
						}
						else if (COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[JRNK]] <
								COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[K]])
						{
							PENJ = PENJ + 1;
							COMMOD9->COLPEN[JOPT] = COMMOD9->COLPEN[JOPT] + 1;
						}

					}

					if (SEP > 0) HPEN = HPEN + (PENJ / SEP);

				}

				goto Label2300;

Label2100:

				for (K = min(IRNK, JRNK); K < max(IRNK, JRNK) - 1; K++)
				{

					if (COMMOD9->RASC[HPERM[JRNK]][HPERM[K]] +
							COMMOD9->RASC[HPERM[K]][HPERM[JRNK]] <= 0) continue;

					HPEN = HPEN +
						(double)(COMMOD9->RASC[HPERM[JRNK]][HPERM[K]]) /
						(double)(COMMOD9->RASC[HPERM[JRNK]][HPERM[K]] +
								(double)(COMMOD9->RASC[HPERM[K]][HPERM[JRNK]]))
						- (double)(COMMOD9->RASC[HPERM[K]][HPERM[JRNK]]) /
						(double)(COMMOD9->RASC[HPERM[JRNK]][HPERM[K]] +
								(double)(COMMOD9->RASC[HPERM[K]][HPERM[JRNK]]));
				}

Label2300:
				;
			}
			else if (IRNK > JRNK)
			{

				if (COMMOD9->RASCon) goto Label2500;

				for (K = min(IRNK, JRNK) + 1; K < max(IRNK, JRNK); K++)
				{
					PENJ = 0.0;
					SEP = 0;

					for (JOPT = 0; JOPT < COMMOD9->NSCT; JOPT++)
					{

						if ((::IORZERO(HPERM[JRNK], JOPT)) ||
								(::IORZERO(HPERM[K], JOPT))) continue;

						SEP = SEP + 1;

						if (COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[K]] >
								COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[JRNK]])
						{
							PENJ = PENJ - 1;
							COMMOD9->COLPEN[JOPT] = COMMOD9->COLPEN[JOPT] + 1;
						}
						else if (COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[K]] <
								COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[JRNK]])
						{
							PENJ = PENJ + 1;
							COMMOD9->COLPEN[JOPT] = COMMOD9->COLPEN[JOPT] + 1;
						}
					}

					if (SEP > 0) HPEN = HPEN + (PENJ / SEP);
				}

				goto Label2700;

Label2500:

				for (K = min(IRNK, JRNK) + 1; K < max(IRNK, JRNK); K++)
				{

					if (COMMOD9->RASC[HPERM[JRNK]][HPERM[K]] +
							COMMOD9->RASC[HPERM[K]][HPERM[JRNK]] <= 0) continue;

					HPEN = HPEN -
						(double)(COMMOD9->RASC[HPERM[JRNK]][HPERM[K]]) /
						(double)((COMMOD9->RASC[HPERM[JRNK]][HPERM[K]]) +
								(COMMOD9->RASC[HPERM[K]][HPERM[JRNK]]))
						+ (double)(COMMOD9->RASC[HPERM[K]][HPERM[JRNK]]) /
						(double)((COMMOD9->RASC[HPERM[JRNK]][HPERM[K]]) +
								(COMMOD9->RASC[HPERM[K]][HPERM[JRNK]]));

				}

Label2700:
				;
			}
		}
	}
	else if (COMMOD9->NABRGEN == 3)
	{

		if (COMMOD9->PENF != 4)
		{
			for (JOPT = 0; JOPT < COMMOD9->NSCT; JOPT++)
			{
				PENJ = 0.0;

				if (::IORZERO(HPERM[max(IRNK, JRNK)], JOPT)) goto Label2900;

				statij = COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[max(IRNK, JRNK)]];

				for (K = min(IRNK, JRNK); K < max(IRNK, JRNK) - 1; K++)
				{

					if (::IORZERO(HPERM[K], JOPT)) goto Label2800;

					SEP = statij - COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[K]];

					if ((COMMOD9->PENF == 2) && (SEP > 0))
					{
						PENJ = PENJ - 1;
					}
					else if ((COMMOD9->PENF == 2) && (SEP < 0))
					{
						PENJ = PENJ + 1;
					}
					else if ((COMMOD9->PENF == 7) && (SEP > 0))
					{
						PENJ = PENJ - (double)(JRNK - K) / (double)COMMOD9->NEVNT;
					}
					else if ((COMMOD9->PENF == 7) && (SEP < 0))
					{
						PENJ = PENJ + (double)(JRNK - K) / (double)COMMOD9->NEVNT;
					}
					else if (COMMOD9->PENF == 3)
					{
						PENJ = PENJ - SEP;
					}

Label2800:
					;
				}

				HPEN = HPEN + PENJ;
				COMMOD9->COLPEN[JOPT] = COMMOD9->COLPEN[JOPT] + PENJ;

Label2900:
				PENJ = 0.0;

				if (::IORZERO(HPERM[min(IRNK, JRNK)], JOPT)) goto Label3000;

				statij = COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[min(IRNK, JRNK)]];

				for (K = min(IRNK, JRNK) + 1; K < max(IRNK, JRNK) - 1; K++)
				{

					if (::IORZERO(HPERM[K], JOPT)) goto Label2950;

					SEP = COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[K]] - statij;

					if ((COMMOD9->PENF == 2) && (SEP > 0))
					{
						PENJ = PENJ - 1;
					}
					else if ((COMMOD9->PENF == 2) && (SEP < 0))
					{
						PENJ = PENJ + 1;
					}
					else if ((COMMOD9->PENF == 7) && (SEP > 0))
					{
						PENJ = PENJ - (double)(K - JRNK) / (double)COMMOD9->NEVNT;
					}
					else if ((COMMOD9->PENF == 7) && (SEP < 0))
					{
						PENJ = PENJ + (double)(K - JRNK) / (double)COMMOD9->NEVNT;
					}
					else if (COMMOD9->PENF == 3)
					{
						PENJ = PENJ - SEP;
					}

Label2950:
					;

				}

				HPEN = HPEN + PENJ;
				COMMOD9->COLPEN[JOPT] = COMMOD9->COLPEN[JOPT] + PENJ;

Label3000:
				;
			}

		}
		else if (COMMOD9->PENF == 4)
		{

			if (COMMOD9->RASCon) goto Label3100;

			for (K = min(IRNK, JRNK); K < max(IRNK, JRNK) - 1; K++)
			{
				PENJ = 0.0;
				SEP = 0;

				for (JOPT = 0; JOPT < COMMOD9->NSCT; JOPT++)
				{

					if (::IORZERO(HPERM[max(IRNK, JRNK)], JOPT)) continue;

					if (::IORZERO(HPERM[K], JOPT)) continue;

					SEP = SEP + 1;

					if (COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[max(IRNK, JRNK)]] >
							COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[K]])
					{
						PENJ = PENJ - 1;
						COMMOD9->COLPEN[JOPT] = COMMOD9->COLPEN[JOPT] + 1;
					}
					else if (COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[max(IRNK, JRNK)]] <
							COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[K]])
					{
						PENJ = PENJ + 1;
						COMMOD9->COLPEN[JOPT] = COMMOD9->COLPEN[JOPT] + 1;
					}

				}

				if (SEP > 0) HPEN = HPEN + (PENJ / SEP);

			}

			goto Label3300;

Label3100:
			;

			for (K = min(IRNK, JRNK); K < max(IRNK, JRNK) - 1; K++)
			{

				if (COMMOD9->RASC[HPERM[max(IRNK, JRNK)]][HPERM[K]] +
						COMMOD9->RASC[HPERM[K]][HPERM[max(IRNK, JRNK)]] <= 0) continue;

				HPEN = HPEN +
					(double)(COMMOD9->RASC[HPERM[max(IRNK, JRNK)]][HPERM[K]]) /
					(double)(COMMOD9->RASC[HPERM[max(IRNK, JRNK)]][HPERM[K]] +
							COMMOD9->RASC[HPERM[K]][HPERM[max(IRNK, JRNK)]])
					- (double)(COMMOD9->RASC[HPERM[K]][HPERM[max(IRNK, JRNK)]]) /
					(double)(COMMOD9->RASC[HPERM[max(IRNK, JRNK)]][HPERM[K]] +
							COMMOD9->RASC[HPERM[K]][HPERM[max(IRNK, JRNK)]]);

			}

Label3300:

			if (COMMOD9->RASCon) goto Label3500;

			for (K = min(IRNK, JRNK) + 1; K < max(IRNK, JRNK) - 1; K++)
			{

				PENJ = 0.0;
				SEP = 0;

				for (JOPT = 0; JOPT < COMMOD9->NSCT; JOPT++)
				{

					if (::IORZERO(HPERM[min(IRNK, JRNK)], JOPT)) continue;

					if (::IORZERO(HPERM[JRNK], JOPT)) continue;

					SEP = SEP + 1;

					if (COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[K]] >
							COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[min(IRNK, JRNK)]])
					{
						PENJ = PENJ - 1;
						COMMOD9->COLPEN[JOPT] = COMMOD9->COLPEN[JOPT] + 1;
					}
					else if (COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[K]] <
							COMMOD9->ISTATIC_0[JOPT*COMMOD9->NEVNT + HPERM[min(IRNK, JRNK)]])
					{
						PENJ = PENJ + 1;
						COMMOD9->COLPEN[JOPT] = COMMOD9->COLPEN[JOPT] + 1;
					}
				}

				if (SEP > 0) HPEN = HPEN + (PENJ / SEP);

			}

			goto Label3700;

Label3500:

			for (K = min(IRNK, JRNK) + 1; K < max(IRNK, JRNK) - 1; K++)
			{

				if (COMMOD9->RASC[HPERM[min(IRNK, JRNK)]][HPERM[K]] +
						COMMOD9->RASC[HPERM[K]][HPERM[min(IRNK, JRNK)]] <= 0) continue;

				HPEN = HPEN -
					(double)(COMMOD9->RASC[HPERM[min(IRNK, JRNK)]][HPERM[K]]) /
					(double)(COMMOD9->RASC[HPERM[min(IRNK, JRNK)]][HPERM[K]] +
							COMMOD9->RASC[HPERM[K]][HPERM[min(IRNK, JRNK)]])
					+ (double)(COMMOD9->RASC[HPERM[K]][HPERM[min(IRNK, JRNK)]]) /
					(double)(COMMOD9->RASC[HPERM[min(IRNK, JRNK)]][HPERM[K]] +
							COMMOD9->RASC[HPERM[K]][HPERM[min(IRNK, JRNK)]]);

			}

Label3700:
			;
		}
	}

	if (COMMOD9->JSPANF == 1)
	{
		::JSPAN(HPERM);
		HPEN = HPEN + COMMOD9->SPANPEN;
	}

Label5000:

	if (COMMOD9->PENF == 5)
	{

		::NEWROYAL(IRNK, JRNK, HPERM, HPEN);
	}

	if (COMMOD9->PENF == 6)
	{

		if (COMMOD9->FB4LF != 0) ::NEWSEQUEL(IRNK, JRNK, HPERM, HPEN);
	}

	if (COMMOD9->PEN2F > 0) ::NEW2PEN(IRNK, JRNK, HPERM, HPEN, 1);

}

void GETEVNT(int I, string& xtitle, int flg1, int flg2, int flg3, int flg4, int flg5, int flg6)
{
	int Nj;
	string xsct, xevent;
	string xtag;

	Nj = 0;
	xsct = "  ";
	xtag = "  ";
	xevent = "  ";
	xtitle = "  ";

	COMMOD* COMMOD9 = COMMOD::Singleton();

	if ((flg1 != 0) || (flg2 != 0))
	{
		if (COMMOD9->EVTFILE.substr(0, 3) != "OFF")
		{
			if (flg2 == 0)
			{
				xtitle = " ";
			}
			else
			{
				::trim(COMMOD9->EVNTNAME[COMMOD9->IROWS[I][0]]);
				xtitle = COMMOD9->EVNTNAME[COMMOD9->IROWS[I][0]];
			}

			if (flg1 != 0)
			{
				::trim(COMMOD9->EVNTNICK[COMMOD9->IROWS[I][0]]);
				::trim(xtitle);

				xtitle = "[" + COMMOD9->EVNTNICK[COMMOD9->IROWS[I][0]] +
					"]   " + xtitle;
			}
		}
		else
		{
			char buff[10];

			::sprintf(buff, "%d", I + 1);
			xtitle = string(buff);
			::trim(xtitle);
			xtitle = "Event " + xtitle;
		}
	}

	if (flg3 != 0)
	{
		if (COMMOD9->IROWS[I][1] == 1)
		{
			::trim(xtitle);
			xtitle = xtitle + " FAD";
		}
		else if (COMMOD9->IROWS[I][1] == 2)
		{
			::trim(xtitle);
			xtitle = xtitle + " LAD";
		}
		else if (COMMOD9->IROWS[I][1] == -1)
		{
			::trim(xtitle);
			xtitle = xtitle + " APP";
		}
		else if (COMMOD9->IROWS[I][1] == -2)
		{
			::trim(xtitle);
			xtitle = xtitle + " DIS";
		}
		else if (COMMOD9->IROWS[I][1] == 3)
		{
			::trim(xtitle);
			xtitle = xtitle + " MID";
		}
		else if (COMMOD9->IROWS[I][1] == -3)
		{
			::trim(xtitle);
			xtitle = xtitle + " BOB";
		}
		else if (COMMOD9->IROWS[I][1] == 4)
		{
			::trim(xtitle);
			xtitle = xtitle + " ASH";
		}
		else if (COMMOD9->IROWS[I][1] == 5)
		{
			::trim(xtitle);
			xtitle = xtitle + " AGE";
		}
		else if (COMMOD9->IROWS[I][1] == 11)
		{
			::trim(xtitle);
			xtitle = xtitle + " MAX";
		}
		else if (COMMOD9->IROWS[I][1] == 12)
		{
			::trim(xtitle);
			xtitle = xtitle + " MIN";
		}
	}

	if (flg4 != 0)
	{
		Nj = COMMOD9->CULLIST[I][2];

		if (Nj > 0)
		{
			char buff[10];

			::sprintf(buff, "%d", Nj);
			xsct = string(buff);
			::trim(xtitle);
			::trim(xsct);
			xtitle = xtitle + " {in " + xsct + " sections}";
		}

	}

	if (flg5 != 0)
	{
		char buff[10];

		::sprintf(buff, "%d", I + 1);
		xevent = string(buff);
		::trim(xtitle);
		::trim(xevent);
		xtitle = xtitle + "  (C# " + " " + xevent + ")";
	}

	if (flg6 != 0)
	{
		xtag = COMMOD9->ETAGNAME[COMMOD9->IROWS[I][5]];
		::trim(xtitle);
		::trim(xtag);
		xtitle = xtitle + "  (tag:" + " " + xtag + ")";
	}

}

double PLACE(int evt, int sec)
{
	COMMOD* COMMOD9 = COMMOD::Singleton();

	if (sec <= COMMOD9->NSCT - 1)
	{
		return COMMOD9->VALEVEL[sec][COMMOD9->HSCTSOL[sec*COMMOD9->NEVNT + COMMOD9->BSTPERM[evt]]];
	}
	else
	{
		return COMMOD9->COMPLVL[COMMOD9->HSCTSOL[sec*COMMOD9->NEVNT + COMMOD9->BSTPERM[evt]]][sec - COMMOD9->NSCT];
	}

}

void UNBASE()
{
	int I, J;
	double fctr;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	for (J = 0; J < COMMOD9->NSCT; J++)
	{

		fctr = COMMOD9->VALEVEL[J][COMMOD9->MXLVL + 2];

		if (fctr != 0.00)
		{

			for (I = 0; I <= COMMOD9->HLEVEL[J]; I++)
			{
				COMMOD9->VALEVEL[J][I] = COMMOD9->VALEVEL[J][I] - fctr;
			}
		}
	}

}

void CalcEPEN()
{
	COMMOD* COMMOD9 = COMMOD::Singleton();
	memset(epen, 0, sizeof(float)*epenCount);

	int h_level = 0;
	int ELVL = 0;
	int EMOVES = 0;
	float DIST = 0.0;
	float WEIGHT = 0.0;
	int IDX = 0;
	int R_IDX = 0;
	int IDX_ISTATIC_0 = 0;

	if ((COMMOD9->PENF == 0) || (COMMOD9->PENF > 1))
	{
		for (int sct = 0; sct < COMMOD9->NSCT; sct++)
		{
			for (int evt = 0; evt < COMMOD9->NEVNT; evt++)
			{
				IDX_ISTATIC_0 = sct*COMMOD9->NEVNT + evt;

				if (COMMOD9->ISTATIC_0[IDX_ISTATIC_0] == -1) continue;

				int h_level = COMMOD9->HLEVEL[sct] + 1;

				ELVL = COMMOD9->ISTATIC_0[IDX_ISTATIC_0];

				for (int k = 0; k < h_level; k++)
				{
					DIST = COMMOD9->VALEVEL[sct][ELVL] - COMMOD9->VALEVEL[sct][k];

					EMOVES = COMMOD9->ISTATIC_1[IDX_ISTATIC_0];

					if (DIST < 0.0)
					{

						R_IDX = COMMOD9->IDX_SCT_RSTATIC[sct] + evt * 2;
						WEIGHT = COMMOD9->NRSTATIC[R_IDX];
					}
					else
					{

						R_IDX = COMMOD9->IDX_SCT_RSTATIC[sct] + evt * 2 + 1;
						WEIGHT = COMMOD9->NRSTATIC[R_IDX];
					}

					IDX = IDX_EVT_SCT_EPEN[sct][evt] + k;

					switch (EMOVES)
					{
						case -1:
						case 0:

							epen[IDX] = ::fabs(DIST) * WEIGHT * COMMOD9->LARGE;

							break;
						case 1:

							if (DIST >= 0)
							{

								epen[IDX] = DIST * WEIGHT;
							}
							else
							{

								epen[IDX] = ::fabs(DIST) * COMMOD9->LARGE;

								if (WEIGHT > 0.0) epen[IDX] = epen[IDX] * WEIGHT;
							}
							break;
						case 2:

							if (DIST <= 0)
							{

								epen[IDX] = ::fabs(DIST) * WEIGHT;
							}
							else
							{

								epen[IDX] = DIST * COMMOD9->LARGE;
								if (WEIGHT > 0.0) epen[IDX] = epen[IDX] * WEIGHT;

							}
							break;
						case 3:

							epen[IDX] = ::fabs(DIST) * WEIGHT;
							break;
					}

				}

			}
		}

	}
	else if (COMMOD9->PENF == 1)
	{
		for (int sct = 0; sct < COMMOD9->NSCT; sct++)
		{

			for (int evt = 0; evt < COMMOD9->NEVNT; evt++)
			{
				IDX_ISTATIC_0 = sct*COMMOD9->NEVNT + evt;

				if (COMMOD9->ISTATIC_0[IDX_ISTATIC_0] == -1) continue;

				int h_level = COMMOD9->HLEVEL[sct] + 1;

				ELVL = COMMOD9->ISTATIC_0[IDX_ISTATIC_0];

				for (int k = 0; k < h_level; k++)
				{
					DIST = (float)(ELVL - k);

					IDX = IDX_EVT_SCT_EPEN[sct][evt] + k;

					EMOVES = COMMOD9->ISTATIC_1[IDX_ISTATIC_0];

					if (DIST < 0.0)
					{

						R_IDX = COMMOD9->IDX_SCT_RSTATIC[sct] + evt * 2;

						WEIGHT = COMMOD9->NRSTATIC[R_IDX];
					}
					else
					{

						R_IDX = COMMOD9->IDX_SCT_RSTATIC[sct] + evt * 2 + 1;

						WEIGHT = COMMOD9->NRSTATIC[R_IDX];
					}

					switch (EMOVES)
					{
						case -1:
						case 0:

							epen[IDX] = ::fabs(DIST) * WEIGHT * COMMOD9->LARGE;
							break;
						case 1:

							if (DIST >= 0)
							{

								epen[IDX] = DIST * WEIGHT;
							}
							else
							{

								epen[IDX] = ::fabs(DIST) * COMMOD9->LARGE;

								if (WEIGHT > 0.0) epen[IDX] = epen[IDX] * WEIGHT;

							}
							break;
						case 2:

							if (DIST <= 0)
							{

								epen[IDX] = ::fabs(DIST) * WEIGHT;
							}
							else
							{

								epen[IDX] = DIST * COMMOD9->LARGE;
								if (WEIGHT > 0.0) epen[IDX] = epen[IDX] * WEIGHT;

							}
							break;
						case 3:

							epen[IDX] = ::fabs(DIST) * WEIGHT;
							break;
					}
				}

			}
		}

	}
	else if (COMMOD9->PENF == -1)
	{
		for (int sct = 0; sct < COMMOD9->NSCT; sct++)
		{
			for (int evt = 0; evt < COMMOD9->NEVNT; evt++)
			{
				IDX_ISTATIC_0 = sct*COMMOD9->NEVNT + evt;

				if (COMMOD9->ISTATIC_0[IDX_ISTATIC_0] == -1) continue;

				int h_level = COMMOD9->HLEVEL[sct] + 1;

				ELVL = COMMOD9->ISTATIC_0[IDX_ISTATIC_0];

				for (int k = 0; k < h_level; k++)
				{
					if ((ELVL - k) > 1)
					{
						DIST = 1.0f + (float)(COMMOD9->ELEVEL[sct][ELVL - 1] - COMMOD9->ELEVEL[sct][k]);
					}
					else if ((ELVL - k) < -1)
					{
						DIST = (float)(COMMOD9->ELEVEL[sct][ELVL + 1] - COMMOD9->ELEVEL[sct][k]) - 1.0f;
					}
					else if (abs(ELVL - k) == 1)
					{
						DIST = (float)(ELVL - k);
					}
					else if (ELVL == k)
					{
						DIST = 0.0f;
					}

					IDX = IDX_EVT_SCT_EPEN[sct][evt] + k;

					EMOVES = COMMOD9->ISTATIC_1[IDX_ISTATIC_0];

					if (DIST < 0.0)
					{

						R_IDX = COMMOD9->IDX_SCT_RSTATIC[sct] + evt * 2;

						WEIGHT = COMMOD9->NRSTATIC[R_IDX];
					}
					else
					{

						R_IDX = COMMOD9->IDX_SCT_RSTATIC[sct] + evt * 2 + 1;

						WEIGHT = COMMOD9->NRSTATIC[R_IDX];
					}

					switch (EMOVES)
					{
						case -1:
						case 0:

							epen[IDX] = ::fabs(DIST) * WEIGHT * COMMOD9->LARGE;
							break;
						case 1:

							if (DIST >= 0)
							{

								epen[IDX] = DIST * WEIGHT;
							}
							else
							{

								epen[IDX] = ::fabs(DIST) * COMMOD9->LARGE;

								if (WEIGHT > 0.0) epen[IDX] = epen[IDX] * WEIGHT;
							}
							break;
						case 2:

							if (DIST <= 0)
							{

								epen[IDX] = ::fabs(DIST) * WEIGHT;
							}
							else
							{

								epen[IDX] = DIST * COMMOD9->LARGE;
								if (WEIGHT > 0.0) epen[IDX] = epen[IDX] * WEIGHT;
							}
							break;
						case 3:

							epen[IDX] = ::fabs(DIST) * WEIGHT;
							break;
					}

				}

			}
		}

	}

}
