#include <iostream>
#include <math.h>
#include "COMMOD.h"
#include "Helper.h"
#include "DOCOEX.h"

using namespace std;
using std::string;

DOCOEX* DOCOEX::g_singleton=NULL;
extern	bool	*allrzero;
extern BYTE G_BYTE_0;
extern BYTE G_BYTE_1;
extern BYTE G_BYTE_2;
extern BYTE G_BYTE_3;

void DOCOEX::RunIt(int flg)
{
	int IROW=0, ISP=0, MROW=0, MSP=0;
	int IAROW=0, IBROW=0, MAROW=0, MBROW=0;
	int AIJ, BIJ, AMJ, BMJ, AIK, BMK;
	int I, J, K, M, IJ, IK, MJ, MK;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	::SetVal(COMMOD9->COEXST,COMMOD9->NSPC*COMMOD9->NSPC, (BYTE)0);

	if (COMMOD9->NSPC == 0) return;

	if (flg == 1) return ;           

	for (I = 0; I < COMMOD9->NSPC; I++)
	{
		COMMOD9->COEXST[I*COMMOD9->NSPC+I] = 3;
	}

	if (flg == 0) goto Label7350;

	//-------------------------------------------------------------------
	//    CALCULATE THE COEXISTENCE MATRIX.          
	//-------------------------------------------------------------------			
	for (J = 0; J < COMMOD9->NSCT; J++)
	{

		for (I = 0; I < COMMOD9->NEVNT; I++)
		{                  
			AIJ = -1;
			BIJ = -1;

			if ((COMMOD9->IROWS[I][1] != 1) && (COMMOD9->IROWS[I][1] != 3)) continue;

			if (COMMOD9->IROWS[I][1] == 1)
			{
				IAROW = I;

				if (COMMOD9->ISTATIC_1[J*COMMOD9->NEVNT+IAROW] == 3) continue;

				ISP = COMMOD9->IROWS[IAROW][2];

				IBROW = COMMOD9->IROWS[IAROW][3];

				if (COMMOD9->ISTATIC_1[J*COMMOD9->NEVNT+IBROW] == 3) continue;
				if (COMMOD9->ISTATIC_1[J*COMMOD9->NEVNT+IAROW] == 1)
				{
					if (!(::RZERO(IAROW,J))) AIJ = COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+IAROW];
					if (!(::RZERO(IBROW,J))) BIJ = COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+IBROW];                          
				}
			}
			else if (COMMOD9->IROWS[I][1] == 3)
			{                       
				IROW = I;
				if (COMMOD9->ISTATIC_1[J*COMMOD9->NEVNT+IROW] == 1)
				{
					if (!(::RZERO(IROW,J))) AIJ = COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+IROW];
				}
				else if (COMMOD9->ISTATIC_1[J*COMMOD9->NEVNT+IROW] == 2)
				{
					if (!(::RZERO(IROW,J))) BIJ = COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+IROW];
				}
			}

			if ((BIJ >= 0) && (AIJ > BIJ))
			{
				COMMOD9->STOPF = 1;                    
			}

			for (M = 0; M < COMMOD9->NEVNT; M++)
			{
				AMJ = -1;
				BMJ = -1;
				if ((COMMOD9->IROWS[M][1] != 1) && (COMMOD9->IROWS[M][1] != 3)) continue;
				if (COMMOD9->IROWS[M][1] == 1)
				{
					MAROW = M;

					if (COMMOD9->ISTATIC_1[J*COMMOD9->NEVNT+MAROW] == 3) continue;

					MSP = COMMOD9->IROWS[MAROW][2];

					MBROW = COMMOD9->IROWS[MAROW][3];

					if (COMMOD9->ISTATIC_1[J*COMMOD9->NEVNT+MBROW] == 3) continue;
					if (COMMOD9->ISTATIC_1[J*COMMOD9->NEVNT+MAROW] == 1)
					{                             
						if (!(::RZERO(MAROW, J))) AMJ = COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+MAROW];
						if (!(::RZERO(MBROW, J))) BMJ = COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+MBROW];                             
					}
				}
				else if (COMMOD9->IROWS[M][1] == 3)
				{                           
					MROW = M;
					if (COMMOD9->ISTATIC_1[J*COMMOD9->NEVNT+MROW] == 1)
					{
						if (!(::RZERO(MROW, J))) AMJ = COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+MROW];
					}
					else if (COMMOD9->ISTATIC_1[J*COMMOD9->NEVNT+MROW] == 2)
					{
						if (!(::RZERO(MROW, J))) BMJ = COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+MROW];
					}
				}

				if (AIJ >= 0)
				{

					if (AIJ == AMJ)
					{
						COMMOD9->COEXST[ISP*COMMOD9->NSPC+MSP] = max(COMMOD9->COEXST[ISP*COMMOD9->NSPC+MSP], (BYTE)2);
						COMMOD9->COEXST[MSP*COMMOD9->NSPC+ISP] = max(COMMOD9->COEXST[MSP*COMMOD9->NSPC+ISP], (BYTE)2);

						if ((BIJ >= 0) && (BMJ >= 0) && (AIJ != BIJ) && (AMJ != BMJ))
						{
							COMMOD9->COEXST[ISP*COMMOD9->NSPC+MSP] = 3;
							COMMOD9->COEXST[MSP*COMMOD9->NSPC+ISP] = 3;
						}

					}
					else if (AIJ == BMJ)
					{
						COMMOD9->COEXST[ISP*COMMOD9->NSPC+MSP] = max(COMMOD9->COEXST[ISP*COMMOD9->NSPC+MSP], (BYTE)2);
						COMMOD9->COEXST[MSP*COMMOD9->NSPC+ISP] = max(COMMOD9->COEXST[MSP*COMMOD9->NSPC+ISP], (BYTE)2);                               
					}
					else if ((AMJ >= 0) && (BIJ >= 0) && (AIJ < AMJ) && (AMJ < BIJ))
					{
						COMMOD9->COEXST[ISP*COMMOD9->NSPC+MSP] = 3;
						COMMOD9->COEXST[MSP*COMMOD9->NSPC+ISP] = 3;
					}
					else if ((BMJ >= 0) && (BIJ >= 0) && (AIJ < BMJ) && (BMJ < BIJ))
					{
						COMMOD9->COEXST[ISP*COMMOD9->NSPC+MSP] = 3;
						COMMOD9->COEXST[MSP*COMMOD9->NSPC+ISP] = 3;
					}
				}


				if (BIJ >= 0)
				{                           
					if (BIJ == BMJ)
					{
						COMMOD9->COEXST[ISP*COMMOD9->NSPC+MSP] = max(COMMOD9->COEXST[ISP*COMMOD9->NSPC+MSP], (BYTE)2);
						COMMOD9->COEXST[MSP*COMMOD9->NSPC+ISP] = max(COMMOD9->COEXST[MSP*COMMOD9->NSPC+ISP], (BYTE)2);

						if ((AIJ >= 0) && (AMJ >= 0) && (AIJ != BIJ) && (AMJ != BMJ))
						{
							COMMOD9->COEXST[ISP*COMMOD9->NSPC+MSP] = 3;
							COMMOD9->COEXST[MSP*COMMOD9->NSPC+ISP] = 3;
						}                              

					}
					else if (BIJ == AMJ)
					{
						COMMOD9->COEXST[ISP*COMMOD9->NSPC+MSP] = max(COMMOD9->COEXST[ISP*COMMOD9->NSPC+MSP], (BYTE)2);
						COMMOD9->COEXST[MSP*COMMOD9->NSPC+ISP] = max(COMMOD9->COEXST[MSP*COMMOD9->NSPC+ISP], (BYTE)2);
					}
				}
			}
		}
	}			

	if (COMMOD9->STOPF == 1)
	{
		cout<<"___________________"<<endl;
		cout<<"INVALID INPUT DATA"<<endl;
		return;
	}


	for (I = 0; I < COMMOD9->NEVNT - 1; I++)
	{
		if ((COMMOD9->IROWS[I][1] > 2) || (COMMOD9->IROWS[I][1] < 1)) continue;
		if (allrzero[I]) continue;

		for (M = I + 1; M < COMMOD9->NEVNT; M++)
		{
			if ((COMMOD9->IROWS[M][1] > 2)) continue;

			if (allrzero[M]) continue;

			ISP = COMMOD9->IROWS[I][2];
			MSP = COMMOD9->IROWS[M][2];

			if (COMMOD9->COEXST[ISP*COMMOD9->NSPC+MSP] > 1) continue;

			for (J = 0; J < COMMOD9->NSCT - 1; J++)
			{
				if ((::RZERO(I, J)) || (::RZERO(M, J))) continue;

				IJ = -1;
				MJ = -1;
				IJ = COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+I];
				MJ = COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+M];

				if ((IJ == -1) || (MJ == -1)) break;

				for (K = J + 1; K < COMMOD9->NSCT; K++)
				{
					if ((::RZERO(I, K)) || (::RZERO(M, K))) continue;

					IK = -1;
					MK = -1;
					IK = COMMOD9->ISTATIC_0[K*COMMOD9->NEVNT+I];
					MK = COMMOD9->ISTATIC_0[K*COMMOD9->NEVNT+M];

					if ((IK == -1) || (MK == -1)) continue;

					if (((IJ < MJ) && (IK > MK)) ||
							((IJ > MJ) && (IK < MK)))
					{
						COMMOD9->COEXST[ISP*COMMOD9->NSPC+MSP] = 2;
						COMMOD9->COEXST[MSP*COMMOD9->NSPC+ISP] = 2;
						break;
					}
				}

				if (COMMOD9->COEXST[ISP*COMMOD9->NSPC+MSP] > 1) break;
			}
		}
	}


	for (I = 0; I < COMMOD9->NEVNT - COMMOD9->NSPC - COMMOD9->NMIDS - 1; I++)
	{
		::FINDROW(I, 1, IAROW);

		if (IAROW == -1) continue;

		ISP = COMMOD9->IROWS[IAROW][2];

		for (M = I + 1; M < COMMOD9->NEVNT - COMMOD9->NSPC - COMMOD9->NMIDS; M++)
		{                  
			::FINDROW(M, 1, MAROW);

			if (MAROW == -1) continue;

			MSP = COMMOD9->IROWS[MAROW][2];

			if (COMMOD9->COEXST[ISP*COMMOD9->NSPC+MSP] == 0)
			{
				for (J = 0; J < COMMOD9->NSCT; J++)
				{
					BIJ = -1;
					AMJ = -1;
					IBROW = COMMOD9->IROWS[IAROW][3];

					if (!(::RZERO(IBROW, J))) BIJ = COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+IBROW];
					if (!(::RZERO(MAROW, J))) AMJ = COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+MAROW];

					if ((BIJ > 0) && (BIJ < AMJ))
					{                                
						for (K = 0; K < COMMOD9->NSCT; K++)
						{
							if (J == K) continue;

							BMK = -1;
							AIK = -1;
							MBROW = COMMOD9->IROWS[MAROW][3];

							if (!(::RZERO(MBROW, K))) BMK = COMMOD9->ISTATIC_0[K*COMMOD9->NEVNT+MBROW];
							if (!(::RZERO(IAROW, K))) AIK = COMMOD9->ISTATIC_0[K*COMMOD9->NEVNT+IAROW];

							if ((BMK > 0) && (BMK < AIK))
							{                                        
								COMMOD9->COEXST[ISP*COMMOD9->NSPC+MSP] = 2;
								COMMOD9->COEXST[MSP*COMMOD9->NSPC+ISP] = 2;
							}
						}
					}
				}
			}
		}
	}


Label7350:

	if ((flg != 1) && (COMMOD9->CDF != 1))
	{              
		for (I = 0; I < COMMOD9->NSPC; I++)
		{
			if (flg == 2)
			{
				//nothing
			}
			else if (flg == 3)
			{
				//nothing
			}
		}             

	}

	return;
}
