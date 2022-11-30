#include <iostream>
#include <math.h>
#include "COMMOD.h"
#include "Helper.h"
#include "DOPTLORD.h"

using namespace std;
using std::string;

DOPTLORD* DOPTLORD::g_singleton=NULL;

void DOPTLORD::UPDATE(int IRW, int MRW, int rong, int rite, int pass)
{
	COMMOD* COMMOD9 = COMMOD::Singleton();

	if (COMMOD9->PTLORD[IRW][MRW] != (int)rong)
	{
		COMMOD9->PTLORD[IRW][MRW] = (int)rite;
	}
	else
	{	
		cout<<"____________________________________"<<endl;
		cout<<"INCONSISTENT ORDERING RELATIONSHIPS!"<<endl;
	}
}

void DOPTLORD::RunIt()
{
	int I=0, J=0, K=0, Jdef=0;
	int IROW=0, MROW=0, KROW=0;
	int LVLIJ=0, LVLMJ=0, IFAD=0, ILAD=0;
	double IUP=0, IDN=0, MUP=0, MDN=0;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	if (COMMOD9->NOTHR <= 0) return;

	::SetVal(COMMOD9->PTLORD,COMMOD9->NEVNT,COMMOD9->NEVNT+2-(COMMOD9->NSPC*2), 0);	

	for (int n = 0; n < COMMOD9->NEVNT; n++)
	{
		if (COMMOD9->IROWS[n][1] > 5)
		{
			cout<<"IROWS["<<n<<",1]="<<COMMOD9->IROWS[n][1]<<endl;
		}
	}

	int IDX = 0;

	for (IROW = 0; IROW < COMMOD9->NEVNT; IROW++)
	{
		IUP = 0.00;
		IDN = 0.00;
		Jdef = 0;

		if (COMMOD9->IROWS[IROW][1] == 5)
		{
			for (J = 0; J < COMMOD9->NSCT; J++)
			{				
				if (COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+IROW] >= 0)
				{
					IDX = COMMOD9->IDX_SCT_RSTATIC[J] + IROW * 2;
					if (IDN == 0.00)
					{					
						IUP = min(COMMOD9->NRSTATIC[IDX], COMMOD9->NRSTATIC[IDX+1]);
						IDN = max(COMMOD9->NRSTATIC[IDX], COMMOD9->NRSTATIC[IDX + 1]);
						Jdef = J;
					}

				}

				if (IUP > 0.00) 
					break;

			}			


			for (MROW = IROW + 1; MROW < COMMOD9->NEVNT; MROW++)
			{
				if ((COMMOD9->IROWS[MROW][1] == 5) &&
						(IROW != MROW))
				{
					MUP = 0.00;
					MDN = 0.00;

					for (K = 0; K < COMMOD9->NSCT; K++)
					{
						if (COMMOD9->ISTATIC_0[MROW+K*COMMOD9->NEVNT] >= 0)
						{						
							IDX = COMMOD9->IDX_SCT_RSTATIC[K] + MROW * 2;
							MUP = min(COMMOD9->NRSTATIC[IDX], COMMOD9->NRSTATIC[IDX + 1]);
							MDN = max(COMMOD9->NRSTATIC[IDX], COMMOD9->NRSTATIC[IDX + 1]);

							if (MDN < IUP)
							{
								UPDATE(MROW, IROW, 1, 2, 0);
								UPDATE(IROW, MROW, 2, 1, 0);
							}

							if (IDN < MUP)
							{	
								UPDATE(IROW, MROW, 1, 2, 0);
								UPDATE(MROW, IROW, 2, 1, 0);
							}

						}

						if (MUP > 0.00) break;
					}
				}
			}
		}
	}

	IFAD = 0;
	ILAD = 0;	
	for (IROW = 0; IROW < COMMOD9->NEVNT; IROW++)
	{
		if (COMMOD9->IROWS[IROW][1] == 3)
		{
			for (MROW = 0; MROW < COMMOD9->NEVNT; MROW++)
			{
				if (COMMOD9->IROWS[IROW][0] != COMMOD9->IROWS[MROW][0]) continue;

				if (COMMOD9->IROWS[MROW][1] == 1)
				{
					IFAD = 1;					
					UPDATE(MROW, IROW, 2, 1, 0);
				}
				else if (COMMOD9->IROWS[MROW][1] == 2)
				{
					ILAD = 1;					
					UPDATE(MROW, IROW, 1, 2, 0);
				}
			}

		}

		IFAD = 0;
		ILAD = 0;
	}


	for (IROW = 0; IROW < COMMOD9->NEVNT; IROW++)
	{
		if (COMMOD9->IROWS[IROW][1] == 11)
		{		
			MROW = COMMOD9->IROWS[IROW][3];			
			UPDATE(MROW, IROW, 1, 2, 0);
			UPDATE(IROW, MROW, 2, 1, 0);

		}
	}


	for (IROW = 0; IROW < COMMOD9->NEVNT; IROW++)
	{
		for (J = 0; J < COMMOD9->NSCT; J++)
		{
			if (!(::IORZERO(IROW, J)))
			{
				LVLIJ = COMMOD9->ISTATIC_0[IROW+J*COMMOD9->NEVNT];

				for (MROW = 0; MROW < COMMOD9->NEVNT; MROW++)
				{
					if (MROW == IROW)
					{
						if (((COMMOD9->IROWS[MROW][1] > 2) || (COMMOD9->IROWS[MROW][1] < 1)))
						{							
							COMMOD9->PTLORD[IROW][MROW] = 3;
						}
					}
					else if (((COMMOD9->IROWS[MROW][1] > 2) || (COMMOD9->IROWS[MROW][1] < 1)) &&
							!(::IORZERO(MROW, J)))
					{

						LVLMJ = COMMOD9->ISTATIC_0[MROW+J*COMMOD9->NEVNT];

						if ((COMMOD9->ISTATIC_1[IROW+J*COMMOD9->NEVNT] == 0) && (LVLIJ > LVLMJ) &&
								((COMMOD9->ISTATIC_1[MROW+J*COMMOD9->NEVNT] == 0) || (COMMOD9->ISTATIC_1[MROW+J*COMMOD9->NEVNT] == 1)))
						{
							UPDATE(IROW, MROW, 1, 2, J);
						}

						else if ((COMMOD9->ISTATIC_1[IROW+J*COMMOD9->NEVNT] == 0) && (LVLIJ < LVLMJ) &&
								((COMMOD9->ISTATIC_1[MROW+J*COMMOD9->NEVNT] == 0) || (COMMOD9->ISTATIC_1[MROW+J*COMMOD9->NEVNT] == 2)))
						{
							UPDATE(IROW, MROW, 2, 1, J);
						}

						else if ((COMMOD9->ISTATIC_1[IROW+J*COMMOD9->NEVNT] == 1) && (LVLIJ < LVLMJ) &&
								((COMMOD9->ISTATIC_1[MROW+J*COMMOD9->NEVNT] == 0) || (COMMOD9->ISTATIC_1[MROW+J*COMMOD9->NEVNT] == 2)))
						{
							UPDATE(IROW, MROW, 2, 1, J);
						}

						else if ((COMMOD9->ISTATIC_1[IROW+J*COMMOD9->NEVNT] == 2) && (LVLIJ > LVLMJ) &&
								((COMMOD9->ISTATIC_1[MROW+J*COMMOD9->NEVNT] == 0) || (COMMOD9->ISTATIC_1[MROW+J*COMMOD9->NEVNT] == 1)))
						{
							UPDATE(IROW, MROW, 1, 2, J);
						}

					}
				}
			}
		}
	}


	for (IROW = 0; IROW < COMMOD9->NEVNT; IROW++)
	{		
		for (MROW = 0; MROW < COMMOD9->NEVNT - (COMMOD9->NSPC * 2); MROW++)
		{
			if (COMMOD9->PTLORD[IROW, MROW] != 0)
			{				
				for (KROW = 0; KROW < COMMOD9->NEVNT - (COMMOD9->NSPC * 2); KROW++)
				{				
					if ((COMMOD9->PTLORD[IROW][MROW] == 1) &&
							(COMMOD9->PTLORD[MROW][KROW] == 1))
					{
						UPDATE(IROW, KROW, 2, 1, 0 - IROW);
					}

					else if ((COMMOD9->PTLORD[IROW][MROW] == 2) &&
							(COMMOD9->PTLORD[MROW][KROW] == 2))
					{
						UPDATE(IROW, KROW, 1, 2, 0 - IROW);
					}
				}
			}
		}
	}

	for (IROW = 0; IROW < COMMOD9->NEVNT; IROW++)
	{
		for (MROW = 0; MROW < COMMOD9->NEVNT - (COMMOD9->NSPC * 2); MROW++)
		{
			if (COMMOD9->PTLORD[IROW, MROW] != 0)
			{
				for (KROW = 0; KROW < COMMOD9->NEVNT - (COMMOD9->NSPC * 2); KROW++)
				{
					if ((COMMOD9->PTLORD[IROW][MROW] == 1) &&
							(COMMOD9->PTLORD[MROW][KROW] == 1))
					{
						UPDATE(IROW, KROW, 2, 1, 0 - IROW);
					}
					else if ((COMMOD9->PTLORD[IROW][MROW] == 2) &&
							(COMMOD9->PTLORD[MROW][KROW] == 2))
					{
						UPDATE(IROW, KROW, 1, 2, 0 - IROW);
					}
				}
			}
		}
	}

	for (IROW = 0; IROW < COMMOD9->NEVNT; IROW++)
	{		
		for (MROW = 0; MROW < COMMOD9->NEVNT - (COMMOD9->NSPC * 2); MROW++)
		{
			if (COMMOD9->PTLORD[IROW][MROW] == 1)
			{
				COMMOD9->PTLORD[IROW][COMMOD9->NEVNT  - (COMMOD9->NSPC * 2)] =
					COMMOD9->PTLORD[IROW][COMMOD9->NEVNT  - (COMMOD9->NSPC * 2)] + 1;
			}
			else if (COMMOD9->PTLORD[IROW][MROW] == 2)
			{
				COMMOD9->PTLORD[IROW][COMMOD9->NEVNT + 1- (COMMOD9->NSPC * 2)] =
					COMMOD9->PTLORD[IROW][COMMOD9->NEVNT + 1 - (COMMOD9->NSPC * 2)] + 1;
			}
		}
	}

	if (COMMOD9->CDF != 1)
	{		
		for (I = 0; I < COMMOD9->NEVNT; I++)
		{
			//nothing
		}		
	}


	return;
}
