#include <iostream>
#include <math.h>
#include "COMMOD.h"
#include "Helper.h"
#include "DOFB4L.h"

using namespace std;
using std::string;

DOFB4L* DOFB4L::g_singleton=NULL;
extern	bool	*allrzero;
void DOFB4L::RunIt()
{
	int ISP=0, MSP=0;
	int IAROW=0, MBROW=0;
	int AIJ=0, BMJ=0;
	int I=0, J=0, M=0;

	COMMOD* COMMOD9 = COMMOD::Singleton();
	
        ::SetVal(COMMOD9->FADLAD,COMMOD9->NSPC*COMMOD9->NSPC, (BYTE)0);

	COMMOD9->NFALA = 0;

	if (COMMOD9->NSPC == 0)  return;
	
	for (I = 0; I < COMMOD9->NSPC; I++)
	{
		COMMOD9->FADLAD[I*COMMOD9->NSPC+I] = 1;
	}
	
	for (I = 0; I < COMMOD9->NEVNT; I++)
	{
	
		if (COMMOD9->IROWS[I][1] != 1) continue;
		
		if (allrzero[I]) continue;

		IAROW = I;
		ISP = COMMOD9->IROWS[IAROW][2];
		

		for (M = 0; M < COMMOD9->NEVNT; M++)
		{

			if (COMMOD9->IROWS[M][1] != 2) continue;
			
			if (allrzero[M]) continue;

			MBROW = M;
			MSP = COMMOD9->IROWS[MBROW][2];
			
			if (COMMOD9->FADLAD[ISP*COMMOD9->NSPC+MSP] == 1) continue;
			
			for (J = 0; J < COMMOD9->NSCT; J++)
			{
			AIJ = -1;

			
			if (COMMOD9->ISTATIC_1[J*COMMOD9->NEVNT+IAROW] != 1) continue;

				
			if (!(::RZERO(IAROW, J))) AIJ = COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+IAROW];
				                                        
			if (AIJ == -1) continue;

			BMJ = -1;

				
			if (COMMOD9->ISTATIC_1[J*COMMOD9->NEVNT+MBROW] != 2) continue;

				   
			if (!(::RZERO(MBROW, J))) BMJ = COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+MBROW];

			if (BMJ == -1) continue;

				
			if (AIJ < BMJ)
			{
				COMMOD9->FADLAD[ISP*COMMOD9->NSPC+MSP] = 1;
			}
			else if ((AIJ == BMJ) && (COMMOD9->COXSTF == 2))
			{
				COMMOD9->FADLAD[ISP*COMMOD9->NSPC+MSP] = 1;
				COMMOD9->FADLAD[MSP*COMMOD9->NSPC+ISP] = 1;

			}
			}
		}
	}


	return;
}
