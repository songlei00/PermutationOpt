#include <iostream>
#include <math.h>
#include "COMMOD.h"
#include "Helper.h"
#include "DORASC.h"

using namespace std;
using std::string;

DORASC* DORASC::g_singleton=NULL;
extern	bool	*allrzero;
void DORASC::RunIt()
{
	int IJ, MJ;
	int I, J, M;            

	COMMOD* COMMOD9 = COMMOD::Singleton();

	::SetVal(COMMOD9->RASC,COMMOD9->NEVNT,COMMOD9->NEVNT,(short int) 0);

	if (COMMOD9->NEVNT == 0)return;

	for (I = 0; I < COMMOD9->NEVNT - 1; I++)
	{
		if (allrzero[I]) continue;

		for (M = I + 1; M < COMMOD9->NEVNT; M++)
		{                  
			if (allrzero[M]) continue;

			for (J = 0; J < COMMOD9->NSCT; J++)
			{
				IJ = 0;

				if (!(::RZERO(I, J))) IJ = COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+I];

				if (IJ == 0) continue;

				MJ = 0;

				if (!(::RZERO(M, J))) MJ = COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+M];

				if (MJ == 0) continue;

				if (IJ < MJ)
				{
					COMMOD9->RASC[I][M] = COMMOD9->RASC[I][M] + 1;
				}
				else if (MJ < IJ)
				{
					COMMOD9->RASC[M][I] = COMMOD9->RASC[M][I] + 1;
				}
			}
		}
	}

	return;
}
