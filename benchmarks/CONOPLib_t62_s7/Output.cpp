#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include "COMMOD.h"
#include "Output.h"

using namespace std;

void Output::STARTOUT(int myperm[])
{
	int L;
	COMMOD* COMMOD9 = COMMOD::Singleton();

	if ((COMMOD9->INITSOL.substr(0, 3) != "OFF") && (COMMOD9->CDF != 1))
	{
		fstream _file(COMMOD9->INITSOL.c_str(),ios::out);

		for (L = 0; L < COMMOD9->NEVNT; L++)
		{
			_file<<::setw(7)<<COMMOD9->IROWS[myperm[L]][0] + 1<<::setw(7)<<COMMOD9->IROWS[myperm[L]][1]<<::setw(7)<<L+1<<"\r\n";
		}

		_file.close();

	}
}

void Output::STEPOUT(int myperm[])
{
	int L;
	COMMOD* COMMOD9 = COMMOD::Singleton();

	if ((COMMOD9->STEPSOL.substr(0, 3) != "OFF") && (COMMOD9->CDF != 1))
	{
		fstream _file(COMMOD9->STEPSOL.c_str(),ios::out);

		for (L = 0; L < COMMOD9->NEVNT; L++)
		{
			_file<<::setw(7)<<COMMOD9->IROWS[myperm[L]][0] + 1<<::setw(7)<<COMMOD9->IROWS[myperm[L]][1]<<::setw(7)<<L+1<<"\r\n";
		}

		_file.close();

	}
}

void Output::COEXOUT()
{
	int I, M;
	COMMOD* COMMOD9 = COMMOD::Singleton();
	ofstream _file;
	_file.open(COMMOD9->COEXFILE.c_str());
	for (I = 0; I < COMMOD9->NSPC; I++)
	{
		for (M = 0; M < COMMOD9->NSPC; M++)
		{
			if (I < M&&COMMOD9->COEXST[I*COMMOD9->NSPC+M] != 0) {
				_file << I << " " << M << " " << (int)COMMOD9->COEXST[I*COMMOD9->NSPC+M] << "\r\n";
			}
		}
	}
	_file.close();
}
void Output:: FADLADOUT()
{
	int I, M;
	COMMOD* COMMOD9 = COMMOD::Singleton();
	ofstream _file;
	_file.open(COMMOD9->FB4LFILE.c_str());
	for (I = 0; I < COMMOD9->NSPC; I++)
	{
		for (M = 0; M < COMMOD9->NSPC; M++)
		{
			if (COMMOD9->FADLAD[I*COMMOD9->NSPC+M] != 0) {
				_file << I << " " << M << "\r\n";				
			}
		}
	}
	_file.close();
}

