#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include "COMMOD.h"
#include "DOINPUT.h"
#include "Helper.h"
#include<stdio.h> 
using namespace std;
using std::string;

DOINPUT* DOINPUT::g_singleton=NULL;


void DOINPUT::BeforeRunIt(int MYID)
{
	int i,j;

	COMMOD* COMMOD9 = COMMOD::Singleton();

	COMMOD9->CALIBF = 0;


	COMMOD9->PERM = new int[COMMOD9->NEVNT];
	::SetVal(COMMOD9->PERM,COMMOD9->NEVNT,0);

	COMMOD9->INIPERM = new int[COMMOD9->NEVNT];
	::SetVal(COMMOD9->INIPERM,COMMOD9->NEVNT,0);

	COMMOD9->PXLPERM = new int[COMMOD9->NEVNT];
	::SetVal(COMMOD9->PXLPERM,COMMOD9->NEVNT,0);

	COMMOD9->TEMPERM = new int[COMMOD9->NEVNT];
	::SetVal(COMMOD9->TEMPERM,COMMOD9->NEVNT,0);

	COMMOD9->BSTPERM = new int[COMMOD9->NEVNT];
	::SetVal(COMMOD9->BSTPERM,COMMOD9->NEVNT,0);

	COMMOD9->LSTPERM = new int[COMMOD9->NEVNT];
	::SetVal(COMMOD9->LSTPERM,COMMOD9->NEVNT,0);



	COMMOD9->HLEVEL = new int[COMMOD9->NSCT + 10];
	::SetVal(COMMOD9->HLEVEL,COMMOD9->NSCT + 10,0);

	COMMOD9->LIMITS = new int*[COMMOD9->NSCT];
	for(i=0;i<COMMOD9->NSCT;i++)COMMOD9->LIMITS[i]=new int[2];
	::SetVal(COMMOD9->LIMITS,COMMOD9->NSCT,2,0);

	COMMOD9->COVER = new int[COMMOD9->NEVNT];
	::SetVal(COMMOD9->COVER,COMMOD9->NEVNT,0);


	COMMOD9->COEXST = new BYTE [COMMOD9->NSPC*COMMOD9->NSPC];			
	::SetVal(COMMOD9->COEXST, COMMOD9->NSPC*COMMOD9->NSPC, (BYTE)0);			

	COMMOD9->PTLORD = new int*[COMMOD9->NEVNT];
	for(i=0;i<COMMOD9->NEVNT;i++)COMMOD9->PTLORD[i]=new int[COMMOD9->NEVNT + 2 - (COMMOD9->NSPC * 2)];
	::SetVal(COMMOD9->PTLORD,COMMOD9->NEVNT,COMMOD9->NEVNT + 2 - (COMMOD9->NSPC * 2),0);

	if (COMMOD9->FB4LF != 0)
	{
		COMMOD9->FADLAD = new BYTE[COMMOD9->NSPC*COMMOD9->NSPC];			
		::SetVal(COMMOD9->FADLAD,COMMOD9->NSPC*COMMOD9->NSPC,(BYTE)0);
	}
	if (MYID == 0) {
		if (COMMOD9->PENF == 4)
		{
			COMMOD9->RASC = new short int*[COMMOD9->NEVNT];
			for (i = 0; i<COMMOD9->NEVNT; i++)COMMOD9->RASC[i] = new short int[COMMOD9->NEVNT];
			::SetVal(COMMOD9->RASC, COMMOD9->NEVNT, COMMOD9->NEVNT, (short int)0);
			cout << "COMMOD9->RASC ����" << MYID << endl;
		}				
	}           



	COMMOD9->ELEVEL= new int*[COMMOD9->NSCT];
	for (i = 0; i<COMMOD9->NSCT; i++)COMMOD9->ELEVEL[i] = new int[COMMOD9->MXLVL + 2];
	::SetVal(COMMOD9->ELEVEL, COMMOD9->NSCT, COMMOD9->MXLVL + 2, 0);


	COMMOD9->NEGATIVE = new int*[COMMOD9->NSPC];
	for(i=0;i<COMMOD9->NSPC;i++)COMMOD9->NEGATIVE[i]=new int[COMMOD9->NSCT + 4];
	::SetVal(COMMOD9->NEGATIVE,COMMOD9->NSPC,COMMOD9->NSCT + 4,0);


	COMMOD9->IROWS = new int*[COMMOD9->NEVNT];
	for(i=0;i<COMMOD9->NEVNT;i++)COMMOD9->IROWS[i]=new int[6];
	::SetVal(COMMOD9->IROWS,COMMOD9->NEVNT,6,0);


	{
		int RSTATICCount = 0;
		int h_Level = 0;
		COMMOD9->IDX_SCT_RSTATIC = new int[COMMOD9->NSCT];
		for (int i = 0; i < COMMOD9->NSCT; i++)
		{
			COMMOD9->IDX_SCT_RSTATIC[i] = RSTATICCount;
			RSTATICCount += COMMOD9->NEVNT * 2;
		}
		COMMOD9->NRSTATIC = new float[RSTATICCount];
		COMMOD9->RSTATICCount = RSTATICCount;				
		memset(COMMOD9->NRSTATIC, 0, sizeof(float)*RSTATICCount);
	}		



	COMMOD9->ISTATIC_0=new short int[(COMMOD9->NSCT + 10)*COMMOD9->NEVNT];
	COMMOD9->ISTATIC_1 = new BYTE[(COMMOD9->NSCT + 10)*COMMOD9->NEVNT];
	::SetVal(COMMOD9->ISTATIC_0, (COMMOD9->NSCT+10)* COMMOD9->NEVNT, (short int)0);
	::SetVal(COMMOD9->ISTATIC_1, (COMMOD9->NSCT+10)* COMMOD9->NEVNT, (BYTE)0);	

	COMMOD9->VALEVEL= new float*[COMMOD9->NSCT];
	for (i = 0; i<COMMOD9->NSCT; i++)COMMOD9->VALEVEL[i] = new float[COMMOD9->MXLVL + 3];
	::SetVal(COMMOD9->VALEVEL, COMMOD9->NSCT, COMMOD9->MXLVL + 3, 0.0f);

	COMMOD9->COMPLVL = new float*[COMMOD9->NEVNT + 3];
	for(i=0;i<COMMOD9->NEVNT+3;i++)COMMOD9->COMPLVL[i]=new float[10];
	::SetVal(COMMOD9->COMPLVL,COMMOD9->NEVNT + 3,10,0.0f);

	COMMOD9->SCJPEN = new double[COMMOD9->NSCT];
	::SetVal(COMMOD9->SCJPEN,COMMOD9->NSCT,0.0);

	COMMOD9->COLPEN = new double[COMMOD9->NSCT];
	::SetVal(COMMOD9->COLPEN,COMMOD9->NSCT,0.0);



	COMMOD9->SCTSOL_SIZE = (COMMOD9->NSCT + 10)*COMMOD9->NEVNT;
	COMMOD9->HSCTSOL = new unsigned short[COMMOD9->SCTSOL_SIZE];		
	memset(COMMOD9->HSCTSOL, 0, sizeof(unsigned short)*COMMOD9->SCTSOL_SIZE);

	COMMOD9->SCTSOL = new unsigned short[COMMOD9->SCTSOL_SIZE];			
	memset(COMMOD9->SCTSOL, 0, sizeof(unsigned short)*COMMOD9->SCTSOL_SIZE);

	COMMOD9->BSCTSOL = new unsigned short[COMMOD9->SCTSOL_SIZE];			
	memset(COMMOD9->BSCTSOL, 0, sizeof(unsigned short)*COMMOD9->SCTSOL_SIZE);

COMMOD9->LOADNAMES=false;

	if (COMMOD9->LOADNAMES) {


		COMMOD9->SECTNAME = new string[COMMOD9->NSCT];
		COMMOD9->SECTNICK = new string[COMMOD9->NSCT];

		COMMOD9->LABLNAME = new string[COMMOD9->MXLBL];
		COMMOD9->EVNTNAME = new string[COMMOD9->NEVNT - COMMOD9->NSPC - COMMOD9->NMIDS];
		COMMOD9->EVNTNICK = new string[COMMOD9->NEVNT - COMMOD9->NSPC - COMMOD9->NMIDS];

		COMMOD9->ETAGNAME = new string[COMMOD9->MXLBL];
		COMMOD9->STAGNAME = new string[COMMOD9->MXLBL];
		for (I = 0; I < COMMOD9->MXLBL; I++)
		{
			COMMOD9->ETAGNAME[I] = "not assigned";
			COMMOD9->STAGNAME[I] = "not assigned";
		}
	}         

	COMMOD9->NMIDS=0;



	COMMOD9->SECTPERM = new int[COMMOD9->NSCT];
	for(i=0;i<COMMOD9->NSCT;i++)COMMOD9->SECTPERM[i]=-1;

	COMMOD9->SECTPROP = new int*[COMMOD9->NSCT];
	for(i=0;i<COMMOD9->NSCT;i++)COMMOD9->SECTPROP[i]=new int[4];
	::SetVal(COMMOD9->SECTPROP,COMMOD9->NSCT,4,0);

	COMMOD9->CULLIST = new int*[COMMOD9->NEVNT];
	for(i=0;i<COMMOD9->NEVNT;i++)COMMOD9->CULLIST[i]=new int[6];
	::SetVal(COMMOD9->CULLIST,COMMOD9->NEVNT,6,0);

	COMMOD9->MXJTAG=0;



	if (COMMOD9->RUNGRF >= 4 || COMMOD9->GRIDF == 1)
	{
		if (COMMOD9->BESTFILE.substr(0, 3) == "ADD" || COMMOD9->BESTFILE.substr(0, 3) == "add")
		{                 
			if (::fileExists(COMMOD9->BESTFILE.substr(3)))
			{
				cout<<"BESTFILE Exists"<<endl;                      
				cout<<"  Loading Event/Position Archive . . ."<<endl;

			}
			else
			{
				//nothing
			}
		}
		else
		{
			//nothing
		}


		if ((COMMOD9->VERYBEST - COMMOD9->USERBEST) < -0.01)
		{
			//nothing
		}
		else if ((COMMOD9->VERYBEST - COMMOD9->USERBEST) > 0.01)
		{
			//nothing
		}


		if (COMMOD9->GRID2F == 1)
		{             

			if (COMMOD9->SCNDFILE.substr(0, 3) == "ADD" || COMMOD9->SCNDFILE.substr(0, 3) == "add")
			{                       
				bool hasFile = true;

				if (hasFile)
				{                       
					cout<<"  Loading 2nd Event/Position Archive . . ."<<endl;                         
				}
				else
				{
					//nothing
				}
			}
			else
			{
				//nothing
			}

		}
	}
}

void DOINPUT::RunIt(int MYID)
{

	BeforeRunIt(MYID);


	COMMOD* COMMOD9 = COMMOD::Singleton();          
	DIGIT[0] = "0";
	DIGIT[1] = "1";
	DIGIT[2] = "2";
	DIGIT[3] = "3";
	DIGIT[4] = "4";
	DIGIT[5] = "5";
	DIGIT[6] = "6";
	DIGIT[7] = "7";
	DIGIT[8] = "8";
	DIGIT[9] = "9";
	MINC = 0.00;
	A = 0;
	I = 0;
	L = 0;
	Ilbl = 0;
	Ityp = 0;
	UNPRDF = 0;
	NSCAL = 0;
	NTAG = 0;
	ITAG = 0;
	JTAG = 0;



	HEVENT=0, HTYPE=0, HSECTION=0, UNPRDF=0, LSTSECT=0, LSTEVNT=0;
	IEVENT=0, ITYPE=0, ISECTION=0, ILEVEL=0, IMOVES=0, IPRD=0, IUNPRD=0;
	JPLACE=0, JINC=0, Ityp=0, Ilbl=0, LSTTYPE=0, MFAD=0, MLAD=0;
	IPAIR=0;
	I=0, J=0, K=0, N=0, M=0, IROW=0, JOPT=0, NENDS=0, JTAG=0, ITAG=0, NTAG=0, NSCAL=0;
	A=0, L=0;
	EVENT=0.0, TYP=0.0, SECTION=0.0, VALUE=0.0, LEVEL=0.0, MOVES=0.0, WTUP=0.0f, WTDN=0.0f;
	MNSPACE=0.0, MUSPACE=0.0, MINC=0.0;

	COMMOD9->IRCOUNT = 0;

	string line="";
	::fstream _file;  

	if (COMMOD9->SOLVEF == 0 && !(COMMOD9->NUDGER))
	{
		//nothing
	}
	else if (COMMOD9->SOLVEF == 0 && COMMOD9->NUDGER)
	{
		cout<<"  Simulated annealing with nudges"<<endl;
	}
	else if (COMMOD9->SOLVEF == 1)
	{
		cout<<"  Greedy Algorithm"<<endl;
	}

	if (COMMOD9->PAUSF == 5 && COMMOD9->SOLVEF == 0) cout<<"     - with auto cooling"<<endl;

	if (COMMOD9->PENF == -1)
	{
		cout<<"  Optimize on Eventual Misfit"<<endl;
	}
	else if (COMMOD9->PENF == 0)
	{
		//nothing
	}
	else if (COMMOD9->PENF == 1)
	{
		cout<<"  Optimize on Level Misfit"<<endl;
	}
	else if (COMMOD9->PENF == 2)
	{
		cout<<"  Optimize on Ordinal Misfit"<<endl;
	}
	else if (COMMOD9->PENF == 3)
	{
		cout<<"  Optimize on Spatial Misfit"<<endl;
	}
	else if (COMMOD9->PENF == 4)
	{
		if (COMMOD9->RASCon)
		{
			cout<<"  Optimize on Rascal Misfit with RASC() array"<<endl;
		}
		else
		{
			cout<<"  Optimize on Rascal Misfit without RASC() array"<<endl;
		}

	}
	else if (COMMOD9->PENF == 5)
	{
		cout<<"  Optimize on Royal Misfit"<<endl;
	}
	else if (COMMOD9->PENF == 6 && COMMOD9->FB4LF != 0)
	{
		cout<<"  Optimize on Sequel Misfit"<<endl;
	}        

	if(0==MYID){
		if (COMMOD9->COXSTF == 4)
		{
			cout<<"  No Coexistences Enforced"<<endl;
		}
		else if (COMMOD9->COXSTF == 3)
		{		
			cout<<"  Strict Coexistence Criteria Enforced"<<endl;
		}
		else if (COMMOD9->COXSTF == 2)
		{		
			cout<<"  Loose Coexistence Criteria Enforced"<<endl;
		}
		else if (COMMOD9->COXSTF == 1)
		{
			cout<<"  Coexistences Read from File"<<endl;
		}
		else if (COMMOD9->COXSTF == 0)
		{
			cout<<"  All taxa forced to co-exist"<<endl;
		}
	}


	if (COMMOD9->CONTRACT == 0)
	{
		//nothing
	}
	else if (COMMOD9->CONTRACT == 3)
	{
		cout<<"  All Ranges Allowed to Contract and Shift"<<endl;
	}
	else if (COMMOD9->CONTRACT == 2)
	{
		cout<<"  All Ranges Allowed to Contract, but not Shift"<<endl;
	}
	else if (COMMOD9->CONTRACT == 1)
	{
		cout<<"  Ranges Contract or Shift only as Permitted by Input File"<<endl;
	}


	::trim(COMMOD9->INFILE);
	double* SectionMaxPen=new double[COMMOD9->NSCT];
	int IDX = 0;
	int nBigWeightEvents = 0;
	bool bFind = false;
	int* tmpBigWeightEventsArray = new int[COMMOD9->NEVNT];
	if (COMMOD9->JSTART >= COMMOD9->NSCT)
	{
		COMMOD9->STOPF = 1;               
		goto Label9999;
	}

	LSTSECT = 0;
	LSTEVNT = 0;
	LSTTYPE = 0;
	IEVENT = 0;
	ISECTION = 0;
	NENDS = 0;
	COMMOD9->NROWS = 0;
	COMMOD9->NOBS = 0;
	COMMOD9->NMIDS = 0;
	COMMOD9->NBOXS = 0;
	IPRD = 1;
	IUNPRD = 0;

	::SetVal(COMMOD9->HLEVEL,COMMOD9->NSCT+10, -1);

	::SetVal(COMMOD9->VALEVEL,COMMOD9->NSCT, COMMOD9->MXLVL + 3, 0.0f);

	::SetVal(COMMOD9->ISTATIC_0, (COMMOD9->NSCT+10)*COMMOD9->NEVNT, (short int)-1);
	::SetVal(COMMOD9->ISTATIC_1, (COMMOD9->NSCT+10)*COMMOD9->NEVNT, (BYTE)0);
	
	
	
	COMMOD9->SectionMaxPenOrder = new int[COMMOD9->NSCT];
	if (!::fileExists(COMMOD9->INFILE))
	{
		cout<<" Unable to find file "<<COMMOD9->INFILE<<endl;
		cout<<"  check .CFG file and file manager"<<endl;
		COMMOD9->STOPF = 1;
		goto Label9999;
	}


	_file.open(COMMOD9->INFILE.c_str(),ios::in);

	while(std::getline(_file,line))
	{
		vector<string> elems = ::split(line, ' ');
		::trim(elems[0]);
		::trim(elems[1]);
		::trim(elems[2]);
		::trim(elems[3]);
		::trim(elems[4]);
		::trim(elems[5]);
		::trim(elems[6]);
		::trim(elems[7]);

		EVENT = strtod(elems[0].c_str(),NULL);
		TYP= strtod(elems[1].c_str(),NULL);
		SECTION= strtod(elems[2].c_str(),NULL);
		VALUE= strtof(elems[3].c_str(),NULL);
		LEVEL= strtod(elems[4].c_str(),NULL);
		MOVES= strtod(elems[5].c_str(),NULL);
		WTUP = strtof(elems[6].c_str(),NULL);
		WTDN = strtof(elems[7].c_str(),NULL);

		ILEVEL = (int)LEVEL;
		ISECTION = (int)SECTION;

		COMMOD9->HLEVEL[ISECTION - 1] =  ::max(COMMOD9->HLEVEL[ISECTION - 1], ILEVEL-1);

		if (((int)EVENT != IEVENT) && ((int)EVENT != IEVENT + 1))
		{
			COMMOD9->STOPF = 1;

		}

		if ((int)TYP == 3)
		{
			switch ((int)MOVES)
			{
				case 1:
				case 2:
					COMMOD9->STOPF = 1;
					UNPRDF = 1;                           
					break;
				case 3:
					if ((int)TYP != ITYPE) COMMOD9->NMIDS = COMMOD9->NMIDS + 1;
					break;
			}
		}

		if (COMMOD9->NMIDS > COMMOD9->NOTHR)
		{
			COMMOD9->STOPF = 1;
			cout<<"Number of MIDs exceeds EVENTS in config file!"<<endl;
		}

		if ((int)TYP == 11)
		{
			if ((int)TYP != ITYPE) COMMOD9->NBOXS = COMMOD9->NBOXS + 1;
		}

		if ((COMMOD9->NBOXS * 2) > COMMOD9->NOTHR)
		{
			COMMOD9->STOPF = 1;
			cout<<"Number of MAX & MIN events exceeds EVENTS in config file!"<<endl;
		}

		if ((int)LEVEL <= 0)
		{
			COMMOD9->STOPF = 1;                 
			cout<<"        revise column 5 in file "<<endl;
		}

		if ((int)LEVEL > COMMOD9->MXLVL)
		{
			COMMOD9->STOPF = 1;                  
			cout<<"    increase limit in configuration file!"<<endl;
		}


		IEVENT = (int)EVENT;
		ITYPE = (int)TYP;
		LSTSECT = ::max(LSTSECT, (int)SECTION);
		LSTEVNT = ::max(LSTEVNT, (int)EVENT);
	}

	_file.close();


	if (COMMOD9->STOPF == 1) goto Label9999;

	if (LSTEVNT != COMMOD9->NEVNT - COMMOD9->NSPC - COMMOD9->NMIDS - COMMOD9->NBOXS)
	{
		COMMOD9->STOPF = 1;             
	}

	if (LSTSECT != COMMOD9->NSCT)
	{
		COMMOD9->STOPF = 1;               
	}
	
	if (COMMOD9->STOPF == 1) goto Label9999;


	//reset NROWS for array-loading loops
	COMMOD9->NROWS = 0;


	//reset event counter
	IEVENT = 0;      

	_file.open(COMMOD9->INFILE.c_str(),ios::in);

Label3:
	COMMOD9->NROWS = COMMOD9->NROWS + 1;
	

	while(std::getline(_file,line))
	{
		vector<string> elems = ::split(line, ' ');
		::trim(elems[0]);
		::trim(elems[1]);
		::trim(elems[2]);
		::trim(elems[3]);
		::trim(elems[4]);
		::trim(elems[5]);
		::trim(elems[6]);
		::trim(elems[7]);

		EVENT = strtod(elems[0].c_str(),NULL);
		TYP= strtod(elems[1].c_str(),NULL);
		SECTION= strtod(elems[2].c_str(),NULL);
		VALUE= strtof(elems[3].c_str(),NULL);
		LEVEL= strtod(elems[4].c_str(),NULL);
		MOVES= strtod(elems[5].c_str(),NULL);
		WTUP = strtof(elems[6].c_str(),NULL);
		WTDN = strtof(elems[7].c_str(),NULL);				

		IEVENT = (int)EVENT;
		ITYPE = (int)TYP;
		ISECTION = (int)SECTION;
		ILEVEL = (int)LEVEL;

		if (ILEVEL > 0)
		{
			if (COMMOD9->VALEVEL[ISECTION - 1][ILEVEL - 1] == 0.0)
			{
				COMMOD9->VALEVEL[ISECTION - 1][ILEVEL - 1] = VALUE;
			}
			else if(COMMOD9->VALEVEL[ISECTION-1][ILEVEL - 1] !=VALUE)
			{
				COMMOD9->STOPF = 1;                      
			}
		}

		COMMOD9->ELEVEL[ISECTION - 1][ILEVEL - 1] = COMMOD9->ELEVEL[ISECTION - 1][ILEVEL - 1] + 1;
		if (ITYPE <= 3)
		{
			COMMOD9->ELEVEL[ISECTION - 1][COMMOD9->MXLVL] = COMMOD9->ELEVEL[ISECTION - 1][COMMOD9->MXLVL] + 1;
		}
		else if (ITYPE > 3)
		{
			COMMOD9->ELEVEL[ISECTION - 1][COMMOD9->MXLVL + 1] = COMMOD9->ELEVEL[ISECTION - 1][COMMOD9->MXLVL + 1] + 1;
		}


		if (ITYPE > 2 || ITYPE < 1)
		{
			IMOVES = (int)MOVES;

			if (((ITYPE > 3) && (ITYPE < 10) && (IMOVES != 0)) ||
					((ITYPE == 11) && (IMOVES != 2)) ||
					((ITYPE == 12) && (IMOVES != 1)) ||
					((ITYPE == 3) && (IMOVES != 3)) ||
					((ITYPE == -1) && (IMOVES != 1)) ||
					((ITYPE == -2) && (IMOVES != 2)))
			{
				COMMOD9->STOPF = 1;						

			}

			NENDS = NENDS + 1;
			IUNPRD = IUNPRD + 1;

			if (COMMOD9->NROWS == 1)
			{
				COMMOD9->IRCOUNT = 1;
			}
			else if ((IEVENT - 1 != HEVENT) || (ITYPE != HTYPE))
			{						
				COMMOD9->IRCOUNT = COMMOD9->IRCOUNT + 1;
			}


			HEVENT = IEVENT - 1;
			HTYPE = ITYPE;
			COMMOD9->IROWS[COMMOD9->IRCOUNT - 1][0] = IEVENT - 1;
			COMMOD9->IROWS[COMMOD9->IRCOUNT - 1][1] = ITYPE;
			COMMOD9->IROWS[COMMOD9->IRCOUNT - 1][2] = IUNPRD;
			COMMOD9->ISTATIC_0[(ISECTION - 1)*COMMOD9->NEVNT+(COMMOD9->IRCOUNT - 1)] = ILEVEL - 1;
			COMMOD9->ISTATIC_1[(ISECTION - 1)*COMMOD9->NEVNT + (COMMOD9->IRCOUNT - 1)] = IMOVES;

			IDX = COMMOD9->IDX_SCT_RSTATIC[ISECTION - 1] + (COMMOD9->IRCOUNT - 1) * 2;
			COMMOD9->NRSTATIC[IDX] = WTUP;
			IDX = COMMOD9->IDX_SCT_RSTATIC[ISECTION - 1] + (COMMOD9->IRCOUNT - 1) * 2 + 1;
			COMMOD9->NRSTATIC[IDX] = WTDN;
		}
		else
		{                   
			COMMOD9->NROWS = COMMOD9->NROWS - 1;
		}

		goto Label3;
	}
	_file.close();

	COMMOD9->NROWS = COMMOD9->NROWS - 1;
	
	if (COMMOD9->STOPF == 1) goto Label9999;

	for (J = 0; J < COMMOD9->NSCT; J++)
	{           
		COMMOD9->VALEVEL[J][COMMOD9->MXLVL + 2] = 1000.00f - COMMOD9->VALEVEL[J][0];

		for (I = 0; I <= COMMOD9->HLEVEL[J]; I++)
		{
			COMMOD9->VALEVEL[J][I] = COMMOD9->VALEVEL[J][I] + COMMOD9->VALEVEL[J][COMMOD9->MXLVL + 2];

		}
	}

	for (J = 0; J < COMMOD9->NSCT; J++)
	{
		if (COMMOD9->HLEVEL[J] < 1) continue;

		for (I = 1; I <= COMMOD9->HLEVEL[J]; I++)
		{
			COMMOD9->ELEVEL[J][I] = COMMOD9->ELEVEL[J][I] + COMMOD9->ELEVEL[J][I - 1];
		}
	}

	

	if (COMMOD9->IRCOUNT != COMMOD9->NEVNT - (COMMOD9->NSPC * 2))
	{
		COMMOD9->STOPF = 1;              
		goto Label9999;
	}

	IEVENT = 0;

	if (COMMOD9->NROWS == 0) IPRD = 1; else IPRD = 0;
	_file.open(COMMOD9->INFILE.c_str(),ios::in);
Label5:
	COMMOD9->NROWS = COMMOD9->NROWS + 1;         

	while(std::getline(_file,line))
	{
		vector<string> elems = ::split(line, ' ');
		::trim(elems[0]);
		::trim(elems[1]);
		::trim(elems[2]);
		::trim(elems[3]);
		::trim(elems[4]);
		::trim(elems[5]);
		::trim(elems[6]);
		::trim(elems[7]);

		EVENT = strtod(elems[0].c_str(),NULL);
		TYP= strtod(elems[1].c_str(),NULL);
		SECTION= strtod(elems[2].c_str(),NULL);
		VALUE= strtof(elems[3].c_str(),NULL);
		LEVEL= strtod(elems[4].c_str(),NULL);
		MOVES= strtod(elems[5].c_str(),NULL);
		WTUP = strtof(elems[6].c_str(),NULL);
		WTDN = strtof(elems[7].c_str(),NULL);

		if ((TYP < 3) && (TYP > 0))
		{
			IEVENT = (int)EVENT;
			ITYPE = (int)TYP;
			ISECTION = (int)SECTION;
			ILEVEL = (int)LEVEL;
			IMOVES = (int)MOVES;
			NENDS = NENDS + 1;

			if (((ITYPE == 1) && (IMOVES < 1)) ||
					((ITYPE == 1) && (IMOVES > 3)) ||
					((ITYPE != 2) && (IMOVES == 2)) ||
					((ITYPE == 2) && (IMOVES < 2)) ||
					((ITYPE == 2) && (IMOVES > 3)) ||
					((ITYPE == 11) && (IMOVES != 2)) ||
					((ITYPE == 12) && (IMOVES != 1)))
			{
				COMMOD9->STOPF = 1;                     
			}

			if (COMMOD9->NROWS == 1)
			{                     
				COMMOD9->IRCOUNT = 1;
			}
			else if ((IEVENT-1 != HEVENT) || (ITYPE != HTYPE))
			{                  
				COMMOD9->IRCOUNT = COMMOD9->IRCOUNT + 1;
				if (IEVENT-1 != HEVENT) IPRD = IPRD + 1;
			}

			HEVENT = IEVENT-1;
			HTYPE = ITYPE;
			COMMOD9->IROWS[COMMOD9->IRCOUNT - 1][0] = IEVENT-1;
			COMMOD9->IROWS[COMMOD9->IRCOUNT - 1][1] = ITYPE;
			COMMOD9->IROWS[COMMOD9->IRCOUNT - 1][2] = IPRD-1;                   
			COMMOD9->ISTATIC_0[(ISECTION - 1)*COMMOD9->NEVNT + (COMMOD9->IRCOUNT - 1)] = ILEVEL - 1;
			COMMOD9->ISTATIC_1[(ISECTION - 1)*COMMOD9->NEVNT + (COMMOD9->IRCOUNT - 1)] = IMOVES;                 

			IDX = COMMOD9->IDX_SCT_RSTATIC[ISECTION - 1] + (COMMOD9->IRCOUNT - 1) * 2;
			COMMOD9->NRSTATIC[IDX] = WTUP;
			IDX = COMMOD9->IDX_SCT_RSTATIC[ISECTION - 1] + (COMMOD9->IRCOUNT - 1) * 2 + 1;
			COMMOD9->NRSTATIC[IDX] = WTDN;

			COMMOD9->NEGATIVE[IPRD - 1][ISECTION - 1] = 1-1;
			COMMOD9->NEGATIVE[IPRD - 1][COMMOD9->NSCT + ITYPE - 1] = COMMOD9->IRCOUNT-1;


			if (COMMOD9->CONTRACT == 0)
			{
				if ((ITYPE == 1) || (ITYPE == 2))
				{
					COMMOD9->ISTATIC_1[(ISECTION - 1)*COMMOD9->NEVNT + (COMMOD9->IRCOUNT - 1)] = ITYPE;
				}
			}
			else if ((COMMOD9->CONTRACT == 2) || (COMMOD9->CONTRACT == 3))
			{                       
				if ((ITYPE == 1) || (ITYPE == 2))
				{
					COMMOD9->ISTATIC_1[(ISECTION - 1)*COMMOD9->NEVNT + (COMMOD9->IRCOUNT - 1)] = 3;
				}
			}

		}


		goto Label5;
	}
	_file.close();


	COMMOD9->NOBS = NENDS;         
	::EVENTSUM(1, MINC);
	::EVENTSUM(2, MINC);

	COMMOD9->NEXCL = 0;

	for (I = 0; I < COMMOD9->NEVNT; I++)
	{
		if (COMMOD9->CULLIST[I][2] == 1) COMMOD9->NEXCL = COMMOD9->NEXCL + 1;
	}

	COMMOD9->JEXCL = 0;
	for (J = 0; J < COMMOD9->NSCT; J++)
	{
		for (I = 0; I < COMMOD9->NEVNT; I++)
		{

			if (COMMOD9->CULLIST[I][2] != 1) continue;

			if (((COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+I] == 0) && (COMMOD9->IROWS[I][1] == 2)) ||
					((COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+I] == COMMOD9->HLEVEL[J]) &&
					 (COMMOD9->IROWS[I][1] == 1)))
			{
				if (COMMOD9->JSPANF > 0) COMMOD9->JSPANF = COMMOD9->JSPANF + 1;
				COMMOD9->SECTPROP[J][1] = COMMOD9->SECTPROP[J][1] + 1;
			}


			if (((COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+I] == 0) && (::abs(COMMOD9->IROWS[I][1]) == 1)) ||
					((COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+I] == COMMOD9->HLEVEL[J]) &&
					 (::abs(COMMOD9->IROWS[I][1]) == 2)))
			{
				COMMOD9->SECTPROP[J][2] = COMMOD9->SECTPROP[J][2] + 1;
			}
		}

		if (COMMOD9->SECTPROP[J][1] > 0)
			COMMOD9->JEXCL = COMMOD9->JEXCL + 1;

	}

	if (COMMOD9->JSPANF > 1)
	{
		COMMOD9->JSPANF = 1;
	}
	else  if (COMMOD9->JSPANF == 1)
	{
		COMMOD9->JSPANF = 0;
	}


	COMMOD9->MAXPEN = 0.0;
	
	
	double iSectionMaxPen;
	double tempPen;
	if (COMMOD9->PENF < 2)
	{
		for (M = 0; M < COMMOD9->NSCT; M++)
		{
			iSectionMaxPen = COMMOD9->MAXPEN;

			for (N = 0; N < COMMOD9->NEVNT; N++)
			{
				tempPen = 0;

				if (::IORZERO(N, M)) continue;

				if (::abs(COMMOD9->IROWS[N][1]) == 1)
				{
					IDX = COMMOD9->IDX_SCT_RSTATIC[M] + N * 2+1;
					if (COMMOD9->PENF == 0)
					{
						tempPen = (COMMOD9->NRSTATIC[IDX] *
								(COMMOD9->VALEVEL[M][COMMOD9->ISTATIC_0[M*COMMOD9->NEVNT+N]] - 1000));
						COMMOD9->MAXPEN = COMMOD9->MAXPEN + tempPen;
					}
					else if (COMMOD9->PENF == 1)
					{

						tempPen = (COMMOD9->NRSTATIC[IDX] *
								(double)(COMMOD9->ISTATIC_0[M*COMMOD9->NEVNT+N]));
						COMMOD9->MAXPEN = COMMOD9->MAXPEN + tempPen;
					}
					else if (COMMOD9->PENF == -1)
					{							
						tempPen = (COMMOD9->NRSTATIC[IDX] *
								(double)(COMMOD9->ELEVEL[M][COMMOD9->ISTATIC_0[M*COMMOD9->NEVNT+N]]));
						COMMOD9->MAXPEN = COMMOD9->MAXPEN + tempPen;
					}
				}

				if (::abs(COMMOD9->IROWS[N][1]) == 2)
				{
					IDX = COMMOD9->IDX_SCT_RSTATIC[M] + N * 2;
					if (COMMOD9->PENF == 0)
					{
						tempPen = (COMMOD9->NRSTATIC[IDX] *
								(COMMOD9->VALEVEL[M][COMMOD9->HLEVEL[M]] -
								 COMMOD9->VALEVEL[M][COMMOD9->ISTATIC_0[M*COMMOD9->NEVNT+N]]));
						COMMOD9->MAXPEN = COMMOD9->MAXPEN + tempPen;
					}
					else if (COMMOD9->PENF == 1)
					{							
						tempPen = (COMMOD9->NRSTATIC[IDX] *
								(double)(COMMOD9->HLEVEL[M] - COMMOD9->ISTATIC_0[M*COMMOD9->NEVNT+N]));
						COMMOD9->MAXPEN = COMMOD9->MAXPEN + tempPen;
					}
					else if (COMMOD9->PENF == -1)
					{							
						tempPen = (COMMOD9->NRSTATIC[IDX] *
								(double)(COMMOD9->ELEVEL[M][COMMOD9->HLEVEL[M]] -
									COMMOD9->ELEVEL[M][COMMOD9->ISTATIC_0[M*COMMOD9->NEVNT+N]]));
						COMMOD9->MAXPEN = COMMOD9->MAXPEN + tempPen;
					}
				}
			}

			SectionMaxPen[M] = COMMOD9->MAXPEN - iSectionMaxPen;
		}
	}

	ASORT_DESC(SectionMaxPen, COMMOD9->SectionMaxPenOrder, COMMOD9->NSCT);

	delete[] SectionMaxPen;

	
	

	

	for (int sct = 0; sct <= 20 && sct < COMMOD9->NSCT; sct++)
	{
		for (int evt = 0; evt < COMMOD9->NEVNT; evt++)
		{
			if (COMMOD9->ISTATIC_0[COMMOD9->SectionMaxPenOrder[sct] * COMMOD9->NEVNT+evt] > -1)
			{
				bFind = false;
				for (int idx = 0; idx < nBigWeightEvents; idx++)
				{
					if (tmpBigWeightEventsArray[idx] == evt)
					{
						bFind = true;
						break;
					}
				}

				if (!bFind)tmpBigWeightEventsArray[nBigWeightEvents++] = evt;
			}
		}
	}		


	for (J = 0; J < COMMOD9->NSCT; J++)
	{
		JOPT = 0;

		for (I = 0; I < COMMOD9->NEVNT - 1; I++)
		{
			if (COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+I] == -1) continue;

			for (K = I + 1; K < COMMOD9->NEVNT; K++)
			{
				if (COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+K] == -1) continue;

				if (COMMOD9->PENF == 2) COMMOD9->MAXPEN = COMMOD9->MAXPEN + 1;

				if (COMMOD9->PENF == 3) COMMOD9->MAXPEN = COMMOD9->MAXPEN +
					::fabs(COMMOD9->VALEVEL[J][COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+I]] -
							COMMOD9->VALEVEL[J][COMMOD9->ISTATIC_0[J*COMMOD9->NEVNT+K]]);

				JOPT = JOPT + 1;
			}
		}

	}

	if (COMMOD9->PENF == 4) COMMOD9->MAXPEN = (COMMOD9->NEVNT * (COMMOD9->NEVNT - 1)) / 2;

	if ((COMMOD9->PENF == 5) || (COMMOD9->PENF == 6)) COMMOD9->MAXPEN =
		(COMMOD9->NEVNT * (COMMOD9->NEVNT - 1));

	COMMOD9->NROWS = COMMOD9->NROWS - 1;

	for (J = 0; J < COMMOD9->NSCT; J++)
	{
		for (I = 0; I < COMMOD9->MXLVL; I++)
		{
			if ((COMMOD9->VALEVEL[J][I] < 999.0) && (COMMOD9->VALEVEL[J][I] != 0))
			{
				COMMOD9->STOPF = 1;                    
			}

			if ((COMMOD9->VALEVEL[J][I] > COMMOD9->VALEVEL[J][I + 1]) &&
					((COMMOD9->VALEVEL[J][I + 2] != 0.0) || (COMMOD9->VALEVEL[J][I + 1] != 0.0)))
			{
				COMMOD9->STOPF = 1;                      
			}
		}
	}

	if (COMMOD9->STOPF == 1) goto Label9999;

	if ((COMMOD9->SCTFILE.substr(0, 3) == "OFF") || (COMMOD9->SCTFILE.substr(0, 3) == "off"))
	{

		cout<<"  Section Names NOT Provided; using numbers only"<<endl;
		char buff[10];
		for (ISECTION = 0; ISECTION < COMMOD9->NSCT; ISECTION++)
		{
			COMMOD9->SECTPERM[ISECTION] = ISECTION;
			if (COMMOD9->LOADNAMES) {
				::sprintf(buff, "%d", ISECTION);
				JNICK = "(" + string(buff) + ")";
				COMMOD9->SECTNAME[ISECTION] = "SECTION " + JNICK;
				COMMOD9->SECTNICK[ISECTION] = JNICK;
			}

			COMMOD9->SECTPROP[ISECTION][0] = 1;
		}
	}
	else
	{
		::trim(COMMOD9->SCTFILE);

		ISECTION = -1;           

		_file.open(COMMOD9->SCTFILE.c_str(),ios::in);

Label201:
		ISECTION = ISECTION + 1;               

		while(std::getline(_file,line))
		{
			vector<string> elems = ::split(line, '\'');
			::trim(elems[0]);
			::trim(elems[1]);
			::trim(elems[2]);
			::trim(elems[3]);
			::trim(elems[4]);
			HSECTION = atoi(elems[0].c_str());

			JPLACE = atoi(elems[2].c_str());
			JINC = atoi(elems[4].c_str());


			if (COMMOD9->LOADNAMES) {
				JNAME = elems[3];
				JNICK = elems[1];
				COMMOD9->SECTNAME[ISECTION] = JNAME;
				COMMOD9->SECTNICK[ISECTION] = JNICK;
			}

			COMMOD9->SECTPERM[JPLACE-1] = ISECTION;
			COMMOD9->SECTPROP[ISECTION][0] = JINC;

			if ((JINC > 1) || (JINC < 0))
			{
				COMMOD9->STOPF = 1;                       
				goto Label206;
			}
			else if (JINC == 1)
			{
				NSCAL = NSCAL + 1;
			}

			if (HSECTION-1 != ISECTION)
			{
				COMMOD9->SCTFILE = "OFF";
				COMMOD9->STOPF = 1;                      
				goto Label206;
			}

			goto Label201;
		}
		_file.close();

		if (ISECTION  != COMMOD9->NSCT)
		{
			COMMOD9->SCTFILE = "OFF";
			COMMOD9->STOPF = 1;                  
		}

		for (J = 0; J < COMMOD9->NSCT; J++)
		{
			if (COMMOD9->SECTPERM[J] == -1)
			{
				COMMOD9->STOPF = 1;                      
			}
		}


	}

Label206:
	if (COMMOD9->SCTFILE.substr(0, 3) != "OFF")
	{
		//nothing
	}

	if (NSCAL == 1)
	{
		cout<<"   WARNING: No sections allocated for scaling"<<endl;
		cout<<"            [col 5 of section list all 0s]"<<endl;
		cout<<"            Scaled composites not possible"<<endl;
	}
	if ((COMMOD9->SCTTAGS.substr(0, 3) != "OFF") &&
			(COMMOD9->SCTTAGS.substr(0, 3) != "off"))
	{
		cout << "  Loading Section Tag Names . . ." << endl;
		if (COMMOD9->LOADNAMES) {					

Label741:

			::trim(TAGSTRING);
			COMMOD9->STAGNAME[Ilbl] = TAGSTRING;
			goto Label741;

			COMMOD9->MXJTAG = Ilbl;
		}
		else {
			COMMOD9->MXJTAG = Ilbl;
		}
	}  


	if ((COMMOD9->STAGFILE.substr(0, 3) != "OFF") &&
			(COMMOD9->STAGFILE.substr(0, 3) != "off"))
	{
		NTAG = 0;
		cout<<"  Loading Tags for each Section. . ."<<endl;

Label711:               
		COMMOD9->SECTPROP[Ilbl][3] = JTAG;
		NTAG = ::max(NTAG, JTAG);
		goto Label711;

		;
	}

	if (COMMOD9->STAGFILE.substr(0, 3) != "OFF")
	{
		//nothing
	}

	if (NTAG > COMMOD9->MXJTAG)
	{
		COMMOD9->STOPF = 1;               
	}
	else if (NTAG < COMMOD9->MXJTAG)
	{
		//nothing
	}

	if (COMMOD9->LOADNAMES) {

		if ((COMMOD9->LBLFILE.substr(0, 3) != "OFF") && (COMMOD9->LBLFILE.substr(0, 3) != "off"))
		{
			::trim(COMMOD9->LBLFILE);
			cout << "  Loading Label Names " << COMMOD9->LBLFILE << " ..." << endl;					

			COMMOD9->LABLNAME[Ilbl] = JNAME;
		}
	}
	if (COMMOD9->LBLFILE.substr(0, 3) != "OFF")
	{
		//nothing
	}


	if ((COMMOD9->EVTFILE.substr(0, 3) == "OFF") || (COMMOD9->EVTFILE.substr(0, 3) == "off"))
	{
		if (COMMOD9->LOADNAMES) {
			cout << "  Event Names NOT Provided; using numbers only" << endl;
			char buff[10];
			for (IEVENT = 0; IEVENT < COMMOD9->NEVNT - COMMOD9->NSPC; IEVENT++)
			{							
				::sprintf(buff, "%d", IEVENT);
				INICK = "(" + string(buff) + ")";

				COMMOD9->EVNTNAME[IEVENT] = "EVENT" + INICK;
				COMMOD9->EVNTNICK[IEVENT] = "Evt." + INICK;
			}
		}
	}
	else
	{
		::trim(COMMOD9->EVTFILE);					
		IEVENT = -1;

		_file.open(COMMOD9->EVTFILE.c_str(), ios::in);

Label301:
		IEVENT = IEVENT + 1;

		while (std::getline(_file, line))
		{
			vector<string> elems = ::split(line, '\'');
			::trim(elems[0]);
			::trim(elems[1]);
			::trim(elems[3]);
			HEVENT = atoi(elems[0].c_str());						
			INAME = elems[3];
			if (COMMOD9->LOADNAMES) {
				INICK = elems[1];

				COMMOD9->EVNTNICK[IEVENT] = INICK;
			}

			if (INAME.substr(0, 1) == "*")
			{
				A = 0;
				Ilbl = 0;
				if (COMMOD9->LOADNAMES) {
					COMMOD9->EVNTNAME[IEVENT] = INAME.substr(5);
				}
				if (INAME.substr(3, 1) == "F") Ityp = 1;
				if (INAME.substr(3, 1) == "L") Ityp = 2;
				if (INAME.substr(3, 1) == "M") Ityp = 3;
				if (INAME.substr(3, 1) == "X") Ityp = 4;
				if (INAME.substr(3, 1) == "N") Ityp = 5;

				for (L = 0; L < 10; L++)
				{
					if (DIGIT[L] == INAME.substr(1, 1))
					{
						Ilbl = Ilbl + ((L - 1) * 10);
						A = A + 1;
					}

					if (DIGIT[L] == INAME.substr(2, 1))
					{
						Ilbl = Ilbl + (L - 1);
						A = A + 1;
					}
				}

				if ((A != 2) || (Ityp == 0))
				{
					COMMOD9->STOPF = 1;
					cout << "* Invalid label for event: " << HEVENT << endl;
				}
				else
				{
					::FINDROW(HEVENT - 1, Ityp, IROW);
					COMMOD9->IROWS[IROW][4] = Ilbl;
				}
			}
			else
			{
				if (COMMOD9->LOADNAMES) {
					COMMOD9->EVNTNAME[IEVENT] = INAME;
				}
			}

			if (HEVENT - 1 != IEVENT)
			{
				COMMOD9->EVTFILE = "OFF";
				COMMOD9->STOPF = 1;							
			}

			if (COMMOD9->STOPF == 1) goto Label333;

			goto Label301;

		}
		_file.close();

	}

Label333:

	if ((COMMOD9->EVTTAGS.substr(0, 3) != "OFF") &&
			(COMMOD9->EVTTAGS.substr(0, 3) != "off"))
	{
		if (COMMOD9->LOADNAMES) {
			cout << "  Loading Event Tag Names . . ." << endl;

Label751:					
			::trim(TAGSTRING);
			COMMOD9->ETAGNAME[Ilbl + 1] = TAGSTRING;

			goto Label751;					

			COMMOD9->MXITAG = Ilbl;
		}
		else {
			cout << "  Loading Event Tag Names . . ." << endl;

			COMMOD9->MXITAG = Ilbl;
		}
	}


	if ((COMMOD9->ETAGFILE.substr(0, 3) != "OFF") &&
			(COMMOD9->ETAGFILE.substr(0, 3) != "off"))
	{
		cout<<"  Loading Tags for each Event . . ."<<endl;

Label761:

		for (I = 0; I < COMMOD9->NEVNT; I++)
		{
			if (COMMOD9->IROWS[I][0] == Ilbl) COMMOD9->IROWS[I][5] = ITAG;
		}

		goto Label761;

	}

	if (COMMOD9->ETAGFILE.substr(0, 3) != "OFF")
	{
		//nothing
	}


	for (I = 0; I < COMMOD9->NEVNT; I++)
	{
		switch (COMMOD9->IROWS[I][1])
		{
			case 11:
				::FINDROW(COMMOD9->IROWS[I][0], 12, IPAIR);
				COMMOD9->IROWS[I][3] = IPAIR;
				break;
			case 12:
				::FINDROW(COMMOD9->IROWS[I][0], 11, IPAIR);
				COMMOD9->IROWS[I][3] = IPAIR;
				break;
			default:
				COMMOD9->IROWS[I][3] = COMMOD9->IROWS[I][0];
				break;
		}
	}

	for (I = 0; I < COMMOD9->NEVNT; I++)
	{
		switch (COMMOD9->IROWS[I][1])
		{
			case 1:
				if (COMMOD9->NEGATIVE[COMMOD9->IROWS[I][2]][COMMOD9->NSCT + 1] >= 0)
				{
					COMMOD9->IROWS[I][3] = COMMOD9->NEGATIVE[COMMOD9->IROWS[I][2]][COMMOD9->NSCT + 1];
				}
				else
				{
					COMMOD9->STOPF = 1;                           
					goto Label9999;
				}
				break;
			case 2:
				if (COMMOD9->NEGATIVE[COMMOD9->IROWS[I][2]][COMMOD9->NSCT] >= 0)
				{
					COMMOD9->IROWS[I][3] = COMMOD9->NEGATIVE[COMMOD9->IROWS[I][2]][COMMOD9->NSCT];
				}
				else
				{
					COMMOD9->STOPF = 1;                          
					goto Label9999;
				}
				break;
			default:
				COMMOD9->IROWS[I][3] = COMMOD9->IROWS[I][0];
				break;
		}
	}

	for (I = 0; I < COMMOD9->NEVNT; I++)
	{
		if (COMMOD9->IROWS[I][1] != 3) continue;

		IEVENT = COMMOD9->IROWS[I][0];
		MFAD = 0;
		MLAD = 0;

		for (M = 0; M < COMMOD9->NEVNT; M++)
		{
			if ((COMMOD9->IROWS[M][0] == IEVENT) && (COMMOD9->IROWS[M][1] == 1)) MFAD = M;

			if ((COMMOD9->IROWS[M][0] == IEVENT) && (COMMOD9->IROWS[M][1] == 2)) MLAD = M;
		}

		if (MFAD == 0)
		{
			COMMOD9->STOPF = 1;
			cout<<" * taxon',I, ' has MID but no FAD"<<endl;
		}
		else if (MLAD == 0)
		{
			COMMOD9->STOPF = 1;
			cout<<" * taxon',I, ' has MID but no LAD"<<endl;
		}
	}


	for (I = 0; I < COMMOD9->NEVNT; I++)
	{
		if ((COMMOD9->IROWS[I][1] != 11) && (COMMOD9->IROWS[I][1] != 12)) continue;

		IEVENT = COMMOD9->IROWS[I][0];               

		MFAD = 0;
		MLAD = 0;

		for (M = 0; M < COMMOD9->NEVNT; M++)
		{
			if ((COMMOD9->IROWS[M][0] == IEVENT) && (COMMOD9->IROWS[M][1] == 11)) MFAD = M;

			if ((COMMOD9->IROWS[M][0] == IEVENT) && (COMMOD9->IROWS[M][1] == 12)) MLAD = M;
		}

		if ((MFAD == 0) && (MLAD != 0))
		{
			COMMOD9->STOPF = 1;
			cout<<" * boxed event "<<I<<" has MIN but no MAX"<<endl;
		}
		else if ((MLAD == 0) && (MFAD != 0))
		{
			COMMOD9->STOPF = 1;
			cout<<" * boxed event "<<I<<" has MAX but no MIN"<<endl;
		}
	}


	for (I = 0; I < COMMOD9->NEVNT; I++)
	{
		switch (COMMOD9->IROWS[I][1])
		{
			case 11:
				::FINDROW(COMMOD9->IROWS[I][0], 12, IPAIR);
				COMMOD9->IROWS[I][3] = IPAIR;
				break;
			case 12:
				::FINDROW(COMMOD9->IROWS[I][0], 11, IPAIR);
				COMMOD9->IROWS[I][3] = IPAIR;
				break;
		}
	}

	COMMOD9->YEVNT = COMMOD9->IROWS[COMMOD9->XEVNT][3];

Label9999:           

	if (UNPRDF == 1)
	{
		cout<<"_______________________________________________"<<endl;
		cout<<"Input includes top-only and/or bottom-only taxa"<<endl;
		cout<<"Change name/extension of current input file "<<endl;
		cout<<"ENTER THE NEW NAME AS PREPFILE IN CONOP9.CFG"<<endl;
		cout<<"AND THEN RUN CONSORT9 BEFORE RETRYING CONOP9"<<endl;

		COMMOD9->STOPF = 1;
	}


	if (COMMOD9->STOPF == 1)
	{
		cout<<"______________________"<<endl;
		cout<<"**INVALID INPUT DATA**"<<endl;
		cout<<endl;
	}
}
