#include <string>
#include <iostream>
#include <fstream>
#include <stdlib.h> 
#include <vector>
#include <map>
#include <time.h>
#include <math.h>
#include <algorithm>
#include "COMMOD.h"
#include "PREPARE.h"
#include "Helper.h"

using namespace std;
using std::string;
using std::vector;
//extern	MPI_File *fp;
PREPARE* PREPARE::g_singleton=NULL;

void PREPARE::InitConfigNamelist(){

	CONOPNET_CNF_NAMELIST= map<string, map<string, string> >();

	map<string, string> getinn_namelist=map<string, string>();
	getinn_namelist.insert(pair<string,string>("PROJECT",""));
	getinn_namelist.insert(pair<string,string>("SECTIONS",""));
	getinn_namelist.insert(pair<string,string>("TAXA",""));
	getinn_namelist.insert(pair<string,string>("EVENTS",""));
	getinn_namelist.insert(pair<string,string>("MAX_LEVELS",""));
	getinn_namelist.insert(pair<string,string>("MAX_LABELS",""));
	getinn_namelist.insert(pair<string,string>("LOADFILE",""));
	getinn_namelist.insert(pair<string,string>("PREPFILE",""));
	getinn_namelist.insert(pair<string,string>("SECTFILE",""));
	getinn_namelist.insert(pair<string,string>("SECTTAGFILE",""));
	getinn_namelist.insert(pair<string,string>("SECTTAGS",""));
	getinn_namelist.insert(pair<string,string>("LABELFILE",""));
	getinn_namelist.insert(pair<string,string>("EVENTFILE",""));
	getinn_namelist.insert(pair<string,string>("EVENTTAGFILE",""));
	getinn_namelist.insert(pair<string,string>("EVENTTAGS",""));
	getinn_namelist.insert(pair<string,string>("BESTKNOWN",""));
	getinn_namelist.insert(pair<string, string>("LOADNAMES", ""));
	getinn_namelist.insert(pair<string, string>("ANNEALONLY", ""));
	getinn_namelist.insert(pair<string, string>("LOADCOEX", ""));
	getinn_namelist.insert(pair<string, string>("LOADCEFB4L", ""));

	CONOPNET_CNF_NAMELIST.insert(pair<string,map<string, string> >("getinn",getinn_namelist));

	map<string, string> getans_namelist=map<string, string>();
	getans_namelist.insert(pair<string,string>("PENALTY",""));
	getans_namelist.insert(pair<string,string>("LETCONTRACT",""));
	getans_namelist.insert(pair<string,string>("WEIGHTING",""));
	getans_namelist.insert(pair<string,string>("USENEGATIVE",""));
	getans_namelist.insert(pair<string,string>("NEARENOUGH",""));
	getans_namelist.insert(pair<string,string>("EXCLUSIVES",""));
	getans_namelist.insert(pair<string,string>("FORCECOEX",""));
	getans_namelist.insert(pair<string,string>("FORCEFB4L",""));
	getans_namelist.insert(pair<string,string>("HOMERANGE",""));
	getans_namelist.insert(pair<string,string>("SMOOTHER",""));
	getans_namelist.insert(pair<string,string>("SQUEEZER",""));
	getans_namelist.insert(pair<string,string>("SHRINKER",""));
	getans_namelist.insert(pair<string,string>("TEASER",""));
	getans_namelist.insert(pair<string,string>("STACKER",""));
	CONOPNET_CNF_NAMELIST.insert(pair<string,map<string, string> >("getans",getans_namelist));

	map<string, string> getrun_namelist=map<string, string>();
	getrun_namelist.insert(pair<string,string>("SOLVER",""));
	getrun_namelist.insert(pair<string,string>("STEPS",""));
	getrun_namelist.insert(pair<string,string>("TRIALS",""));
	getrun_namelist.insert(pair<string,string>("STARTEMP",""));
	getrun_namelist.insert(pair<string,string>("RATIO",""));
	getrun_namelist.insert(pair<string,string>("HOODSIZE",""));
	getrun_namelist.insert(pair<string,string>("STARTYPE",""));
	getrun_namelist.insert(pair<string,string>("STARTSECT",""));
	getrun_namelist.insert(pair<string,string>("STARTEVENT",""));
	getrun_namelist.insert(pair<string,string>("SHOWMOVIES",""));
	getrun_namelist.insert(pair<string,string>("TRAJECTORY",""));
	getrun_namelist.insert(pair<string,string>("VIDEOMODE",""));
	getrun_namelist.insert(pair<string,string>("PAUSES",""));
	getrun_namelist.insert(pair<string,string>("CURVFILE",""));
	getrun_namelist.insert(pair<string,string>("CRV2FILE",""));
	CONOPNET_CNF_NAMELIST.insert(pair<string,map<string, string> >("getrun",getrun_namelist));

	map<string, string> getout_namelist=map<string, string>();
	getrun_namelist.insert(pair<string,string>("COLUMNS",""));
	getrun_namelist.insert(pair<string,string>("UNLOADMAIN",""));
	getrun_namelist.insert(pair<string,string>("FITS_OUT",""));
	getrun_namelist.insert(pair<string,string>("CNFG_OUT",""));
	getrun_namelist.insert(pair<string,string>("SEQN_OUT",""));
	getrun_namelist.insert(pair<string,string>("INCR_OUT",""));
	getrun_namelist.insert(pair<string,string>("LOC_OUT",""));
	getrun_namelist.insert(pair<string,string>("OBS_OUT",""));
	getrun_namelist.insert(pair<string,string>("COMP_OUT",""));
	getrun_namelist.insert(pair<string,string>("UNLOADSECT",""));
	getrun_namelist.insert(pair<string,string>("SECT_OUT",""));
	getrun_namelist.insert(pair<string,string>("UNLOADEVNT",""));
	getrun_namelist.insert(pair<string,string>("EVNT_OUT",""));
	getrun_namelist.insert(pair<string,string>("COEX_OUT",""));
	getrun_namelist.insert(pair<string,string>("RUNLOGFILE",""));
	getrun_namelist.insert(pair<string,string>("CULLFILE",""));
	getrun_namelist.insert(pair<string,string>("SOLNLIST",""));
	getrun_namelist.insert(pair<string,string>("STARTFILE",""));
	getrun_namelist.insert(pair<string,string>("STEPFILE",""));
	getrun_namelist.insert(pair<string,string>("BESTARTFILE",""));
	getrun_namelist.insert(pair<string,string>("COMPOSFILE",""));
	getrun_namelist.insert(pair<string,string>("COMPOSNMBR",""));
	getrun_namelist.insert(pair<string,string>("COMPOSTYPE",""));
	getrun_namelist.insert(pair<string,string>("OBSDFILE",""));
	getrun_namelist.insert(pair<string,string>("PLCDFILE",""));
	getrun_namelist.insert(pair<string,string>("EXTNFILE",""));
	getrun_namelist.insert(pair<string,string>("COEXISTFILE",""));
	getrun_namelist.insert(pair<string,string>("FAD_LADFILE",""));
	getrun_namelist.insert(pair<string,string>("ORDERFILE",""));
	CONOPNET_CNF_NAMELIST.insert(pair<string,map<string, string> >("getout",getrun_namelist));
}  

vector<string> PREPARE::GetKeyValuePair(string line)
{	
	if(line.find("=")==line.npos)throw "error in reading config file.";

	vector<string> elems = ::split(line, '=');
	::trim(elems[0]);
	::trim(elems[1]);
	::transform(elems[0].begin(),elems[0].end(),elems[0].begin(),::toupper);
	if(elems[1].find('\'')==0)elems[1]=elems[1].substr(1,elems[1].length()-2);

	return elems;            
}

void PREPARE::ReadConfigNamelist(){
	COMMOD* COMMOD9 = COMMOD::Singleton();
	if(!::fileExists(COMMOD9->CNFGFILE))throw "error: config file not found!";
	::fstream _file;
	_file.open(COMMOD9->CNFGFILE.c_str(),ios::in);


	bool bMaybeEnd1 = false;
	bool bMaybeEnd2 = false;
	string currentNamelistType = "";

	string line;
	while(std::getline(_file,line))
	{
		::trim(line);
		if ((line== "") || (line.find("/")==0))
		{
			if (bMaybeEnd2) { break; }
			else if (bMaybeEnd1 == false)
			{
				bMaybeEnd1 = true;
				continue;
			}
			else if (bMaybeEnd2 == false)
			{
				bMaybeEnd2 = true;
				continue;
			}

		}

		if (line.find("&")==0)
		{

			currentNamelistType = line.substr(1);
			::trim(currentNamelistType);
			bMaybeEnd1 = false;
			bMaybeEnd2 = false;
		}
		else if (line.find("_")==0)
		{
			continue;
		}
		else
		{ 
			vector<string> kvp=GetKeyValuePair(line);
			CONOPNET_CNF_NAMELIST[currentNamelistType][kvp[0]] = kvp[1];
		}
	}



	//getins	
	FREQUENCY = atoi(CONOPNET_CNF_NAMELIST["getinn"]["FREQUENCY"].c_str());
	PROJECT = CONOPNET_CNF_NAMELIST["getinn"]["PROJECT"];
	SECTIONS = atoi(CONOPNET_CNF_NAMELIST["getinn"]["SECTIONS"].c_str());
	TAXA = atoi(CONOPNET_CNF_NAMELIST["getinn"]["TAXA"].c_str());
	EVENTS = atoi(CONOPNET_CNF_NAMELIST["getinn"]["EVENTS"].c_str());
	MAX_LEVELS = atoi(CONOPNET_CNF_NAMELIST["getinn"]["MAX_LEVELS"].c_str());
	MAX_LABELS = atoi(CONOPNET_CNF_NAMELIST["getinn"]["MAX_LABELS"].c_str());
	LOADFILE = CONOPNET_CNF_NAMELIST["getinn"]["LOADFILE"];
	PREPFILE = CONOPNET_CNF_NAMELIST["getinn"]["PREPFILE"];
	SECTFILE = CONOPNET_CNF_NAMELIST["getinn"]["SECTFILE"];
	SECTTAGFILE = CONOPNET_CNF_NAMELIST["getinn"]["SECTTAGFILE"];
	SECTTAGS = CONOPNET_CNF_NAMELIST["getinn"]["SECTTAGS"];
	LABELFILE = CONOPNET_CNF_NAMELIST["getinn"]["LABELFILE"];
	EVENTFILE = CONOPNET_CNF_NAMELIST["getinn"]["EVENTFILE"];
	EVENTTAGFILE = CONOPNET_CNF_NAMELIST["getinn"]["EVENTTAGFILE"];
	EVENTTAGS = CONOPNET_CNF_NAMELIST["getinn"]["EVENTTAGS"];
	BESTKNOWN = strtod(CONOPNET_CNF_NAMELIST["getinn"]["BESTKNOWN"].c_str(),NULL);
	LOADNAMES = CONOPNET_CNF_NAMELIST["getinn"]["LOADNAMES"];
	ANNEALONLY = CONOPNET_CNF_NAMELIST["getinn"]["ANNEALONLY"];
	LOADCOEX = CONOPNET_CNF_NAMELIST["getinn"]["LOADCOEX"];
	LOADCEFB4L = CONOPNET_CNF_NAMELIST["getinn"]["LOADCEFB4L"];

	//getans
	PENALTY = CONOPNET_CNF_NAMELIST["getans"]["PENALTY"];
	LETCONTRACT = CONOPNET_CNF_NAMELIST["getans"]["LETCONTRACT"];
	WEIGHTING = CONOPNET_CNF_NAMELIST["getans"]["WEIGHTING"];
	USENEGATIVE = CONOPNET_CNF_NAMELIST["getans"]["USENEGATIVE"];
	NEARENOUGH = strtod(CONOPNET_CNF_NAMELIST["getans"]["NEARENOUGH"].c_str(),NULL);
	EXCLUSIVES = CONOPNET_CNF_NAMELIST["getans"]["EXCLUSIVES"];
	FORCECOEX = CONOPNET_CNF_NAMELIST["getans"]["FORCECOEX"];
	FORCEFB4L = CONOPNET_CNF_NAMELIST["getans"]["FORCEFB4L"];
	HOMERANGE = CONOPNET_CNF_NAMELIST["getans"]["HOMERANGE"];
	SMOOTHER = strtod(CONOPNET_CNF_NAMELIST["getans"]["SMOOTHER"].c_str(),NULL);
	SQUEEZER = strtod(CONOPNET_CNF_NAMELIST["getans"]["SQUEEZER"].c_str(),NULL);
	SHRINKER = strtod(CONOPNET_CNF_NAMELIST["getans"]["SHRINKER"].c_str(),NULL);
	TEASER = strtod(CONOPNET_CNF_NAMELIST["getans"]["TEASER"].c_str(),NULL);
	STACKER = CONOPNET_CNF_NAMELIST["getans"]["STACKER"];

	//getrun
	SOLVER = CONOPNET_CNF_NAMELIST["getrun"]["SOLVER"];
	STEPS = atoi(CONOPNET_CNF_NAMELIST["getrun"]["STEPS"].c_str());
	TRIALS = atoi(CONOPNET_CNF_NAMELIST["getrun"]["TRIALS"].c_str());
	STARTEMP = strtod(CONOPNET_CNF_NAMELIST["getrun"]["STARTEMP"].c_str(),NULL);
	RATIO = strtod(CONOPNET_CNF_NAMELIST["getrun"]["RATIO"].c_str(),NULL);
	HOODSIZE = CONOPNET_CNF_NAMELIST["getrun"]["HOODSIZE"];
	STARTYPE = CONOPNET_CNF_NAMELIST["getrun"]["STARTYPE"];
	STARTSECT = atoi(CONOPNET_CNF_NAMELIST["getrun"]["STARTSECT"].c_str());
	STARTEVENT = atoi(CONOPNET_CNF_NAMELIST["getrun"]["STARTEVENT"].c_str());
	SHOWMOVIES = CONOPNET_CNF_NAMELIST["getrun"]["SHOWMOVIES"];
	TRAJECTORY = CONOPNET_CNF_NAMELIST["getrun"]["TRAJECTORY"];
	VIDEOMODE = CONOPNET_CNF_NAMELIST["getrun"]["VIDEOMODE"];
	PAUSES = CONOPNET_CNF_NAMELIST["getrun"]["PAUSES"];
	CURVFILE = CONOPNET_CNF_NAMELIST["getrun"]["CURVFILE"];
	CRV2FILE = CONOPNET_CNF_NAMELIST["getrun"]["CRV2FILE"];

	//getout
	COLUMNS = atoi(CONOPNET_CNF_NAMELIST["getout"]["COLUMNS"].c_str());
	UNLOADMAIN = CONOPNET_CNF_NAMELIST["getout"]["UNLOADMAIN"];
	FITS_OUT = CONOPNET_CNF_NAMELIST["getout"]["FITS_OUT"];
	CNFG_OUT = CONOPNET_CNF_NAMELIST["getout"]["CNFG_OUT"];
	SEQN_OUT = CONOPNET_CNF_NAMELIST["getout"]["SEQN_OUT"];
	INCR_OUT = CONOPNET_CNF_NAMELIST["getout"]["INCR_OUT"];
	LOC_OUT = CONOPNET_CNF_NAMELIST["getout"]["LOC_OUT"];
	OBS_OUT = CONOPNET_CNF_NAMELIST["getout"]["OBS_OUT"];
	COMP_OUT = CONOPNET_CNF_NAMELIST["getout"]["COMP_OUT"];
	UNLOADSECT = CONOPNET_CNF_NAMELIST["getout"]["UNLOADSECT"];
	SECT_OUT = CONOPNET_CNF_NAMELIST["getout"]["SECT_OUT"];
	UNLOADEVNT = CONOPNET_CNF_NAMELIST["getout"]["UNLOADEVNT"];
	EVNT_OUT = CONOPNET_CNF_NAMELIST["getout"]["EVNT_OUT"];
	COEX_OUT = CONOPNET_CNF_NAMELIST["getout"]["COEX_OUT"];
	RUNLOGFILE = CONOPNET_CNF_NAMELIST["getout"]["RUNLOGFILE"];
	CULLFILE = CONOPNET_CNF_NAMELIST["getout"]["CULLFILE"];
	SOLNLIST = CONOPNET_CNF_NAMELIST["getout"]["SOLNLIST"];
	STARTFILE = CONOPNET_CNF_NAMELIST["getout"]["STARTFILE"];
	STEPFILE = CONOPNET_CNF_NAMELIST["getout"]["STEPFILE"];
	BESTARTFILE = CONOPNET_CNF_NAMELIST["getout"]["BESTARTFILE"];
	COMPOSFILE = CONOPNET_CNF_NAMELIST["getout"]["COMPOSFILE"];
	COMPOSNMBR = atoi(CONOPNET_CNF_NAMELIST["getout"]["COMPOSNMBR"].c_str());
	COMPOSTYPE = CONOPNET_CNF_NAMELIST["getout"]["COMPOSTYPE"];
	OBSDFILE = CONOPNET_CNF_NAMELIST["getout"]["OBSDFILE"];
	PLCDFILE = CONOPNET_CNF_NAMELIST["getout"]["PLCDFILE"];
	EXTNFILE = CONOPNET_CNF_NAMELIST["getout"]["EXTNFILE"];
	COEXISTFILE = CONOPNET_CNF_NAMELIST["getout"]["COEXISTFILE"];
	FAD_LADFILE = CONOPNET_CNF_NAMELIST["getout"]["FAD_LADFILE"];
	ORDERFILE = CONOPNET_CNF_NAMELIST["getout"]["ORDERFILE"];
}

void PREPARE::RunIt(int MYID)
{

	srand((unsigned)time(NULL)*(MYID+1));

	COMMOD* COMMOD9 = COMMOD::Singleton();
	COMMOD9->CNFGFILE.assign("CONOP9.CFG");
	COMMOD9->PLOTFILE.assign("PlotData.txt");	
	//*******************************************
	//initialize some COMMOD names
	COMMOD9->PROJNAME.assign("      ");
	COMMOD9->PROJTYP1.assign("      ");
	COMMOD9->PROJTYP2.assign("      ");
	COMMOD9->NINNER = 0;
	COMMOD9->NOUTER = 0;
	COMMOD9->DFLTPEN = false;
	COMMOD9->DFLTCOMP = false;
	COMMOD9->DFLTNABR = false;
	COMMOD9->DFLTSTART = false;
	COMMOD9->DFLTWT = false;
	COMMOD9->DFLTCOEX = false;
	COMMOD9->DFLTFb4L = false;
	COMMOD9->RASCon = false;

	COMMOD9->NSCT = 0;
	COMMOD9->NEVNT = 0;
	COMMOD9->NSPC = 0;
	COMMOD9->NOTHR = 0;
	//----------------------------------------------------------------------
	//   "LARGE" IS A MULTIPLIER TO DISCOURAGE RANGE CONTRACTIONS
	COMMOD9->LARGE = 7777.0f;
	//----------------------------------------------------------------------
	COMMOD9->GRAPHF = 1;
	COMMOD9->RUNGRF = 2;
	//----------------------------------------------------------------------
	COMMOD9->TERMF = 1;

	COMMOD9->CALIBF = 0;	

	if(!::fileExists(COMMOD9->CNFGFILE))
	{
		cout<<" ABORTING: Unable to find "<<COMMOD9->CNFGFILE<<endl;
		cout<<" Should be in same directory as CONOP++.EXE"<<endl;
		COMMOD9->STOPF=1;
		return;
	}

	InitConfigNamelist();
	ReadConfigNamelist();

	COMMOD9->FREQUENCY = FREQUENCY;
	COMMOD9->NEAR = NEARENOUGH;
	COMMOD9->NSCT = SECTIONS;
	COMMOD9->NSPC = TAXA;
	COMMOD9->MAXDIV = COMMOD9->NSPC;
	COMMOD9->TWOSPC = 2 * TAXA;
	COMMOD9->MXCOEX = (COMMOD9->NSPC * (COMMOD9->NSPC - 1)) / 2;
	COMMOD9->MXFALA = (COMMOD9->NSPC * COMMOD9->NSPC);
	COMMOD9->NEVNT = EVENTS + (2 * TAXA);
	COMMOD9->NOTHR = EVENTS;
	COMMOD9->MXLVL = MAX_LEVELS;
	COMMOD9->MXLBL = MAX_LABELS;
	COMMOD9->NSTOP = STEPS;
	COMMOD9->NOUTER = STEPS;
	COMMOD9->NINNER = TRIALS;
	COMMOD9->NTRIES = 0;
	COMMOD9->NTRAJ = 1;

	if (COMMOD9->NTRAJ > COMMOD9->NINNER)
		COMMOD9->NTRAJ = COMMOD9->NINNER;

	if ((STARTSECT > 0) && (STARTSECT <= COMMOD9->NSCT))
		COMMOD9->JSTART = STARTSECT-1;
	else
		COMMOD9->JSTART = 0;

	if ((STARTEVENT > 0) && (STARTEVENT <= COMMOD9->NEVNT))
	{
		COMMOD9->ESTART = STARTEVENT - 1;
		COMMOD9->XEVNT = STARTEVENT - 1;
	}
	else
	{

		COMMOD9->XEVNT =int(random(0,COMMOD9->NEVNT));       
		COMMOD9->ESTART = -1;
	}  

	COMMOD9->STARTT = STARTEMP;
	if (COMMOD9->STARTT < 0.0) COMMOD9->STARTT = 0.0;

	COMMOD9->R = 0.98;
	if (RATIO <= 1.00) COMMOD9->R = RATIO;

	COMMOD9->Rx = COMMOD9->R;

	COMMOD9->KMX = 100000000.0;

	COMMOD9->KTS = std::max(0.0, TEASER);

	if (COMMOD9->KTS > 0.0) COMMOD9->KMX = std::min(COMMOD9->KMX, COMMOD9->KTS);

	COMMOD9->KSM = std::max(0.0, SMOOTHER);

	if (COMMOD9->KSM > 0.0) COMMOD9->KMX = std::min(COMMOD9->KMX, COMMOD9->KSM);

	COMMOD9->KSQ = std::max(0.0, SQUEEZER);

	if (COMMOD9->KSQ > 0.0) COMMOD9->KMX = std::min(COMMOD9->KMX, COMMOD9->KSQ);

	COMMOD9->KSH = std::max(0.0, SHRINKER);

	if (COMMOD9->KSH > 0.0) COMMOD9->KMX = std::min(COMMOD9->KMX, COMMOD9->KSH);

	if (COMMOD9->KMX == 100000000.0) COMMOD9->KMX = 1.0;

	COMMOD9->INFILE.assign(LOADFILE);
	COMMOD9->RAWFILE.assign(PREPFILE);

	COMMOD9->SCTFILE.assign(SECTFILE);
	COMMOD9->STAGFILE.assign(SECTTAGFILE);
	COMMOD9->SCTTAGS.assign(SECTTAGS);

	if (COMMOD9->SCTFILE.substr(0, 3) == "off") COMMOD9->SCTFILE = "OFF" + COMMOD9->SCTFILE.substr(3);

	COMMOD9->USERBEST = BESTKNOWN;
	::transform(LABELFILE.begin(),LABELFILE.end(),LABELFILE.begin(),::toupper);
	COMMOD9->LBLFILE.assign(LABELFILE);

	if (COMMOD9->LBLFILE.substr(0, 3) == "off") COMMOD9->LBLFILE = "OFF" + COMMOD9->LBLFILE.substr(3);

	COMMOD9->EVTFILE.assign(EVENTFILE);
	COMMOD9->ETAGFILE.assign(EVENTTAGFILE);
	COMMOD9->EVTTAGS.assign(EVENTTAGS);
	COMMOD9->INITSOL.assign(STARTFILE);
	::transform(STEPFILE.begin(),STEPFILE.end(),STEPFILE.begin(),::toupper);
	COMMOD9->STEPSOL.assign(STEPFILE);
	COMMOD9->STEPTMP.assign("Steptemp.dat");

	if (COMMOD9->STEPSOL.substr(0, 3) == "off") COMMOD9->STEPSOL = "OFF" + COMMOD9->STEPSOL.substr(3);

	COMMOD9->BESTSOL.assign(BESTARTFILE);
	COMMOD9->OUTMAIN.assign(UNLOADMAIN);
	COMMOD9->OUTSECT.assign(UNLOADSECT);
	COMMOD9->OUTEVNT.assign(UNLOADEVNT);

	::trim(EXCLUSIVES);
	::transform(EXCLUSIVES.begin(),EXCLUSIVES.end(),EXCLUSIVES.begin(),::toupper);

	if(EXCLUSIVES=="YES" || EXCLUSIVES=="ON")
	{
		COMMOD9->JSPANF = 1;
	}
	else if(EXCLUSIVES=="NO" || EXCLUSIVES=="OFF")
	{
		COMMOD9->JSPANF = 0;
	}
	else
	{
		COMMOD9->JSPANF = 0;
	}
	::transform(LOADNAMES.begin(), LOADNAMES.end(), LOADNAMES.begin(), ::toupper);
	if (LOADNAMES == "NO") {
		COMMOD9->LOADNAMES = false;
	}
	else {
		COMMOD9->LOADNAMES = true;
	}
	::transform(ANNEALONLY.begin(), ANNEALONLY.end(), ANNEALONLY.begin(), ::toupper);
	if (ANNEALONLY == "YES") {
		COMMOD9->ANNEALONLY = true;
	}
	else {
		COMMOD9->ANNEALONLY = false;
	}
	::transform(LOADCOEX.begin(), LOADCOEX.end(), LOADCOEX.begin(), ::toupper);
	if (LOADCOEX == "YES") {
		COMMOD9->LOADCOEX = true;
	}
	else {
		COMMOD9->LOADCOEX = false;
	}
	::transform(LOADCEFB4L.begin(), LOADCEFB4L.end(), LOADCEFB4L.begin(), ::toupper);

	if (LOADCEFB4L == "YES") {
		COMMOD9->LOADCEFb4L = true;
	}
	else {
		COMMOD9->LOADCEFb4L = false;
	}

	COMMOD9->SLNF = 1;	

	::transform(SOLNLIST.begin(),SOLNLIST.end(),SOLNLIST.begin(),::toupper);

	if (SOLNLIST.substr(0, 3)== "OFF") COMMOD9->SLNF = 0;	

	if ((SOLNLIST.substr(0, 3)== "OFF") ||(SOLNLIST.substr(0, 3)== "NEW"))
	{
		COMMOD9->SLNFILE.assign(SOLNLIST.substr(3));		       
		COMMOD9->TTLFILE.assign(SOLNLIST.substr(3, SOLNLIST.find_first_of(".") - 3 + 1) + "ttl");
	}
	else
	{
		COMMOD9->SLNFILE.assign(SOLNLIST);		            
		COMMOD9->TTLFILE.assign(SOLNLIST.substr(0, SOLNLIST.find_first_of(".") + 1) + "ttl");
	}

	if (SOLNLIST.substr(0, 3) == "NEW")
	{
		fstream _file;
		_file.open(COMMOD9->SLNFILE.c_str(),ios::out);
		_file.close();
		_file.open(COMMOD9->TTLFILE.c_str(),ios::out);
		_file.close();   
	}


	COMMOD9->NCOL = 7;
	COMMOD9->CDF = 0;
	if (COLUMNS > 0) COMMOD9->NCOL = COLUMNS;

	if (COLUMNS <= 0) COMMOD9->CDF = 1;

	COMMOD9->ABFILE.assign(OBSDFILE);
	COMMOD9->ALBETFILE.assign(PLCDFILE);
	COMMOD9->DELTAFILE.assign(EXTNFILE);
	COMMOD9->GRIDF = 0;

	::transform(CURVFILE.begin(),CURVFILE.end(),CURVFILE.begin(),::toupper);
	if (CURVFILE.substr(0, 3)== "OFF") CURVFILE = CURVFILE.substr(3);

	COMMOD9->BESTFILE.assign(CURVFILE);

	if (COMMOD9->BESTFILE.substr(0, 3)!= "OFF") COMMOD9->GRIDF = 1;

	::transform(CRV2FILE.begin(),CRV2FILE.end(),CRV2FILE.begin(),::toupper);
	if (CRV2FILE.substr(0, 3)== "OFF") CRV2FILE = CRV2FILE.substr(3);

	COMMOD9->SCNDFILE.assign(CRV2FILE);
	COMMOD9->CMPSTFILE.assign(COMPOSFILE);
	COMMOD9->COMPNO = COMPOSNMBR;
	if (COMMOD9->COMPNO == 0) COMMOD9->SLNF = 0;

	//abbreviations
	//STANDARD - STD ST S
	//MAXIMUM  - MAX MX M X
	//AVERAGE  - AVG AV A V
	//ZERO     - ZRO Z
	//Z-SCORE  - Z
	//MINIMUM  - MIN
	::transform(COMPOSTYPE.begin(),COMPOSTYPE.end(),COMPOSTYPE.begin(),::toupper);
	::trim(COMPOSTYPE);

	if(COMPOSTYPE=="ORD")
	{
		COMMOD9->COMPF = 1;
	}
	else if(COMPOSTYPE=="STD" || COMPOSTYPE=="AST" || COMPOSTYPE=="AVS")
	{
		COMMOD9->COMPF = 2;
	}
	else if(COMPOSTYPE=="ZTD" || COMPOSTYPE=="ZST" || COMPOSTYPE=="ZAS")
	{
		COMMOD9->COMPF = 3;
	}
	else if(COMPOSTYPE=="MAX")
	{
		COMMOD9->COMPF = 4;
	}
	else if(COMPOSTYPE=="ZMX")
	{
		COMMOD9->COMPF = 5;
	}
	else if(COMPOSTYPE=="AVG")
	{
		COMMOD9->COMPF = 6;
	}
	else if(COMPOSTYPE=="MIN")
	{
		COMMOD9->COMPF = 7;
	}
	else if(COMPOSTYPE=="ZAV" || COMPOSTYPE=="NIL" || COMPOSTYPE=="ZRO" || COMPOSTYPE=="ZMN")
	{
		COMMOD9->COMPF = 8;
	}
	else if(COMPOSTYPE=="MST" || COMPOSTYPE=="XST" || COMPOSTYPE=="ZMS")
	{
		COMMOD9->COMPF = 9;
	}
	else if(COMPOSTYPE=="ZXS" || COMPOSTYPE=="ZMS")
	{
		COMMOD9->COMPF = 10;
	}
	else
	{
		COMMOD9->DFLTCOMP = true;
		COMMOD9->COMPF = 3;
	}

	COMMOD9->COEXFILE.assign(COEXISTFILE);
	COMMOD9->FB4LFILE.assign(FAD_LADFILE);
	COMMOD9->ORDRFILE.assign(ORDERFILE);
	COMMOD9->RUNLOG.assign(RUNLOGFILE);
	COMMOD9->CIRCFILE.assign(CULLFILE);
	COMMOD9->PROJNAME.assign(PROJECT);
	::trim(PENALTY);
	::trim(FORCECOEX);
	::trim(LETCONTRACT);
	COMMOD9->PROJTYP1 = PENALTY + "  COEX=" + FORCECOEX +  "  LTCTRCT=" + LETCONTRACT;
	::trim(SOLVER);
	::trim(HOODSIZE);
	COMMOD9->PROJTYP2 = SOLVER + "  " + HOODSIZE;

	COMMOD9->NABRGEN = 1;
	::transform(HOODSIZE.begin(),HOODSIZE.end(),HOODSIZE.begin(),::toupper);
	if(HOODSIZE=="BIG")
	{
		COMMOD9->NABRGEN = 1;
	}
	else if(HOODSIZE=="SMALL" || HOODSIZE=="REOPT")
	{
		COMMOD9->NABRGEN = 2;
	}
	else if(HOODSIZE=="DOUBLE")
	{
		COMMOD9->NABRGEN = 3;
	}
	else
	{
		COMMOD9->DFLTNABR = true;
		COMMOD9->NABRGEN = 1;
	}



	COMMOD9->CONTF = 0;
	::transform(STARTYPE.begin(),STARTYPE.end(),STARTYPE.begin(),::toupper);
	::trim(STARTYPE);
	if(STARTYPE=="SECT")
	{
		//start from one section
		COMMOD9->INIGEN = 1;
	}
	else if(STARTYPE=="FILE")
	{
		//start from the STARTFILE
		COMMOD9->INIGEN = 2;
	}
	else if(STARTYPE=="RAND")
	{
		//start from a random sequence
		COMMOD9->INIGEN = 3;
	}
	else if(STARTYPE=="BEST")
	{
		//start from the BESTFILE
		COMMOD9->INIGEN = 4;
	}
	else if(STARTYPE=="STEP")
	{
		//start from the STEPFILE
		COMMOD9->INIGEN = 5;
	}
	else if(STARTYPE=="CONT")
	{
		//start from the STEPFILE
		COMMOD9->CONTF = 1;
		COMMOD9->INIGEN = 5;
	}
	else if(STARTYPE=="PART")
	{
		COMMOD9->INIGEN = 6;
	}
	else
	{
		COMMOD9->DFLTSTART = true;
		COMMOD9->INIGEN = 3;
	}

	if ((EVENTS > 0) && (STARTYPE == "SECT")) COMMOD9->INIGEN = 3;

	COMMOD9->WTF = 1;
	COMMOD9->cutco = 0;
	::transform(WEIGHTING.begin(),WEIGHTING.end(),WEIGHTING.begin(),::toupper);
	::trim(WEIGHTING);

	if(WEIGHTING=="OFF")
	{
		COMMOD9->WTF = 0;
	}
	else if(WEIGHTING=="ON" || WEIGHTING=="FILE")
	{
		COMMOD9->WTF = 1;
	}
	else if(WEIGHTING=="COEX" || WEIGHTING=="COEX^-1")
	{
		COMMOD9->WTF = 2;
	}
	else if(WEIGHTING=="COEX2" || WEIGHTING=="COEX^-2")
	{
		COMMOD9->WTF = 22;
	}
	else if(WEIGHTING=="COEX3" || WEIGHTING=="COEX^-3")
	{
		COMMOD9->WTF = 23;
	}
	else if(WEIGHTING=="COEX%1")
	{
		COMMOD9->WTF = 3;
		COMMOD9->cutco = 1;
	}
	else if(WEIGHTING=="COEX%2")
	{
		COMMOD9->WTF = 3;
		COMMOD9->cutco = 2;
	}
	else if(WEIGHTING=="COEX%5")
	{
		COMMOD9->WTF = 3;
		COMMOD9->cutco = 5;
	}
	else if(WEIGHTING=="COEX%10")
	{
		COMMOD9->WTF = 3;
		COMMOD9->cutco = 10;
	}
	else if(WEIGHTING=="COEX%15")
	{
		COMMOD9->WTF = 3;
		COMMOD9->cutco = 15;
	}
	else if(WEIGHTING=="COEX%20")
	{
		COMMOD9->WTF = 3;
		COMMOD9->cutco = 20;
	}
	else
	{
		COMMOD9->DFLTWT = true;
	}

	::transform(FORCECOEX.begin(),FORCECOEX.end(),FORCECOEX.begin(),::toupper);
	::trim(FORCECOEX);

	if(FORCECOEX=="OFF")
	{
		COMMOD9->COXSTF = 4;
	}
	else if(FORCECOEX=="SS")
	{
		COMMOD9->COXSTF = 3;
	}
	else if(FORCECOEX=="SL" || FORCECOEX=="ON")
	{
		COMMOD9->COXSTF = 2;
	}
	else if(FORCECOEX=="ALL")
	{
		COMMOD9->COXSTF = 0;
	}
	else
	{
		COMMOD9->DFLTCOEX = true;
		COMMOD9->COXSTF = 2;
	}


	if (COMMOD9->NSPC == 0) COMMOD9->COXSTF = 4;

	::transform(FORCEFB4L.begin(),FORCEFB4L.end(),FORCEFB4L.begin(),::toupper);
	::trim(FORCEFB4L);
	if(FORCEFB4L=="OFF" || FORCEFB4L=="NO" || FORCEFB4L=="N")
	{		
		COMMOD9->FB4LF = 0;
	}
	else if(FORCEFB4L=="ON" || FORCEFB4L=="YES" || FORCEFB4L=="Y" || FORCEFB4L=="SS")
	{		
		COMMOD9->FB4LF = 1;
	}
	else if(FORCEFB4L=="SL")
	{	
		COMMOD9->FB4LF = 1;
		COMMOD9->COXSTF = 2;
	}
	else if(FORCEFB4L=="FILE" || FORCEFB4L=="DISK")
	{
		COMMOD9->FB4LF = 2;
	}
	else
	{
		COMMOD9->DFLTFb4L = true;
		COMMOD9->FB4LF = 0;
	}	

	if (COMMOD9->NSPC == 0) COMMOD9->FB4LF = 0;

	::transform(LETCONTRACT.begin(),LETCONTRACT.end(),LETCONTRACT.begin(),::toupper);
	::trim(LETCONTRACT);
	if(LETCONTRACT=="SL" || LETCONTRACT=="ON")
	{
		COMMOD9->CONTRACT = 3;
	}
	else if(LETCONTRACT=="SS")
	{
		COMMOD9->CONTRACT = 2;
	}
	else if(LETCONTRACT=="FILE")
	{
		COMMOD9->CONTRACT = 1;
	}
	else
	{
		COMMOD9->CONTRACT = 0;
	}

	::transform(USENEGATIVE.begin(),USENEGATIVE.end(),USENEGATIVE.begin(),::toupper);
	::trim(USENEGATIVE);
	if(USENEGATIVE=="SS" || USENEGATIVE=="ON")
	{
		COMMOD9->NEGATF = 1;
	}
	else if(USENEGATIVE=="SL" || USENEGATIVE=="COEX")
	{
		COMMOD9->NEGATF = 2;
	}
	else if(USENEGATIVE=="OFF")
	{
		COMMOD9->NEGATF = 0;
	}
	else
	{
		COMMOD9->NEGATF = 0;
	}


	::transform(HOMERANGE.begin(),HOMERANGE.end(),HOMERANGE.begin(),::toupper);
	::trim(HOMERANGE);
	if(HOMERANGE=="SS")
	{
		COMMOD9->HOMEF = 1;
	}
	else if(HOMERANGE=="SL" || HOMERANGE=="COEX")
	{
		COMMOD9->HOMEF = 2;
	}
	else
	{
		COMMOD9->HOMEF = 1;
	}


	::transform(SOLVER.begin(),SOLVER.end(),SOLVER.begin(),::toupper);
	::trim(SOLVER);

	if(SOLVER=="ANNEAL" || SOLVER=="REOPT")
	{
		COMMOD9->SOLVEF = 0;
	}
	else if(SOLVER=="ANNUDG")
	{
		COMMOD9->SOLVEF = 0;
		COMMOD9->NUDGER = true;
	}
	else if(SOLVER=="GREEDY")
	{
		COMMOD9->SOLVEF = 2;
	}
	else if(SOLVER=="TEMPER")
	{
		COMMOD9->SOLVEF = 3;
	}
	else if(SOLVER=="SQUEEZE" || SOLVER=="SQUASH")
	{
		COMMOD9->SOLVEF = 4;
	}
	else if(SOLVER=="SQUEAL")
	{
		COMMOD9->SOLVEF = 5;
	}
	else if(SOLVER=="SHRINK")
	{
		COMMOD9->SOLVEF = 6;
	}
	else if(SOLVER=="SHREAL")
	{
		COMMOD9->SOLVEF = 7;
	}
	else if(SOLVER=="STACK" || SOLVER=="TEASE")
	{
		COMMOD9->SOLVEF = 8;
	}
	else if(SOLVER=="STEAL" || SOLVER=="ANNEAS")
	{
		COMMOD9->SOLVEF = 9;
	}
	else
	{
		COMMOD9->SOLVEF = 0;
	}

	COMMOD9->PEN2F = 0;
	if (COMMOD9->SOLVEF >= 4) COMMOD9->PEN2F = 1;
	if ((COMMOD9->KSM > 0.0) || (COMMOD9->KSQ > 0.0) ||
			(COMMOD9->KSH > 0.0) || (COMMOD9->KTS > 0.0)) COMMOD9->PEN2F = 1;
	COMMOD9->PEN2M = COMMOD9->PEN2F;

	::transform(PENALTY.begin(),PENALTY.end(),PENALTY.begin(),::toupper);
	::trim(PENALTY);
	if(PENALTY=="EVENTUAL")
	{
		COMMOD9->PENF = -1;
	}
	else if(PENALTY=="INTERVAL")
	{
		COMMOD9->PENF = 0;
	}
	else if(PENALTY=="LEVEL")
	{
		COMMOD9->PENF = 1;
	}
	else if(PENALTY=="ORDINAL")
	{
		COMMOD9->PENF = 2;
	}
	else if(PENALTY=="SPATIAL")
	{
		COMMOD9->PENF = 3;
	}
	else if(PENALTY=="RASCAL")
	{
		COMMOD9->PENF = 4;
	}
	else if(PENALTY=="RASCER")
	{
		COMMOD9->PENF = 4;
		COMMOD9->RASCon = true;
	}
	else if(PENALTY=="ROYAL")
	{
		COMMOD9->PENF = 5;
	}
	else if(PENALTY=="SEQUEL")
	{
		COMMOD9->PENF = 6;		
		COMMOD9->FB4LF = 1;
	}
	else if(PENALTY=="MOMENTAL")
	{
		COMMOD9->PENF = 7;			   
		if (COMMOD9->FB4LF == 0) COMMOD9->FB4LF = 1;
	}
	else
	{
		COMMOD9->DFLTPEN = true;
		COMMOD9->PENF = 1;
	}      

	COMMOD9->PENM = COMMOD9->PENF;

	::transform(STACKER.begin(),STACKER.end(),STACKER.begin(),::toupper);
	::trim(STACKER);
	if(STACKER=="OFF")
	{
		COMMOD9->STKF = 0;
	}
	else if(STACKER=="THRU")
	{
		COMMOD9->STKF = 1;
		COMMOD9->STAKNAME.assign("THRU");
	}
	else if(STACKER=="INCL")
	{
		COMMOD9->STKF = 2;
		COMMOD9->STAKNAME.assign("INCL");
	}
	else if(STACKER=="DIST")
	{
		COMMOD9->STKF = 3;
		COMMOD9->STAKNAME.assign("DIST");
	}
	else if(STACKER=="FREQ")
	{
		COMMOD9->STKF = 4;
		COMMOD9->STAKNAME.assign("FREQ");
	}
	else if(STACKER=="EXIT")
	{
		COMMOD9->STKF = 5;
		COMMOD9->STAKNAME.assign("EXIT");
	}
	else if(STACKER=="PROP")
	{
		COMMOD9->STKF = 6;
		COMMOD9->STAKNAME.assign("PROP");
	}
	else if(STACKER=="OLAP" || STACKER=="SPAN")
	{
		COMMOD9->STKF = 7;
		COMMOD9->STAKNAME.assign("SPAN");
	}
	else if(STACKER=="COEX")
	{
		COMMOD9->STKF = 8;
		COMMOD9->STAKNAME.assign("COEX");
	}
	else if(STACKER=="FB4L")
	{
		COMMOD9->STKF = 9;
		COMMOD9->STAKNAME.assign("FB4L");
		if (COMMOD9->FB4LF == 0) COMMOD9->FB4LF = 1;
	}
	else
	{
		COMMOD9->STKF = 0;
	}

	if((COMMOD9->STKF!=0)&&(COMMOD9->PENF>=2)&&(COMMOD9->PENF!=6)&&
			(COMMOD9->STKF<7))COMMOD9->STKF=7;

	if(COMMOD9->PENF==6)COMMOD9->STKF=8;	

	COMMOD9->FITOUTF = 1;
	::transform(FITS_OUT.begin(),FITS_OUT.end(),FITS_OUT.begin(),::toupper);
	::trim(FITS_OUT);
	if (FITS_OUT == "OFF") COMMOD9->FITOUTF = 0;

	COMMOD9->RUNOUTF = 1;
	::transform(CNFG_OUT.begin(),CNFG_OUT.end(),CNFG_OUT.begin(),::toupper);
	::trim(CNFG_OUT);
	if (CNFG_OUT == "OFF") COMMOD9->RUNOUTF = 0;

	COMMOD9->SEQOUTF = 1;
	::transform(SEQN_OUT.begin(),SEQN_OUT.end(),SEQN_OUT.begin(),::toupper);
	::trim(SEQN_OUT);
	if (SEQN_OUT == "OFF") COMMOD9->SEQOUTF = 0;

	COMMOD9->INCOUTF = 1;
	::transform(INCR_OUT.begin(),INCR_OUT.end(),INCR_OUT.begin(),::toupper);
	::trim(INCR_OUT);
	if (INCR_OUT == "OFF") COMMOD9->INCOUTF = 0;

	COMMOD9->LOCOUTF = 1;
	::transform(LOC_OUT.begin(),LOC_OUT.end(),LOC_OUT.begin(),::toupper);
	::trim(LOC_OUT);
	if (LOC_OUT == "OFF") COMMOD9->LOCOUTF = 0;

	COMMOD9->OBSOUTF = 1;
	::transform(OBS_OUT.begin(),OBS_OUT.end(),OBS_OUT.begin(),::toupper);
	::trim(OBS_OUT);
	if (OBS_OUT == "OFF") COMMOD9->OBSOUTF = 0;

	COMMOD9->CMPOUTF = 1;
	::transform(COMP_OUT.begin(),COMP_OUT.end(),COMP_OUT.begin(),::toupper);
	::trim(COMP_OUT);
	if (COMP_OUT == "OFF") COMMOD9->CMPOUTF = 0;

	COMMOD9->SCTOUTF = 1;
	::transform(SECT_OUT.begin(),SECT_OUT.end(),SECT_OUT.begin(),::toupper);
	::trim(SECT_OUT);
	if (SECT_OUT == "OFF") COMMOD9->SCTOUTF = 0;

	if (SECT_OUT == "MIN") COMMOD9->SCTOUTF = 2;

	COMMOD9->COXOUTF = 1;
	::transform(COEX_OUT.begin(),COEX_OUT.end(),COEX_OUT.begin(),::toupper);
	::trim(COEX_OUT);
	if (COEX_OUT == "OFF") COMMOD9->COXOUTF = 0;

	if (COEX_OUT == "COUNT") COMMOD9->COXOUTF = 2;

	COMMOD9->EVTOUTF = 1;
	::transform(EVNT_OUT.begin(),EVNT_OUT.end(),EVNT_OUT.begin(),::toupper);
	::trim(EVNT_OUT);
	if (EVNT_OUT == "OFF") COMMOD9->EVTOUTF = 0;


	COMMOD9->AUTF = 0;
	COMMOD9->ADAF = 0;
	COMMOD9->PAUSF = 0;
	COMMOD9->WALKF = 0;
	::transform(PAUSES.begin(),PAUSES.end(),PAUSES.begin(),::toupper);
	::trim(PAUSES);
	if(PAUSES=="OFF")
	{
		COMMOD9->PAUSF = 1;
	}
	else if(PAUSES=="RPT" || PAUSES=="REP" || PAUSES=="REPEAT")
	{
		COMMOD9->PAUSF = 2;
	}
	else if(PAUSES=="ADD")
	{
		COMMOD9->PAUSF = 3;
	}
	else if(PAUSES=="BAR")
	{
		COMMOD9->PAUSF = 4;
	}
	else if(PAUSES=="AUT" || PAUSES=="AUTO")
	{
		COMMOD9->PAUSF = 5;
		COMMOD9->NOUTER = 1;
	}
	else if(PAUSES=="ADA" || PAUSES=="ADA")
	{
		COMMOD9->PAUSF = 5;
		COMMOD9->NOUTER = 1;
		COMMOD9->ADAF = 1;
	}
	else if(PAUSES=="WLK" || PAUSES=="WALK")
	{
		COMMOD9->PAUSF = 1;
		COMMOD9->NOUTER = 1;
	}   


	COMMOD9->FIXF = 0;
	COMMOD9->MOVEF = 0;
	::transform(SHOWMOVIES.begin(),SHOWMOVIES.end(),SHOWMOVIES.begin(),::toupper);
	::trim(SHOWMOVIES);
	if(SHOWMOVIES=="OFF")
	{
		COMMOD9->RUNGRF = 0;
	}
	else if(SHOWMOVIES=="DIV")
	{
		COMMOD9->RUNGRF = 0;
		COMMOD9->MOVEF = 1;
	}
	else if(SHOWMOVIES=="PEN" || SHOWMOVIES=="CHT" || SHOWMOVIES=="BST")
	{
		COMMOD9->RUNGRF = 2;
	}
	else if(SHOWMOVIES=="EVT")
	{
		COMMOD9->RUNGRF = 4;
		COMMOD9->GRIDF = 1;
	}
	else if(SHOWMOVIES=="LAG")
	{
		COMMOD9->RUNGRF = 4;
		COMMOD9->FIXF = 1;
		COMMOD9->GRIDF = 1;
	}
	else if(SHOWMOVIES=="AIM")
	{
		COMMOD9->RUNGRF = 4;
		COMMOD9->FIXF = 2;
		COMMOD9->GRIDF = 1;
	}
	else if(SHOWMOVIES=="FIX")
	{
		COMMOD9->RUNGRF = 4;
		COMMOD9->FIXF = 3;
		COMMOD9->GRIDF = 1;
	}
	else if(SHOWMOVIES=="END")
	{
		COMMOD9->RUNGRF = 4;
		COMMOD9->FIXF = 4;
		COMMOD9->GRIDF = 1;
	}
	else if(SHOWMOVIES=="FAR")
	{
		COMMOD9->RUNGRF = 4;
		COMMOD9->FIXF = 5;
		COMMOD9->GRIDF = 1;
	}
	else
	{
		COMMOD9->RUNGRF = 2;
	}

	::transform(CURVFILE.begin(),CURVFILE.end(),CURVFILE.begin(),::toupper);
	if (CURVFILE.substr(0, 3) == "OFF") COMMOD9->GRIDF = 1;

	COMMOD9->TRJGRF = 0;
	::transform(TRAJECTORY.begin(),TRAJECTORY.end(),TRAJECTORY.begin(),::toupper);
	::trim(TRAJECTORY);
	if(TRAJECTORY=="OFF")
	{
		COMMOD9->TRJGRF = 0;
	}
	else if(TRAJECTORY=="ALL")
	{
		COMMOD9->TRJGRF = 4;
	}
	else if(TRAJECTORY=="BAD")
	{
		COMMOD9->TRJGRF = 3;
	}
	else if(TRAJECTORY=="GOOD")
	{
		COMMOD9->TRJGRF = 2;
	}
	else if(TRAJECTORY=="COOL")
	{
		COMMOD9->TRJGRF = 1;
	}


	COMMOD9->GRID2F = 0;
	if ((COMMOD9->SOLVEF == 8) || (COMMOD9->SOLVEF == 9) || (COMMOD9->KTS > 0.0)) COMMOD9->GRID2F = 1;

	if (COMMOD9->CDF != 1)
	{
		::fstream _file;
		_file.open(COMMOD9->RUNLOG.c_str(),ios::out);		

		_file.close();		
	}

	return;
}
