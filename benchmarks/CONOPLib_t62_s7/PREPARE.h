#include<string>
#include<vector>
#include<map>

using std::string;
using std::vector;
using std::map;

class PREPARE
{
	public:
		static PREPARE* Singleton()
		{
			if (g_singleton == NULL)
				g_singleton = new PREPARE;

			return g_singleton;
		}

		void RunIt(int MYID);	 

		string UPPER;

	private:
		PREPARE() { };
		~PREPARE() {};

		void InitConfigNamelist();
		void ReadConfigNamelist();	
		vector<string> GetKeyValuePair(string line);

		static PREPARE* g_singleton;

		string COEX_OUT, EVNT_OUT, SECT_OUT, FITS_OUT, CNFG_OUT,
		       SEQN_OUT, INCR_OUT, LOC_OUT, OBS_OUT, COMP_OUT;

		map<string, map<string, string> > CONOPNET_CNF_NAMELIST;

		int FREQUENCY, SECTIONS, TAXA, EVENTS, MAX_LEVELS, MAX_LABELS, STEPS, TRIALS,
		    STARTSECT, COLUMNS, COMPOSNMBR, STARTEVENT, STAGNANTLOOP;
		double STARTEMP, RATIO, SMOOTHER, SQUEEZER,
		       SHRINKER, TEASER, BESTKNOWN, NEARENOUGH;

		string LOADFILE, SECTFILE, EVENTFILE, UNLOADMAIN, UNLOADSECT,
		       UNLOADEVNT, ORDERFILE, RUNLOGFILE, BESTARTFILE, LABELFILE,
		       SECTTAGFILE, SECTTAGS, EVENTTAGFILE, EVENTTAGS,
		       STARTFILE, SOLNLIST, PREPFILE, OBSDFILE, PLCDFILE, STEPFILE,
		       EXTNFILE, CURVFILE, CRV2FILE,
		       COEXISTFILE, FAD_LADFILE, COMPOSFILE, CULLFILE;
		string PROJECT;
		string HOODSIZE, STARTYPE, SOLVER, PENALTY, VIDEOMODE;
		string WEIGHTING, PAUSES;
		string FORCECOEX, FORCEFB4L, LETCONTRACT,
		       STACKER, SHOWMOVIES, TRAJECTORY, COMPOSTYPE,
		       USENEGATIVE, HOMERANGE, EXCLUSIVES, LOADNAMES, ANNEALONLY, LOADCOEX, LOADCEFB4L;
};
