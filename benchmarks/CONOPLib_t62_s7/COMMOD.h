#include <string>

using std::string;

typedef unsigned char BYTE;


class COMMOD
{	
	public:

		int MXLVL, MXLBL;       
		int WTF, cutco;
		int CONTRACT, COXSTF, FB4LF, INIGEN, NABRGEN;
		int SOLVEF, ONEGRF, SLNF;
		int JSTART, RUNGRF, TWOGRF, FNCGRF, FIXF, PAUSF, WALKF, GRIDF, GRID2F;
		int MOVEF, STKF, HOMEF, NOTHR, MAXDIV;
		int NTRIES, LSTRIES, CNTRIES, NBETR, NSTOP;
		int NTRAJ, NCALIB, CALIBF, GRAPHF, TRJGRF, CHTGRF, CTRF;
		int PENF, PEN2F, PENM, PEN2M;
		int VIDEOF, COMPNO, COMPF, XEVNT, YEVNT, XPOSN, ESTART;
		float LARGE;
		int COXOUTF, EVTOUTF, SCTOUTF, FITOUTF, RUNOUTF, SEQOUTF, CONTF;
		int INCOUTF, LOCOUTF, OBSOUTF, CMPOUTF, NEGATF, JSPANF, NEXCL, JEXCL;

		int NREPS, NROWS, NOBS, NMIDS, NBOXS, NSCT, NSPC, NEVNT, TWOSPC, FREQUENCY;
		int TERMF, NCOL;
		int NFALA, MXFALA, NCOEX, MXCOEX;
		int NCOEX_EXCLUDE;//added by HXD 2020-8-18

		int MXITAG, MXJTAG;
		int COEXON, COEXOFF;
		int NEWRANK;
		int STOPF, AUTF, ADAF, CDF;
		int NINNER;
		double NOUTER;
		double R, Rx, T, STARTT, KSM, KSQ, KSH, KTS, KMX, CPUTM;
		double UU;
		bool LOADNAMES, ANNEALONLY, LOADCOEX, LOADCEFb4L;

		//-------------------------------//
		//File names for input and output//
		//-------------------------------//     
		string INFILE;
		string RAWFILE ;
		string SCTFILE ;
		string EVTFILE ;
		string ETAGFILE ;
		string STAGFILE ;
		string SCTTAGS ;
		string EVTTAGS ;
		string LBLFILE ;
		string SLNFILE ;
		string TTLFILE ;
		string OUTMAIN ;
		string OUTSECT ;
		string OUTEVNT ;
		string ABFILE ;
		string ALBETFILE ;
		string DELTAFILE ;
		string BESTFILE ;
		string SCNDFILE ;
		string STEPSOL ;
		string STEPTMP ;
		string CMPSTFILE ;
		string COEXFILE ;
		string FB4LFILE ;
		string ORDRFILE ;
		string CNFGFILE ;
		string LOGFILE ;
		string CIRCFILE ;
		string INITSOL ;
		string BESTSOL ;
		string RUNLOG ;
		string TMPFILE ;
		string ASKFILE ;
		string USRFILE ;
		string PLOTFILE ;
		string PROJNAME , PROJTYP1 ,PROJTYP2 ;
		string STAKNAME ;

		//----------------------------------------//
		//storage for full section and event names//
		//as well as abbreviated nicknames        //
		//----------------------------------------//
		string* SECTNAME;
		string* SECTNICK ;
		string* EVNTNAME ;
		string* EVNTNICK ;
		string* LABLNAME ;
		string* ETAGNAME ;
		string* STAGNAME ;

		//---------------------------------------------------------------//
		//an array for the plotting order for sections                   //
		//an array for the JEXCL sections with exclusive taxa at the ends//
		//---------------------------------------------------------------//
		int* SECTPERM, *EXCLSECT; 

		//-------------------------------//
		//an array for section properties//
		//-------------------------------//
		int** SECTPROP;


		//-------------------------------//
		//flags for "augmented" annealing//
		//-------------------------------//
		bool NUDGUP, NUDGDN, NUDGER;


		//-------------------------------------//
		//flags for default options chosen in  //
		//response to unrecognized .CFG entries//
		//-------------------------------------//
		bool DFLTPEN, DFLTCOMP, DFLTNABR, DFLTSTART;
		bool DFLTWT, DFLTCOEX, DFLTFb4L;


		//----------------------------------------------------------//
		//"faster" version of RASCAL penalty which uses RASC() array//
		//----------------------------------------------------------//
		bool RASCon;


		//----------------------------------//
		//counts the records of input events//
		//----------------------------------//
		int IRCOUNT;


		int *HLEVEL;
		int* RANGE, *COVER;

		int** LIMITS;        
		int **PTLORD;

		short int **RASC;		
		BYTE *FADLAD;		
		BYTE *COEXST;

		int** NEGATIVE, **ELEVEL;     

		int** IROWS;
		int* IROWS_3;		
		short int * ISTATIC_0;
		BYTE *ISTATIC_1;	

		float* NRSTATIC;
		int* IDX_SCT_RSTATIC;
		int RSTATICCount;
		float** VALEVEL;

		float**COMPLVL;


		int** CULLIST;

		int* PERM, *INIPERM, *PXLPERM, *TEMPERM, *BSTPERM, *LSTPERM;

		double BSTPEN, BS2PEN, NXTPEN, PEN, TOOPEN, TO2PEN, MAXPEN, SMPEN, SQPEN,
		       SHPEN, INTPEN, ORDPEN, LVLPEN, RSCPEN, ROYPEN, SEQPEN, SPTPEN, INIPEN,
		       SPANPEN, VERYBEST, USERBEST, TWINBEST, LCLBST, SQBST, SHBST, TSPEN,
		       BSTPLUS, TSBST, SQLST, SHLST, TSLST, NGTPEN, NGHPEN, AUTPEN, NEAR,
		       ASPNPEN, RVLPEN, MOMPEN, CRTPEN;

		double* SCJPEN, *COLPEN;

		int SCTSOL_SIZE;
		unsigned short* HSCTSOL;

		unsigned short	*SCTSOL, *BSCTSOL;
		int HSCTSOL_CHANGE_FLAG;
		int *HSCTSOL_STATE, *HSCTSOL_LASTPOSITION, *HSCTSOL_LASTVALUE;

		int SOL_SIZE; 

		double Erare, Srare, up95, dn95;

		int STAGNANTLOOP;
		int* SectionMaxPenOrder;


		//for output		
		string OutmainSB;
		string BestSolnSB;
		string DeltaSB;
		string AlbetSB;
		string AbSB;
		string CmpstSB;
		string TtlSB;
		string SlnSB;

		bool DEBUG_MODE;


		static COMMOD* Singleton(void)
		{
			if(NULL==g_singleton)
				g_singleton=new COMMOD;

			return g_singleton;
		}

	private:
		static COMMOD* g_singleton;
		COMMOD(void){
			NUDGUP=false, NUDGDN=false, NUDGER=false;	

		};
		~COMMOD(void){if(g_singleton!=NULL)delete g_singleton;}; 			      


};
