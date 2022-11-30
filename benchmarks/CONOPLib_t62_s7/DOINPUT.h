#include <string>
class DOINPUT
{
	public:
		static DOINPUT* Singleton()
		{
			if (g_singleton == NULL)
				g_singleton = new DOINPUT();

			return g_singleton;
		}

		void RunIt(int MYID);

	private:
		DOINPUT() { };
		~DOINPUT() {};	

		void BeforeRunIt(int MYID);

		static DOINPUT* g_singleton;

		int HEVENT, HTYPE, HSECTION, UNPRDF, LSTSECT, LSTEVNT;
		int IEVENT, ITYPE, ISECTION, ILEVEL, IMOVES, IPRD, IUNPRD;
		int JPLACE, JINC, Ityp, Ilbl, LSTTYPE, MFAD, MLAD;
		int IPAIR;
		int I, J, K, N, M, IROW, JOPT, NENDS, JTAG, ITAG, NTAG, NSCAL;
		int A, L;
		double EVENT, TYP, SECTION, LEVEL, MOVES;
		float  VALUE,WTUP, WTDN;
		double MNSPACE, MUSPACE, MINC;
		string JNAME, TAGSTRING;
		string JNICK;
		string INAME;
		string INICK;
		string DIGIT[10];
};
