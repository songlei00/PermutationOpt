#include <string>

class ANNEAL
{
	public:
		static ANNEAL* Singleton()
		{
			if (g_singleton == NULL)
				g_singleton = new ANNEAL;

			return g_singleton;
		}

		void RunIt(int NUMBERPROCESS,int MYID);

	private:
		ANNEAL() { };
		~ANNEAL() {};

		void FILLBEST(int ISCOR, double RSCOR);

		static	 ANNEAL* g_singleton;
};
