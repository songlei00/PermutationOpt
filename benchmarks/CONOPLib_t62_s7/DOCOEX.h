#include <string>

class DOCOEX
{
	public:
		static DOCOEX* Singleton()
		{
			if (g_singleton == NULL)
				g_singleton = new DOCOEX;

			return g_singleton;
		}

		void RunIt(int flg);

	private:
		DOCOEX() { };
		~DOCOEX() {};

		static DOCOEX* g_singleton;
};
