#include <string>

class DOPTLORD
{
	public:
		static DOPTLORD* Singleton()
		{
			if (g_singleton == NULL)
				g_singleton = new DOPTLORD;

			return g_singleton;
		}

		void RunIt();

	private:
		DOPTLORD() { };
		~DOPTLORD() {};

		void UPDATE(int IRW, int MRW, int rong, int rite, int pass);

		static DOPTLORD* g_singleton;
};
