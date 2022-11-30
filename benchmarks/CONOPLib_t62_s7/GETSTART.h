#include <string>

class GETSTART
{
	public:
		static GETSTART* Singleton()
		{
			if (g_singleton == NULL)
				g_singleton = new GETSTART;

			return g_singleton;
		}

		void RunIt(int OUT);

	private:
		GETSTART() { };
		~GETSTART() {};


		static GETSTART* g_singleton;
};
