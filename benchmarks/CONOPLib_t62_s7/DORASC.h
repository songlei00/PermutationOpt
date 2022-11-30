#include <string>

class DORASC
{
	public:
		static DORASC* Singleton()
		{
			if (g_singleton == NULL)
				g_singleton = new DORASC;

			return g_singleton;
		}

		void RunIt();

	private:
		DORASC() { };
		~DORASC() {};


		static DORASC* g_singleton;
};
