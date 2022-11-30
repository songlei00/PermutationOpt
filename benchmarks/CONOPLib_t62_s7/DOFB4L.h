#include <string>

class DOFB4L
{
	public:
		static DOFB4L* Singleton()
		{
			if (g_singleton == NULL)
				g_singleton = new DOFB4L;

			return g_singleton;
		}

		void RunIt();

	private:
		DOFB4L() { };
		~DOFB4L() {};


		static DOFB4L* g_singleton;
};
