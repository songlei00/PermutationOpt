g++ -shared -fPIC -c COMMOD.cpp Helper.cpp PREPARE.cpp DOINPUT.cpp  DOCOEX.cpp DOFB4L.cpp DOPTLORD.cpp DORASC.cpp GETSTART.cpp Output.cpp  ANNEAL.cpp CONOP.cpp
g++ -shared -fPIC -o CONOPLib.so COMMOD.o Helper.o PREPARE.o DOINPUT.o  DOCOEX.o DOFB4L.o DOPTLORD.o DORASC.o GETSTART.o Output.o  ANNEAL.o CONOP.o

python call.py 
