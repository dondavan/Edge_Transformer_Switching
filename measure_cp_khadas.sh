git pull
scons Werror=1 debug=0 asserts=0 neon=1 opencl=1 examples=1 os=linux arch=armv8a measure=1 -j4
sshpass -p 'khadas' scp build/libarm_compute* Khadas.local:/home/khadas/run/lib 
sshpass -p 'khadas' scp build/examples/graph_bert_* Khadas.local:/home/khadas/run