FLAGS=`pkg-config --cflags opencv4`
LIBS=`pkg-config --libs opencv4`
all: cpu gpu
gpu:
	nvcc -c src/gauss_gpu.cu
	nvcc -ccbin g++ gauss_gpu.o src/main.cpp -o exec_gpu ${LIBS} ${FLAGS} -lcuda
cpu:
	g++ -o exec_cpu  src/gauss_cpu.cpp ${FLAGS} ${LIBS}
clean:
	rm exec_*
	rm *.o
	rm images/*_cpu.jpg
	rm images/*_gpu.jpg
