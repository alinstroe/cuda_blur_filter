FLAGS=`pkg-config --cflags opencv`
LIBS=`pkg-config --libs opencv`
all: cpu gpu
gpu:
	nvcc -c blur.cu
	nvcc -ccbin g++ blur.o main.cpp -o blur_gpu ${LIBS} ${FLAGS} -lcuda -lcudart
cpu:
	g++ -o blur_cpu  blur_cpu.cpp ${FLAGS} ${LIBS}
clean:
	rm gauss_exec
	rm blur.o
	rm blur_cpu
