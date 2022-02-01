FLAGS=`pkg-config --cflags opencv`
LIBS=`pkg-config --libs opencv`
all: cpu gpu
gpu:
	nvcc -c blur.cu
	nvcc -ccbin g++ blur.o main.cpp -o blur_gpu ${LIBS} ${FLAGS} -lcuda
cpu:
	g++ -o blur_cpu  blur_cpu.cpp ${FLAGS} ${LIBS}
clean:
	rm blur_gpu
	rm blur.o
	rm blur_cpu
	rm *_cpu.jpg
	rm *_gpu.jpg
