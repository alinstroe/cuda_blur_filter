FLAGS=`pkg-config --cflags opencv`
LIBS=`pkg-config --libs opencv`
all: cpu gpu
gpu:
	nvcc -c blur.cu
	nvcc -ccbin g++ blur.o main.cpp -I/usr/include/opencv -lopencv_core -lopencv_highgui -lopencv_imgproc -lcuda -lcudart -o blur_gpu `pkg-config opencv --cflags --libs` -lcudart
cpu:
	g++ -o blur_cpu  blur_cpu.cpp `pkg-config opencv --cflags --libs`
clean:
	rm gauss_exec
	rm blur.o
	rm blur_cpu
	rm core.*
