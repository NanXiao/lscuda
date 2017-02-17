lscuda: lscuda.cu
	nvcc -Wno-deprecated-gpu-targets -o $@ $^

clean:
	rm -f lscuda
