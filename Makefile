
.PHONY: clean bdist

voltrace/backend/voltrace_backend.so: voltrace/backend/*.c
	clang -O3 -march=native -shared -fPIC -ffast-math -Wno-extern-initializer ./voltrace/backend/voltrace-backend.c -o voltrace/backend/voltrace_backend.so -lm 

clean:
	rm -rf ./build ./dist
	rm voltrace/backend/voltrace_backend.so

sdist:
	python3 ./setup.py sdist


