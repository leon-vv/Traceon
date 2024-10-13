
.PHONY: clean bdist

traceon/backend/traceon_backend.so: traceon/backend/traceon-backend.c
	clang -O3 -march=native -shared -fPIC -ffast-math -Wno-extern-initializer ./traceon/backend/traceon-backend.c -o traceon/backend/traceon_backend.so -lm 

clean:
	rm -rf ./build ./dist
	rm traceon/backend/traceon_backend.so

sdist:
	python3 ./setup.py sdist


