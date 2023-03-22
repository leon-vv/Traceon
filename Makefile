
.PHONY: clean bdist

clean:
	rm -rf ./build ./dist
	rm traceon/backend/traceon_backend.so

sdist:
	python3 ./setup.py sdist

traceon/backend/traceon_backend.so: traceon/backend/traceon-backend.c
	clang -O3 -shared -fPIC -ffast-math ./traceon/backend/traceon-backend.c -o traceon/backend/traceon_backend.so

