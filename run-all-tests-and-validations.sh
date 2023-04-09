
echo "================ RUNNING TESTS\n\n"

python3 ./tests/plane-intersection.py
python3 ./tests/triangle-integral.py

echo "\n\n================ RUNNING VALIDATIONS\n\n"

echo "\n===== Capacitance sphere"

python3 ./validation/capacitance-sphere.py 
python3 ./validation/capacitance-sphere.py --symmetry 3d

echo "\n===== Dohi "

python3 ./validation/dohi.py

echo "\n===== Edwards"

python3 ./validation/edwards2007.py
python3 ./validation/edwards2007.py --symmetry 3d

echo "\n===== Simple mirror"

python3 ./validation/simple-mirror.py 
python3 ./validation/simple-mirror.py --symmetry 3d

echo "\n===== Spherical capacitor floating conductor"

python3 ./validation/spherical-capacitor-floating-conductor.py
python3 ./validation/spherical-capacitor-floating-conductor.py --symmetry 3d

echo "\n===== Spherical capacitor tracing"

python3 ./validation/spherical-capacitor.py 
python3 ./validation/spherical-capacitor.py --symmetry 3d

echo "\n===== Two cylinder edwards"

python3 ./validation/two-cylinder-edwards.py
