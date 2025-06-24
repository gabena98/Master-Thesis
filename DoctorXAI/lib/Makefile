all: generator.so

python.zig:
	zig translate-c /home/gbenanti/.venv/lib/python3.10 -I/home/gbenanti/.venv/lib/python3.10 -I/home/gbenanti/.venv/lib/python3.10 > python.zig

generator.so: main.zig numpy_data.zig python.zig
	zig build-lib -I/home/gbenanti/.venv/lib/python3.10 -I/home/gbenanti/.venv/lib/python3.10 -dynamic -fPIC --name generator -lc main.zig
	mv libgenerator.so generator.so
	rm libgenerator.so.o

release: main.zig out.zig numpy_data.zig python.zig
	zig build-lib -I/usr/include -I/usr/include/python3.11 -dynamic -fPIC --name generator -Doptimize=ReleaseFast main.zig
	mv libgenerator.so generator.so
	rm libgenerator.so.o

clean:
	rm -f generator.so