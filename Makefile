.PHONY: all clean

all: demo2 demo5

demo2: demo2_wasm.cpp
	emcc demo2_wasm.cpp -o demo2_wasm.js -O3 -sALLOW_MEMORY_GROWTH

demo5: demo5_wasm_webgpu.cpp
	emcc demo5_wasm_webgpu.cpp -o demo5_wasm_webgpu.js --use-port=emdawnwebgpu --closure=1 -O3 -sALLOW_MEMORY_GROWTH -sMAX_WEBGL_VERSION=0

clean:
	rm -f demo2_wasm.js demo2_wasm.wasm demo5_wasm_webgpu.js demo5_wasm_webgpu.wasm
