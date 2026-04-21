# webgl-webgpu-draw-times-benchmark
- webgl\webgpu等渲染方式，对渲染对象个数（gldraw次数）的显存开销和性能测试
- 通过每个渲染对象只包含一个三角面，允许设置总数实现


## 结论

- webgl下不同方式的承受能力是相同的，包含threejs、js直接调用、emcc-c++-wasm调用，单帧性能差距很小
- webgpu下，threejs下，显存占用极高，性能常常只有webgl的一半
- emcc-c++-webgpu下，不炸显存的情况下性能可以达到webgl两倍，但显存占用极高
- webgpu显存占用极高原因不明


# WebGL/WebGPU Draw Times Benchmark

- Benchmarks the VRAM usage and performance of rendering a varying number of draw calls using WebGL, WebGPU, etc.
- Each draw call represents a single triangle; the total number of draw calls is configurable.

## Conclusions

- Under WebGL, performance and VRAM usage are comparable across different implementation methods, including Three.js, raw WebGL API, and EMCC C++ WASM. Single-frame performance differences are negligible.
- Under WebGPU with Three.js, VRAM usage is extremely high, and performance is often only half that of WebGL.
- Under WebGPU with EMCC C++, performance can reach twice that of WebGL without out-of-memory errors, but VRAM usage is also extremely high.
- The cause of the extremely high VRAM usage in WebGPU remains unclear.

