// demo5_wasm_webgpu.cpp — C++ / WASM → WebGPU
// 与 demo3_webgl.html 功能对齐：可配置三角面片数量、独立drawCall、
// 3D相机控制、FPS显示
// 编译: emcc demo5_wasm_webgpu.cpp -o demo5_wasm_webgpu.js \
//      --use-port=emdawnwebgpu --closure=1 -O3 -sALLOW_MEMORY_GROWTH \
//      -sMAX_WEBGL_VERSION=0

#include <webgpu/webgpu_cpp.h>

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <emscripten.h>
#include <emscripten/html5.h>

// ======================== 数学工具 ========================
struct Vec3 { float x, y, z; };

static void vec3Sub(const Vec3 &a, const Vec3 &b, Vec3 &out) {
    out.x = a.x - b.x; out.y = a.y - b.y; out.z = a.z - b.z;
}
static void vec3Cross(const Vec3 &a, const Vec3 &b, Vec3 &out) {
    out.x = a.y*b.z - a.z*b.y;
    out.y = a.z*b.x - a.x*b.z;
    out.z = a.x*b.y - a.y*b.x;
}
static float vec3Len(const Vec3 &a) {
    return std::sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}
static void vec3Normalize(const Vec3 &a, Vec3 &out) {
    float l = vec3Len(a);
    if (l > 0) { float il = 1.0f / l; out.x=a.x*il; out.y=a.y*il; out.z=a.z*il; }
}

static void mat4Perspective(float *m, float fovY, float aspect,
                            float nearZ, float farZ) {
    std::memset(m, 0, 16*sizeof(float));
    float f = 1.0f / std::tan(fovY * 0.5f);
    float nf = 1.0f / (nearZ - farZ);
    m[0]  = f / aspect;
    m[5]  = f;
    m[10] = (farZ + nearZ) * nf;
    m[11] = -1.0f;
    m[14] = 2.0f * farZ * nearZ * nf;
}

static void mat4LookAt(float *m, const Vec3 &eye, const Vec3 &center,
                       const Vec3 &up) {
    std::memset(m, 0, 16*sizeof(float));
    Vec3 zAxis; vec3Sub(eye, center, zAxis); vec3Normalize(zAxis, zAxis);
    Vec3 xAxis; vec3Cross(up, zAxis, xAxis); vec3Normalize(xAxis, xAxis);
    Vec3 yAxis; vec3Cross(zAxis, xAxis, yAxis);
    m[0]=xAxis.x; m[1]=xAxis.y; m[2]=xAxis.z; m[3]=0;
    m[4]=yAxis.x; m[5]=yAxis.y; m[6]=yAxis.z; m[7]=0;
    m[8]=zAxis.x; m[9]=zAxis.y; m[10]=zAxis.z; m[11]=0;
    m[12]=-(xAxis.x*eye.x + xAxis.y*eye.y + xAxis.z*eye.z);
    m[13]=-(yAxis.x*eye.x + yAxis.y*eye.y + yAxis.z*eye.z);
    m[14]=-(zAxis.x*eye.x + zAxis.y*eye.y + zAxis.z*eye.z);
    m[15]=1;
}

static void mat4Mul(float *r, const float *a, const float *b) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            r[j*4+i] = 0;
            for (int k = 0; k < 4; ++k)
                r[j*4+i] += a[k*4+i] * b[j*4+k];
        }
}

// ======================== 随机数 (简易) ========================
static uint32_t rng_state = 12345;
static float rngFloat() {
    rng_state = rng_state * 1103515245 + 12345;
    return (float)((rng_state >> 16) & 0x7FFF) / 32767.0f;
}

// ======================== 顶点格式 ========================
struct Vertex { float x, y, z, r, g, b, a; };

// ======================== 三角面片数据 ========================
struct TriangleData {
    Vertex verts[3];
};

// ======================== 相机状态 ========================
static float cameraDistance = 12.0f;
static float cameraRotX = 0.0f;
static float cameraRotY = 0.0f;

// ======================== 三角面片管理 ========================
static TriangleData *gTriangles = nullptr;
static int gTriangleCount = 100;

static void generateTriangles(int count) {
    if (gTriangles) { free(gTriangles); gTriangles = nullptr; }
    if (count <= 0) { gTriangleCount = 0; return; }
    gTriangles = (TriangleData*)malloc(count * sizeof(TriangleData));
    gTriangleCount = count;
    for (int i = 0; i < count; i++) {
        float cx = (rngFloat() - 0.5f) * 10.0f;
        float cy = (rngFloat() - 0.5f) * 10.0f;
        float cz = (rngFloat() - 0.5f) * 10.0f;
        float sz = 0.1f + rngFloat() * 0.3f;
        for (int v = 0; v < 3; v++) {
            gTriangles[i].verts[v].x = cx + (rngFloat() - 0.5f) * sz;
            gTriangles[i].verts[v].y = cy + (rngFloat() - 0.5f) * sz;
            gTriangles[i].verts[v].z = cz + (rngFloat() - 0.5f) * sz;
            gTriangles[i].verts[v].r = rngFloat();
            gTriangles[i].verts[v].g = rngFloat();
            gTriangles[i].verts[v].b = rngFloat();
            gTriangles[i].verts[v].a = 1.0f;
        }
    }
}

// ======================== JS 交互 ========================
static void rebuildTriangleBuffers(); // 前向声明

// ======================== JS 交互 ========================
extern "C" {
EMSCRIPTEN_KEEPALIVE
int getTriangleCount() {
    return gTriangleCount;
}

EMSCRIPTEN_KEEPALIVE
void setTriangleCount(int count) {
    generateTriangles(count);
    rebuildTriangleBuffers();
}
}

// ======================== WebGPU 全局对象 ========================
static const wgpu::Instance instance = wgpuCreateInstance(nullptr);
static wgpu::Adapter   adapter;
static wgpu::Device    device;
static wgpu::Queue     queue;
static wgpu::Surface   surface;
static wgpu::RenderPipeline pipeline;
static wgpu::Buffer    uniformBuffer;
static wgpu::BindGroup bindGroup;
static wgpu::VertexBufferLayout bufferLayout;

// ======================== 每个三角面片的 vertex buffer ========================
static struct TriBuffer {
    wgpu::Buffer buf;
} *gTriBuffers = nullptr;
static int gTriBufferCount = 0;

static void rebuildTriangleBuffers() {
    // 释放旧 buffer
    for (int i = 0; i < gTriBufferCount; i++) {
        if (gTriBuffers[i].buf) gTriBuffers[i].buf.Destroy();
    }
    free(gTriBuffers);

    int count = gTriangleCount;
    gTriBuffers = (TriBuffer*)calloc(count, sizeof(TriBuffer));
    gTriBufferCount = count;

    for (int i = 0; i < count; i++) {
        wgpu::BufferDescriptor desc;
        desc.usage = wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst;
        desc.size = sizeof(gTriangles[i].verts);
        desc.mappedAtCreation = true;
        gTriBuffers[i].buf = device.CreateBuffer(&desc);
        void *mapped = gTriBuffers[i].buf.GetMappedRange();
        std::memcpy(mapped, gTriangles[i].verts, sizeof(gTriangles[i].verts));
        gTriBuffers[i].buf.Unmap();
    }
}

// ======================== Uniform 布局 ========================
struct Uniforms {
    float viewProj[16];
};

// ======================== WGSL 着色器 ========================
static const char shaderCode[] = R"(
    struct Uniforms {
        viewProj: mat4x4<f32>,
    };
    @group(0) @binding(0) var<uniform> u: Uniforms;

    struct VSOut {
        @builtin(position) position: vec4<f32>,
        @location(0)       color:    vec4<f32>,
    };

    @vertex
    fn vs(@location(0) pos: vec3<f32>,
          @location(1) col: vec4<f32>) -> VSOut {
        var o: VSOut;
        o.position = u.viewProj * vec4<f32>(pos, 1.0);
        o.color = col;
        return o;
    }

    @fragment
    fn fs(@location(0) col: vec4<f32>) -> @location(0) vec4<f32> {
        return col;
    }
)";

// ======================== 初始化管线 ========================
static void initPipeline() {
    // Uniform buffer
    wgpu::BufferDescriptor ubDesc;
    ubDesc.size = sizeof(Uniforms);
    ubDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    uniformBuffer = device.CreateBuffer(&ubDesc);

    // Shader module
    wgpu::ShaderModule shaderModule;
    {
        wgpu::ShaderSourceWGSL wgslDesc;
        wgslDesc.code = shaderCode;
        wgpu::ShaderModuleDescriptor smDesc;
        smDesc.nextInChain = &wgslDesc;
        shaderModule = device.CreateShaderModule(&smDesc);
    }

    // Vertex buffer layout
    wgpu::VertexAttribute attrs[2];
    attrs[0].format = wgpu::VertexFormat::Float32x3;
    attrs[0].offset = 0;
    attrs[0].shaderLocation = 0;
    attrs[1].format = wgpu::VertexFormat::Float32x4;
    attrs[1].offset = 3 * sizeof(float);
    attrs[1].shaderLocation = 1;

    bufferLayout.arrayStride = sizeof(Vertex);
    bufferLayout.attributeCount = 2;
    bufferLayout.attributes = attrs;

    // Bind group layout
    wgpu::BindGroupLayoutEntry entry{};
    entry.binding = 0;
    entry.visibility = wgpu::ShaderStage::Vertex;
    entry.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutDescriptor bglDesc;
    bglDesc.entryCount = 1;
    bglDesc.entries = &entry;
    auto bgl = device.CreateBindGroupLayout(&bglDesc);

    // Bind group
    wgpu::BindGroupEntry bgEntry{};
    bgEntry.binding = 0;
    bgEntry.buffer = uniformBuffer;
    bgEntry.size = sizeof(Uniforms);

    wgpu::BindGroupDescriptor bgDesc;
    bgDesc.layout = bgl;
    bgDesc.entryCount = 1;
    bgDesc.entries = &bgEntry;
    bindGroup = device.CreateBindGroup(&bgDesc);

    // Render pipeline
    wgpu::ColorTargetState colorTarget;
    colorTarget.format = wgpu::TextureFormat::BGRA8Unorm;

    wgpu::FragmentState fragState;
    fragState.module = shaderModule;
    fragState.targetCount = 1;
    fragState.targets = &colorTarget;

    wgpu::VertexState vertState;
    vertState.module = shaderModule;
    vertState.bufferCount = 1;
    vertState.buffers = &bufferLayout;

    wgpu::PipelineLayoutDescriptor plDesc;
    auto bglPtr = bgl;
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = &bglPtr;

    wgpu::RenderPipelineDescriptor rpDesc;
    rpDesc.layout = device.CreatePipelineLayout(&plDesc);
    rpDesc.vertex = vertState;
    rpDesc.fragment = &fragState;
    rpDesc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    rpDesc.primitive.cullMode = wgpu::CullMode::None; // 双面渲染

    pipeline = device.CreateRenderPipeline(&rpDesc);
    printf("[WebGPU] 管线创建完成\n");
}

// ======================== 渲染 ========================
static void renderFrame() {
    // 获取 canvas 尺寸
    int w = 0, h = 0;
    EM_ASM({
        var c = document.getElementById('canvas');
        if (c) { setValue($0, c.width, 'i32'); setValue($1, c.height, 'i32'); }
    }, &w, &h);
    if (w <= 0 || h <= 0) return;

    wgpu::SurfaceTexture surfaceTexture;
    surface.GetCurrentTexture(&surfaceTexture);
    if (!surfaceTexture.texture) return;

    wgpu::TextureView backbuffer = surfaceTexture.texture.CreateView();

    // 计算 view * proj
    float proj[16], view[16], vp[16];
    float aspect = (float)w / (float)h;
    mat4Perspective(proj, 3.14159265f / 4.0f, aspect, 0.1f, 1000.0f);

    float eyeX = cameraDistance * std::sin(cameraRotY) * std::cos(cameraRotX);
    float eyeY = cameraDistance * std::sin(cameraRotX);
    float eyeZ = cameraDistance * std::cos(cameraRotY) * std::cos(cameraRotX);
    Vec3 eye = {eyeX, eyeY, eyeZ};
    Vec3 center = {0, 0, 0};
    Vec3 up = {0, 1, 0};
    mat4LookAt(view, eye, center, up);
    mat4Mul(vp, proj, view);

    Uniforms uniforms;
    std::memcpy(uniforms.viewProj, vp, sizeof(vp));
    queue.WriteBuffer(uniformBuffer, 0, &uniforms, sizeof(uniforms));

    // Render pass
    wgpu::RenderPassColorAttachment colorAtt;
    colorAtt.view = backbuffer;
    colorAtt.loadOp = wgpu::LoadOp::Clear;
    colorAtt.storeOp = wgpu::StoreOp::Store;
    colorAtt.clearValue = {0, 0, 0, 1};

    wgpu::RenderPassDescriptor renderPass;
    renderPass.colorAttachmentCount = 1;
    renderPass.colorAttachments = &colorAtt;

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    {
        wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPass);
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bindGroup);

        // 每个三角面片独立 drawCall
        for (int i = 0; i < gTriBufferCount; i++) {
            if (gTriBuffers[i].buf) {
                pass.SetVertexBuffer(0, gTriBuffers[i].buf);
                pass.Draw(3);
            }
        }
        pass.End();
    }
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);
}

// ======================== FPS ========================
static double gFrameCount = 0;
static double gLastFpsTime = 0;
static int gFps = 0;

// ======================== 主循环 ========================
static void mainLoop() {
    gFrameCount++;
    double now = emscripten_get_now();
    if (now - gLastFpsTime >= 1000.0) {
        gFps = (int)(gFrameCount * 1000.0 / (now - gLastFpsTime));
        gFrameCount = 0;
        gLastFpsTime = now;

        // 更新 HTML 中的 FPS 显示
        EM_ASM({
            var el = document.getElementById('fps-display');
            if (el) el.textContent = 'FPS: ' + $0 + ' | 三角面片: ' + $1;
        }, gFps, gTriangleCount);
    }

    renderFrame();
}

// ======================== 鼠标事件 ========================
static bool isDragging = false;
static float lastMouseX = 0, lastMouseY = 0;

static EM_BOOL onMouseDown(int eventType, const EmscriptenMouseEvent *e, void *userData) {
    isDragging = true;
    lastMouseX = (float)e->clientX;
    lastMouseY = (float)e->clientY;
    return EM_TRUE;
}

static EM_BOOL onMouseMove(int eventType, const EmscriptenMouseEvent *e, void *userData) {
    if (!isDragging) return EM_FALSE;
    float dx = (float)e->clientX - lastMouseX;
    float dy = (float)e->clientY - lastMouseY;
    cameraRotY += dx * 0.01f;
    cameraRotX += dy * 0.01f;
    cameraRotX = std::max(-1.5f, std::min(1.5f, cameraRotX));
    lastMouseX = (float)e->clientX;
    lastMouseY = (float)e->clientY;
    return EM_TRUE;
}

static EM_BOOL onMouseUp(int eventType, const EmscriptenMouseEvent *e, void *userData) {
    isDragging = false;
    return EM_TRUE;
}

static EM_BOOL onWheel(int eventType, const EmscriptenWheelEvent *e, void *userData) {
    cameraDistance += e->deltaY * 0.01f;
    cameraDistance = std::max(1.0f, std::min(100.0f, cameraDistance));
    return EM_TRUE;
}

// ======================== resize ========================
static EM_BOOL onResize(int eventType, const EmscriptenUiEvent *e, void *userData) {
    EM_ASM({
        var c = document.getElementById('canvas');
        if (c) { c.width = window.innerWidth; c.height = window.innerHeight; }
    });
    return EM_TRUE;
}

// ======================== Surface/Device/Adapter ========================
static void setupSurface() {
    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector canvasDesc;
    canvasDesc.selector = "#canvas";

    wgpu::SurfaceDescriptor surfDesc;
    surfDesc.nextInChain = &canvasDesc;
    surface = instance.CreateSurface(&surfDesc);

    wgpu::SurfaceCapabilities caps;
    surface.GetCapabilities(adapter, &caps);

    wgpu::SurfaceConfiguration config;
    config.device = device;
    config.format = caps.formats[0];
    config.usage = wgpu::TextureUsage::RenderAttachment;
    config.width = 800;
    config.height = 600;
    config.alphaMode = wgpu::CompositeAlphaMode::Auto;
    config.presentMode = wgpu::PresentMode::Fifo;
    surface.Configure(&config);
}

static void onRequestDevice(wgpu::RequestDeviceStatus status,
                            wgpu::Device dev, wgpu::StringView msg) {
    assert(status == wgpu::RequestDeviceStatus::Success);
    if (status != wgpu::RequestDeviceStatus::Success) return;
    device = dev;
    queue = device.GetQueue();

    setupSurface();
    generateTriangles(gTriangleCount);
    initPipeline();
    rebuildTriangleBuffers();

    printf("[WebGPU] 资源初始化完成，开始渲染\n");
    emscripten_set_main_loop(mainLoop, 0, true);
}

static void onRequestAdapter(wgpu::RequestAdapterStatus status,
                             wgpu::Adapter adap, wgpu::StringView msg) {
    assert(status == wgpu::RequestAdapterStatus::Success);
    if (status != wgpu::RequestAdapterStatus::Success) return;
    adapter = adap;

    wgpu::DeviceDescriptor devDesc;
    devDesc.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType type, wgpu::StringView msg) {
            printf("[WebGPU Error] type=%d: %.*s\n", (int)type,
                   (int)msg.length, msg.data);
        });

    adapter.RequestDevice(&devDesc, wgpu::CallbackMode::AllowSpontaneous,
                          onRequestDevice);
}

// ======================== main ========================
int main() {
    printf("[WASM WebGPU] 启动...\n");

    // 事件监听
    emscripten_set_mousedown_callback("#canvas", nullptr, true, onMouseDown);
    emscripten_set_mousemove_callback("#canvas", nullptr, true, onMouseMove);
    emscripten_set_mouseup_callback("#canvas", nullptr, true, onMouseUp);
    emscripten_set_wheel_callback("#canvas", nullptr, true, onWheel);
    emscripten_set_resize_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, nullptr, true, onResize);

    // 初始 canvas 尺寸
    EM_ASM({
        var c = document.getElementById('canvas');
        if (c) { c.width = window.innerWidth; c.height = window.innerHeight; }
    });

    instance.RequestAdapter(nullptr, wgpu::CallbackMode::AllowSpontaneous,
                            onRequestAdapter);
    return 0;
}
