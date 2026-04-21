#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <emscripten.h>
#include <emscripten/html5.h>
#include <GLES3/gl3.h>

// ==================== 数学工具 ====================

static void compute_perspective(float *m, float fov, float aspect, float near, float far) {
    memset(m, 0, 16 * sizeof(float));
    float f = 1.0f / tanf(fov / 2.0f);
    float nf = 1.0f / (near - far);
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (far + near) * nf;
    m[11] = -1.0f;
    m[14] = 2.0f * far * near * nf;
}

static void compute_lookat(float *m,
    float eye_x, float eye_y, float eye_z,
    float center_x, float center_y, float center_z,
    float up_x, float up_y, float up_z) {
    
    float zx = eye_x - center_x;
    float zy = eye_y - center_y;
    float zz = eye_z - center_z;
    float len = 1.0f / sqrtf(zx*zx + zy*zy + zz*zz);
    float za[3] = {zx*len, zy*len, zz*len};
    
    float xx = up_y * za[2] - up_z * za[1];
    float xy = up_z * za[0] - up_x * za[2];
    float xz = up_x * za[1] - up_y * za[0];
    len = 1.0f / sqrtf(xx*xx + xy*xy + xz*xz);
    float xa[3] = {xx*len, xy*len, xz*len};
    
    float yx = za[1]*xa[2] - za[2]*xa[1];
    float yy = za[2]*xa[0] - za[0]*xa[2];
    float yz = za[0]*xa[1] - za[1]*xa[0];
    
    m[0]=xa[0]; m[1]=yx;  m[2]=za[0]; m[3]=0;
    m[4]=xa[1]; m[5]=yy;  m[6]=za[1]; m[7]=0;
    m[8]=xa[2]; m[9]=yz;  m[10]=za[2]; m[11]=0;
    m[12]=-(xa[0]*eye_x + xa[1]*eye_y + xa[2]*eye_z);
    m[13]=-(yx*eye_x + yy*eye_y + yz*eye_z);
    m[14]=-(za[0]*eye_x + za[1]*eye_y + za[2]*eye_z);
    m[15]=1;
}

// ==================== 全局状态 ====================

static double canvas_width = 800;
static double canvas_height = 600;
static float camera_distance = 5.0f;
static float camera_rot_x = 0.0f;
static float camera_rot_y = 0.0f;

// FPS
static int frame_count = 0;
static double last_time = 0.0;
static int current_fps = 0;

// 三角形数据（C++侧维护）
static int triangle_count = 0;
static GLuint *pos_buffers = NULL;  // 每个三角形一个position buffer
static GLuint *col_buffers = NULL;  // 每个三角形一个color buffer

// Shader program
static GLuint program = 0;
static GLuint a_position = 0;
static GLuint a_color = 0;
static GLint u_projection = -1;
static GLint u_view = -1;

static float rand_float() {
    return (float)rand() / (float)RAND_MAX;
}

// ==================== 三角形管理 ====================

static void create_triangle_buffer(GLuint buf, float *data, int count) {
    glBindBuffer(GL_ARRAY_BUFFER, buf);
    glBufferData(GL_ARRAY_BUFFER, count * sizeof(float), data, GL_STATIC_DRAW);
}

static void rebuild_all_triangles(int count) {
    // 删除旧缓冲
    for (int i = 0; i < triangle_count; i++) {
        glDeleteBuffers(1, &pos_buffers[i]);
        glDeleteBuffers(1, &col_buffers[i]);
    }
    if (pos_buffers) free(pos_buffers);
    if (col_buffers) free(col_buffers);
    
    triangle_count = count;
    pos_buffers = (GLuint*)malloc(count * sizeof(GLuint));
    col_buffers = (GLuint*)malloc(count * sizeof(GLuint));
    
    int floats_per_triangle = 9; // 3 vertices * 3 floats
    
    for (int i = 0; i < count; i++) {
        float cx = (rand_float() - 0.5f) * 10.0f;
        float cy = (rand_float() - 0.5f) * 10.0f;
        float cz = (rand_float() - 0.5f) * 10.0f;
        float size = 0.1f + rand_float() * 0.3f;
        
        float positions[9];
        float colors[9];
        
        for (int v = 0; v < 3; v++) {
            int idx = v * 3;
            positions[idx]   = cx + (rand_float() - 0.5f) * size;
            positions[idx+1] = cy + (rand_float() - 0.5f) * size;
            positions[idx+2] = cz + (rand_float() - 0.5f) * size;
            
            colors[idx]   = rand_float();
            colors[idx+1] = rand_float();
            colors[idx+2] = rand_float();
        }
        
        glGenBuffers(1, &pos_buffers[i]);
        create_triangle_buffer(pos_buffers[i], positions, 9);
        
        glGenBuffers(1, &col_buffers[i]);
        create_triangle_buffer(col_buffers[i], colors, 9);
    }
}

// ==================== Shader编译 ====================

static GLuint compile_shader(GLenum type, const char *source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    
    GLint status = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        char log[512];
        glGetShaderInfoLog(shader, 512, NULL, log);
        emscripten_console_log("Shader compile error:");
        emscripten_console_log(log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static void init_shaders() {
    const char *vs_source = 
        "attribute vec3 aPosition;"
        "attribute vec3 aColor;"
        "uniform mat4 uProjection;"
        "uniform mat4 uView;"
        "varying vec3 vColor;"
        "void main() {"
        "  gl_Position = uProjection * uView * vec4(aPosition, 1.0);"
        "  vColor = aColor;"
        "}";
    
    const char *fs_source = 
        "precision mediump float;"
        "varying vec3 vColor;"
        "void main() {"
        "  gl_FragColor = vec4(vColor, 1.0);"
        "}";
    
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_source);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_source);
    
    program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    
    GLint status = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (!status) {
        char log[512];
        glGetProgramInfoLog(program, 512, NULL, log);
        emscripten_console_log("Program link error:");
        emscripten_console_log(log);
    }
    
    glUseProgram(program);
    
    a_position = glGetAttribLocation(program, "aPosition");
    a_color    = glGetAttribLocation(program, "aColor");
    u_projection = glGetUniformLocation(program, "uProjection");
    u_view       = glGetUniformLocation(program, "uView");
}

// ==================== Canvas / WebGL 上下文 ====================

static EMSCRIPTEN_WEBGL_CONTEXT_HANDLE gl_ctx = 0;

static void init_webgl() {
    EmscriptenWebGLContextAttributes attrs;
    emscripten_webgl_init_context_attributes(&attrs);
    attrs.antialias = 1;
    attrs.alpha = 0;
    attrs.majorVersion = 2;
    attrs.minorVersion = 0;
    attrs.explicitSwapControl = 0;
    
    gl_ctx = emscripten_webgl_create_context("#gl-canvas", &attrs);
    emscripten_webgl_make_context_current(gl_ctx);
}

// ==================== 交互回调 ====================

static EM_BOOL on_mouse_move(int event_type, const EmscriptenMouseEvent *e, void *userdata) {
    static int prev_x = 0, prev_y = 0;
    static int is_down = 0;
    
    if (event_type == EMSCRIPTEN_EVENT_MOUSEDOWN) {
        is_down = 1;
        prev_x = e->clientX;
        prev_y = e->clientY;
        return 1;
    }
    if (event_type == EMSCRIPTEN_EVENT_MOUSEUP) {
        is_down = 0;
        return 1;
    }
    if (event_type == EMSCRIPTEN_EVENT_MOUSEMOVE) {
        if (is_down) {
            float dx = (float)(e->clientX - prev_x);
            float dy = (float)(e->clientY - prev_y);
            camera_rot_y += dx * 0.01f;
            camera_rot_x += dy * 0.01f;
            prev_x = e->clientX;
            prev_y = e->clientY;
        }
        return 1;
    }
    return 0;
}

static EM_BOOL on_wheel(int event_type, const EmscriptenWheelEvent *e, void *userdata) {
    camera_distance += (float)(e->deltaY * 0.01);
    if (camera_distance < 1.0f) camera_distance = 1.0f;
    if (camera_distance > 20.0f) camera_distance = 20.0f;
    return 1;
}

static EM_BOOL on_resize(int event_type, const EmscriptenUiEvent *e, void *userdata) {
    emscripten_get_element_css_size("#gl-canvas", &canvas_width, &canvas_height);
    glViewport(0, 0, (int)canvas_width, (int)canvas_height);
    return 1;
}

// ==================== 导出给JS调用的函数 ====================

#ifdef __cplusplus
extern "C" {
#endif

EMSCRIPTEN_KEEPALIVE
void set_canvas_size(int width, int height) {
    canvas_width = width;
    canvas_height = height;
}

EMSCRIPTEN_KEEPALIVE
void update_triangle_count(int count) {
    if (count > 0) {
        rebuild_all_triangles(count);
    }
}

EMSCRIPTEN_KEEPALIVE
void update_camera_rotation(float dx, float dy) {
    camera_rot_y += dx * 0.01f;
    camera_rot_x += dy * 0.01f;
}

EMSCRIPTEN_KEEPALIVE
int get_current_fps() {
    return current_fps;
}

EMSCRIPTEN_KEEPALIVE
void update_camera_zoom(float delta) {
    camera_distance += delta * 0.01f;
    if (camera_distance < 1.0f) camera_distance = 1.0f;
    if (camera_distance > 20.0f) camera_distance = 20.0f;
}

#ifdef __cplusplus
}
#endif

// ==================== 渲染循环 ====================

static void main_loop() {
    // FPS
    double now = emscripten_get_now();
    frame_count++;
    if (now - last_time >= 1000.0) {
        current_fps = (int)((frame_count * 1000.0) / (now - last_time));
        frame_count = 0;
        last_time = now;
    }
    
    // 清屏
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    // 不启用CULL_FACE，双面渲染
    
    // 计算矩阵
    float projection[16];
    float view[16];
    float aspect = (float)canvas_width / (float)canvas_height;
    compute_perspective(projection, 3.14159265f / 4.0f, aspect, 0.1f, 1000.0f);
    
    float eye_x = camera_distance * sinf(camera_rot_y) * cosf(camera_rot_x);
    float eye_y = camera_distance * sinf(camera_rot_x);
    float eye_z = camera_distance * cosf(camera_rot_y) * cosf(camera_rot_x);
    compute_lookat(view, eye_x, eye_y, eye_z, 0, 0, 0, 0, 1, 0);
    
    // 设置uniform
    glUniformMatrix4fv(u_projection, 1, GL_FALSE, projection);
    glUniformMatrix4fv(u_view, 1, GL_FALSE, view);
    
    // 每个三角形独立drawCall
    for (int i = 0; i < triangle_count; i++) {
        glBindBuffer(GL_ARRAY_BUFFER, pos_buffers[i]);
        glEnableVertexAttribArray(a_position);
        glVertexAttribPointer(a_position, 3, GL_FLOAT, GL_FALSE, 0, 0);
        
        glBindBuffer(GL_ARRAY_BUFFER, col_buffers[i]);
        glEnableVertexAttribArray(a_color);
        glVertexAttribPointer(a_color, 3, GL_FLOAT, GL_FALSE, 0, 0);
        
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    
    // 更新JS侧FPS显示
    emscripten_run_script("if(window._updateFpsDisplay) window._updateFpsDisplay();");
}

// ==================== 入口 ====================

int main() {
    srand((unsigned int)time(NULL));
    last_time = emscripten_get_now();
    
    // 初始化WebGL
    init_webgl();
    
    // 初始canvas尺寸
    emscripten_get_element_css_size("#gl-canvas", &canvas_width, &canvas_height);
    glViewport(0, 0, (int)canvas_width, (int)canvas_height);
    
    // 初始化Shader
    init_shaders();
    
    // 初始三角形
    rebuild_all_triangles(100);
    
    // 注册事件回调
    emscripten_set_mousemove_callback("#gl-canvas", NULL, 1, on_mouse_move);
    emscripten_set_mousedown_callback("#gl-canvas", NULL, 1, on_mouse_move);
    emscripten_set_mouseup_callback("#gl-canvas", NULL, 1, on_mouse_move);
    emscripten_set_wheel_callback("#gl-canvas", NULL, 1, on_wheel);
    emscripten_set_resize_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, NULL, 0, on_resize);
    
    // 开始渲染循环
    emscripten_set_main_loop(main_loop, 0, 1);
    
    return 0;
}
