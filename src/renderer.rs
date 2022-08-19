use glam::{vec3, vec4, UVec2, Vec2, Vec3, Vec4};
use std::collections::HashMap;

pub struct FrameBuffer<VSInput, PSInput> {
    w: usize,
    h: usize,
    vertex_shader: fn(&VSInput, &mut ShaderContext) -> Vec4,
    // vs_uniform:todo!(),
    pixel_shader: fn(&PSInput, &ShaderContext) -> Vec4,
    // ps_uniform:todo!(),
}

// application -> geometry processing -> rasterization -> pixel processing
// 看一下realtime rendering的第二章

impl<VSInput, PSInput> FrameBuffer<VSInput, PSInput> {

    // TODO 要区分vary和uniform，不要把他们都放到VSInput里
    pub fn compute_triangle(&self, vs_inputs: &[VSInput; 3]) -> Option<[Vertex; 3]> {
        let mut triangle: [Vertex; 3] = [Vertex::new(), Vertex::new(), Vertex::new()];

        for i in 0..3 {
            let mut vertex = &mut triangle[i];

            vertex.pos = (self.vertex_shader)(&vs_inputs[i], &mut vertex.context);

            let w = vertex.pos.w;
            if w == 0.0 {
                return None;
            }
            if vertex.pos.z <= 0.0 || vertex.pos.z >= w{
                return None;
            }
            
        }

        None
    }
}

pub struct Vertex {
    context: ShaderContext,
    rhw: f32,   // w倒数
    pos: Vec4,  // 坐标
    spf: Vec2,  // 浮点数屏幕坐标
    spi: UVec2, // 整数屏幕坐标
}

impl Vertex {
    pub fn new() -> Self {
        Self {
            context: ShaderContext::new(),
            rhw: 0.0,
            pos: Vec4::ZERO,
            spf: Vec2::ZERO,
            spi: UVec2::ZERO,
        }
    }
}

pub struct ShaderContext {
    pub varying_float: HashMap<u32, f32>, // 浮点数 varying 列表
    pub varying_vec2: HashMap<u32, Vec2>, // 二维矢量 varying 列表
    pub varying_vec3: HashMap<u32, Vec3>, // 三维矢量 varying 列表
    pub varying_vec4: HashMap<u32, Vec4>, // 四维矢量 varying 列表
}

impl ShaderContext {
    fn new() -> ShaderContext {
        ShaderContext {
            varying_float: HashMap::new(),
            varying_vec2: HashMap::new(),
            varying_vec3: HashMap::new(),
            varying_vec4: HashMap::new(),
        }
    }
}
