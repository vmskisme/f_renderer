use glam::{vec3, vec4, IVec2, UVec2, Vec2, Vec3, Vec4};
use std::cmp::{max, min};
use std::fs;
use std::{collections::HashMap, task::Context};

#[inline]
pub fn vec4_to_u8_array(v: Vec4) -> [u8; 4] {
    [
        (v.x * 255.0).clamp(0.0, 255.0) as u8,
        (v.y * 255.0).clamp(0.0, 255.0) as u8,
        (v.z * 255.0).clamp(0.0, 255.0) as u8,
        (v.w * 255.0).clamp(0.0, 255.0) as u8,
    ]
}

#[inline]
pub fn u8_array_to_vec4(v: [u8; 4]) -> Vec4 {
    Vec4::new(
        (v[0] as f32) / 255.0,
        (v[1] as f32) / 255.0,
        (v[2] as f32) / 255.0,
        (v[3] as f32) / 255.0,
    )
}

#[inline]
pub fn u8_array_mul_f32(v: [u8; 4], a: f32) -> [u8; 4] {
    [
        ((v[0] as f32) * a) as u8,
        ((v[1] as f32) * a) as u8,
        ((v[2] as f32) * a) as u8,
        ((v[3] as f32) * a) as u8,
    ]
}

#[inline]
fn is_top_left(a: IVec2, b: IVec2) -> bool {
    ((a.y == b.y) && (a.x < b.x)) || (a.y > b.y)
}

pub struct Renderer<VSInput, PSInput, VSUniform, PSUniform> {
    w: usize,
    h: usize,
    vertex_shader: fn(&VSUniform, &VSInput, &mut ShaderContext) -> Vec4,
    vs_uniform: VSUniform,
    pixel_shader: fn(&PSUniform, &PSInput, &ShaderContext) -> Vec4,
    ps_uniform: PSUniform,
}

// application -> geometry processing -> rasterization -> pixel processing
// 看一下realtime rendering的第二章

impl<VSInput, PSInput, VSUniform, PSUniform> Renderer<VSInput, PSInput, VSUniform, PSUniform> {
    pub fn geometry_processing(&self, vs_inputs: &[VSInput; 3]) -> Option<Vec<[Vertex; 3]>> {
        let mut vertices = vec![];

        for i in 0..3 {
            let mut vertex = Vertex::new();

            vertex.pos = (self.vertex_shader)(&self.vs_uniform, &vs_inputs[i], &mut vertex.context);

            vertices.push(vertex);
        }

        enum Plane {
            X_LEFT,
            X_RIGHT,
            Y_UP,
            Y_DOWN,
            Z_NEAR,
            Z_FAR,
            W_PLANE,
        }

        const EPSILON: f32 = 1.0e-5;

        #[inline]
        fn insides(plane: &Plane, vertex: &Vertex) -> bool {
            let w = vertex.pos.w;
            match plane {
                Plane::X_LEFT => vertex.pos.x >= -w,
                Plane::X_RIGHT => vertex.pos.x <= w,
                Plane::Y_UP => vertex.pos.y <= w,
                Plane::Y_DOWN => vertex.pos.y >= -w,
                Plane::Z_FAR => vertex.pos.z <= vertex.pos.w,
                Plane::Z_NEAR => vertex.pos.z >= 0.0, // todo <= -w
                Plane::W_PLANE => vertex.pos.w <= EPSILON, // todo 现在还有问题
            }
        }

        #[inline]
        fn calculate_intersect_ratio(plane: &Plane, a: &Vertex, b: &Vertex) -> f32 {
            let a_w = a.pos.w;
            let b_w = b.pos.w;
            match plane {
                Plane::X_LEFT => (a.pos.x + a_w) / (a_w + a.pos.x - b_w - b.pos.x),
                Plane::X_RIGHT => (a_w - a.pos.x) / (a_w - b_w - a.pos.x + b.pos.x),
                Plane::Y_UP => (a_w - a.pos.y) / (a_w - b_w - a.pos.y + b.pos.y),
                Plane::Y_DOWN => (a.pos.y + a_w) / (a_w + a.pos.y - b_w - b.pos.y),
                Plane::Z_FAR => (a.pos.z + a_w) / (a_w + a.pos.z - b_w - b.pos.z),
                Plane::Z_NEAR => a_w / (a_w - b_w), // ...todo
                Plane::W_PLANE => (EPSILON - a_w) / (b_w - a_w), // todo
            }
        }

        #[inline]
        fn vertex_intersect(a: &Vertex, b: &Vertex, ratio: f32) -> Vertex {
            let mut new_vertex = Vertex::new();
            new_vertex.pos = a.pos + ratio * (b.pos - a.pos);

            for item in a.context.varying_float.iter() {
                let a_vary = item.1;
                let b_vary = b.context.varying_float[item.0];
                new_vertex
                    .context
                    .varying_float
                    .insert(*item.0, a_vary + ratio * (b_vary - a_vary));
            }

            for item in a.context.varying_vec2.iter() {
                let a_c = *item.1;
                let b_c = b.context.varying_vec2[item.0];
                new_vertex
                    .context
                    .varying_vec2
                    .insert(*item.0, a_c + ratio * (b_c - a_c));
            }

            for item in a.context.varying_vec3.iter() {
                let a_c = *item.1;
                let b_c = b.context.varying_vec3[item.0];
                new_vertex
                    .context
                    .varying_vec3
                    .insert(*item.0, a_c + ratio * (b_c - a_c));
            }

            for item in a.context.varying_vec4.iter() {
                let a_c = *item.1;
                let b_c = b.context.varying_vec4[item.0];
                new_vertex
                    .context
                    .varying_vec4
                    .insert(*item.0, a_c + ratio * (b_c - a_c));
            }

            new_vertex
        }

        let mut valid_vertices = vec![];

        let mut valid_vertices_index = vec![];
        for i in 0..3 {
            let mut all_inside = true;
            let a = &vertices[i];
            for j in (i + 1)..3 {
                // todo 这里有冗余计算，因为外层循环一轮后inside已经对所有点完成了计算
                let b = &vertices[j];

                for plane in [
                    Plane::X_LEFT,
                    Plane::X_RIGHT,
                    Plane::Y_UP,
                    Plane::Y_DOWN,
                    Plane::Z_NEAR,
                    Plane::Z_FAR,
                    // Plane::W_PLANE, // todo
                ] {
                    let a_inside = insides(&plane, &a);
                    let b_inside = insides(&plane, &b);

                    if a_inside != b_inside {
                        let ratio = calculate_intersect_ratio(&plane, &a, &b);
                        let new_vertex = vertex_intersect(&a, &b, ratio);
                        if new_vertex.pos.w.abs() > EPSILON {
                            valid_vertices.push(new_vertex); // 这里的点有可能还是越界的
                        }
                    }

                    all_inside = all_inside && a_inside;
                }
            }

            if all_inside {
                valid_vertices_index.push(i);
            }
        }

        for v in valid_vertices_index {
            let vertex = &vertices[v];
            if vertex.pos.w == 0.0 {
                continue;
            }
            valid_vertices.push(vertex.clone()); // todo try to remove clone
        }

        if valid_vertices.len() < 3 {
            return None;
        }

        // todo 处理vertices的顺序

        let mut triangles = vec![];

        for i in 0..valid_vertices.len() {
            let vertex = &mut valid_vertices[i];
            let w = vertex.pos.w;
            vertex.rhw = 1.0 / w;

            // 齐次坐标空间 /w 归一化到单位体积 cvv
            vertex.pos = vertex.pos * vertex.rhw;

            // 计算屏幕坐标
            vertex.spf.x = (vertex.pos.x + 1.0) * (self.w as f32) * 0.5;
            vertex.spf.y = (1.0 - vertex.pos.y) * (self.h as f32) * 0.5;

            vertex.spi.x = (vertex.spf.x + 0.5) as i32;
            vertex.spi.y = (vertex.spf.y + 0.5) as i32;
        }

        for i in 0..valid_vertices.len() - 2 {
            let a = valid_vertices[i].clone();
            let b = valid_vertices[i + 1].clone();
            let c = valid_vertices[i + 2].clone();
            triangles.push([a, b, c]);
        }

        Some(triangles)
    }

    fn rasterization() {
        todo!()
    }
}

#[derive(Clone)]
pub struct Vertex {
    context: ShaderContext,
    rhw: f32,   // w倒数
    pos: Vec4,  // 坐标
    spf: Vec2,  // 浮点数屏幕坐标
    spi: IVec2, // 整数屏幕坐标
}

impl Vertex {
    pub fn new() -> Self {
        Self {
            context: ShaderContext::new(),
            rhw: 0.0,
            pos: Vec4::ZERO,
            spf: Vec2::ZERO,
            spi: IVec2::ZERO,
        }
    }
}

#[derive(Clone)]
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

pub struct FrameBuffer {
    pub width: u32,
    pub height: u32,
    bits: Vec<u8>,
}

impl FrameBuffer {
    pub fn new(width: u32, height: u32) -> FrameBuffer {
        return FrameBuffer {
            width: width,
            height: height,
            bits: vec![0; (width * height * 4) as usize],
        };
    }

    pub fn fill(&mut self, color: [u8; 4]) {
        for i in 0..self.height {
            for j in 0..self.width {
                self.set_pixel(j, i, color);
            }
        }
    }

    #[inline]
    pub fn set_pixel(&mut self, x: u32, y: u32, color: [u8; 4]) {
        let offset = (y * self.width * 4 + x * 4) as usize;
        for i in 0..4 {
            self.bits[offset + i] = color[i];
        }
    }

    #[inline]
    fn get_pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let offset = (y * self.width * 4 + x * 4) as usize;
        [
            self.bits[offset],
            self.bits[offset + 1],
            self.bits[offset + 2],
            self.bits[offset + 3],
        ]
    }

    pub fn sample_2d(&self, uv: Vec2) -> Vec4 {
        let x = uv.x * (self.width as f32);
        let y = uv.y * (self.height as f32);
        let a = x.fract();
        let b = y.fract();

        let x1 = (x as u32).clamp(0, self.width - 1);
        let y1 = (y as u32).clamp(0, self.width - 1);
        let x2 = (x1 + 1).clamp(0, self.width - 1);
        let y2 = (y1 + 1).clamp(0, self.width - 1);

        let c11 = u8_array_to_vec4(self.get_pixel(x1, y1)) * (1.0 - a) * (1.0 - b);
        let c12 = u8_array_to_vec4(self.get_pixel(x1, y2)) * (1.0 - a) * b;
        let c21 = u8_array_to_vec4(self.get_pixel(x2, y1)) * a * (1.0 - b);
        let c22 = u8_array_to_vec4(self.get_pixel(x2, y2)) * a * b;

        vec4(
            c11.x + c12.x + c21.x + c22.x,
            c11.y + c12.y + c21.y + c22.y,
            c11.z + c12.z + c21.z + c22.z,
            c11.w + c12.w + c21.w + c22.w,
        )
    }

    pub fn draw_line(&mut self, x1: u32, y1: u32, x2: u32, y2: u32, color: [u8; 4]) {
        let (x1, x2) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
        let (y1, y2) = if y1 < y2 { (y1, y2) } else { (y2, y1) };
        match (x1 == x2, y1 == y2) {
            (true, true) => {
                self.set_pixel(x1, y1, color);
            }
            (true, false) => {
                for y in y1..y2 {
                    self.set_pixel(x1, y, color);
                }
            }
            (false, true) => {
                for x in x1..x2 {
                    self.set_pixel(x, y1, color);
                }
            }
            (false, false) => {
                let dx = x2 - x1;
                let dy = y2 - y1;
                let mut rem = 0;
                if dx > dy {
                    let mut y = y1;
                    for x in x1..x2 {
                        self.set_pixel(x, y, color);
                        rem += dy;
                        if rem >= dx {
                            y += 1;
                            rem -= dx;
                            self.set_pixel(x, y, color);
                        }
                    }
                    self.set_pixel(x2, y2, color);
                } else {
                    let mut x = x1;
                    for y in y1..y2 {
                        self.set_pixel(x, y, color);
                        rem += dx;
                        if rem >= dy {
                            x += 1;
                            rem -= dy;
                            self.set_pixel(x, y, color);
                        }
                    }
                    self.set_pixel(x2, y2, color);
                }
            }
        }
    }
}
