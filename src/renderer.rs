use glam::{vec3, vec4, UVec2,IVec2, Vec2, Vec3, Vec4};
use std::{collections::HashMap, task::Context};
use std::cmp::{min, max};
use std::{fs};



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

pub fn vec3_to_vec4(v: Vec3, w: f32) -> Vec4 {
    Vec4::new(v.x, v.y, v.z, w)
}

fn is_top_left(a: IVec2, b: IVec2) -> bool {
    ((a.y == b.y) && (a.x < b.x)) || (a.y > b.y)
}


pub struct Renderer<VSInput, PSInput> {
    w: usize,
    h: usize,
    vertex_shader: fn(&VSInput, &mut ShaderContext) -> Vec4,
    // vs_uniform:todo!(),
    pixel_shader: fn(&PSInput, &ShaderContext) -> Vec4,
    // ps_uniform:todo!(),
}

// application -> geometry processing -> rasterization -> pixel processing
// 看一下realtime rendering的第二章

impl<VSInput, PSInput> Renderer<VSInput, PSInput> {
    // TODO 要区分vary和uniform，不要把他们都放到VSInput里

    pub fn compute_triangle(&self, vs_inputs: &[VSInput; 3]) -> Option<Vec<[Vertex; 3]>> {
        let mut vertices = vec![];
        for i in 0..3 {
            let mut vertex = Vertex::new();

            vertex.pos = (self.vertex_shader)(&vs_inputs[i], &mut vertex.context);

            let w = vertex.pos.w;
            if w == 0.0 {
                return None;
            }

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
                Plane::Z_NEAR => vertex.pos.z >= 0.0, // <= -w
                Plane::W_PLANE => vertex.pos.w <= EPSILON, // 现在还有问题
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

        fn vertex_intersect(a: &Vertex, b: &Vertex, ratio: f32) -> Vertex {
            let mut new_v = Vertex::new();
            new_v.pos = a.pos + ratio * (b.pos - a.pos);

            for item in a.context.varying_float.iter() {
                let a_c = item.1;
                let b_c = b.context.varying_float.get(item.0).unwrap();
                new_v
                    .context
                    .varying_float
                    .insert(*item.0, a_c + ratio * (b_c - a_c));
            }

            for item in a.context.varying_vec2.iter() {
                let a_c = *item.1;
                let b_c = *b.context.varying_vec2.get(item.0).unwrap();
                new_v
                    .context
                    .varying_vec2
                    .insert(*item.0, a_c + ratio * (b_c - a_c));
            }

            for item in a.context.varying_vec3.iter() {
                let a_c = *item.1;
                let b_c = *b.context.varying_vec3.get(item.0).unwrap();
                new_v
                    .context
                    .varying_vec3
                    .insert(*item.0, a_c + ratio * (b_c - a_c));
            }

            for item in a.context.varying_vec4.iter() {
                let a_c = *item.1;
                let b_c = *b.context.varying_vec4.get(item.0).unwrap();
                new_v
                    .context
                    .varying_vec4
                    .insert(*item.0, a_c + ratio * (b_c - a_c));
            }

            new_v
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
                ] {
                    let a_inside = insides(&plane, &a);
                    let b_inside = insides(&plane, &b);

                    if a_inside != b_inside {
                        let ratio = calculate_intersect_ratio(&plane, &a, &b);
                        let new_vertex = vertex_intersect(&a, &b, ratio);
                        if new_vertex.pos.w.abs() > EPSILON {
                            valid_vertices.push(new_vertex);
                        }
                    }

                    all_inside = all_inside && a_inside;
                }
            }

            if all_inside {
                valid_vertices_index.push(i); //todo remove clone
            }
        }

        for v in valid_vertices_index {
            let old_v = vertices[v].clone(); //todo 
            valid_vertices.push(old_v);
        }

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

        Some(triangles)  // TODO 比不做裁剪多了非常多三角形
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
    pub bits: Vec<Vec<Vec4>>,
}

impl FrameBuffer {
    pub fn new(width: u32, height: u32) -> FrameBuffer {
        return FrameBuffer {
            width: width,
            height: height,
            bits: vec![vec![Vec4::ZERO; width as usize]; height as usize],
        };
    }

    pub fn fill(&mut self, color: Vec4) {
        for i in 0..self.height as usize {
            for j in 0..self.width as usize {
                self.bits[i][j] = color;
            }
        }
    }

    pub fn load_file(path: &str) -> Self {
        let mut buf = fs::read(path).unwrap();
        if buf[0] != 0x42 || buf[1] != 0x4d {
            panic!("bmp error");
        }

        let offset = u32::from_le_bytes([buf[10], buf[11], buf[12], buf[13]]);

        let info = BitMapInfoHeader {
            bi_size: u32::from_le_bytes([buf[14], buf[15], buf[16], buf[17]]),
            bi_width: u32::from_le_bytes([buf[18], buf[19], buf[20], buf[21]]),
            bi_height: u32::from_le_bytes([buf[22], buf[23], buf[24], buf[25]]),
            bi_planes: u16::from_le_bytes([buf[26], buf[27]]),
            bi_bit_count: u16::from_le_bytes([buf[28], buf[29]]),
            bi_compression: u32::from_le_bytes([buf[30], buf[31], buf[32], buf[33]]),
            bi_size_image: u32::from_le_bytes([buf[34], buf[35], buf[36], buf[37]]),
            bi_x_pels_per_meter: u32::from_le_bytes([buf[37], buf[38], buf[39], buf[40]]),
            bi_y_pels_per_meter: u32::from_le_bytes([buf[41], buf[42], buf[43], buf[44]]),
            bi_clr_used: u32::from_le_bytes([buf[14], buf[45], buf[46], buf[47]]),
            bi_clr_important: u32::from_le_bytes([buf[48], buf[49], buf[50], buf[51]]),
        };

        let pixel_size: u32 = if info.bi_bit_count == 24 { 3 } else { 4 };
        let pitch = (info.bi_width * pixel_size + 3) & (!3);
        let padding = (pitch - info.bi_width * pixel_size) as usize;

        let mut bits = vec![vec![Vec4::ZERO; info.bi_width as usize]; info.bi_height as usize];
        let mut index = offset as usize;

        for y in 0..info.bi_height as usize {
            for x in 0..info.bi_width as usize {
                let color;
                if pixel_size == 3 {
                    color = u8_array_to_vec4([buf[index + 2], buf[index + 1], buf[index], 255]);
                } else {
                    color = u8_array_to_vec4([
                        buf[index + 2],
                        buf[index + 1],
                        buf[index],
                        buf[index + 3],
                    ]);
                }
                bits[y][x] = color;
                index += pixel_size as usize;
            }

            index += padding;
        }

        Self {
            width: info.bi_width,
            height: info.bi_height,
            bits: bits,
        }
    }

    pub fn merge(
        mesh: &Vec<((u32, u32), FrameBuffer)>,
        row_w: u32,
        col_h: u32,
        width: u32,
        height: u32,
    ) -> FrameBuffer {
        if mesh.len() == 0 {
            return FrameBuffer::new(0, 0);
        }
        let mut result = FrameBuffer::new(width, height);

        for ((col, row), buf) in mesh {
            for i in 0..buf.height {
                for j in 0..buf.width {
                    result.set_pixel(
                        row * row_w + j,
                        col * col_h + i,
                        buf.bits[i as usize][j as usize],
                    );
                }
            }
        }
        return result;
    }

    #[inline]
    fn get_pixel(&self, x: u32, y: u32) -> Vec4 {
        self.bits[y as usize][x as usize]
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

        let c11 = self.get_pixel(x1, y1) * (1.0 - a) * (1.0 - b);
        let c12 = self.get_pixel(x1, y2) * (1.0 - a) * b;
        let c21 = self.get_pixel(x2, y1) * a * (1.0 - b);
        let c22 = self.get_pixel(x2, y2) * a * b;

        vec4(
            c11.x + c12.x + c21.x + c22.x,
            c11.y + c12.y + c21.y + c22.y,
            c11.z + c12.z + c21.z + c22.z,
            c11.w + c12.w + c21.w + c22.w,
        )
    }

    pub fn set_pixel(&mut self, x: u32, y: u32, color: Vec4) {
        self.bits[y as usize][x as usize] = color;
    }

    pub fn draw_line(&mut self, x1: u32, y1: u32, x2: u32, y2: u32, color: Vec4) {
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


pub struct BitMapInfoHeader {
    pub bi_size: u32,
    pub bi_width: u32,
    pub bi_height: u32,
    pub bi_planes: u16,
    pub bi_bit_count: u16,
    pub bi_compression: u32,
    pub bi_size_image: u32,
    pub bi_x_pels_per_meter: u32,
    pub bi_y_pels_per_meter: u32,
    pub bi_clr_used: u32,
    pub bi_clr_important: u32,
}
