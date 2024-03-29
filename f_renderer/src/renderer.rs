use glam::{vec3, vec4, IVec2, UVec2, Vec2, Vec3, Vec4};
use std::cmp::{max, min};
use std::f32::consts::PI;
use std::ops::{Add, Mul, Sub};

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
fn is_top_left(a: IVec2, b: IVec2) -> bool {
    ((a.y == b.y) && (a.x < b.x)) || (a.y > b.y)
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

pub struct Renderer {}

impl Renderer {
    const EPSILON: f32 = 1.0e-5;

    #[inline]
    fn insides<T>(plane: &Plane, vertex: &Vertex<T>) -> bool {
        let w = vertex.pos.w;
        match plane {
            Plane::X_LEFT => vertex.pos.x >= -w,
            Plane::X_RIGHT => vertex.pos.x <= w,
            Plane::Y_UP => vertex.pos.y <= w,
            Plane::Y_DOWN => vertex.pos.y >= -w,
            Plane::Z_FAR => vertex.pos.z <= vertex.pos.w,
            Plane::Z_NEAR => vertex.pos.z >= 0.0, // todo <= -w
            Plane::W_PLANE => vertex.pos.w >= Self::EPSILON,
        }
    }

    #[inline]
    fn calculate_intersect_ratio<T>(plane: &Plane, a: &Vertex<T>, b: &Vertex<T>) -> f32 {
        let a_w = a.pos.w;
        let b_w = b.pos.w;
        match plane {
            Plane::X_LEFT => -(a.pos.x + a_w) / (b_w + b.pos.x - a.pos.x - a_w),
            Plane::X_RIGHT => (a_w - a.pos.x) / (a_w - b_w - a.pos.x + b.pos.x),
            Plane::Y_UP => (a_w - a.pos.y) / (a_w - b_w - a.pos.y + b.pos.y),
            Plane::Y_DOWN => -(a.pos.y + a_w) / (b_w + b.pos.y - a_w - a.pos.y),
            Plane::Z_FAR => (a_w - a.pos.z) / (a_w - b_w - a.pos.z + b.pos.z),
            Plane::Z_NEAR => a_w / (a_w - b_w), // ...todo
            Plane::W_PLANE => (Self::EPSILON - a_w) / (b_w - a_w), // todo
        }
    }

    #[inline]
    fn vertex_intersect<
        ShaderContext: Add<Output = ShaderContext>
            + Sub<Output = ShaderContext>
            + Mul<f32, Output = ShaderContext>
            + Copy
            + Clone
            + Default,
    >(
        a: &Vertex<ShaderContext>,
        b: &Vertex<ShaderContext>,
        ratio: f32,
    ) -> Vertex<ShaderContext> {
        let mut new_vertex = Vertex::default();
        new_vertex.pos = a.pos + ratio * (b.pos - a.pos);

        new_vertex.context = a.context + (b.context - a.context) * ratio;

        new_vertex
    }

    pub fn geometry_processing<
        ShaderContext: Add<Output = ShaderContext>
            + Sub<Output = ShaderContext>
            + Mul<f32, Output = ShaderContext>
            + Copy
            + Clone
            + Default,
        VSInput,
        VSUniform,
        F: Fn(&VSUniform, &VSInput, &mut ShaderContext) -> Vec4
    >(
        width: u32,
        height: u32,
        vs_inputs: &[VSInput; 3],
        vertex_shader: &F,
        vs_uniform: &VSUniform,
    ) -> Option<Vec<[Vertex<ShaderContext>; 3]>> {
        let mut vertices = vec![Vertex::default(), Vertex::default(), Vertex::default()];

        for i in 0..3 {
            let pos = (vertex_shader)(vs_uniform, &vs_inputs[i], &mut vertices[i].context);
            if pos.w == 0.0 {
                return None;
            }
            vertices[i].pos = pos;
        }

        const PLANE_LIST: [Plane; 6] = [
            Plane::X_LEFT,
            Plane::X_RIGHT,
            Plane::Y_UP,
            Plane::Y_DOWN,
            Plane::Z_NEAR,
            Plane::Z_FAR,
            // Plane::W_PLANE, // todo
        ];
        let mut inside_list = [
            [false; PLANE_LIST.len()],
            [false; PLANE_LIST.len()],
            [false; PLANE_LIST.len()],
        ];

        let mut all_insides = true;
        for i in 0..3 {
            let v = &vertices[i];
            let mut v_all_inside = true;
            for (j, plane) in PLANE_LIST.iter().enumerate() {
                let is_inside = Self::insides(plane, v);
                inside_list[i][j] = is_inside;
                v_all_inside &= is_inside;
            }
            all_insides &= v_all_inside;
        }

        let mut valid_vertices = vec![];
        if !all_insides {
            for i in 0..3 {
                let a = &vertices[i];
                for j in (i + 1)..3 {
                    let b = &vertices[j];

                    for (plane_index, plane) in PLANE_LIST.iter().enumerate() {
                        let a_inside = inside_list[i][plane_index];
                        let b_inside = inside_list[j][plane_index];

                        if a_inside != b_inside {
                            let ratio = Self::calculate_intersect_ratio(&plane, &a, &b);
                            let new_vertex = Self::vertex_intersect(&a, &b, ratio);
                            if new_vertex.pos.w.abs() > Self::EPSILON {
                                valid_vertices.push(new_vertex);
                            }
                        }
                    }
                }
            }
            valid_vertices.extend(vertices);
        } else {
            valid_vertices = vertices;
        }

        if valid_vertices.len() < 3 {
            return None;
        }

        let mut centroid = Vec2::new(0.0, 0.0);

        for vertex in valid_vertices.iter() {
            centroid.x += vertex.pos.x;
            centroid.y += vertex.pos.y;
        }

        centroid *= 1.0 / valid_vertices.len() as f32;

        // let mut vertices_tuple: Vec<(Vertex<ShaderContext>, f32)> = valid_vertices
        //     .into_iter()
        //     .map(|a| {
        //         let forward_a = Vec2::new(a.pos.x - centroid.x, a.pos.y - centroid.y);
        //         let mut atan_a = forward_a.y.atan2(forward_a.x);
        //         if atan_a < 0.0 {
        //             atan_a += PI * 2.0;
        //         }
        //         (a, atan_a)
        //     })
        //     .collect();

        // vertices_tuple.sort_by(|a, b| a.1.total_cmp(&b.1));

        // let mut valid_vertices: Vec<Vertex<ShaderContext>> = vertices_tuple.into_iter().map(|x| x.0).collect();

        valid_vertices.sort_by(|a, b| {
            let forward_a = Vec2::new(a.pos.x - centroid.x, a.pos.y - centroid.y);
            let forward_b = Vec2::new(b.pos.x - centroid.x, b.pos.y - centroid.y);
            let mut atan_a = forward_a.y.atan2(forward_a.x); //todo  Redundant calculations
            let mut atan_b = forward_b.y.atan2(forward_b.x); // todo Redundant calculations
            if atan_a < 0.0 {
                atan_a += PI * 2.0;
            }
            if atan_b < 0.0 {
                atan_b += PI * 2.0;
            }

            atan_a.total_cmp(&atan_b)
        });

        for i in 0..valid_vertices.len() {
            let vertex = &mut valid_vertices[i];
            let w = vertex.pos.w;
            vertex.rhw = 1.0 / w;

            // normalized device coordinate
            vertex.pos = vertex.pos * vertex.rhw;

            // screen coordinate
            vertex.spf.x = (vertex.pos.x + 1.0) * (width as f32) * 0.5;
            vertex.spf.y = (1.0 - vertex.pos.y) * (height as f32) * 0.5;

            // int screen coordinate
            vertex.spi.x = (vertex.spf.x + 0.5) as i32;
            vertex.spi.y = (vertex.spf.y + 0.5) as i32;
        }

        if valid_vertices.len() == 3 {
            return Some(vec![[
                valid_vertices[0],
                valid_vertices[1],
                valid_vertices[2],
            ]]);
        }

        let mut triangles = vec![];

        let mut last_vertex_index = valid_vertices.len() - 1;

        while last_vertex_index > 3 {
            let a = valid_vertices[last_vertex_index];
            let b = valid_vertices[last_vertex_index - 1];
            triangles.push([valid_vertices[0], b, a]);
            last_vertex_index -= 1;
        }

        let a = valid_vertices[0];
        let b = valid_vertices[2];
        let c = valid_vertices[3];
        triangles.push([a, b, c]);

        let c = valid_vertices[2];
        let b = valid_vertices[1];
        let a = valid_vertices[0];
        triangles.push([a, b, c]);

        Some(triangles)
    }

    pub fn rasterization<ShaderContext, PSUniform, F>(
        width_range: (i32, i32),
        height_range: (i32, i32),
        triangle: &[Vertex<ShaderContext>; 3],
        pixel_shader: &F,
        ps_uniform: &PSUniform,
        frame_buffer: &mut FrameBuffer,
        depth_buffer: &mut [f32],
    ) where
        ShaderContext: Add<Output = ShaderContext>
            + Sub<Output = ShaderContext>
            + Mul<f32, Output = ShaderContext>
            + Copy
            + Clone,
        F: Fn(&PSUniform, &ShaderContext) -> Vec4,
    {
        let mut min_x = triangle[0].spi.x.clamp(width_range.0, width_range.1);
        let mut max_x = min_x;
        let mut min_y = triangle[0].spi.y.clamp(height_range.0, height_range.1);
        let mut max_y = min_y;

        for k in 1..3 {
            let vertex = &triangle[k];

            min_x = min(min_x, vertex.spi.x).clamp(width_range.0, width_range.1);
            max_x = max(max_x, vertex.spi.x).clamp(width_range.0, width_range.1);

            min_y = min(min_y, vertex.spi.y).clamp(height_range.0, height_range.1);
            max_y = max(max_y, vertex.spi.y).clamp(height_range.0, height_range.1);
        }

        let v01 = triangle[1].pos - triangle[0].pos;
        let v02 = triangle[2].pos - triangle[0].pos;

        let v01 = Vec3::new(v01.x, v01.y, v01.z);
        let v02 = Vec3::new(v02.x, v02.y, v02.z);
        let normal = v01.cross(v02);

        let mut vtx = [&triangle[0], &triangle[1], &triangle[2]];

        if normal.z > 0.0 {
            vtx[1] = &triangle[2];
            vtx[2] = &triangle[1];
        }

        let p0 = vtx[0].spi;
        let p1 = vtx[1].spi;
        let p2 = vtx[2].spi;

        let top_left_01 = is_top_left(p0, p1);
        let top_left_12 = is_top_left(p1, p2);
        let top_left_20 = is_top_left(p2, p0);

        for cy in min_y..max_y {
            let index_y = (cy - height_range.0) as usize;
            for cx in min_x..max_x {
                let px = Vec2::new(cx as f32 + 0.5, cy as f32 + 0.5);
                let index_x = (cx - width_range.0) as usize;

                // Edge Equation
                let E01 = -(cx - p0.x) * (p1.y - p0.y) + (cy - p0.y) * (p1.x - p0.x);
                let E12 = -(cx - p1.x) * (p2.y - p1.y) + (cy - p1.y) * (p2.x - p1.x);
                let E20 = -(cx - p2.x) * (p0.y - p2.y) + (cy - p2.y) * (p0.x - p2.x);

                if E01 < if top_left_01 { 0 } else { 1 } {
                    continue;
                }
                if E12 < if top_left_12 { 0 } else { 1 } {
                    continue;
                }
                if E20 < if top_left_20 { 0 } else { 1 } {
                    continue;
                }

                let s0 = vtx[0].spf - px;
                let s1 = vtx[1].spf - px;
                let s2 = vtx[2].spf - px;

                let mut a = s1.perp_dot(s2).abs();
                let mut b = s2.perp_dot(s0).abs();
                let mut c = s0.perp_dot(s1).abs();

                let s = a + b + c;
                if s == 0.0 {
                    continue;
                }

                a = a * (1.0 / s);
                b = b * (1.0 / s);
                c = c * (1.0 / s);

                let rhw = vtx[0].rhw * a + vtx[1].rhw * b + vtx[2].rhw * c;

                let index = index_y * width_range.1 as usize + index_x;
                if rhw < depth_buffer[index] {
                    continue;
                }
                depth_buffer[index] = rhw;

                let w = 1.0 / if rhw != 0.0 { rhw } else { 1.0 };

                let c0 = vtx[0].rhw * a * w;
                let c1 = vtx[1].rhw * b * w;
                let c2 = vtx[2].rhw * c * w;

                let i0 = vtx[0].context;
                let i1 = vtx[1].context;
                let i2 = vtx[2].context;

                let input = i0 * c0 + i1 * c1 + i2 * c2;

                let color = (pixel_shader)(ps_uniform, &input);
                frame_buffer.set_pixel(index_x as u32, index_y as u32, vec4_to_u8_array(color));
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct Vertex<T> {
    context: T,
    rhw: f32, //reciprocal of w
    pos: Vec4,
    spf: Vec2,  // screen coordinate
    spi: IVec2, // int screen coordinate
}

impl<T> Default for Vertex<T>
where
    T: Default,
{
    fn default() -> Self {
        Self {
            context: T::default(),
            rhw: 0.0,
            pos: Vec4::ZERO,
            spf: Vec2::ZERO,
            spi: IVec2::ZERO,
        }
    }
}

#[derive(Clone)]
pub struct FrameBuffer {
    width: u32,
    height: u32,
    buffer: Vec<u8>,
}

impl FrameBuffer {
    pub fn new(width: u32, height: u32) -> FrameBuffer {
        return FrameBuffer {
            width: width,
            height: height,
            buffer: vec![0; (width * height * 4) as usize],
        };
    }

    pub fn load_file(path: &str) -> Self {
        let image = image::open(path).unwrap();

        let width = image.width() as usize;
        let height = image.height() as usize;
        let mut bits = vec![0 as u8; width * height * 4];
        let color = image.as_bytes();

        match image.color() {
            image::ColorType::Rgb8 => {
                println!("rgb {}", path);
                for y in 0..height {
                    for x in 0..width {
                        let index = y * width * 4 + x * 4;
                        let color_index = y * width * 3 + x * 3;
                        bits[index] = color[color_index + 2];
                        bits[index + 1] = color[color_index + 1];
                        bits[index + 2] = color[color_index + 0];
                        bits[index + 3] = 255;
                    }
                }
            }
            image::ColorType::Rgba8 => {
                println!("rgba {}", path);
                for y in 0..height {
                    for x in 0..width {
                        let index = y * width * 4 + x * 4;
                        bits[index] = color[index + 2];
                        bits[index + 1] = color[index + 1];
                        bits[index + 2] = color[index];
                        bits[index + 3] = color[index + 3];
                    }
                }
            }
            _ => {
                panic!("invalid color type")
            }
        }

        Self {
            width: image.width(),
            height: image.height(),
            buffer: bits,
        }
    }

    pub fn get_data(&self) -> &Vec<u8> {
        &self.buffer
    }

    pub fn clear(&mut self) {
        self.buffer.fill(0);
    }

    pub fn get_size(&self) -> u32 {
        self.width * self.height * 4
    }

    pub fn fill(&mut self, color: [u8; 4]) {
        let size = self.get_size() as usize;
        let mut i = 0;
        while i < size{
            for (offset, bit) in color.iter().enumerate(){
                self.buffer[i + offset] = *bit;
            }
            i += 4;
        }
    }

    #[inline]
    pub fn set_pixel(&mut self, x: u32, y: u32, color: [u8; 4]) {
        let offset = (y * self.width * 4 + x * 4) as usize;
        for i in 0..4 {
            //todo 这样存y会反过来
            self.buffer[offset + i] = color[i];
        }
    }

    #[inline]
    pub fn get_pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let offset = (y * self.width * 4 + x * 4) as usize;
        [
            self.buffer[offset],
            self.buffer[offset + 1],
            self.buffer[offset + 2],
            self.buffer[offset + 3],
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
