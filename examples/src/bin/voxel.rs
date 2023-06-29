use f_renderer::buffer_vec::BufferVec;
use f_renderer::matrix_util;
use f_renderer::wgpu_base::{Rgba, WgpuRenderer};
use futures_lite;
use glam::{Mat4, Vec4Swizzles};
use glam::{Vec2, Vec3, Vec4};
use matrix_util::set_rotate;
use rand::{self, Rng};
use std::cmp::min;
use std::f32::consts::{E, PI};
use std::iter::Sum;
use std::time::Instant;
use wgpu::BufferUsages;
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn main() {
    let w = 960;
    let h = 540;
    let width = w as f32;
    let height = h as f32;

    let proj_inverse = Mat4::perspective_lh(PI * 0.25, width / height, 0.1, 100.0).inverse();
    let mut camera = camera::new(
        Vec3::new(0.0, 0.0, 5.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    );
    let mut look_at_inverse = camera.look_at().inverse();
    let model_inverse = Mat4::IDENTITY.inverse();

    let level = 3;
    let voxel = Voxel::gen_randomly(level);
    let length = 2.0;
    let cube_range = CubeRange::new(Vec3::new(0.0, 0.0, 0.0), length);
    let voxel_cube = VoxelCube::new(voxel, level, length);

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    window.set_inner_size(PhysicalSize::new(w, h));
    let mut state = futures_lite::future::block_on(WgpuRenderer::new(&window));
    let mut src_data: BufferVec<Rgba> = BufferVec::new(BufferUsages::COPY_SRC);

    let mut mouse_right_press = false;
    let mut mouse_middle_press = false;
    let mut cursor_pos = Vec2::new(0.0, 0.0);
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    WindowEvent::MouseWheel { delta, .. } => match delta {
                        MouseScrollDelta::LineDelta(x, y) => {
                            let (x, y) = (*x, *y);
                            let mut forward = (camera.eye - camera.center).normalize();
                            let distance = camera.eye.distance(camera.center);
                            if (-1.0 < distance && y > 0.0) || (distance < 20.0 && y < 0.0) {
                                forward = forward * (distance - y * 0.2);
                                camera.eye = forward + camera.center;
                                look_at_inverse = camera.look_at().inverse();
                            }
                        }
                        MouseScrollDelta::PixelDelta(p) => {}
                    },
                    WindowEvent::MouseInput { state, button, .. } => match button {
                        MouseButton::Right => {
                            mouse_right_press = match state {
                                ElementState::Pressed => true,
                                ElementState::Released => false,
                            }
                        }
                        MouseButton::Middle => {
                            mouse_middle_press = match state {
                                ElementState::Pressed => true,
                                ElementState::Released => false,
                            }
                        }
                        _ => {}
                    },
                    WindowEvent::CursorMoved { position, .. } => {
                        let x = position.x as f32;
                        let y = position.y as f32;

                        let theta_x = x - cursor_pos.x;
                        let theta_y = y - cursor_pos.y;
                        let forward = camera.center - camera.eye;
                        let right = forward.cross(camera.up).normalize();
                        if mouse_right_press {
                            let mut forward = Vec4::from((forward, 1.0));
                            let ratio = 0.005;
                            let rotate_horizon_mat = set_rotate(camera.up, theta_x * PI * ratio);
                            let rotate_vertical_mat = set_rotate(right, -theta_y * PI * ratio);

                            forward = rotate_vertical_mat * rotate_horizon_mat * forward;
                            let new_forward = Vec3::new(forward.x, forward.y, forward.z);
                            camera.up = right.cross(new_forward).normalize();
                            camera.eye = camera.center - Vec3::new(forward.x, forward.y, forward.z);
                        } else if mouse_middle_press {
                            let up = camera.up.normalize();
                            let ratio = 0.01;
                            let offset = (up * theta_y + right * theta_x) * ratio;
                            camera.center -= offset;
                            camera.eye -= offset;
                        }
                        look_at_inverse = camera.look_at().inverse();

                        cursor_pos = Vec2::new(x, y);
                    }
                    _ => {}
                }
            }
        }
        Event::RedrawRequested(window_id) if window_id == window.id() => {
            // cargo run --release --bin voxel
            let start_time = Instant::now();
            src_data.clear();
            for y in 0..h {
                for x in 0..w {
                    let screen_pos = Vec2::new(x as f32, y as f32);
                    let ndc_pos = Vec2::new(
                        screen_pos.x * 2.0 / width - 1.0,
                        1.0 - (screen_pos.y * 2.0 / height),
                    );
                    let dir = model_inverse
                        * look_at_inverse
                        * proj_inverse
                        * Vec4::new(ndc_pos.x, ndc_pos.y, 1.0, 1.0);
                    let ray = Ray::new(camera.eye, dir.truncate().normalize());

                    if let Some((start, end)) = voxel_cube.intersect(ray) {
                        if let Some(color) = voxel_cube.ray_cast(ray, start, end) {
                            src_data.push(color);
                        } else {
                            src_data.push(Rgba::new());
                        }
                    } else {
                        src_data.push(Rgba::new());
                    }
                }
            }

            src_data.write_buffer(state.device(), state.queue());
            match state.render_by_buffer(&src_data) {
                Ok(_) => {}
                Err(wgpu::SurfaceError::Lost) => state.resize(state.size()),
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                Err(e) => eprintln!("{:?}", e),
            }
            let elapsed_time = start_time.elapsed();
            println!("fps: {}", 1.0 / elapsed_time.as_secs_f32());
        }
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        _ => {}
    });
}

struct VoxelRenderer {
    voxel_cube: VoxelCube,
    cube_range: CubeRange,
    width: u32,
    height: u32,
}

impl VoxelRenderer {}

struct camera {
    eye: Vec3,
    center: Vec3,
    up: Vec3,
}

impl camera {
    fn new(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        Self {
            eye: eye,
            center: center,
            up: up,
        }
    }

    fn look_at(&self) -> Mat4 {
        Mat4::look_at_lh(self.eye, self.center, self.up)
    }
}

type Level = u32;

#[derive(Clone, Copy)]
struct Ray {
    pos: Vec3,
    dir: Vec3,
}

impl Ray {
    fn new(pos: Vec3, dir: Vec3) -> Self {
        Self { pos: pos, dir: dir }
    }
}

struct VoxelCube {
    length: f32,
    level: Level,
    voxel: Voxel,
}

impl VoxelCube {
    fn new(voxel: Voxel, level: Level, length: f32) -> Self {
        Self {
            length: length,
            level: level,
            voxel: voxel,
        }
    }

    fn intersect(&self, ray: Ray) -> Option<(Vec3, Vec3)> {
        let length = self.length;

        // 先不考虑射线起点位于立方体内的情况
        if (ray.pos.x > length && ray.pos.x < 0.0)
            || (ray.pos.y > length && ray.pos.y < 0.0)
            || (ray.pos.z > length && ray.pos.z < 0.0)
        {
            return None;
        }

        let mut intersect_points = Vec::new();

        const AXIS_LIST: [Vec3; 3] = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];

        // 设点与平面相交于某点b， 则点b到平面内一点a所代表的向量与平面法向量n垂直, 有(b - a) * n = 0, 可求得射线t值， t > 0则交于平面
        for axis in AXIS_LIST {
            let n = axis;
            let unit_diagonal = Vec3::ONE - axis; // 单位正方形的对角位置

            if ray.dir.cross(n) == Vec3::ZERO {
                // 与轴平行
                let pos = unit_diagonal * ray.pos; // 降维的pos

                if pos.x >= 0.0
                    && pos.x <= length
                    && pos.y >= 0.0
                    && pos.y <= length
                    && pos.z >= 0.0
                    && pos.z <= length
                {
                    intersect_points.push(pos);
                    intersect_points.push(pos + length * unit_diagonal);
                    break;
                }
                continue;
            }

            let dir_dot_n = ray.dir.dot(n);
            if dir_dot_n == 0.0 {
                // 射线方向与平面法向量垂直, 则与另外两个平面之一平行
                continue;
            }
            for a in [Vec3::ZERO, axis * length] {
                // 三个近平面都交于原点, 远平面相交于对角点
                let ap = a - ray.pos;
                if ap.cross(ray.dir) == Vec3::ZERO {
                    // a就是交点
                    intersect_points.push(a);
                } else {
                    let ap_dot_n = ap.dot(n);
                    let t = ap_dot_n / dir_dot_n;
                    if t >= 0.0 {
                        // t >= 0表示射线沿正向移动，即存在交点
                        let b = ray.pos + ray.dir * t;
                        if b.y >= 0.0
                            && b.y <= length
                            && b.z >= 0.0
                            && b.z <= length
                            && b.x >= 0.0
                            && b.x <= length
                        {
                            // todo 误差
                            intersect_points.push(b)
                        }
                    }
                }
            }
        }

        if intersect_points.is_empty() {
            return None;
        }

        if intersect_points.len() < 2 {
            return Some((intersect_points[0], intersect_points[0]));
        }

        intersect_points.sort_by(|a, b| a.distance(ray.pos).total_cmp(&b.distance(ray.pos)));

        if intersect_points.len() > 2 {
            let mut i = 1;
            for j in 0..intersect_points.len() {
                if intersect_points[i] != intersect_points[j] {
                    intersect_points[i] = intersect_points[j];
                    i = j;
                }
            }
        }

        Some((intersect_points[0], intersect_points[1]))
    }

    fn ray_cast(&self, ray: Ray, in_pos: Vec3, out_pos: Vec3) -> Option<Rgba> {
        let length = self.length;
        let t_max_vec = (out_pos - in_pos) / ray.dir;
        let t_max = t_max_vec.x.min(t_max_vec.y).min(t_max_vec.z);
        let per_t = self.length / 2.0_f32.powi(self.level as i32) * 0.01;
        let mut t = 0.0;
        let cube_range = CubeRange::new(Vec3::new(0.0, 0.0, 0.0), length);
        let mut result = Rgba::new();
        while t <= t_max {
            if let Some(leaf) = Self::find_leaf(&self.voxel, cube_range, in_pos + t * ray.dir) {
                return Some(leaf.color); // todo
            }
            if t >= t_max {
                break;
            }
            t = (t + per_t).min(t_max);
        }

        None
    }

    fn find_leaf(voxel: &Voxel, range: CubeRange, pos: Vec3) -> Option<Leaf> {
        let valid_mask = voxel.valid_mask;
        let leaf_mask = voxel.leaf_mask;
        let mut children_index = 0;
        let mut leaf_index = 0;
        for i in 0..(8 as u8) {
            let bit = 1 << i;
            if (bit & valid_mask) != bit {
                continue;
            }
            let is_leaf = (bit & leaf_mask) == bit;

            let (is_inside, new_range) = Self::check_inside(range, pos, bit);
            if is_inside {
                if is_leaf {
                    return Some(voxel.leaves[leaf_index]);
                }
                return Self::find_leaf(&voxel.children[children_index], new_range, pos);
            }

            if is_leaf {
                leaf_index += 1;
            } else {
                children_index += 1;
            }
        }

        None
    }

    fn check_inside_cube(cube_range: CubeRange, pos: Vec3) -> bool {
        cube_range.root.x <= pos.x
            && pos.x < (cube_range.root.x + cube_range.length)
            && cube_range.root.y <= pos.y
            && pos.y < (cube_range.root.y + cube_range.length)
            && cube_range.root.z <= pos.z
            && pos.z < (cube_range.root.z + cube_range.length)
    }

    fn get_sub_cube_range(cube_range: CubeRange, bit: u8) -> CubeRange {
        let length = cube_range.length;
        let root = cube_range.root;
        let half = length * 0.5;
        match bit {
            1 => CubeRange::new(root, half),
            2 => CubeRange::new(Vec3::new(root.x + half, root.y, root.z), half),
            4 => CubeRange::new(Vec3::new(root.x, root.y, root.z + half), half),
            8 => CubeRange::new(Vec3::new(root.x + half, root.y, root.z + half), half),
            16 => CubeRange::new(Vec3::new(root.x, root.y + half, root.z), half),
            32 => CubeRange::new(Vec3::new(root.x + half, root.y + half, root.z), half),
            64 => CubeRange::new(Vec3::new(root.x, root.y + half, root.z + half), half),
            128 => CubeRange::new(Vec3::new(root.x + half, root.y + half, root.z + half), half),
            _ => cube_range,
        }
    }

    fn check_inside(cube_range: CubeRange, pos: Vec3, bit: u8) -> (bool, CubeRange) {
        let sub_cube_range = Self::get_sub_cube_range(cube_range, bit);
        (Self::check_inside_cube(sub_cube_range, pos), sub_cube_range)
    }
}

#[derive(Clone, Copy)]
struct CubeRange {
    root: Vec3,
    length: f32,
}

impl CubeRange {
    fn new(pos: Vec3, length: f32) -> Self {
        Self {
            root: pos,
            length: length,
        }
    }
}

#[derive(Clone, Copy)]
struct Leaf {
    id: u32,
    color: Rgba,
}

impl Leaf {
    fn new(id: u32) -> Self {
        Self {
            id: id,
            color: Rgba::WHITE,
        }
    }

    fn new_by_color(rgba: Rgba) -> Self {
        Self { id: 0, color: rgba }
    }
}

struct Voxel {
    valid_mask: u8,
    leaf_mask: u8,
    children: Vec<Voxel>,
    leaves: Vec<Leaf>,
}

impl Voxel {
    fn new() -> Self {
        Self {
            valid_mask: 0,
            leaf_mask: 0,
            children: vec![],
            leaves: vec![],
        }
    }

    fn new_full() -> Self {
        Self {
            valid_mask: u8::MAX,
            leaf_mask: u8::MAX,
            children: vec![],
            leaves: vec![Leaf::new(0); 8],
        }
    }

    fn new_test() -> Self {
        let mut leaves = vec![];
        for i in 0..7 {
            leaves.push(Leaf::new_by_color(Rgba::new_randomly()));
        }
        Self {
            valid_mask: u8::MAX - 1,
            leaf_mask: u8::MAX - 1,
            children: vec![],
            leaves: leaves,
        }
    }

    fn gen_randomly(level: Level) -> Self {
        let mut voxel = Self::new();
        let mut rng = rand::thread_rng();

        for i in 0..8 {
            let bit = (1 << i) as u8;
            let is_valid = rng.gen_bool(70.0 / 100.0);
            if is_valid {
                voxel.valid_mask += bit;
                let is_leaf = if level > 0 {
                    rng.gen_bool(30.0 / 100.0)
                } else {
                    true
                };
                if is_leaf {
                    voxel.leaf_mask += bit;
                    voxel.leaves.push(Leaf::new_by_color(Rgba::new_randomly()));
                } else {
                    voxel.children.push(Self::gen_randomly(level - 1));
                }
            }
        }

        voxel
    }

    fn depth_first(voxel: &Voxel, level: Level) {
        let mut valid_mask = voxel.valid_mask;
        let mut leaf_mask = voxel.leaf_mask;
        let mut children_index = 0;
        let mut leaf_index = 0;
        for i in 0..8 {
            let bit = (1 << i) as u8;
            if (bit & valid_mask) != bit {
                continue;
            }
            let is_leaf = (bit & leaf_mask) == bit;
            if !is_leaf {
                Self::depth_first(&voxel.children[children_index], level + 1);
                children_index += 1;
            } else {
                leaf_index += 1;
            }
        }
    }

    fn get_leaves_count(voxel: &Voxel) -> usize {
        let mut valid_mask = voxel.valid_mask;
        let mut leaf_mask = voxel.leaf_mask;
        let mut children_index = 0;
        let mut leaf_index = 0;
        let mut leaves_num = voxel.leaves.len();
        for i in 0..8 {
            let bit = (1 << i) as u8;
            if (bit & valid_mask) != bit {
                continue;
            }
            let is_leaf = (bit & leaf_mask) == bit;
            if !is_leaf {
                leaves_num += Self::get_leaves_count(&voxel.children[children_index]);
                children_index += 1;
            } else {
                leaf_index += 1;
            }
        }
        return leaves_num;
    }
}
