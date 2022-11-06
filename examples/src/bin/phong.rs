

use f_renderer::*;
use camera::Camera;
use glam::{Mat4, Vec2, Vec3, Vec4};
use matrix_util::{set_identity, set_perspective, set_rotate};
use renderer::{FrameBuffer, Renderer};
use vector_util::{reflect};

use std::ops::{Add, Mul, Sub};

use ash::util::*;
pub use ash::{Device, Instance};
use obj_loader::Model;
use std::f32::consts::PI;
use std::time::Instant;
use winit::event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent};



fn main() {
    const RECIPROCAL_SCALE: u32 = 1;
    const WIDTH: u32 = 1920 / RECIPROCAL_SCALE;
    const HEIGHT: u32 = 1080 / RECIPROCAL_SCALE;

    #[derive(Clone, Copy)]
    struct VSUniform {
        model: Mat4,
        view: Mat4,
        proj: Mat4,
    }

    #[derive(Clone, Copy)]
    enum PLACE {
        BODY,
        FACE,
        HAIR,
    }

    #[derive(Clone, Copy)]
    struct PSUniform<'a> {
        place: PLACE,
        sample_2d_body: &'a FrameBuffer,
        sample_2d_face: &'a FrameBuffer,
        sample_2d_hair: &'a FrameBuffer,
        view_pos: Vec3
    }

    #[derive(Clone, Copy)]
    struct VSInput {
        pub pos: Vec3,
        pub uv: Vec2,
        pub normal: Vec3,
    }

    impl VSInput {
        pub const ZERO: Self = Self {
            pos: Vec3::ZERO,
            uv: Vec2::ZERO,
            normal: Vec3::ZERO,
        };
    }

    #[derive(Clone, Copy)]
    struct ShaderContext {
        uv: Vec2,
        normal: Vec3,
        pos: Vec3,
    }

    impl Mul<f32> for ShaderContext{
        type Output = Self;
        fn mul(self, rhs: f32) -> Self::Output {
            Self {
                uv: self.uv * rhs,
                normal: self.normal * rhs,
                pos: self.pos * rhs,
            }
        }
    }

    impl Add for ShaderContext {
        type Output = Self;
        fn add(self, rhs: Self) -> Self::Output {
            Self {
                uv: self.uv + rhs.uv,
                normal: self.normal + rhs.normal,
                pos: self.pos + rhs.pos,
            }
        }
    }

    impl Sub for ShaderContext {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self {
                uv: self.uv - rhs.uv,
                normal: self.normal - rhs.normal,
                pos: self.pos - rhs.pos,
            }
        }  
    }

    impl Default for ShaderContext{
        fn default() -> Self {
            Self {
                uv: Vec2::ZERO,
                normal: Vec3::ZERO,
                pos: Vec3::ZERO,
            }
        }
    }

    let vertex_shader = |
        vs_uniform: &VSUniform,
        vs_input: &VSInput,
        context: &mut ShaderContext,
    | -> Vec4 {
        let mvp = vs_uniform.proj * vs_uniform.view * vs_uniform.model;
        context.uv = vs_input.uv;
        let normal = vs_input.normal;
        context.normal = Vec3::new(normal.x, normal.y, normal.z);
        let pos = vs_uniform.model * Vec4::from((vs_input.pos, 1.0));
        context.pos = Vec3::new(pos.x, pos.y, pos.z);
        mvp * Vec4::from((vs_input.pos, 1.0))
    };

    const LIGHT_COLOR: Vec3 = Vec3::new(1.0, 1.0, 1.0);
    const LIGHT_POS: Vec3 = Vec3::new(1.2, 1.0, 2.0);

    const AMBIENT_STRENGTH: f32 = 0.1;
    const SPECULAR_STRENGTH: f32 = 0.5;
    let pixel_shader = |ps_uniform: &PSUniform, context: &ShaderContext| -> Vec4 {
        let ambient_color = LIGHT_COLOR * AMBIENT_STRENGTH;

        let normal = context.normal.normalize();
        let light_dir = (LIGHT_POS - context.pos).normalize();
        let diff = normal.dot(light_dir).max(0.0);
        let diffuse = diff * LIGHT_COLOR;
        
        let view_dir = (ps_uniform.view_pos - context.pos).normalize();
        let reflect_dir = reflect(-light_dir, normal);
        let spec = view_dir.dot(reflect_dir).max(0.0).powi(32);
        let specular = SPECULAR_STRENGTH * spec * LIGHT_COLOR;

        let uv = context.uv;
        let color = match ps_uniform.place {
            PLACE::BODY => ps_uniform.sample_2d_body.sample_2d(uv),
            PLACE::FACE => ps_uniform.sample_2d_face.sample_2d(uv),
            PLACE::HAIR => ps_uniform.sample_2d_hair.sample_2d(uv),
        };

        color * Vec4::from((ambient_color + diffuse + specular, 1.0))
    };

    let mat_model = set_identity();

    let mut camera_1 = Camera::new(
        Vec3::new(0.0, 1.0, 3.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    );

    let mat_proj = set_perspective(PI * 0.25, WIDTH as f32 / HEIGHT as f32, 0.1, 100.0);

    let qiyana_body = Model::new("./assets/obj/qiyana/qiyanabody.obj");
    let body_diffuse = FrameBuffer::load_file("./assets/obj/qiyana/qiyanabody_diffuse.tga");
    let qiyana_face = Model::new("./assets/obj/qiyana/qiyanaface.obj");
    let face_diffuse = FrameBuffer::load_file("./assets/obj/qiyana/qiyanaface_diffuse.tga");
    let qiyana_hair = Model::new("./assets/obj/qiyana/qiyanahair.obj");
    let hair_diffuse = FrameBuffer::load_file("./assets/obj/qiyana/qiyanahair_diffuse.tga");

    let mut vs_uniform = VSUniform {
        model: mat_model,
        view: camera_1.mat_look_at,
        proj: mat_proj,
    };
    let mut ps_uniform = PSUniform {
        place: PLACE::HAIR,
        sample_2d_body: &body_diffuse,
        sample_2d_face: &face_diffuse,
        sample_2d_hair: &hair_diffuse,
        view_pos: camera_1.eye,
    };

    unsafe {
        fn init_vertex_input(model: &Model) -> Vec<[VSInput; 3]>{
            let mut vertices_input = vec![];
            for i in 0..model.faces_len() {
                let mut inputs = [VSInput::ZERO, VSInput::ZERO, VSInput::ZERO];
                for j in 0..3 {
                    inputs[j] = VSInput {
                        pos: model.vert(i, j),
                        uv: model.uv(i, j),
                        normal: model.normal(i, j),
                    };
                }
                vertices_input.push(inputs);
            }
            vertices_input
        }

        let body_vertices_input = init_vertex_input(&qiyana_body);
        let face_vertices_input = init_vertex_input(&qiyana_face);
        let hair_vertices_input = init_vertex_input(&qiyana_hair);
    
        let mut frame_buffer = FrameBuffer::new(WIDTH, HEIGHT);
        let mut depth_buffer = vec![0.0 as f32; (WIDTH * HEIGHT) as usize];

        let mut mouse_right_press = false;
        let mut mouse_middle_press = false;
        let mut cursor_pos = Vec2::new(0.0, 0.0);

        let render_func = |event: Event<()>, image_slice: &mut Align<u8>| {
            match event {
                Event::WindowEvent {
                    event: WindowEvent::MouseWheel { delta, .. },
                    ..
                } => {
                    use winit::event::MouseScrollDelta::*;
                    match delta {
                        LineDelta(x, y) => {
                            let mut forward = (camera_1.eye - camera_1.at).normalize();
                            let distance = camera_1.eye.distance(camera_1.at);
                            if (-1.0 < distance && y > 0.0) || (distance < 20.0 && y < 0.0) {
                                forward = forward * (distance - y * 0.2);
                                let new_eye = forward + camera_1.at;
                                let new_at = new_eye - camera_1.eye + camera_1.at;
                                camera_1.eye = forward + camera_1.at;
                                // camera_1.at = new_at;
                                camera_1.cal_look_at();

                                vs_uniform.view = camera_1.mat_look_at;
                                ps_uniform.view_pos = camera_1.eye;
                            }
                        }
                        PixelDelta(p) => {}
                    }
                }
                Event::WindowEvent {
                    event:
                        WindowEvent::MouseInput {
                            button: MouseButton::Right,
                            ..
                        },
                    ..
                } => {
                    if let Event::WindowEvent { event, .. } = event {
                        if let WindowEvent::MouseInput { state, .. } = event {
                            mouse_right_press = match state {
                                ElementState::Pressed => true,
                                ElementState::Released => false,
                            }
                        }
                    }
                }
                Event::WindowEvent {
                    event:
                        WindowEvent::MouseInput {
                            button: MouseButton::Middle,
                            ..
                        },
                    ..
                } => {
                    if let Event::WindowEvent { event, .. } = event {
                        if let WindowEvent::MouseInput { state, .. } = event {
                            mouse_middle_press = match state {
                                ElementState::Pressed => true,
                                ElementState::Released => false,
                            }
                        }
                    }
                }
                Event::WindowEvent {
                    event: WindowEvent::CursorMoved { .. },
                    ..
                } => {
                    if let Event::WindowEvent { event, .. } = event {
                        if let WindowEvent::CursorMoved { position, .. } = event {
                            let x = position.x as f32;
                            let y = position.y as f32;

                            let theta_x = x - cursor_pos.x;
                            let theta_y = y - cursor_pos.y;
                            let forward = camera_1.at - camera_1.eye;
                            let right = forward.cross(camera_1.up).normalize();
                            if mouse_right_press {
                                let mut forward = Vec4::from((forward, 1.0));
                                let ratio = 0.005;
                                let rotate_horizon_mat =
                                    set_rotate(camera_1.up, theta_x * PI * ratio);
                                let rotate_vertical_mat = set_rotate(right, -theta_y * PI * ratio);

                                forward = rotate_vertical_mat * rotate_horizon_mat * forward;
                                let new_forward = Vec3::new(forward.x, forward.y, forward.z);
                                camera_1.up = right.cross(new_forward).normalize();
                                camera_1.eye =
                                    camera_1.at - Vec3::new(forward.x, forward.y, forward.z);
                            } else if mouse_middle_press {
                                let up = camera_1.up.normalize();
                                let ratio = 0.01;
                                let offset = (up * theta_y + right * theta_x) * ratio;
                                camera_1.at -= offset;
                                camera_1.eye -= offset;
                            }
                            camera_1.cal_look_at();
                            vs_uniform.view = camera_1.mat_look_at;
                            ps_uniform.view_pos = camera_1.eye;

                            cursor_pos = Vec2::new(x, y);
                        }
                    }
                }
                Event::MainEventsCleared => {
                    let start_time = Instant::now();
                    frame_buffer.fill([30, 30, 30, 255]);
                    depth_buffer.fill(0.0);

                    let mut vertices = vec![];

                    for input in body_vertices_input.iter() {
                        if let Some(out_vertices) = Renderer::geometry_processing(
                            WIDTH,
                            HEIGHT,
                            input,
                            &vertex_shader,
                            &vs_uniform,
                        ) {
                            vertices.extend(out_vertices);
                        }
                    }

                    let body_offset = vertices.len();
                    for input in face_vertices_input.iter() {
                        if let Some(out_vertices) = Renderer::geometry_processing(
                            WIDTH,
                            HEIGHT,
                            input,
                            &vertex_shader,
                            &vs_uniform,
                        ) {
                            vertices.extend(out_vertices);
                        }
                    }

                    let face_offset = vertices.len();
                    for input in hair_vertices_input.iter() {
                        if let Some(out_vertices) = Renderer::geometry_processing(
                            WIDTH,
                            HEIGHT,
                            input,
                            &vertex_shader,
                            &vs_uniform,
                        ) {
                            vertices.extend(out_vertices);
                        }
                    }

                    let hair_offset = vertices.len();

                    for i in 0..vertices.len() {
                        let vertex_s = &vertices[i];

                        if i <= body_offset {
                            ps_uniform.place = PLACE::BODY;
                        } else if body_offset < i && i <= face_offset {
                            ps_uniform.place = PLACE::FACE;
                        } else if face_offset < i && i <= hair_offset {
                            ps_uniform.place = PLACE::HAIR;
                        }

                        Renderer::rasterization(
                            (0, WIDTH as i32),
                            (0, HEIGHT as i32),
                            vertex_s,
                            &pixel_shader,
                            &ps_uniform,
                            &mut frame_buffer,
                            &mut depth_buffer,
                        );
                    }

                    let elapsed_time = start_time.elapsed();
                    println!("fps: {}", 1.0 / elapsed_time.as_secs_f32());

                    image_slice.copy_from_slice(frame_buffer.get_data());
                }
                _ => {}
            }
        };

        use vulkan_base::DisplayBase;

        let display_base = DisplayBase::new(WIDTH, HEIGHT);
        display_base.render_loop(render_func);
    }
}
