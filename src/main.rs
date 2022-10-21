mod camera;
mod input_controller;
mod matrix_util;
mod obj_loader;
mod renderer;
mod vulkan_base;

use camera::Camera;
use glam::{Mat4, Vec2, Vec3, Vec4};
use matrix_util::{set_identity, set_perspective, set_rotate};
use renderer::{FrameBuffer, Interpolable, Renderer};

use ash::util::*;
use ash::vk;
pub use ash::{Device, Instance};
use obj_loader::Model;
use std::default::Default;
use std::f32::consts::PI;
use std::time::{Instant};
use vulkan_base::{find_memorytype_index, record_submit_commandbuffer, VulkanBase};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::{
    event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn main() {
    const WIDTH: u32 = 960;
    const HEIGHT: u32 = 540;

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
    }

    impl Interpolable for ShaderContext {
        fn new() -> Self {
            Self {
                uv: Vec2::ZERO,
                normal: Vec3::ZERO,
            }
        }

        fn add(self, rhs: Self) -> Self {
            Self {
                uv: self.uv + rhs.uv,
                normal: self.normal + rhs.normal,
            }
        }

        fn mul(self, rhs: f32) -> Self {
            Self {
                uv: self.uv * rhs,
                normal: self.normal * rhs,
            }
        }

        fn sub(self, rhs: Self) -> Self {
            Self {
                uv: self.uv - rhs.uv,
                normal: self.normal - rhs.normal,
            }
        }
    }

    fn vertex_shader(
        vs_uniform: &VSUniform,
        vs_input: &VSInput,
        context: &mut ShaderContext,
    ) -> Vec4 {
        let mvp = vs_uniform.proj * vs_uniform.view * vs_uniform.model;
        let model_lt = vs_uniform.model.inverse().transpose();
        context.uv = vs_input.uv;
        let normal = model_lt * Vec4::from((vs_input.normal, 1.0));
        context.normal = Vec3::new(normal.x, normal.y, normal.z);
        mvp * Vec4::from((vs_input.pos, 1.0))
    }

    fn pixel_shader(ps_uniform: &PSUniform, context: &ShaderContext) -> Vec4 {
        let uv = context.uv;
        let n = context.normal;
        let l = Vec3::new(1.0, 1.0, 0.85).normalize();
        let color = match ps_uniform.place {
            PLACE::BODY => ps_uniform.sample_2d_body.sample_2d(uv),
            PLACE::FACE => ps_uniform.sample_2d_face.sample_2d(uv),
            PLACE::HAIR => ps_uniform.sample_2d_hair.sample_2d(uv),
        };
        // color * (n.dot(l).clamp(0.0, 1.0) + 0.1)
        color
    }

    let mat_model = set_identity();

    let mut camera_1 = Camera::new(
        Vec3::new(0.0, 1.0, 3.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    );

    let mat_proj = set_perspective(PI * 0.25, WIDTH as f32 / HEIGHT as f32, 0.1, 100.0);

    let qiyana_body = Model::new("./obj/qiyana/qiyanabody.obj");
    let qiyana_face = Model::new("./obj/qiyana/qiyanaface.obj");
    let qiyana_hair = Model::new("./obj/qiyana/qiyanahair.obj");

    let mut vs_uniform = VSUniform {
        model: mat_model,
        view: camera_1.mat_look_at,
        proj: mat_proj,
    };
    let mut ps_uniform = PSUniform {
        place: PLACE::HAIR,
        sample_2d_body: &qiyana_body.diffuse_map,
        sample_2d_face: &qiyana_face.diffuse_map,
        sample_2d_hair: &qiyana_hair.diffuse_map,
    };

    let mut test_renderer: Renderer<VSInput, VSUniform, PSUniform, ShaderContext> = Renderer::new(
        WIDTH,
        HEIGHT,
        vs_uniform,
        vertex_shader,
        ps_uniform,
        pixel_shader,
    );

    let mut frame_buffer = FrameBuffer::new(WIDTH, HEIGHT);
    let mut depth_buffer = vec![vec![0.0; WIDTH as usize]; HEIGHT as usize];

    unsafe {
        let base = VulkanBase::new(WIDTH, HEIGHT);

        let view_ports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: base.surface_resolution.width as f32,
            height: base.surface_resolution.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [base.surface_resolution.into()];

        let image_buffer_info = vk::BufferCreateInfo {
            size: (std::mem::size_of::<u8>() * frame_buffer.get_size() as usize) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let image_buffer = base.device.create_buffer(&image_buffer_info, None).unwrap();
        let image_buffer_memory_req = base.device.get_buffer_memory_requirements(image_buffer);
        let image_buffer_memory_index = find_memorytype_index(
            &image_buffer_memory_req,
            &base.device_memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("Unable to find suitable memory type for the vertex buffer.");

        let image_buffer_allocate_info = vk::MemoryAllocateInfo {
            allocation_size: image_buffer_memory_req.size,
            memory_type_index: image_buffer_memory_index,
            ..Default::default()
        };
        let image_buffer_memory = base
            .device
            .allocate_memory(&image_buffer_allocate_info, None)
            .unwrap();
        let image_ptr = base
            .device
            .map_memory(
                image_buffer_memory,
                0,
                image_buffer_memory_req.size,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap();
        let mut image_slice = Align::new(
            image_ptr,
            std::mem::align_of::<u8>() as u64,
            image_buffer_memory_req.size,
        );

        // image_slice.copy_from_slice(&frame_buffer.bits);
        base.device.unmap_memory(image_buffer_memory);
        base.device
            .bind_buffer_memory(image_buffer, image_buffer_memory, 0)
            .unwrap();

        let buffer_region = vk::BufferImageCopy::builder()
            .image_subresource(
                vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .build(),
            )
            .image_extent(vk::Extent3D {
                width: WIDTH,
                height: HEIGHT,
                depth: 1,
            })
            .build();

        let mut mouse_right_press = false;
        let mut mouse_middle_press = false;
        let mut cursor_pos = Vec2::new(0.0, 0.0);

        base.event_loop
            .borrow_mut()
            .run_return(|event, _, control_flow| {
                *control_flow = ControlFlow::Poll;
                match event {
                    Event::WindowEvent {
                        event: WindowEvent::CloseRequested,
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    Event::WindowEvent {
                        event: WindowEvent::MouseWheel { delta, phase, .. },
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
                                    camera_1.at = new_at;
                                    camera_1.cal_look_at();

                                    vs_uniform.view = camera_1.mat_look_at;
                                    test_renderer.set_vs_uniform(vs_uniform);
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
                        if let Event::WindowEvent { window_id, event } = event {
                            if let WindowEvent::MouseInput {
                                device_id,
                                state,
                                button,
                                modifiers,
                            } = event
                            {
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
                        if let Event::WindowEvent { window_id, event } = event {
                            if let WindowEvent::MouseInput {
                                device_id,
                                state,
                                button,
                                modifiers,
                            } = event
                            {
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
                                    let rotate_vertical_mat =
                                        set_rotate(right, -theta_y * PI * ratio);

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
                                test_renderer.set_vs_uniform(vs_uniform);

                                cursor_pos = Vec2::new(x, y);
                            }
                        }
                    }
                    Event::MainEventsCleared => {
                        let start_time = Instant::now();
                        let (present_index, _) = base
                            .swapchain_loader
                            .acquire_next_image(
                                base.swapchain,
                                std::u64::MAX,
                                base.present_complete_semaphore,
                                vk::Fence::null(),
                            )
                            .unwrap();

                        let present_images = base
                            .swapchain_loader
                            .get_swapchain_images(base.swapchain)
                            .unwrap();

                        frame_buffer.fill([30, 30, 30, 255]);
                        depth_buffer = vec![vec![0.0; WIDTH as usize]; HEIGHT as usize];

                        let mut vertices = vec![];
                        let body_offset = 0;
                        for i in 0..qiyana_body.faces_len() {
                            let mut inputs = [VSInput::ZERO, VSInput::ZERO, VSInput::ZERO];
                            for j in 0..3 {
                                inputs[j] = VSInput {
                                    pos: qiyana_body.vert(i, j),
                                    uv: qiyana_body.uv(i, j),
                                    normal: qiyana_body.normal(i, j),
                                };
                            }
                            let option_vertices = test_renderer.geometry_processing(&inputs);

                            if option_vertices.is_some() {
                                vertices.extend(option_vertices.unwrap());
                            }
                        }

                        let face_offset = vertices.len();
                        for i in 0..qiyana_face.faces_len() {
                            let mut inputs = [VSInput::ZERO, VSInput::ZERO, VSInput::ZERO];
                            for j in 0..3 {
                                inputs[j] = VSInput {
                                    pos: qiyana_face.vert(i, j),
                                    uv: qiyana_face.uv(i, j),
                                    normal: qiyana_face.normal(i, j),
                                };
                            }
                            let option_vertices = test_renderer.geometry_processing(&inputs);

                            if option_vertices.is_some() {
                                vertices.extend(option_vertices.unwrap());
                            }
                        }

                        let hair_offset = vertices.len();
                        for i in 0..qiyana_hair.faces_len() {
                            let mut inputs = [VSInput::ZERO, VSInput::ZERO, VSInput::ZERO];
                            for j in 0..3 {
                                inputs[j] = VSInput {
                                    pos: qiyana_hair.vert(i, j),
                                    uv: qiyana_hair.uv(i, j),
                                    normal: qiyana_hair.normal(i, j),
                                };
                            }
                            let option_vertices = test_renderer.geometry_processing(&inputs);

                            if option_vertices.is_some() {
                                vertices.extend(option_vertices.unwrap());
                            }
                        }

                        // let start_time = Instant::now();

                        for i in 0..vertices.len() {
                            let vertex_s = &vertices[i];

                            if i == body_offset {
                                ps_uniform.place = PLACE::BODY;
                                test_renderer.set_ps_uniform(ps_uniform);
                            } else if i == face_offset {
                                ps_uniform.place = PLACE::FACE;
                                test_renderer.set_ps_uniform(ps_uniform);
                            } else if i == hair_offset {
                                ps_uniform.place = PLACE::HAIR;
                                test_renderer.set_ps_uniform(ps_uniform);
                            }

                            test_renderer.rasterization(
                                (0, WIDTH as i32),
                                (0, HEIGHT as i32),
                                vertex_s,
                                &mut frame_buffer,
                                &mut depth_buffer,
                            );
                        }

                        image_slice.copy_from_slice(&frame_buffer.bits);

                        let swap_chain_image = present_images[present_index as usize];

                        record_submit_commandbuffer(
                            &base.device,
                            base.draw_command_buffer,
                            base.draw_commands_reuse_fence,
                            base.present_queue,
                            &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
                            &[base.present_complete_semaphore],
                            &[base.rendering_complete_semaphore],
                            |device, draw_command_buffer| {
                                device.cmd_set_viewport(draw_command_buffer, 0, &view_ports);
                                device.cmd_set_scissor(draw_command_buffer, 0, &scissors);

                                let mut layout_transition_barriers =
                                    vk::ImageMemoryBarrier::builder()
                                        .image(swap_chain_image)
                                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                                        .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                                        .subresource_range(
                                            vk::ImageSubresourceRange::builder()
                                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                                .layer_count(1)
                                                .level_count(1)
                                                .build(),
                                        )
                                        .build();

                                device.cmd_pipeline_barrier(
                                    draw_command_buffer,
                                    vk::PipelineStageFlags::TRANSFER,
                                    vk::PipelineStageFlags::TRANSFER,
                                    vk::DependencyFlags::empty(),
                                    &[],
                                    &[],
                                    &[layout_transition_barriers],
                                );

                                device.cmd_copy_buffer_to_image(
                                    draw_command_buffer,
                                    image_buffer,
                                    swap_chain_image,
                                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                                    &[buffer_region],
                                );

                                layout_transition_barriers.new_layout =
                                    vk::ImageLayout::PRESENT_SRC_KHR;
                                layout_transition_barriers.old_layout =
                                    vk::ImageLayout::TRANSFER_DST_OPTIMAL;

                                device.cmd_pipeline_barrier(
                                    draw_command_buffer,
                                    vk::PipelineStageFlags::TRANSFER,
                                    vk::PipelineStageFlags::TRANSFER,
                                    vk::DependencyFlags::empty(),
                                    &[],
                                    &[],
                                    &[layout_transition_barriers],
                                );
                            },
                        );
                        //let mut present_info_err = mem::zeroed();
                        let wait_semaphors = [base.rendering_complete_semaphore];
                        let swapchains = [base.swapchain];
                        let image_indices = [present_index];
                        let present_info = vk::PresentInfoKHR::builder()
                            .wait_semaphores(&wait_semaphors) // &base.rendering_complete_semaphore)
                            .swapchains(&swapchains)
                            .image_indices(&image_indices)
                            .build();

                        base.swapchain_loader
                            .queue_present(base.present_queue, &present_info)
                            .unwrap();

                        let elapsed_time = start_time.elapsed();
                        println!("fps: {}", 1.0 / elapsed_time.as_secs_f32());
                    }
                    _ => (),
                }
            });

        base.device.device_wait_idle().unwrap();
        base.device.free_memory(image_buffer_memory, None);
        base.device.destroy_buffer(image_buffer, None);
    }
}
