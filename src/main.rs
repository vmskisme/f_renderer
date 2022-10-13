mod camera;
mod input_controller;
mod matrix_util;
mod renderer;
mod vulkan_base;

use camera::Camera;
use glam::{IVec2, Mat4, UVec2, Vec2, Vec3, Vec4};
use matrix_util::{mul_vec4, set_identity, set_look_at, set_perspective, set_scale};
use renderer::{u8_array_to_vec4, vec4_to_u8_array, FrameBuffer, Renderer, ShaderContext};

use ash::util::*;
use ash::vk;
pub use ash::{Device, Instance};
use std::default::Default;
use std::f32::consts::PI;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use vulkan_base::{find_memorytype_index, record_submit_commandbuffer, VulkanBase};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::{
    event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use crate::matrix_util::set_rotate;

fn main() {
    const WIDTH: u32 = 1920;
    const HEIGHT: u32 = 1080;

    #[derive(Clone, Copy)]
    struct VSUniform {
        model: Mat4,
        view: Mat4,
        proj: Mat4,
    }

    struct PSUniform {}

    struct VSInput {
        pos: Vec3,
        color: Vec4,
    }

    const COLOR: u32 = 1;
    fn vertex_shader(
        vs_uniform: &VSUniform,
        vs_input: &VSInput,
        context: &mut ShaderContext,
    ) -> Vec4 {
        context.varying_vec4.insert(COLOR, vs_input.color);
        let mvp = vs_uniform.proj * vs_uniform.view * vs_uniform.model;
        mul_vec4(mvp, Vec4::from((vs_input.pos, 1.0)))
    }

    fn pixel_shader(ps_uniform: &PSUniform, context: &ShaderContext) -> Vec4 {
        context.varying_vec4[&COLOR]
    }

    let mat_model = set_identity();

    let mut camera_1 = Camera::new(
        Vec3::new(0.0, 0.0, 3.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    );

    let mat_proj = set_perspective(PI * 0.25, WIDTH as f32 / HEIGHT as f32, 0.1, 100.0);

    let mut vs_uniform = VSUniform {
        model: mat_model,
        view: camera_1.mat_look_at,
        proj: mat_proj,
    };
    let ps_uniform = PSUniform {};

    let mut test_renderer: Renderer<VSInput, VSUniform, PSUniform> = Renderer::new(
        WIDTH,
        HEIGHT,
        vs_uniform,
        vertex_shader,
        ps_uniform,
        pixel_shader,
    );

    let triangle = [
        VSInput {
            pos: Vec3::new(0.0, 1.0, 0.0),
            color: Vec4::new(1.0, 0.0, 0.0, 1.0),
        },
        VSInput {
            pos: Vec3::new(1.0, -1.0, 0.0),
            color: Vec4::new(0.0, 1.0, 0.0, 1.0),
        },
        VSInput {
            pos: Vec3::new(-1.0, -1.0, 0.0),
            color: Vec4::new(0.0, 0.0, 1.0, 1.0),
        },
    ];

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

        image_slice.copy_from_slice(&frame_buffer.bits);
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

        let now = SystemTime::now();

        let mut mouse_right_press = false;
        let mut cursor_pos: (f32, f32) = (0.0, 0.0);

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
                                if (2.0 < distance && y > 0.0) || (distance < 20.0 && y < 0.0){
                                    forward = forward * (distance - y * 0.2);
                                    let new_eye = forward + camera_1.at;
                                    camera_1.eye = new_eye;
                                    camera_1.set_look_at();

                                    vs_uniform.view = camera_1.mat_look_at;
                                    test_renderer.set_vs_uniform(vs_uniform);
                                }
                            }
                            PixelDelta(p) => {

                            }
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
                        event: WindowEvent::CursorMoved { .. },
                        ..
                    } => {
                        if let Event::WindowEvent { event, .. } = event {
                            if let WindowEvent::CursorMoved { position, .. } = event {
                                let x = position.x as f32;
                                let y = position.y as f32;

                                if mouse_right_press {
                                    let forward = camera_1.at - camera_1.eye;
                                    let right = forward.cross(camera_1.up);
                                    let mut forward = Vec4::from((forward, 1.0));
                                    let theta_h = (x - cursor_pos.0) * 0.005;
                                    let theta_v = -(y - cursor_pos.1) * 0.005;

                                    let rotate_horizon_mat = set_rotate(camera_1.up, theta_h * PI);
                                    let rotate_vertical_mat = set_rotate(right, theta_v * PI);

                                    forward = mul_vec4(rotate_horizon_mat, forward);
                                    forward = mul_vec4(rotate_vertical_mat, forward);
                                    let new_forward = Vec3::new(forward.x, forward.y, forward.z);
                                    camera_1.up = right.cross(new_forward).normalize();
                                    camera_1.eye =
                                        camera_1.at - Vec3::new(forward.x, forward.y, forward.z);

                                    camera_1.set_look_at();

                                    vs_uniform.view = camera_1.mat_look_at;
                                    test_renderer.set_vs_uniform(vs_uniform);
                                }

                                cursor_pos = (x, y);
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

                        frame_buffer.fill([70, 70, 70, 255]);
                        depth_buffer = vec![vec![0.0; WIDTH as usize]; HEIGHT as usize];

                        let option_vertices = test_renderer.geometry_processing(&triangle);
                        let mut vertices = vec![];
                        if option_vertices.is_some() {
                            vertices = option_vertices.unwrap();
                        }
                        
                        for vertex_s in vertices {
                            test_renderer.rasterization(
                                (0, WIDTH as i32),
                                (0, HEIGHT as i32),
                                &vertex_s,
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
