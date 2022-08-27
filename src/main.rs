mod matrix_util;
mod renderer;
mod vulkan_base;

use glam::{vec3, vec4, IVec2, Mat4, UVec2, Vec2, Vec3, Vec4};
use matrix_util::{mul_vec4, set_identity, set_look_at, set_perspective, set_scale};
use renderer::{u8_array_to_vec4, vec4_to_u8_array, FrameBuffer, Renderer, ShaderContext};

use ash::util::*;
use ash::vk;
pub use ash::{Device, Instance};
use std::default::Default;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use vulkan_base::{find_memorytype_index, record_submit_commandbuffer, VulkanBase};

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
    let mut mat_view = set_look_at(
        Vec3::new(0.0, 0.0, 3.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    );
    let mat_proj = set_perspective(3.1415926 * 0.25, WIDTH as f32 / HEIGHT as f32, 0.1, 100.0);

    let mut vs_uniform = VSUniform {
        model: mat_model,
        view: mat_view,
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

        base.render_loop(|| {
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

            let init_elapsed = now.elapsed().unwrap().as_secs_f32();
            mat_view = set_look_at(
                Vec3::new(init_elapsed.sin() * 3.0, 0.0, init_elapsed.cos() * 3.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );

            vs_uniform.view = mat_view;
            test_renderer.set_vs_uniform(vs_uniform);

            let vertices = test_renderer.geometry_processing(&triangle).unwrap();

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

                    let mut layout_transition_barriers = vk::ImageMemoryBarrier::builder()
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

                    layout_transition_barriers.new_layout = vk::ImageLayout::PRESENT_SRC_KHR;
                    layout_transition_barriers.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;

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
        });

        base.device.device_wait_idle().unwrap();
        base.device.free_memory(image_buffer_memory, None);
        base.device.destroy_buffer(image_buffer, None);
    }
}

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
fn save_file(frame_buffer: &FrameBuffer) {
    let path = Path::new("out/lorem_ipsum.bmp");
    let display = path.display();

    let with_alpha = true;

    let mut file = match File::create(&path) {
        Err(why) => panic!("couldn't create {}: {:?}", display, why),
        Ok(file) => file,
    };

    let width: u32 = frame_buffer.width;
    let height: u32 = frame_buffer.height;
    let pixel_size = if with_alpha { 3 } else { 3 } as u32;
    let pitch = (width * pixel_size + 3) & (!3);
    let info = BitMapInfoHeader {
        bi_size: 40,
        bi_width: width,
        bi_height: height,
        bi_planes: 1,
        bi_bit_count: if with_alpha { 32 } else { 24 },
        bi_compression: 0,
        bi_size_image: pitch * height,
        bi_x_pels_per_meter: 0xb12,
        bi_y_pels_per_meter: 0xb12,
        bi_clr_used: 0,
        bi_clr_important: 0,
    };

    let offset = 54 as u32;
    let bf_size = info.bi_size_image + offset;
    let zero = 0 as u32;

    file.write_all(&[0x42]).expect("write error");
    file.write_all(&[0x4d]).expect("write error");
    file.write_all(&bf_size.to_le_bytes()).expect("write error");
    file.write_all(&zero.to_le_bytes()).expect("write error");
    file.write_all(&offset.to_le_bytes()).expect("write error");

    file.write_all(&info.bi_size.to_le_bytes())
        .expect("write error");
    file.write_all(&info.bi_width.to_le_bytes())
        .expect("write error");
    file.write_all(&info.bi_height.to_le_bytes())
        .expect("write error");
    file.write_all(&info.bi_planes.to_le_bytes())
        .expect("write error");
    file.write_all(&info.bi_bit_count.to_le_bytes())
        .expect("write error");
    file.write_all(&info.bi_compression.to_le_bytes())
        .expect("write error");
    file.write_all(&info.bi_size_image.to_le_bytes())
        .expect("write error");
    file.write_all(&info.bi_x_pels_per_meter.to_le_bytes())
        .expect("write error");
    file.write_all(&info.bi_y_pels_per_meter.to_le_bytes())
        .expect("write error");
    file.write_all(&info.bi_clr_used.to_le_bytes())
        .expect("write error");
    file.write_all(&info.bi_clr_important.to_le_bytes())
        .expect("write error");

    let padding: u32 = pitch - width * pixel_size;

    for y in 0..height {
        for x in 0..width {
            let color = frame_buffer.get_pixel(x, y);
            if with_alpha {
                file.write_all(&[color[2], color[1], color[0], color[3]])
                    .expect("write error");
            } else {
                file.write_all(&[color[2], color[1], color[0]])
                    .expect("write error");
            }
        }
        let mut i = 0;
        while i < padding {
            file.write_all(&[0 as u8; 16]).expect("write error");
            i += 16;
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
