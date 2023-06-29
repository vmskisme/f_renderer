use crate::buffer_vec::BufferVec;
use bytemuck;
use glam::{Vec2, Vec3, Vec4};
use wgpu::{
    include_spirv_raw, util::DeviceExt, Backend, Backends, Buffer, BufferAddress, BufferUsages,
    Device, DeviceDescriptor, Extent3d, ImageCopyBuffer, ImageCopyTexture, ImageDataLayout,
    Instance, InstanceDescriptor, Origin3d, Queue, RenderPipeline, Surface, SurfaceConfiguration,
    Texture, TextureAspect, TextureUsages,
};
use winit::{dpi::PhysicalSize, event::WindowEvent, window::Window};
use rand::{self, Rng};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Rgba {
    b: u8,
    g: u8,
    r: u8,
    a: u8,
}

impl Rgba {
    pub fn new() -> Self {
        Self {
            r: 0,
            g: 0,
            b: 0,
            a: 255,
        }
    }

    pub fn new_randomly() -> Self{
        let mut rng = rand::thread_rng();
        Self { b: rng.gen_range(0..=255), g: rng.gen_range(0..=255), r: rng.gen_range(0..=255), a: 255 }   
    }

    pub const WHITE: Rgba = Rgba{b: 255, g: 255, r: 255, a: 255};
}

pub struct WgpuRenderer {
    surface: Surface,
    device: Device,
    queue: Queue,
    surface_config: SurfaceConfiguration,
    size: PhysicalSize<u32>,
}

impl WgpuRenderer {
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    pub async fn new(window: &Window) -> Self {
        let size = window.inner_size();
        let instance = Instance::new(InstanceDescriptor::default());
        let surface = unsafe { instance.create_surface(window).unwrap() };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::SPIRV_SHADER_PASSTHROUGH,
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let adapter = instance
            .enumerate_adapters(wgpu::Backends::all())
            .filter(|adapter| {
                // Check if this adapter supports our surface
                adapter.is_surface_supported(&surface)
            })
            .next()
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_DST,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &config);

        Self {
            surface: surface,
            device: device,
            queue: queue,
            surface_config: config,
            size: size,
        }
    }

    pub fn size(&self) -> PhysicalSize<u32> {
        self.size
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    pub fn render_by_buffer(&mut self, buffer: &BufferVec<Rgba>) -> Result<(), wgpu::SurfaceError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        let output = self.surface.get_current_texture()?;

        let source = ImageCopyBuffer {
            buffer: buffer.buffer().unwrap(),
            layout: ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.size.width * 4),
                rows_per_image: Some(self.size.height),
            },
        };

        let dst_data = ImageCopyTexture {
            texture: &output.texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        };

        let copy_size = Extent3d {
            width: self.size.width,
            height: self.size.height,
            depth_or_array_layers: 1,
        };

        encoder.copy_buffer_to_texture(source, dst_data, copy_size);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}