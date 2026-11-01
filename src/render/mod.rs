pub mod env;
pub mod scene;

use std::sync::Arc;

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::model::{AlphaMode, Model, Vertex};
use env::{HdrImage, MATCAP_SIZE};
pub use scene::FrameUniforms;

pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DISPLAY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const MSAA: u32 = 4;
const ENV_LEVELS: u32 = 6;
const SHADOW_SIZE: u32 = 256;

#[derive(Clone)]
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
}

impl Gpu {
    pub fn headless() -> Result<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .context("no GPU adapter found")?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("model-turntable"),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .context("creating GPU device")?;
        Ok(Gpu {
            device,
            queue,
            adapter_name: adapter.get_info().name,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialUniforms {
    base_color: [f32; 4],
    emissive: [f32; 4],
    params: [f32; 4],
    flags: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadowUniforms {
    proj: Mat4,
    local: Mat4,
    p: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrefilterUniforms {
    p: [f32; 4],
}

pub struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    material: usize,
    blend: bool,
    centroid: Vec3,
}

pub struct GpuModel {
    meshes: Vec<GpuMesh>,
    materials: Vec<wgpu::BindGroup>,
}

pub struct Environment {
    view: wgpu::TextureView,
    pub sh: [Vec3; 9],
    pub max_lod: f32,
}

pub struct Matcap {
    view: wgpu::TextureView,
}

pub struct GroundShadow {
    view: wgpu::TextureView,
}

pub struct Renderer {
    pub gpu: Gpu,
    frame_bgl: wgpu::BindGroupLayout,
    material_bgl: wgpu::BindGroupLayout,
    opaque: wgpu::RenderPipeline,
    blend: wgpu::RenderPipeline,
    background: wgpu::RenderPipeline,
    ground: wgpu::RenderPipeline,
    shadow_bgl: wgpu::BindGroupLayout,
    shadow: wgpu::RenderPipeline,
    down_bgl: wgpu::BindGroupLayout,
    down: wgpu::RenderPipeline,
    prefilter_bgl: wgpu::BindGroupLayout,
    prefilter: wgpu::RenderPipeline,
    env_sampler: wgpu::Sampler,
    clamp_sampler: wgpu::Sampler,
    material_sampler: wgpu::Sampler,
    white_srgb: wgpu::TextureView,
    white_linear: wgpu::TextureView,
    flat_normal: wgpu::TextureView,
    black: wgpu::TextureView,
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

const VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<Vertex>() as u64,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4],
};

const POSITION_ONLY_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<Vertex>() as u64,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
};

const PREMULTIPLIED: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
};

impl Renderer {
    pub fn new(gpu: Gpu) -> Renderer {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let util = device.create_shader_module(wgpu::include_wgsl!("util.wgsl"));
        let prefilter_shader = device.create_shader_module(wgpu::include_wgsl!("prefilter.wgsl"));

        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                texture_entry(1, true),
                sampler_entry(2),
                texture_entry(3, true),
                texture_entry(4, true),
                sampler_entry(5),
            ],
        });
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                texture_entry(1, true),
                texture_entry(2, true),
                texture_entry(3, true),
                texture_entry(4, true),
                texture_entry(5, true),
                sampler_entry(6),
            ],
        });
        let main_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("main"),
            bind_group_layouts: &[Some(&frame_bgl), Some(&material_bgl)],
            immediate_size: 0,
        });
        let frame_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("frame only"),
            bind_group_layouts: &[Some(&frame_bgl)],
            immediate_size: 0,
        });

        let msaa = wgpu::MultisampleState {
            count: MSAA,
            mask: !0,
            alpha_to_coverage_enabled: false,
        };
        let depth = |write: bool, compare: wgpu::CompareFunction| {
            Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(write),
                depth_compare: Some(compare),
                stencil: Default::default(),
                bias: Default::default(),
            })
        };
        let make = |label: &str,
                    layout: &wgpu::PipelineLayout,
                    vs: &str,
                    fs: &str,
                    buffers: &[Option<wgpu::VertexBufferLayout>],
                    blend: Option<wgpu::BlendState>,
                    depth_stencil: Option<wgpu::DepthStencilState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil,
                multisample: msaa,
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: COLOR_FORMAT,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let opaque = make(
            "opaque",
            &main_layout,
            "vs_main",
            "fs_main",
            &[Some(VERTEX_LAYOUT)],
            None,
            depth(true, wgpu::CompareFunction::Less),
        );
        let blend = make(
            "blend",
            &main_layout,
            "vs_main",
            "fs_main",
            &[Some(VERTEX_LAYOUT)],
            Some(PREMULTIPLIED),
            depth(false, wgpu::CompareFunction::Less),
        );
        let background = make(
            "background",
            &frame_layout,
            "vs_fullscreen",
            "fs_background",
            &[],
            None,
            depth(false, wgpu::CompareFunction::Always),
        );
        let ground = make(
            "ground",
            &frame_layout,
            "vs_ground",
            "fs_ground",
            &[],
            Some(PREMULTIPLIED),
            depth(false, wgpu::CompareFunction::LessEqual),
        );

        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow"),
            entries: &[uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT)],
        });
        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow"),
            bind_group_layouts: &[Some(&shadow_bgl)],
            immediate_size: 0,
        });
        let max_blend = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Max,
        };
        let shadow = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow bake"),
            layout: Some(&shadow_layout),
            vertex: wgpu::VertexState {
                module: &util,
                entry_point: Some("vs_shadow"),
                compilation_options: Default::default(),
                buffers: &[Some(POSITION_ONLY_LAYOUT)],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &util,
                entry_point: Some("fs_shadow"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState {
                        color: max_blend,
                        alpha: max_blend,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let down_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("downsample"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                texture_entry(1, false),
            ],
        });
        let down_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("downsample"),
            bind_group_layouts: &[Some(&down_bgl)],
            immediate_size: 0,
        });
        let down = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("downsample"),
            layout: Some(&down_layout),
            vertex: wgpu::VertexState {
                module: &util,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &util,
                entry_point: Some("fs_downsample"),
                compilation_options: Default::default(),
                targets: &[Some(COLOR_FORMAT.into())],
            }),
            multiview_mask: None,
            cache: None,
        });

        let prefilter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prefilter"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                texture_entry(1, true),
                sampler_entry(2),
            ],
        });
        let prefilter_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prefilter"),
            bind_group_layouts: &[Some(&prefilter_bgl)],
            immediate_size: 0,
        });
        let prefilter = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("prefilter"),
            layout: Some(&prefilter_layout),
            vertex: wgpu::VertexState {
                module: &prefilter_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &prefilter_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(HDR_FORMAT.into())],
            }),
            multiview_mask: None,
            cache: None,
        });

        let env_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("env"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let clamp_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let material_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        });

        let solid = |rgba: [u8; 4], format: wgpu::TextureFormat| {
            device
                .create_texture_with_data(
                    &gpu.queue,
                    &wgpu::TextureDescriptor {
                        label: Some("solid"),
                        size: wgpu::Extent3d {
                            width: 1,
                            height: 1,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    &rgba,
                )
                .create_view(&Default::default())
        };
        let white_srgb = solid([255; 4], wgpu::TextureFormat::Rgba8UnormSrgb);
        let white_linear = solid([255; 4], wgpu::TextureFormat::Rgba8Unorm);
        let flat_normal = solid([128, 128, 255, 255], wgpu::TextureFormat::Rgba8Unorm);
        let black = solid([0, 0, 0, 255], wgpu::TextureFormat::Rgba8Unorm);

        Renderer {
            gpu,
            frame_bgl,
            material_bgl,
            opaque,
            blend,
            background,
            ground,
            shadow_bgl,
            shadow,
            down_bgl,
            down,
            prefilter_bgl,
            prefilter,
            env_sampler,
            clamp_sampler,
            material_sampler,
            white_srgb,
            white_linear,
            flat_normal,
            black,
        }
    }

    fn device(&self) -> &wgpu::Device {
        &self.gpu.device
    }

    pub fn upload_model(&self, model: &Model) -> GpuModel {
        let device = self.device();
        let textures: Vec<wgpu::TextureView> = model
            .textures
            .iter()
            .map(|t| self.upload_texture(t).create_view(&Default::default()))
            .collect();

        let materials = model
            .materials
            .iter()
            .map(|m| {
                let u = MaterialUniforms {
                    base_color: m.base_color,
                    emissive: [
                        m.emissive[0],
                        m.emissive[1],
                        m.emissive[2],
                        m.occlusion_strength,
                    ],
                    params: [
                        m.metallic,
                        m.roughness,
                        m.normal_scale,
                        match m.alpha_mode {
                            AlphaMode::Mask(c) => c,
                            _ => 0.5,
                        },
                    ],
                    flags: [
                        match m.alpha_mode {
                            AlphaMode::Opaque => 0.0,
                            AlphaMode::Mask(_) => 1.0,
                            AlphaMode::Blend => 2.0,
                        },
                        m.normal_tex.is_some() as u32 as f32,
                        m.unlit as u32 as f32,
                        0.0,
                    ],
                };
                let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("material"),
                    contents: bytemuck::bytes_of(&u),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let tex = |i: Option<usize>, fallback: &wgpu::TextureView| -> wgpu::TextureView {
                    i.and_then(|i| textures.get(i)).unwrap_or(fallback).clone()
                };
                let views = [
                    tex(m.base_color_tex, &self.white_srgb),
                    tex(m.metal_rough_tex, &self.white_linear),
                    tex(m.normal_tex, &self.flat_normal),
                    tex(m.occlusion_tex, &self.white_linear),
                    tex(m.emissive_tex, &self.white_srgb),
                ];
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("material"),
                    layout: &self.material_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&views[0]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&views[1]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&views[2]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&views[3]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&views[4]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::Sampler(&self.material_sampler),
                        },
                    ],
                })
            })
            .collect();

        let meshes = model
            .meshes
            .iter()
            .map(|m| {
                let centroid = m
                    .vertices
                    .iter()
                    .fold(Vec3::ZERO, |a, v| a + Vec3::from(v.pos))
                    / m.vertices.len().max(1) as f32;
                GpuMesh {
                    vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("vertices"),
                        contents: bytemuck::cast_slice(&m.vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                    indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("indices"),
                        contents: bytemuck::cast_slice(&m.indices),
                        usage: wgpu::BufferUsages::INDEX,
                    }),
                    index_count: m.indices.len() as u32,
                    material: m.material,
                    blend: model.materials[m.material].alpha_mode == AlphaMode::Blend,
                    centroid,
                }
            })
            .collect();

        GpuModel { meshes, materials }
    }

    fn upload_texture(&self, t: &crate::model::Texture) -> wgpu::Texture {
        let max_dim = self.device().limits().max_texture_dimension_2d;
        let mut levels = vec![(t.width, t.height, t.rgba.clone())];
        while levels[0].0 > max_dim || levels[0].1 > max_dim {
            let (w, h, d) = &levels[0];
            levels[0] = downsample_rgba(*w, *h, d);
        }
        loop {
            let (w, h, d) = levels.last().unwrap();
            if *w == 1 && *h == 1 {
                break;
            }
            let next = downsample_rgba(*w, *h, d);
            levels.push(next);
        }
        let format = if t.srgb {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
        let data: Vec<u8> = levels.iter().flat_map(|l| l.2.iter().copied()).collect();
        self.device().create_texture_with_data(
            &self.gpu.queue,
            &wgpu::TextureDescriptor {
                label: Some("material texture"),
                size: wgpu::Extent3d {
                    width: levels[0].0,
                    height: levels[0].1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: levels.len() as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &data,
        )
    }

    pub fn create_environment(&self, img: &HdrImage) -> Environment {
        let device = self.device();
        let mips = img.mip_chain();
        let src_data: Vec<u8> = mips
            .iter()
            .flat_map(|(_, _, d)| bytemuck::cast_slice::<half::f16, u8>(d).iter().copied())
            .collect();
        let src = device.create_texture_with_data(
            &self.gpu.queue,
            &wgpu::TextureDescriptor {
                label: Some("hdri source"),
                size: wgpu::Extent3d {
                    width: img.width,
                    height: img.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: mips.len() as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &src_data,
        );
        let src_view = src.create_view(&Default::default());

        let levels = ENV_LEVELS.min(mips.len() as u32);
        let filtered = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdri prefiltered"),
            size: wgpu::Extent3d {
                width: img.width,
                height: img.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        for level in 0..levels {
            let roughness = level as f32 / (levels - 1).max(1) as f32;
            let params = PrefilterUniforms {
                p: [
                    roughness,
                    img.width as f32,
                    img.height as f32,
                    mips.len() as f32,
                ],
            };
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("prefilter params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prefilter"),
                layout: &self.prefilter_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                    },
                ],
            });
            let view = filtered.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("prefilter"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.prefilter);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        self.gpu.queue.submit([encoder.finish()]);

        Environment {
            view: filtered.create_view(&Default::default()),
            sh: img.sh9(),
            max_lod: (levels - 1) as f32,
        }
    }

    pub fn create_matcap(&self, data: &[half::f16]) -> Matcap {
        let tex = self.device().create_texture_with_data(
            &self.gpu.queue,
            &wgpu::TextureDescriptor {
                label: Some("matcap"),
                size: wgpu::Extent3d {
                    width: MATCAP_SIZE,
                    height: MATCAP_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(data),
        );
        Matcap {
            view: tex.create_view(&Default::default()),
        }
    }

    pub fn empty_shadow(&self) -> GroundShadow {
        GroundShadow {
            view: self.black.clone(),
        }
    }

    /// Top-down height-falloff mask of the model, blurred into a soft contact shadow.
    /// Lives in the model's local XZ plane so it spins with the model.
    pub fn bake_shadow(
        &self,
        model: &GpuModel,
        local: Mat4,
        bounds: &scene::Framing,
    ) -> GroundShadow {
        let device = self.device();
        let ext = bounds.shadow_extent;
        let height = (bounds.y_max - bounds.y_min).max(1e-4);
        // Looking down -Y: local X -> clip X, local Z -> clip Y (flipped so texture v follows +Z).
        let view = glam::camera::rh::view::look_at_mat4(
            Vec3::new(0.0, bounds.y_max + 1.0, 0.0),
            Vec3::new(0.0, bounds.y_min, 0.0),
            Vec3::NEG_Z,
        );
        let proj =
            glam::camera::rh::proj::directx::orthographic(-ext, ext, -ext, ext, 0.0, height + 2.0);
        let u = ShadowUniforms {
            proj: proj * view,
            local,
            p: [bounds.y_min, height * 0.15, 0.0, 0.0],
        };
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shadow params"),
            contents: bytemuck::bytes_of(&u),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow"),
            layout: &self.shadow_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow bake"),
            size: wgpu::Extent3d {
                width: SHADOW_SIZE,
                height: SHADOW_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view_t = target.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow bake"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_t,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.shadow);
            pass.set_bind_group(0, &bind, &[]);
            for m in &model.meshes {
                pass.set_vertex_buffer(0, m.vertices.slice(..));
                pass.set_index_buffer(m.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..m.index_count, 0, 0..1);
            }
        }
        let pixels = self.read_texture(encoder, &target, SHADOW_SIZE, SHADOW_SIZE);
        let mask: Vec<f32> = pixels
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| p[0] as f32 / 255.0)
            .collect();

        let n = SHADOW_SIZE as usize;
        let tight = blur(&mask, n, 3);
        let wide = blur(&mask, n, 22);
        let out: Vec<u8> = tight
            .iter()
            .zip(&wide)
            .enumerate()
            .flat_map(|(i, (t, w))| {
                // Fade out towards the edge of the quad so it never shows a hard border.
                let x = (i % n) as f32 / n as f32 * 2.0 - 1.0;
                let y = (i / n) as f32 / n as f32 * 2.0 - 1.0;
                let edge = (1.0 - (x * x + y * y).sqrt()).clamp(0.0, 0.25) * 4.0;
                let v = ((t * 0.5 + w * 0.95).min(1.0) * edge * 255.0) as u8;
                [v, v, v, 255]
            })
            .collect();
        let tex = device.create_texture_with_data(
            &self.gpu.queue,
            &wgpu::TextureDescriptor {
                label: Some("ground shadow"),
                size: wgpu::Extent3d {
                    width: SHADOW_SIZE,
                    height: SHADOW_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &out,
        );
        GroundShadow {
            view: tex.create_view(&Default::default()),
        }
    }

    fn read_texture(
        &self,
        mut encoder: wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let device = self.device();
        let row = width * 4;
        let padded =
            row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.gpu.queue.submit([encoder.finish()]);
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        if !matches!(rx.recv(), Ok(Ok(()))) {
            return vec![0; (row * height) as usize];
        }
        let data = slice.get_mapped_range().expect("mapped readback");
        let mut out = Vec::with_capacity((row * height) as usize);
        for y in 0..height as usize {
            let start = y * padded as usize;
            out.extend_from_slice(&data[start..start + row as usize]);
        }
        drop(data);
        buffer.unmap();
        out
    }

    pub fn render(
        &self,
        target: &RenderTarget,
        model: &GpuModel,
        res: &FrameResources,
        uniforms: &FrameUniforms,
        draw_background: bool,
        draw_ground: bool,
    ) {
        let device = self.device();
        let queue = &self.gpu.queue;
        queue.write_buffer(&target.frame_buffer, 0, bytemuck::bytes_of(uniforms));
        let frame_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame"),
            layout: &self.frame_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: target.frame_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&res.env.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&res.matcap.view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&res.shadow.view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.clamp_sampler),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.msaa_view,
                    depth_slice: None,
                    resolve_target: Some(&target.resolve_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_bind_group(0, &frame_bind, &[]);
            if draw_background {
                pass.set_pipeline(&self.background);
                pass.draw(0..3, 0..1);
            }
            if draw_ground {
                pass.set_pipeline(&self.ground);
                pass.draw(0..6, 0..1);
            }
            pass.set_pipeline(&self.opaque);
            for m in model.meshes.iter().filter(|m| !m.blend) {
                pass.set_bind_group(1, &model.materials[m.material], &[]);
                pass.set_vertex_buffer(0, m.vertices.slice(..));
                pass.set_index_buffer(m.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..m.index_count, 0, 0..1);
            }
            let mut blended: Vec<&GpuMesh> = model.meshes.iter().filter(|m| m.blend).collect();
            if !blended.is_empty() {
                let cam = Vec3::from_slice(&uniforms.camera_pos[..3]);
                let dist = |m: &GpuMesh| {
                    uniforms
                        .model
                        .transform_point3(m.centroid)
                        .distance_squared(cam)
                };
                blended.sort_by(|a, b| dist(b).total_cmp(&dist(a)));
                pass.set_pipeline(&self.blend);
                for m in blended {
                    pass.set_bind_group(1, &model.materials[m.material], &[]);
                    pass.set_vertex_buffer(0, m.vertices.slice(..));
                    pass.set_index_buffer(m.indices.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..m.index_count, 0, 0..1);
                }
            }
        }
        if let Some(out) = &target.output {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("downsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &out.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.down);
            pass.set_bind_group(0, &out.bind, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
    }

    /// Premultiplied RGBA8 (sRGB) pixels of the final image.
    pub fn read_target(&self, target: &RenderTarget) -> Vec<u8> {
        let encoder = self.device().create_command_encoder(&Default::default());
        self.read_texture(encoder, target.final_texture(), target.width, target.height)
    }

    pub fn create_target(&self, width: u32, height: u32, supersample: u32) -> RenderTarget {
        let device = self.device();
        let max = device.limits().max_texture_dimension_2d;
        let mut ss = supersample.max(1);
        while ss > 1 && (width * ss > max || height * ss > max) {
            ss -= 1;
        }
        let (rw, rh) = (width * ss, height * ss);
        let size = wgpu::Extent3d {
            width: rw,
            height: rh,
            depth_or_array_layers: 1,
        };
        let tex = |label, format, samples, usage, view_formats| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats,
            })
        };
        let msaa = tex(
            "msaa color",
            COLOR_FORMAT,
            MSAA,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            &[],
        );
        let depth = tex(
            "depth",
            DEPTH_FORMAT,
            MSAA,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            &[],
        );
        let resolve = tex(
            "resolve",
            COLOR_FORMAT,
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            &[DISPLAY_FORMAT],
        );
        let resolve_view = resolve.create_view(&Default::default());
        let output = (ss > 1).then(|| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("output"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: COLOR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[DISPLAY_FORMAT],
            });
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("downsample params"),
                contents: bytemuck::cast_slice(&[ss, 0, 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("downsample"),
                layout: &self.down_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&resolve_view),
                    },
                ],
            });
            OutputStage {
                view: texture.create_view(&Default::default()),
                texture,
                bind,
            }
        });
        let display_view = output
            .as_ref()
            .map(|o| &o.texture)
            .unwrap_or(&resolve)
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(DISPLAY_FORMAT),
                ..Default::default()
            });
        RenderTarget {
            width,
            display_view,
            height,
            msaa_view: msaa.create_view(&Default::default()),
            depth_view: depth.create_view(&Default::default()),
            resolve_view,
            resolve,
            output,
            frame_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frame uniforms"),
                size: std::mem::size_of::<FrameUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }
}

pub struct FrameResources {
    pub env: Arc<Environment>,
    pub matcap: Arc<Matcap>,
    pub shadow: Arc<GroundShadow>,
}

struct OutputStage {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind: wgpu::BindGroup,
}

pub struct RenderTarget {
    pub width: u32,
    pub height: u32,
    display_view: wgpu::TextureView,
    msaa_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    resolve: wgpu::Texture,
    resolve_view: wgpu::TextureView,
    output: Option<OutputStage>,
    frame_buffer: wgpu::Buffer,
}

impl RenderTarget {
    pub fn final_texture(&self) -> &wgpu::Texture {
        self.output
            .as_ref()
            .map(|o| &o.texture)
            .unwrap_or(&self.resolve)
    }

    /// The final image viewed as plain unorm, so egui (which composites in
    /// gamma space) receives the sRGB-encoded bytes without a decode.
    pub fn display_view(&self) -> &wgpu::TextureView {
        &self.display_view
    }
}

fn downsample_rgba(w: u32, h: u32, data: &[u8]) -> (u32, u32, Vec<u8>) {
    let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
    let mut out = vec![0u8; (nw * nh * 4) as usize];
    for y in 0..nh {
        for x in 0..nw {
            for c in 0..4 {
                let mut s = 0u32;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let sx = (x * 2 + dx).min(w - 1);
                    let sy = (y * 2 + dy).min(h - 1);
                    s += data[((sy * w + sx) * 4 + c) as usize] as u32;
                }
                out[((y * nw + x) * 4 + c) as usize] = ((s + 2) / 4) as u8;
            }
        }
    }
    (nw, nh, out)
}

/// Separable box blur, three passes approximates a gaussian.
fn blur(src: &[f32], n: usize, radius: usize) -> Vec<f32> {
    let mut a = src.to_vec();
    let mut b = vec![0.0; src.len()];
    let r = radius as isize;
    for _ in 0..3 {
        for horizontal in [true, false] {
            for y in 0..n {
                for x in 0..n {
                    let mut s = 0.0;
                    for k in -r..=r {
                        let (sx, sy) = if horizontal {
                            ((x as isize + k).clamp(0, n as isize - 1) as usize, y)
                        } else {
                            (x, (y as isize + k).clamp(0, n as isize - 1) as usize)
                        };
                        s += a[sy * n + sx];
                    }
                    b[y * n + x] = s / (2 * r + 1) as f32;
                }
            }
            std::mem::swap(&mut a, &mut b);
        }
    }
    a
}
