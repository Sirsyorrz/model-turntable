use std::{collections::HashMap, path::Path};

use anyhow::{Context, Result, anyhow};
use base64::Engine;
use glam::{Mat3, Mat4, Vec3};
use gltf::{buffer, image::Source};

use super::{
    AlphaMode, Material, Mesh, Model, Texture, Vertex, compute_smooth_normals, flat_normal,
};
use crate::settings::{Shading, UpAxis};

pub fn load(path: &Path) -> Result<Model> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let gltf::Gltf { document, blob } = gltf::Gltf::from_slice(&bytes).context("parsing glTF")?;
    let base = path.parent().unwrap_or(Path::new("."));
    let buffers =
        gltf::import_buffers(&document, Some(base), blob).context("loading glTF buffers")?;

    let mut loader = Loader {
        base,
        buffers: &buffers,
        images: document.images().collect(),
        textures: Vec::new(),
        texture_cache: HashMap::new(),
        warnings: Vec::new(),
    };

    let materials: Vec<Material> = document.materials().map(|m| loader.material(&m)).collect();
    let default_material = materials.len();

    let mut meshes = Vec::new();
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next());
    match scene {
        Some(scene) => {
            for node in scene.nodes() {
                visit_node(
                    &node,
                    Mat4::IDENTITY,
                    &buffers,
                    default_material,
                    &mut meshes,
                );
            }
        }
        None => {
            for mesh in document.meshes() {
                push_mesh(
                    &mesh,
                    Mat4::IDENTITY,
                    &buffers,
                    default_material,
                    &mut meshes,
                );
            }
        }
    }

    let mut materials = materials;
    materials.push(Material::default());

    Ok(Model {
        path: path.to_path_buf(),
        meshes,
        materials,
        textures: loader.textures,
        native_up: UpAxis::Y,
        default_shading: Shading::Smooth,
        warnings: loader.warnings,
    })
}

fn visit_node(
    node: &gltf::Node,
    parent: Mat4,
    buffers: &[buffer::Data],
    default_material: usize,
    out: &mut Vec<Mesh>,
) {
    let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        push_mesh(&mesh, world, buffers, default_material, out);
    }
    for child in node.children() {
        visit_node(&child, world, buffers, default_material, out);
    }
}

fn push_mesh(
    mesh: &gltf::Mesh,
    world: Mat4,
    buffers: &[buffer::Data],
    default_material: usize,
    out: &mut Vec<Mesh>,
) {
    let normal_mat = Mat3::from_mat4(world).inverse().transpose();
    for prim in mesh.primitives() {
        if prim.mode() != gltf::mesh::Mode::Triangles {
            continue;
        }
        let reader = prim.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
        let Some(positions) = reader.read_positions() else {
            continue;
        };
        let positions: Vec<Vec3> = positions
            .map(|p| world.transform_point3(Vec3::from(p)))
            .collect();
        let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|n| n.collect());
        let uvs: Option<Vec<[f32; 2]>> = reader.read_tex_coords(0).map(|t| t.into_f32().collect());
        let colors: Option<Vec<[f32; 4]>> =
            reader.read_colors(0).map(|c| c.into_rgba_f32().collect());
        let indices: Vec<u32> = match reader.read_indices() {
            Some(i) => i.into_u32().collect(),
            None => (0..positions.len() as u32).collect(),
        };

        let mut vertices: Vec<Vertex> = positions
            .iter()
            .enumerate()
            .map(|(i, p)| Vertex {
                pos: p.to_array(),
                normal: normals
                    .as_ref()
                    .and_then(|n| n.get(i))
                    .map(|n| {
                        (normal_mat * Vec3::from(*n))
                            .normalize_or(Vec3::Y)
                            .to_array()
                    })
                    .unwrap_or([0.0; 3]),
                uv: uvs
                    .as_ref()
                    .and_then(|t| t.get(i))
                    .copied()
                    .unwrap_or([0.0; 2]),
                color: colors
                    .as_ref()
                    .and_then(|c| c.get(i))
                    .copied()
                    .unwrap_or([1.0; 4]),
            })
            .collect();
        let indices: Vec<u32> = indices
            .into_iter()
            .filter(|&i| (i as usize) < vertices.len())
            .collect();
        let indices = indices[..indices.len() / 3 * 3].to_vec();

        if normals.is_none() {
            // Spec says missing normals mean flat shading.
            let mut flat = Vec::with_capacity(indices.len());
            for tri in indices.as_chunks::<3>().0 {
                let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| vertices[i as usize]);
                let n = flat_normal(a.pos.into(), b.pos.into(), c.pos.into()).to_array();
                for mut v in [a, b, c] {
                    v.normal = n;
                    flat.push(v);
                }
            }
            vertices = flat;
            let indices = (0..vertices.len() as u32).collect();
            out.push(Mesh {
                vertices,
                indices,
                material: prim.material().index().unwrap_or(default_material),
            });
            continue;
        }
        if vertices.iter().any(|v| v.normal == [0.0; 3]) {
            compute_smooth_normals(&mut vertices, &indices);
        }
        out.push(Mesh {
            vertices,
            indices,
            material: prim.material().index().unwrap_or(default_material),
        });
    }
}

struct Loader<'a> {
    base: &'a Path,
    buffers: &'a [buffer::Data],
    images: Vec<gltf::Image<'a>>,
    textures: Vec<Texture>,
    texture_cache: HashMap<(usize, bool), Option<usize>>,
    warnings: Vec<String>,
}

impl Loader<'_> {
    fn material(&mut self, m: &gltf::Material) -> Material {
        let pbr = m.pbr_metallic_roughness();
        let emissive_strength = m.emissive_strength().unwrap_or(1.0);
        Material {
            base_color: pbr.base_color_factor(),
            base_color_tex: pbr
                .base_color_texture()
                .and_then(|t| self.texture(t.texture(), true)),
            metallic: pbr.metallic_factor(),
            roughness: pbr.roughness_factor(),
            metal_rough_tex: pbr
                .metallic_roughness_texture()
                .and_then(|t| self.texture(t.texture(), false)),
            normal_tex: m
                .normal_texture()
                .and_then(|t| self.texture(t.texture(), false)),
            normal_scale: m.normal_texture().map(|t| t.scale()).unwrap_or(1.0),
            occlusion_tex: m
                .occlusion_texture()
                .and_then(|t| self.texture(t.texture(), false)),
            occlusion_strength: m.occlusion_texture().map(|t| t.strength()).unwrap_or(1.0),
            emissive: m.emissive_factor().map(|c| c * emissive_strength),
            emissive_tex: m
                .emissive_texture()
                .and_then(|t| self.texture(t.texture(), true)),
            alpha_mode: match m.alpha_mode() {
                gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                gltf::material::AlphaMode::Mask => AlphaMode::Mask(m.alpha_cutoff().unwrap_or(0.5)),
                gltf::material::AlphaMode::Blend => AlphaMode::Blend,
            },
            unlit: m.unlit(),
        }
    }

    fn texture(&mut self, tex: gltf::Texture, srgb: bool) -> Option<usize> {
        let image_index = tex.source().index();
        if let Some(cached) = self.texture_cache.get(&(image_index, srgb)) {
            return *cached;
        }
        let result = match self.decode(image_index) {
            Ok(img) => {
                self.textures.push(Texture::from_image(img, srgb));
                Some(self.textures.len() - 1)
            }
            Err(e) => {
                let name = self.images[image_index]
                    .name()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("image {image_index}"));
                self.warnings.push(format!("Texture {name}: {e:#}"));
                None
            }
        };
        self.texture_cache.insert((image_index, srgb), result);
        result
    }

    fn decode(&self, image_index: usize) -> Result<image::DynamicImage> {
        let bytes: Vec<u8> = match self.images[image_index].source() {
            Source::View { view, .. } => {
                let data = &self.buffers[view.buffer().index()].0;
                data[view.offset()..view.offset() + view.length()].to_vec()
            }
            Source::Uri { uri, .. } => {
                if let Some(rest) = uri.strip_prefix("data:") {
                    let (_, b64) = rest
                        .split_once(";base64,")
                        .ok_or_else(|| anyhow!("unsupported data URI"))?;
                    base64::engine::general_purpose::STANDARD.decode(b64)?
                } else {
                    let path = self.base.join(percent_decode(uri));
                    std::fs::read(&path).with_context(|| format!("missing {}", path.display()))?
                }
            }
        };
        Ok(image::load_from_memory(&bytes)?)
    }
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Ok(hex) = std::str::from_utf8(&bytes[i + 1..i + 3])
            && let Ok(v) = u8::from_str_radix(hex, 16)
        {
            out.push(v);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}
