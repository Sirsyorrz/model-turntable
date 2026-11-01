use std::{collections::HashMap, path::Path};

use anyhow::{Context, Result, anyhow};
use glam::{Mat3, Mat4, Vec3, Vec4};

use super::{AlphaMode, Material, Mesh, Model, Texture, Vertex, resolve_texture_path};
use crate::settings::{Shading, UpAxis};

/// Loads FBX and OBJ (ufbx handles both), converted to right-handed Y-up.
pub fn load(path: &Path) -> Result<Model> {
    let opts = ufbx::LoadOpts {
        target_axes: ufbx::CoordinateAxes {
            right: ufbx::CoordinateAxis::PositiveX,
            up: ufbx::CoordinateAxis::PositiveY,
            front: ufbx::CoordinateAxis::PositiveZ,
        },
        target_unit_meters: 1.0,
        space_conversion: ufbx::SpaceConversion::ModifyGeometry,
        generate_missing_normals: true,
        use_blender_pbr_material: true,
        load_external_files: true,
        ignore_missing_external_files: true,
        ignore_animation: true,
        ..Default::default()
    };
    let path_str = path.to_str().ok_or_else(|| anyhow!("non UTF-8 path"))?;
    let scene = ufbx::load_file(path_str, opts)
        .map_err(|e| anyhow!("{}", e.description))
        .with_context(|| format!("loading {}", path.display()))?;

    let mut loader = Loader {
        dir: path.parent().unwrap_or(Path::new(".")).to_path_buf(),
        textures: Vec::new(),
        texture_cache: HashMap::new(),
        materials: Vec::new(),
        material_ids: HashMap::new(),
        warnings: Vec::new(),
    };

    let mut meshes = Vec::new();
    let mut tri = Vec::new();
    for node in scene.nodes.iter() {
        let Some(mesh) = node.mesh.as_ref() else {
            continue;
        };
        let world = to_mat4(&node.geometry_to_world);
        let normal_mat = Mat3::from_mat4(world).inverse().transpose();
        let mat_list = if node.materials.is_empty() {
            &mesh.materials
        } else {
            &node.materials
        };

        let parts: Vec<(u32, Vec<u32>)> = if mesh.material_parts.is_empty() {
            vec![(0, (0..mesh.faces.len() as u32).collect())]
        } else {
            mesh.material_parts
                .iter()
                .map(|p| (p.index, p.face_indices.to_vec()))
                .collect()
        };

        for (mat_index, faces) in parts {
            let material = match mat_list.get(mat_index as usize) {
                Some(m) => loader.material(m),
                None => loader.default_material(),
            };
            let mut vertices = Vec::new();
            for fi in faces {
                let face = mesh.faces[fi as usize];
                tri.resize((mesh.max_face_triangles * 3).max(3), 0);
                let n = ufbx::triangulate_face(&mut tri, mesh, face) as usize;
                for &ix in &tri[..n * 3] {
                    let ix = ix as usize;
                    let p = mesh.vertex_position[ix];
                    let pos = world.transform_point3(Vec3::new(p.x as f32, p.y as f32, p.z as f32));
                    let normal = if mesh.vertex_normal.exists {
                        let n = mesh.vertex_normal[ix];
                        (normal_mat * Vec3::new(n.x as f32, n.y as f32, n.z as f32))
                            .normalize_or(Vec3::Y)
                    } else {
                        Vec3::Y
                    };
                    let uv = if mesh.vertex_uv.exists {
                        let t = mesh.vertex_uv[ix];
                        [t.x as f32, 1.0 - t.y as f32]
                    } else {
                        [0.0; 2]
                    };
                    vertices.push(Vertex {
                        pos: pos.to_array(),
                        normal: normal.to_array(),
                        uv,
                        color: [1.0; 4],
                    });
                }
            }
            if vertices.is_empty() {
                continue;
            }
            let indices = (0..vertices.len() as u32).collect();
            meshes.push(Mesh {
                vertices,
                indices,
                material,
            });
        }
    }

    Ok(Model {
        path: path.to_path_buf(),
        meshes,
        materials: loader.materials,
        textures: loader.textures,
        native_up: UpAxis::Y,
        default_shading: Shading::Smooth,
        warnings: loader.warnings,
    })
}

fn to_mat4(m: &ufbx::Matrix) -> Mat4 {
    Mat4::from_cols(
        Vec4::new(m.m00 as f32, m.m10 as f32, m.m20 as f32, 0.0),
        Vec4::new(m.m01 as f32, m.m11 as f32, m.m21 as f32, 0.0),
        Vec4::new(m.m02 as f32, m.m12 as f32, m.m22 as f32, 0.0),
        Vec4::new(m.m03 as f32, m.m13 as f32, m.m23 as f32, 1.0),
    )
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum TexKey {
    Single(u32, bool),
    MetalRough(Option<u32>, Option<u32>),
}

struct Loader {
    dir: std::path::PathBuf,
    textures: Vec<Texture>,
    texture_cache: HashMap<TexKey, Option<usize>>,
    materials: Vec<Material>,
    material_ids: HashMap<u32, usize>,
    warnings: Vec<String>,
}

impl Loader {
    fn default_material(&mut self) -> usize {
        *self.material_ids.entry(u32::MAX).or_insert_with(|| {
            self.materials.push(Material::default());
            self.materials.len() - 1
        })
    }

    fn material(&mut self, m: &ufbx::Material) -> usize {
        if let Some(&i) = self.material_ids.get(&m.element.element_id) {
            return i;
        }
        let pbr = &m.pbr;
        let base = vec4(&pbr.base_color);
        let base_factor = if pbr.base_factor.has_value {
            pbr.base_factor.value_vec4.x as f32
        } else {
            1.0
        };
        let base_tex = map_texture(&pbr.base_color).or_else(|| map_texture(&m.fbx.diffuse_color));
        let base_color_tex = base_tex.and_then(|t| self.single(t, true));

        let rough_tex = map_texture(&pbr.roughness);
        let metal_tex = map_texture(&pbr.metalness);
        let metal_rough_tex = if rough_tex.is_some() || metal_tex.is_some() {
            self.metal_rough(rough_tex, metal_tex)
        } else {
            None
        };

        let normal_tex = map_texture(&pbr.normal_map)
            .or_else(|| map_texture(&m.fbx.normal_map))
            .and_then(|t| self.single(t, false));
        let occlusion_tex = map_texture(&pbr.ambient_occlusion).and_then(|t| self.single(t, false));
        let emissive_tex = map_texture(&pbr.emission_color).and_then(|t| self.single(t, true));

        let emission = vec4(&pbr.emission_color);
        let emission_factor = if pbr.emission_factor.has_value {
            pbr.emission_factor.value_vec4.x as f32
        } else {
            1.0
        };
        let mut emissive = [emission.x, emission.y, emission.z].map(|c| c * emission_factor);
        if emissive_tex.is_some() && emissive == [0.0; 3] {
            emissive = [1.0; 3];
        }

        let opacity = if pbr.opacity.has_value {
            pbr.opacity.value_vec4.x as f32
        } else {
            1.0
        };
        let alpha_mode = if opacity < 0.99 {
            AlphaMode::Blend
        } else if base_color_tex.is_some_and(|t| self.textures[t].has_transparency()) {
            AlphaMode::Mask(0.5)
        } else {
            AlphaMode::Opaque
        };

        let has_rough = pbr.roughness.has_value;
        let material = Material {
            // FBX textures replace the colour rather than multiply it.
            base_color: if base_color_tex.is_some() {
                [1.0, 1.0, 1.0, opacity]
            } else {
                [
                    base.x * base_factor,
                    base.y * base_factor,
                    base.z * base_factor,
                    opacity,
                ]
            },
            base_color_tex,
            metallic: if metal_tex.is_some() {
                1.0
            } else {
                pbr.metalness.value_vec4.x as f32
            },
            roughness: if rough_tex.is_some() || !has_rough {
                if rough_tex.is_some() { 1.0 } else { 0.5 }
            } else {
                pbr.roughness.value_vec4.x as f32
            },
            metal_rough_tex,
            normal_tex,
            normal_scale: 1.0,
            occlusion_tex,
            occlusion_strength: 1.0,
            emissive,
            emissive_tex,
            alpha_mode,
            unlit: false,
        };
        self.materials.push(material);
        let i = self.materials.len() - 1;
        self.material_ids.insert(m.element.element_id, i);
        i
    }

    fn single(&mut self, tex: &ufbx::Texture, srgb: bool) -> Option<usize> {
        let key = TexKey::Single(tex.element.element_id, srgb);
        if let Some(c) = self.texture_cache.get(&key) {
            return *c;
        }
        let result = self.decode(tex).map(|img| {
            self.textures.push(Texture::from_image(img, srgb));
            self.textures.len() - 1
        });
        self.texture_cache.insert(key, result);
        result
    }

    /// FBX stores roughness and metalness as separate greyscale maps; pack them
    /// into one texture with the glTF channel layout the shader expects.
    fn metal_rough(
        &mut self,
        rough: Option<&ufbx::Texture>,
        metal: Option<&ufbx::Texture>,
    ) -> Option<usize> {
        let key = TexKey::MetalRough(
            rough.map(|t| t.element.element_id),
            metal.map(|t| t.element.element_id),
        );
        if let Some(c) = self.texture_cache.get(&key) {
            return *c;
        }
        let r = rough.and_then(|t| self.decode(t)).map(|i| i.into_luma8());
        let m = metal.and_then(|t| self.decode(t)).map(|i| i.into_luma8());
        let result = if r.is_none() && m.is_none() {
            None
        } else {
            let (w, h) = r
                .as_ref()
                .or(m.as_ref())
                .map(|i| i.dimensions())
                .unwrap_or((1, 1));
            let sample = |img: &Option<image::GrayImage>, x: u32, y: u32, missing: u8| match img {
                Some(i) => {
                    let (iw, ih) = i.dimensions();
                    i.get_pixel(x * iw / w, y * ih / h).0[0]
                }
                None => missing,
            };
            let mut rgba = Vec::with_capacity((w * h * 4) as usize);
            for y in 0..h {
                for x in 0..w {
                    // Missing roughness defaults to 0.5, missing metal to 0 (factors are 1.0).
                    rgba.extend_from_slice(&[0, sample(&r, x, y, 128), sample(&m, x, y, 0), 255]);
                }
            }
            self.textures.push(Texture {
                width: w,
                height: h,
                rgba,
                srgb: false,
            });
            Some(self.textures.len() - 1)
        };
        self.texture_cache.insert(key, result);
        result
    }

    fn decode(&mut self, tex: &ufbx::Texture) -> Option<image::DynamicImage> {
        let tex = if tex.content.is_empty() && !tex.has_file {
            tex.file_textures.iter().next().map(|t| &**t).unwrap_or(tex)
        } else {
            tex
        };
        let name = if tex.filename.is_empty() {
            tex.element.name.to_string()
        } else {
            tex.filename.to_string()
        };
        let result = if !tex.content.is_empty() {
            image::load_from_memory(&tex.content).map_err(|e| anyhow!(e))
        } else {
            let candidates = [
                tex.relative_filename.as_ref(),
                tex.filename.as_ref(),
                tex.absolute_filename.as_ref(),
            ];
            match resolve_texture_path(&self.dir, &candidates) {
                Some(p) => image::open(&p).with_context(|| p.display().to_string()),
                None => Err(anyhow!("file not found")),
            }
        };
        match result {
            Ok(img) => Some(img),
            Err(e) => {
                self.warnings.push(format!("Texture {name}: {e:#}"));
                None
            }
        }
    }
}

fn map_texture(map: &ufbx::MaterialMap) -> Option<&ufbx::Texture> {
    if map.feature_disabled {
        return None;
    }
    map.texture.as_deref()
}

fn vec4(map: &ufbx::MaterialMap) -> glam::Vec4 {
    let v = &map.value_vec4;
    glam::Vec4::new(v.x as f32, v.y as f32, v.z as f32, v.w as f32)
}
