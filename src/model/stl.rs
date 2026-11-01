use std::path::Path;

use anyhow::{Context, Result, bail};
use glam::Vec3;

use super::{Mesh, Model, Vertex, flat_normal};
use crate::settings::{Shading, UpAxis};

pub fn load(path: &Path) -> Result<Model> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let triangles = if is_ascii(&data) {
        parse_ascii(&data)?
    } else {
        parse_binary(&data)?
    };
    if triangles.is_empty() {
        bail!("STL has no triangles");
    }

    // Stored normals are frequently zero or wrong, so recompute them.
    let mut vertices = Vec::with_capacity(triangles.len() * 3);
    for [a, b, c] in triangles {
        let n = flat_normal(a, b, c).to_array();
        for p in [a, b, c] {
            vertices.push(Vertex {
                pos: p.to_array(),
                normal: n,
                uv: [0.0; 2],
                color: [1.0; 4],
            });
        }
    }
    let indices = (0..vertices.len() as u32).collect();

    Ok(Model {
        path: path.to_path_buf(),
        meshes: vec![Mesh {
            vertices,
            indices,
            material: 0,
        }],
        materials: Vec::new(),
        textures: Vec::new(),
        native_up: UpAxis::Z,
        default_shading: Shading::Flat,
        warnings: Vec::new(),
    })
}

/// Binary STLs may also start with "solid", so check the body for "facet" too.
fn is_ascii(data: &[u8]) -> bool {
    let head = &data[..data.len().min(512)];
    let text = String::from_utf8_lossy(head).to_ascii_lowercase();
    text.trim_start().starts_with("solid") && text.contains("facet") && !head.contains(&0)
}

fn parse_binary(data: &[u8]) -> Result<Vec<[Vec3; 3]>> {
    if data.len() < 84 {
        bail!("file too small to be an STL");
    }
    // Some exporters write a bogus count, so trust the file size when they disagree.
    let declared = u32::from_le_bytes(data[80..84].try_into().unwrap()) as usize;
    let available = (data.len() - 84) / 50;
    let count = if declared <= available {
        declared
    } else {
        available
    };
    let f = |o: usize| f32::from_le_bytes(data[o..o + 4].try_into().unwrap());
    Ok((0..count)
        .map(|i| {
            let o = 84 + i * 50 + 12;
            let v = |k: usize| Vec3::new(f(o + k * 12), f(o + k * 12 + 4), f(o + k * 12 + 8));
            [v(0), v(1), v(2)]
        })
        .collect())
}

fn parse_ascii(data: &[u8]) -> Result<Vec<[Vec3; 3]>> {
    let text = String::from_utf8_lossy(data);
    let mut tris = Vec::new();
    let mut cur = Vec::with_capacity(3);
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() != Some("vertex") {
            continue;
        }
        let xyz: Vec<f32> = parts.filter_map(|p| p.parse().ok()).collect();
        if xyz.len() != 3 {
            bail!("malformed vertex line: {line}");
        }
        cur.push(Vec3::new(xyz[0], xyz[1], xyz[2]));
        if cur.len() == 3 {
            tris.push([cur[0], cur[1], cur[2]]);
            cur.clear();
        }
    }
    Ok(tris)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_stl() {
        let src = b"solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n";
        assert!(is_ascii(src));
        assert_eq!(parse_ascii(src).unwrap().len(), 1);
    }

    #[test]
    fn binary_with_bogus_count() {
        let mut d = vec![0u8; 84 + 50];
        d[80..84].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(parse_binary(&d).unwrap().len(), 1);
    }
}
