const PI: f32 = 3.14159265359;
const SAMPLES: u32 = 256u;

struct Params {
    // x: roughness, y: source width, z: source height, w: source mip count
    p: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var src_sampler: sampler;

struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> Out {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: Out;
    out.clip = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

fn equirect_to_dir(uv: vec2<f32>) -> vec3<f32> {
    let phi = (uv.x - 0.5) * 2.0 * PI;
    let theta = uv.y * PI;
    return vec3<f32>(sin(theta) * sin(phi), cos(theta), -sin(theta) * cos(phi));
}

fn dir_to_equirect(d: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(atan2(d.x, -d.z) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI);
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    var bits = i;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return vec2<f32>(f32(i) / f32(n), f32(bits) * 2.3283064365386963e-10);
}

@fragment
fn fs_main(in: Out) -> @location(0) vec4<f32> {
    let n = equirect_to_dir(in.uv);
    let roughness = params.p.x;
    if roughness < 0.001 {
        return vec4<f32>(textureSampleLevel(src, src_sampler, in.uv, 0.0).rgb, 1.0);
    }
    let a = roughness * roughness;
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if abs(n.y) > 0.999 {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tx = normalize(cross(up, n));
    let ty = cross(n, tx);
    let texel_solid_angle = 4.0 * PI / (params.p.y * params.p.z);

    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    for (var i = 0u; i < SAMPLES; i++) {
        let xi = hammersley(i, SAMPLES);
        let phi = 2.0 * PI * xi.x;
        let cos_t = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
        let sin_t = sqrt(1.0 - cos_t * cos_t);
        let h = normalize(tx * (sin_t * cos(phi)) + ty * (sin_t * sin(phi)) + n * cos_t);
        let l = normalize(2.0 * dot(n, h) * h - n);
        let n_dot_l = dot(n, l);
        if n_dot_l <= 0.0 {
            continue;
        }
        // Filtered importance sampling: read from a blurrier mip where samples are sparse.
        let n_dot_h = max(dot(n, h), 0.0);
        let a2 = a * a;
        let d = (n_dot_h * n_dot_h * (a2 - 1.0) + 1.0);
        let pdf = a2 / (PI * d * d) * 0.25;
        let sample_solid_angle = 1.0 / (f32(SAMPLES) * pdf + 1e-6);
        let lod = clamp(0.5 * log2(sample_solid_angle / texel_solid_angle) + 1.0, 0.0, params.p.w - 1.0);
        sum += textureSampleLevel(src, src_sampler, dir_to_equirect(l), lod).rgb * n_dot_l;
        weight += n_dot_l;
    }
    return vec4<f32>(sum / max(weight, 1e-4), 1.0);
}
