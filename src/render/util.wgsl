struct ShadowParams {
    // local (pivot-centred) space -> ortho clip space, looking down -Y
    proj: mat4x4<f32>,
    // raw model space -> local space
    local: mat4x4<f32>,
    // x: ground y, y: falloff height
    p: vec4<f32>,
};

@group(0) @binding(0) var<uniform> shadow: ShadowParams;

struct ShadowOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) height: f32,
};

@vertex
fn vs_shadow(@location(0) pos: vec3<f32>) -> ShadowOut {
    var out: ShadowOut;
    let lp = shadow.local * vec4<f32>(pos, 1.0);
    out.clip = shadow.proj * lp;
    out.height = lp.y - shadow.p.x;
    return out;
}

@fragment
fn fs_shadow(in: ShadowOut) -> @location(0) vec4<f32> {
    // Anything overhead blocks some sky light; parts near the ground block most.
    let v = 0.45 + 0.55 * exp(-max(in.height, 0.0) / shadow.p.y);
    return vec4<f32>(v, v, v, 1.0);
}

struct DownParams {
    factor: vec4<u32>,
};

@group(0) @binding(0) var<uniform> down: DownParams;
@group(0) @binding(1) var down_src: texture_2d<f32>;

@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_downsample(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let k = i32(down.factor.x);
    let base = vec2<i32>(floor(pos.xy)) * k;
    var sum = vec4<f32>(0.0);
    for (var y = 0; y < k; y++) {
        for (var x = 0; x < k; x++) {
            sum += textureLoad(down_src, base + vec2<i32>(x, y), 0);
        }
    }
    return sum / f32(k * k);
}
