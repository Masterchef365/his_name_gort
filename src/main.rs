use std::os::windows::prelude::MetadataExt;

use idek::{prelude::*, IndexBuffer, MultiPlatformCamera};

fn main() -> Result<()> {
    launch::<TriangleApp>(Settings::default().vr_if_any_args())
}

const DIMS: usize = 3;

struct TriangleApp {
    original_verts: Vec<Vertex>,
    verts: VertexBuffer,
    indices: IndexBuffer,
    shader: Shader,
    camera: MultiPlatformCamera,
}

impl App for TriangleApp {
    fn init(ctx: &mut Context, platform: &mut Platform) -> Result<Self> {
        let shader = ctx.shader(
            DEFAULT_VERTEX_SHADER,
            DEFAULT_FRAGMENT_SHADER,
            Primitive::Lines,
        )?;

        let (original_verts, indices) = line_cube(1., [1.; 3]);

        Ok(Self {
            verts: ctx.vertices(&original_verts, true)?,
            indices: ctx.indices(&indices, false)?,
            original_verts,
            shader,
            camera: MultiPlatformCamera::new(platform),
        })
    }

    fn frame(&mut self, ctx: &mut Context, _: &mut Platform) -> Result<Vec<DrawCmd>> {
        let mut new_verts = self.original_verts.clone();
        let time = ctx.start_time().elapsed().as_secs_f32();
        let anim = time;
        let axis = (0, 2);

        new_verts.iter_mut().for_each(|v| v.pos = n_rotate(axis, anim, v.pos));

        ctx.update_vertices(self.verts, &new_verts)?;

        Ok(vec![DrawCmd::new(self.verts)
            .indices(self.indices)
            .shader(self.shader)])
    }

    fn event(
        &mut self,
        ctx: &mut Context,
        platform: &mut Platform,
        mut event: Event,
    ) -> Result<()> {
        if self.camera.handle_event(&mut event) {
            ctx.set_camera_prefix(self.camera.get_prefix())
        }
        idek::close_when_asked(platform, &event);
        Ok(())
    }
}

//fn project<const D: 

fn line_cube(scale: f32, color: [f32; 3]) -> (Vec<Vertex>, Vec<u32>) {
    let vertices = n_cube_vertices::<3>(scale)
        .map(|pos| Vertex { pos, color })
        .collect();

    let indices = n_cube_line_indices(3).collect();

    (vertices, indices)
}

fn collect_array<T, const N: usize>(i: impl Iterator<Item = T>) -> [T; N]
where
    T: Default + Copy,
{
    let mut array = [T::default(); N];
    let n_filled = array.iter_mut().zip(i).map(|(arr, i)| *arr = i).count();
    assert_eq!(n_filled, N, "Iterator could not fill array");
    array
}

type Vector<const D: usize> = [f32; D];

fn n_cube_vertices<const D: usize>(
    scale: f32,
) -> impl Iterator<Item = Vector<D>> {
    let rank = D as u32;
    let f = move |i: u32, dim: u32| if i & 1 << dim == 0 { scale } else { -scale };
    (0..1u32 << rank).map(move |i| collect_array((0..rank).map(|dim| f(i, dim))))
}

fn n_cube_line_indices(rank: u32) -> impl Iterator<Item = u32> {
    (0..rank)
        .map(move |dim| {
            let mask = u32::MAX << dim;
            (0..1 << (rank - 1)).map(move |combo| {
                let high_bits = (combo & mask) << 1;
                let low_bits = combo & !mask;
                (0..=1).map(move |bit| high_bits | (bit << dim) | low_bits)
            })
        })
        .flatten()
        .flatten()
}

fn n_rotate<const D: usize>(axis: (usize, usize), angle: f32, mut v: Vector<D>) -> Vector<D> {
    let (a, b) = axis;
    let (va, vb) = (v[a], v[b]);
    v[a] = va * angle.cos() - vb * angle.sin();
    v[b] = va * angle.sin() + vb * angle.cos();
    v
}

