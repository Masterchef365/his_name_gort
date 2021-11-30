use std::os::windows::prelude::MetadataExt;

use idek::{prelude::*, IndexBuffer, MultiPlatformCamera};

fn main() -> Result<()> {
    launch::<HisNameGort>(Settings::default().vr_if_any_args())
}

type DynSparseMatrix = Vec<Vec<f32>>;

/// Given coefficients for each dimension, project onto the dimension `proj`
// https://www.heldermann-verlag.de/jgg/jgg01_05/jgg0404.pdf but I've modified it in ways I don't really understand
fn make_projection(coeffs: &[f32], proj: usize) -> DynSparseMatrix {
    let dims = coeffs.len();
    let mut rows = vec![];
    for row in dims - proj..dims {
        let mut row_data = vec![];
        for col in 0..row + 1 {
            let mut entry = 1.;
            for (dim, &coeff) in (0..row).zip(coeffs) {
                if col > dim + 1 {
                    continue;
                }
                entry *= &match (col <= dim, row == dim + 1) {
                    (true, true) => -coeff.sin(),
                    (false, false) => coeff.sin(),
                    _ => coeff.cos(),
                };
            }
            row_data.push(entry);
        }
        rows.push(row_data);
    }
    rows
}

fn project_3d<const D: usize>(v: Vector<D>) -> Vector<3> {
    let mut out_vect3 = [0.0; 3];
    out_vect3.copy_from_slice(&v[..3]);
    out_vect3
}

fn project_nd<const D: usize>(v: Vector<D>, matrix: &DynSparseMatrix) -> Vector<3> {
    let mut out_vect3 = [0.0; 3];

    for (out, mat) in out_vect3.iter_mut().zip(matrix) {
        *out = mat.iter().zip(&v).map(|(m, v)| m * v).sum::<f32>();
    }

    out_vect3
}

const DIMS: usize = 7;

struct HisNameGort {
    original_verts: Vec<VertexN<DIMS>>,
    verts: VertexBuffer,
    indices: IndexBuffer,
    shader: Shader,
    camera: MultiPlatformCamera,
    matrix: DynSparseMatrix,
}

impl App for HisNameGort {
    fn init(ctx: &mut Context, platform: &mut Platform) -> Result<Self> {
        let shader = ctx.shader(
            DEFAULT_VERTEX_SHADER,
            DEFAULT_FRAGMENT_SHADER,
            Primitive::Lines,
        )?;

        let original_verts = n_cube_vertices_color(1.).collect::<Vec<_>>();
        let indices = n_cube_line_indices(DIMS as _).collect::<Vec<u32>>();

        //let matrix = make_projection(&[0.3, 1.4, 2.3, 9.0, 1.4, 3.2], 3);
        let matrix = make_projection(&[2.; DIMS], 3);
        let verts = convert_verts(&original_verts, 0., &matrix);

        Ok(Self {
            verts: ctx.vertices(&verts, true)?,
            indices: ctx.indices(&indices, false)?,
            original_verts,
            shader,
            camera: MultiPlatformCamera::new(platform),
            matrix,
        })
    }

    fn frame(&mut self, ctx: &mut Context, _: &mut Platform) -> Result<Vec<DrawCmd>> {
        let time = ctx.start_time().elapsed().as_secs_f32();
        let anim = time / 10.;

        let new_verts: Vec<Vertex> = convert_verts(&self.original_verts, anim, &self.matrix);

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

#[derive(Copy, Clone, Debug)]
struct VertexN<const D: usize> {
    pos: Vector<D>,
    color: Vector<3>,
}

impl Into<Vertex> for VertexN<3> {
    fn into(self) -> Vertex {
        let VertexN { pos, color } = self;
        Vertex { pos, color }
    }
}

impl<const D: usize> VertexN<D> {
    fn map_pos<const U: usize>(self, f: impl FnOnce(Vector<D>) -> Vector<U>) -> VertexN<U> {
        VertexN {
            pos: f(self.pos),
            color: self.color,
        }
    }
}

fn convert_verts<const D: usize>(
    original: &[VertexN<D>],
    anim: f32,
    matrix: &DynSparseMatrix,
) -> Vec<Vertex> {
    let color = [1.; 3];
    original
        .iter()
        .copied()
        //.map(|v| n_rotate((0, 1), anim, v))
        .map(|v| v.map_pos(|v| n_rotate((0, 1), anim * 2., v)))
        .map(|v| v.map_pos(|v| n_rotate((1, 2), 3., v)))
        .map(|v| v.map_pos(|v| n_rotate((2, 3), anim, v)))
        .map(|v| v.map_pos(|v| n_rotate((3, 4), 0.2, v)))
        .map(|v| v.map_pos(|v| n_rotate((4, 5), anim * 3., v)))
        //.map(|v| n_rotate((1, 3), 2.5, v))
        //.map(|v| n_rotate((1, 2), anim / 3., v))
        //.map(|pos| project_3d(pos))
        .map(|v| v.map_pos(|pos| project_nd(pos, matrix)))
        .map(|v| v.into())
        .collect()
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

fn n_cube_vertices_color<const D: usize>(scale: f32) -> impl Iterator<Item = VertexN<D>> {
    let rank = D as u32;
    let pos_dim = move |i: u32, dim: u32| if i & 1 << dim == 0 { scale } else { -scale };
    let color_dim = move |i: u32, dim: u32| if i & 1 << dim == 0 { 1. } else { 0. };
    (0..1u32 << rank).map(move |i| VertexN {
        pos: collect_array((0..rank).map(|dim| pos_dim(i, dim))),
        color: collect_array((0..3).map(|dim| color_dim(i, dim))),
    })
}

fn n_cube_vertices<const D: usize>(scale: f32) -> impl Iterator<Item = Vector<D>> {
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