use idek::{prelude::*, IndexBuffer, MultiPlatformCamera};

fn main() -> Result<()> {
    launch::<TriangleApp>(Settings::default().vr_if_any_args())
}

struct TriangleApp {
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
        let (vertices, indices) = line_cube(1., [1.; 3]);
        Ok(Self {
            verts: ctx.vertices(&vertices, false)?,
            indices: ctx.indices(&indices, false)?,
            shader,
            camera: MultiPlatformCamera::new(platform),
        })
    }

    fn frame(&mut self, _ctx: &mut Context, _: &mut Platform) -> Result<Vec<DrawCmd>> {
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

fn line_cube(scale: f32, color: [f32; 3]) -> (Vec<Vertex>, Vec<u32>) {
    let vertices = n_cube_vertices(scale, color, 3)
        .map(|vert| Vertex {
            pos: [vert.pos[0], vert.pos[1], vert.pos[2]],
            color: vert.color,
        })
        .collect();

    let indices = n_cube_line_indices(3).collect();

    (vertices, indices)
}

struct VertexN {
    pos: Vec<f32>,
    color: [f32; 3], // TODO: Multidimensional color??
}

fn n_cube_vertices(scale: f32, color: [f32; 3], rank: u32) -> impl Iterator<Item = VertexN> {
    let f = move |i: u32, dim: u32| if i & 1 << dim == 0 { scale } else { -scale };
    (0..1u32 << rank)
        .map(move |i| (0..rank).map(|dim| f(i, dim)).collect::<Vec<f32>>())
        .map(move |pos| VertexN { pos, color })
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
