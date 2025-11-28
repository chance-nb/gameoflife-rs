#![allow(clippy::cast_precision_loss)]
use bitvec::index::BitIdx;
use bitvec::order::Msb0;
use bitvec::store::BitStore as _;
use core::ops::Not;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read as _, Write as _};
use std::path::PathBuf;
use three_d::{CpuTexture, GUI, Key, PhysicalPoint, ScissorBox, egui, vec4};

use three_d::core::{ClearState, Context, Program, RenderStates, VertexBuffer, vec3};
use three_d::{
    CoreError, Event, FrameInput, Interpolation, MouseButton, Texture2D, TextureData, Vector2,
    Vector3, Wrapping, vec2,
};
use winit::dpi::PhysicalSize;

#[derive(Debug, Default)]
struct FrameCounter {
    fps: f64,
    frames: u64,
    time: f64, // in ms
}
impl FrameCounter {
    const PROBE_INTERVAL: f64 = 1000.;
    const SECOND_INTERVAL: f64 = Self::PROBE_INTERVAL / 1000.;
    pub const fn add_frame(&mut self) {
        self.frames += 1;
    }
    pub fn update(&mut self, delta: f64) {
        self.time += delta;
        if self.time >= Self::PROBE_INTERVAL {
            self.time -= Self::PROBE_INTERVAL;
            self.fps = self.frames as f64 / Self::SECOND_INTERVAL;
            self.frames = 0;
        }
    }
}

#[derive(Default, Debug, Clone, Copy)]
enum ActiveItem {
    #[default]
    A,
    B,
}
impl Not for ActiveItem {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Self::A => Self::B,
            Self::B => Self::A,
        }
    }
}

struct SwapItem<T> {
    pub active: ActiveItem,
    pub item_a: T,
    pub item_b: T,
}
impl<T> SwapItem<T> {
    const fn new(texture_a: T, texture_b: T) -> Self {
        Self {
            item_a: texture_a,
            item_b: texture_b,
            active: ActiveItem::A,
        }
    }
    fn swap(&mut self) {
        self.active = !self.active;
    }
    const fn get_active_mut(&mut self) -> &mut T {
        match self.active {
            ActiveItem::A => &mut self.item_a,
            ActiveItem::B => &mut self.item_b,
        }
    }
    /// Returns the textures of this as a tuple (`primary`, `secondary`)
    const fn get_both_mut(&mut self) -> (&mut T, &mut T) {
        match self.active {
            ActiveItem::A => (&mut self.item_a, &mut self.item_b),
            ActiveItem::B => (&mut self.item_b, &mut self.item_a),
        }
    }
}

#[derive(Default)]
struct LoadSaveState {
    last_loaded_file: Option<PathBuf>,
    last_saved_file: Option<PathBuf>,
}

struct GridSize {
    current: Vector2<u32>,
    configured: Vector2<u32>,
}
impl GridSize {
    const DEFAULT_SIZE: Vector2<u32> = vec2(100, 100);
}
impl Default for GridSize {
    fn default() -> Self {
        Self {
            current: Self::DEFAULT_SIZE,
            configured: Self::DEFAULT_SIZE,
        }
    }
}

struct Assets {
    edit_program: Program,
    conway_program: Program,
    zoom_program: Program,
    fullscreen_quad: VertexBuffer<Vector3<f32>>,
    fullscreen_quad_uvs: VertexBuffer<Vector2<f32>>,
    gui: GUI,
}
impl Assets {
    fn init(context: &Context) -> Self {
        let conway_program = Program::from_source(
            context,
            include_str!("passthrough.vert"),
            include_str!("game_of_life.frag"),
        )
        .unwrap();

        let zoom_program = Program::from_source(
            context,
            include_str!("zoom.vert"),
            include_str!("sample_texture.frag"),
        )
        .unwrap();

        let edit_program = Program::from_source(
            context,
            include_str!("passthrough.vert"),
            include_str!("edit.frag"),
        )
        .unwrap();

        let fullscreen_quad = VertexBuffer::new_with_data(
            context,
            &[
                vec3(1., -1., 0.0),  // bottom right
                vec3(-1., -1., 0.0), // bottom left
                vec3(1., 1., 0.0),   // top right
                vec3(1., 1., 0.0),   // top right
                vec3(-1., -1., 0.0), // bottom left
                vec3(-1., 1., 0.0),  // top left
            ],
        );
        let fullscreen_quad_uvs = three_d::VertexBuffer::new_with_data(
            context,
            &[
                vec2(1.0, 0.0), // Corresponds to (1., -1., 0.0) -> Bottom Right
                vec2(0.0, 0.0), // Corresponds to (-1., -1., 0.0) -> Bottom Left
                vec2(1.0, 1.0), // Corresponds to (1., 1., 0.0) -> Top Right
                vec2(1.0, 1.0), // Corresponds to (1., 1., 0.0) -> Top Right
                vec2(0.0, 0.0), // Corresponds to (-1., -1., 0.0) -> Bottom Left
                vec2(0.0, 1.0), // Corresponds to (-1., 1., 0.0) -> Top Left
            ],
        );

        let gui = GUI::new(context);

        Self {
            edit_program,
            conway_program,
            zoom_program,
            fullscreen_quad,
            fullscreen_quad_uvs,
            gui,
        }
    }
}

pub enum Operation {
    Load,
    LoadLast,
    Save,
    SaveLast,
    ApplyGridSize,
    WindowResizeAdjust(PhysicalSize<u32>),
    RerandomiseGrid,
    ClearGrid,
}

pub struct GameOfLifeRunner {
    pending_operations: Vec<Operation>,
    grid: SwapItem<Texture2D>,
    zoom: Vector3<f32>, // (x, y, zoom level)
    paused: bool,
    grid_size: GridSize,
    time_since_last_frame: f64,
    target_frame_rate: f64,
    simulation_frame_counter: FrameCounter,
    load_save_state: LoadSaveState,
    current_window_size: PhysicalSize<u32>,
    assets: Assets,
}
impl GameOfLifeRunner {
    pub fn new(context: &Context, window_size: PhysicalSize<u32>) -> Self {
        let grid_texture = Self::make_grid_texture(
            GridSize::DEFAULT_SIZE,
            context,
            Self::generate_random_grid_data(GridSize::DEFAULT_SIZE),
        );

        Self {
            pending_operations: Vec::new(),
            grid: grid_texture,
            zoom: vec3(0., 0., 1.),
            time_since_last_frame: 0.,
            target_frame_rate: 10.,
            grid_size: GridSize::default(),
            load_save_state: LoadSaveState::default(),
            paused: false,
            simulation_frame_counter: FrameCounter::default(),
            current_window_size: window_size,
            assets: Assets::init(context),
        }
    }

    pub fn add_operation(&mut self, operation: Operation) {
        self.pending_operations.push(operation);
    }

    fn generate_random_grid_data(grid_size: Vector2<u32>) -> Vec<u8> {
        (0..grid_size.x * grid_size.y).fold(Vec::new(), |mut vec, _| {
            vec.push(rand::random::<u8>());
            vec
        })
    }

    fn generate_empty_grid_data(grid_size: Vector2<u32>) -> Vec<u8> {
        vec![0; (grid_size.x * grid_size.y) as usize]
    }

    fn make_grid_texture(
        size: Vector2<u32>,
        context: &Context,
        new_grid_data: Vec<u8>,
    ) -> SwapItem<Texture2D> {
        let new_texture_data = TextureData::RU8(new_grid_data);

        let new_cpu_texture = CpuTexture {
            name: "ConwayGrid".to_owned(),
            data: new_texture_data,
            width: size.x,
            height: size.y,
            min_filter: Interpolation::Nearest,
            mag_filter: Interpolation::Nearest,
            mipmap: None,
            wrap_s: Wrapping::Repeat,
            wrap_t: Wrapping::Repeat,
        };

        let texture_a = Texture2D::new(context, &new_cpu_texture);
        let texture_b = Texture2D::new(context, &new_cpu_texture);

        SwapItem::new(texture_a, texture_b)
    }

    fn draw_pixel(&mut self, frame_input: &FrameInput, position: PhysicalPoint, erase: bool) {
        // normalized x and y in the windowspace
        let x_window = -2. * (f64::from(position.x) / f64::from(frame_input.viewport.width) - 0.5);
        let y_window = -2. * (f64::from(position.y) / f64::from(frame_input.viewport.height) - 0.5);
        // normalized x and y on the texture
        let x_tex_normal = (x_window / f64::from(self.zoom.z)) + f64::from(self.zoom.x);
        let y_tex_normal = (y_window / f64::from(self.zoom.z)) + f64::from(self.zoom.y);

        let gsx = self.grid_size.current.x as f32;
        let gsy = self.grid_size.current.y as f32;
        #[expect(clippy::cast_possible_truncation)]
        let x_tex = ((-x_tex_normal).mul_add(f64::from(gsx), f64::from(gsx)) / 2.) as i32;
        #[expect(clippy::cast_possible_truncation)]
        let y_tex = ((-y_tex_normal).mul_add(f64::from(gsy), f64::from(gsy)) / 2.) as i32;

        let sbox = ScissorBox {
            x: x_tex,
            y: y_tex,
            width: 1,
            height: 1,
        };
        let tex = self.grid.get_active_mut();
        tex.as_color_target(None)
            .write_partially::<CoreError>(sbox, || {
                self.assets
                    .edit_program
                    .use_vertex_attribute("position", &self.assets.fullscreen_quad);
                self.assets.edit_program.use_uniform(
                    "color",
                    if erase {
                        vec4(0., 0., 0., 1.)
                    } else {
                        vec4(1., 0., 0., 1.)
                    },
                );
                self.assets.edit_program.draw_arrays(
                    RenderStates::default(),
                    frame_input.viewport,
                    self.assets.fullscreen_quad.vertex_count(),
                );
                Ok(())
            })
            .unwrap();
    }

    fn apply_configured_grid_size(&mut self, context: &Context) {
        let raw_data = self.grid.get_active_mut().as_color_target(None).read();

        // make a new Vec the the size of the new grid
        let mut new_grid_data: Vec<u8> = Vec::with_capacity(
            (self.grid_size.configured.x * self.grid_size.configured.y) as usize,
        );

        // put old data into the new one
        // the Vec represents the grid linewise from the top left
        for y in 0..self.grid_size.configured.y {
            for x in 0..self.grid_size.configured.x {
                if x >= self.grid_size.current.x || y >= self.grid_size.current.y {
                    // we're outside the old grid, so set it to not alive
                    new_grid_data.push(0);
                } else {
                    // the width of one line * the line below the current one + how far to the
                    // right in the next line we are
                    let pos = self.grid_size.current.x * y + x;
                    new_grid_data.push(raw_data[pos as usize]);
                }
            }
        }

        let texture = Self::make_grid_texture(self.grid_size.configured, context, new_grid_data);

        self.grid_size.current = self.grid_size.configured;
        self.grid = texture;
    }

    fn window_resize_adjust(&mut self, new_window_size: PhysicalSize<u32>, context: &Context) {
        let ratio_x =
            f64::from(self.grid_size.current.x) / f64::from(self.current_window_size.width);
        let ratio_y =
            f64::from(self.grid_size.current.y) / f64::from(self.current_window_size.height);
        let new_grid_size = Vector2 {
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            x: (ratio_x * f64::from(new_window_size.width)).floor() as u32,
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            y: (ratio_y * f64::from(new_window_size.height)).floor() as u32,
        };
        if self.grid_size.current != new_grid_size {
            self.current_window_size = PhysicalSize {
                #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                width: (f64::from(new_grid_size.x) / ratio_x) as u32,
                #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                height: (f64::from(new_grid_size.y) / ratio_y) as u32,
            };
            self.grid_size.configured = new_grid_size;
            self.apply_configured_grid_size(context);
        }
    }

    fn pick_and_save_file(&mut self) -> std::io::Result<()> {
        let file = rfd::FileDialog::new()
            .add_filter("gameoflife-rs", &["gameoflife-rs"])
            .set_title("Save State as")
            .save_file()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidFilename,
                    "Couldn't pick file path",
                )
            })?;

        self.save_file(file)
    }

    // storage format: 4 bytes width
    //                 4 bytes height
    //                 ceil((width*height)/8) bytes, last byte is padded with 0s
    pub fn save_file(&mut self, file_path: PathBuf) -> std::io::Result<()> {
        let mut file = BufWriter::new(File::create(file_path.clone())?);

        let current_state: Vec<u8> = self.grid.get_active_mut().as_color_target(None).read();

        let num_cells = self.grid_size.current.x * self.grid_size.current.y;

        let mut bitified_state: Vec<u8> = Vec::new();
        let mut curr_u8 = 0u8;
        for (i, cell) in current_state.iter().enumerate() {
            if i % 8 == 0 && i != 0 {
                bitified_state.push(curr_u8);
                curr_u8 = 0;
            }
            curr_u8 <<= 1;

            if *cell == 255 {
                curr_u8 += 1;
            }
        }

        if !num_cells.is_multiple_of(8) {
            curr_u8 <<= 8 - (num_cells % 8);
        }
        bitified_state.push(curr_u8);

        file.write_all(&self.grid_size.current.x.to_be_bytes())?;
        file.write_all(&self.grid_size.current.y.to_be_bytes())?;

        file.write_all(bitified_state.as_slice())?;

        self.load_save_state.last_saved_file = Some(file_path);

        Ok(())
    }

    fn pick_and_load_file(&mut self, context: &Context) -> std::io::Result<()> {
        let file = rfd::FileDialog::new()
            .add_filter("gameoflife-rs", &["gameoflife-rs"])
            .set_title("Save State as")
            .pick_file()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidFilename,
                    "Couldn't pick file path",
                )
            })?;

        self.load_file(file, context)
    }

    pub fn load_file(&mut self, file_path: PathBuf, context: &Context) -> std::io::Result<()> {
        let mut file = BufReader::new(File::open(file_path.clone())?).bytes();

        macro_rules! next_byte {
            () => {
                file.next().ok_or(std::io::Error::new(
                    std::io::ErrorKind::InvalidFilename,
                    "Missing Data",
                ))??
            };
        }

        let width = u32::from_be_bytes([next_byte!(), next_byte!(), next_byte!(), next_byte!()]);
        let height = u32::from_be_bytes([next_byte!(), next_byte!(), next_byte!(), next_byte!()]);

        let mut data: Vec<u8> = Vec::new();
        for byte in file.flatten() {
            for i in BitIdx::range_all() {
                data.push(if byte.get_bit::<Msb0>(i) { 255u8 } else { 0u8 });
            }
        }

        data.truncate(width as usize * height as usize);

        let size = vec2(width, height);
        self.grid = Self::make_grid_texture(size, context, data);
        self.grid_size.current = size;
        self.grid_size.configured = size;

        self.paused = true;

        self.load_save_state.last_loaded_file = Some(file_path);

        Ok(())
    }

    pub fn render_frame(&mut self, frame_input: &mut FrameInput) -> bool {
        if self.target_frame_rate != 0. && !self.paused {
            self.time_since_last_frame += frame_input.elapsed_time;
        }

        let mut need_rerender = self.make_gui(frame_input);
        need_rerender |= self.update(frame_input);

        // gui doesn't work if it's not drawn for the first couple frames, for whatever reason
        need_rerender |= frame_input.accumulated_time < 100.;
        if need_rerender {
            frame_input
                .screen()
                // Clear the color and depth of the screen render target
                .clear(ClearState::color_and_depth(0.8, 0.8, 0.8, 1.0, 1.0))
                .write::<CoreError>(|| {
                    self.assets
                        .zoom_program
                        .use_vertex_attribute("position", &self.assets.fullscreen_quad);
                    self.assets
                        .zoom_program
                        .use_texture("conway", self.grid.get_active_mut());
                    self.assets.zoom_program.use_uniform("zoom", self.zoom);
                    // self.zoom_program
                    //     .use_uniform("time", frame_input.accumulated_time as f32 / 1000.);
                    self.assets
                        .conway_program
                        .use_vertex_attribute("uv", &self.assets.fullscreen_quad_uvs);

                    self.assets.zoom_program.draw_arrays(
                        RenderStates::default(),
                        frame_input.viewport,
                        self.assets.fullscreen_quad.vertex_count(),
                    );
                    Ok(())
                })
                .unwrap()
                .write::<CoreError>(|| self.assets.gui.render())
                .unwrap();
        }

        self.simulation_frame_counter
            .update(frame_input.elapsed_time);
        need_rerender
    }

    fn update(&mut self, frame_input: &FrameInput) -> bool {
        let mut need_rerender = false;

        need_rerender |= self.apply_pending_operations(frame_input);

        need_rerender |= self.handle_events(&frame_input.events, frame_input);

        let target_time_per_frame = (1.0 / self.target_frame_rate) * 1000.;
        if self.time_since_last_frame >= target_time_per_frame && !self.paused {
            self.time_since_last_frame -= target_time_per_frame;
            self.update_simulation();

            self.simulation_frame_counter.add_frame();
            need_rerender = true;
        }

        need_rerender
    }

    fn update_simulation(&mut self) {
        let (active, inactive) = self.grid.get_both_mut();

        let conway_target = inactive.as_color_target(None);
        conway_target
            .write::<CoreError>(|| {
                self.assets
                    .conway_program
                    .use_vertex_attribute("position", &self.assets.fullscreen_quad);
                self.assets
                    .conway_program
                    .use_vertex_attribute("uv", &self.assets.fullscreen_quad_uvs);
                self.assets.conway_program.use_texture("conway", active);
                self.assets
                    .conway_program
                    .use_uniform("texture_size", self.grid_size.current);

                self.assets.conway_program.draw_arrays(
                    RenderStates::default(),
                    conway_target.viewport(),
                    self.assets.fullscreen_quad.vertex_count(),
                );
                Ok(())
            })
            .unwrap();

        self.grid.swap();
    }

    fn apply_pending_operations(&mut self, frame_input: &FrameInput) -> bool {
        let mut operation_applied = false;
        while let Some(operation) = self.pending_operations.pop() {
            match operation {
                Operation::Save => match self.pick_and_save_file() {
                    Ok(()) => println!("saved succesfully!"),
                    Err(e) => println!("{e}"),
                },

                Operation::Load => match self.pick_and_load_file(&frame_input.context) {
                    Ok(()) => {
                        println!("loaded succesfully!");
                    }
                    Err(e) => println!("{e}"),
                },

                Operation::LoadLast => {
                    if let Some(file) = &self.load_save_state.last_loaded_file {
                        match self.load_file(file.clone(), &frame_input.context) {
                            Ok(()) => {
                                println!("loaded succesfully!");
                            }
                            Err(e) => println!("{e}"),
                        }
                    }
                }

                Operation::SaveLast => {
                    if let Some(file) = &self.load_save_state.last_saved_file {
                        match self.save_file(file.clone()) {
                            Ok(()) => {
                                println!("saved succesfully!");
                            }
                            Err(e) => println!("{e}"),
                        }
                    }
                }

                Operation::ApplyGridSize => {
                    self.apply_configured_grid_size(&frame_input.context);
                    self.zoom = vec3(0., 0., 1.);
                }

                Operation::RerandomiseGrid => {
                    self.grid
                        .get_active_mut()
                        .fill(&Self::generate_random_grid_data(self.grid_size.current));
                    self.update_simulation();
                    self.time_since_last_frame = 0.;
                }

                Operation::ClearGrid => {
                    self.grid
                        .get_active_mut()
                        .fill(&Self::generate_empty_grid_data(self.grid_size.current));
                }

                Operation::WindowResizeAdjust(new_size) => {
                    self.window_resize_adjust(new_size, &frame_input.context);
                }
            }
            operation_applied = true;
        }

        operation_applied
    }

    fn handle_events(&mut self, events: &Vec<Event>, frame_input: &FrameInput) -> bool {
        let mut rerender_required = false;
        for event in events {
            match event {
                Event::MousePress {
                    button,
                    position,
                    handled: false,
                    ..
                } => {
                    match button {
                        MouseButton::Left => self.draw_pixel(frame_input, *position, false),
                        MouseButton::Right => self.draw_pixel(frame_input, *position, true),
                        MouseButton::Middle => {}
                    }
                    rerender_required = true;
                }
                Event::MouseMotion {
                    button,
                    delta,
                    handled: false,
                    position,
                    ..
                } => match button {
                    Some(MouseButton::Middle) => {
                        let x_window = 2. * (delta.0 / (frame_input.viewport.width as f32));
                        let y_window = -2. * (delta.1 / (frame_input.viewport.height as f32));
                        self.zoom.x += x_window / self.zoom.z;
                        self.zoom.y += y_window / self.zoom.z;
                        rerender_required = true;
                    }
                    Some(MouseButton::Left) => {
                        self.draw_pixel(frame_input, *position, false);
                        rerender_required = true;
                    }
                    Some(MouseButton::Right) => {
                        self.draw_pixel(frame_input, *position, true);
                        rerender_required = true;
                    }
                    _ => {}
                },
                // zoom
                Event::MouseWheel {
                    delta: (_, delta),
                    position,
                    handled: false,
                    ..
                }
                | Event::PinchGesture {
                    delta,
                    position,
                    handled: false,
                    ..
                } => {
                    let new_zoom_z = self.zoom.z * (1.0 + delta / 100.);
                    if new_zoom_z < 1. {
                        // zooming out makes no sense, so just reset everything
                        self.zoom.z = 1.;
                        self.zoom.x = 0.;
                        self.zoom.y = 0.;
                    } else {
                        // normalized x and y in the windowspace
                        let x_window =
                            -2. * (position.x / (frame_input.viewport.width as f32) - 0.5);
                        let y_window =
                            -2. * (position.y / (frame_input.viewport.height as f32) - 0.5);
                        // normalized x and y on the texture
                        let x_tex = (x_window / self.zoom.z) + self.zoom.x;
                        let y_tex = (y_window / self.zoom.z) + self.zoom.y;

                        self.zoom.z = new_zoom_z;

                        // we don't want to put where the mouse was on the origin,
                        // so we shift it back to the windowspace position
                        self.zoom.x = x_tex - (x_window / self.zoom.z);
                        self.zoom.y = y_tex - (y_window / self.zoom.z);
                    }
                    rerender_required = true;
                }
                Event::KeyPress {
                    kind: Key::Space,
                    handled: false,
                    ..
                } => self.paused = !self.paused,
                _ => {}
            }
        }
        rerender_required
    }

    fn make_gui(&mut self, frame_input: &mut FrameInput) -> bool {
        self.assets.gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |gui_context| {
                use three_d::egui::{Button, Checkbox, Label, Slider, SliderClamping};
                egui::Window::new("").show(gui_context, |ui| {
                    ui.add(Label::new(format!(
                        "FPS: {}",
                        self.simulation_frame_counter.fps
                    )));
                    ui.add(
                        Slider::new(&mut self.target_frame_rate, 0.0..=200.)
                            .clamping(SliderClamping::Never)
                            .text("Target Framerate"),
                    );
                    ui.add(
                        Slider::new(&mut self.grid_size.configured.x, 1..=1000)
                            .clamping(SliderClamping::Never)
                            .text("Grid width"),
                    );
                    ui.add(
                        Slider::new(&mut self.grid_size.configured.y, 1..=1000)
                            .clamping(SliderClamping::Never)
                            .text("Grid height"),
                    );
                    if ui.add(Button::new("Apply Grid Size")).clicked() {
                        self.pending_operations.push(Operation::ApplyGridSize);
                    }

                    ui.horizontal_wrapped(|ui| {
                        ui.set_max_width(250.);
                        if ui.add(Button::new("Rerandomise")).clicked() {
                            self.pending_operations.push(Operation::RerandomiseGrid);
                        }
                        if ui.add(Button::new("Clear")).clicked() {
                            self.pending_operations.push(Operation::ClearGrid);
                        }
                        if ui.add(Button::new("Save")).clicked() {
                            self.pending_operations.push(Operation::Save);
                        }
                        if ui.add(Button::new("Load")).clicked() {
                            self.pending_operations.push(Operation::Load);
                        }
                        if self.load_save_state.last_loaded_file.is_some()
                            && ui.add(Button::new("Load again")).clicked()
                        {
                            self.pending_operations.push(Operation::LoadLast);
                        }
                        if self.load_save_state.last_saved_file.is_some()
                            && ui.add(Button::new("Save again")).clicked()
                        {
                            self.pending_operations.push(Operation::SaveLast);
                        }
                    });

                    ui.add(Checkbox::new(&mut self.paused, "Pause")).clicked();
                });
            },
        )
    }
}
