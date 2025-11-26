#![allow(clippy::cast_precision_loss)]
use bitvec::index::BitIdx;
use bitvec::order::Msb0;
use bitvec::store::BitStore as _;
use core::ops::Not;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read as _, Write as _};
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

// state machine would be detrimentally overcomplex; any state represented by the bools is valid
#[allow(clippy::struct_excessive_bools)]
pub struct ConwaysGameOfLife {
    grid_size: Vector2<u32>,
    config_grid_size: Vector2<u32>,
    should_resize_grid: bool,
    should_save: bool,
    should_load: bool,
    conway_texture: SwapItem<Texture2D>,
    conway_program: Program,
    zoom_program: Program,
    zoom: Vector3<f32>, // (x, y, zoom level)
    edit_program: Program,
    fullscreen_quad: VertexBuffer<Vector3<f32>>,
    fullscreen_quad_uvs: VertexBuffer<Vector2<f32>>,
    gui: GUI,
    frame_time: f64,
    frame_rate: f64,
    paused: bool,
    frame_counter: FrameCounter,
    current_window_size: PhysicalSize<u32>,
    resized: bool,
}
impl ConwaysGameOfLife {
    pub fn new(context: &Context, window_size: PhysicalSize<u32>) -> Self {
        const DEFAULT_GRID_SIZE: Vector2<u32> = vec2(100, 100);
        let conway_texture = Self::make_texture(
            DEFAULT_GRID_SIZE,
            context,
            Self::generate_random_conway_data(DEFAULT_GRID_SIZE),
        );

        let conway_program = Program::from_source(
            context,
            include_str!("passthrough.vert"),
            include_str!("conway.frag"),
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

        let zoom = vec3(0., 0., 1.);

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
            grid_size: DEFAULT_GRID_SIZE,
            config_grid_size: DEFAULT_GRID_SIZE,
            should_resize_grid: false,
            should_save: false,
            should_load: false,
            conway_texture,
            conway_program,
            zoom_program,
            zoom,
            edit_program,
            fullscreen_quad,
            fullscreen_quad_uvs,
            gui,
            frame_time: 0.,
            frame_rate: 10.,
            paused: false,
            frame_counter: FrameCounter::default(),
            current_window_size: window_size,
            resized: false,
        }
    }

    fn generate_random_conway_data(grid_size: Vector2<u32>) -> Vec<u8> {
        (0..grid_size.x * grid_size.y).fold(Vec::new(), |mut vec, _| {
            vec.push(rand::random::<u8>());
            vec
        })
    }

    fn generate_empty_conway_data(grid_size: Vector2<u32>) -> Vec<u8> {
        vec![0; (grid_size.x * grid_size.y) as usize]
    }

    fn make_texture(
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

    fn resize_grid(&mut self, context: &Context) {
        self.should_resize_grid = false;

        // die Daten des Rasters holen
        let raw_data = self
            .conway_texture
            .get_active_mut()
            .as_color_target(None)
            .read();

        // make a new Vec the the size of the new grid
        let mut new_grid_data: Vec<u8> =
            Vec::with_capacity((self.config_grid_size.x * self.config_grid_size.y) as usize);

        // put old data into the new one
        // the Vec represents the grid linewise from the top left
        for y in 0..self.config_grid_size.y {
            for x in 0..self.config_grid_size.x {
                if x >= self.grid_size.x || y >= self.grid_size.y {
                    // we're outside the old grid, so set it to not alive
                    new_grid_data.push(0);
                } else {
                    // the width of one line * the line below the current one + how far to the
                    // right in the next line we are
                    let pos = self.grid_size.x * y + x;
                    new_grid_data.push(raw_data[pos as usize]);
                }
            }
        }

        let texture = Self::make_texture(self.config_grid_size, context, new_grid_data);

        self.grid_size = self.config_grid_size;
        self.conway_texture = texture;
        self.resized = true;
    }

    pub fn resize(&mut self, new_window_size: PhysicalSize<u32>, context: &Context) {
        let ratio_x = f64::from(self.grid_size.x) / f64::from(self.current_window_size.width);
        let ratio_y = f64::from(self.grid_size.y) / f64::from(self.current_window_size.height);
        let new_grid_size = Vector2 {
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            x: (ratio_x * f64::from(new_window_size.width)).floor() as u32,
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            y: (ratio_y * f64::from(new_window_size.height)).floor() as u32,
        };
        if self.grid_size != new_grid_size {
            self.current_window_size = PhysicalSize {
                #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                width: (f64::from(new_grid_size.x) / ratio_x) as u32,
                #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                height: (f64::from(new_grid_size.y) / ratio_y) as u32,
            };
            self.config_grid_size = new_grid_size;
            self.resize_grid(context);
        }
        self.resized = true;
    }

    // storage format: 4 bytes width
    //                 4 bytes height
    //                 ceil((width*height)/8) bytes, last byte is padded with 0s
    pub fn save(&mut self) -> std::io::Result<()> {
        let file = rfd::FileDialog::new()
            .add_filter("gaofli", &["conway-rs"])
            .set_title("Save State as")
            .save_file()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidFilename,
                    "Couldn't pick file path",
                )
            })?;

        let current_state: Vec<u8> = self
            .conway_texture
            .get_active_mut()
            .as_color_target(None)
            .read();

        let num_cells = self.grid_size.x * self.grid_size.y;
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

        let mut file = BufWriter::new(File::create(file)?);

        file.write_all(&self.grid_size.x.to_be_bytes())?;
        file.write_all(&self.grid_size.y.to_be_bytes())?;

        file.write_all(bitified_state.as_slice())?;

        Ok(())
    }

    pub fn load(&mut self, context: &Context) -> std::io::Result<()> {
        let file = rfd::FileDialog::new()
            .add_filter("conway-rs", &["conway-rs"])
            .set_title("Save State as")
            .pick_file()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidFilename,
                    "Couldn't pick file path",
                )
            })?;

        let mut file = BufReader::new(File::open(file)?).bytes();

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
        self.conway_texture = Self::make_texture(size, context, data);
        self.config_grid_size = size;
        self.grid_size = size;

        Ok(())
    }

    // Die wichtigste methode - hier wird ein neues Frame gerendert
    pub fn render_frame(&mut self, frame_input: &mut FrameInput) -> bool {
        if self.frame_rate != 0. && !self.paused {
            self.frame_time += frame_input.elapsed_time;
        }

        let mut need_rerender = self.gui(frame_input);

        if self.should_save {
            self.should_save = false;
            match self.save() {
                Ok(()) => println!("saved succesfully!"),
                Err(e) => println!("{e}"),
            }
        }

        if self.should_load {
            self.should_load = false;
            match self.load(&frame_input.context) {
                Ok(()) => {
                    println!("loaded succesfully!");
                    self.paused = true;
                }
                Err(e) => println!("{e}"),
            }
        }

        need_rerender |= self.handle_events(&frame_input.events, frame_input);

        let target_time_per_frame = (1.0 / self.frame_rate) * 1000.;
        let update_sim = self.frame_time >= target_time_per_frame && !self.paused;
        if self.should_resize_grid {
            self.resize_grid(&frame_input.context);
            self.zoom = vec3(0., 0., 1.);
        }
        if update_sim {
            self.frame_time -= target_time_per_frame;
            {
                let (active, inactive) = self.conway_texture.get_both_mut();
                // inactive.fill(&conway_data_empty);
                let conway_target = inactive.as_color_target(None);
                conway_target
                    .write::<CoreError>(|| {
                        self.conway_program
                            .use_vertex_attribute("position", &self.fullscreen_quad);
                        self.conway_program
                            .use_vertex_attribute("uv", &self.fullscreen_quad_uvs);
                        self.conway_program.use_texture("conway", active);
                        self.conway_program
                            .use_uniform("texture_size", self.grid_size);

                        self.conway_program.draw_arrays(
                            RenderStates::default(),
                            conway_target.viewport(),
                            self.fullscreen_quad.vertex_count(),
                        );
                        Ok(())
                    })
                    .unwrap();
                self.conway_texture.swap();
            };
            self.frame_counter.add_frame();
        }

        // gui doesn't work if it's not drawn for the first couple frames, for whatever reason
        need_rerender |= update_sim || frame_input.accumulated_time < 100. || self.resized;
        if need_rerender {
            frame_input
                .screen()
                // Clear the color and depth of the screen render target
                .clear(ClearState::color_and_depth(0.8, 0.8, 0.8, 1.0, 1.0))
                .write::<CoreError>(|| {
                    self.zoom_program
                        .use_vertex_attribute("position", &self.fullscreen_quad);
                    self.zoom_program
                        .use_texture("conway", self.conway_texture.get_active_mut());
                    self.zoom_program.use_uniform("zoom", self.zoom);
                    // self.zoom_program
                    //     .use_uniform("time", frame_input.accumulated_time as f32 / 1000.);
                    self.conway_program
                        .use_vertex_attribute("uv", &self.fullscreen_quad_uvs);

                    self.zoom_program.draw_arrays(
                        RenderStates::default(),
                        frame_input.viewport,
                        self.fullscreen_quad.vertex_count(),
                    );
                    Ok(())
                })
                .unwrap()
                .write::<CoreError>(|| self.gui.render())
                .unwrap();
        }

        self.frame_counter.update(frame_input.elapsed_time);
        need_rerender
    }

    fn gui(&mut self, frame_input: &mut FrameInput) -> bool {
        self.gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |gui_context| {
                use three_d::egui::{Button, Checkbox, Label, Slider, SliderClamping};
                egui::Window::new("").show(gui_context, |ui| {
                    ui.add(Label::new(format!("FPS: {}", self.frame_counter.fps)));
                    ui.add(
                        Slider::new(&mut self.frame_rate, 0.0..=200.)
                            .clamping(SliderClamping::Never)
                            .text("Target Framerate"),
                    );
                    ui.add(
                        Slider::new(&mut self.config_grid_size.x, 1..=1000)
                            .clamping(SliderClamping::Never)
                            .text("Grid width"),
                    );
                    ui.add(
                        Slider::new(&mut self.config_grid_size.y, 1..=1000)
                            .clamping(SliderClamping::Never)
                            .text("Grid height"),
                    );
                    if ui.add(Button::new("Apply Grid Size")).clicked() {
                        self.should_resize_grid = true;
                    }
                    if ui.add(Button::new("Rerandomize")).clicked() {
                        self.conway_texture
                            .get_active_mut()
                            .fill(&Self::generate_random_conway_data(self.grid_size));
                    }
                    if ui.add(Button::new("Clear")).clicked() {
                        self.conway_texture
                            .get_active_mut()
                            .fill(&Self::generate_empty_conway_data(self.grid_size));
                    }
                    if ui.add(Button::new("Save")).clicked() {
                        self.should_save = true;
                    }
                    if ui.add(Button::new("Load")).clicked() {
                        self.should_load = true;
                    }
                    ui.add(Checkbox::new(&mut self.paused, "Pause")).clicked();
                });
            },
        )
    }

    fn draw_pixel(&mut self, frame_input: &FrameInput, position: PhysicalPoint, erase: bool) {
        // normalized x and y in the windowspace
        let x_window = -2. * (f64::from(position.x) / f64::from(frame_input.viewport.width) - 0.5);
        let y_window = -2. * (f64::from(position.y) / f64::from(frame_input.viewport.height) - 0.5);
        // normalized x and y on the texture
        let x_tex_normal = (x_window / f64::from(self.zoom.z)) + f64::from(self.zoom.x);
        let y_tex_normal = (y_window / f64::from(self.zoom.z)) + f64::from(self.zoom.y);

        let gsx = self.grid_size.x as f32;
        let gsy = self.grid_size.y as f32;
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
        let tex = self.conway_texture.get_active_mut();
        tex.as_color_target(None)
            .write_partially::<CoreError>(sbox, || {
                self.edit_program
                    .use_vertex_attribute("position", &self.fullscreen_quad);
                self.edit_program.use_uniform(
                    "color",
                    if erase {
                        vec4(0., 0., 0., 1.)
                    } else {
                        vec4(1., 0., 0., 1.)
                    },
                );
                self.edit_program.draw_arrays(
                    RenderStates::default(),
                    frame_input.viewport,
                    self.fullscreen_quad.vertex_count(),
                );
                Ok(())
            })
            .unwrap();
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
}
