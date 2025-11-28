#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

use three_d::{FrameInputGenerator, SurfaceSettings, WindowedContext};
use winit::{
    dpi::PhysicalSize, event::Event, event_loop::EventLoop,
    platform::wayland::WindowBuilderExtWayland as _, window::WindowBuilder,
};

mod game_of_life;
use crate::game_of_life::{GameOfLifeRunner, Operation};

fn main() {
    const INIT_WINDOW_SIZE: PhysicalSize<u32> = PhysicalSize::new(1000, 1000);

    let event_loop = EventLoop::new();
    let window_builder = WindowBuilder::new()
        .with_title("Conway's Game of life")
        .with_name("conway-rs", "conway-rs")
        .with_inner_size(INIT_WINDOW_SIZE);

    let window = window_builder.build(&event_loop).unwrap();
    let context = WindowedContext::from_winit_window(
        &window,
        SurfaceSettings {
            vsync: false,
            ..Default::default()
        },
    )
    .unwrap();

    let mut game_of_life_runner = GameOfLifeRunner::new(&context, INIT_WINDOW_SIZE);

    if let Some(file_to_load) = std::env::args().nth(1) {
        match game_of_life_runner.load_file(file_to_load.into(), &context) {
            Ok(()) => {}
            Err(e) => match e.kind() {
                std::io::ErrorKind::NotFound => println!("Couldn't load file: File not found"),
                _ => println!("Couldn't load file: {e}"),
            },
        }
    }

    let mut frame_input_generator = FrameInputGenerator::from_winit_window(&window);

    event_loop.run(move |event, _, control_flow| match event {
        Event::MainEventsCleared => window.request_redraw(),
        Event::RedrawRequested(_) => {
            let mut frame_input = frame_input_generator.generate(&context);
            let need_redraw = game_of_life_runner.render_frame(&mut frame_input);
            if need_redraw {
                match context.swap_buffers() {
                    Ok(()) => {}
                    Err(e) => println!("ERROR: Failed to swap Framebuffers: {e}"),
                }
            }
            control_flow.set_poll();
            window.request_redraw();
        }
        Event::WindowEvent { ref event, .. } => {
            frame_input_generator.handle_winit_window_event(event);
            match event {
                winit::event::WindowEvent::Resized(physical_size) => {
                    context.resize(*physical_size);
                    game_of_life_runner
                        .add_operation(Operation::WindowResizeAdjust(*physical_size));
                }
                winit::event::WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    context.resize(**new_inner_size);

                    game_of_life_runner
                        .add_operation(Operation::WindowResizeAdjust(**new_inner_size));
                }
                winit::event::WindowEvent::CloseRequested => {
                    control_flow.set_exit();
                }
                _ => (),
            }
        }
        _ => {}
    })
}
