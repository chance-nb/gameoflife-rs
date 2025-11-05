use three_d::{FrameInputGenerator, SurfaceSettings, WindowedContext};
use winit::{
    dpi::PhysicalSize, event::Event, event_loop::EventLoop,
    platform::wayland::WindowBuilderExtWayland, window::WindowBuilder,
};

use crate::game_of_life::ConwaysGameOfLife;

mod game_of_life;

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

    let mut conways_game_of_life = ConwaysGameOfLife::new(&context, INIT_WINDOW_SIZE);

    let mut frame_input_generator = FrameInputGenerator::from_winit_window(&window);

    event_loop.run(move |event, _, control_flow| match event {
        Event::MainEventsCleared => window.request_redraw(),
        Event::RedrawRequested(_) => {
            let mut frame_input = frame_input_generator.generate(&context);
            let need_redraw = conways_game_of_life.render_frame(&mut frame_input);
            if need_redraw {
                match context.swap_buffers() {
                    Ok(_) => {}
                    Err(e) => println!("ERROR: Failed to swap Framebuffers: {}", e),
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
                    conways_game_of_life.resize(*physical_size, &context);
                }
                winit::event::WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    context.resize(**new_inner_size);
                    conways_game_of_life.resize(**new_inner_size, &context);
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
