use anyhow::Result;
use ash::vk::{self};
use gpu_allocator::MemoryLocation;
use gui::{
    imgui::{DrawData, Ui},
    imgui_rs_vulkan_renderer::Renderer,
    GuiContext,
};
use simple_logger::SimpleLogger;
use std::{marker::PhantomData, time::Instant};
use vulkan::*;
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const IN_FLIGHT_FRAMES: u32 = 2;
const VULKAN_VERSION: VkVersion = VkVersion::from_major_minor(1, 3);

pub struct BaseApp<B: App> {
    phantom: PhantomData<B>,
    pub swapchain: VkSwapchain,
    render_pass: VkRenderPass,
    framebuffers: Vec<VkFramebuffer>,
    pub command_pool: VkCommandPool,
    pub storage_images: Vec<ImageAndView>,
    command_buffers: Vec<VkCommandBuffer>,
    in_flight_frames: InFlightFrames,
    pub context: VkContext,
}

pub trait App: Sized {
    type Gui: Gui;

    fn new(base: &mut BaseApp<Self>) -> Result<Self>;

    fn update(&self, base: &BaseApp<Self>, gui: &mut Self::Gui, image_index: usize) -> Result<()>;

    fn record_command(
        &self,
        base: &BaseApp<Self>,
        buffer: &VkCommandBuffer,
        image_index: usize,
    ) -> Result<()>;

    fn on_recreate_swapchain(&self, storage_images: &[ImageAndView]) -> Result<()>;
}

pub trait Gui: Sized {
    fn new() -> Result<Self>;

    fn build(&mut self, ui: &Ui);
}

impl Gui for () {
    fn new() -> Result<Self> {
        Ok(())
    }

    fn build(&mut self, _ui: &Ui) {}
}

pub fn run<A: App + 'static>(app_name: &str, width: u32, height: u32) -> Result<()> {
    SimpleLogger::default().env().init()?;

    let (window, event_loop) = create_window(app_name, width, height);
    let mut base_app = BaseApp::new(&window, app_name, width, height)?;
    let mut ui = A::Gui::new()?;
    let app = A::new(&mut base_app)?;
    let mut gui_context = GuiContext::new(
        &base_app.context,
        &base_app.context.command_pool,
        &base_app.render_pass,
        &window,
        IN_FLIGHT_FRAMES as _,
    )?;
    let mut is_swapchain_dirty = false;
    let mut last_frame = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        let app = &app; // Make sure it is dropped before base_app

        gui_context.handle_event(&window, &event);

        match event {
            Event::NewEvents(_) => {
                let now = Instant::now();
                gui_context.update_delta_time(now - last_frame);
                last_frame = now;
            }
            // On resize
            Event::WindowEvent {
                event: WindowEvent::Resized(..),
                ..
            } => {
                log::debug!("Window has been resized");
                is_swapchain_dirty = true;
            }
            // Draw
            Event::MainEventsCleared => {
                if is_swapchain_dirty {
                    let dim = window.inner_size();
                    if dim.width > 0 && dim.height > 0 {
                        base_app
                            .recreate_swapchain(dim.width, dim.height)
                            .expect("Failed to recreate swapchain");
                        app.on_recreate_swapchain(base_app.storage_images.as_slice())
                            .expect("Error on recreate swapchain callback");
                        gui_context
                            .set_render_pass(&base_app.render_pass)
                            .expect("Failed to set gui render pass");
                    } else {
                        return;
                    }
                }

                is_swapchain_dirty = base_app
                    .draw(&window, app, &mut gui_context, &mut ui)
                    .expect("Failed to tick");
            }
            // Exit app on request to close window
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            // Wait for gpu to finish pending work before closing app
            Event::LoopDestroyed => base_app
                .wait_for_gpu()
                .expect("Failed to wait for gpu to finish work"),
            _ => (),
        }
    });
}

fn create_window(app_name: &str, width: u32, height: u32) -> (Window, EventLoop<()>) {
    log::debug!("Creating window and event loop");
    let events_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(app_name)
        .with_inner_size(PhysicalSize::new(width, height))
        .with_resizable(true)
        .build(&events_loop)
        .unwrap();

    (window, events_loop)
}

impl<B: App> BaseApp<B> {
    fn new(window: &Window, app_name: &str, width: u32, height: u32) -> Result<Self> {
        log::info!("Create application");

        // Vulkan context
        let required_extensions = [
            "VK_KHR_swapchain",
            "VK_KHR_ray_tracing_pipeline",
            "VK_KHR_acceleration_structure",
            "VK_KHR_deferred_host_operations",
        ];
        let mut context =
            VkContext::new(window, VULKAN_VERSION, Some(app_name), &required_extensions)?;

        let command_pool = context.create_command_pool(
            context.graphics_queue_family,
            Some(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
        )?;

        let swapchain = VkSwapchain::new(&context, width, height)?;

        let render_pass = create_render_pass(&context, &swapchain)?;

        let framebuffers = swapchain.get_framebuffers(&render_pass)?;

        let storage_images = create_storage_images(
            &mut context,
            swapchain.format,
            swapchain.extent,
            swapchain.images.len(),
        )?;

        let command_buffers = create_command_buffers(&command_pool, &swapchain)?;

        let in_flight_frames = InFlightFrames::new(&context, IN_FLIGHT_FRAMES)?;

        Ok(Self {
            phantom: PhantomData,
            context,
            command_pool,
            swapchain,
            render_pass,
            framebuffers,
            storage_images,
            command_buffers,
            in_flight_frames,
        })
    }

    fn recreate_swapchain(&mut self, width: u32, height: u32) -> Result<()> {
        log::debug!("Recreating the swapchain");

        self.wait_for_gpu()?;

        // Swapchain and dependent resources
        self.framebuffers.clear();
        self.swapchain.resize(&self.context, width, height)?;

        let render_pass = create_render_pass(&self.context, &self.swapchain)?;
        let _ = std::mem::replace(&mut self.render_pass, render_pass);

        let framebuffers = self.swapchain.get_framebuffers(&self.render_pass)?;
        let _ = std::mem::replace(&mut self.framebuffers, framebuffers);

        // Recreate storage image for RT and update descriptor set
        let storage_images = create_storage_images(
            &mut self.context,
            self.swapchain.format,
            self.swapchain.extent,
            self.swapchain.images.len(),
        )?;

        let _ = std::mem::replace(&mut self.storage_images, storage_images);

        Ok(())
    }

    pub fn wait_for_gpu(&self) -> Result<()> {
        self.context.device_wait_idle()
    }

    fn draw(
        &mut self,
        window: &Window,
        base_app: &B,
        gui_context: &mut GuiContext,
        gui: &mut B::Gui,
    ) -> Result<bool> {
        // Generate UI

        gui_context
            .platform
            .prepare_frame(gui_context.imgui.io_mut(), window)?;
        let ui = gui_context.imgui.frame();

        gui.build(&ui);

        gui_context.platform.prepare_render(&ui, window);
        let draw_data = ui.render();

        // Drawing the frame
        self.in_flight_frames.next();
        self.in_flight_frames.fence().wait(None)?;

        let next_image_result = self.swapchain.acquire_next_image(
            std::u64::MAX,
            self.in_flight_frames.image_available_semaphore(),
        );
        let image_index = match next_image_result {
            Ok(AcquiredImage { index, .. }) => index as usize,
            Err(err) => match err.downcast_ref::<vk::Result>() {
                Some(&vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(true),
                _ => panic!("Error while acquiring next image. Cause: {}", err),
            },
        };

        base_app.update(self, gui, image_index)?;

        self.in_flight_frames.fence().reset()?;

        let command_buffer = &self.command_buffers[image_index];

        self.record_command_buffer(
            command_buffer,
            image_index,
            base_app,
            &mut gui_context.renderer,
            draw_data,
        )?;

        self.context.graphics_queue.submit(
            command_buffer,
            Some(self.in_flight_frames.image_available_semaphore()),
            Some(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT),
            Some(self.in_flight_frames.render_finished_semaphore()),
            self.in_flight_frames.fence(),
        )?;

        let signal_semaphores = [self.in_flight_frames.render_finished_semaphore()];
        let present_result = self.swapchain.queue_present(
            image_index as _,
            &signal_semaphores,
            &self.context.present_queue,
        );
        match present_result {
            Ok(true) => return Ok(true),
            Err(err) => match err.downcast_ref::<vk::Result>() {
                Some(&vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(true),
                _ => panic!("Failed to present queue. Cause: {}", err),
            },
            _ => {}
        }

        Ok(false)
    }

    fn record_command_buffer(
        &self,
        buffer: &VkCommandBuffer,
        image_index: usize,
        base_app: &B,
        gui_renderer: &mut Renderer,
        draw_data: &DrawData,
    ) -> Result<()> {
        let swapchain_image = &self.swapchain.images[image_index];
        let framebuffer = &self.framebuffers[image_index];
        let storage_image = &self.storage_images[image_index];

        let storage_image = &storage_image.image;

        buffer.reset()?;

        buffer.begin(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)?;

        base_app.record_command(self, buffer, image_index)?;

        // Copy ray tracing result into swapchain
        buffer.transition_layout(
            swapchain_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        );

        buffer.transition_layout(
            storage_image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        );

        buffer.copy_image(
            storage_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        buffer.transition_layout(
            swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::COLOR_ATTACHMENT_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        );

        buffer.transition_layout(
            storage_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::TRANSFER_READ,
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TOP_OF_PIPE,
        );

        // Gui pass
        buffer.begin_render_pass(&self.render_pass, framebuffer);

        gui_renderer.cmd_draw(buffer.inner, draw_data)?;

        buffer.end_render_pass();

        buffer.end()?;

        Ok(())
    }
}

fn create_render_pass(context: &VkContext, swapchain: &VkSwapchain) -> Result<VkRenderPass> {
    let attachment_descs = [vk::AttachmentDescription::builder()
        .format(swapchain.format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .build()];

    let color_attachment_refs = [vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build()];

    let subpass_descs = [vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)
        .build()];

    let subpass_deps = [vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )
        .build()];

    let render_pass_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachment_descs)
        .subpasses(&subpass_descs)
        .dependencies(&subpass_deps);

    context.create_render_pass(&render_pass_info)
}

fn create_storage_images(
    context: &mut VkContext,
    format: vk::Format,
    extent: vk::Extent2D,
    count: usize,
) -> Result<Vec<ImageAndView>> {
    let mut images = Vec::with_capacity(count);

    for _ in 0..count {
        let image = context.create_image(
            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::STORAGE,
            MemoryLocation::GpuOnly,
            format,
            extent.width,
            extent.height,
        )?;

        let view = image.create_image_view()?;

        context.execute_one_time_commands(|cmd_buffer| {
            cmd_buffer.transition_layout(
                &image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
            );
        })?;

        images.push(ImageAndView { image, view })
    }

    Ok(images)
}

fn create_command_buffers(
    pool: &VkCommandPool,
    swapchain: &VkSwapchain,
) -> Result<Vec<VkCommandBuffer>> {
    pool.allocate_command_buffers(vk::CommandBufferLevel::PRIMARY, swapchain.images.len() as _)
}

pub struct ImageAndView {
    pub view: VkImageView,
    pub image: VkImage,
}

struct InFlightFrames {
    sync_objects: Vec<SyncObjects>,
    current_frame: usize,
}

struct SyncObjects {
    image_available_semaphore: VkSemaphore,
    render_finished_semaphore: VkSemaphore,
    fence: VkFence,
}

impl InFlightFrames {
    fn new(context: &VkContext, frame_count: u32) -> Result<Self> {
        let sync_objects = (0..frame_count)
            .map(|_i| {
                let image_available_semaphore = context.create_semaphore()?;
                let render_finished_semaphore = context.create_semaphore()?;
                let fence = context.create_fence(Some(vk::FenceCreateFlags::SIGNALED))?;

                Ok(SyncObjects {
                    image_available_semaphore,
                    render_finished_semaphore,
                    fence,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            sync_objects,
            current_frame: 0,
        })
    }

    fn next(&mut self) {
        self.current_frame = (self.current_frame + 1) % self.sync_objects.len();
    }

    fn image_available_semaphore(&self) -> &VkSemaphore {
        &self.sync_objects[self.current_frame].image_available_semaphore
    }

    fn render_finished_semaphore(&self) -> &VkSemaphore {
        &self.sync_objects[self.current_frame].render_finished_semaphore
    }

    fn fence(&self) -> &VkFence {
        &self.sync_objects[self.current_frame].fence
    }
}
