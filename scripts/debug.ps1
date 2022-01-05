$env:VK_LAYER_PATH = "$env:VULKAN_SDK\Bin"
$env:VK_INSTANCE_LAYERS = "VK_LAYER_KHRONOS_validation"

cargo run

$env:VK_LAYER_PATH = ""
$env:VK_INSTANCE_LAYERS = ""
