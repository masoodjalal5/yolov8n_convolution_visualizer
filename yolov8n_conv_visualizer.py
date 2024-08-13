# Visualizes input, output and the weights of a convolutional layer
# 'k' for next channel
# 'j' for previous channel
# 'm' for next layer
# 'n' for previous layer

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8n model
model = YOLO('yolov8n.pt').model

# Load and preprocess your image
image_path = "test_img.jpg"  # Replace with the path to your image
image = Image.open(image_path).convert("RGB")

# Define the transformation: Resize to 640x640 and convert to tensor
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Store data for navigation
stored_data = []

# Define a hook function to capture and visualize input and output after each Conv2d layer
def hook_fn(module, input, output, layer_num, layer_name):
    print(f"Layer {layer_num}: {layer_name}")
    
    # Detach input and output and convert to numpy
    input_np = input[0][0].detach().cpu().numpy()
    output_np = output[0].detach().cpu().numpy()

    print(f"Input shape: {input[0].shape}")
    print(f"Output shape: {output.shape}")
    print("\n")
    
    # Store the data
    stored_data.append({
        'layer_num': layer_num,
        'layer_name': layer_name,
        'input': input_np,
        'output': output_np,
        'kernel_weights': module.weight.detach().cpu().numpy() if isinstance(module, nn.Conv2d) else None
    })

# Register hooks on each Conv2d layer with layer numbers and names
hooks = []
layer_count = 0
for layer in model.modules():
    if isinstance(layer, nn.Conv2d):  # If the layer is a Conv2d layer
        hooks.append(layer.register_forward_hook(
            lambda module, input, output, layer_num=layer_count, layer_name=layer.__class__.__name__:
            hook_fn(module, input, output, layer_num, layer_name)
        ))
        layer_count += 1

# Run the image through the model
output = model(input_tensor)

# Remove hooks after use
for hook in hooks:
    hook.remove()

# Function to visualize channels and kernels
def visualize_layer_data(layer_data, channel_index=0):
    input_np = layer_data['input']
    output_np = layer_data['output']
    kernel_weights = layer_data['kernel_weights']

    print(f"Visualizing Layer {layer_data['layer_num']} - {layer_data['layer_name']}")
    print(f"Channel: {channel_index+1}/{output_np.shape[0]}")

    # Clear the axes
    axes[0].clear()
    axes[1].clear()
    axes[2].clear()

    # Input Visualization
    if input_np.shape[0] == 3:  # If the input has 3 channels, visualize as an image
        input_np = np.moveaxis(input_np, 0, -1)  # Convert from (C, H, W) to (H, W, C)
        axes[0].imshow(input_np)
        axes[0].set_title(f'Input of {layer_data["layer_name"]}')
    else:  # For other cases, visualize the first channel only
        axes[0].imshow(input_np[0], cmap='gray')
        axes[0].set_title(f'Input of {layer_data["layer_name"]}')

    # Output Visualization
    axes[1].imshow(output_np[channel_index], cmap='gray')
    axes[1].set_title(f'Output of {layer_data["layer_name"]}, Channel {channel_index+1}')

    # Kernel Weights Visualization (if available)
    if kernel_weights is not None:
        axes[2].imshow(kernel_weights[channel_index][0], cmap='gray')
        axes[2].set_title(f'Kernel Weights, Channel {channel_index+1}')
    else:
        axes[2].axis('off')

    # Redraw the figure with the updated content
    fig.canvas.draw()

# Interactive display of channels
current_layer_index = 0
current_channel_index = 0

def update_visualization():
    global current_layer_index, current_channel_index
    layer_data = stored_data[current_layer_index]
    visualize_layer_data(layer_data, current_channel_index)

def on_key(event):
    global current_layer_index, current_channel_index

    if event.key == 'k':  # Next channel
        current_channel_index = (current_channel_index + 1) % stored_data[current_layer_index]['output'].shape[0]
        update_visualization()
    elif event.key == 'j':  # Previous channel
        current_channel_index = (current_channel_index - 1) % stored_data[current_layer_index]['output'].shape[0]
        update_visualization()
    elif event.key == 'm':  # Next layer
        current_layer_index = (current_layer_index + 1) % len(stored_data)
        current_channel_index = 0
        update_visualization()
    elif event.key == 'n':  # Previous layer
        current_layer_index = (current_layer_index - 1) % len(stored_data)
        current_channel_index = 0
        update_visualization()

# Create the figure and axes once
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Connect the key press event
fig.canvas.mpl_connect('key_press_event', on_key)

# Initial display
update_visualization()

plt.show()

