import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import warnings
import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("Running on GPU⚡")
except ImportError:
    GPU_AVAILABLE = False
    print("Running on CPU 💻")
import importlib.util
import sys
import logging
from pathlib import Path
from glob import glob
import torchvision.models as models
from torchvision.models import (
    VGG11_Weights, VGG11_BN_Weights, VGG13_Weights, VGG13_BN_Weights,
    VGG16_Weights, VGG16_BN_Weights, VGG19_Weights, VGG19_BN_Weights,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights,
    ResNeXt50_32X4D_Weights, ResNeXt101_32X8D_Weights, ResNeXt101_64X4D_Weights,
    Wide_ResNet50_2_Weights, Wide_ResNet101_2_Weights,
    AlexNet_Weights, DenseNet121_Weights, DenseNet161_Weights, DenseNet169_Weights, DenseNet201_Weights,
    Inception_V3_Weights, GoogLeNet_Weights, MobileNet_V2_Weights, MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights, MNASNet0_5_Weights, MNASNet0_75_Weights, MNASNet1_0_Weights,
    MNASNet1_3_Weights, EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights,
    EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights, EfficientNet_B6_Weights,
    EfficientNet_B7_Weights, EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights,
    RegNet_X_400MF_Weights, RegNet_X_800MF_Weights, RegNet_X_1_6GF_Weights, RegNet_X_3_2GF_Weights,
    RegNet_X_8GF_Weights, RegNet_X_16GF_Weights, RegNet_X_32GF_Weights,
    RegNet_Y_400MF_Weights, RegNet_Y_800MF_Weights, RegNet_Y_1_6GF_Weights, RegNet_Y_3_2GF_Weights,
    RegNet_Y_8GF_Weights, RegNet_Y_16GF_Weights, RegNet_Y_32GF_Weights, RegNet_Y_128GF_Weights,
    ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights, ViT_H_14_Weights,
    Swin_T_Weights, Swin_S_Weights, Swin_B_Weights, Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights,
    MaxVit_T_Weights, ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights
)
import torch

try:
    import customtkinter as ctk
    print("Running on CustomTkinter")
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
except ImportError:
    print("Running on Fallback Tkinter")
    import tkinter as tk
    from tkinter import ttk
    ctk = None
from tkinter import filedialog, messagebox, simpledialog
from visualisation_app import NpyVisualizerApp  # Import the optimized NpyVisualizerApp

# Setup logging
logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore", category=UserWarning)

# Globals
selected_model = None
features = {}
layer_shapes = {}
input_image = None
selected_model_name = ""
mode = None
model_file_path = None
npz_file_path = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.debug(f"Using device: {device}")

MODELS_DICT = {
    "VGG11": (models.vgg11, VGG11_Weights.DEFAULT),
    "VGG11_BN": (models.vgg11_bn, VGG11_BN_Weights.DEFAULT),
    "VGG13": (models.vgg13, VGG13_Weights.DEFAULT),
    "VGG13_BN": (models.vgg13_bn, VGG13_BN_Weights.DEFAULT),
    "VGG16": (models.vgg16, VGG16_Weights.DEFAULT),
    "VGG16_BN": (models.vgg16_bn, VGG16_BN_Weights.DEFAULT),
    "VGG19": (models.vgg19, VGG19_Weights.DEFAULT),
    "VGG19_BN": (models.vgg19_bn, VGG19_BN_Weights.DEFAULT),
    "ResNet18": (models.resnet18, ResNet18_Weights.DEFAULT),
    "ResNet34": (models.resnet34, ResNet34_Weights.DEFAULT),
    "ResNet50": (models.resnet50, ResNet50_Weights.DEFAULT),
    "ResNet101": (models.resnet101, ResNet101_Weights.DEFAULT),
    "ResNet152": (models.resnet152, ResNet152_Weights.DEFAULT),
    "ResNeXt50_32x4d": (models.resnext50_32x4d, ResNeXt50_32X4D_Weights.DEFAULT),
    "ResNeXt101_32x8d": (models.resnext101_32x8d, ResNeXt101_32X8D_Weights.DEFAULT),
    "ResNeXt101_64x4d": (models.resnext101_64x4d, ResNeXt101_64X4D_Weights.DEFAULT),
    "Wide_ResNet50_2": (models.wide_resnet50_2, Wide_ResNet50_2_Weights.DEFAULT),
    "Wide_ResNet101_2": (models.wide_resnet101_2, Wide_ResNet101_2_Weights.DEFAULT),
    "AlexNet": (models.alexnet, AlexNet_Weights.DEFAULT),
    "DenseNet121": (models.densenet121, DenseNet121_Weights.DEFAULT),
    "DenseNet161": (models.densenet161, DenseNet161_Weights.DEFAULT),
    "DenseNet169": (models.densenet169, DenseNet169_Weights.DEFAULT),
    "DenseNet201": (models.densenet201, DenseNet201_Weights.DEFAULT),
    "Inception_V3": (models.inception_v3, Inception_V3_Weights.DEFAULT),
    "GoogLeNet": (models.googlenet, GoogLeNet_Weights.DEFAULT),
    "MobileNet_V2": (models.mobilenet_v2, MobileNet_V2_Weights.DEFAULT),
    "MobileNet_V3_Small": (models.mobilenet_v3_small, MobileNet_V3_Small_Weights.DEFAULT),
    "MobileNet_V3_Large": (models.mobilenet_v3_large, MobileNet_V3_Large_Weights.DEFAULT),
    "MNASNet0_5": (models.mnasnet0_5, MNASNet0_5_Weights.DEFAULT),
    "MNASNet0_75": (models.mnasnet0_75, MNASNet0_75_Weights.DEFAULT),
    "MNASNet1_0": (models.mnasnet1_0, MNASNet1_0_Weights.DEFAULT),
    "MNASNet1_3": (models.mnasnet1_3, MNASNet1_3_Weights.DEFAULT),
    "EfficientNet_B0": (models.efficientnet_b0, EfficientNet_B0_Weights.DEFAULT),
    "EfficientNet_B1": (models.efficientnet_b1, EfficientNet_B1_Weights.DEFAULT),
    "EfficientNet_B2": (models.efficientnet_b2, EfficientNet_B2_Weights.DEFAULT),
    "EfficientNet_B3": (models.efficientnet_b3, EfficientNet_B3_Weights.DEFAULT),
    "EfficientNet_B4": (models.efficientnet_b4, EfficientNet_B4_Weights.DEFAULT),
    "EfficientNet_B5": (models.efficientnet_b5, EfficientNet_B5_Weights.DEFAULT),
    "EfficientNet_B6": (models.efficientnet_b6, EfficientNet_B6_Weights.DEFAULT),
    "EfficientNet_B7": (models.efficientnet_b7, EfficientNet_B7_Weights.DEFAULT),
    "EfficientNet_V2_S": (models.efficientnet_v2_s, EfficientNet_V2_S_Weights.DEFAULT),
    "EfficientNet_V2_M": (models.efficientnet_v2_m, EfficientNet_V2_M_Weights.DEFAULT),
    "EfficientNet_V2_L": (models.efficientnet_v2_l, EfficientNet_V2_L_Weights.DEFAULT),
    "RegNet_X_400MF": (models.regnet_x_400mf, RegNet_X_400MF_Weights.DEFAULT),
    "RegNet_X_800MF": (models.regnet_x_800mf, RegNet_X_800MF_Weights.DEFAULT),
    "RegNet_X_1_6GF": (models.regnet_x_1_6gf, RegNet_X_1_6GF_Weights.DEFAULT),
    "RegNet_X_3_2GF": (models.regnet_x_3_2gf, RegNet_X_3_2GF_Weights.DEFAULT),
    "RegNet_X_8GF": (models.regnet_x_8gf, RegNet_X_8GF_Weights.DEFAULT),
    "RegNet_X_16GF": (models.regnet_x_16gf, RegNet_X_16GF_Weights.DEFAULT),
    "RegNet_X_32GF": (models.regnet_x_32gf, RegNet_X_32GF_Weights.DEFAULT),
    "RegNet_Y_400MF": (models.regnet_y_400mf, RegNet_Y_400MF_Weights.DEFAULT),
    "RegNet_Y_800MF": (models.regnet_y_800mf, RegNet_Y_800MF_Weights.DEFAULT),
    "RegNet_Y_1_6GF": (models.regnet_y_1_6gf, RegNet_Y_1_6GF_Weights.DEFAULT),
    "RegNet_Y_3_2GF": (models.regnet_y_3_2gf, RegNet_Y_3_2GF_Weights.DEFAULT),
    "RegNet_Y_8GF": (models.regnet_y_8gf, RegNet_Y_8GF_Weights.DEFAULT),
    "RegNet_Y_16GF": (models.regnet_y_16gf, RegNet_Y_16GF_Weights.DEFAULT),
    "RegNet_Y_32GF": (models.regnet_y_32gf, RegNet_Y_32GF_Weights.DEFAULT),
    "RegNet_Y_128GF": (models.regnet_y_128gf, RegNet_Y_128GF_Weights.DEFAULT),
    "ViT_B_16": (models.vit_b_16, ViT_B_16_Weights.DEFAULT),
    "ViT_B_32": (models.vit_b_32, ViT_B_32_Weights.DEFAULT),
    "ViT_L_16": (models.vit_l_16, ViT_L_16_Weights.DEFAULT),
    "ViT_L_32": (models.vit_l_32, ViT_L_32_Weights.DEFAULT),
    "ViT_H_14": (models.vit_h_14, ViT_H_14_Weights.DEFAULT),
    "Swin_T": (models.swin_t, Swin_T_Weights.DEFAULT),
    "Swin_S": (models.swin_s, Swin_S_Weights.DEFAULT),
    "Swin_B": (models.swin_b, Swin_B_Weights.DEFAULT),
    "Swin_V2_T": (models.swin_v2_t, Swin_V2_T_Weights.DEFAULT),
    "Swin_V2_S": (models.swin_v2_s, Swin_V2_S_Weights.DEFAULT),
    "Swin_V2_B": (models.swin_v2_b, Swin_V2_B_Weights.DEFAULT),
    "MaxVit_T": (models.maxvit_t, MaxVit_T_Weights.DEFAULT),
    "ConvNeXt_Tiny": (models.convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
    "ConvNeXt_Small": (models.convnext_small, ConvNeXt_Small_Weights.DEFAULT),
    "ConvNeXt_Base": (models.convnext_base, ConvNeXt_Base_Weights.DEFAULT),
    "ConvNeXt_Large": (models.convnext_large, ConvNeXt_Large_Weights.DEFAULT),
}


def to_device(tensor):
    """Move tensor to the appropriate device (GPU/CPU)."""
    return tensor.to(device)

def to_gpu(arr):
    """Transfer array to GPU if available, else keep on CPU."""
    return cp.asarray(arr) if GPU_AVAILABLE else np.asarray(arr)

def to_cpu(arr):
    """Transfer array to CPU for NumPy operations."""
    return cp.asnumpy(arr) if GPU_AVAILABLE and isinstance(arr, cp.ndarray) else np.asarray(arr)

def get_model(model_name, download_weights, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load a pretrained model from torchvision.models with optional weights."""
    if model_name not in MODELS_DICT:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODELS_DICT.keys())}")
    
    model_fn, weights = MODELS_DICT[model_name]
    model = model_fn(weights=weights if download_weights else None)
    return model.to(device)

    


def ask_download_weights():
    """Prompt user to download pretrained weights."""
    while True:
        ans = input("Download pretrained weights for the models? (yes/no): ").strip().lower()
        if ans in ['yes', 'y']:
            return True
        elif ans in ['no', 'n']:
            return False
        print("Please enter 'yes' or 'no'.")

download_weights = ask_download_weights()
# Initialize only selected models to save memory; dropdown will handle loading
PRETRAINED_MODELS = {}  # We'll load models on-demand when selected from dropdown

def preprocess_image(image_path, transform_choice, mode_value, target_size, use_rgb):
    """Preprocess image based on mode and transformation choice."""
    try:
        image = Image.open(image_path)
        num_channels = 3 if use_rgb or mode_value == "Pretrained" else 1
        image = image.convert("RGB" if num_channels == 3 else "L")
        
        transform = transforms.Compose([
            transforms.Resize(target_size if transform_choice != "Crop" else max(target_size), 
                             interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(target_size) if transform_choice == "Crop" else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406] if num_channels == 3 else [0.5],
                std=[0.229, 0.224, 0.225] if num_channels == 3 else [0.5]
            )
        ])
        return to_device(transform(image).unsqueeze(0))
    except Exception as e:
        logging.error(f"Failed to preprocess image {image_path}: {str(e)}")
        raise ValueError(f"Failed to preprocess image {image_path}: {str(e)}")

def extract_features(layer_name, arch_window=None):
    """Extract features from images for a specific layer."""
    global selected_model, selected_model_name, npz_file_path, mode, width_var, height_var, channel_var
    if arch_window:
        arch_window.destroy()
    if not selected_model:
        messagebox.showerror("Error", "Please load/select a model first!")
        return

    folder_path = filedialog.askdirectory(title="Select Folder of Images")
    if not folder_path:
        messagebox.showwarning("Warning", "No image folder selected.")
        return

    transform_choice = simpledialog.askstring("Select Transformation", "Choose one: Resize, Crop, None", parent=root)
    if transform_choice not in ["Resize", "Crop", "None"]:
        messagebox.showerror("Invalid Choice", "Please choose either 'Resize', 'Crop', or 'None'.")
        return

    image_files = [os.path.join(r, f) for r, _, fs in os.walk(folder_path) 
                   for f in fs if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        messagebox.showerror("Error", "No valid image files found in the selected folder.")
        return

    target_size = (224, 224) if mode.get() == "Pretrained" else (int(height_var.get()), int(width_var.get()))
    use_rgb = channel_var.get() if mode.get() == "Custom" else True

    # Create loading screen
    loading_window = ctk.CTkToplevel(root) if ctk else tk.Toplevel(root)
    loading_window.title("Processing Images")
    loading_window.geometry("400x150")
    loading_window.transient(root)
    loading_window.grab_set()
    frame = ctk.CTkFrame(loading_window) if ctk else ttk.Frame(loading_window, padding=20)
    frame.pack(expand=True, fill="both")
    progress_label = ctk.CTkLabel(frame, text=f"Processing: 0/{len(image_files)} images") if ctk else ttk.Label(frame, text=f"Processing: 0/{len(image_files)} images")
    progress_label.pack(pady=5)
    file_label = ctk.CTkLabel(frame, text="Current file: None") if ctk else ttk.Label(frame, text="Current file: None")
    file_label.pack(pady=5)
    progress_bar = ctk.CTkProgressBar(frame, mode="determinate") if ctk else ttk.Progressbar(frame, mode="determinate", maximum=len(image_files))
    progress_bar.pack(pady=10, fill="x")
    if ctk:
        progress_bar.set(0)

    def hook_fn(module, input, output):
        out = output
        if out.dim() == 4:
            out = nn.AdaptiveAvgPool2d((1, 1))(out).squeeze(-1).squeeze(-1)
        hook_fn.output = out.detach().cpu().numpy()

    hook_fn.output = None
    handle = None
    for name, module in selected_model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook_fn)
            break
    else:
        loading_window.destroy()
        messagebox.showerror("Error", f"Layer {layer_name} not found!")
        return

    failed = []
    saved_file_paths = []
    feature_list = []
    label_list = []
    for i, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        progress_label.configure(text=f"Processing: {i}/{len(image_files)} images") if ctk else progress_label.config(text=f"Processing: {i}/{len(image_files)} images")
        file_label.configure(text=f"Current file: {filename}") if ctk else file_label.config(text=f"Current file: {filename}")
        if ctk:
            progress_bar.set(i / len(image_files))
        else:
            progress_bar["value"] = i
        root.update()
        try:
            image_tensor = preprocess_image(image_path, transform_choice, mode.get(), target_size, use_rgb)
            with torch.no_grad():
                selected_model(image_tensor)
            if hook_fn.output is not None:
                out_folder = Path("features") / selected_model_name / f"{layer_name} ({transform_choice})" / os.path.relpath(os.path.dirname(image_path), folder_path)
                out_folder.mkdir(parents=True, exist_ok=True)
                save_path = out_folder / f"{os.path.splitext(filename)[0]}.npy"
                np.save(save_path, hook_fn.output)
                saved_file_paths.append(str(save_path.relative_to("features")))
                feature_list.append(to_gpu(hook_fn.output.flatten()))
                label_list.append(os.path.basename(os.path.dirname(image_path)))
        except Exception as e:
            failed.append(f"{filename}: {str(e)}")

    handle.remove()
    loading_window.destroy()

    if feature_list:
        try:
            features_array = to_cpu(np.vstack([to_cpu(f) for f in feature_list]))
            labels_array = np.array(label_list)
            npz_save_path = Path("features") / selected_model_name / f"{layer_name} ({transform_choice})" / f"{selected_model_name}_{layer_name}_{transform_choice}_all_classes.npz"
            npz_save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(npz_save_path, features=features_array, labels=labels_array)
            npz_file_path = str(npz_save_path.resolve())
            saved_file_paths.append(str(npz_save_path.relative_to("features")))
            with open("last_npz_path.txt", "w") as f:
                f.write(npz_file_path)
            logging.debug(f"Saved NPZ file: {npz_file_path}")
        except Exception as e:
            logging.error(f"Failed to save .npz file: {e}")
            messagebox.showerror("Error", f"Failed to save .npz file: {e}")
            return

    msg = f"Model: {selected_model_name}\nLayer: {layer_name}\nSuccessfully extracted features for {len(feature_list)} images.\n"
    if failed:
        msg += f"Failed for {len(failed)} images:\n{', '.join(failed)}\n"
    msg += f"Saved files:\n{', '.join(saved_file_paths)}"
    messagebox.showinfo("Done", msg)

def compute_layer_shapes():
    """Compute output shapes for all model layers."""
    global selected_model, layer_shapes, mode, width_var, height_var, channel_var
    if not selected_model:
        return
    layer_shapes.clear()
    channels = 3 if mode.get() == "Pretrained" or channel_var.get() else 1
    input_size = (224, 224) if mode.get() == "Pretrained" else (int(height_var.get()), int(width_var.get()))
    dummy_input = to_device(torch.randn(1, channels, *input_size))
    hooks = []

    def hook_fn(name):
        def hook(module, input, output):
            out = output
            if out.dim() == 4:
                out = nn.AdaptiveAvgPool2d((1, 1))(out).squeeze(-1).squeeze(-1)
            layer_shapes[name] = list(out.shape)
        return hook

    for name, module in selected_model.named_modules():
        hooks.append(module.register_forward_hook(hook_fn(name)))

    try:
        with torch.no_grad():
            selected_model(dummy_input)
    except Exception as e:
        logging.error(f"Failed to compute layer shapes: {e}")
        messagebox.showerror("Error", f"Failed to compute layer shapes: {e}")
    finally:
        for hook in hooks:
            hook.remove()

def show_model_architecture():
    """Display model architecture with layer shapes and extraction buttons."""
    global selected_model
    if not selected_model:
        messagebox.showerror("Error", "Please load/select a model first!")
        return

    compute_layer_shapes()
    if not layer_shapes:
        messagebox.showerror("Error", "No layers found in the model!")
        return

    arch_window = ctk.CTkToplevel(root) if ctk else tk.Toplevel(root)
    arch_window.title("Model Architecture")
    arch_window.geometry("600x500")
    if not ctk:
        arch_window.configure(bg="#f7f9fc")

    (ctk.CTkLabel(arch_window, text="Layer Naming Convention: [Batch, Features]") if ctk else
     ttk.Label(arch_window, text="Layer Naming Convention: [Batch, Features]", font=("Segoe UI", 10), background="#f7f9fc")).pack(pady=5)

    canvas = ctk.CTkCanvas(arch_window) if ctk else tk.Canvas(arch_window, bg="#ffffff", highlightthickness=0)
    scrollbar = ctk.CTkScrollbar(arch_window, command=canvas.yview) if ctk else ttk.Scrollbar(arch_window, orient="vertical", command=canvas.yview)
    scrollable_frame = ctk.CTkFrame(canvas) if ctk else ttk.Frame(canvas)
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True, padx=10, pady=5)
    scrollbar.pack(side="right", fill="y")

    def on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", on_mousewheel)
    canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
    canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    for name, shape in layer_shapes.items():
        (ctk.CTkButton(scrollable_frame, text=f"{name} - [{', '.join(map(str, shape))}]",
                       command=lambda n=name: extract_features(n, arch_window)) if ctk else
         ttk.Button(scrollable_frame, text=f"{name} - [{', '.join(map(str, shape))}]",
                    command=lambda n=name: extract_features(n, arch_window))).pack(anchor="w", pady=4, padx=10, fill="x")

def select_pretrained_model(name):
    """Select a pretrained model."""
    global selected_model, selected_model_name, npz_file_path
    # Load model on-demand if not already loaded
    if name not in PRETRAINED_MODELS:
        PRETRAINED_MODELS[name] = get_model(name, download_weights)
    selected_model = PRETRAINED_MODELS[name]
    selected_model.eval()
    selected_model_name = name
    npz_file_path = None
    logging.debug(f"Selected pretrained model: {name}")
    compute_layer_shapes()
    messagebox.showinfo("Model Selected", f"Loaded pretrained model: {name}")


def show_model_file_contents(file_path):
    """Display contents of a model definition file."""
    try:
        with open(file_path, "r") as file:
            class_content = "".join(line for line in file if line.strip().startswith("class "))
        if not class_content:
            messagebox.showerror("Error", "No class definition found in the model file.")
            return
        viewer = ctk.CTkToplevel() if ctk else tk.Toplevel()
        viewer.title(f"Viewing Class: {os.path.basename(file_path)}")
        text_area = ctk.CTkTextbox(viewer, wrap="none") if ctk else tk.Text(viewer, wrap="none")
        text_area.insert("1.0", class_content)
        if not ctk:
            text_area.config(state="disabled")
        text_area.pack(expand=True, fill="both")
        if not ctk:
            y_scroll = tk.Scrollbar(viewer, orient="vertical", command=text_area.yview)
            x_scroll = tk.Scrollbar(viewer, orient="horizontal", command=text_area.xview)
            y_scroll.pack(side="right", fill="y")
            x_scroll.pack(side="bottom", fill="x")
            text_area.config(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
    except Exception as e:
        logging.error(f"Could not read model file: {e}")
        messagebox.showerror("Error", f"Could not read model file: {str(e)}")

def choose_model_class(class_names):
    """Prompt user to select a model class from a list."""
    chooser = ctk.CTkToplevel() if ctk else tk.Toplevel()
    chooser.title("Choose Model Class")
    chooser.geometry("300x250")
    if not ctk:
        chooser.configure(bg="white")
    (ctk.CTkLabel(chooser, text="Select a model class:") if ctk else
     tk.Label(chooser, text="Select a model class:", font=("Arial", 11), bg="white")).pack(pady=10)
    
    listbox_frame = ctk.CTkFrame(chooser) if ctk else ttk.Frame(chooser)
    listbox_frame.pack(padx=10, pady=5, fill="both", expand=True)
    listbox = tk.Listbox(listbox_frame, selectmode=tk.SINGLE, font=("Arial", 10), height=6)
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar = ctk.CTkScrollbar(listbox_frame, command=listbox.yview) if ctk else ttk.Scrollbar(listbox_frame, orient="vertical", command=listbox.yview)
    scrollbar.pack(side="right", fill="y")
    listbox.config(yscrollcommand=scrollbar.set)
    for name in class_names:
        listbox.insert(tk.END, name)
    selected = ctk.StringVar() if ctk else tk.StringVar()

    def confirm_selection():
        try:
            selected.set(listbox.get(listbox.curselection()))
            chooser.destroy()
        except:
            messagebox.showwarning("Warning", "Please select a class.")

    (ctk.CTkButton(chooser, text="Confirm", command=confirm_selection) if ctk else
     ttk.Button(chooser, text="Confirm", command=confirm_selection)).pack(pady=10)
    (ctk.CTkButton(chooser, text="Cancel", command=chooser.destroy) if ctk else
     ttk.Button(chooser, text="Cancel", command=chooser.destroy)).pack(pady=5)
    chooser.transient(root)
    chooser.grab_set()
    chooser.wait_window()
    return selected.get()

def select_model_file():
    """Load a custom model definition from a .py file."""
    global selected_model, selected_model_name, model_file_path, npz_file_path
    file_path = filedialog.askopenfilename(filetypes=[("Python Files", "*.py")])
    if not file_path:
        messagebox.showwarning("Warning", "No model file selected.")
        return
    try:
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        model_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = model_module
        spec.loader.exec_module(model_module)
        model_classes = [attr for attr in dir(model_module) if isinstance(getattr(model_module, attr), type)]
        if not model_classes:
            messagebox.showerror("Error", "No model class found in the selected file.")
            return
        chosen_class_name = choose_model_class(model_classes)
        if not chosen_class_name:
            messagebox.showinfo("Info", "No class selected.")
            return
        selected_model = getattr(model_module, chosen_class_name)().to(device)
        selected_model_name = chosen_class_name
        npz_file_path = None
        model_file_path = file_path
        logging.debug(f"Loaded custom model: {selected_model_name}")
        messagebox.showinfo("Architecture Loaded", f"{selected_model_name} architecture loaded. Now load weights.")
    except Exception as e:
        logging.error(f"Failed to import model: {e}")
        messagebox.showerror("Error", f"Failed to import model: {str(e)}")

def load_model_weights():
    """Load weights for a custom model from a .pth file."""
    global selected_model, selected_model_name, npz_file_path
    if not selected_model:
        messagebox.showerror("Error", "Please load a model architecture file first!")
        return
    file_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pth")])
    if not file_path:
        messagebox.showwarning("Warning", "No model weights selected.")
        return
    try:
        selected_model.load_state_dict(torch.load(file_path, map_location=device))
        selected_model.eval()
        npz_file_path = None
        logging.debug(f"Loaded weights for {selected_model_name}")
        messagebox.showinfo("Model Loaded", f"{selected_model_name} weights loaded successfully!")
        if model_file_path:
            show_model_file_contents(model_file_path)
    except Exception as e:
        logging.error(f"Failed to load model weights: {e}")
        messagebox.showerror("Error", f"Failed to load model weights: {str(e)}")

def run_visualisation():
    """Launch NpyVisualizerApp with the extracted features."""
    global npz_file_path, dim_var, algo_var, dis_var
    if not npz_file_path:
        last_npz_path_file = Path("last_npz_path.txt")
        if last_npz_path_file.exists():
            with open(last_npz_path_file, "r") as f:
                npz_file_path = f.read().strip()
            logging.debug(f"Loaded npz_file_path from last_npz_path.txt: {npz_file_path}")
        else:
            npz_files = glob("features/**/*.npz", recursive=True)
            if npz_files:
                npz_file_path = max(npz_files, key=os.path.getmtime)
                logging.debug(f"Fallback to latest .npz file: {npz_file_path}")

    if not npz_file_path or not Path(npz_file_path).exists():
        logging.error(f"No valid feature file found. Path: {npz_file_path}")
        messagebox.showerror("Error", f"No valid feature file found. Please extract features first! Path: {npz_file_path}")
        return

    try:
        npz_file_path = str(Path(npz_file_path).resolve())
        logging.debug(f"Launching NpyVisualizerApp with: {npz_file_path}")
        viz_window = ctk.CTkToplevel(root) if ctk else tk.Toplevel(root)
        viz_window.title("CNN Embedding Analysis")
        app = NpyVisualizerApp(viz_window, dim_var.get(), algo_var.get(), dis_var.get(), npz_file_path)
    except Exception as e:
        logging.error(f"Failed to launch visualization: {e}")
        messagebox.showerror("Error", f"Failed to launch visualization: {str(e)}")

def main_screen():
    """Create the main GUI for model selection and feature extraction."""
    global root, mode, width_var, height_var, channel_var, dim_var, algo_var, dis_var
    root = ctk.CTk() if ctk else tk.Tk()
    root.title("Model Feature Extractor")
    root.geometry("700x750")
    if not ctk:
        root.configure(bg="#f7f9fc")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", font=("Segoe UI", 10), background="#f7f9fc")
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("TCheckbutton", font=("Segoe UI", 10))
        style.configure("TCombobox", font=("Segoe UI", 10), padding=6)
        style.configure("Modern.TLabelframe", background="#ffffff", borderwidth=1, relief="solid")
        style.configure("Modern.TLabelframe.Label", background="#ffffff", font=("Segoe UI", 11, "bold"))

    (ctk.CTkLabel(root, text="Model Feature Extractor", font=("Segoe UI", 14, "bold")) if ctk else
     ttk.Label(root, text="Model Feature Extractor", font=("Segoe UI", 14, "bold"))).pack(pady=20)

    mode = ctk.StringVar(value="Pretrained") if ctk else tk.StringVar(value="Pretrained")
    mode_frame = ctk.CTkFrame(root) if ctk else ttk.LabelFrame(root, text="Model Mode", style="Modern.TLabelframe")
    mode_frame.pack(pady=10, padx=20, fill="x")
    (ctk.CTkRadioButton(mode_frame, text="Pretrained", variable=mode, value="Pretrained") if ctk else
     ttk.Radiobutton(mode_frame, text="Pretrained", variable=mode, value="Pretrained")).pack(side="left", padx=20, pady=10)
    (ctk.CTkRadioButton(mode_frame, text="Custom (.py + .pth)", variable=mode, value="Custom") if ctk else
     ttk.Radiobutton(mode_frame, text="Custom (.py + .pth)", variable=mode, value="Custom")).pack(side="left", padx=20)

    model_frame = ctk.CTkFrame(root) if ctk else ttk.LabelFrame(root, text="Step 1: Select Model", style="Modern.TLabelframe")
    model_frame.pack(pady=10, padx=20, fill="x")

    def update_model_frame():
        for widget in model_frame.winfo_children():
            widget.destroy()
        if mode.get() == "Pretrained":
            # Create dropdown for pretrained models
            (ctk.CTkLabel(model_frame, text="Select Pretrained Model:") if ctk else
             ttk.Label(model_frame, text="Select Pretrained Model:")).pack(pady=(10, 3))
            model_var = ctk.StringVar(value="Select a model") if ctk else tk.StringVar(value="Select a model")
            model_names = sorted(MODELS_DICT.keys())  # Sort for better UX
            (ctk.CTkOptionMenu(model_frame, variable=model_var, values=["Select a model"] + model_names,
                               command=lambda name: select_pretrained_model(name) if name != "Select a model" else None) if ctk else
             ttk.Combobox(model_frame, textvariable=model_var, values=["Select a model"] + model_names,
                          state="readonly")).pack(pady=5, padx=20, fill="x")
            if not ctk:
                model_frame.children["!combobox"].bind("<<ComboboxSelected>>",
                    lambda e: select_pretrained_model(model_var.get()) if model_var.get() != "Select a model" else None)
        else:
            (ctk.CTkButton(model_frame, text="Select Model Definition (.py)", command=select_model_file) if ctk else
             ttk.Button(model_frame, text="Select Model Definition (.py)", command=select_model_file)).pack(pady=5, fill="x")
            (ctk.CTkButton(model_frame, text="Load Model Weights (.pth)", command=load_model_weights) if ctk else
             ttk.Button(model_frame, text="Load Model Weights (.pth)", command=load_model_weights)).pack(pady=5, fill="x")
            (ctk.CTkLabel(model_frame, text="Input Dimensions (Width x Height):") if ctk else
             ttk.Label(model_frame, text="Input Dimensions (Width x Height):")).pack(pady=(10, 5))
            dim_frame = ctk.CTkFrame(model_frame) if ctk else ttk.Frame(model_frame)
            dim_frame.pack(pady=5)
            global width_var, height_var, channel_var
            width_var = ctk.StringVar(value="224") if ctk else tk.StringVar(value="224")
            height_var = ctk.StringVar(value="224") if ctk else tk.StringVar(value="224")
            (ctk.CTkEntry(dim_frame, textvariable=width_var, width=80) if ctk else
             ttk.Entry(dim_frame, textvariable=width_var, width=8)).pack(side="left", padx=5)
            (ctk.CTkLabel(dim_frame, text="x") if ctk else ttk.Label(dim_frame, text="x")).pack(side="left")
            (ctk.CTkEntry(dim_frame, textvariable=height_var, width=80) if ctk else
             ttk.Entry(dim_frame, textvariable=height_var, width=8)).pack(side="left", padx=5)
            channel_var = ctk.BooleanVar(value=True) if ctk else tk.BooleanVar(value=True)
            (ctk.CTkCheckBox(model_frame, text="Use RGB Channel", variable=channel_var) if ctk else
             ttk.Checkbutton(model_frame, text="Use RGB Channel", variable=channel_var)).pack(pady=10)

    mode.trace_add("write", lambda *args: update_model_frame())
    update_model_frame()

    (ctk.CTkLabel(root, text="Step 2: View Model Architecture", font=("Segoe UI", 14, "bold")) if ctk else
     ttk.Label(root, text="Step 2: View Model Architecture", font=("Segoe UI", 14, "bold"))).pack(pady=5)
    (ctk.CTkButton(root, text="View Model Architecture", command=show_model_architecture) if ctk else
     ttk.Button(root, text="View Model Architecture", command=show_model_architecture)).pack(pady=10, padx=20, fill="x")

    vis_frame = ctk.CTkFrame(root) if ctk else ttk.LabelFrame(root, text="Step 3: Embedding Analysis", style="Modern.TLabelframe")
    vis_frame.pack(pady=10, padx=20, fill="x")
    (ctk.CTkLabel(vis_frame, text="Select Dimension:") if ctk else ttk.Label(vis_frame, text="Select Dimension:")).pack(pady=(10, 3))
    dim_var = ctk.StringVar(value="2D") if ctk else tk.StringVar(value="2D")
    (ctk.CTkOptionMenu(vis_frame, variable=dim_var, values=["2D", "3D"]) if ctk else
     ttk.Combobox(vis_frame, textvariable=dim_var, values=["2D", "3D"], state="readonly")).pack(pady=5, padx=20, fill="x")
    (ctk.CTkLabel(vis_frame, text="Feature Reduction Algorithm:") if ctk else
     ttk.Label(vis_frame, text="Feature Reduction Algorithm:")).pack(pady=(10, 3))
    algo_var = ctk.StringVar(value="PCA") if ctk else tk.StringVar(value="PCA")
    (ctk.CTkOptionMenu(vis_frame, variable=algo_var, values=["PCA", "TruncatedSVD"]) if ctk else
     ttk.Combobox(vis_frame, textvariable=algo_var, values=["PCA", "TruncatedSVD"], state="readonly")).pack(pady=5, padx=20, fill="x")
    (ctk.CTkLabel(vis_frame, text="Distance Metric:") if ctk else ttk.Label(vis_frame, text="Distance Metric:")).pack(pady=(10, 3))
    dis_var = ctk.StringVar(value="Euclidean") if ctk else tk.StringVar(value="euclidean")
    (ctk.CTkOptionMenu(vis_frame, variable=dis_var, values=["euclidean", "cosine", "cityblock", "canberra"]) if ctk else
     ttk.Combobox(vis_frame, textvariable=dis_var, values=["euclidean", "cosine", "cityblock", "canberra"], state="readonly")).pack(pady=5, padx=20, fill="x")
    (ctk.CTkButton(vis_frame, text="Run Embedding Analysis", command=run_visualisation) if ctk else
     ttk.Button(vis_frame, text="Run Embedding Analysis", command=run_visualisation)).pack(pady=15)

    root.mainloop()


if __name__ == "__main__":
    main_screen()