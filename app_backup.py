# ```python
import tkinter as tk
from tkinter import *
from tkinter import ttk, messagebox, filedialog, simpledialog
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import warnings
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("Running on GPU⚡")
except ImportError:
    import numpy as np
    GPU_AVAILABLE = False
    print("Running on CPU 💻")
import importlib.util
import sys
import logging
from pathlib import Path
from glob import glob
from torchvision.models import VGG16_Weights, ResNet18_Weights, AlexNet_Weights
from visualisation_app import NpyVisualizerApp

# Setup logging
logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Globals
dim_options=""
algo_options=""
selected_model = None
features = {}
layer_shapes = {}
input_image = None
selected_model_name = ""
mode = None
model_file_path = None
npz_file_path = None

def ask_download_weights():
    while True:
        ans = input("Download pretrained weights for VGG16, ResNet18, AlexNet? (yes/no): ").strip().lower()
        if ans in ['yes', 'y']:
            return True
        elif ans in ['no', 'n']:
            return False
        else:
            print("Please enter 'yes' or 'no'.")

def get_model(model_name, download_weights):
    if model_name == "VGG16":
        return models.vgg16(weights=VGG16_Weights.DEFAULT if download_weights else None)
    elif model_name == "ResNet18":
        return models.resnet18(weights=ResNet18_Weights.DEFAULT if download_weights else None)
    elif model_name == "AlexNet":
        return models.alexnet(weights=AlexNet_Weights.DEFAULT if download_weights else None)
    else:
        raise ValueError(f"Unknown model: {model_name}")

download_weights = ask_download_weights()
PRETRAINED_MODELS = {
    "VGG16": get_model("VGG16", download_weights),
    "ResNet18": get_model("ResNet18", download_weights),
    "AlexNet": get_model("AlexNet", download_weights)
}

def preprocess_image(image_path, transform_choice):
    global mode, width_var, height_var, channel_var
    image = Image.open(image_path)
    target_size = (224, 224) if mode.get() == "Pretrained" else (int(height_var.get()), int(width_var.get()))
    num_channels = 3

    try:
        if mode.get() == "Pretrained":
            if transform_choice == "Resize":
                transform = transforms.Compose([
                    transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            elif transform_choice == "Crop":
                transform = transforms.Compose([
                    transforms.Resize(max(target_size), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(target_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            elif transform_choice == "None":
                transform = transforms.Compose([
                    transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            image = image.convert("RGB")
        else:
            if channel_var.get():
                image = image.convert("RGB")
                num_channels = 3
            else:
                image = image.convert("L")
                num_channels = 1
            if transform_choice == "Resize":
                transform = transforms.Compose([
                    transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels)
                ])
            elif transform_choice == "Crop":
                transform = transforms.Compose([
                    transforms.Resize(max(target_size), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(target_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels)
                ])
            elif transform_choice == "None":
                transform = transforms.Compose([
                    transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels)
                ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        logging.error(f"Failed to preprocess image {image_path}: {str(e)}")
        raise ValueError(f"Failed to preprocess image {image_path}: {str(e)}")

def extract_features(layer_name, arch_window=None):
    global selected_model, features
    if arch_window is not None:
        arch_window.destroy()
    if selected_model is None:
        messagebox.showerror("Error", "Please load/select a model first!")
        return

    folder_prompt_window = tk.Toplevel(root)
    folder_prompt_window.title("Image Folder Selection")
    folder_prompt_window.geometry("400x200")
    frame = ttk.Frame(folder_prompt_window, padding=20)
    frame.pack(expand=True, fill="both")
    ttk.Label(frame, text=f"Layer: {layer_name}", font=("Arial", 10)).pack(pady=20)
    ttk.Button(frame, text="Step 3: Select Image Folder",
               command=lambda: handle_folder_selection(folder_prompt_window, layer_name)).pack(pady=10)

def handle_folder_selection(window, layer_name):
    global selected_model, selected_model_name, npz_file_path
    folder_path = filedialog.askdirectory(title="Select Folder of Images")
    if not folder_path:
        messagebox.showwarning("Warning", "No image folder selected.")
        return

    transform_choice = simpledialog.askstring(
        "Select Transformation",
        "Choose one: Resize, Crop, None",
        parent=root
    )
    if transform_choice not in ["Resize", "Crop", "None"]:
        messagebox.showerror("Invalid Choice", "Please choose either 'Resize', 'Crop', or 'None'.")
        return

    window.destroy()

    # Count total images for progress bar
    image_files = []
    for root_dir, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root_dir, filename))
    total_images = len(image_files)
    if total_images == 0:
        messagebox.showerror("Error", "No valid image files found in the selected folder.")
        return

    # Create loading screen
    loading_window = tk.Toplevel(root)
    loading_window.title("Processing Images")
    loading_window.geometry("400x150")
    loading_window.transient(root)
    loading_window.grab_set()
    frame = ttk.Frame(loading_window, padding=20)
    frame.pack(expand=True, fill="both")
    progress_label = ttk.Label(frame, text=f"Processing: 0/{total_images} images")
    progress_label.pack(pady=5)
    file_label = ttk.Label(frame, text="Current file: None")
    file_label.pack(pady=5)
    progress_bar = ttk.Progressbar(frame, mode="determinate", maximum=total_images, value=0)
    progress_bar.pack(pady=10, fill="x")
    root.update()

    def hook_fn(module, input, output):
        if output.dim() == 4:
            output = nn.AdaptiveAvgPool2d((1, 1))(output)
            output = output.squeeze(-1).squeeze(-1)
        hook_fn.output = output.detach().numpy()

    hook_fn.output = None
    for name, module in selected_model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook_fn)
            break
    else:
        loading_window.destroy()
        messagebox.showerror("Error", f"Layer {layer_name} not found!")
        return

    failed = []
    success_count = 0
    saved_file_paths = []
    feature_list = []
    label_list = []
    current_image = 0

    for image_path in image_files:
        current_image += 1
        filename = os.path.basename(image_path)
        try:
            # Update loading screen
            progress_label.config(text=f"Processing: {current_image}/{total_images} images")
            file_label.config(text=f"Current file: {filename}")
            progress_bar["value"] = current_image
            root.update()

            image_tensor = preprocess_image(image_path, transform_choice)
            with torch.no_grad():
                selected_model(image_tensor)
            if hook_fn.output is not None:
                root_dir = os.path.dirname(image_path)
                relative_path = os.path.relpath(root_dir, folder_path)
                out_folder = Path("features") / selected_model_name / f"{layer_name} ({transform_choice})" / relative_path
                out_folder.mkdir(parents=True, exist_ok=True)
                save_path = out_folder / (os.path.splitext(filename)[0] + ".npy")
                np.save(save_path, hook_fn.output)
                saved_file_paths.append(str(Path(save_path).relative_to("features")))
                success_count += 1

                features = hook_fn.output.flatten()
                feature_list.append(features)

                label = os.path.basename(root_dir)
                label_list.append(label)
        except Exception as e:
            failed.append(f"{image_path}: {str(e)}")

    handle.remove()
    loading_window.destroy()

    if feature_list:
        try:
            features_array = np.vstack(feature_list)
            labels_array = np.array(label_list)
            npz_save_path = Path("features") / selected_model_name / f"{layer_name} ({transform_choice})" / f"{selected_model_name}_{layer_name}_{transform_choice}_all_classes.npz"
            npz_save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(npz_save_path, features=features_array, labels=labels_array)
            npz_file_path = str(npz_save_path.resolve())
            saved_file_paths.append(str(npz_save_path.relative_to("features")))
            logging.debug(f"NPZ file saved at: {npz_file_path}")
            print(f"NPZ file saved at: {npz_file_path}")  # Debug
            # Persist npz_file_path
            with open("last_npz_path.txt", "w") as f:
                f.write(npz_file_path)
            logging.debug(f"Saved npz_file_path to last_npz_path.txt: {npz_file_path}")
        except ValueError as e:
            logging.error(f"Failed to save .npz file: {e}")
            messagebox.showerror("Error", f"Failed to save .npz file: {e}")
            return

    saved_files = "\n".join(saved_file_paths)
    msg = (
        f"Model: {selected_model_name}\n"
        f"Layer: {layer_name}\n"
        f"Successfully extracted features for {success_count} images.\n"
    )
    if failed:
        msg += f"Failed for {len(failed)} images:\n{', '.join(str(f) for f in failed)}\n"
    msg += f"\nSaved files:\n{saved_files}"
    messagebox.showinfo("Done", msg)

def compute_layer_shapes():
    global selected_model, layer_shapes, mode, width_var, height_var, channel_var
    if selected_model is None:
        return
    layer_shapes.clear()
    if mode.get() == "Pretrained":
        dummy_input = torch.randn(1, 3, 224, 224)
    else:
        channels = 3 if channel_var.get() else 1
        try:
            input_width = int(width_var.get())
            input_height = int(height_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid width and height values.")
            return
        dummy_input = torch.randn(1, channels, input_height, input_width)
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

    with torch.no_grad():
        try:
            selected_model(dummy_input)
        except Exception as e:
            logging.error(f"Failed to compute layer shapes: {e}")
            messagebox.showerror("Error", f"Failed to compute layer shapes: {e}")

    for hook in hooks:
        hook.remove()

def show_model_architecture():
    global selected_model
    if not selected_model:
        messagebox.showerror("Error", "Please load/select a model first!")
        return

    compute_layer_shapes()
    if not layer_shapes:
        messagebox.showerror("Error", "No layers found in the model!")
        return

    # Create a new window
    arch_window = tk.Toplevel(root)
    arch_window.title("Model Architecture")
    arch_window.geometry("600x500")
    arch_window.configure(bg="#f7f9fc")

    # Label for naming convention
    ttk.Label(arch_window, text="Layer Naming Convention: [Batch, Features]",
              font=("Segoe UI", 10), background="#f7f9fc").pack(pady=5)

    # Container frame
    container = ttk.Frame(arch_window)
    container.pack(fill="both", expand=True, padx=10, pady=5)

    # Create canvas + scrollbar
    canvas = tk.Canvas(container, borderwidth=0, background="#ffffff", highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    # Configure canvas scrolling
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows/macOS
    canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux scroll up
    canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))   # Linux scroll down

    canvas.configure(yscrollcommand=scrollbar.set)

    # Pack canvas and scrollbar
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Populate scrollable frame with model layers
    for name, shape in layer_shapes.items():
        shape_text = f"{name} - [{', '.join(map(str, shape))}]"
        ttk.Button(scrollable_frame, text=shape_text,
                   command=lambda n=name, w=arch_window: extract_features(n, w)
                   ).pack(anchor="w", pady=4, padx=10, fill="x")


def select_pretrained_model(name):
    global selected_model, selected_model_name, npz_file_path
    selected_model = PRETRAINED_MODELS[name]
    selected_model.eval()
    selected_model_name = name
    npz_file_path = None  # Reset npz_file_path on model change
    logging.debug(f"Model changed to {name}, npz_file_path reset to: {npz_file_path}")
    print(f"Model changed to {name}, npz_file_path reset to: {npz_file_path}")  # Debug
    compute_layer_shapes()
    messagebox.showinfo("Model Selected", f"Loaded pretrained model: {name}")

def show_model_file_contents(file_path):
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
        class_start_index = next((i for i, line in enumerate(lines) if line.strip().startswith("class ")), None)
        if class_start_index is None:
            messagebox.showerror("Error", "No class definition found in the model file.")
            return
        class_content = "".join(lines[class_start_index:])
        viewer = tk.Toplevel()
        viewer.title(f"Viewing Class: {os.path.basename(file_path)}")
        text_area = tk.Text(viewer, wrap="none")
        text_area.insert("1.0", class_content)
        text_area.config(state="disabled")
        text_area.pack(expand=True, fill="both")
        y_scroll = tk.Scrollbar(viewer, orient="vertical", command=text_area.yview)
        y_scroll.pack(side="right", fill="y")
        text_area.config(yscrollcommand=y_scroll.set)
        x_scroll = tk.Scrollbar(viewer, orient="horizontal", command=text_area.xview)
        x_scroll.pack(side="bottom", fill="x")
        text_area.config(xscrollcommand=x_scroll.set)
    except Exception as e:
        logging.error(f"Could not read model file: {e}")
        messagebox.showerror("Error", f"Could not read model file:\n{str(e)}")

def choose_model_class(class_names):
    chooser = tk.Toplevel()
    chooser.title("Choose Model Class")
    chooser.geometry("300x250")
    chooser.configure(bg="white")
    label = tk.Label(chooser, text="Select a model class:", font=("Arial", 11), bg="white")
    label.pack(pady=10)
    listbox_frame = ttk.Frame(chooser)
    listbox_frame.pack(padx=10, pady=5, fill="both", expand=True)
    listbox = tk.Listbox(listbox_frame, selectmode=tk.SINGLE, font=("Arial", 10), height=6)
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=listbox.yview)
    scrollbar.pack(side="right", fill="y")
    listbox.config(yscrollcommand=scrollbar.set)
    for name in class_names:
        listbox.insert(tk.END, name)
    selected = tk.StringVar()

    def confirm_selection():
        try:
            selected.set(listbox.get(listbox.curselection()))
            chooser.destroy()
        except:
            messagebox.showwarning("Warning", "Please select a class.")

    confirm_btn = ttk.Button(chooser, text="Confirm", command=confirm_selection, width=15)
    confirm_btn.pack(pady=10)
    close_btn = ttk.Button(chooser, text="Cancel", command=chooser.destroy, width=15)
    close_btn.pack(pady=5)
    chooser.transient()
    chooser.grab_set()
    chooser.wait_window()
    return selected.get() if selected.get() else None

def select_model_file():
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
        model_class = getattr(model_module, chosen_class_name)
        selected_model = model_class()
        selected_model_name = model_class.__name__
        npz_file_path = None  # Reset npz_file_path on model change
        logging.debug(f"Model file selected, npz_file_path reset to: {npz_file_path}")
        print(f"Model file selected, npz_file_path reset to: {npz_file_path}")  # Debug
        messagebox.showinfo("Architecture Loaded", f"{selected_model_name} architecture loaded. Now load weights.")
        model_file_path = file_path
    except Exception as e:
        logging.error(f"Failed to import model: {e}")
        messagebox.showerror("Error", f"Failed to import model:\n{str(e)}")

def load_model_weights():
    global selected_model, model_file_path, npz_file_path
    if selected_model is None:
        messagebox.showerror("Error", "Please load a model architecture file first!")
        return
    file_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pth")])
    if not file_path:
        messagebox.showwarning("Warning", "No model weights selected.")
        return
    try:
        selected_model.load_state_dict(torch.load(file_path, map_location=torch.device("cpu")))
        selected_model.eval()
        npz_file_path = None  # Reset npz_file_path on weights change
        logging.debug(f"Weights loaded, npz_file_path reset to: {npz_file_path}")
        print(f"Weights loaded, npz_file_path reset to: {npz_file_path}")  # Debug
        messagebox.showinfo("Model Loaded", f"{selected_model_name} weights loaded successfully!")
        if model_file_path:
            show_model_file_contents(model_file_path)
    except Exception as e:
        logging.error(f"Failed to load model weights: {e}")
        messagebox.showerror("Error", f"Failed to load model weights:\n{str(e)}")

def run_visualisation():
    global npz_file_path
    logging.debug(f"run_visualisation called with npz_file_path: {npz_file_path}")
    print(f"run_visualisation called with npz_file_path: {npz_file_path}")  # Debug
    # Load persisted npz_file_path if not set
    if npz_file_path is None:
        last_npz_path_file = Path("last_npz_path.txt")
        if last_npz_path_file.exists():
            with open(last_npz_path_file, "r") as f:
                npz_file_path = f.read().strip()
            logging.debug(f"Loaded npz_file_path from last_npz_path.txt: {npz_file_path}")
            print(f"Loaded npz_file_path from last_npz_path.txt: {npz_file_path}")  # Debug
        else:
            # Fallback to latest .npz file in features directory
            npz_files = glob("features/**/*.npz", recursive=True)
            if npz_files:
                npz_file_path = max(npz_files, key=os.path.getmtime)
                logging.debug(f"Fallback to latest .npz file: {npz_file_path}")
                print(f"Fallback to latest .npz file: {npz_file_path}")  # Debug

    # Normalize path
    if npz_file_path:
        npz_path = Path(npz_file_path)
        try:
            npz_file_path = str(npz_path.resolve())
            logging.debug(f"Normalized npz_file_path: {npz_file_path}")
            print(f"Normalized npz_file_path: {npz_file_path}")  # Debug
        except Exception as e:
            logging.error(f"Failed to resolve path {npz_file_path}: {e}")
            messagebox.showerror("Error", f"Invalid path: {npz_file_path}\n{str(e)}")
            return

    # Check if file exists
    if npz_file_path is None or not Path(npz_file_path).exists():
        logging.error(f"No valid feature file found. Path: {npz_file_path}")
        messagebox.showerror("Error", f"No valid feature file found. Please extract features first! Path: {npz_file_path}")
        return

    # Verify file accessibility
    try:
        with open(npz_file_path, "rb") as f:
            pass
        logging.debug(f"File is accessible: {npz_file_path}")
        print(f"File is accessible: {npz_file_path}")  # Debug
    except Exception as e:
        logging.error(f"Cannot access file {npz_file_path}: {e}")
        messagebox.showerror("Error", f"Cannot access file: {npz_file_path}\n{str(e)}")
        return

    try:
        logging.debug(f"Launching NpyVisualizerApp with: {npz_file_path}")
        print(f"Launching NpyVisualizerApp with: {npz_file_path}")  # Debug
        print(f"File exists check: {os.path.isfile(npz_file_path)}")  # Debug
        viz_window = tk.Toplevel(root)
        viz_window.title("CNN Embedding Analysis")
        app = NpyVisualizerApp(viz_window, dim_var.get(), algo_var.get(), npz_file_path)
    except Exception as e:
        logging.error(f"Failed to launch visualization: {e}")
        messagebox.showerror("Error", f"Failed to launch visualization: {str(e)}")

# import tkinter as tk
# from tkinter import ttk
# from tkinter import StringVar

def main_screen():
    global root, mode, width_var, height_var, channel_var,dim_var, algo_var
    root = tk.Tk()
    root.title("Model Feature Extractor")
    root.geometry("700x750")
    root.configure(bg="#f7f9fc")

    # Modern style
    style = ttk.Style()
    style.theme_use("clam")

    # Universal font
    default_font = ("Segoe UI", 10)
    heading_font = ("Segoe UI", 14, "bold")

    style.configure("TLabel", font=default_font, background="#f7f9fc")
    style.configure("TButton", font=default_font, padding=6)
    style.configure("TCheckbutton", font=default_font)
    style.configure("TCombobox", padding=6)

    # Custom modern label frame
    style.configure("Modern.TLabelframe",
                    background="#ffffff",
                    borderwidth=1,
                    relief="solid")

    style.configure("Modern.TLabelframe.Label",
                    background="#ffffff",
                    font=("Segoe UI", 11, "bold"))

    # Heading
    ttk.Label(root, text="🧠 Model Feature Extractor", font=heading_font, anchor="center").pack(pady=20)

    # Mode selector
    mode = tk.StringVar(value="Pretrained")
    mode_frame = ttk.LabelFrame(root, text="Model Mode", style="Modern.TLabelframe")
    mode_frame.pack(pady=10, padx=20, fill="x")

    ttk.Radiobutton(mode_frame, text="Pretrained", variable=mode, value="Pretrained").pack(side="left", padx=20, pady=10)
    ttk.Radiobutton(mode_frame, text="Custom (.py + .pth)", variable=mode, value="Custom").pack(side="left", padx=20)

    # Model selection
    model_options_frame = ttk.LabelFrame(root, text="Step 1: Select Model", style="Modern.TLabelframe")
    model_options_frame.pack(pady=10, padx=20, fill="x")

    def update_buttons():
        for widget in model_options_frame.winfo_children():
            widget.destroy()
        if mode.get() == "Pretrained":
            for model_name in PRETRAINED_MODELS.keys():
                ttk.Button(model_options_frame, text=model_name,
                           command=lambda n=model_name: select_pretrained_model(n)).pack(pady=4, padx=10, fill="x")
        else:
            ttk.Button(model_options_frame, text="Select Model Definition (.py)", command=select_model_file).pack(pady=5)
            ttk.Button(model_options_frame, text="Load Model Weights (.pth)", command=load_model_weights).pack(pady=5)

            ttk.Label(model_options_frame, text="Input Dimensions (Width x Height):").pack(pady=(10, 5))
            dim_frame = ttk.Frame(model_options_frame)
            dim_frame.pack(pady=5)

            width_var = tk.StringVar(value="224")
            height_var = tk.StringVar(value="224")

            ttk.Entry(dim_frame, textvariable=width_var, width=8).pack(side="left", padx=5)
            ttk.Label(dim_frame, text="x").pack(side="left")
            ttk.Entry(dim_frame, textvariable=height_var, width=8).pack(side="left", padx=5)

            channel_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(model_options_frame, text="Use RGB Channel", variable=channel_var).pack(pady=10)

    mode.trace_add("write", lambda *args: update_buttons())
    update_buttons()

    # Step 2
    stp2frm = ttk.LabelFrame(root, text="Step 2: View Model Architecture", style="Modern.TLabelframe")
    stp2frm.pack(pady=10, padx=20, fill="x")
    ttk.Button(stp2frm, text="View Model Architecture", command=show_model_architecture).pack(pady=10)

    # Step 3
    stp3frm = ttk.LabelFrame(root, text="Step 3: Embedding Analysis", style="Modern.TLabelframe")
    stp3frm.pack(pady=10, padx=20, fill="x")

    ttk.Label(stp3frm, text="Select Dimension:").pack(pady=(10, 3))
    dim_options = ["2D","3D"]
    dim_var = StringVar(value=dim_options[0])
    ttk.Combobox(stp3frm, textvariable=dim_var, values=dim_options, state="readonly").pack(pady=5, padx=20, fill="x")

    ttk.Label(stp3frm, text="Feature Reduction Algorithm:").pack(pady=(10, 3))
    algo_options = ["PCA", "TruncatedSVD"]
    algo_var = StringVar(value=algo_options[0])
    ttk.Combobox(stp3frm, textvariable=algo_var, values=algo_options, state="readonly").pack(pady=5, padx=20, fill="x")

    ttk.Button(stp3frm, text="Run Embedding Analysis", command=run_visualisation).pack(pady=15)

    root.mainloop()


if __name__ == "__main__":
    main_screen()
# ```