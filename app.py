import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import warnings
import numpy as np
import importlib.util
import sys
from torchvision.models import VGG16_Weights, ResNet18_Weights, AlexNet_Weights

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Globals
selected_model = None
features = {}
layer_shapes = {}
input_image = None
selected_model_name = ""
mode = None  # Will be initialized in main_screen()
model_file_path = None

# Ask user whether to download pretrained weights for all models
def ask_download_weights():
    while True:
        ans = input("Do you want to download pretrained weights for all models (VGG16, ResNet18, and AlexNet)? (yes/no): ").strip().lower()
        if ans in ['yes', 'y', 'Y']:
            return True
        elif ans in ['no', 'n', 'N']:
            return False
        else:
            print("Please enter 'yes' or 'no'.")

# Decide weights based on user input
def get_model(model_name, download_weights):
    if model_name == "VGG16":
        return models.vgg16(weights=VGG16_Weights.DEFAULT if download_weights else None)
    elif model_name == "ResNet18":
        return models.resnet18(weights=ResNet18_Weights.DEFAULT if download_weights else None)
    elif model_name == "AlexNet":
        return models.alexnet(weights=AlexNet_Weights.DEFAULT if download_weights else None)
    else:
        raise ValueError(f"Unknown model: {model_name}")

# Ask user once if they want to download weights for all models
download_weights = ask_download_weights()

# Load models conditionally based on user input
PRETRAINED_MODELS = {
    "VGG16": get_model("VGG16", download_weights),
    "ResNet18": get_model("ResNet18", download_weights),
    "AlexNet": get_model("AlexNet", download_weights)
}


import os
from tkinter import messagebox, simpledialog
from PIL import Image
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def preprocess_image(image_path, transform_choice):
    global mode
    global width_var, height_var, channel_var

    # Check if mode is 'Pretrained'
    if mode.get() == "Pretrained":
        if transform_choice == "Resize":
            transform = transforms.Compose([
                transforms.Resize(224, interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif transform_choice == "Crop":
            # Check if the image dimensions are large enough
            image = Image.open(image_path)
            if image.size[0] >= 224 and image.size[1] >= 224:
                # No need to resize, just apply center crop
                transform = transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                # Resize first if the image is too small, then apply center crop
                transform = transforms.Compose([
                    transforms.Resize(224, interpolation=Image.BILINEAR),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        elif transform_choice == "None":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        image = Image.open(image_path).convert("RGB")
    
    else:  # For other modes
        input_width = int(width_var.get())  # Get width input value
        input_height = int(height_var.get())  # Get height input value
        
        image = Image.open(image_path)
        
        if channel_var.get():  # RGB mode
            image = image.convert("RGB")
            num_channels = 3
        else:  # Grayscale mode
            image = image.convert("L")
            num_channels = 1

        # Select the transformation based on user input
        if transform_choice == "Resize":
            transform = transforms.Compose([
                transforms.Resize((input_height, input_width)),
                transforms.ToTensor(),
                # pixel values are in the range [0, 1] after ToTensor,
                # input is transformed from [0, 1] → [-1, 1], centered around 0
                transforms.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels)
            ])
            
        elif transform_choice == "Crop":
            transform = transforms.Compose([
                transforms.CenterCrop((input_height, input_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels)
            ])
            
        elif transform_choice == "None":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels)
            ])

    return transform(image).unsqueeze(0)


# Extract features
def extract_features(layer_name, arch_window=None):
    global selected_model, features
    
    if arch_window is not None:
        arch_window.destroy()  # Close the architecture window

    if selected_model is None:
        messagebox.showerror("Error", "Please load/select a model first!")
        return

    # New window with button to select folder
    folder_prompt_window = tk.Toplevel(root)
    folder_prompt_window.title("Image Folder Selection")
    folder_prompt_window.geometry("400x200")  # Set width x height

    # Optional: add some padding and centering
    frame = ttk.Frame(folder_prompt_window, padding=20)
    frame.pack(expand=True, fill="both")

    ttk.Label(frame, text=f"Layer: {layer_name}", font=("Arial", 10)).pack(pady=20)
    ttk.Button(frame, text="Step 3: Select Image Folder",
            command=lambda: handle_folder_selection(folder_prompt_window, layer_name)).pack(pady=10)



def handle_folder_selection(window, layer_name):
    global selected_model, selected_model_name

    folder_path = filedialog.askdirectory(title="Select Folder of Images")
    if not folder_path:
        messagebox.showwarning("Warning", "No image folder selected.")
        return
    
    # Ask the user for the transformation they want, just once
    transform_choice = simpledialog.askstring(
        "Select Transformation",
        "Choose one: Resize, Crop, None",
        parent=root  # Assuming 'root' is the Tkinter root window
    )

    # Validate the user input for transformation choice
    if transform_choice not in ["Resize", "Crop", "None"]:
        messagebox.showerror("Invalid Choice", "Please choose either 'Resize', 'Crop', or 'None'.")
        return

    # Close the prompt window
    window.destroy()

    # Hook for the layer
    # runs during the forward pass of a model layer
    def hook_fn(module, input, output):
        hook_fn.output = output.detach().numpy() # saves output to a static variable hook_fn.output.

    hook_fn.output = None
    for name, module in selected_model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook_fn) #attaches the hook_fn to the target layer.
            break
    else:
        messagebox.showerror("Error", f"Layer {layer_name} not found!")
        return

    # Process all images
    failed = []
    success_count = 0
    saved_file_paths = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(folder_path, filename)
            try:
                image_tensor = preprocess_image(file_path, transform_choice)
                with torch.no_grad():
                    selected_model(image_tensor)
                if hook_fn.output is not None:
                    out_folder = os.path.join("features", selected_model_name, f"{layer_name} ({transform_choice})")
                    os.makedirs(out_folder, exist_ok=True)
                    save_path = os.path.join(out_folder, os.path.splitext(filename)[0] + ".npy")
                    np.save(save_path, hook_fn.output)

                    # Store relative path instead of full path
                    relative_path = os.path.relpath(save_path, start="features")
                    saved_file_paths.append(relative_path)

                    success_count += 1
            except Exception as e:
                failed.append(filename)

    handle.remove()

    saved_files = "\n".join(saved_file_paths)

    msg = (
        f"Model: {selected_model_name}\n"
        f"Layer: {layer_name}\n"
        f"Successfully extracted features for {success_count} images.\n"
    )
    if failed:
        msg += f"Failed for {len(failed)} images.\n"

    msg += f"\nSaved .npy files:\n{saved_files}"

    messagebox.showinfo("Done", msg)

# Compute layer shapes
def compute_layer_shapes():
    global selected_model, layer_shapes, mode
    global width_var, height_var, channel_var
    
    if selected_model is None:
        return

    layer_shapes.clear()

    if mode.get() == "Pretrained":
        dummy_input = torch.randn(1, 3, 224, 224)
    else:
        # Decide number of channels based on the checkbox (True = RGB = 3, False = Grayscale = 1)
        channels = 3 if channel_var.get() else 1
        
        input_width = int(width_var.get())
        input_height = int(height_var.get())
        dummy_input = torch.randn(1, channels, input_height, input_width)
    hooks = []

    def hook_fn(name):
        def hook(module, input, output):
            layer_shapes[name] = list(output.shape)
        return hook

    for name, module in selected_model.named_modules():
        hooks.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        selected_model(dummy_input)

    for hook in hooks:
        hook.remove()

# Show model architecture
def show_model_architecture():
    global selected_model
    if not selected_model:
        messagebox.showerror("Error", "Please load/select a model first!")
        return

    compute_layer_shapes()

    arch_window = tk.Toplevel(root)
    arch_window.title("Model Architecture")
    ttk.Label(arch_window, text="Layer Naming Convention: [Batch, Channels, Height, Width]", font=("Arial", 10)).pack()

    for name, shape in layer_shapes.items():
        shape_text = f"{name} - [{', '.join(map(str, shape))}]"
        ttk.Button(arch_window, text=shape_text,
                   command=lambda n=name, w=arch_window: extract_features(n, w)).pack()

# Load pretrained model
def select_pretrained_model(name):
    global selected_model, selected_model_name
    selected_model = PRETRAINED_MODELS[name]
    selected_model.eval()
    selected_model_name = name
    compute_layer_shapes()
    messagebox.showinfo("Model Selected", f"Loaded pretrained model: {name}")

def show_model_file_contents(file_path):
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Find where class definitions begin, class CustomCNN
        class_start_index = next((i for i, line in enumerate(lines) if line.strip().startswith("class ")), None)

        if class_start_index is None:
            messagebox.showerror("Error", "No class definition found in the model file.")
            return

        class_content = "".join(lines[class_start_index:]) #Joins all lines from the class definition to the end of the file

        # Create a new window to display the content
        viewer = tk.Toplevel() #new pop-up
        viewer.title(f"Viewing Class: {os.path.basename(file_path)}")

        # Text widget with scrollbars
        text_area = tk.Text(viewer, wrap="none")
        text_area.insert("1.0", class_content)
        text_area.config(state="disabled") #disable editing
        text_area.pack(expand=True, fill="both")

        # Add Scrollbars
        y_scroll = tk.Scrollbar(viewer, orient="vertical", command=text_area.yview)
        y_scroll.pack(side="right", fill="y")
        text_area.config(yscrollcommand=y_scroll.set)

        x_scroll = tk.Scrollbar(viewer, orient="horizontal", command=text_area.xview)
        x_scroll.pack(side="bottom", fill="x")
        text_area.config(xscrollcommand=x_scroll.set)

    except Exception as e:
        messagebox.showerror("Error", f"Could not read model file:\n{str(e)}")

import tkinter.simpledialog

import tkinter as tk
from tkinter import ttk, messagebox

def choose_model_class(class_names): #from list of classes in model definition, choose one
    chooser = tk.Toplevel()
    chooser.title("Choose Model Class")
    chooser.geometry("300x250")  # Set window size

    # Styling for the window
    chooser.configure(bg="white")

    # Add label with a cleaner font and padding
    label = tk.Label(chooser, text="Select a model class:", font=("Arial", 11), bg="white")
    label.pack(pady=10)

    # Create a scrollable list of class names
    listbox_frame = ttk.Frame(chooser)
    listbox_frame.pack(padx=10, pady=5, fill="both", expand=True)

    # Listbox with scroll bar
    listbox = tk.Listbox(listbox_frame, selectmode=tk.SINGLE, font=("Arial", 10), height=6)
    listbox.pack(side="left", fill="both", expand=True)

    scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=listbox.yview)
    scrollbar.pack(side="right", fill="y")
    listbox.config(yscrollcommand=scrollbar.set)

    for name in class_names:
        listbox.insert(tk.END, name)

    selected = tk.StringVar()

    # Confirm selection button
    def confirm_selection():
        try:
            selected.set(listbox.get(listbox.curselection()))
            chooser.destroy()
        except:
            messagebox.showwarning("Warning", "Please select a class.")

    # Use a button with a more refined appearance
    confirm_btn = ttk.Button(chooser, text="Confirm", command=confirm_selection, width=15)
    confirm_btn.pack(pady=10)

    # Optional: Add a close button in case the user wants to cancel
    close_btn = ttk.Button(chooser, text="Cancel", command=chooser.destroy, width=15)
    close_btn.pack(pady=5)

    # Keep the window on top and modal
    chooser.transient()  # keep on top
    chooser.grab_set()   # block interaction with other windows
    chooser.wait_window()  # wait until the window is closed

    return selected.get() if selected.get() else None


# Load model from .py
def select_model_file():
    global selected_model, selected_model_name, model_file_path

    file_path = filedialog.askopenfilename(filetypes=[("Python Files", "*.py")]) #Allows the user to select a custom .py file that contains a model definition.
    if not file_path:
        messagebox.showwarning("Warning", "No model file selected.")
        return

    try:
        module_name = os.path.splitext(os.path.basename(file_path))[0] #before .py name of model, Gets the file name
        spec = importlib.util.spec_from_file_location(module_name, file_path) # (blueprint) for how to load the file.
        model_module = importlib.util.module_from_spec(spec) #actual Python module object.
        sys.modules[module_name] = model_module
        spec.loader.exec_module(model_module) # loads the Python file and executes the code

        model_classes = [attr for attr in dir(model_module) if isinstance(getattr(model_module, attr), type)] # lists all items (functions, classes, variables) inside the loaded module.
        #Keeps only the class names (ignores functions, variables, etc.).
        
        if not model_classes: # filters for only class names 
            messagebox.showerror("Error", "No model class found in the selected file.")
            return

        chosen_class_name = choose_model_class(model_classes) #ask the user to pick one of the classes found as the main model.
        if not chosen_class_name:
            messagebox.showinfo("Info", "No class selected.")
            return

        model_class = getattr(model_module, chosen_class_name) #Retrieves the chosen class using getattr().
        selected_model = model_class() # creates an instance of it and 
        selected_model_name = model_class.__name__ #stores the object and class name.

        messagebox.showinfo("Architecture Loaded", f"{selected_model_name} architecture loaded. Now load weights.")

        
        # show_model_file_contents(file_path)
        model_file_path = file_path 

    except Exception as e:
        messagebox.showerror("Error", f"Failed to import model:\n{str(e)}")

# Load weights
#  load the .pth file (model weights) into a previously loaded model
def load_model_weights():
    global selected_model, model_file_path

    if selected_model is None:
        messagebox.showerror("Error", "Please load a model architecture file first!")
        return

    file_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pth")]) #.pth files store saved weights of a PyTorch model
    if not file_path:
        messagebox.showwarning("Warning", "No model weights selected.")
        return

    try:
        # load_state_dict() inserts those weights
        selected_model.load_state_dict(torch.load(file_path, map_location=torch.device("cpu"))) #Loads the model’s saved parameters using PyTorch’s torch.load()
        selected_model.eval() #in inference mode
        
        
        
        # compute_layer_shapes()
        messagebox.showinfo("Model Loaded", f"{selected_model_name} weights loaded successfully!")
        
        if model_file_path:
            show_model_file_contents(model_file_path)
            
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model weights:\n{str(e)}")

# GUI
def main_screen():
    global root, mode
    global width_var, height_var, channel_var
    root = tk.Tk()
    mode = tk.StringVar(value="Pretrained")

    # root.title("Unified PyTorch Feature Extractor")
    root.title("Model Feature Extractor")

    ttk.Label(root, text="Model Feature Extractor", font=("Arial", 14)).pack(pady=10)

    # Mode selector
    mode_frame = ttk.LabelFrame(root, text="Model Mode")
    mode_frame.pack(pady=5, padx=10, fill="x")
    ttk.Radiobutton(mode_frame, text="Pretrained", variable=mode, value="Pretrained").pack(side="left", padx=10)
    ttk.Radiobutton(mode_frame, text="Custom (.py + .pth)", variable=mode, value="Custom").pack(side="left", padx=10)

    # Dynamic model selection section
    model_options_frame = ttk.LabelFrame(root, text="Step 1: Select Model")
    model_options_frame.pack(pady=10, padx=10, fill="x")

    def update_buttons():
        
        global width_var, height_var, channel_var
        
        for widget in model_options_frame.winfo_children():
            widget.destroy()

        if mode.get() == "Pretrained":
            for model_name in PRETRAINED_MODELS.keys():
                ttk.Button(model_options_frame, text=f" {model_name} ",
                           command=lambda n=model_name: select_pretrained_model(n)).pack(pady=2)
        else:
            
            ttk.Button(model_options_frame, text="Select Model Definition (.py)",
                       command=select_model_file).pack(pady=2)
            ttk.Button(model_options_frame, text="Load Model Weights (.pth)",
                       command=load_model_weights).pack(pady=2)
            
            # Add width and height input fields for custom models
            ttk.Label(model_options_frame, text="Input Dimensions (m X n):").pack(pady=5)

            # Create a frame to hold the width and height entries and center it
            dim_frame = ttk.Frame(model_options_frame)
            dim_frame.pack(pady=5)  # Centered by default inside model_options_frame

            width_var = tk.StringVar(value="")  # No default value
            height_var = tk.StringVar(value="")  # No default value

            ttk.Entry(dim_frame, textvariable=width_var, width=10).pack(side="left", padx=5)
            ttk.Label(dim_frame, text="X").pack(side="left")
            ttk.Entry(dim_frame, textvariable=height_var, width=10).pack(side="left", padx=5)
            
            
            # Add checkbox to select whether to use RGB for images
            channel_var = tk.BooleanVar(value=True)  # Default to RGB
            ttk.Checkbutton(model_options_frame, text="Use RGB Channel (True/False)", variable=channel_var).pack(pady=5)

    mode.trace_add("write", lambda *args: update_buttons()) #which calls a function whenever the value of the variable changes.
    update_buttons()

    ttk.Button(root, text="Step 2: View Model Architecture", command=show_model_architecture).pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    main_screen()
    
"""
HOW TO RUN

1)
32 X 32 
checkbox = True

custom_cnn_definition.py
saved_models/custom_cnn.pth

Screenshots/folder1



2)
28 X 28 
checkbox = False

custom_cnn2_definition.py
saved_models/custom_cnn2.pth

Screenshots/folder1



"""     




"""
ask_download_weights() -> get_model() ->  main_screen() -> 
                                                            1) select_pretrained_model() -> compute_layer_shapes() -> hook_fn()
                                                            2) select_model_file() -> choose_model_class() 
                                                                                   -> getattr()
                                                            3) load_model_weights() -> show_model_file_contents()
                                                            4) show_model_architecture() -> compute_layer_shapes()
                                                                                         -> extract_features()
                                                                                                -> handle_folder_selection()
                                                                                                        -> preprocess_image()


"""
