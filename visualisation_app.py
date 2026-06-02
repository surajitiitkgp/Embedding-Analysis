
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import tkinter as tk
from tkinter import Frame, ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Circle
import os
import warnings
import logging
from pathlib import Path
import zipfile
from scipy.spatial.distance import pdist, squareform, cdist

# Setup logging
logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.simplefilter("ignore", category=UserWarning)

class NpyVisualizerApp:
    def __init__(self, root, dim_options="2D", algo_options="PCA", dis_opt="euclidean", file_path=None):
        self.root = root
        self.root.title("CNN Embedding Analysis for Semantic Relationships")
        self.root.geometry("1400x900")
        self.root.configure(bg="#121212")
        
        # Setup modern dark theme ttk styling
        self.setup_styles()

        # Main Layout Grid configuration
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Create Left Sidebar Frame
        self.sidebar = tk.Frame(self.root, bg="#1e1e1e", width=380, padx=15, pady=15)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self.sidebar.grid_propagate(False) # Keep width fixed

        # Sidebar Title
        title_label = tk.Label(self.sidebar, text="EMBEDDING VISUALIZER", font=("Segoe UI", 16, "bold"), bg="#1e1e1e", fg="#00adb5")
        title_label.pack(anchor="w", pady=(0, 15))

        # Config Panel (using LabelFrame styled for dark theme)
        self.controls_frame = tk.LabelFrame(self.sidebar, text="Configuration", font=("Segoe UI", 10, "bold"),
                                            bg="#1e1e1e", fg="#00adb5", bd=1, relief="solid", padx=10, pady=10)
        self.controls_frame.pack(fill="x", pady=(0, 15))

        # Setup Config variables
        self.dim = tk.StringVar(value=dim_options)
        self.opt = tk.StringVar(value=algo_options)
        self.distance_metric = tk.StringVar(value=dis_opt)

        # Dimension Dropdown
        lbl_dim = tk.Label(self.controls_frame, text="Dimension:", font=("Segoe UI", 9), bg="#1e1e1e", fg="#ffffff")
        lbl_dim.grid(row=0, column=0, sticky="w", pady=5)
        self.dropdown_dim = tk.OptionMenu(self.controls_frame, self.dim, "2D", "3D")
        self.dropdown_dim.config(bg="#2d2d2d", fg="#ffffff", activebackground="#3d3d3d", activeforeground="#ffffff",
                                 font=("Segoe UI", 9), bd=0, highlightthickness=0)
        self.dropdown_dim["menu"].config(bg="#2d2d2d", fg="#ffffff", activebackground="#00adb5", activeforeground="#ffffff")
        self.dropdown_dim.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=5)

        # Reduction Method Dropdown
        lbl_opt = tk.Label(self.controls_frame, text="Reduction Method:", font=("Segoe UI", 9), bg="#1e1e1e", fg="#ffffff")
        lbl_opt.grid(row=1, column=0, sticky="w", pady=5)
        self.dropdown_opt = tk.OptionMenu(self.controls_frame, self.opt, "PCA", "TruncatedSVD")
        self.dropdown_opt.config(bg="#2d2d2d", fg="#ffffff", activebackground="#3d3d3d", activeforeground="#ffffff",
                                 font=("Segoe UI", 9), bd=0, highlightthickness=0)
        self.dropdown_opt["menu"].config(bg="#2d2d2d", fg="#ffffff", activebackground="#00adb5", activeforeground="#ffffff")
        self.dropdown_opt.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=5)

        # Distance Metric Dropdown
        lbl_dist = tk.Label(self.controls_frame, text="Distance Metric:", font=("Segoe UI", 9), bg="#1e1e1e", fg="#ffffff")
        lbl_dist.grid(row=2, column=0, sticky="w", pady=5)
        self.dropdown_distance = tk.OptionMenu(self.controls_frame, self.distance_metric, "euclidean", "cosine", "cityblock", "canberra")
        self.dropdown_distance.config(bg="#2d2d2d", fg="#ffffff", activebackground="#3d3d3d", activeforeground="#ffffff",
                                      font=("Segoe UI", 9), bd=0, highlightthickness=0)
        self.dropdown_distance["menu"].config(bg="#2d2d2d", fg="#ffffff", activebackground="#00adb5", activeforeground="#ffffff")
        self.dropdown_distance.grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=5)

        self.controls_frame.columnconfigure(1, weight=1)

        # Buttons Panel
        self.buttons_frame = tk.Frame(self.sidebar, bg="#1e1e1e")
        self.buttons_frame.pack(fill="x", pady=(0, 15))

        self.btn_load_folder = tk.Button(self.buttons_frame, text="Load Folder of NPY Files", command=self.load_folder)
        self.btn_load_folder.pack(fill="x", side="top", pady=(0, 8))
        self.style_button(self.btn_load_folder, bg_color="#00adb5", hover_color="#00f5d4")

        self.btn_load_file = tk.Button(self.buttons_frame, text="Load NPZ File", command=self.load_npz)
        self.btn_load_file.pack(fill="x", side="top")
        self.style_button(self.btn_load_file, bg_color="#3a3b3c", hover_color="#5a5b5c")

        # Console Log Panel
        lbl_console = tk.Label(self.sidebar, text="Console Output Logs", font=("Segoe UI", 10, "bold"), bg="#1e1e1e", fg="#ffffff")
        lbl_console.pack(anchor="w", pady=(0, 5))

        console_frame = tk.Frame(self.sidebar, bg="#1e1e1e")
        console_frame.pack(fill="both", expand=True)

        self.metadata_text = tk.Text(console_frame, bg="#121212", fg="#a9b7c6", insertbackground="#ffffff",
                                     font=("Consolas", 9), bd=0, wrap="word")
        self.metadata_text.pack(side="left", fill="both", expand=True)

        self.scrollbar = tk.Scrollbar(console_frame, command=self.metadata_text.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.metadata_text.config(yscrollcommand=self.scrollbar.set)

        # Create Right Main Panel
        self.main_panel = tk.Frame(self.root, bg="#121212")
        self.main_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        self.main_panel.columnconfigure(0, weight=1)
        self.main_panel.rowconfigure(0, weight=1)

        # Notebook tabs
        self.notebook = ttk.Notebook(self.main_panel, style="Custom.TNotebook")
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # Create notebook tabs
        self.tab_plots = tk.Frame(self.notebook, bg="#1e1e1e")
        self.tab_heatmap = tk.Frame(self.notebook, bg="#1e1e1e")
        self.tab_radius = tk.Frame(self.notebook, bg="#1e1e1e")
        self.tab_deformity = tk.Frame(self.notebook, bg="#1e1e1e")
        self.tab_metrics = tk.Frame(self.notebook, bg="#1e1e1e")

        self.notebook.add(self.tab_plots, text="Embedding Plots")
        self.notebook.add(self.tab_heatmap, text="Inter-Centroid Heatmap")
        self.notebook.add(self.tab_radius, text="Radius Analysis")
        self.notebook.add(self.tab_deformity, text="Deformity Index")
        self.notebook.add(self.tab_metrics, text="Clustering Metrics")

        # Embed plot_frame inside tab_plots
        self.plot_frame = tk.Frame(self.tab_plots, bg="#1e1e1e")
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Initialize core variables
        self.data = None
        self.file_path = None
        self.labels = None
        self.max_samples = 1000
        self.distances_per_class = {}

        if file_path:
            try:
                file_path = str(Path(file_path).resolve())
                logging.debug(f"Initializing with file_path: {file_path}")
                print(f"Initializing with file_path: {file_path}")
                if os.path.exists(file_path):
                    self.file_path = file_path
                    self.load_npz(auto=True)
                else:
                    self.insert_error(f"File not found: {file_path}")
                    logging.error(f"File not found: {file_path}")
                    messagebox.showerror("Error", f"File not found: {file_path}")
            except (FileNotFoundError, ValueError) as e:
                self.insert_error(f"Invalid path: {file_path}: {e}")
                logging.error(f"Invalid path: {file_path}: {e}")
                messagebox.showerror("Error", f"Invalid path: {file_path}\n{e}")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('default')
        
        # Configure Notebook Style
        style.configure("Custom.TNotebook", background="#121212", borderwidth=0)
        style.configure("Custom.TNotebook.Tab", background="#2d2d2d", foreground="#ffffff",
                        font=("Segoe UI", 10, "bold"), padding=[18, 6])
        style.map("Custom.TNotebook.Tab",
                  background=[("selected", "#00adb5"), ("active", "#3d3d3d")],
                  foreground=[("selected", "#ffffff"), ("active", "#ffffff")])
        
        # Configure Treeview Style
        style.configure("Treeview", background="#2d2d2d", foreground="#ffffff",
                        fieldbackground="#2d2d2d", font=("Segoe UI", 10), rowheight=25)
        style.configure("Treeview.Heading", background="#3d3d3d", foreground="#00adb5",
                        font=("Segoe UI", 10, "bold"), relief="flat")
        style.map("Treeview.Heading",
                  background=[("active", "#4d4d4d")],
                  foreground=[("active", "#00adb5")])
        style.map("Treeview",
                  background=[("selected", "#00adb5")],
                  foreground=[("selected", "#ffffff")])

    def style_button(self, btn, bg_color="#00adb5", hover_color="#00f5d4", fg_color="#ffffff"):
        btn.config(
            bg=bg_color, fg=fg_color, activebackground=hover_color, activeforeground=fg_color,
            font=("Segoe UI", 10, "bold"), bd=0, relief="flat", padx=10, pady=8, cursor="hand2"
        )
        btn.bind("<Enter>", lambda e: btn.config(bg=hover_color))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg_color))

    def load_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder Containing .npy Files")
        if not folder_path:
            return  # User cancelled

        if not os.path.isdir(folder_path):
            self.insert_error("Selected path is not a valid folder!")
            return

        self.file_path = folder_path
        self.insert_metadata("Info", f"Loading folder: {folder_path}")

        features_list = []
        labels_list = []
        expected_shape = None

        npy_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.npy')]
        if not npy_files:
            self.insert_error("No .npy files found in the selected folder!")
            return

        self.insert_metadata("Info", f"Found {len(npy_files)} .npy files. Processing...")

        for fname in sorted(npy_files):
            file_path = os.path.join(folder_path, fname)
            try:
                data = np.load(file_path, allow_pickle=True)

                # Case 1: Single feature vector saved as (1, 1, C, H, W) or (C, H, W) from CNN
                if data.ndim >= 3:
                    if data.shape[0] == 1 or data.ndim == 3:
                        flattened = data.reshape(1, -1)  # (1, features)
                        if expected_shape is None:
                            expected_shape = flattened.shape[1:]
                        elif flattened.shape[1:] != expected_shape:
                            self.insert_metadata("Warning", f"Shape mismatch: {fname} → {flattened.shape}, expected (*, {expected_shape})")
                            continue
                        features_list.append(flattened)
                        label = os.path.splitext(fname)[0].split('_')[0]  # e.g., "cat_image.npy" → "cat"
                        labels_list.append(label)
                        self.insert_metadata("Success", f"Loaded {fname} → Class: {label} | Shape: {flattened.shape}")
                    else:
                        self.insert_metadata("Warning", f"Skipping {fname}: unexpected batch size {data.shape[0]}")

                # Case 2: Already a flat vector (n_features,)
                elif data.ndim == 1:
                    processed = self.preprocess_1d_array(data)
                    if processed is None:
                        self.insert_metadata("Warning", f"Skipping {fname}: 1D array couldn't be reshaped")
                        continue
                    features_list.append(processed['features'])
                    if 'labels' in processed:
                        labels_list.extend(processed['labels'])
                    else:
                        label = os.path.splitext(fname)[0].split('_')[0]
                        labels_list.extend([label] * processed['features'].shape[0])
                    self.insert_metadata("Success", f"Processed 1D {fname} → {processed['features'].shape}")

                # Case 3: Already (n_samples, n_features) — rare but possible
                elif data.ndim == 2:
                    features_list.append(data)
                    label = os.path.splitext(fname)[0].split('_')[0]
                    labels_list.extend([label] * data.shape[0])
                    self.insert_metadata("Success", f"Loaded pre-flattened {fname} → {data.shape}")

                else:
                    self.insert_metadata("Warning", f"Skipping {fname}: unsupported shape {data.shape}")

            except Exception as e:
                self.insert_metadata("Error", f"Failed to load {fname}: {str(e)}")
                logging.error(f"Load failed {file_path}: {e}")

        if not features_list:
            self.insert_error("No valid data was loaded from any file!")
            return

        # Stack all features
        try:
            self.data = {'features': np.vstack(features_list)}
            self.labels = np.array(labels_list, dtype=str)

            n_samples, n_features = self.data['features'].shape
            self.insert_metadata("Success", f"SUCCESS! Loaded {n_samples} samples × {n_features} features")
            self.insert_metadata("Info", f"Unique classes: {len(np.unique(self.labels))} → {np.unique(self.labels)}")

            # Final safety checks
            if n_samples < 2 or n_features < 2:
                self.insert_error(f"Not enough data: {n_samples} samples, {n_features} features")
                self.data = None
                self.labels = None
                return

            self.display_metadata()
            self.visualize_all()

        except Exception as e:
            self.insert_error(f"Failed to combine features: {e}")
            logging.exception("Stacking failed")
            
    def load_npz(self, auto=False):
        file_path = self.file_path if auto else filedialog.askopenfilename(filetypes=[("NPY/NPZ files", "*.npy *.npz")])
        if not file_path or not os.path.exists(file_path):
            self.metadata_text.delete(1.0, tk.END)
            self.insert_metadata("Error", "File not found!")
            logging.error("File not found in manual selection")
            return
        self.file_path = file_path
        logging.debug(f"Attempting to load NPZ file: {file_path}")
        print(f"Attempting to load NPZ file: {file_path}")
        try:
            if file_path.endswith('.npz'):
                with zipfile.ZipFile(file_path, 'r') as zf:
                    if zf.testzip() is not None:
                        raise zipfile.BadZipFile("NPZ file is corrupted")
                logging.debug(f"NPZ file integrity verified: {file_path}")
                print(f"NPZ file integrity verified: {file_path}")
                data = np.load(file_path)
                self.data = {}
                self.labels = data.get('labels')
                for key in data:
                    if key == 'labels':
                        continue
                    arr = data[key]
                    if self.is_feature_array(arr):
                        self.data[key] = arr
                    else:
                        self.insert_metadata("Warning", f"Skipping {key}: not a valid feature array (shape {arr.shape}, dtype {arr.dtype})")
                        logging.warning(f"Skipped {key}: invalid feature array")
                if not self.data:
                    self.insert_metadata("Error", "No valid feature arrays found in .npz file!")
                    logging.error("No valid feature arrays in .npz file")
                    return
                if 'features' in self.data:
                    n_samples, n_features = self.data['features'].shape
                    if n_features < 2:
                        self.insert_metadata("Error", f"Too few features ({n_features}) for visualization. Need at least 2.")
                        self.data = None
                        logging.error(f"Too few features: {n_features}")
                        return
                    if n_samples < 2:
                        self.insert_metadata("Error", f"Too few samples ({n_samples}) for visualization. Need at least 2.")
                        self.data = None
                        logging.error(f"Too few samples: {n_samples}")
                        return
                    if self.labels is not None and len(self.labels) != n_samples:
                        self.insert_metadata("Error", f"Labels shape {self.labels.shape} does not match features {self.data['features'].shape}.")
                        self.labels = None
                        logging.error(f"Labels shape mismatch: {self.labels.shape} vs {self.data['features'].shape}")
            elif file_path.endswith('.npy'):
                data = np.load(file_path)
                if data.ndim == 1:
                    self.data = self.preprocess_1d_array(data)
                    if self.data is None:
                        self.insert_metadata("Error", "Cannot process 1D array: unable to reshape or segment.")
                        logging.error("Failed to process 1D .npy array")
                        return
                elif data.ndim == 4 and data.shape[0] == 1:
                    self.data = {'features': data.reshape(1, -1)}
                    label = os.path.splitext(os.path.basename(file_path))[0]
                    self.labels = np.array([label])
                else:
                    self.data = {'data': data}
                    self.labels = None
            else:
                self.insert_metadata("Error", "File must be .npy or .npz")
                logging.error("Invalid file extension")
                return
            self.display_metadata()
            self.visualize_all()
            logging.debug(f"Successfully loaded file: {file_path}")
            print(f"Successfully loaded file: {file_path}")
        except zipfile.BadZipFile as e:
            self.insert_error(f"Corrupted NPZ file: {e}")
            logging.error(f"Corrupted NPZ file: {e}")
            messagebox.showerror("Error", f"Corrupted NPZ file: {file_path}\n{e}")
        except (FileNotFoundError, ValueError) as e:
            self.insert_error(f"Error loading file: {e}")
            logging.error(f"Error loading file: {e}")
            messagebox.showerror("Error", f"Error loading file: {file_path}\n{e}")

    def preprocess_1d_array(self, data):
        n_elements = len(data)
        self.insert_metadata("Info", f"Processing 1D array with {n_elements} elements...")
        possible_feature_sizes = [512, 1024, 2048, 4096]
        for n_features in possible_feature_sizes:
            if n_elements % n_features == 0:
                n_samples = n_elements // n_features
                if n_samples > 1 and n_samples <= self.max_samples * 10:
                    features = data.reshape(n_samples, n_features)
                    self.insert_metadata("Info", f"Reshaped to {n_samples} samples x {n_features} features.")
                    n_clusters = min(10, n_samples)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = kmeans.fit_predict(StandardScaler().fit_transform(features))
                    self.insert_metadata("Info", f"Generated {n_clusters} pseudo-labels using K-means.")
                    return {'features': features, 'labels': labels}
        window_size = 1024
        n_samples = n_elements // window_size
        if n_samples > 1:
            features = data[:n_samples * window_size].reshape(n_samples, window_size)
            self.insert_metadata("Info", f"Segmented into {n_samples} samples x {window_size} features.")
            n_clusters = min(10, n_samples)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(StandardScaler().fit_transform(features))
            self.insert_metadata("Info", f"Generated {n_clusters} pseudo-labels using K-means.")
            return {'features': features, 'labels': labels}
        self.insert_metadata("Error", "Failed to reshape or segment 1D array into features.")
        return None

    def is_feature_array(self, arr):
        return arr.ndim == 2 and arr.shape[1] >= 2 and np.issubdtype(arr.dtype, np.number)

    def insert_metadata(self, tag, message):
        self.metadata_text.insert(tk.END, f"{tag}: {message}\n")
        self.metadata_text.see(tk.END)
        logging.debug(f"{tag}: {message}")

    def insert_error(self, message):
        self.metadata_text.delete(1.0, tk.END)
        self.insert_metadata("Error", message)

    def display_metadata(self):
        self.metadata_text.delete(1.0, tk.END)
        self.insert_metadata("Info", f"File/Folder: {self.file_path}")
        if isinstance(self.data, dict):
            self.insert_metadata("Info", "Contains arrays:")
            for key, arr in self.data.items():
                self.insert_metadata("Info", f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
            if self.labels is not None:
                self.insert_metadata("Info", f"  labels: shape={self.labels.shape}, dtype={self.labels.dtype}, unique={len(np.unique(self.labels))}")
        else:
            self.insert_metadata("Info", f"Array: shape={self.data.shape}, dtype={self.data.dtype}")

    def compute_mode_distance(self, distances):
        if distances.size == 0:
            self.insert_metadata("Warning", "No distances available for mode calculation. Returning 0.0.")
            return 0.0
        hist, bin_edges = np.histogram(distances, bins=20, density=True)
        mode_idx = np.argmax(hist)
        mode_distance = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
        return mode_distance

    def compute_max_radius(self, features_tsne, labels, centroids, unique_labels):
        max_radii = {}
        point_counts = {}
        shifted_centroids = {}
        metric = self.distance_metric.get()
        
        for label, centroid in centroids:
            own_mask = labels == label
            other_mask = labels != label
            own_points = features_tsne[own_mask]
            other_points = features_tsne[other_mask]
            
            if own_points.size == 0:
                self.insert_metadata("Warning", f"Class {label}: No own points. Setting radius to 0.")
                max_radii[label] = 0.0
                point_counts[label] = (0, 0)
                shifted_centroids[label] = centroid
                continue
            
            own_distances = cdist(own_points, [centroid], metric=metric).flatten()
            other_distances = cdist(other_points, [centroid], metric=metric).flatten() if other_points.size > 0 else np.array([])
            
            self.insert_metadata("Debug", f"Class {label}: {own_points.shape[0]} own points, {other_points.shape[0]} other points")
            
            if own_distances.size == 0:
                max_radii[label] = 0.0
                point_counts[label] = (0, 0)
                shifted_centroids[label] = centroid
                continue
            
            own_indices = np.argsort(own_distances)
            sorted_own_distances = own_distances[own_indices]
            
            last_valid_radius = 0.0
            last_own_count = 0
            last_other_count = 0
            last_valid_centroid = centroid
            
            for k in range(1, len(sorted_own_distances) + 1):
                radius = sorted_own_distances[k - 1]
                own_count = k
                other_count = np.sum(other_distances <= radius)
                self.insert_metadata("Debug", f"Class {label}, k={k}, Radius {radius:.4f}: own_count={own_count}, other_count={other_count}")
                
                if own_count > other_count:
                    last_valid_radius = radius
                    last_own_count = own_count
                    last_other_count = other_count
                    points_within_radius = own_points[own_indices[:k]]
                    if points_within_radius.size > 0:
                        last_valid_centroid = np.mean(points_within_radius, axis=0)
                    else:
                        self.insert_metadata("Warning", f"Class {label}: No points within radius {radius:.4f}. Keeping original centroid.")
                        last_valid_centroid = centroid
            
            max_radii[label] = last_valid_radius
            point_counts[label] = (last_own_count, last_other_count)
            shifted_centroids[label] = last_valid_centroid
            
            self.insert_metadata("Info", f"Class {label}: Final radius={last_valid_radius:.4f}, own_count={last_own_count}, other_count={last_other_count}")
            self.insert_metadata("Info", f"Class {label}: Shifted centroid=({last_valid_centroid[0]:.4f}, {last_valid_centroid[1]:.4f}{', ' + f'{last_valid_centroid[2]:.4f}' if len(last_valid_centroid) == 3 else ''})")
        
        return max_radii, point_counts, shifted_centroids    
    
    def display_radius_table(self, max_radii, point_counts, shifted_centroids, table_data, feature_name, dim, method):
        # Clear previous widgets in tab_radius
        for widget in self.tab_radius.winfo_children():
            widget.destroy()

        # Title
        lbl_title = tk.Label(self.tab_radius, text=f"Centroid Radius Analysis: {feature_name} ({method} {dim})",
                             font=("Segoe UI", 12, "bold"), bg="#1e1e1e", fg="#00adb5")
        lbl_title.pack(anchor="w", padx=15, pady=10)

        # Table container
        table_frame = tk.Frame(self.tab_radius, bg="#1e1e1e")
        table_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        columns = ["Class", "Max Radius", "Own Points", "Other Points", "Mean Dist", "Mode Dist", "Max Dist", "Shifted Centroid"]
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", style="Treeview")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor="center")
        for idx, label in enumerate(sorted(max_radii.keys())):
            own_count, other_count = point_counts[label]
            mean_dist, mode_dist, max_dist = table_data[idx][1:4]
            shifted_centroid = shifted_centroids[label]
            centroid_str = f"({shifted_centroid[0]:.4f}, {shifted_centroid[1]:.4f}{', ' + f'{shifted_centroid[2]:.4f}' if len(shifted_centroid) == 3 else ''})"
            tree.insert("", tk.END, values=[f"Class {label}", f"{max_radii[label]:.4f}", own_count, other_count, mean_dist, mode_dist, max_dist, centroid_str])
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        output_dir = os.path.dirname(self.file_path) if os.path.isfile(self.file_path) else self.file_path
        radius_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_radius_{method.lower()}_{dim.lower()}.csv")
        with open(radius_file, 'w') as f:
            f.write("Class,Max Radius,Own Points,Other Points,Mean Distance,Mode Distance,Max Distance,Shifted Centroid\n")
            for idx, label in enumerate(sorted(max_radii.keys())):
                own_count, other_count = point_counts[label]
                mean_dist, mode_dist, max_dist = table_data[idx][1:4]
                shifted_centroid = shifted_centroids[label]
                centroid_str = f"({shifted_centroid[0]:.4f}, {shifted_centroid[1]:.4f}{', ' + f'{shifted_centroid[2]:.4f}' if len(shifted_centroid) == 3 else ''})"
                f.write(f"\"Class {label}\",{max_radii[label]:.4f},{own_count},{other_count},{mean_dist},{mode_dist},{max_dist},\"{centroid_str}\"\n")
        self.insert_metadata("Info", f"Centroid radius table saved to {radius_file}")

    def compute_clustering_metrics(self, features, true_labels):
        n_samples = features.shape[0]
        n_clusters = min(10, n_samples)
        self.insert_metadata("Info", f"Computing clustering metrics for {n_samples} samples...")
        features = StandardScaler().fit_transform(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        pred_labels = kmeans.fit_predict(features)
        if true_labels is None:
            self.insert_metadata("Warning", "No true labels provided. Using K-means pseudo-labels as true labels.")
            true_labels = pred_labels
        true_labels = np.array(true_labels, dtype=str)
        pred_labels = np.array(pred_labels, dtype=int)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        unique_true_labels = np.unique(true_labels)
        unique_pred_labels = np.unique(pred_labels)
        contingency_matrix = np.zeros((len(unique_true_labels), len(unique_pred_labels)))
        for i, true_label in enumerate(unique_true_labels):
            for j, pred_label in enumerate(unique_pred_labels):
                contingency_matrix[i, j] = np.sum((true_labels == true_label) & (pred_labels == pred_label))
        purity = np.sum(np.max(contingency_matrix, axis=0)) / n_samples
        rand_index = adjusted_rand_score(true_labels, pred_labels)
        self.insert_metadata("Info", f"NMI: {nmi:.4f}, Purity Index: {purity:.4f}, Rand Index: {rand_index:.4f}")
        return {
            'Dataset': os.path.basename(self.file_path),
            'NMI': nmi,
            'Purity Index': purity,
            'Rand Index': rand_index
        }

    def display_clustering_metrics(self):
        if self.data is None:
            self.insert_metadata("Error", "No data loaded to compute clustering metrics.")
            return

        for widget in self.tab_metrics.winfo_children():
            widget.destroy()

        lbl_title = tk.Label(self.tab_metrics, text="Clustering Metrics Summary",
                             font=("Segoe UI", 12, "bold"), bg="#1e1e1e", fg="#00adb5")
        lbl_title.pack(anchor="w", padx=15, pady=10)

        table_frame = tk.Frame(self.tab_metrics, bg="#1e1e1e")
        table_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        columns = ["Dataset", "NMI", "Purity Index", "Rand Index"]
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", style="Treeview")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor="center")

        for key, arr in self.data.items():
            if self.is_feature_array(arr):
                metrics = self.compute_clustering_metrics(arr, self.labels)
                tree.insert("", tk.END, values=[
                    metrics['Dataset'],
                    f"{metrics['NMI']:.4f}",
                    f"{metrics['Purity Index']:.4f}",
                    f"{metrics['Rand Index']:.4f}"
                ])

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        output_dir = os.path.dirname(self.file_path) if os.path.isfile(self.file_path) else self.file_path
        metrics_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_clustering_metrics.csv")
        with open(metrics_file, 'w') as f:
            f.write("Dataset,NMI,Purity Index,Rand Index\n")
            for key, arr in self.data.items():
                if self.is_feature_array(arr):
                    metrics = self.compute_clustering_metrics(arr, self.labels)
                    f.write(f"\"{metrics['Dataset']}\",{metrics['NMI']:.4f},{metrics['Purity Index']:.4f},{metrics['Rand Index']:.4f}\n")
        self.insert_metadata("Info", f"Clustering metrics saved to {metrics_file}")

    def clear_plot_frame(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        self.distances_per_class = {}

    def is_label_array(self, arr):
        return arr.ndim == 1 and len(np.unique(arr)) <= 100

    def get_label_array(self, data, feature_key):
        return self.labels, 'labels' if self.labels is not None else (None, None)

    def compute_deformity_index(self, features_tsne, labels, centroids_dict, max_radii, point_counts):
        metric = self.distance_metric.get()
        deformity_scores = {}
        for label in np.unique(labels):
            mask = labels == label
            points = features_tsne[mask]
            if len(points) < 3:
                deformity_scores[label] = 1.0
                self.insert_metadata("DEFORMITY", f"Class {label}: 1.000 (too few points)")
                continue
            centroid = centroids_dict[label]
            distances = cdist(points, [centroid], metric=metric).flatten()
            compactness = np.std(distances) / (np.mean(distances) + 1e-8)
            own_count, other_count = point_counts.get(label, (len(points), 0))
            overlap_penalty = 1.0 - (own_count / (own_count + other_count + 1e-8))
            if len(points) > 1:
                nn_dists = np.sort(cdist(points, points, metric=metric), axis=1)[:, 1]
                sparsity = np.mean(distances) / (np.mean(nn_dists) + 1e-8)
                sparsity_penalty = min(sparsity / 5.0, 1.0)
            else:
                sparsity_penalty = 0.0
            deformity = 0.4 * compactness + 0.4 * overlap_penalty + 0.2 * sparsity_penalty
            deformity = np.clip(deformity, 0.0, 1.0)
            deformity_scores[label] = deformity
            quality = "Excellent" if deformity < 0.25 else "Good" if deformity < 0.45 else "Moderate" if deformity < 0.65 else "Poor" if deformity < 0.85 else "Very Poor"
            self.insert_metadata("DEFORMITY", f"Class {label}: {deformity:.3f} [{quality}] | Compact={compactness:.3f}, Overlap={overlap_penalty:.3f}, Sparse={sparsity_penalty:.3f}")
        avg = np.mean(list(deformity_scores.values()))
        overall = "Excellent" if avg < 0.25 else "Good" if avg < 0.45 else "Moderate" if avg < 0.65 else "Poor"
        self.insert_metadata("DEFORMITY", f"OVERALL DEFORMITY INDEX: {avg:.3f} to {overall} cluster quality")
        return deformity_scores, avg

    def display_deformity_table(self, scores, avg_deformity, feature_name, dim, method):
        # Clear previous widgets in tab_deformity
        for widget in self.tab_deformity.winfo_children():
            widget.destroy()

        # Banner Layout
        header_frame = tk.Frame(self.tab_deformity, bg="#1e1e1e")
        header_frame.pack(fill="x", padx=15, pady=10)

        lbl_title = tk.Label(header_frame, text=f"DEFORMITY INDEX REPORT - {feature_name} ({method} {dim})",
                             font=("Segoe UI", 12, "bold"), bg="#1e1e1e", fg="#ff5252")
        lbl_title.pack(anchor="w")

        quality = "Excellent" if avg_deformity < 0.25 else "Good" if avg_deformity < 0.45 else "Moderate" if avg_deformity < 0.65 else "Poor" if avg_deformity < 0.85 else "Very Poor"
        color = "#00ff80" if avg_deformity < 0.45 else "#ffff00" if avg_deformity < 0.85 else "#ff5252"
        
        lbl_score = tk.Label(header_frame, text=f"Overall Deformity Index: {avg_deformity:.4f}  [{quality}]",
                             font=("Segoe UI", 11, "bold"), bg="#1e1e1e", fg=color)
        lbl_score.pack(anchor="w", pady=(5, 0))

        # Table container
        table_frame = tk.Frame(self.tab_deformity, bg="#1e1e1e")
        table_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        columns = ("Class", "Deformity Index", "Quality")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", style="Treeview")
        tree.heading("Class", text="Class")
        tree.heading("Deformity Index", text="Deformity Index")
        tree.heading("Quality", text="Quality")
        tree.column("Class", width=200, anchor="center")
        tree.column("Deformity Index", width=220, anchor="center")
        tree.column("Quality", width=180, anchor="center")

        for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            q = "Excellent" if score < 0.25 else "Good" if score < 0.45 else "Moderate" if score < 0.65 else "Poor" if score < 0.85 else "Very Poor"
            tree.insert("", "end", values=(f"Class {label}", f"{score:.4f}", q))

        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.insert_metadata("Info", "Deformity Index table updated")

    def save_deformity_report(self, scores, avg, feature_name, dim, method):
        output_dir = os.path.dirname(self.file_path) if os.path.isfile(self.file_path) else self.file_path
        base = os.path.splitext(os.path.basename(self.file_path))[0]
        path = os.path.join(output_dir, f"{base}_deformity_index_{method.lower()}_{dim.lower()}.csv")
        with open(path, 'w') as f:
            f.write("Class,Deformity_Index,Quality\n")
            for lbl, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                q = "Excellent" if s < 0.25 else "Good" if s < 0.45 else "Moderate" if s < 0.65 else "Poor" if s < 0.85 else "Very Poor"
                f.write(f"{lbl},{s:.4f},{q}\n")
            f.write(f"\nAverage,{avg:.4f},Overall\n")
        self.insert_metadata("Info", f"Deformity Index report saved: {os.path.basename(path)}")

    def visualize_features(self, features, labels=None, feature_name="Features", labels_name=None, dim="2D", method="PCA"):
        if not self.is_feature_array(features):
            if features.ndim == 4 and features.shape[0] == 1:
                features = features.reshape(1, -1)
            else:
                self.insert_metadata("Error", f"Cannot visualize {feature_name} with shape {features.shape}.")
                self.fallback_visualization(features, feature_name, labels=labels)
                return
        n_samples, n_features = features.shape
        if n_samples > self.max_samples:
            self.insert_metadata("Info", f"Downsampling {feature_name} from {n_samples} to {self.max_samples} samples.")
            indices = np.random.choice(n_samples, self.max_samples, replace=False)
            features = features[indices]
            labels = labels[indices] if labels is not None else None
            n_samples = self.max_samples
        elif n_samples < 2:
            self.insert_metadata("Error", f"Single sample for {feature_name}; {dim} t-SNE not applicable.")
            self.fallback_visualization(features, feature_name, labels=labels)
            return
        self.insert_metadata("Info", f"Normalizing features for {feature_name}...")
        features = StandardScaler().fit_transform(features)
        if n_features > 50:
            n_components = min(n_samples, n_features, 50)
            self.insert_metadata("Info", f"Reducing {n_features} features to {n_components} with {method}...")
            try:
                reducer = PCA(n_components=n_components, random_state=42) if method == "PCA" else TruncatedSVD(n_components=n_components, random_state=42)
                features = reducer.fit_transform(features)
            except ValueError as e:
                self.insert_error(f"{method} failed for {feature_name}: {e}")
                self.fallback_visualization(features, feature_name, labels=labels)
                return
        n_components = 2 if dim == "2D" else 3
        perplexity = min(5, n_samples - 1) if n_samples < 50 else min(30, n_samples - 1)
        self.insert_metadata("Info", f"Generating {dim} t-SNE plot (perplexity={perplexity})...")
        self.root.update()
        try:
            try:
                tsne = TSNE(n_components=n_components, perplexity=perplexity, max_iter=1000, random_state=42, n_jobs=-1)
            except TypeError:
                tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=1000, random_state=42, n_jobs=-1)
            features_tsne = tsne.fit_transform(features)
            fig = plt.figure(figsize=(20, 8))
            if dim == "2D":
                ax = fig.add_subplot(111)
                ax.set_aspect('equal', adjustable='box')
                x, y = features_tsne[:, 0], features_tsne[:, 1]
            else:
                ax = fig.add_subplot(111, projection='3d')
                x, y, z = features_tsne[:, 0], features_tsne[:, 1], features_tsne[:, 2]
            centroids = []
            shifted_centroids_list = []
            self.distances_per_class = {}
            table_data = []
            if labels is not None:
                labels = np.array(labels, dtype=str)
                unique_labels = np.unique(labels)
                self.insert_metadata("Info", f"Found {len(unique_labels)} unique labels: {unique_labels}")
                if len(unique_labels) > 100:
                    self.insert_metadata("Warning", f"Excessive classes ({len(unique_labels)}). Consider reducing class count.")
                if len(unique_labels) <= 40:
                    cmap_tab20 = plt.colormaps['tab20']
                    cmap_tab20b = plt.colormaps['tab20b']
                    colors = [cmap_tab20(i / 20) for i in range(20)] + [cmap_tab20b(i / 20) for i in range(20)]
                    colors = colors[:len(unique_labels)]
                else:
                    cmap = plt.colormaps['viridis']
                    colors = [cmap(i / len(unique_labels)) for i in range(len(unique_labels))]
                for idx, label in enumerate(unique_labels):
                    mask = labels == label
                    if np.sum(mask) > 0:
                        if dim == "2D":
                            ax.scatter(x[mask], y[mask], c=[colors[idx]], marker=".", label=f"Class {label}", alpha=0.5, s=80)
                            centroid = np.mean(features_tsne[mask], axis=0)
                            ax.scatter([centroid[0]], [centroid[1]], c=[colors[idx]], marker='X', s=200, edgecolors='black', label="Centroid" if idx == 0 else "")
                            class_points = features_tsne[mask]
                            for point in class_points:
                                ax.plot([point[0], centroid[0]], [point[1], centroid[1]], c=colors[idx], alpha=0.3, linewidth=0.5)
                        else:
                            ax.scatter(x[mask], y[mask], z[mask], c=[colors[idx]], marker=".", label=f"Class {label}", alpha=0.5, s=80)
                            centroid = np.mean(features_tsne[mask], axis=0)
                            ax.scatter([centroid[0]], [centroid[1]], [centroid[2]], c=[colors[idx]], marker='X', s=200, edgecolors='black', label="Centroid" if idx == 0 else "")
                            class_points = features_tsne[mask]
                            for point in class_points:
                                ax.plot([point[0], centroid[0]], [point[1], centroid[1]], [point[2], centroid[2]], c=colors[idx], alpha=0.3, linewidth=0.5)
                        centroids.append((label, centroid))
                        distances = cdist(features_tsne[mask], [centroid], metric=self.distance_metric.get()).flatten()
                        self.distances_per_class[label] = distances
                        mode_distance = self.compute_mode_distance(distances)
                        table_data.append([f"{label}", f"{np.mean(distances):.4f}", f"{mode_distance:.4f}", f"{np.max(distances):.4f}"])
                max_radii, point_counts, shifted_centroids = self.compute_max_radius(features_tsne, labels, centroids, unique_labels)
                centroids_dict = dict(centroids)
                deformity_scores, avg_deformity = self.compute_deformity_index(features_tsne, labels, centroids_dict, max_radii, point_counts)
                self.display_deformity_table(deformity_scores, avg_deformity, feature_name, dim, method)
                self.save_deformity_report(deformity_scores, avg_deformity, feature_name, dim, method)
                self.display_radius_table(max_radii, point_counts, shifted_centroids, table_data, feature_name, dim, method)
                
                shifted_legend_added = False
                for label, centroid in centroids:
                    radius = max_radii.get(label, 0.0)
                    shifted_centroid = shifted_centroids.get(label, centroid)
                    if radius > 0:
                        if dim == "2D":
                            circle = Circle(centroid, radius, color=colors[unique_labels.tolist().index(label)], 
                                            fill=False, linestyle='--', alpha=0.7)
                            ax.add_patch(circle)
                            lbl_shifted = "Shifted Centroid" if not shifted_legend_added else ""
                            shifted_legend_added = True
                            ax.scatter([shifted_centroid[0]], [shifted_centroid[1]], c=[colors[unique_labels.tolist().index(label)]], 
                                       marker='*', s=200, edgecolors='black', label=lbl_shifted)
                        else:
                            u = np.linspace(0, 2 * np.pi, 20)
                            v = np.linspace(0, np.pi, 20)
                            x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + centroid[0]
                            y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + centroid[1]
                            z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + centroid[2]
                            ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color=colors[unique_labels.tolist().index(label)], 
                                              alpha=0.3, linestyle='--')
                            lbl_shifted = "Shifted Centroid" if not shifted_legend_added else ""
                            shifted_legend_added = True
                            ax.scatter([shifted_centroid[0]], [shifted_centroid[1]], [shifted_centroid[2]], 
                                       c=[colors[unique_labels.tolist().index(label)]], marker='*', s=200, edgecolors='black', 
                                       label=lbl_shifted)
                    shifted_centroids_list.append((label, shifted_centroid))

                # Build clean non-overlapping legend
                handles, legend_labels = ax.get_legend_handles_labels()
                class_handles = []
                class_labels = []
                general_handles = []
                general_labels = []
                for h, l in zip(handles, legend_labels):
                    if l in ["Centroid", "Shifted Centroid"]:
                        general_handles.append(h)
                        general_labels.append(l)
                    else:
                        class_handles.append(h)
                        class_labels.append(l)

                max_legend_classes = 20
                shown_handles = class_handles[:max_legend_classes] + general_handles
                shown_labels = class_labels[:max_legend_classes] + general_labels

                ax.legend(shown_handles, shown_labels, title=f"Classes: {len(unique_labels)}",
                          bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=8, ncol=2)

                self.insert_metadata("Info", f"Visualizing classes: {unique_labels}")
                max_table_classes = 10
                if len(unique_labels) > max_table_classes:
                    self.insert_metadata("Info", f"Showing stats for {max_table_classes} classes in table. Full stats in distance file.")
                    table_data = table_data[:max_table_classes]
                if table_data:
                    table_columns = ["Class", "Mean Dist", "Mode Dist", "Max Dist"]
                    table = ax.table(cellText=table_data, colLabels=table_columns, bbox=[1.05, 0.0, 0.65, 0.4], loc='right')
                    table.auto_set_font_size(False)
                    table.set_fontsize(7)
                    table.scale(1, 1.5)
                    table.auto_set_column_width(col=list(range(len(table_columns))))
                    self.insert_metadata("Info", "In-plot table with distances displayed.")
            else:
                if dim == "2D":
                    ax.scatter(x, y, c='blue', alpha=0.5, s=80)
                else:
                    ax.scatter(x, y, z, c='blue', alpha=0.5, s=80)
            ax.set_title(f"{dim} t-SNE: Semantic Relationships ({feature_name})")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            if dim == "3D":
                ax.set_zlabel("t-SNE 3")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.subplots_adjust(right=0.57 if dim == "2D" else 0.45, top=1, bottom=0.1)
            try:
                # Wrap each plot and its toolbar in a separate container frame
                plot_container = tk.Frame(self.plot_frame, bg="#1e1e1e")
                plot_container.pack(side="left", fill="both", expand=True, padx=10, pady=10)

                canvas = FigureCanvasTkAgg(fig, master=plot_container)
                canvas.draw()
                canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
                
                toolbar = NavigationToolbar2Tk(canvas, plot_container)
                toolbar.update()
                output_dir = os.path.dirname(self.file_path) if os.path.isfile(self.file_path) else self.file_path
                output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_tsne_{dim.lower()}_{method.lower()}.png")
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                self.insert_metadata("Info", f"{dim} t-SNE plot saved to {output_file}")
                centroid_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_centroids_{dim.lower()}.txt")
                with open(centroid_file, 'w') as f:
                    f.write(f"Class Centroids in {dim} t-SNE Space ({'x, y' if dim == '2D' else 'x, y, z'}):\n")
                    for label, centroid in centroids:
                        f.write(f"Class {label} Original: ({centroid[0]:.4f}, {centroid[1]:.4f}{', ' + f'{centroid[2]:.4f}' if dim == '3D' else ''})\n")
                    f.write(f"\nShifted Centroids in {dim} t-SNE Space ({'x, y' if dim == '2D' else 'x, y, z'}):\n")
                    for label, shifted_centroid in shifted_centroids_list:
                        f.write(f"Class {label} Shifted: ({shifted_centroid[0]:.4f}, {shifted_centroid[1]:.4f}{', ' + f'{shifted_centroid[2]:.4f}' if dim == '3D' else ''})\n")
                self.insert_metadata("Info", f"Centroid coordinates saved to {centroid_file}")
                for label, centroid in centroids:
                    self.insert_metadata("Info", f"Original Centroid for Class {label}: ({centroid[0]:.4f}, {centroid[1]:.4f}{', ' + f'{centroid[2]:.4f}' if dim == '3D' else ''})")
                for label, shifted_centroid in shifted_centroids_list:
                    self.insert_metadata("Info", f"Shifted Centroid for Class {label}: ({shifted_centroid[0]:.4f}, {shifted_centroid[1]:.4f}{', ' + f'{shifted_centroid[2]:.4f}' if dim == '3D' else ''})")
                distance_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_distances_{dim.lower()}.csv")
                with open(distance_file, 'w') as f:
                    f.write("Class,Mean Distance,Mode Distance,Max Distance,Individual Distances\n")
                    for label, distances in sorted(self.distances_per_class.items(), key=lambda x: x[0]):
                        mode_distance = self.compute_mode_distance(distances)
                        f.write(f"\"Class {label}\",{np.mean(distances):.4f},{mode_distance:.4f},{np.max(distances):.4f},\"{';'.join([f'{d:.4f}' for d in distances])}\"\n")
                self.insert_metadata("Info", f"Distances saved to {distance_file}")
                if len(centroids) > 1:
                    centroid_array = np.array([c[1] for c in centroids])
                    centroid_labels = [c[0] for c in centroids]
                    dist_matrix = squareform(pdist(centroid_array, metric=self.distance_metric.get()))
                    # Clear tab_heatmap
                    for widget in self.tab_heatmap.winfo_children():
                        widget.destroy()

                    lbl_heatmap_title = tk.Label(self.tab_heatmap, text=f"Inter-Centroid Distances: {feature_name} ({method} {dim})",
                                                 font=("Segoe UI", 12, "bold"), bg="#1e1e1e", fg="#00adb5")
                    lbl_heatmap_title.pack(anchor="w", padx=15, pady=10)

                    # Sub-frame for canvas and scrollbars
                    heatmap_container = tk.Frame(self.tab_heatmap, bg="#1e1e1e")
                    heatmap_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))

                    canvas_scroll = tk.Canvas(heatmap_container, bg="#1e1e1e", bd=0, highlightthickness=0)
                    canvas_scroll.pack(side=tk.LEFT, fill="both", expand=True)
                    
                    scrollbar_y = tk.Scrollbar(heatmap_container, orient=tk.VERTICAL, command=canvas_scroll.yview)
                    scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
                    canvas_scroll.config(yscrollcommand=scrollbar_y.set)
                    
                    scrollbar_x = tk.Scrollbar(self.tab_heatmap, orient=tk.HORIZONTAL, command=canvas_scroll.xview)
                    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X, padx=15)
                    canvas_scroll.config(xscrollcommand=scrollbar_x.set)

                    inner_frame = tk.Frame(canvas_scroll, bg="#1e1e1e")
                    canvas_scroll.create_window((0, 0), window=inner_frame, anchor="nw")

                    heatmap_fig, heatmap_ax = plt.subplots(figsize=(max(6, len(centroid_labels)), max(6, len(centroid_labels))))
                    heatmap_fig.patch.set_facecolor('#1e1e1e')
                    heatmap_ax.set_facecolor('#1e1e1e')
                    heatmap_ax.tick_params(colors='#ffffff')
                    heatmap_ax.xaxis.label.set_color('#ffffff')
                    heatmap_ax.yaxis.label.set_color('#ffffff')
                    heatmap_ax.title.set_color('#00adb5')

                    sns.heatmap(dist_matrix, annot=True, fmt='.2f', xticklabels=centroid_labels, yticklabels=centroid_labels,
                                cmap=plt.colormaps['viridis'], cbar_kws={'label': f'{self.distance_metric.get().capitalize()} Distance'}, ax=heatmap_ax)
                    heatmap_ax.set_title(f"Inter-Centroid Distances: {feature_name} ({method} {dim})")
                    plt.tight_layout()
                    
                    plot_canvas = FigureCanvasTkAgg(heatmap_fig, master=inner_frame)
                    plot_canvas.draw()
                    plot_canvas.get_tk_widget().pack(fill="both", expand=True)
                    
                    inner_frame.update_idletasks()
                    canvas_scroll.config(scrollregion=canvas_scroll.bbox("all"))
                    heatmap_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_inter_centroid_distances_{method.lower()}_{dim.lower()}.png")
                    heatmap_fig.savefig(heatmap_file, dpi=300, bbox_inches='tight')
                    self.insert_metadata("Info", f"Inter-centroid distance heatmap saved to {heatmap_file}")
                    plt.close(heatmap_fig)
            except (ValueError, MemoryError) as e:
                self.insert_error(f"Failed to render {dim} t-SNE plot for {feature_name}: {e}")
            self.insert_metadata("Success", f"{dim} t-SNE completed for {feature_name}. Closer points indicate similarity.")
            plt.close(fig)
        except (ValueError, MemoryError) as e:
            self.insert_error(f"{dim} t-SNE failed for {feature_name}: {e}")
            plt.close('all')
            self.fallback_visualization(features, feature_name, labels=labels)

    def fallback_visualization(self, arr, arr_name, labels=None):
        try:
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            if arr.ndim == 1:
                ax.plot(arr[:1000], label=arr_name, alpha=0.6)
                ax.set_title(f"First 1000 Elements: {arr_name}")
                ax.set_xlabel("Index")
                ax.set_ylabel("Value")
            elif arr.ndim == 2 and arr.shape[1] >= 2:
                x = arr[:1000, 0]
                y = arr[:1000, 1]
                if labels is not None:
                    labels = np.array(labels[:1000], dtype=str)
                    unique_labels = np.unique(labels)
                    self.insert_metadata("Info", f"Found {len(unique_labels)} unique labels in fallback: {unique_labels}")
                    if len(unique_labels) <= 40:
                        cmap_tab20 = plt.colormaps['tab20']
                        cmap_tab20b = plt.colormaps['tab20b']
                        colors = [cmap_tab20(i / 20) for i in range(20)] + [cmap_tab20b(i / 20) for i in range(20)]
                        colors = colors[:len(unique_labels)]
                    else:
                        cmap = plt.colormaps['viridis']
                        colors = [cmap(i / len(unique_labels)) for i in range(len(unique_labels))]
                    for idx, label in enumerate(unique_labels):
                        mask = labels == label
                        if np.sum(mask) > 0:
                            ax.scatter(x[mask], y[mask], c=[colors[idx]], marker=".", label=f"Class {label}", alpha=0.5, s=80)
                    if len(unique_labels) <= 20:
                        ax.legend(title=f"Classes: {len(unique_labels)}", bbox_to_anchor=(1.1, 1), loc='upper left',
                                 borderaxespad=0., fontsize=8, ncol=2)
                    self.insert_metadata("Info", f"Visualizing classes in fallback: {unique_labels}")
                else:
                    ax.scatter(x, y, c='blue', alpha=0.5, s=80)
                ax.set_title(f"First Two Features: {arr_name}")
                ax.set_xlabel("Feature 0")
                ax.set_ylabel("Feature 1")
            elif arr.ndim == 4 and arr.shape[0] == 1:
                ax.imshow(arr[0, 0, :, :], cmap=plt.colormaps['viridis'])
                ax.set_title(f"Feature Map (Channel 0): {arr_name}")
                ax.set_xlabel("Width")
                ax.set_ylabel("Height")
            else:
                self.insert_metadata("Error", f"No visualization possible for {arr_name} with shape {arr.shape}")
                plt.close(fig)
                return
            ax.grid(True, alpha=0.3)
            try:
                # Wrap fallback plot and toolbar in container frame
                plot_container = tk.Frame(self.plot_frame, bg="#1e1e1e")
                plot_container.pack(side="left", fill="both", expand=True, padx=10, pady=10)

                canvas = FigureCanvasTkAgg(fig, master=plot_container)
                canvas.draw()
                canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
                
                toolbar = NavigationToolbar2Tk(canvas, plot_container)
                toolbar.update()
                output_dir = os.path.dirname(self.file_path) if os.path.isfile(self.file_path) else self.file_path
                output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_fallback_{arr_name}.png")
                fig.savefig(output_file, dpi=100, bbox_inches='tight')
                self.insert_metadata("Info", f"Saved fallback plot to {output_file}")
                
            except (ValueError, MemoryError) as e:
                self.insert_error(f"Failed to render fallback plot for {arr_name}: {e}")
            self.insert_metadata("Info", f"Fallback visualization completed for {arr_name}.")
            plt.close(fig)
        except (ValueError, MemoryError) as e:
            self.insert_error(f"Failed to generate fallback visualization for {arr_name}: {e}")
            plt.close('all')

    def visualize_all(self):
        if self.data is None:
            return
        self.clear_plot_frame()
        for key, arr in self.data.items():
            if self.is_feature_array(arr):
                self.visualize_features(arr, labels=self.labels, feature_name=key, labels_name='labels', 
                                       dim=self.dim.get(), method=self.opt.get())
            else:
                self.insert_metadata("Info", f"Skipping {key}: not a valid feature array (shape {arr.shape}, dtype {arr.dtype})")
            self.root.update()
        self.display_clustering_metrics()

def main(file_path=None):
    root = tk.Tk()
    app = NpyVisualizerApp(root, "2D", "PCA", "euclidean")
    root.mainloop()

if __name__ == "__main__":
    main()