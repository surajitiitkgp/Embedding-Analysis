# ```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import Frame, ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
import warnings
import logging
from pathlib import Path
import zipfile
from scipy.spatial.distance import pdist, squareform

# Setup logging
logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

warnings.simplefilter("ignore", category=UserWarning)

class NpyVisualizerApp:
    def __init__(self, root, dim_options, algo_options, dis_opt, file_path=None):
        self.root = root
        self.root.title("CNN Embedding Analysis for Semantic Relationships")
        self.root.geometry("1200x800")
        self.frame = Frame(self.root)
        self.frame.pack(padx=10, pady=10, fill="both", expand=True)
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(0, weight=0)
        self.frame.rowconfigure(1, weight=0)
        self.frame.rowconfigure(2, weight=0)
        self.frame.rowconfigure(3, weight=0)
        self.frame.rowconfigure(4, weight=0)
        self.frame.rowconfigure(5, weight=1)
        
        self.dim = tk.StringVar(value=dim_options)
        self.dropdown_label1 = tk.Label(self.frame, text="Choose Dimension:")
        self.dropdown_label1.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.dropdown_dim = tk.OptionMenu(self.frame, self.dim, "2D", "3D")
        self.dropdown_dim.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.opt = tk.StringVar(value=algo_options)
        self.dropdown_label = tk.Label(self.frame, text="Choose Reduction Method:")
        self.dropdown_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.dropdown_opt = tk.OptionMenu(self.frame, self.opt, "PCA", "TruncatedSVD")
        self.dropdown_opt.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.distance_metric = tk.StringVar(value=dis_opt)
        self.dropdown_label_distance = tk.Label(self.frame, text="Choose Distance Metric:")
        self.dropdown_label_distance.grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.dropdown_distance = tk.OptionMenu(self.frame, self.distance_metric, "euclidean", "cosine", "cityblock", "canberra")
        self.dropdown_distance.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.btn_load_folder = tk.Button(self.frame, text="Load Folder of NPY Files", command=self.load_folder)
        self.btn_load_folder.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        self.btn_load_file = tk.Button(self.frame, text="Load NPZ File", command=self.load_npz)
        self.btn_load_file.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        self.metadata_text = tk.Text(self.frame, height=12, width=50)
        self.metadata_text.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.scrollbar = tk.Scrollbar(self.frame, command=self.metadata_text.yview)
        self.scrollbar.grid(row=4, column=2, sticky="ns")
        self.metadata_text.config(yscrollcommand=self.scrollbar.set)
        self.plot_frame = tk.Frame(self.frame)
        self.plot_frame.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
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
            except Exception as e:
                self.insert_error(f"Invalid path: {file_path}: {e}")
                logging.error(f"Invalid path: {file_path}: {e}")
                messagebox.showerror("Error", f"Invalid path: {file_path}\n{e}")   

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path or not os.path.exists(folder_path):
            self.insert_metadata("Error", "Folder not found!")
            logging.error("Folder not found")
            return
        self.file_path = folder_path
        try:
            features = []
            labels = []
            expected_shape = None
            for fname in os.listdir(folder_path):
                if fname.endswith('.npy'):
                    file_path = os.path.join(folder_path, fname)
                    data = np.load(file_path)
                    if data.ndim == 4 and data.shape[0] == 1:
                        flattened = data.flatten()
                        if expected_shape is None:
                            expected_shape = flattened.shape
                        elif flattened.shape != expected_shape:
                            self.insert_error(
                                f"Dimension mismatch in {fname}: got shape {flattened.shape}, expected {expected_shape}"
                            )
                            logging.error(f"Dimension mismatch in {fname}")
                            return
                        features.append(flattened)
                        label = os.path.splitext(fname)[0].split('_')[0]
                        labels.append(label)
                    elif data.ndim == 1:
                        processed_data = self.preprocess_1d_array(data)
                        if processed_data is None:
                            self.insert_metadata("Warning", f"Skipping {fname}: cannot process 1D array")
                            logging.warning(f"Skipped {fname}: cannot process 1D array")
                            continue
                        features.append(processed_data['features'])
                        labels.append(processed_data['labels'])
                    else:
                        self.insert_metadata("Warning", f"Skipping {fname}: expected 4D array with batch=1 or 1D array, got shape {data.shape}")
                        logging.warning(f"Skipped {fname}: invalid shape")
            if features:
                try:
                    self.data = {'features': np.vstack(features)}
                    self.labels = np.array(labels)
                    n_samples, n_features = self.data['features'].shape
                    if n_features < 2:
                        self.insert_metadata("Error", f"Too few features ({n_features}) for visualization.")
                        self.data = None
                        logging.error(f"Too few features: {n_features}")
                        return
                    if n_samples < 2:
                        self.insert_metadata("Error", f"Too few samples ({n_samples}) for visualization.")
                        self.data = None
                        logging.error(f"Too few samples: {n_samples}")
                        return
                    if self.labels is not None and len(self.labels) != n_samples:
                        self.insert_metadata("Error", f"Labels shape {self.labels.shape} does not match features {self.data['features'].shape}.")
                        self.labels = None
                        logging.error(f"Labels shape mismatch: {self.labels.shape} vs {self.data['features'].shape}")
                    self.display_metadata()
                    self.visualize_all()
                    logging.debug(f"Successfully loaded folder: {folder_path}")
                    print(f"Successfully loaded folder: {folder_path}")
                except ValueError as e:
                    self.insert_error(f"Failed to stack features: {e}")
                    logging.error(f"Failed to stack features: {e}")
            else:
                self.insert_metadata("Error", "No valid .npy files found in folder!")
                logging.error("No valid .npy files in folder")
        except Exception as e:
            self.insert_error(f"Error loading folder: {e}")
            logging.error(f"Error loading folder: {e}")

    def load_npz(self, auto=False):
        file_path = self.file_path
        if not auto:
            file_path = filedialog.askopenfilename(filetypes=[("NPY/NPZ files", "*.npy *.npz")])
            if not file_path or not os.path.exists(file_path):
                self.metadata_text.delete(1.0, tk.END)
                self.insert_metadata("Error", "File not found!")
                logging.error("File not found in manual selection")
                return
            self.file_path = file_path
        logging.debug(f"Attempting to load NPZ file: {file_path}")
        print(f"Attempting to load NPZ file: {file_path}")
        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                if zf.testzip() is not None:
                    raise zipfile.BadZipFile("NPZ file is corrupted")
            logging.debug(f"NPZ file integrity verified: {file_path}")
            print(f"NPZ file integrity verified: {file_path}")
            if file_path.endswith('.npy'):
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
            elif file_path.endswith('.npz'):
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
        except FileNotFoundError as e:
            self.insert_error(f"File not found: {e}")
            logging.error(f"File not found: {e}")
            messagebox.showerror("Error", f"File not found: {file_path}\n{e}")
        except Exception as e:
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
            return 0.0
        hist, bin_edges = np.histogram(distances, bins=20, density=True)
        mode_idx = np.argmax(hist)
        mode_distance = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
        return mode_distance

    def compute_max_radius(self, features_tsne, labels, centroids, unique_labels):
        max_radii = {}
        point_counts = {}
        for idx, (label, centroid) in enumerate(centroids):
            own_mask = labels == label
            other_mask = labels != label
            own_points = features_tsne[own_mask]
            other_points = features_tsne[other_mask]
            own_distances = np.sqrt(np.sum((own_points - centroid) ** 2, axis=1)) if own_points.size > 0 else np.array([])
            other_distances = np.sqrt(np.sum((other_points - centroid) ** 2, axis=1)) if other_points.size > 0 else np.array([])
            
            # Log the number of points for debugging
            self.insert_metadata("Debug", f"Class {label}: {own_points.shape[0]} own points, {other_points.shape[0]} other points")
            
            # Calculate maximum possible distance
            max_distance = max(np.max(own_distances) if own_distances.size > 0 else 0, 
                            np.max(other_distances) if other_distances.size > 0 else 0)
            if max_distance == 0:
                self.insert_metadata("Warning", f"Class {label}: No valid distances (empty class or single point). Setting radius to 0.")
                max_radii[label] = 0.0
                point_counts[label] = (0, 0)
                continue
            
            # Use a finer step size to increase precision
            step = max_distance / 500  # Increased from 100 to 500 for finer granularity
            radius = 0.0
            last_valid_radius = 0.0
            last_own_count = 0
            last_other_count = 0
            
            # Increment radius until other_count >= own_count or max_distance is reached
            while radius <= max_distance:
                own_count = np.sum(own_distances <= radius) if own_distances.size > 0 else 0
                other_count = np.sum(other_distances <= radius) if other_distances.size > 0 else 0
                
                # Log counts for debugging
                self.insert_metadata("Debug", f"Class {label}, Radius {radius:.4f}: own_count={own_count}, other_count={other_count}")
                
                if own_count > 0 and own_count > other_count:
                    last_valid_radius = radius
                    last_own_count = own_count
                    last_other_count = other_count
                elif own_count == 0 and own_points.size > 0:
                    # If no own points are within radius but class has points, continue to increase radius
                    pass
                else:
                    # Stop if other_count >= own_count or own_count is 0 and no further points are expected
                    break
                
                radius += step
            
            # Fallback: If no valid radius found, use the maximum own_distance or a small default
            if last_valid_radius == 0.0 and own_distances.size > 0:
                self.insert_metadata("Warning", f"Class {label}: No radius found where own_count > other_count. Using max own_distance.")
                last_valid_radius = np.max(own_distances) if own_distances.size > 0 else step
                last_own_count = np.sum(own_distances <= last_valid_radius) if own_distances.size > 0 else 0
                last_other_count = np.sum(other_distances <= last_valid_radius) if other_distances.size > 0 else 0
            
            max_radii[label] = last_valid_radius
            point_counts[label] = (last_own_count, last_other_count)
            
            # Log the final radius and counts
            self.insert_metadata("Info", f"Class {label}: Final radius={last_valid_radius:.4f}, own_count={last_own_count}, other_count={last_other_count}")
        
        return max_radii, point_counts
    

    
    def display_radius_table(self, max_radii, point_counts, feature_name, dim, method):
        radius_window = tk.Toplevel(self.root)
        radius_window.title(f"Centroid Radius Analysis: {feature_name} ({method} {dim})")
        radius_window.geometry("600x400")
        radius_frame = tk.Frame(radius_window)
        radius_frame.pack(fill="both", expand=True)
        columns = ["Class", "Max Radius", "Own Points", "Other Points"]
        tree = ttk.Treeview(radius_frame, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        for label in sorted(max_radii.keys()):
            own_count, other_count = point_counts[label]
            tree.insert("", tk.END, values=[f"Class {label}", f"{max_radii[label]:.4f}", own_count, other_count])
        tree.pack(fill="both", expand=True)
        output_dir = os.path.dirname(self.file_path) if os.path.isfile(self.file_path) else self.file_path
        radius_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_radius_{method.lower()}_{dim.lower()}.csv")
        with open(radius_file, 'w') as f:
            f.write("Class,Max Radius,Own Points,Other Points\n")
            for label in sorted(max_radii.keys()):
                own_count, other_count = point_counts[label]
                f.write(f"\"Class {label}\",{max_radii[label]:.4f},{own_count},{other_count}\n")
        self.insert_metadata("Info", f"Centroid radius table saved to {radius_file}")

    def clear_plot_frame(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        self.distances_per_class = {}

    def is_label_array(self, arr):
        return arr.ndim == 1 and len(np.unique(arr)) <= 100

    def get_label_array(self, data, feature_key):
        return self.labels, 'labels' if self.labels is not None else (None, None)

    def visualize_featuresPCA(self, features, labels=None, feature_name="Features", labels_name=None):
        if not self.is_feature_array(features):
            if features.ndim == 4 and features.shape[0] == 1:
                features = features.reshape(1, -1)
            else:
                self.insert_metadata("Error", f"Cannot visualize {feature_name} with shape {features.shape}.")
                self.fallback_visualization(features, feature_name, labels=labels)
                return
        n_samples, n_features = features.shape
        indices = None
        if n_samples > self.max_samples:
            self.insert_metadata("Info", f"Downsampling {feature_name} from {n_samples} to {self.max_samples} samples.")
            indices = np.random.choice(n_samples, self.max_samples, replace=False)
            features = features[indices]
            labels = labels[indices] if labels is not None else None
            n_samples = self.max_samples
        elif n_samples < 2:
            self.insert_metadata("Error", f"Single sample for {feature_name}; 2D t-SNE not applicable.")
            self.fallback_visualization(features, feature_name, labels=labels)
            return
        self.insert_metadata("Info", f"Normalizing features for {feature_name}...")
        features = StandardScaler().fit_transform(features)
        if n_features > 50:
            n_components = min(n_samples, n_features, 50)
            self.insert_metadata("Info", f"Reducing {n_features} features to {n_components} with PCA...")
            try:
                pca = PCA(n_components=n_components, random_state=42)
                features = pca.fit_transform(features)
            except ValueError as e:
                self.insert_error(f"PCA failed for {feature_name}: {e}")
                self.fallback_visualization(features, feature_name, labels=labels)
                return
        perplexity = min(5, n_samples - 1) if n_samples < 50 else min(30, n_samples - 1)
        self.insert_metadata("Info", f"Generating 2D t-SNE plot (perplexity={perplexity})...")
        self.root.update()
        try:
            tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42, n_jobs=-1)
            features_tsne = tsne.fit_transform(features)
            fig, ax = plt.subplots(figsize=(20, 8))
            x = features_tsne[:, 0]
            y = features_tsne[:, 1]
            centroids = []
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
                        ax.scatter(x[mask], y[mask], c=[colors[idx]], marker=".", label=f"Class {label}", alpha=0.5, s=80)
                        centroid = np.mean(features_tsne[mask], axis=0)
                        centroids.append((label, centroid))
                        ax.scatter([centroid[0]], [centroid[1]], c=[colors[idx]], marker='X', s=200, edgecolors='black')
                        class_points = features_tsne[mask]
                        for point in class_points:
                            ax.plot([point[0], centroid[0]], [point[1], centroid[1]], c=colors[idx], alpha=0.3, linewidth=0.5)
                        distances = np.sqrt(np.sum((class_points - centroid) ** 2, axis=1))
                        self.distances_per_class[label] = distances
                        mode_distance = self.compute_mode_distance(distances)
                        table_data.append([f"Class {label}", f"{np.mean(distances):.4f}", f"{mode_distance:.4f}", f"{np.max(distances):.4f}"])
                max_radii, point_counts = self.compute_max_radius(features_tsne, labels, centroids, unique_labels)
                self.display_radius_table(max_radii, point_counts, feature_name, "2D", "PCA")
                max_legend_classes = 20
                if len(unique_labels) > max_legend_classes:
                    self.insert_metadata("Info", f"Showing {max_legend_classes} classes in legend. Full class list in centroid/distance files.")
                    handles, legend_labels = ax.get_legend_handles_labels()
                    ax.legend(handles[:max_legend_classes*2], legend_labels[:max_legend_classes*2], title=f"Classes: {len(unique_labels)}",
                            bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0., fontsize=8, ncol=4)
                else:
                    ax.legend(title=f"Classes: {len(unique_labels)}", bbox_to_anchor=(1.1, 1), loc='upper left',
                            borderaxespad=0., fontsize=8, ncol=2)
                self.insert_metadata("Info", f"Visualizing classes: {unique_labels}")
                max_table_classes = 10
                if len(unique_labels) > max_table_classes:
                    self.insert_metadata("Info", f"Showing stats for {max_table_classes} classes in table. Full stats in distance file.")
                    table_data = table_data[:max_table_classes]
                if table_data:
                    table_columns = ["Class", "Mean Distance", "Mode Distance", "Max Distance"]
                    table = ax.table(cellText=table_data, colLabels=table_columns, bbox=[1.1, 0.0, 0.7, 0.5], loc='right')
                    table.auto_set_font_size(False)
                    table.set_fontsize(7)
                    table.scale(1, 1.5)
            else:
                ax.scatter(x, y, c='blue', alpha=0.5, s=80)
            ax.set_title(f"2D t-SNE: Semantic Relationships ({feature_name})")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.subplots_adjust(right=0.57, top=1, bottom=0.1)
            try:
                canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(side="left", padx=10, pady=10)
                toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
                toolbar.update()
                output_dir = os.path.dirname(self.file_path) if os.path.isfile(self.file_path) else self.file_path
                output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_tsne_2d.png")
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                self.insert_metadata("Info", f"2D t-SNE plot saved to {output_file}")
                centroid_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_centroids_2d.txt")
                with open(centroid_file, 'w') as f:
                    f.write("Class Centroids in 2D t-SNE Space (x, y):\n")
                    for label, centroid in centroids:
                        f.write(f"Class {label}: ({centroid[0]:.4f}, {centroid[1]:.4f})\n")
                self.insert_metadata("Info", f"Centroid coordinates saved to {centroid_file}")
                for label, centroid in centroids:
                    self.insert_metadata("Info", f"Centroid for Class {label}: ({centroid[0]:.4f}, {centroid[1]:.4f})")
                distance_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_distances_2d.csv")
                with open(distance_file, 'w') as f:
                    f.write("Class,Mean Distance,Mode Distance,Max Distance,Individual Distances\n")
                    for label, distances in sorted(self.distances_per_class.items(), key=lambda x: x[0]):
                        mode_distance = self.compute_mode_distance(distances)
                        f.write(f"\"Class {label}\",{np.mean(distances):.4f},{mode_distance:.4f},{np.max(distances):.4f},\"{';'.join([f'{d:.4f}' for d in distances])}\"\n")
                self.insert_metadata("Info", f"Distances saved to {distance_file}")
            except Exception as e:
                self.insert_error(f"Failed to render 2D t-SNE plot for {feature_name}: {e}")
            self.insert_metadata("Success", f"2D t-SNE completed for {feature_name}. Closer points indicate similarity.")
            plt.close(fig)
            # Move heatmap to a separate window if centroids exist
            if len(centroids) > 1:
                centroid_array = np.array([c[1] for c in centroids])
                centroid_labels = [c[0] for c in centroids]
                dist_matrix = squareform(pdist(centroid_array, metric=self.distance_metric.get()))
                heatmap_window = tk.Toplevel(self.root)
                heatmap_window.title(f"Inter-Centroid Distances: {feature_name} (PCA 2D)")
                heatmap_window.geometry("800x600")
                heatmap_frame = tk.Frame(heatmap_window)
                heatmap_frame.pack(fill="both", expand=True)
                heatmap_fig, heatmap_ax = plt.subplots(figsize=(max(6, len(centroid_labels)), max(6, len(centroid_labels))))
                sns.heatmap(dist_matrix, annot=True, fmt='.2f', xticklabels=centroid_labels, yticklabels=centroid_labels,
                            cmap=plt.colormaps['viridis'], cbar_kws={'label': f'{self.distance_metric.get().capitalize()} Distance'}, ax=heatmap_ax)
                heatmap_ax.set_title(f"Inter-Centroid Distances: {feature_name} (PCA 2D)")
                plt.tight_layout()
                heatmap_canvas = FigureCanvasTkAgg(heatmap_fig, master=heatmap_frame)
                heatmap_canvas.draw()
                heatmap_canvas.get_tk_widget().pack(fill="both", expand=True)
                output_dir = os.path.dirname(self.file_path) if os.path.isfile(self.file_path) else self.file_path
                heatmap_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_inter_centroid_distances_pca_2d.png")
                heatmap_fig.savefig(heatmap_file, dpi=300, bbox_inches='tight')
                self.insert_metadata("Info", f"Inter-centroid distance heatmap saved to {heatmap_file}")
                plt.close(heatmap_fig)
        except Exception as e:
            self.insert_error(f"2D t-SNE failed for {feature_name}: {e}")
            plt.close('all')
            self.fallback_visualization(features, feature_name, labels=labels)

    def visualize_featuresTruncatedSVD(self, features, labels=None, feature_name="Features", labels_name=None):
        if not self.is_feature_array(features):
            if features.ndim == 4 and features.shape[0] == 1:
                features = features.reshape(1, -1)
            else:
                self.insert_metadata("Error", f"Cannot visualize {feature_name} with shape {features.shape}.")
                self.fallback_visualization(features, feature_name, labels=labels)
                return
        n_samples, n_features = features.shape
        indices = None
        if n_samples > self.max_samples:
            self.insert_metadata("Info", f"Downsampling {feature_name} from {n_samples} to {self.max_samples} samples.")
            indices = np.random.choice(n_samples, self.max_samples, replace=False)
            features = features[indices]
            labels = labels[indices] if labels is not None else None
            n_samples = self.max_samples
        elif n_samples < 2:
            self.insert_metadata("Error", f"Single sample for {feature_name}; 2D t-SNE not applicable.")
            self.fallback_visualization(features, feature_name, labels=labels)
            return
        self.insert_metadata("Info", f"Normalizing features for {feature_name}...")
        features = StandardScaler().fit_transform(features)
        if n_features > 50:
            n_components = min(n_samples, n_features, 50)
            self.insert_metadata("Info", f"Reducing {n_features} features to {n_components} with TruncatedSVD...")
            try:
                pca = TruncatedSVD(n_components=n_components, random_state=42)
                features = pca.fit_transform(features)
            except ValueError as e:
                self.insert_error(f"TruncatedSVD failed for {feature_name}: {e}")
                self.fallback_visualization(features, feature_name, labels=labels)
                return
        perplexity = min(5, n_samples - 1) if n_samples < 50 else min(30, n_samples - 1)
        self.insert_metadata("Info", f"Generating 2D t-SNE plot (perplexity={perplexity})...")
        self.root.update()
        try:
            tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42, n_jobs=-1)
            features_tsne = tsne.fit_transform(features)
            fig, ax = plt.subplots(figsize=(20, 8))
            x = features_tsne[:, 0]
            y = features_tsne[:, 1]
            centroids = []
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
                        ax.scatter(x[mask], y[mask], c=[colors[idx]], marker=".", label=f"Class {label}", alpha=0.5, s=80)
                        centroid = np.mean(features_tsne[mask], axis=0)
                        centroids.append((label, centroid))
                        ax.scatter([centroid[0]], [centroid[1]], c=[colors[idx]], marker='X', s=200, edgecolors='black')
                        class_points = features_tsne[mask]
                        for point in class_points:
                            ax.plot([point[0], centroid[0]], [point[1], centroid[1]], c=colors[idx], alpha=0.3, linewidth=0.5)
                        distances = np.sqrt(np.sum((class_points - centroid) ** 2, axis=1))
                        self.distances_per_class[label] = distances
                        mode_distance = self.compute_mode_distance(distances)
                        table_data.append([f"Class {label}", f"{np.mean(distances):.4f}", f"{mode_distance:.4f}", f"{np.max(distances):.4f}"])
                max_radii, point_counts = self.compute_max_radius(features_tsne, labels, centroids, unique_labels)
                self.display_radius_table(max_radii, point_counts, feature_name, "2D", "TruncatedSVD")
                max_legend_classes = 20
                if len(unique_labels) > max_legend_classes:
                    self.insert_metadata("Info", f"Showing {max_legend_classes} classes in legend. Full class list in centroid/distance files.")
                    handles, legend_labels = ax.get_legend_handles_labels()
                    ax.legend(handles[:max_legend_classes*2], legend_labels[:max_legend_classes*2], title=f"Classes: {len(unique_labels)}",
                            bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0., fontsize=8, ncol=4)
                else:
                    ax.legend(title=f"Classes: {len(unique_labels)}", bbox_to_anchor=(1.1, 1), loc='upper left',
                            borderaxespad=0., fontsize=8, ncol=2)
                self.insert_metadata("Info", f"Visualizing classes: {unique_labels}")
                max_table_classes = 10
                if len(unique_labels) > max_table_classes:
                    self.insert_metadata("Info", f"Showing stats for {max_table_classes} classes in table. Full stats in distance file.")
                    table_data = table_data[:max_table_classes]
                if table_data:
                    table_columns = ["Class", "Mean Distance", "Mode Distance", "Max Distance"]
                    table = ax.table(cellText=table_data, colLabels=table_columns, bbox=[1.1, 0.0, 0.7, 0.5], loc='right')
                    table.auto_set_font_size(False)
                    table.set_fontsize(7)
                    table.scale(1, 1.5)
            else:
                ax.scatter(x, y, c='blue', alpha=0.5, s=80)
            ax.set_title(f"2D t-SNE: Semantic Relationships ({feature_name})")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.subplots_adjust(right=0.57, top=1, bottom=0.1)
            try:
                canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(side="left", padx=10, pady=10)
                toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
                toolbar.update()
                output_dir = os.path.dirname(self.file_path) if os.path.isfile(self.file_path) else self.file_path
                output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_tsne_2d.png")
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                self.insert_metadata("Info", f"2D t-SNE plot saved to {output_file}")
                centroid_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_centroids_2d.txt")
                with open(centroid_file, 'w') as f:
                    f.write("Class Centroids in 2D t-SNE Space (x, y):\n")
                    for label, centroid in centroids:
                        f.write(f"Class {label}: ({centroid[0]:.4f}, {centroid[1]:.4f})\n")
                self.insert_metadata("Info", f"Centroid coordinates saved to {centroid_file}")
                for label, centroid in centroids:
                    self.insert_metadata("Info", f"Centroid for Class {label}: ({centroid[0]:.4f}, {centroid[1]:.4f})")
                distance_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_distances_2d.csv")
                with open(distance_file, 'w') as f:
                    f.write("Class,Mean Distance,Mode Distance,Max Distance,Individual Distances\n")
                    for label, distances in sorted(self.distances_per_class.items(), key=lambda x: x[0]):
                        mode_distance = self.compute_mode_distance(distances)
                        f.write(f"\"Class {label}\",{np.mean(distances):.4f},{mode_distance:.4f},{np.max(distances):.4f},\"{';'.join([f'{d:.4f}' for d in distances])}\"\n")
                self.insert_metadata("Info", f"Distances saved to {distance_file}")
            except Exception as e:
                self.insert_error(f"Failed to render 2D t-SNE plot for {feature_name}: {e}")
            self.insert_metadata("Success", f"2D t-SNE completed for {feature_name}. Closer points indicate similarity.")
            plt.close(fig)
            # Move heatmap to a separate window if centroids exist
            if len(centroids) > 1:
                centroid_array = np.array([c[1] for c in centroids])
                centroid_labels = [c[0] for c in centroids]
                dist_matrix = squareform(pdist(centroid_array, metric=self.distance_metric.get()))
                heatmap_window = tk.Toplevel(self.root)
                heatmap_window.title(f"Inter-Centroid Distances: {feature_name} (SVD 2D)")
                heatmap_window.geometry("800x600")
                heatmap_frame = tk.Frame(heatmap_window)
                heatmap_frame.pack(fill="both", expand=True)
                heatmap_fig, heatmap_ax = plt.subplots(figsize=(max(6, len(centroid_labels)), max(6, len(centroid_labels))))
                sns.heatmap(dist_matrix, annot=True, fmt='.2f', xticklabels=centroid_labels, yticklabels=centroid_labels,
                            cmap=plt.colormaps['viridis'], cbar_kws={'label': f'{self.distance_metric.get().capitalize()} Distance'}, ax=heatmap_ax)
                heatmap_ax.set_title(f"Inter-Centroid Distances: {feature_name} (SVD 2D)")
                plt.tight_layout()
                heatmap_canvas = FigureCanvasTkAgg(heatmap_fig, master=heatmap_frame)
                heatmap_canvas.draw()
                heatmap_canvas.get_tk_widget().pack(fill="both", expand=True)
                output_dir = os.path.dirname(self.file_path) if os.path.isfile(self.file_path) else self.file_path
                heatmap_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_inter_centroid_distances_svd_2d.png")
                heatmap_fig.savefig(heatmap_file, dpi=300, bbox_inches='tight')
                self.insert_metadata("Info", f"Inter-centroid distance heatmap saved to {heatmap_file}")
                plt.close(heatmap_fig)
        except Exception as e:
            self.insert_error(f"2D t-SNE failed for {feature_name}: {e}")
            plt.close('all')
            self.fallback_visualization(features, feature_name, labels=labels)

    def visualize_featuresPCA3D(self, features, labels=None, feature_name="Features", labels_name=None):
        if not self.is_feature_array(features):
            if features.ndim == 4 and features.shape[0] == 1:
                features = features.reshape(1, -1)
            else:
                self.insert_metadata("Error", f"Cannot visualize {feature_name} with shape {features.shape}.")
                self.fallback_visualization(features, feature_name, labels=labels)
                return
        n_samples, n_features = features.shape
        indices = None
        if n_samples > self.max_samples:
            self.insert_metadata("Info", f"Downsampling {feature_name} from {n_samples} to {self.max_samples} samples.")
            indices = np.random.choice(n_samples, self.max_samples, replace=False)
            features = features[indices]
            labels = labels[indices] if labels is not None else None
            n_samples = self.max_samples
        elif n_samples < 2:
            self.insert_metadata("Error", f"Single sample for {feature_name}; 3D t-SNE not applicable.")
            self.fallback_visualization(features, feature_name, labels=labels)
            return
        self.insert_metadata("Info", f"Normalizing features for {feature_name}...")
        features = StandardScaler().fit_transform(features)
        if n_features > 50:
            n_components = min(n_samples, n_features, 50)
            self.insert_metadata("Info", f"Reducing {n_features} features to {n_components} with PCA...")
            try:
                pca = PCA(n_components=n_components, random_state=42)
                features = pca.fit_transform(features)
            except ValueError as e:
                self.insert_error(f"PCA failed for {feature_name}: {e}")
                self.fallback_visualization(features, feature_name, labels=labels)
                return
        perplexity = min(5, n_samples - 1) if n_samples < 50 else min(30, n_samples - 1)
        self.insert_metadata("Info", f"Generating 3D t-SNE plot (perplexity={perplexity})...")
        self.root.update()
        try:
            tsne = TSNE(n_components=3, perplexity=perplexity, n_iter=1000, random_state=42, n_jobs=-1)
            features_tsne = tsne.fit_transform(features)
            fig = plt.figure(figsize=(20, 8))
            ax = fig.add_subplot(111, projection='3d')
            x = features_tsne[:, 0]
            y = features_tsne[:, 1]
            z = features_tsne[:, 2]
            centroids = []
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
                        ax.scatter(x[mask], y[mask], z[mask], c=[colors[idx]], marker=".", label=f"Class {label}", alpha=0.5, s=80)
                        centroid = np.mean(features_tsne[mask], axis=0)
                        centroids.append((label, centroid))
                        ax.scatter([centroid[0]], [centroid[1]], [centroid[2]], c=[colors[idx]], marker='X', s=200, edgecolors='black')
                        class_points = features_tsne[mask]
                        for point in class_points:
                            ax.plot([point[0], centroid[0]], [point[1], centroid[1]], [point[2], centroid[2]], c=colors[idx], alpha=0.3, linewidth=0.5)
                        distances = np.sqrt(np.sum((class_points - centroid) ** 2, axis=1))
                        self.distances_per_class[label] = distances
                        mode_distance = self.compute_mode_distance(distances)
                        table_data.append([f"Class {label}", f"{np.mean(distances):.4f}", f"{mode_distance:.4f}", f"{np.max(distances):.4f}"])
                max_radii, point_counts = self.compute_max_radius(features_tsne, labels, centroids, unique_labels)
                self.display_radius_table(max_radii, point_counts, feature_name, "3D", "PCA")
                max_legend_classes = 20
                if len(unique_labels) > max_legend_classes:
                    self.insert_metadata("Info", f"Showing {max_legend_classes} classes in legend. Full class list in centroid/distance files.")
                    handles, legend_labels = ax.get_legend_handles_labels()
                    ax.legend(handles[:max_legend_classes*2], legend_labels[:max_legend_classes*2], title=f"Classes: {len(unique_labels)}",
                            bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0., fontsize=8, ncol=2)
                else:
                    ax.legend(title=f"Classes: {len(unique_labels)}", bbox_to_anchor=(1.1, 1), loc='upper left',
                            borderaxespad=0., fontsize=8, ncol=2)
                self.insert_metadata("Info", f"Visualizing classes: {unique_labels}")
                max_table_classes = 10
                if len(unique_labels) > max_table_classes:
                    self.insert_metadata("Info", f"Showing stats for {max_table_classes} classes in table. Full stats in distance file.")
                    table_data = table_data[:max_table_classes]
                if table_data:
                    table_columns = ["Class", "Mean Distance", "Mode Distance", "Max Distance"]
                    table = ax.table(cellText=table_data, colLabels=table_columns, bbox=[1.1, 0.0, 0.5, 0.5], loc='right')
                    table.auto_set_font_size(False)
                    table.set_fontsize(7)
                    table.scale(1, 1.5)
                if len(centroids) > 1:
                    centroid_array = np.array([c[1] for c in centroids])
                    centroid_labels = [c[0] for c in centroids]
                    dist_matrix = squareform(pdist(centroid_array, metric=self.distance_metric.get()))
                    heatmap_window = tk.Toplevel(self.root)
                    heatmap_window.title(f"Inter-Centroid Distances: {feature_name} (PCA 3D)")
                    heatmap_window.geometry("800x600")
                    heatmap_frame = tk.Frame(heatmap_window)
                    heatmap_frame.pack(fill="both", expand=True)
                    canvas = tk.Canvas(heatmap_frame)
                    canvas.pack(side=tk.LEFT, fill="both", expand=True)
                    scrollbar = tk.Scrollbar(heatmap_frame, orient=tk.VERTICAL, command=canvas.yview)
                    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                    canvas.config(yscrollcommand=scrollbar.set)
                    inner_frame = tk.Frame(canvas)
                    canvas.create_window((0, 0), window=inner_frame, anchor="nw")
                    heatmap_fig, heatmap_ax = plt.subplots(figsize=(max(6, len(centroid_labels)), max(6, len(centroid_labels))))
                    sns.heatmap(dist_matrix, annot=True, fmt='.2f', xticklabels=centroid_labels, yticklabels=centroid_labels,
                                cmap=plt.colormaps['viridis'], cbar_kws={'label': f'{self.distance_metric.get().capitalize()} Distance'}, ax=heatmap_ax)
                    heatmap_ax.set_title(f"Inter-Centroid Distances: {feature_name} (PCA 3D)")
                    plt.tight_layout()
                    plot_canvas = FigureCanvasTkAgg(heatmap_fig, master=inner_frame)
                    plot_canvas.draw()
                    plot_canvas.get_tk_widget().pack(fill="both", expand=True)
                    inner_frame.update_idletasks()
                    canvas.config(scrollregion=canvas.bbox("all"))
                    output_dir = os.path.dirname(self.file_path) if os.path.isfile(self.file_path) else self.file_path
                    heatmap_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_inter_centroid_distances_pca_3d.png")
                    heatmap_fig.savefig(heatmap_file, dpi=300, bbox_inches='tight')
                    self.insert_metadata("Info", f"Inter-centroid distance heatmap saved to {heatmap_file}")
                    plt.close(heatmap_fig)
            else:
                ax.scatter(x, y, z, c='blue', alpha=0.5, s=80)
            ax.set_title(f"3D t-SNE: Semantic Relationships ({feature_name})")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_zlabel("t-SNE 3")
            plt.tight_layout()
            fig.subplots_adjust(right=0.45)
            try:
                canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(side="left", padx=10, pady=10)
                toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
                toolbar.update()
                output_dir = os.path.dirname(self.file_path) if os.path.isfile(self.file_path) else self.file_path
                output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_tsne_3d.png")
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                self.insert_metadata("Info", f"3D t-SNE plot saved to {output_file}")
                centroid_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_centroids_3d.txt")
                with open(centroid_file, 'w') as f:
                    f.write("Class Centroids in 3D t-SNE Space (x, y, z):\n")
                    for label, centroid in centroids:
                        f.write(f"Class {label}: ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})\n")
                self.insert_metadata("Info", f"Centroid coordinates saved to {centroid_file}")
                for label, centroid in centroids:
                    self.insert_metadata("Info", f"Centroid for Class {label}: ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")
                distance_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_distances_3d.csv")
                with open(distance_file, 'w') as f:
                    f.write("Class,Mean Distance,Mode Distance,Max Distance,Individual Distances\n")
                    for label, distances in sorted(self.distances_per_class.items(), key=lambda x: x[0]):
                        mode_distance = self.compute_mode_distance(distances)
                        f.write(f"\"Class {label}\",{np.mean(distances):.4f},{mode_distance:.4f},{np.max(distances):.4f},\"{';'.join([f'{d:.4f}' for d in distances])}\"\n")
                self.insert_metadata("Info", f"Distances saved to {distance_file}")
            except Exception as e:
                self.insert_error(f"Failed to render 3D t-SNE plot for {feature_name}: {e}")
            self.insert_metadata("Success", f"3D t-SNE completed for {feature_name}. Closer points indicate similarity.")
            plt.close(fig)
        except Exception as e:
            self.insert_error(f"3D t-SNE failed for {feature_name}: {e}")
            plt.close('all')
            self.fallback_visualization(features, feature_name, labels=labels)

    def visualize_featuresSVD3D(self, features, labels=None, feature_name="Features", labels_name=None):
        if not self.is_feature_array(features):
            if features.ndim == 4 and features.shape[0] == 1:
                features = features.reshape(1, -1)
            else:
                self.insert_metadata("Error", f"Cannot visualize {feature_name} with shape {features.shape}.")
                self.fallback_visualization(features, feature_name, labels=labels)
                return
        n_samples, n_features = features.shape
        indices = None
        if n_samples > self.max_samples:
            self.insert_metadata("Info", f"Downsampling {feature_name} from {n_samples} to {self.max_samples} samples.")
            indices = np.random.choice(n_samples, self.max_samples, replace=False)
            features = features[indices]
            labels = labels[indices] if labels is not None else None
            n_samples = self.max_samples
        elif n_samples < 2:
            self.insert_metadata("Error", f"Single sample for {feature_name}; 3D t-SNE not applicable.")
            self.fallback_visualization(features, feature_name, labels=labels)
            return
        self.insert_metadata("Info", f"Normalizing features for {feature_name}...")
        features = StandardScaler().fit_transform(features)
        if n_features > 50:
            n_components = min(n_samples, n_features, 50)
            self.insert_metadata("Info", f"Reducing {n_features} features to {n_components} with TruncatedSVD...")
            try:
                pca = TruncatedSVD(n_components=n_components, random_state=42)
                features = pca.fit_transform(features)
            except ValueError as e:
                self.insert_error(f"TruncatedSVD failed for {feature_name}: {e}")
                self.fallback_visualization(features, feature_name, labels=labels)
                return
        perplexity = min(5, n_samples - 1) if n_samples < 50 else min(30, n_samples - 1)
        self.insert_metadata("Info", f"Generating 3D t-SNE plot (perplexity={perplexity})...")
        self.root.update()
        try:
            tsne = TSNE(n_components=3, perplexity=perplexity, n_iter=1000, random_state=42, n_jobs=-1)
            features_tsne = tsne.fit_transform(features)
            fig = plt.figure(figsize=(20, 8))
            ax = fig.add_subplot(111, projection='3d')
            x = features_tsne[:, 0]
            y = features_tsne[:, 1]
            z = features_tsne[:, 2]
            centroids = []
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
                        ax.scatter(x[mask], y[mask], z[mask], c=[colors[idx]], marker=".", label=f"Class {label}", alpha=0.5, s=80)
                        centroid = np.mean(features_tsne[mask], axis=0)
                        centroids.append((label, centroid))
                        ax.scatter([centroid[0]], [centroid[1]], [centroid[2]], c=[colors[idx]], marker='X', s=200, edgecolors='black')
                        class_points = features_tsne[mask]
                        for point in class_points:
                            ax.plot([point[0], centroid[0]], [point[1], centroid[1]], [point[2], centroid[2]], c=colors[idx], alpha=0.3, linewidth=0.5)
                        distances = np.sqrt(np.sum((class_points - centroid) ** 2, axis=1))
                        self.distances_per_class[label] = distances
                        mode_distance = self.compute_mode_distance(distances)
                        table_data.append([f"Class {label}", f"{np.mean(distances):.4f}", f"{mode_distance:.4f}", f"{np.max(distances):.4f}"])
                max_radii, point_counts = self.compute_max_radius(features_tsne, labels, centroids, unique_labels)
                self.display_radius_table(max_radii, point_counts, feature_name, "3D", "TruncatedSVD")
                max_legend_classes = 20
                if len(unique_labels) > max_legend_classes:
                    self.insert_metadata("Info", f"Showing {max_legend_classes} classes in legend. Full class list in centroid/distance files.")
                    handles, legend_labels = ax.get_legend_handles_labels()
                    ax.legend(handles[:max_legend_classes*2], legend_labels[:max_legend_classes*2], title=f"Classes: {len(unique_labels)}",
                            bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0., fontsize=8, ncol=2)
                else:
                    ax.legend(title=f"Classes: {len(unique_labels)}", bbox_to_anchor=(1.1, 1), loc='upper left',
                            borderaxespad=0., fontsize=8, ncol=2)
                self.insert_metadata("Info", f"Visualizing classes: {unique_labels}")
                max_table_classes = 10
                if len(unique_labels) > max_table_classes:
                    self.insert_metadata("Info", f"Showing stats for {max_table_classes} classes in table. Full stats in distance file.")
                    table_data = table_data[:max_table_classes]
                if table_data:
                    table_columns = ["Class", "Mean Distance", "Mode Distance", "Max Distance"]
                    table = ax.table(cellText=table_data, colLabels=table_columns, bbox=[1.1, 0.0, 0.5, 0.5], loc='right')
                    table.auto_set_font_size(False)
                    table.set_fontsize(7)
                    table.scale(1, 1.5)
                if len(centroids) > 1:
                    centroid_array = np.array([c[1] for c in centroids])
                    centroid_labels = [c[0] for c in centroids]
                    dist_matrix = squareform(pdist(centroid_array, metric=self.distance_metric.get()))
                    heatmap_window = tk.Toplevel(self.root)
                    heatmap_window.title(f"Inter-Centroid Distances: {feature_name} (TruncatedSVD 3D)")
                    heatmap_window.geometry("800x600")
                    heatmap_frame = tk.Frame(heatmap_window)
                    heatmap_frame.pack(fill="both", expand=True)
                    canvas = tk.Canvas(heatmap_frame)
                    canvas.pack(side=tk.LEFT, fill="both", expand=True)
                    scrollbar = tk.Scrollbar(heatmap_frame, orient=tk.VERTICAL, command=canvas.yview)
                    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                    canvas.config(yscrollcommand=scrollbar.set)
                    inner_frame = tk.Frame(canvas)
                    canvas.create_window((0, 0), window=inner_frame, anchor="nw")
                    heatmap_fig, heatmap_ax = plt.subplots(figsize=(max(6, len(centroid_labels)), max(6, len(centroid_labels))))
                    sns.heatmap(dist_matrix, annot=True, fmt='.2f', xticklabels=centroid_labels, yticklabels=centroid_labels,
                                cmap=plt.colormaps['viridis'], cbar_kws={'label': f'{self.distance_metric.get().capitalize()} Distance'}, ax=heatmap_ax)
                    heatmap_ax.set_title(f"Inter-Centroid Distances: {feature_name} (TruncatedSVD 3D)")
                    plt.tight_layout()
                    plot_canvas = FigureCanvasTkAgg(heatmap_fig, master=inner_frame)
                    plot_canvas.draw()
                    plot_canvas.get_tk_widget().pack(fill="both", expand=True)
                    inner_frame.update_idletasks()
                    canvas.config(scrollregion=canvas.bbox("all"))
                    output_dir = os.path.dirname(self.file_path) if os.path.isfile(self.file_path) else self.file_path
                    heatmap_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_inter_centroid_distances_svd_3d.png")
                    heatmap_fig.savefig(heatmap_file, dpi=300, bbox_inches='tight')
                    self.insert_metadata("Info", f"Inter-centroid distance heatmap saved to {heatmap_file}")
                    plt.close(heatmap_fig)
            else:
                ax.scatter(x, y, z, c='blue', alpha=0.5, s=80)
            ax.set_title(f"3D t-SNE: Semantic Relationships ({feature_name})")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_zlabel("t-SNE 3")
            plt.tight_layout()
            fig.subplots_adjust(right=0.45)
            try:
                canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(side="left", padx=10, pady=10)
                toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
                toolbar.update()
                output_dir = os.path.dirname(self.file_path) if os.path.isfile(self.file_path) else self.file_path
                output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_tsne_3d.png")
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                self.insert_metadata("Info", f"3D t-SNE plot saved to {output_file}")
                centroid_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_centroids_3d.txt")
                with open(centroid_file, 'w') as f:
                    f.write("Class Centroids in 3D t-SNE Space (x, y, z):\n")
                    for label, centroid in centroids:
                        f.write(f"Class {label}: ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})\n")
                self.insert_metadata("Info", f"Centroid coordinates saved to {centroid_file}")
                for label, centroid in centroids:
                    self.insert_metadata("Info", f"Centroid for Class {label}: ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")
                distance_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_distances_3d.csv")
                with open(distance_file, 'w') as f:
                    f.write("Class,Mean Distance,Mode Distance,Max Distance,Individual Distances\n")
                    for label, distances in sorted(self.distances_per_class.items(), key=lambda x: x[0]):
                        mode_distance = self.compute_mode_distance(distances)
                        f.write(f"\"Class {label}\",{np.mean(distances):.4f},{mode_distance:.4f},{np.max(distances):.4f},\"{';'.join([f'{d:.4f}' for d in distances])}\"\n")
                self.insert_metadata("Info", f"Distances saved to {distance_file}")
            except Exception as e:
                self.insert_error(f"Failed to render 3D t-SNE plot for {feature_name}: {e}")
            self.insert_metadata("Success", f"3D t-SNE completed for {feature_name}. Closer points indicate similarity.")
            plt.close(fig)
        except Exception as e:
            self.insert_error(f"3D t-SNE failed for {feature_name}: {e}")
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
                canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(side="left", padx=10, pady=10)
                toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
                toolbar.update()
                output_dir = os.path.dirname(self.file_path) if os.path.isfile(self.file_path) else self.file_path
                output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_fallback_{arr_name}.png")
                fig.savefig(output_file, dpi=100, bbox_inches='tight')
                self.insert_metadata("Info", f"Saved fallback plot to {output_file}")
            except Exception as e:
                self.insert_error(f"Failed to render fallback plot for {arr_name}: {e}")
            self.insert_metadata("Info", f"Fallback visualization completed for {arr_name}.")
            plt.close(fig)
        except Exception as e:
            self.insert_error(f"Failed to generate fallback visualization for {arr_name}: {e}")
            plt.close('all')

    def visualize_all(self):
        if self.data is None:
            return
        self.clear_plot_frame()
        if self.opt.get() == "PCA" and self.dim.get() == "2D":
            for key, arr in self.data.items():
                if self.is_feature_array(arr):
                    self.visualize_featuresPCA(arr, labels=self.labels, feature_name=key, labels_name='labels')
                else:
                    self.insert_metadata("Info", f"Skipping {key}: not a valid feature array (shape {arr.shape}, dtype {arr.dtype})")
                self.root.update()
        elif self.opt.get() == "TruncatedSVD" and self.dim.get() == "2D":
            for key, arr in self.data.items():
                if self.is_feature_array(arr):
                    self.visualize_featuresTruncatedSVD(arr, labels=self.labels, feature_name=key, labels_name='labels')
                else:
                    self.insert_metadata("Info", f"Skipping {key}: not a valid feature array (shape {arr.shape}, dtype {arr.dtype})")
                self.root.update()
        elif self.opt.get() == "PCA" and self.dim.get() == "3D":
            for key, arr in self.data.items():
                if self.is_feature_array(arr):
                    self.visualize_featuresPCA3D(arr, labels=self.labels, feature_name=key, labels_name='labels')
                else:
                    self.insert_metadata("Info", f"Skipping {key}: not a valid feature array (shape {arr.shape}, dtype {arr.dtype})")
                self.root.update()
        elif self.opt.get() == "TruncatedSVD" and self.dim.get() == "3D":
            for key, arr in self.data.items():
                if self.is_feature_array(arr):
                    self.visualize_featuresSVD3D(arr, labels=self.labels, feature_name=key, labels_name='labels')
                else:
                    self.insert_metadata("Info", f"Skipping {key}: not a valid feature array (shape {arr.shape}, dtype {arr.dtype})")
                self.root.update()

def main(file_path=None):
    root = tk.Tk()
    app = NpyVisualizerApp(root, "2D", "PCA", "euclidean")
    root.mainloop()

if __name__ == "__main__":
    main()
