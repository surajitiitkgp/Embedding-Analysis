# ```python
import numpy as np
import matplotlib.pyplot as plt
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

# Setup logging
logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

warnings.simplefilter("ignore", category=UserWarning)

class NpyVisualizerApp:
    def __init__(self, root, file_path=None):
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
        self.frame.rowconfigure(4, weight=1)
        self.dim = tk.StringVar(value="2D")
        self.dropdown_label1 = tk.Label(self.frame, text="Choose Dimension:")
        self.dropdown_label1.grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.dropdown_dim = tk.OptionMenu(self.frame, self.dim, "2D", "3D")
        self.dropdown_dim.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.opt = tk.StringVar(value="PCA")
        self.dropdown_label = tk.Label(self.frame, text="Choose Reduction Method:")
        self.dropdown_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.dropdown_opt = tk.OptionMenu(self.frame, self.opt, "PCA", "TruncatedSVD")
        self.dropdown_opt.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.btn_load_folder = tk.Button(self.frame, text="Load Folder of NPY Files", command=self.load_folder)
        self.btn_load_folder.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        # Disable manual NPZ loading button to prevent confusion
        self.btn_load_file = tk.Button(self.frame, text="Load NPZ File", command=self.load_npz)
        self.btn_load_file.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self.metadata_text = tk.Text(self.frame, height=12, width=50)
        self.metadata_text.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.scrollbar = tk.Scrollbar(self.frame, command=self.metadata_text.yview)
        self.scrollbar.grid(row=3, column=2, sticky="ns")
        self.metadata_text.config(yscrollcommand=self.scrollbar.set)
        self.plot_frame = tk.Frame(self.frame)
        self.plot_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.data = None
        self.file_path = None
        self.labels = None
        self.max_samples = 1000
        self.distances_per_class = {}
        if file_path:
            try:
                # Normalize and validate path
                file_path = str(Path(file_path).resolve())
                logging.debug(f"Initializing with file_path: {file_path}")
                print(f"Initializing with file_path: {file_path}")  # Debug
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
        print(f"Attempting to load NPZ file: {file_path}")  # Debug
        try:
            # Validate file integrity
            with zipfile.ZipFile(file_path, 'r') as zf:
                if zf.testzip() is not None:
                    raise zipfile.BadZipFile("NPZ file is corrupted")
            logging.debug(f"NPZ file integrity verified: {file_path}")
            print(f"NPZ file integrity verified: {file_path}")  # Debug
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
            print(f"Successfully loaded file: {file_path}")  # Debug
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
                    return {'features': features, 'inferred_labels': labels}
        window_size = 1024
        n_samples = n_elements // window_size
        if n_samples > 1:
            features = data[:n_samples * window_size].reshape(n_samples, window_size)
            self.insert_metadata("Info", f"Segmented into {n_samples} samples x {window_size} features.")
            n_clusters = min(10, n_samples)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(StandardScaler().fit_transform(features))
            self.insert_metadata("Info", f"Generated {n_clusters} pseudo-labels using K-means.")
            return {'features': features, 'inferred_labels': labels}
        self.insert_metadata("Error", "Failed to reshape or segment 1D array into features.")
        return None

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
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(folder_path, file_name)
                    data = np.load(file_path)
                    if data.ndim == 4 and data.shape[0] == 1:
                        flattened = data.flatten()
                        if expected_shape is None:
                            expected_shape = flattened.shape
                        elif flattened.shape != expected_shape:
                            self.insert_error(
                                f"Dimension mismatch in {file_name}: got shape {flattened.shape}, expected {expected_shape}"
                            )
                            logging.error(f"Dimension mismatch in {file_name}")
                            return
                        features.append(flattened)
                        label = os.path.splitext(file_name)[0].split('_')[0]
                        labels.append(label)
                    else:
                        self.insert_metadata("Warning", f"Skipping {file_name}: expected 4D array with batch=1, got shape {data.shape}")
                        logging.warning(f"Skipped {file_name}: invalid shape")
            if features:
                try:
                    self.data = {'features': np.vstack(features)}
                    self.labels = np.array(labels)
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
                    self.display_metadata()
                    self.visualize_all()
                    logging.debug(f"Successfully loaded folder: {folder_path}")
                except ValueError as e:
                    self.insert_error(f"Failed to stack features: {e}")
                    logging.error(f"Failed to stack features: {e}")
            else:
                self.insert_metadata("Error", "No valid .npy files found in folder!")
                logging.error("No valid .npy files in folder")
        except Exception as e:
            self.insert_error(f"Error loading folder: {e}")
            logging.error(f"Error loading folder: {e}")

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

    def clear_plot_frame(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        self.distances_per_class = {}

    def is_label_array(self, arr):
        return arr.ndim == 1 and len(np.unique(arr)) <= 100

    def is_feature_array(self, arr):
        return arr.ndim == 2 and arr.shape[1] >= 2 and np.issubdtype(arr.dtype, np.number)

    def get_label_array(self, data, feature_key):
        return self.labels, 'labels' if self.labels is not None else (None, None)

    def visualize_featuresPCA(self, features, labels=None, feature_name="Features", labels_name=None):
        if not self.is_feature_array(features):
            if features.ndim == 4 and features.shape[0] == 1:
                features = features.reshape(1, -1)
            else:
                self.insert_metadata("Error", f"Cannot visualize {feature_name} with shape {features.shape}. Expected 2D or 4D (1, C, H, W).")
                self.fallback_visualization(features, feature_name, labels=labels)
                return
        n_samples, n_features = features.shape
        indices = None
        if n_samples > self.max_samples:
            self.insert_metadata("Info", f"Downsampling {feature_name} from {n_samples} to {self.max_samples} samples for t-SNE.")
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
            self.insert_metadata("Info", f"Reducing {n_features} features to {n_components} with PCA for t-SNE...")
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
            distance_text = []
            if labels is not None:
                labels = np.array(labels, dtype=str)
                unique_labels = np.unique(labels)
                if len(unique_labels) <= 100:
                    colors = plt.cm.get_cmap('hsv')(np.linspace(0, 1, len(unique_labels)))
                    for idx, label in enumerate(unique_labels):
                        mask = labels == label
                        if np.sum(mask) > 0:
                            ax.scatter(x[mask], y[mask], c=[colors[idx]], marker=".", label=f"Class {label}", alpha=0.7, s=60)
                            centroid = np.mean(features_tsne[mask], axis=0)
                            centroids.append((label, centroid))
                            ax.scatter([centroid[0]], [centroid[1]], c=[colors[idx]], marker='X', s=200, edgecolors='black', label=f"Centroid {label}")
                            class_points = features_tsne[mask]
                            for point in class_points:
                                ax.plot([point[0], centroid[0]], [point[1], centroid[1]], c=colors[idx], alpha=0.3, linewidth=0.5)
                            distances = np.sqrt(np.sum((class_points - centroid) ** 2, axis=1))
                            self.distances_per_class[label] = distances
                            distance_text.append(f"Class {label} Distance Stats:\nMean={np.mean(distances):.4f}, Min={np.min(distances):.4f}, Max={np.max(distances):.4f}")
                    ax.legend(title=f"Classes: {len(unique_labels)}", fancybox=True, frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                    self.insert_metadata("Info", f"Visualizing classes: {unique_labels}")
                    distance_str = "\n".join(sorted(distance_text, key=lambda x: x.split()[1]))
                    ax.text(1.10, 0.7, distance_str, transform=ax.transAxes, fontsize=8, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
                else:
                    self.insert_metadata("Warning", f"Too many unique labels ({len(unique_labels)}) for {feature_name}; using single color.")
                    ax.scatter(x, y, c='blue', alpha=0.7, s=60)
            else:
                ax.scatter(x, y, c='blue', alpha=0.7, s=60)
            ax.set_title(f"2D t-SNE: Semantic Relationships ({feature_name})")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.subplots_adjust(right=0.60, top=1, bottom=0.1)
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
                distance_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_distances_2d.txt")
                with open(distance_file, 'w') as f:
                    f.write("Distances from Data Points to Centroids in 2D t-SNE Space:\n")
                    for label, distances in sorted(self.distances_per_class.items(), key=lambda x: x[0]):
                        f.write(f"\nClass {label}:\n")
                        f.write(f"  Mean Distance: {np.mean(distances):.4f}\n")
                        f.write(f"  Min Distance: {np.min(distances):.4f}\n")
                        f.write(f"  Max Distance: {np.max(distances):.4f}\n")
                        f.write("  Individual Distances:\n")
                        for i, dist in enumerate(distances):
                            f.write(f"    Point {i}: {dist:.4f}\n")
                self.insert_metadata("Info", f"Distances saved to {distance_file}")
            except Exception as e:
                self.insert_error(f"Failed to render 2D t-SNE plot for {feature_name}: {e}")
            self.insert_metadata("Success", f"2D t-SNE completed for {feature_name}. Closer points indicate similarity.")
            plt.close(fig)
        except Exception as e:
            self.insert_error(f"2D t-SNE failed for {feature_name}: {e}")
            plt.close('all')
            self.fallback_visualization(features, feature_name, labels=labels)

    def visualize_featuresTruncatedSVD(self, features, labels=None, feature_name="Features", labels_name=None):
        if not self.is_feature_array(features):
            if features.ndim == 4 and features.shape[0] == 1:
                features = features.reshape(1, -1)
            else:
                self.insert_metadata("Error", f"Cannot visualize {feature_name} with shape {features.shape}. Expected 2D or 4D (1, C, H, W).")
                self.fallback_visualization(features, feature_name, labels=labels)
                return
        n_samples, n_features = features.shape
        indices = None
        if n_samples > self.max_samples:
            self.insert_metadata("Info", f"Downsampling {n_samples} to {self.max_samples} samples for {feature_name}.")
            indices = np.random.choice(n_samples, self.max_samples, replace=False)
            features = features[indices]
            labels = labels[indices] if labels is not None else None
            n_samples = self.max_samples
        elif n_samples < 2:
            self.insert_metadata("Error", f"Too few samples for {feature_name}: {n_samples}. Need at least 2.")
            self.fallback_visualization(features, feature_name, labels=labels)
            return
        self.insert_metadata("Info", f"Normalizing features for {feature_name}...")
        features = StandardScaler().fit_transform(features)
        if n_features > 50:
            n_components = min(n_samples, n_features, 50)
            self.insert_metadata("Info", f"Reducing {n_features} features to {n_components} with TruncatedSVD for t-SNE...")
            try:
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                features = svd.fit_transform(features)
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
            distance_text = []
            if labels is not None:
                labels = np.array(labels, dtype=str)
                unique_labels = np.unique(labels)
                if len(unique_labels) <= 100:
                    colors = plt.cm.get_cmap('hsv')(np.linspace(0, 1, len(unique_labels)))
                    for idx, label in enumerate(unique_labels):
                        mask = labels == label
                        if np.sum(mask) > 0:
                            ax.scatter(x[mask], y[mask], c=[colors[idx]], marker=".", label=f"Class {label}", alpha=0.7, s=60)
                            centroid = np.mean(features_tsne[mask], axis=0)
                            centroids.append((label, centroid))
                            ax.scatter([centroid[0]], [centroid[1]], c=[colors[idx]], marker='X', s=200, edgecolors='black', label=f"Centroid {label}")
                            class_points = features_tsne[mask]
                            for point in class_points:
                                ax.plot([point[0], centroid[0]], [point[1], centroid[1]], c=colors[idx], alpha=0.3, linewidth=0.5)
                            distances = np.sqrt(np.sum((class_points - centroid) ** 2, axis=1))
                            self.distances_per_class[label] = distances
                            distance_text.append(f"Class {label} Distance Stats:\nMean={np.mean(distances):.4f}, Min={np.min(distances):.4f}, Max={np.max(distances):.4f}")
                    ax.legend(title=f"Classes: {len(unique_labels)}", fancybox=True, frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                    self.insert_metadata("Info", f"Visualizing classes: {unique_labels}")
                    distance_str = "\n".join(sorted(distance_text, key=lambda x: x.split()[1]))
                    ax.text(1.10, 0.7, distance_str, transform=ax.transAxes, fontsize=8, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
                else:
                    self.insert_metadata("Warning", f"Too many unique labels ({len(unique_labels)}) for {feature_name}; using single color.")
                    ax.scatter(x, y, c='blue', alpha=0.7, s=60)
            else:
                ax.scatter(x, y, c='blue', alpha=0.7, s=60)
            ax.set_title(f"2D t-SNE: Semantic Relationships ({feature_name})")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.subplots_adjust(right=0.60, top=1, bottom=0.1)
            try:
                canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(side="left", padx=10, pady=10)
                toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
                toolbar.update()
                output_dir = os.path.dirname(self.file_path) if os.path.isfile(self.file_path) else self.file_path
                output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_tsne_2d_svd.png")
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                self.insert_metadata("Info", f"2D t-SNE plot saved to {output_file}")
                centroid_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_centroids_2d_svd.txt")
                with open(centroid_file, 'w') as f:
                    f.write("Class Centroids in 2D t-SNE Space (TruncatedSVD) (x, y):\n")
                    for label, centroid in centroids:
                        f.write(f"Class {label}: ({centroid[0]:.4f}, {centroid[1]:.4f})\n")
                self.insert_metadata("Info", f"Centroid coordinates saved to {centroid_file}")
                for label, centroid in centroids:
                    self.insert_metadata("Info", f"Centroid for Class {label}: ({centroid[0]:.4f}, {centroid[1]:.4f})")
                distance_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_distances_2d_svd.txt")
                with open(distance_file, 'w') as f:
                    f.write("Distances from Data Points to Centroids in 2D t-SNE Space (TruncatedSVD):\n")
                    for label, distances in sorted(self.distances_per_class.items(), key=lambda x: x[0]):
                        f.write(f"\nClass {label}:\n")
                        f.write(f"  Mean Distance: {np.mean(distances):.4f}\n")
                        f.write(f"  Min Distance: {np.min(distances):.4f}\n")
                        f.write(f"  Max Distance: {np.max(distances):.4f}\n")
                        f.write("  Individual Distances:\n")
                        for i, dist in enumerate(distances):
                            f.write(f"    Point {i}: {dist:.4f}\n")
                self.insert_metadata("Info", f"Distances saved to {distance_file}")
            except Exception as e:
                self.insert_error(f"Failed to render 2D t-SNE plot for {feature_name}: {e}")
            self.insert_metadata("Success", f"2D t-SNE completed for {feature_name}. Closer points indicate similarity.")
            plt.close(fig)
        except Exception as e:
            self.insert_error(f"2D t-SNE failed for {feature_name}: {e}")
            plt.close('all')
            self.fallback_visualization(features, feature_name, labels=labels)

    def visualize_featuresPCA3D(self, features, labels=None, feature_name="Features", labels_name=None):
        if not self.is_feature_array(features):
            if features.ndim == 4 and features.shape[0] == 1:
                features = features.reshape(1, -1)
            else:
                self.insert_metadata("Error", f"Cannot visualize {feature_name} with shape {features.shape}. Expected 2D or 4D (1, C, H, W).")
                self.fallback_visualization(features, feature_name, labels=labels)
                return
        n_samples, n_features = features.shape
        indices = None
        if n_samples > self.max_samples:
            self.insert_metadata("Info", f"Downsampling {feature_name} from {n_samples} to {self.max_samples} samples for t-SNE.")
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
            self.insert_metadata("Info", f"Reducing {n_features} features to {n_components} with PCA for t-SNE...")
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
            distance_text = []
            if labels is not None:
                labels = np.array(labels, dtype=str)
                unique_labels = np.unique(labels)
                if len(unique_labels) <= 100:
                    colors = plt.cm.get_cmap('hsv')(np.linspace(0, 1, len(unique_labels)))
                    for idx, label in enumerate(unique_labels):
                        mask = labels == label
                        if np.sum(mask) > 0:
                            ax.scatter(x[mask], y[mask], z[mask], c=[colors[idx]], marker=".", label=f"Class {label}", alpha=0.3, s=60)
                            centroid = np.mean(features_tsne[mask], axis=0)
                            centroids.append((label, centroid))
                            ax.scatter([centroid[0]], [centroid[1]], [centroid[2]], c=[colors[idx]], marker='X', s=200, edgecolors='black', label=f"Centroid {label}")
                            class_points = features_tsne[mask]
                            for point in class_points:
                                ax.plot([point[0], centroid[0]], [point[1], centroid[1]], [point[2], centroid[2]], c=colors[idx], alpha=0.3, linewidth=0.5)
                            distances = np.sqrt(np.sum((class_points - centroid) ** 2, axis=1))
                            self.distances_per_class[label] = distances
                            distance_text.append(f"Class {label} Distance Stats:\nMean={np.mean(distances):.4f}, Min={np.min(distances):.4f}, Max={np.max(distances):.4f}")
                    ax.legend(title=f"Classes: {len(unique_labels)}", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                    self.insert_metadata("Info", f"Visualizing semantic relationships for classes: {unique_labels}")
                    distance_str = "\n".join(sorted(distance_text, key=lambda x: x.split()[1]))
                    ax.text2D(1.10, 0.7, distance_str, transform=ax.transAxes, fontsize=8, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
                else:
                    self.insert_metadata("Warning", f"Too many unique labels ({len(unique_labels)}) for {feature_name}; using single color.")
                    ax.scatter(x, y, z, c='blue', alpha=0.7, s=60)
            else:
                ax.scatter(x, y, z, c='blue', alpha=0.7, s=60)
            ax.set_title(f"3D t-SNE: Semantic Relationships Between Classes ({feature_name})")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_zlabel("t-SNE 3")
            plt.tight_layout()
            fig.subplots_adjust(right=0.50)
            try:
                canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(side="left", padx=10, pady=10)
                toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
                toolbar.update()
                output_dir = os.path.dirname(self.file_path) if os.path.isfile(self.file_path) else self.file_path
                output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_tsne_3d.png")
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                self.insert_metadata("Info", f"3D t-SNE plot showing semantic relationships saved to {output_file}")
                centroid_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_centroids_3d.txt")
                with open(centroid_file, 'w') as f:
                    f.write("Class Centroids in 3D t-SNE Space (x, y, z):\n")
                    for label, centroid in centroids:
                        f.write(f"Class {label}: ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})\n")
                self.insert_metadata("Info", f"Centroid coordinates saved to {centroid_file}")
                for label, centroid in centroids:
                    self.insert_metadata("Info", f"Centroid for Class {label}: ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")
                distance_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_distances_3d.txt")
                with open(distance_file, 'w') as f:
                    f.write("Distances from Data Points to Centroids in 3D t-SNE Space:\n")
                    for label, distances in sorted(self.distances_per_class.items(), key=lambda x: x[0]):
                        f.write(f"\nClass {label}:\n")
                        f.write(f"  Mean Distance: {np.mean(distances):.4f}\n")
                        f.write(f"  Min Distance: {np.min(distances):.4f}\n")
                        f.write(f"  Max Distance: {np.max(distances):.4f}\n")
                        f.write("  Individual Distances:\n")
                        for i, dist in enumerate(distances):
                            f.write(f"    Point {i}: {dist:.4f}\n")
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
                self.insert_metadata("Error", f"Cannot visualize {feature_name} with shape {features.shape}. Expected 2D or 4D (1, C, H, W).")
                self.fallback_visualization(features, feature_name, labels=labels)
                return
        n_samples, n_features = features.shape
        indices = None
        if n_samples > self.max_samples:
            self.insert_metadata("Info", f"Downsampling {feature_name} from {n_samples} to {self.max_samples} samples for t-SNE.")
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
            self.insert_metadata("Info", f"Reducing {n_features} features to {n_components} with TruncatedSVD for t-SNE...")
            try:
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                features = svd.fit_transform(features)
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
            distance_text = []
            if labels is not None:
                labels = np.array(labels, dtype=str)
                unique_labels = np.unique(labels)
                if len(unique_labels) <= 100:
                    colors = plt.cm.get_cmap('hsv')(np.linspace(0, 1, len(unique_labels)))
                    for idx, label in enumerate(unique_labels):
                        mask = labels == label
                        if np.sum(mask) > 0:
                            ax.scatter(x[mask], y[mask], z[mask], c=[colors[idx]], marker=".", label=f"Class {label}", alpha=0.3, s=60)
                            centroid = np.mean(features_tsne[mask], axis=0)
                            centroids.append((label, centroid))
                            ax.scatter([centroid[0]], [centroid[1]], [centroid[2]], c=[colors[idx]], marker='X', s=200, edgecolors='black', label=f"Centroid {label}")
                            class_points = features_tsne[mask]
                            for point in class_points:
                                ax.plot([point[0], centroid[0]], [point[1], centroid[1]], [point[2], centroid[2]], c=colors[idx], alpha=0.3, linewidth=0.5)
                            distances = np.sqrt(np.sum((class_points - centroid) ** 2, axis=1))
                            self.distances_per_class[label] = distances
                            distance_text.append(f"Class {label} Distance Stats:\nMean={np.mean(distances):.4f}, Min={np.min(distances):.4f}, Max={np.max(distances):.4f}")
                    ax.legend(title=f"Classes: {len(unique_labels)}", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                    self.insert_metadata("Info", f"Visualizing semantic relationships for classes: {unique_labels}")
                    distance_str = "\n".join(sorted(distance_text, key=lambda x: x.split()[1]))
                    ax.text2D(1.10, 0.7, distance_str, transform=ax.transAxes, fontsize=8, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
                else:
                    self.insert_metadata("Warning", f"Too many unique labels ({len(unique_labels)}) for {feature_name}; using single color.")
                    ax.scatter(x, y, z, c='blue', alpha=0.7, s=60)
            else:
                ax.scatter(x, y, z, c='blue', alpha=0.7, s=60)
            ax.set_title(f"3D t-SNE: Semantic Relationships Between Classes ({feature_name})")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_zlabel("t-SNE 3")
            plt.tight_layout()
            fig.subplots_adjust(right=0.50)
            try:
                canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(side="left", padx=10, pady=10)
                toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
                toolbar.update()
                output_dir = os.path.dirname(self.file_path) if os.path.isfile(self.file_path) else self.file_path
                output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_tsne_3d_svd.png")
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                self.insert_metadata("Info", f"3D t-SNE plot showing semantic relationships saved to {output_file}")
                centroid_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_centroids_3d_svd.txt")
                with open(centroid_file, 'w') as f:
                    f.write("Class Centroids in 3D t-SNE Space (TruncatedSVD) (x, y, z):\n")
                    for label, centroid in centroids:
                        f.write(f"Class {label}: ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})\n")
                self.insert_metadata("Info", f"Centroid coordinates saved to {centroid_file}")
                for label, centroid in centroids:
                    self.insert_metadata("Info", f"Centroid for Class {label}: ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")
                distance_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_distances_3d_svd.txt")
                with open(distance_file, 'w') as f:
                    f.write("Distances from Data Points to Centroids in 3D t-SNE Space (TruncatedSVD):\n")
                    for label, distances in sorted(self.distances_per_class.items(), key=lambda x: x[0]):
                        f.write(f"\nClass {label}:\n")
                        f.write(f"  Mean Distance: {np.mean(distances):.4f}\n")
                        f.write(f"  Min Distance: {np.min(distances):.4f}\n")
                        f.write(f"  Max Distance: {np.max(distances):.4f}\n")
                        f.write("  Individual Distances:\n")
                        for i, dist in enumerate(distances):
                            f.write(f"    Point {i}: {dist:.4f}\n")
                self.insert_metadata("Info", f"Distances saved to {distance_file}")
            except Exception as e:
                self.insert_error(f"Failed to render 3D t-SNE plot for {feature_name}: {e}")
            self.insert_metadata("Success", f"3D t-SNE completed for {feature_name}. Closer points indicate similarity.")
            plt.close(fig)
        except Exception as e:
            self.insert_error(f"3D t-SNE failed for {feature_name}: {e}")
            plt.close('all')
            self.fallback_visualization(features, feature_name, labels=labels)

    def visualize_labels(self, labels, label_name="Labels"):
        self.insert_metadata("Info", f"Label array: {label_name}, Unique labels: {len(np.unique(labels))}")

    def fallback_visualization(self, arr, arr_name, labels=None):
        try:
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            centroids = []
            self.distances_per_class = {}
            distance_text = []
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
                    if len(unique_labels) <= 100:
                        colors = plt.cm.get_cmap('hsv')(np.linspace(0, 1, len(unique_labels)))
                        for idx, label in enumerate(unique_labels):
                            mask = labels == label
                            if np.sum(mask) > 0:
                                ax.scatter(x[mask], y[mask], c=[colors[idx]], marker=".", label=f"Class {label}", alpha=0.7, s=60)
                                centroid = np.mean(np.array([x[mask], y[mask]]).T, axis=0)
                                centroids.append((label, centroid))
                                ax.scatter([centroid[0]], [centroid[1]], c=[colors[idx]], marker='X', s=200, edgecolors='black', label=f"Centroid {label}")
                                class_points = np.array([x[mask], y[mask]]).T
                                for point in class_points:
                                    ax.plot([point[0], centroid[0]], [point[1], centroid[1]], c=colors[idx], alpha=0.3, linewidth=0.5)
                                distances = np.sqrt(np.sum((class_points - centroid) ** 2, axis=1))
                                self.distances_per_class[label] = distances
                                distance_text.append(f"Class {label} Distance Stats:\nMean={np.mean(distances):.4f}, Min={np.min(distances):.4f}, Max={np.max(distances):.4f}")
                        ax.legend(title=f"Classes: {len(unique_labels)}", fancybox=True, frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')
                        self.insert_metadata("Info", f"Visualizing classes in fallback: {unique_labels}")
                        distance_str = "\n".join(sorted(distance_text, key=lambda x: x.split()[1]))
                        ax.text(1.10, 0.7, distance_str, transform=ax.transAxes, fontsize=8, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
                    else:
                        self.insert_metadata("Warning", f"Too many unique labels ({len(unique_labels)}) for {arr_name}; using single color.")
                        ax.scatter(x, y, c='blue', alpha=0.6)
                else:
                    ax.scatter(x, y, c='blue', alpha=0.6)
                ax.set_title(f"First Two Features: {arr_name}")
                ax.set_xlabel("Feature 0")
                ax.set_ylabel("Feature 1")
                if centroids:
                    output_dir = os.path.dirname(self.file_path) if os.path.isfile(self.file_path) else self.file_path
                    centroid_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_fallback_centroids_{arr_name}.txt")
                    with open(centroid_file, 'w') as f:
                        f.write("Class Centroids in Fallback 2D Space (x, y):\n")
                        for label, centroid in centroids:
                            f.write(f"Class {label}: ({centroid[0]:.4f}, {centroid[1]:.4f})\n")
                    self.insert_metadata("Info", f"Centroid coordinates saved to {centroid_file}")
                    distance_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(self.file_path))[0]}_fallback_distances_{arr_name}.txt")
                    with open(distance_file, 'w') as f:
                        f.write("Distances from Data Points to Centroids in Fallback 2D Space:\n")
                        for label, distances in sorted(self.distances_per_class.items(), key=lambda x: x[0]):
                            f.write(f"\nClass {label}:\n")
                            f.write(f"  Mean Distance: {np.mean(distances):.4f}\n")
                            f.write(f"  Min Distance: {np.min(distances):.4f}\n")
                            f.write(f"  Max Distance: {np.max(distances):.4f}\n")
                            f.write("  Individual Distances:\n")
                            for i, dist in enumerate(distances):
                                f.write(f"    Point {i}: {dist:.4f}\n")
                    self.insert_metadata("Info", f"Distances saved to {distance_file}")
            elif arr.ndim == 4 and arr.shape[0] == 1:
                ax.imshow(arr[0, 0, :, :], cmap='viridis')
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
            plt.close(fig)
        except Exception as e:
            self.insert_error(f"Failed to visualize fallback for {arr_name}: {e}")
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
    app = NpyVisualizerApp(root, file_path)
    root.mainloop()

if __name__ == "__main__":
    main()
# ```