import streamlit as st
import numpy as np
import cv2
import torch
import torchvision
import plotly.graph_objects as go
import os
import glob

from PIL import Image
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter
)
from torchvision.transforms import functional as F
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os
import gdown




# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PruneNet Playground",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 100%;
}

[data-testid="stSidebar"] {
    background-color: #161A23;
}

h1, h2, h3, h4 {
    color: white;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
}
div[data-testid="stVerticalBlockBorderWrapper"]{
    height: 760px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "mango_maskrcnn.pth"


@st.cache_resource
def get_model_file():

    if not os.path.exists(
        "mango_maskrcnn.pth"
    ):

        gdown.download(
            "https://drive.google.com/uc?id=1jAV4hpKG1gitAduyYmC8tE6IKNLlEauw",
            "mango_maskrcnn.pth",
            quiet=False
        )

    return "mango_maskrcnn.pth"
MODEL_PATH = get_model_file()
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================================================
# FIXED COLORS (NO FLICKER)
# =========================================================

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
]

# =========================================================
# SESSION STATE
# =========================================================

if "selected_tree" not in st.session_state:
    st.session_state.selected_tree = 0

if "sample_image" not in st.session_state:
    st.session_state.sample_image = None

# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=False
    )

    in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features,
            2
        )
    )

    in_features_mask = (
        model.roi_heads.mask_predictor.conv5_mask.in_channels
    )

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        256,
        2
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model

# =========================================================
# FILTER OVERLAPS
# =========================================================

def filter_overlapping_masks(
    masks,
    boxes,
    threshold=0.3
):

    if len(masks) == 0:
        return [], []

    sorted_indices = np.argsort(
        [np.sum(m) for m in masks]
    )[::-1]

    masks = [masks[i] for i in sorted_indices]
    boxes = [boxes[i] for i in sorted_indices]

    keep_masks = np.ones(
        len(masks),
        dtype=bool
    )

    for i in range(len(masks)):

        if not keep_masks[i]:
            continue

        for j in range(i + 1, len(masks)):

            if keep_masks[j]:

                overlap = np.logical_and(
                    masks[i],
                    masks[j]
                ).sum()

                min_area = min(
                    np.sum(masks[i]),
                    np.sum(masks[j])
                )

                if min_area == 0:
                    continue

                overlap_ratio = overlap / min_area

                if overlap_ratio > threshold:
                    keep_masks[j] = False

    filtered_masks = [
        masks[i]
        for i in range(len(masks))
        if keep_masks[i]
    ]

    filtered_boxes = [
        boxes[i]
        for i in range(len(boxes))
        if keep_masks[i]
    ]

    return filtered_masks, filtered_boxes

# =========================================================
# CREATE OVERLAY
# =========================================================

@st.cache_data(show_spinner=False)
def create_overlay(image_rgb, masks):

    overlay = image_rgb.copy()

    for idx, mask in enumerate(masks):

        color = COLORS[idx % len(COLORS)]

        color_layer = np.zeros_like(overlay)

        for c in range(3):

            color_layer[:, :, c] = (
                mask.astype(np.uint8) * color[c]
            )

        overlay = cv2.addWeighted(
            overlay,
            1.0,
            color_layer,
            0.45,
            0
        )

        ys, xs = np.where(mask)

        if len(xs) > 0:

            center_x = int(np.mean(xs))
            center_y = int(np.mean(ys))

            cv2.putText(
                overlay,
                f"{idx}",
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3
            )

    return overlay

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("PruneNet Playground")

uploaded_file = st.sidebar.file_uploader(
    "Upload RGB Image",
    type=["jpg", "jpeg", "png"]
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.5,
    0.05
)

# =========================================================
# MAIN UI
# =========================================================

st.title("Interactive Tree Canopy Analysis")

if (
    uploaded_file is None
    and
    st.session_state.sample_image is None
):

    st.markdown("## Featured Samples")

    sample_paths = []

    for root, _, files in os.walk("samples"):

        for file in files:

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                sample_paths.append(
                    os.path.join(root, file)
                )

    sample_paths = sorted(sample_paths)

    if len(sample_paths) > 0:

        cols_per_row = 4

        for row_start in range(
            0,
            len(sample_paths),
            cols_per_row
        ):

            cols = st.columns(cols_per_row)

            for col_idx in range(cols_per_row):

                img_idx = row_start + col_idx

                if img_idx >= len(sample_paths):
                    break

                img_path = sample_paths[img_idx]

                with cols[col_idx]:

                    st.image(
                        img_path,
                        use_container_width=True
                    )

                    st.markdown(
                        f"**{os.path.basename(img_path)}**"
                    )

                    if st.button(
                        "Open",
                        key=f"sample_{img_idx}"
                    ):

                        st.session_state.sample_image = img_path

                        st.session_state.selected_tree = 0

                        st.rerun()

if st.session_state.sample_image is not None:

    st.success(
        f"Selected Sample: "
        f"{os.path.basename(st.session_state.sample_image)}"
    )

if (
    uploaded_file is not None
    or
    st.session_state.sample_image is not None
):

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image_bgr = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

    else:

        image_bgr = cv2.imread(
            st.session_state.sample_image
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )
    
    # =====================================================
    # LOAD MODEL
    # =====================================================

    with st.spinner("Loading model..."):

        model = load_model()

    # =====================================================
    # INFERENCE
    # =====================================================

    tensor_img = F.to_tensor(
        Image.fromarray(image_rgb)
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        preds = model(tensor_img)[0]

    pred_masks_raw = preds["masks"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()
    boxes = preds["boxes"].cpu().numpy()

    # =====================================================
    # FILTER
    # =====================================================

    pred_masks = [
        pred_masks_raw[i, 0] > 0.5
        for i in range(len(scores))
        if scores[i] > confidence_threshold
    ]

    pred_boxes = [
        boxes[i]
        for i in range(len(scores))
        if scores[i] > confidence_threshold
    ]

    pred_masks, pred_boxes = filter_overlapping_masks(
        pred_masks,
        pred_boxes
    )
    if st.session_state.selected_tree >= len(pred_masks):
        st.session_state.selected_tree = 0

    if len(pred_masks) == 0:

        st.warning("No trees detected.")
        st.stop()

    # =====================================================
    # OVERLAY
    # =====================================================

    overlay = create_overlay(
        image_rgb,
        pred_masks
    )

# =====================================================
# QUADRANT LAYOUT
# =====================================================
    if st.button(
        "⬅ Back"
    ):
        st.session_state.sample_image = None
        st.session_state.selected_tree = 0
        st.rerun()
    col1, col2, col3 = st.columns(3)    

    # =====================================================
    # LEFT PANEL
    # =====================================================

    with col1:
        with st.container(border=True):

            st.subheader("Segmentation Overlay")

            st.image(
                overlay,
                use_container_width=True
            )
            
            st.image(
                segmentation_display,
                use_container_width=True
            )

            st.markdown("### Select Tree")

            button_cols = st.columns(4)

            for idx in range(len(pred_masks)):

                if button_cols[idx % 4].button(
                    f"Tree {idx}",
                    key=f"tree_{idx}"
                ):
                    st.session_state.selected_tree = idx
                    st.rerun()

        # =====================================================
        # SELECTED TREE
        # =====================================================

        selected_tree = st.session_state.selected_tree

        if selected_tree is None:

            st.info(
                "Select a tree to analyze."
            )

            st.stop()

        mask = pred_masks[selected_tree]
        # =====================================================
        # EXTRACT TREE
        # =====================================================

        masked_tree = image_rgb.copy()

        masked_tree[~mask] = 0

        gray = cv2.cvtColor(
            masked_tree,
            cv2.COLOR_RGB2GRAY
        )

        gray_smoothed = gaussian_filter(
            gray.astype(np.float32),
            sigma=5
        )

        # =====================================================
        # DISTANCE TRANSFORM
        # =====================================================

        dist_transform = distance_transform_edt(
            gray > 0
        )

        max_dist = np.max(dist_transform)

        center_y, center_x = np.unravel_index(
            np.argmax(dist_transform),
            dist_transform.shape
        )

        radius = int(max_dist)

        reduced_radius = int(radius * 0.9)

        # =====================================================
        # ANALYSIS MASK (distance from boundary)
        # =====================================================
        
        margin = int(radius * 0.10)   # remove outer 10% of canopy
        
        analysis_mask = dist_transform >= margin
        
        reduced_inscribed_masked_image = cv2.bitwise_and(
            masked_tree,
            masked_tree,
            mask=analysis_mask.astype(np.uint8) * 255
        )
        # =====================================================
        # POINT CLOUD
        # =====================================================

        circle_y_indices, circle_x_indices = np.where(
            analysis_mask
        )

        depth = (
            gray[
                circle_y_indices,
                circle_x_indices
            ].astype(np.float32)
            / 255.0
        )

        depth = depth * 100

        height, width = gray.shape
        
        x = circle_x_indices.astype(np.float32)
        y = circle_y_indices.astype(np.float32)
        
        # Invert image Y-axis
        y = height - y
        
        # Rotate 180° in the XY plane
        x = width - x
        y = height - y
        
        points_3d = np.column_stack(
        (
            x,
            y,
            depth.astype(np.float32)
        ))

        sorted_indices = np.argsort(-depth)

        points_3d = points_3d[sorted_indices]

        colors = (
            reduced_inscribed_masked_image[
                circle_y_indices,
                circle_x_indices
            ] / 255.0
        )

        colors = colors[sorted_indices]


    # =====================================================
    # PSEUDO DEPTH OPENINGS
    # =====================================================

        valid_circle = analysis_mask

        pseudo_depth_vals = (
            gray[valid_circle]
            / 255.0
        )

        pseudo_thresh = np.percentile(
            pseudo_depth_vals,
            10
        )

        pseudo_openings = (
            (gray / 255.0)
            <= pseudo_thresh
        ) & valid_circle

        opening_flags = pseudo_openings[
            circle_y_indices,
            circle_x_indices
        ]

        opening_flags = opening_flags[
            sorted_indices
        ]
        pseudo_opening_pixels = np.sum(
            pseudo_openings
        )
        colors[opening_flags] = [1.0, 0.0, 0.0]
        total_tree_pixels = np.sum(
            mask
        )

        pseudo_opening_percent = (
            pseudo_opening_pixels
            /
            total_tree_pixels
        ) * 100
        # =====================================================
        # OPENING OVERLAY
        # =====================================================
        
        overlay = masked_tree.copy()
        
        # Highlight detected openings in red
        overlay[pseudo_openings] = [255, 0, 0]
        
        # Crop to tree bounding box
        ys, xs = np.where(mask)
        
        ymin, ymax = np.min(ys), np.max(ys)
        xmin, xmax = np.min(xs), np.max(xs)
        
        overlay = overlay[
            ymin:ymax+1,
            xmin:xmax+1
        ]
        
        # Add border
        overlay = cv2.copyMakeBorder(
            overlay,
            50,
            50,
            50,
            50,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        
        # Resize for display
        overlay = cv2.resize(
            overlay,
            (600, 600)
        )
    # =====================================================
    # RIGHT PANEL
    # =====================================================

    with col2:
        with st.container(border=True):

            st.subheader(
                f"Tree {selected_tree} - 3D Canopy"
            )

            x_vals = points_3d[:, 0]
            y_vals = points_3d[:, 1]
            z_vals = points_3d[:, 2]

            rgb_colors = [
                f'rgb({int(c[0]*255)},'
                f'{int(c[1]*255)},'
                f'{int(c[2]*255)})'
                for c in colors
            ]

            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=x_vals,
                        y=y_vals,
                        z=z_vals,
                        mode='markers',
                        marker=dict(
                            size=2,
                            color=rgb_colors,
                            opacity=0.8
                        )
                    )
                ]
            )

            fig.update_layout(
                height=700,
                paper_bgcolor="#0E1117",
                plot_bgcolor="#0E1117",
                font=dict(color="white"),
                margin=dict(l=0, r=0, t=20, b=0),
                scene=dict(
                    bgcolor="#0E1117",
                    xaxis=dict(title="X"),
                    yaxis=dict(title="Y"),
                    zaxis=dict(title="Depth"),
                    aspectmode="data",
            
                    camera=dict(
                        eye=dict(
                            x=0,
                            y=0,
                            z=2.5
                        ),
                        center=dict(
                            x=0,
                            y=0,
                            z=0
                        ),
                        up=dict(
                            x=0,
                            y=-1,
                            z=0
                        )
                    )
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )
    with col3:
        with st.container(border=True):

            st.subheader(
                "Detected Crown Openings"
            )

            scale = np.linspace(
                255,
                0,
                400
            ).astype(np.uint8)

            scale = np.tile(
                scale.reshape(-1, 1),
                (1, 40)
            )

            scale = cv2.applyColorMap(
                scale,
                cv2.COLORMAP_JET
            )

            scale = cv2.cvtColor(
                scale,
                cv2.COLOR_BGR2RGB
            )

                st.image(
                    overlay,
                    use_container_width=True
                )
st.sidebar.markdown("---")
st.sidebar.subheader("Results")

st.sidebar.metric(
    "Opening %",
    f"{pseudo_opening_percent:.2f}%"
)

st.sidebar.metric(
    "Tree Pixels",
    f"{int(total_tree_pixels):,}"
)

st.sidebar.metric(
    "Opening Pixels",
    f"{int(pseudo_opening_pixels):,}"
)
