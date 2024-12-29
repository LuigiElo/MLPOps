#!/usr/bin/env python
# coding: utf-8

# # Combining Grounding DINO with Segment Anything (SAM) for text-based mask generation
# 
# This is based on the popular [Grounded Segment Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) project.
# 
# <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/grounded_sam.png"
# alt="drawing" width="900"/>
# 
# <small> Grounded SAM overview. Taken from the <a href="https://github.com/IDEA-Research/Grounded-Segment-Anything">original repository</a>. </small>
# 
import os
from pathlib import Path
import cv2
import numpy as np
from sklearn.metrics import f1_score
#from typing import Dict, Any
import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import cv2
import torch
import requests
import numpy as np
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

get_ipython().system('pip install --upgrade -q git+https://github.com/huggingface/transformers')

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))


# ## Plot Utils

def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: Optional[str] = None
) -> None:
    annotated_image = annotate(image, detections)
    plt.imshow(annotated_image)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()

def random_named_css_colors(num_colors: int) -> List[str]:
    """
    Returns a list of randomly selected named CSS colors.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - list: List of randomly selected named CSS colors.
    """
    # List of named CSS colors
    named_css_colors = [
        'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond',
        'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite',
        'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
        'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
        'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
        'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
        'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
        'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive',
        'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
        'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
        'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey',
        'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
        'whitesmoke', 'yellow', 'yellowgreen'
    ]

    # Sample random named CSS colors
    return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))

def plot_detections_plotly(
    image: np.ndarray,
    detections: List[DetectionResult],
    class_colors: Optional[Dict[str, str]] = None
) -> None:
    # If class_colors is not provided, generate random colors for each class
    if class_colors is None:
        num_detections = len(detections)
        colors = random_named_css_colors(num_detections)
        class_colors = {}
        for i in range(num_detections):
            class_colors[i] = colors[i]


    fig = px.imshow(image)

    # Add bounding boxes
    shapes = []
    annotations = []
    for idx, detection in enumerate(detections):
        label = detection.label
        box = detection.box
        score = detection.score
        mask = detection.mask

        polygon = mask_to_polygon(mask)

        fig.add_trace(go.Scatter(
            x=[point[0] for point in polygon] + [polygon[0][0]],
            y=[point[1] for point in polygon] + [polygon[0][1]],
            mode='lines',
            line=dict(color=class_colors[idx], width=2),
            fill='toself',
            name=f"{label}: {score:.2f}"
        ))

        xmin, ymin, xmax, ymax = box.xyxy
        shape = [
            dict(
                type="rect",
                xref="x", yref="y",
                x0=xmin, y0=ymin,
                x1=xmax, y1=ymax,
                line=dict(color=class_colors[idx])
            )
        ]
        annotation = [
            dict(
                x=(xmin+xmax) // 2, y=(ymin+ymax) // 2,
                xref="x", yref="y",
                text=f"{label}: {score:.2f}",
            )
        ]

        shapes.append(shape)
        annotations.append(annotation)

    # Update layout
    button_shapes = [dict(label="None",method="relayout",args=["shapes", []])]
    button_shapes = button_shapes + [
        dict(label=f"Detection {idx+1}",method="relayout",args=["shapes", shape]) for idx, shape in enumerate(shapes)
    ]
    button_shapes = button_shapes + [dict(label="All", method="relayout", args=["shapes", sum(shapes, [])])]

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        # margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        updatemenus=[
            dict(
                type="buttons",
                direction="up",
                buttons=button_shapes
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Show plot
    fig.show()


# ## Utils

def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a polygon to a segmentation mask.

    Args:
    - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
    - image_shape (tuple): Shape of the image (height, width) for the mask.

    Returns:
    - np.ndarray: Segmentation mask with the polygon filled.
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")

    return image

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks


# ## Grounded Segment Anything (SAM)
# 
# The approach:
# 1. use Grounding DINO to detect a given set of texts ('players' in this case, so both players from Real Madrid and Manchester United) in the image. The output is a set of bounding boxes.
# 2. prompt Segment Anything (SAM) with the bounding boxes, for which the model will output segmentation masks.

def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results

def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, threshold, detector_id)
    detections = segment(image, detections, polygon_refinement, segmenter_id)

    return np.array(image), detections


# ### Inference
# (Change images paths )

#image_url = "/content/drive/MyDrive/DTU/MLOPs/archive/images/Frame 1  (74).jpg"
image_url = "mlsopsbasic/data/archive/images/Frame 1  (74).jpg"
#mask_url ="/content/drive/MyDrive/DTU/MLOPs/archive/images/Frame 1  (74).jpg___fuse.png"
mask_url = "mlsopsbasic/data/archive/images/Frame 1  (74).jpg__fuse.png"
labels = ["players from Real Madrid and Manchester United"]
threshold = 0.3

detector_id = "IDEA-Research/grounding-dino-tiny"
segmenter_id = "facebook/sam-vit-base"

# Function to process a single image
def process_image(image_file, labels, threshold):
  image_array, detections = grounded_segmentation(
      image=str(image_file),
      labels=labels,
      threshold=threshold,
      polygon_refinement=True,
      detector_id=detector_id,
      segmenter_id=segmenter_id
  )
  return image_array, detections


# In[11]:


image_array, detections = process_image(image_url, labels, threshold)
print("Image_array: ", image_array.shape)
print("Detections: ", detections)

ground_truth_mask = cv2.imread(str(mask_url), cv2.IMREAD_GRAYSCALE)
ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)

def calculate_iou(detection_mask, ground_truth_mask):
    intersection = np.logical_and(detection_mask, ground_truth_mask)
    union = np.logical_or(detection_mask, ground_truth_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_pixel_accuracy(detection_mask, ground_truth_mask):
  return np.mean(detection_mask == ground_truth_mask)

def calculate_mean_absolute_error(detection_mask, ground_truth_mask):
  return np.mean(np.abs(detection_mask.astype(float) - ground_truth_mask.astype(float)))


# ## trial - evaluation functions
def evaluate_detection(detection_mask: np.ndarray, ground_truth_mask: np.ndarray) -> Dict[str, float]:
  # Make masks binary
  detection_mask = detection_mask.astype(bool)
  ground_truth_mask = ground_truth_mask.astype(bool)

  # Intersection over Union - IoU
  iou = calculate_iou(detection_mask, ground_truth_mask)

  # Pixel Accuracy
  pixel_accuracy = calculate_pixel_accuracy(detection_mask, ground_truth_mask)

  # Mean Absolute Error
  mae = calculate_mean_absolute_error(detection_mask, ground_truth_mask)

  # F1 score
  f1 = f1_score(ground_truth_mask.flatten(), detection_mask.flatten())

  return {
      'iou': iou,
      'pixel_accuracy': pixel_accuracy,
      'mae': mae,
      'f1_score': f1
  }

evaluate_detection(detections[0].mask, ground_truth_mask)


# Let's visualize the results:

plot_detections(image_array, detections)

plot_detections_plotly(image_array, detections)


# ## Iterating through dataset and getting name

#root_dir = Path("/content/drive/MyDrive/DTU/MLOPs/archive/images")
root_dir = Path("/mlsopsbasic/data/archive/images")

# See if root directory exists
print(f"Root directory exists: {root_dir.exists()}")

def process_football_dataset(root_dir):
    root_dir = Path(root_dir)

    # Verify root directory exists
    #print(f"Root directory exists: {root_dir.exists()}")

    # Get all original images (files without ___fuse.png or ___save.png)
    image_files = [f for f in root_dir.glob("Frame*.jpg") if not str(f).endswith(("___fuse.png", "___save.png"))]
    print(f"Number of images found: {len(image_files)}")

    all_results = []

    for jpg_file in image_files:
        print(f"Processing {jpg_file.name}")

        # Construct paths for corresponding mask files
        fuse_mask = root_dir / f"{jpg_file.name}___fuse.png"
        save_mask = root_dir / f"{jpg_file.name}___save.png"

        # Verify mask files exist
        if fuse_mask.exists() and save_mask.exists():
            print(f"Fuse mask exists: {fuse_mask}")
            print(f"Save mask exists: {save_mask}")

            # Read image and masks
            image = cv2.imread(str(jpg_file))

            # You can choose which mask to use (fuse or save) based on your needs
            # Here we're using the fuse mask as an example
            ground_truth_mask = cv2.imread(str(fuse_mask), cv2.IMREAD_GRAYSCALE)
            ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)

            # Apply the detection method - no plots
            image_array, detections = process_image(str(jpg_file), labels, threshold)

            # Resize ground truth mask to match the shape of the detection mask
            if len(detections) > 0:  # Check if any detections were made
                ground_truth_mask = cv2.resize(
                    ground_truth_mask,
                    (detections[0].mask.shape[1], detections[0].mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

                # Evaluate detection
                result = evaluate_detection(detections[0].mask, ground_truth_mask)
                result['image'] = jpg_file.name

                all_results.append(result)

                print(f"Processed {jpg_file.name}; Results: {result}")
            else:
                print(f"No detections found for {jpg_file.name}")
        else:
            print(f"Mask files not found for {jpg_file.name}")

    return all_results

# Usage
#root_dir = Path("/path/to/your/football/dataset/images")
results = process_football_dataset(root_dir)

print(len(results))
print(results)

# Calculate averages
if results:
    # Averages for each metric across all images
    avg_metrics = {}
    for metric in ['iou', 'pixel_accuracy', 'mae', 'f1_score']:
        avg_metrics[metric] = sum(result[metric] for result in results) / len(results)

    print("Average metrics across all images:")
    for metric, value in avg_metrics.items():
        print(f"Average {metric}: {value:.4f}")

    # Averages per split
    splits = set(result['split'] for result in results)
    for split in splits:
        split_results = [result for result in results if result['split'] == split]
        print(f"\nAverage metrics for {split}:")
        for metric in ['iou', 'pixel_accuracy', 'mae', 'f1_score']:
            avg = sum(result[metric] for result in split_results) / len(split_results)
            print(f"Average {metric}: {avg:.4f}")

    # Overall averages without splits
    print("\nOverall average metrics:")
    for metric in ['iou', 'pixel_accuracy', 'mae', 'f1_score']:
        overall_avg = sum(result[metric] for result in results) / len(results)
        print(f"Overall average {metric}: {overall_avg:.4f}")

print(results[0])

# Plot All_results // Calculate TP, FP, FN TN -- and then derive FPR and FNR.

def calculate_tp(detection_mask: np.ndarray, ground_truth_mask: np.ndarray) -> int:
    """Calculate True Positives."""
    return np.sum(np.logical_and(detection_mask, ground_truth_mask))

def calculate_fp(detection_mask: np.ndarray, ground_truth_mask: np.ndarray) -> int:
    """Calculate False Positives."""
    return np.sum(np.logical_and(detection_mask, np.logical_not(ground_truth_mask)))

def calculate_fn(detection_mask: np.ndarray, ground_truth_mask: np.ndarray) -> int:
    """Calculate False Negatives."""
    return np.sum(np.logical_and(np.logical_not(detection_mask), ground_truth_mask))

def calculate_tn(detection_mask: np.ndarray, ground_truth_mask: np.ndarray) -> int:
    """Calculate True Negatives."""
    return np.sum(np.logical_and(np.logical_not(detection_mask), np.logical_not(ground_truth_mask)))

def calculate_fpr(fp: int, tn: int) -> float:
    """Calculate False Positive Rate."""
    return fp / (fp + tn) if (fp + tn) > 0 else 0

def calculate_fnr(fn: int, tp: int) -> float:
    """Calculate False Negative Rate."""
    return fn / (fn + tp) if (fn + tp) > 0 else 0

def calculate_all_metrics(detection_mask: np.ndarray, ground_truth_mask: np.ndarray) -> dict:
    """Calculate all metrics."""
    # Ensure masks are binary
    detection_mask = detection_mask.astype(bool)
    ground_truth_mask = ground_truth_mask.astype(bool)

    tp = calculate_tp(detection_mask, ground_truth_mask)
    fp = calculate_fp(detection_mask, ground_truth_mask)
    fn = calculate_fn(detection_mask, ground_truth_mask)
    tn = calculate_tn(detection_mask, ground_truth_mask)
    fpr = calculate_fpr(fp, tn)
    fnr = calculate_fnr(fn, tp)

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'FPR': fpr,
        'FNR': fnr
    }

def derive_fpr_fnr_metrics(result):
    # Assuming binary classification and that pixel_accuracy is (TP + TN) / (TP + TN + FP + FN)
    total_pixels = 1  # Normalized to 1 for simplicity
    correct_pixels = result['pixel_accuracy']
    incorrect_pixels = 1 - correct_pixels

    # Derive TP, FP, FN, TN
    tp = result['f1_score'] * correct_pixels  # Approximation
    fn = correct_pixels - tp
    fp = incorrect_pixels - fn
    tn = total_pixels - (tp + fp + fn)

    # Calculate FPR and FNR
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'FPR': fpr,
        'FNR': fnr
    }

# Apply the derivation to all results
for result in results:
    result.update(derive_fpr_fnr_metrics(result))

# Calculate averages
avg_metrics = {}
for metric in ['TP', 'FP', 'FN', 'TN', 'FPR', 'FNR', 'iou', 'pixel_accuracy', 'mae', 'f1_score']:
    avg_metrics[metric] = sum(result[metric] for result in results) / len(results)

print("Average metrics across all images:")
for metric, value in avg_metrics.items():
    print(f"Average {metric}: {value:.4f}")

# Calculate averages per split
splits = set(result['split'] for result in results)
for split in splits:
    split_results = [result for result in results if result['split'] == split]
    print(f"\nAverage metrics for {split}:")
    for metric in ['TP', 'FP', 'FN', 'TN', 'FPR', 'FNR', 'iou', 'pixel_accuracy', 'mae', 'f1_score']:
        avg = sum(result[metric] for result in split_results) / len(split_results)
        print(f"Average {metric}: {avg:.4f}")

# Overall averages
print("\nOverall average metrics:")
for metric in ['TP', 'FP', 'FN', 'TN', 'FPR', 'FNR', 'iou', 'pixel_accuracy', 'mae', 'f1_score']:
    overall_avg = sum(result[metric] for result in results) / len(results)
    print(f"Overall average {metric}: {overall_avg:.4f}")


#print(all_results[0])
print(results[0])

# Save results in a file
import json
from pathlib import Path
import os

# Define the path to the new directory
#output_dir = "/content/drive/MyDrive/output"
output_dir = "/MLOPs/project/MLPOps/"

# Create the directory
os.makedirs(output_dir, exist_ok=True)

print(f"Directory '{output_dir}' created successfully!")

# Define the file path (modify as needed)
output_file = Path("/content/drive/MyDrive/output/all_results.json")
output_file = Path("/MLOPs/project/MLPOps/all_results.json")

# Save the results to a JSON file
with open(output_file, 'w') as f:
    #json.dump(all_results, f, indent=4)
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")