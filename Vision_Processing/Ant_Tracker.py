"""
Ant tracking system using color-based detection and DBSCAN clustering.
Tracks individual ants across video frames and optionally detects
when an ant passes over a fungus region.
"""

import cv2 as cv
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt


def detect_ant_coordinates(frame):
    """
    Detects pixels that likely belong to an ant based on color thresholds.

    Ants are identified by pixels where the red channel dominates over green
    and blue (reddish-brown tone), and whose absolute brightness falls within
    a mid-dark range to exclude both highlights and deep shadows.

    Args:
        frame: A BGR image frame (numpy array of shape H x W x 3).

    Returns:
        A numpy array of shape (N, 2) containing [x, y] coordinates of all
        pixels that match the ant color criteria. Returns an empty array if
        no matching pixels are found.
    """
    coordinates = []
    # frame[:,:,0] is blue, frame[:,:,1] is green, frame[:,:,2] is red
    
    # Extract channels and cast to int32 to prevent overflow in arithmetic
    b = frame[:, :, 0].astype(np.int32)
    g = frame[:, :, 1].astype(np.int32)
    r = frame[:, :, 2].astype(np.int32)
    
    # Conditions for a pixel to be considered a possible ant segment
    cond_green  = r > (1.35 * g)   # Red significantly exceeds green
    cond_blue   = r > (1.35 * b)   # Red significantly exceeds blue
    cond_thresh = r > 68            # Minimum brightness (not pure black)
    cond_dark_r = r < 140           # Maximum red (not too bright)
    cond_dark_g = g < 90            # Green stays low
    cond_dark_b = b < 90            # Blue stays low
    
    # Combine all conditions
    ant_mask = cond_green & cond_blue & cond_thresh & cond_dark_r & cond_dark_g & cond_dark_b
    y_indices, x_indices = np.where(ant_mask)
    for x, y in zip(x_indices, y_indices):
        coordinates.append([x, y])
        
    return np.array(coordinates)


def individualize_points(coordinates, eps, min_samples, size_bias=1, use_average_filter=True):
    """
    Clusters a set of 2D points using DBSCAN and optionally removes clusters
    that are significantly smaller than the average cluster size.

    Args:
        coordinates:        Array of shape (N, 2) with [x, y] point coordinates.
        eps:                Maximum distance between two points to be considered
                            neighbors in DBSCAN.
        min_samples:        Minimum number of points required to form a dense
                            region (core point) in DBSCAN.
        size_bias:          Multiplier applied to the average cluster size when
                            deciding which small clusters to discard. Values < 1
                            make the filter more lenient. Default is 1.
        use_average_filter: If True, removes clusters whose size is below
                            (average_cluster_size * size_bias). Default is True.

    Returns:
        A numpy array of integer labels, one per input point. Label -1 indicates
        noise (points not assigned to any cluster).
    """
    # Cluster with DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
    labels = clustering.labels_

    # Remove clusters that are likely noise based on size
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) != 1 and use_average_filter:
        noise_count = np.count_nonzero(labels == -1)
        num_real_clusters = len(unique_labels) - 1  # Exclude noise label (-1)
        avg_points_per_cluster = ((len(labels) - noise_count) / num_real_clusters) * size_bias
        is_small = (counts < avg_points_per_cluster) & (unique_labels != -1)
        labels_to_remove = unique_labels[is_small]
        labels[np.isin(labels, labels_to_remove)] = -1

    return labels


def remove_noise(labels, coordinates, mask):
    """
    Sets noise points (label == -1) to False in the provided boolean mask.

    Args:
        labels:      Array of DBSCAN labels, one per coordinate. -1 denotes noise.
        coordinates: Array of shape (N, 2) with [x, y] coordinates corresponding
                     to each label.
        mask:        2D boolean numpy array (H x W) where True marks detected pixels.

    Returns:
        The updated mask with noise pixel positions set to False.
    """
    # Find indices where label is -1 (noise)
    noise_indices = np.where(labels == -1)
    noise_coords = coordinates[noise_indices]
    noise_x = noise_coords[:, 0]
    noise_y = noise_coords[:, 1]

    # Mark noise pixels as False in the mask
    mask[noise_y, noise_x] = False
    return mask


def get_cluster_bounding_coords(labels, coordinates_x, coordinates_y, label):
    """
    Returns the min/max x and y values for all points belonging to a given cluster label.

    Args:
        labels:         Array of cluster labels, one per point.
        coordinates_x:  Array of x coordinates for all points.
        coordinates_y:  Array of y coordinates for all points.
        label:          The cluster label whose bounding coordinates are needed.

    Returns:
        A tuple (x_min, x_max, y_min, y_max) representing the bounding box edges
        of the specified cluster.
    """
    x_of_label = coordinates_x[labels == label]
    y_of_label = coordinates_y[labels == label]
    return min(x_of_label), max(x_of_label), min(y_of_label), max(y_of_label)


def get_bounding_box_for_label(labels, coordinates_x, coordinates_y, label):
    """
    Computes the top-left and bottom-right corner coordinates of the bounding
    rectangle for a given cluster label.

    Args:
        labels:         Array of cluster labels, one per point.
        coordinates_x:  Array of x coordinates for all points.
        coordinates_y:  Array of y coordinates for all points.
        label:          The cluster label to compute the bounding box for.

    Returns:
        A tuple of two (x, y) tuples: (top_left_corner, bottom_right_corner).
    """
    x_min, x_max, y_min, y_max = get_cluster_bounding_coords(labels, coordinates_x, coordinates_y, label)
    top_left = (x_min, y_min)
    bottom_right = (x_max, y_max)
    return top_left, bottom_right


def draw_bounding_box(frame, top_left, bottom_right, label):
    """
    Draws a labeled green bounding rectangle onto a frame in-place.

    Args:
        frame:        The BGR image frame to draw on (modified in-place).
        top_left:     (x, y) tuple for the top-left corner of the rectangle.
        bottom_right: (x, y) tuple for the bottom-right corner of the rectangle.
        label:        Integer label to display as text near the top-left corner.

    Returns:
        None. The frame is modified in-place.
    """
    cv.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
    cv.putText(frame, f"{label}", top_left, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def apply_roi_mask(frame, roi_corners):
    """
    Applies a rectangular region-of-interest (ROI) mask to a frame, blacking
    out everything outside the specified rectangle.

    Args:
        frame:       The BGR image frame to mask.
        roi_corners: A list of two (x, y) tuples: [top_left, bottom_right],
                     defining the visible rectangular region.

    Returns:
        A new BGR frame where only the pixels inside the ROI are retained;
        all other pixels are set to zero.
    """
    h, w = frame.shape[:2]
    
    # Create a black mask
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Fill the ROI rectangle with white
    cv.rectangle(mask, roi_corners[0], roi_corners[1], 255, thickness=-1)
    
    # Apply the mask to the frame
    return cv.bitwise_and(frame, frame, mask=mask)


def get_cluster_centroid(labels, coordinates_x, coordinates_y, label):
    """
    Computes the centroid (mean position) of all points belonging to a cluster.

    Args:
        labels:         Array of cluster labels, one per point.
        coordinates_x:  Array of x coordinates for all points.
        coordinates_y:  Array of y coordinates for all points.
        label:          The cluster label whose centroid is to be computed.

    Returns:
        A tuple (cx, cy) of integer pixel coordinates representing the centroid
        of the specified cluster.
    """
    x_of_label = coordinates_x[labels == label]
    y_of_label = coordinates_y[labels == label]
    cx = int(sum(x_of_label) / len(x_of_label))
    cy = int(sum(y_of_label) / len(x_of_label))
    return (cx, cy)


def get_frame_centroid(unique_labels, labels, coordinates_x, coordinates_y,
                       x_trajectory, y_trajectory):
    """
    Determines the best centroid estimate for the tracked ant in the current frame.

    If only one real cluster exists, its centroid is returned directly. If multiple
    clusters are present, the one whose centroid is closest to the last known position
    is selected, assuming the ant cannot teleport between frames.

    Args:
        unique_labels:   Sorted array of unique DBSCAN labels present in the frame
                         (may include -1 for noise).
        labels:          Full array of per-point cluster labels.
        coordinates_x:   Array of x coordinates for all detected points.
        coordinates_y:   Array of y coordinates for all detected points.
        x_trajectory:    List of x coordinates from all previously tracked centroids.
        y_trajectory:    List of y coordinates from all previously tracked centroids.

    Returns:
        A tuple (cx, cy) of integer pixel coordinates for the chosen centroid.
    """
    # Only one valid cluster — use it directly
    if len(unique_labels) == 1:
        return get_cluster_centroid(labels, coordinates_x, coordinates_y, unique_labels[0])

    # Two labels and the first is noise — use the only real cluster
    elif len(unique_labels) == 2 and unique_labels[0] == -1:
        return get_cluster_centroid(labels, coordinates_x, coordinates_y, unique_labels[1])

    # Multiple real clusters — pick the one closest to the last known position
    else:
        centroids_x = []
        centroids_y = []

        # Skip noise label (-1) if present
        start_index = 1 if unique_labels[0] == -1 else 0

        for i in range(start_index, len(unique_labels)):
            cx, cy = get_cluster_centroid(labels, coordinates_x, coordinates_y, unique_labels[i])
            centroids_x.append(cx)
            centroids_y.append(cy)

        # Last known position
        prev_x = x_trajectory[-1]
        prev_y = y_trajectory[-1]

        centroids_x = np.array(centroids_x)
        centroids_y = np.array(centroids_y)

        # Squared Euclidean distances from each centroid to the last known position
        distances = ((centroids_x - prev_x) ** 2 + (centroids_y - prev_y) ** 2).tolist()

        closest_index = distances.index(min(distances))
        return (int(centroids_x[closest_index]), int(centroids_y[closest_index]))


def shift_roi_mask(delta_x, delta_y, roi_corners):
    """
    Translates the ROI mask coordinates by a given displacement.

    This keeps the tracking window centered on the ant as it moves between frames.

    Args:
        delta_x:     Horizontal displacement in pixels (positive = right).
        delta_y:     Vertical displacement in pixels (positive = down).
        roi_corners: A list of two (x, y) tuples: [top_left, bottom_right].

    Returns:
        A new list of two (x, y) tuples representing the shifted ROI corners.
    """
    return [
        (roi_corners[0][0] + delta_x, roi_corners[0][1] + delta_y),
        (roi_corners[1][0] + delta_x, roi_corners[1][1] + delta_y)
    ]


def extract_background(video, n):
    """
    Estimates the static background of a video by computing the median of
    randomly sampled frames.

    Taking the median effectively suppresses moving objects (like ants), leaving
    only the background elements (like the fungus).

    Args:
        video: An OpenCV VideoCapture object for the input video.
        n:     Number of random frames to sample for the median computation.

    Returns:
        A uint8 numpy array of shape (H, W, 3) representing the estimated
        background image.
    """
    # Sample n random frame indices from the video
    frame_ids = video.get(cv.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=n)

    frames = []
    for fid in frame_ids:
        video.set(cv.CAP_PROP_POS_FRAMES, fid)
        ret, frame = video.read()
        if ret:
            frames.append(frame)

    # Pixel-wise median collapses moving objects into the background
    return np.median(frames, axis=0).astype(dtype=np.uint8)


def detect_fungus_coordinates(image):
    """
    Detects pixels that likely belong to the fungus based on color conditions.

    The fungus is identified by pixels where red exceeds both green and blue
    (warm tone), with a slight green-over-blue bias typical of the fungus color.

    Args:
        image: A BGR image (numpy array of shape H x W x 3), typically the
               estimated background frame.

    Returns:
        A tuple of:
            - coordinates: numpy array of shape (N, 2) with [x, y] coordinates
              of all detected fungus pixels.
            - mask: 2D boolean numpy array (H x W) where True marks fungus pixels.
    """
    # image[:,:,0] is blue, image[:,:,1] is green, image[:,:,2] is red
    b = image[:, :, 0].astype(np.int32)
    g = image[:, :, 1].astype(np.int32)
    r = image[:, :, 2].astype(np.int32)

    # Color conditions for fungus pixels
    cond_green      = r > (1 * g)         # Red at least equals green
    cond_blue       = r > (1 * b)          # Red at least equals blue
    cond_thresh     = r > 40               # Minimum brightness
    cond_green_blue = g > b * 1.03         # Slight green-over-blue warmth

    fungus_mask = cond_green & cond_blue & cond_thresh & cond_green_blue

    y_indices, x_indices = np.where(fungus_mask)
    return np.column_stack((x_indices, y_indices)), fungus_mask


def fill_fungus_gaps(mask):
    """
    Fills interior gaps in the fungus mask by flood-filling enclosed False regions.

    A False pixel is considered an interior gap if there are True (fungus) pixels
    in all four cardinal directions (left, right, up, down) along its row and column.
    This creates a solid filled region rather than a sparse outline.

    Args:
        mask: 2D boolean numpy array (H x W) representing detected fungus pixels.

    Returns:
        A tuple of:
            - coordinates: numpy array of shape (N, 2) with [x, y] coordinates
              of all pixels in the filled fungus region (including gaps).
            - filled_mask: 2D boolean numpy array (H x W) with interior holes filled.
    """
    not_fungus = ~mask

    # For each pixel, check if a fungus pixel exists in each direction
    fungus_to_right  = np.fliplr(np.logical_or.accumulate(np.fliplr(mask), axis=1))
    fungus_to_left   = np.logical_or.accumulate(mask, axis=1)
    fungus_below     = np.flipud(np.logical_or.accumulate(np.flipud(mask), axis=0))
    fungus_above     = np.logical_or.accumulate(mask, axis=0)

    # A gap pixel is enclosed if fungus exists in all four directions
    gap_pixels  = not_fungus & fungus_to_right & fungus_to_left & fungus_below & fungus_above
    filled_mask = mask | gap_pixels

    y_indices, x_indices = np.where(filled_mask)
    return np.column_stack((x_indices, y_indices)), filled_mask


def plot_trajectory(x_coords, y_coords, point_size, colors, frame_width, frame_height):
    """
    Plots a scatter graph of trajectory points with the y-axis flipped to match
    image coordinate conventions (origin at top-left).

    Args:
        x_coords:     List or array of x coordinates to plot.
        y_coords:     List or array of y coordinates to plot (already flipped:
                      pass (frame_height - original_y) before calling).
        point_size:   Marker size for the scatter plot.
        colors:       Color or list of colors for each point.
        frame_width:  Width of the video frame, used to set x-axis limits.
        frame_height: Height of the video frame, used to set y-axis limits.

    Returns:
        None. Displays the scatter plot inline.
    """
    plt.scatter(x_coords, y_coords, s=point_size, c=colors)
    plt.xlim(0, frame_width)
    plt.ylim(0, frame_height)


def main():
    # ── Load first frame ──────────────────────────────────────────────────────
    video = cv.VideoCapture("ant_video.mp4")
    is_valid, frame = video.read()
    rows    = frame.shape[0]
    columns = frame.shape[1]

    # ── Detect ants in first frame ────────────────────────────────────────────
    ant_coordinates = detect_ant_coordinates(frame)
    ant_labels      = individualize_points(ant_coordinates, 10, 50, size_bias=0.55)

    # Draw detection rectangles for each cluster
    coord_x = ant_coordinates[:, 0]
    coord_y = ant_coordinates[:, 1]
    bounding_boxes = []
    label_list = np.unique(ant_labels).tolist()
    label_list.remove(-1)
    frame_with_boxes = frame.copy()

    for label in label_list:
        top_left, bottom_right = get_bounding_box_for_label(ant_labels, coord_x, coord_y, label)
        bounding_boxes.append([top_left, bottom_right])
        draw_bounding_box(frame_with_boxes, top_left, bottom_right, label)

    cv.imshow("First Frame", frame_with_boxes)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # ── Select ant label to track ─────────────────────────────────────────────
    selected_label = int(input(f"Select one of the following labels to track its path: {label_list}  "))

    # Initial ROI mask around the chosen ant
    selected_bbox   = bounding_boxes[label_list.index(selected_label)]
    roi_corners     = [selected_bbox[0], selected_bbox[1]]
    initial_centroid = get_cluster_centroid(ant_labels, coord_x, coord_y, selected_label)

    x_trajectory = [initial_centroid[0]]
    y_trajectory = [initial_centroid[1]]

    detect_fungus = input("Do you want to know when the ant was over the fungus? 1. Yes, 0. No  ")

    # ── Fungus detection setup ────────────────────────────────────────────────
    on_fungus_flags = []
    fungus_filled_mask = None

    if detect_fungus == "1":
        background_image = extract_background(video, 100)

        # Detect and fill fungus region
        fungus_coords, fungus_mask        = detect_fungus_coordinates(background_image)
        fungus_labels                      = individualize_points(fungus_coords, 10, 30)
        fungus_mask                        = remove_noise(fungus_labels, fungus_coords, fungus_mask)
        _, fungus_filled_mask              = fill_fungus_gaps(fungus_mask)

        # Return to frame 1 (extract_background moved the read position)
        video.set(cv.CAP_PROP_POS_FRAMES, 1)

        # Check if starting centroid is over fungus
        is_on_fungus = fungus_filled_mask[y_trajectory[-1], x_trajectory[-1]]
        on_fungus_flags.append(is_on_fungus)

    # ── Main tracking loop ────────────────────────────────────────────────────
    frame_skip    = 1   # Process every Nth frame (1 = every frame)
    frame_counter = 1
    centroid      = initial_centroid

    while True:
        is_valid, frame = video.read()
        display_frame = frame.copy()

        if is_valid:
            # Apply ROI mask to focus on the tracked ant
            frame = apply_roi_mask(frame, roi_corners)

            if frame_counter % frame_skip == 0:
                ant_coordinates = detect_ant_coordinates(frame)

                if len(ant_coordinates) != 0:
                    # Cluster detected pixels
                    ant_labels  = individualize_points(ant_coordinates, 8, 20, use_average_filter=True)
                    coord_x     = ant_coordinates[:, 0]
                    coord_y     = ant_coordinates[:, 1]

                    # Find the centroid closest to last known position
                    unique_labels = np.unique(ant_labels)
                    centroid = get_frame_centroid(unique_labels, ant_labels, coord_x, coord_y,
                                                  x_trajectory, y_trajectory)
                    x_trajectory.append(centroid[0])
                    y_trajectory.append(centroid[1])

                    # Shift ROI to follow the ant
                    if len(x_trajectory) >= 2:
                        delta_x    = x_trajectory[-1] - x_trajectory[-2]
                        delta_y    = y_trajectory[-1] - y_trajectory[-2]
                        roi_corners = shift_roi_mask(delta_x, delta_y, roi_corners)

                    # Check if ant is over the fungus
                    if detect_fungus == "1":
                        is_on_fungus = fungus_filled_mask[y_trajectory[-1], x_trajectory[-1]]
                        on_fungus_flags.append(is_on_fungus)
                        
                        # #Graph travel detecting fungus
                        # point_colors = [(1, 0.6, 0) if v else (0, 0.5, 1) for v in on_fungus_flags]
                        # plot_trajectory(x_trajectory, rows - np.array(y_trajectory), 2, point_colors, columns, rows)
                        # plt.show()
                       
                    # #Graph travel
                    # else: 
                    #     plot_trajectory(x_trajectory, rows - np.array(y_trajectory), 2, (0, 0.5, 1), columns, rows)
                    #     plt.show()
            # Display video with centroid marker
            frame_counter += 1
            cv.circle(display_frame, centroid, 5, (0, 255, 0), 6)
            cv.imshow("Video", display_frame)
            if cv.waitKey(1) & 0xFF == ord('d'):
                break
        else:
            break

    video.release()
    cv.destroyAllWindows()

    # ── Final trajectory plot ─────────────────────────────────────────────────
    # Orange = on fungus, blue = off fungus
    if(detect_fungus=="1"):
        point_colors = [(1, 0.6, 0) if v else (0, 0.5, 1) for v in on_fungus_flags]
        plot_trajectory(x_trajectory, rows - np.array(y_trajectory), 2, point_colors, columns, rows)
    else:
        plot_trajectory(x_trajectory, rows - np.array(y_trajectory), 2, (0, 0.5, 1), columns, rows)
    plot_trajectory([x_trajectory[0]],  [rows - y_trajectory[0]],  2, ["green"], columns, rows)
    plot_trajectory([x_trajectory[-1]], [rows - y_trajectory[-1]], 2, ["red"],   columns, rows)
    plt.show()


main()
