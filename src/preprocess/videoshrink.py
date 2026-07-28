import cv2
import argparse
import sys

def add_letterbox_to_video(input_video, output_video, target_width=640, target_height=400, fill=0.85):
    """
    Add black bars to video to achieve target aspect ratio.

    Args:
        input_video: Path to input video file
        output_video: Path to output video file
        target_width: Target frame width (default: 640 for RITnet)
        target_height: Target frame height (default: 400 for RITnet)
        fill: Fill ratio - percentage of frame the content should occupy (0-1)
    """

    # Open input video
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {input_video}")
        sys.exit(1)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Input video: {original_width}x{original_height} @ {fps:.2f} FPS ({frame_count} frames)")

    # Calculate output dimensions with fill ratio
    output_w = target_width
    output_h = target_height

    # Calculate scaled frame dimensions (content should occupy 'fill' of output)
    scaled_w = int(output_w * fill)
    scaled_h = int(output_h * fill)

    # Maintain aspect ratio of original frame
    aspect_ratio = original_width / original_height
    target_aspect = output_w / output_h

    if aspect_ratio > target_aspect:
        # Original is wider - scale by width
        scaled_h = int(scaled_w / aspect_ratio)
    else:
        # Original is taller - scale by height
        scaled_w = int(scaled_h * aspect_ratio)

    # Calculate padding (center the content)
    pad_top = (output_h - scaled_h) // 2
    pad_bottom = output_h - scaled_h - pad_top
    pad_left = (output_w - scaled_w) // 2
    pad_right = output_w - scaled_w - pad_left

    print(f"Output video: {output_w}x{output_h}")
    print(f"Scaled content: {scaled_w}x{scaled_h} (fill={fill*100:.0f}%)")
    print(f"Padding (T/B/L/R): {pad_top}/{pad_bottom}/{pad_left}/{pad_right}")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use MJPEG codec for broad compatibility
    out = cv2.VideoWriter(output_video, fourcc, fps, (output_w, output_h))

    if not out.isOpened():
        print(f"ERROR: Cannot create output video: {output_video}")
        sys.exit(1)

    # Process frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to scaled dimensions
        resized = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        # Create output frame with black bars
        output_frame = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)  # Black
        )

        # Write frame
        out.write(output_frame)

        frame_idx += 1
        if frame_idx % 30 == 0:
            progress = (frame_idx / frame_count) * 100
            print(f"Progress: {frame_idx}/{frame_count} ({progress:.1f}%)")

    # Release resources
    cap.release()
    out.release()

    print(f"✓ Done! Output saved to: {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add black bars to video for RITnet inference (640x400 with OpenEDS domain matching)"
    )
    parser.add_argument("input", help="Input video file")
    parser.add_argument("output", help="Output video file")
    parser.add_argument("--width", type=int, default=640, help="Target frame width (default: 640)")
    parser.add_argument("--height", type=int, default=400, help="Target frame height (default: 400)")
    parser.add_argument("--fill", type=float, default=0.85, help="Fill ratio (0-1, default: 0.85)")

    args = parser.parse_args()

    if not 0 < args.fill <= 1:
        print("ERROR: --fill must be between 0 and 1")
        sys.exit(1)

    add_letterbox_to_video(args.input, args.output, args.width, args.height, args.fill)