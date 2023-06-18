import cv2
import os
import sys
# Path to the input video
input_video = sys.argv[1]
video_path = input_video
demo_path = '/home/mert/pyskl/demo/demo.mp4'


def get_video_dimensions(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Read the first frame to get the dimensions
    success, frame = video.read()

    if success:
        # Get the width and height of the frame
        height, width, _ = frame.shape
        # Return the width and height values
        return width, height
    else:
        print("Failed to read the video file.")

    # Release the video object
    video.release()


# Call the function to get the video dimensions
width, height = get_video_dimensions(demo_path)

# Print the width and height values
print(f"Width: {width}, Height: {height}")

# Load the video
video = cv2.VideoCapture(video_path)

# Get the original video's frame dimensions
original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the target dimensions for resizing
target_width = width
target_height = height

# Create a VideoWriter object to save the resized video
modified_string = video_path.replace(".", "resized.")
output_path = modified_string
output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (target_width, target_height))

# Loop through the frames of the original video
while True:
    # Read the next frame
    ret, frame = video.read()

    if ret:
        # Resize the frame to the target dimensions
        resized_frame = cv2.resize(frame, (target_width, target_height))

        # Write the resized frame to the output video
        output_video.write(resized_frame)
    else:
        break

# Release the video capture and writer objects
video.release()
output_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
####################################################################################################################################################
import ast
import numpy as np

# Read the data from the text file
with open('/home/mert/pyskl/box2.txt', 'r') as file:
    content = file.read()

# Parse the data using ast.literal_eval()
data = ast.literal_eval(content)

# Convert the data to a numpy array
data_array = np.array(data)


####################################################################################################################




# Open the video file
video_path = output_path
cap = cv2.VideoCapture(video_path)

# Define the coordinates of the bounding boxes for each frame
coordinates_per_frame = data_array

# Read the first frame to get video properties
ret, frame = cap.read()
if not ret:
    raise ValueError("Failed to read video file")

# Get the video's width, height, and frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create a VideoWriter object to save the output video
modified_string = output_path.replace(".", "box.")
output_path = modified_string

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process each frame of the video
frame_index = 0
while ret:
    # Draw bounding boxes on the current frame
    if frame_index < len(coordinates_per_frame):
        coordinates = coordinates_per_frame[frame_index]
        for coord in coordinates:
            xmin, ymin, xmax, ymax, _ = coord  # Ignore the accuracy (5th parameter)
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    # Write the frame with bounding boxes to the output video
    out.write(frame)

    # Read the next frame
    ret, frame = cap.read()
    frame_index += 1

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
