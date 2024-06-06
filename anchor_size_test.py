import json
import os
from collections import defaultdict
from PIL import Image



with open('./dataset_pvc/test_dataset.json', 'r') as f:
    annotations = json.load(f)

# Loop through the annotations and calculate object sizes
object_sizes = []
for ann in annotations['annotations']:
    bbox = ann['bbox']
    width, height = bbox[2], bbox[3]
    object_sizes.append((width, height))

# Calculate average object size
avg_width = sum([size[0] for size in object_sizes]) / len(object_sizes)
avg_height = sum([size[1] for size in object_sizes]) / len(object_sizes)
min_width = min([size[0] for size in object_sizes])
min_height = min([size[1] for size in object_sizes])
max_width = max([size[0] for size in object_sizes])
max_height = max([size[1] for size in object_sizes])

print("Average object size: {} x {}".format(avg_width, avg_height))
print("Minimum object size: {} x {}".format(min_width, min_height))
print("Maximum object size: {} x {}".format(max_width, max_height))



root_dir = "./dataset_pvc/img_test"  # Replace with your root directory
image_sizes = []

# Loop through all files in the root directory and its subdirectories
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):  # Only process image files
            img_path = os.path.join(root, file)
            img = Image.open(img_path)
            width, height = img.size
            image_sizes.append((width, height))

# Calculate average image size
avg_width = sum([size[0] for size in image_sizes]) / len(image_sizes)
avg_height = sum([size[1] for size in image_sizes]) / len(image_sizes)
print("Average image size: {} x {}".format(avg_width, avg_height))


# Group annotations by class
# annotations_by_class = {}
# for ann in annotations['annotations']:
#     class_id = ann['category_id']
#     bbox = ann['bbox']
#     if class_id not in annotations_by_class:
#         annotations_by_class[class_id] = []
#     annotations_by_class[class_id].append(bbox)
#
# # Calculate average object size for each class
# for class_id, bboxes in annotations_by_class.items():
#     avg_width = sum([bbox[2] for bbox in bboxes]) / len(bboxes)
#     avg_height = sum([bbox[3] for bbox in bboxes]) / len(bboxes)
#     print("Class {}: Average object size: {} x {}".format(class_id, avg_width, avg_height))

# Group object sizes by class
class_sizes = defaultdict(list)
for ann in annotations['annotations']:
    class_id = ann['category_id']
    bbox = ann['bbox']
    width, height = bbox[2], bbox[3]
    class_sizes[class_id].append((width, height))

# Calculate average object size for each class
avg_sizes = {}
for class_id, sizes in class_sizes.items():
    avg_width = sum([size[0] for size in sizes]) / len(sizes)
    avg_height = sum([size[1] for size in sizes]) / len(sizes)
    avg_sizes[class_id] = (avg_width, avg_height)

# Sort classes by area (height * width)
sorted_classes = sorted(avg_sizes.items(), key=lambda x: x[1][0] * x[1][1], reverse=True)

# Print results
for class_id, size in sorted_classes:
    print("Class {}: {} x {} (area = {})".format(class_id, size[0], size[1], size[0] * size[1]))



# Calculate area size for each class
class_sizes = {}
for ann in annotations['annotations']:
    class_id = ann['category_id']
    bbox = ann['bbox']
    width, height = bbox[2], bbox[3]
    area = width * height
    if class_id not in class_sizes:
        class_sizes[class_id] = []
    class_sizes[class_id].append(area)

# Find the class with the maximum area size
max_class = max(class_sizes, key=lambda x: sum(class_sizes[x]))
max_area = sum(class_sizes[max_class])
max_width, max_height = 0, 0
for ann in annotations['annotations']:
    class_id = ann['category_id']
    if class_id == max_class:
        bbox = ann['bbox']
        width, height = bbox[2], bbox[3]
        if width * height > max_width * max_height:
            max_width, max_height = width, height

# Find the class with the minimum area size
min_class = min(class_sizes, key=lambda x: sum(class_sizes[x]))
min_area = sum(class_sizes[min_class])
min_width, min_height = float('inf'), float('inf')
for ann in annotations['annotations']:
    class_id = ann['category_id']
    if class_id == min_class:
        bbox = ann['bbox']
        width, height = bbox[2], bbox[3]
        if width * height < min_width * min_height:
            min_width, min_height = width, height

# Print the results
print("Class with maximum area size: {}".format(max_class))
print("Width: {}, Height: {}".format(max_width, max_height))
print("Area size: {}".format(max_area))

print("Class with minimum area size: {}".format(min_class))
print("Width: {}, Height: {}".format(min_width, min_height))
print("Area size: {}".format(min_area))