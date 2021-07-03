# Active Learning in Object Detection
## Overview
Using prediction results, selecting images that are most informative. These images tend to have inconsistent confidence score among different classes.

Four methods are supported in this program:
* Max Entropy (single model)
* Sum Entropy (single model)
* Mutual Information (multi models)
* KL divergence on score distribution (multi models)

For detailed description of the method, please read this [documentation](https://www.notion.so/Documentations-e448843a6f604c22848958206e81ca53).

## Usage
Command line arguments include:
- `--method, -m`: Active learning method. Choose one: sum_entropy, max_entropy, mutual_information or KL_score
-  `--pred_path, -p`: Paths to prediction results. You can pass in predictions from one or multiple models, separating by commas. Eg: ./pred1.json,./pred2.json
-  `--gt_path, -g`: Path to ground truth annotations in coco format
-  `--image_root, -i`: Directory containing images
-  `--visualize, -v`: Flag to visualize selected image. Default = True
-  `--num_query, -n`: Number of selected images

Example python command to run the program with max entropy
```python
python main.py -m max_entropy -p predictions.json -n 20 -i ./images -g groundtruths.json
```
