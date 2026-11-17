import logging
import os.path
import sys
from sklearn import metrics
import json


# [_, input_dir, output_dir] = sys.argv

# truth_dir = os.path.join(input_dir, 'ref')
# submit_dir = os.path.join(input_dir, 'res')

# if not os.path.isdir(submit_dir):
#     logging.warning("prediction not found for {0}".format(submit_dir))
#     # exit(0)

# if len(os.listdir(submit_dir)):
#     logging.warning("more than one prediction file was found for {0}".format(submit_dir))
#     # exit(0)


def read_json(input_path):
    with open(input_path, encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data

ground_truth = read_json(r"matonto_test_pairs.json")
submit_prediction = read_json(r'pred_edges_gemma3_4b')
file_name = "score_gemma3_4b.txt"

if len(ground_truth) != len(submit_prediction) :
    logging.warning("The size of prediction doesn't match the ground truth")
    # exit(0)


def calculate_precision_recall_f1(ground_truth, predictions):
    total_correct = 0
    total_predicted = len(predictions)
    total_ground_truth = len(ground_truth)
    
    # Convert ground truth and predictions to sets of tuples for easy comparison
    ground_truth_set = {(item["parent"].lower(), item["child"].lower()) for item in ground_truth}
    predictions_set = {(item["parent"].lower(), item["child"].lower()) for item in predictions}
    
    # Calculate correctly predicted pairs (intersection of sets)
    correct_predictions = ground_truth_set.intersection(predictions_set)
    total_correct = len(correct_predictions)
    
    # Calculate precision, recall, and F1-score
    precision = total_correct / total_predicted if total_predicted > 0 else 0
    recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0
    
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    
    return precision, recall, f1_score

precision, recall, f1_score = calculate_precision_recall_f1(ground_truth, submit_prediction)

output_dir = r'/home/gplsi/mikel/scripts4/llms4ol_2025_iris/LLMs4OL_2025/LLMs4OL/src/TaskC/matonto/'

with open(os.path.join(output_dir, file_name), "w") as scores_file:
    scores_file.write('{0}: {1}\n'.format("Precision", precision))
    scores_file.write('{0}: {1}\n'.format("Recall", recall))
    scores_file.write('{0}: {1}\n'.format("F1", f1_score))
