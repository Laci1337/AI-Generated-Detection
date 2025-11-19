import os
from typing import Any, Optional
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def collect_model_outputs(model: torch.nn.Module, device: torch.device, test_loader) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    model.eval()
    output_list: list[torch.Tensor] = []
    label_list: list[torch.Tensor] = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            output_list.append(outputs)
            label_list.append(labels.to(device))
            
    model.train()
            
    return output_list, label_list
    

def run_test(border: float, model_output_data: tuple[list[torch.Tensor], list[torch.Tensor]]) -> float:
    correct = 0
    total = 0
    
    if model_output_data is None:
        raise ValueError('No input for test.')
    
    model_output_list, model_label_list = model_output_data

    for outputs, labels in zip(model_output_list, model_label_list):
        labels = labels.float().unsqueeze(1)

        probs = torch.sigmoid(outputs)
        predicted = (probs > border).float()

        print("Labels: " + str(labels))
        print("Outputs: " + str(probs) + "\n")

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    return (correct / total)

def run_detailed_test(border: float, model_output_data: tuple[list[torch.Tensor], list[torch.Tensor]]) -> tuple[float, float, float, float]:

    if model_output_data is None:
        raise ValueError('No input for test.')

    model_output_list, model_label_list = model_output_data

    TP = 0.0
    FP = 0.0
    TN = 0.0
    FN = 0.0

    for outputs, labels in zip(model_output_list, model_label_list):
        labels = labels.float().unsqueeze(1)

        probs = torch.sigmoid(outputs)
        predicted = (probs > border).float()

        TP += ((predicted == 1) & (labels == 1)).sum().item()
        FP += ((predicted == 1) & (labels == 0)).sum().item()
        TN += ((predicted == 0) & (labels == 0)).sum().item()
        FN += ((predicted == 0) & (labels == 1)).sum().item()

    return TP, FP, TN, FN


def find_best_border(model_output_data: tuple[list[torch.Tensor], list[torch.Tensor]]) -> None:
    x_data = []
    y_data = []

    border = 0.01

    for _ in range(99):
        print(f"Inspected border: {border}")
        x_data.append(border)
        y_data.append(100.0 * run_test(border, model_output_data))
        
        border += 0.01

    plt.plot(x_data, y_data)
    plt.show()
    

def find_best_border_for_fpr(model_output_data: tuple[list[torch.Tensor], list[torch.Tensor]], target_fpr: Optional[float] = None) -> Optional[float]:
    x_data = []
    y_data = []

    border = 0.01

    for _ in range(99):
        x_data.append(border)
        _, fp, tn, _ = run_detailed_test(border, model_output_data)
        y_data.append(calculate_fpr(fp, tn))
        print(f"Inspected border: {border}, fp: {fp}, tn: {tn}")
        
        border += 0.01

    plt.plot(x_data, y_data)
    plt.show()
    
    if target_fpr is not None:    
        min_distance = abs(y_data[0] - target_fpr)
        best_border = x_data[0]
        
        for border_, fpr in zip(x_data, y_data):
            if abs(fpr - target_fpr) < min_distance:
                min_distance = abs(fpr - target_fpr)
                best_border = border_
                
        return best_border
    
    return None

def find_best_border_for_youden(model_output_data: tuple[list[torch.Tensor], list[torch.Tensor]]) -> Optional[float]:
    x_data = []
    y_data = []

    border = 0.01

    for _ in range(99):
        x_data.append(border)
        tp, fp, tn, fn = run_detailed_test(border, model_output_data)
        youden = calculate_tpr(tp, fn) - calculate_fpr(fp, tn)
        y_data.append(youden)
        print(f"Inspected border: {border}, youden: {youden}")
        
        border += 0.01

    plt.plot(x_data, y_data)
    plt.show()
    
    max_youden = max(y_data)
    best_border_idx = y_data.index(max_youden)
    
    return x_data[best_border_idx]
    
    
def calculate_tpr(tp: int, fn: int) -> float:
    return tp / (tp + fn)

def calculate_fpr(fp: int, tn: int) -> float:
    return fp / (fp + tn)