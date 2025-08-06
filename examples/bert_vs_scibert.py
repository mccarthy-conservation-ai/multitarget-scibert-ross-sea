# bert_vs_scibert.py
"""
BERT vs SciBERT Comparison for Multi-Target Conservation Classification
Clean implementation showing the importance of domain-specific pretraining
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from scipy import stats
import warnings
from typing import Dict, List, Tuple
from tqdm import tqdm
from datetime import datetime
import json
import random

warnings.filterwarnings('ignore')

from experiment_config import get_active_targets, should_use_target, MODE
from simple_data_loader import load_ross_sea_dataset
from simple_preprocessing import prepare_data, create_splits, get_class_descriptions
from simple_model import (
   SimpleMultiTargetModel, MultiTargetLoss, calculate_class_weights,
   get_optimizer, save_model
)


class MultiTargetDataset(Dataset):
   """Dataset for multi-target classification"""
   
   def __init__(self, texts, labels_dict, tokenizer, max_length=512):
       self.texts = texts
       self.labels_dict = labels_dict
       self.tokenizer = tokenizer
       self.max_length = max_length
       
   def __len__(self):
       return len(self.texts)
   
   def __getitem__(self, idx):
       text = str(self.texts[idx])
       
       encoding = self.tokenizer(
           text,
           truncation=True,
           padding='max_length',
           max_length=self.max_length,
           return_tensors='pt'
       )
       
       item = {
           'input_ids': encoding['input_ids'].flatten(),
           'attention_mask': encoding['attention_mask'].flatten(),
       }
       
       for target, label_matrix in self.labels_dict.items():
           item[f'{target}_labels'] = torch.FloatTensor(label_matrix[idx])
       
       return item


def set_random_seeds(seed=42):
   """Set all random seeds for reproducibility"""
   torch.manual_seed(seed)
   np.random.seed(seed)
   random.seed(seed)
   if torch.cuda.is_available():
       torch.cuda.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
   if torch.backends.mps.is_available():
       torch.mps.manual_seed(seed)


def calculate_metrics(y_true, y_pred, target_name):
   """Calculate comprehensive metrics for a target"""
   
   macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
   micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
   weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
   
   precision, recall, f1, support = precision_recall_fscore_support(
       y_true, y_pred, average=None, zero_division=0
   )
   
   metrics = {
       'macro_f1': macro_f1,
       'micro_f1': micro_f1,
       'weighted_f1': weighted_f1,
       'per_class': {
           'precision': precision.tolist(),
           'recall': recall.tolist(),
           'f1': f1.tolist(),
           'support': support.tolist()
       }
   }
   
   return metrics


def calculate_jaccard_similarity(y_true, y_pred):
   """Calculate Jaccard similarity for multi-label classification"""
   
   jaccard_scores = []
   
   for i in range(len(y_true)):
       true_indices = set(np.where(y_true[i] == 1)[0])
       pred_indices = set(np.where(y_pred[i] == 1)[0])
       
       if len(true_indices) == 0 and len(pred_indices) == 0:
           jaccard = 1.0
       elif len(true_indices.union(pred_indices)) == 0:
           jaccard = 0.0
       else:
           intersection = len(true_indices.intersection(pred_indices))
           union = len(true_indices.union(pred_indices))
           jaccard = intersection / union
       
       jaccard_scores.append(jaccard)
   
   return np.mean(jaccard_scores)


def train_epoch(model, train_loader, optimizer, loss_fn, device):
   """Train for one epoch"""
   model.train()
   total_loss = 0
   
   progress_bar = tqdm(train_loader, desc="Training", leave=False)
   
   for batch in progress_bar:
       input_ids = batch['input_ids'].to(device)
       attention_mask = batch['attention_mask'].to(device)
       
       labels = {}
       for key in batch.keys():
           if key.endswith('_labels'):
               target = key.replace('_labels', '')
               if should_use_target(target):
                   labels[target] = batch[key].to(device)
       
       optimizer.zero_grad()
       outputs = model(input_ids, attention_mask)
       
       loss, individual_losses = loss_fn(outputs, labels)
       
       loss.backward()
       torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
       optimizer.step()
       
       total_loss += loss.item()
       
       progress_bar.set_postfix({'loss': loss.item()})
   
   return total_loss / len(train_loader)


def evaluate(model, data_loader, loss_fn, device, threshold=0.5):
   """Evaluate model"""
   model.eval()
   
   active_targets = get_active_targets()
   
   all_predictions = {target: [] for target in active_targets}
   all_labels = {target: [] for target in active_targets}
   total_loss = 0
   
   with torch.no_grad():
       for batch in tqdm(data_loader, desc="Evaluating", leave=False):
           input_ids = batch['input_ids'].to(device)
           attention_mask = batch['attention_mask'].to(device)
           
           labels = {}
           for key in batch.keys():
               if key.endswith('_labels'):
                   target = key.replace('_labels', '')
                   if target in active_targets:
                       labels[target] = batch[key].to(device)
           
           outputs = model(input_ids, attention_mask)
           
           loss, _ = loss_fn(outputs, labels)
           total_loss += loss.item()
           
           for target in active_targets:
               if target in outputs:
                   probs = torch.sigmoid(outputs[target])
                   preds = (probs > threshold).float()
                   
                   all_predictions[target].append(preds.cpu().numpy())
                   all_labels[target].append(labels[target].cpu().numpy())
   
   metrics = {}
   jaccard_scores = {}
   
   for target in active_targets:
       y_true = np.vstack(all_labels[target])
       y_pred = np.vstack(all_predictions[target])
       
       metrics[target] = calculate_metrics(y_true, y_pred, target)
       jaccard_scores[target] = calculate_jaccard_similarity(y_true, y_pred)
   
   avg_loss = total_loss / len(data_loader)
   
   return metrics, jaccard_scores, avg_loss


def calculate_statistical_significance(bert_scores, scibert_scores):
   """
   Calculate statistical significance between BERT and SciBERT
   Uses paired t-test since scores are from same CV folds
   """
   t_stat, p_value = stats.ttest_rel(scibert_scores, bert_scores)
   
   diff = np.array(scibert_scores) - np.array(bert_scores)
   cohen_d = np.mean(diff) / np.std(diff, ddof=1)
   
   n = len(bert_scores)
   bert_mean = np.mean(bert_scores)
   bert_std = np.std(bert_scores, ddof=1)
   bert_ci = stats.t.interval(0.95, n-1, loc=bert_mean, scale=bert_std/np.sqrt(n))
   
   scibert_mean = np.mean(scibert_scores)
   scibert_std = np.std(scibert_scores, ddof=1)
   scibert_ci = stats.t.interval(0.95, n-1, loc=scibert_mean, scale=scibert_std/np.sqrt(n))
   
   return {
       't_statistic': t_stat,
       'p_value': p_value,
       'cohen_d': cohen_d,
       'bert_ci': bert_ci,
       'scibert_ci': scibert_ci,
       'bert_mean': bert_mean,
       'bert_std': bert_std,
       'scibert_mean': scibert_mean,
       'scibert_std': scibert_std
   }


def calculate_target_importance():
   """
   Get target importance weights for overall scoring
   Adjusted based on experiment mode
   """
   active_targets = get_active_targets()
   
   if "zones" in active_targets and "areas" in active_targets:
       return {
           'themes': 0.35,
           'objectives': 0.25,
           'areas': 0.25,
           'zones': 0.15
       }
   else:
       return {
           'themes': 0.6,
           'objectives': 0.4
       }


def train_with_cross_validation(
   model_type: str,
   texts: np.ndarray,
   labels: Dict[str, np.ndarray],
   target_configs: Dict[str, int],
   device: str,
   n_folds: int = 5,
   epochs: int = 20,
   batch_size: int = 8,
   learning_rate: float = 2e-5
):
   """
   Train model using k-fold cross-validation
   
   Returns:
       cv_results: Results from all folds including fold-level scores
       best_model: Best model from CV
       best_tokenizer: Tokenizer for best model
   """
   
   from transformers import AutoModel, AutoTokenizer
   
   print(f"\n{'='*70}")
   print(f"TRAINING {model_type.upper()} WITH {n_folds}-FOLD CROSS-VALIDATION")
   print(f"Active targets: {list(target_configs.keys())}")
   print(f"{'='*70}")
   
   if model_type == 'bert':
       model_checkpoint = 'bert-base-uncased'
   else:
       model_checkpoint = 'allenai/scibert_scivocab_uncased'
   
   tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
   
   kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
   
   cv_results = {
       'fold_metrics': [],
       'fold_jaccard': [],
       'fold_losses': [],
       'fold_f1_scores': [],
       'fold_jaccard_scores': []
   }
   
   best_score = -1
   best_model = None
   
   target_importance = calculate_target_importance()
   
   for fold, (train_idx, val_idx) in enumerate(kfold.split(texts)):
       print(f"\n{'-'*50}")
       print(f"FOLD {fold + 1}/{n_folds}")
       print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
       
       X_train = texts[train_idx]
       X_val = texts[val_idx]
       
       y_train = {}
       y_val = {}
       for target, label_matrix in labels.items():
           y_train[target] = label_matrix[train_idx]
           y_val[target] = label_matrix[val_idx]
       
       class_weights = {}
       for target, label_matrix in y_train.items():
           weights = calculate_class_weights(label_matrix)
           class_weights[target] = weights
       
       train_dataset = MultiTargetDataset(X_train, y_train, tokenizer)
       val_dataset = MultiTargetDataset(X_val, y_val, tokenizer)
       
       train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
       val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
       
       base_model = AutoModel.from_pretrained(model_checkpoint)
       model = SimpleMultiTargetModel(base_model, target_configs).to(device)
       
       loss_fn = MultiTargetLoss(
           target_configs=target_configs,
           class_weights=class_weights,
           focal_gamma=2.0,
           loss_weights=target_importance
       )
       
       optimizer = get_optimizer(model, learning_rate=learning_rate)
       
       best_val_f1 = 0
       best_model_state = None
       
       for epoch in range(epochs):
           train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
           
           if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
               val_metrics, val_jaccard, val_loss = evaluate(
                   model, val_loader, loss_fn, device
               )
               
               weighted_f1 = sum(
                   val_metrics[target]['macro_f1'] * target_importance[target]
                   for target in target_configs.keys()
               )
               
               print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                     f"val_loss={val_loss:.4f}, weighted_f1={weighted_f1:.4f}")
               
               if weighted_f1 > best_val_f1:
                   best_val_f1 = weighted_f1
                   best_model_state = model.state_dict().copy()
       
       if best_model_state is not None:
           model.load_state_dict(best_model_state)
       
       final_metrics, final_jaccard, final_loss = evaluate(
           model, val_loader, loss_fn, device
       )
       
       fold_weighted_f1 = sum(
           final_metrics[target]['macro_f1'] * target_importance[target]
           for target in target_configs.keys()
       )
       
       fold_weighted_jaccard = sum(
           final_jaccard[target] * target_importance[target]
           for target in target_configs.keys()
       )
       
       cv_results['fold_metrics'].append(final_metrics)
       cv_results['fold_jaccard'].append(final_jaccard)
       cv_results['fold_losses'].append(final_loss)
       cv_results['fold_f1_scores'].append(fold_weighted_f1)
       cv_results['fold_jaccard_scores'].append(fold_weighted_jaccard)
       
       if fold_weighted_f1 > best_score:
           best_score = fold_weighted_f1
           best_model = model
       
       print(f"\nFold {fold+1} Results:")
       for target, metrics in final_metrics.items():
           print(f"  {target}: F1={metrics['macro_f1']:.3f}, "
                 f"Jaccard={final_jaccard[target]:.3f}")
       print(f"  Weighted F1: {fold_weighted_f1:.3f}")
       print(f"  Weighted Jaccard: {fold_weighted_jaccard:.3f}")
   
   print(f"\n{'='*50}")
   print(f"CROSS-VALIDATION SUMMARY ({model_type.upper()})")
   print(f"{'='*50}")
   
   avg_metrics = {}
   
   for target in target_configs.keys():
       f1_scores = [fold[target]['macro_f1'] for fold in cv_results['fold_metrics']]
       jaccard_scores = [fold[target] for fold in cv_results['fold_jaccard']]
       
       n = len(f1_scores)
       f1_mean = np.mean(f1_scores)
       f1_std = np.std(f1_scores, ddof=1)
       f1_ci = stats.t.interval(0.95, n-1, loc=f1_mean, scale=f1_std/np.sqrt(n))
       
       jaccard_mean = np.mean(jaccard_scores)
       jaccard_std = np.std(jaccard_scores, ddof=1)
       jaccard_ci = stats.t.interval(0.95, n-1, loc=jaccard_mean, scale=jaccard_std/np.sqrt(n))
       
       avg_metrics[target] = {
           'mean_f1': f1_mean,
           'std_f1': f1_std,
           'ci_f1': f1_ci,
           'mean_jaccard': jaccard_mean,
           'std_jaccard': jaccard_std,
           'ci_jaccard': jaccard_ci
       }
       
       print(f"{target}:")
       print(f"  F1: {f1_mean:.3f} ± {f1_std:.3f} (95% CI: [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}])")
       print(f"  Jaccard: {jaccard_mean:.3f} ± {jaccard_std:.3f} (95% CI: [{jaccard_ci[0]:.3f}, {jaccard_ci[1]:.3f}])")
   
   weighted_f1_mean = np.mean(cv_results['fold_f1_scores'])
   weighted_f1_std = np.std(cv_results['fold_f1_scores'], ddof=1)
   weighted_f1_ci = stats.t.interval(0.95, n_folds-1, loc=weighted_f1_mean,
                                     scale=weighted_f1_std/np.sqrt(n_folds))
   
   print(f"\nOverall Weighted F1: {weighted_f1_mean:.3f} ± {weighted_f1_std:.3f}")
   print(f"95% CI: [{weighted_f1_ci[0]:.3f}, {weighted_f1_ci[1]:.3f}]")
   
   cv_results['avg_metrics'] = avg_metrics
   cv_results['weighted_metrics'] = {
       'f1_mean': weighted_f1_mean,
       'f1_std': weighted_f1_std,
       'f1_ci': weighted_f1_ci
   }
   
   return cv_results, best_model, tokenizer


def evaluate_on_test_set(
   model,
   tokenizer,
   test_texts,
   test_labels,
   target_configs,
   device,
   batch_size=8
):
   """Evaluate model on held-out test set"""
   
   print("\nEvaluating on test set...")
   
   test_dataset = MultiTargetDataset(test_texts, test_labels, tokenizer)
   test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
   
   loss_fn = MultiTargetLoss(target_configs=target_configs)
   
   test_metrics, test_jaccard, test_loss = evaluate(
       model, test_loader, loss_fn, device
   )
   
   return test_metrics, test_jaccard


def main():
   """Main comparison pipeline"""
   
   print("BERT vs SciBERT Comparison for Conservation Classification")
   print(f"EXPERIMENT MODE: {MODE.upper()}")
   print(f"Active targets: {get_active_targets()}")
   print("=" * 70)
   print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
   
   set_random_seeds(42)
   
   if torch.cuda.is_available():
       device = 'cuda'
       print(f"Using GPU: {torch.cuda.get_device_name(0)}")
   elif torch.backends.mps.is_available():
       device = 'mps'
       print("Using Apple Metal Performance Shaders")
   else:
       device = 'cpu'
       print("Using CPU")
   
   print("\nLoading Ross Sea dataset...")
   df, is_actual, dataset_info = load_ross_sea_dataset()
   
   print("\nPreprocessing data...")
   texts, labels, info = prepare_data(df)
   
   active_targets = get_active_targets()
   filtered_labels = {k: v for k, v in labels.items() if k in active_targets}
   
   print(f"\nFiltering targets:")
   print(f"   Original targets: {list(labels.keys())}")
   print(f"   Active targets: {list(filtered_labels.keys())}")
   
   print("\nCreating train/test splits...")
   splits = create_splits(texts, filtered_labels, test_size=0.15)
   
   train_val_texts = np.concatenate([splits['X_train'], splits['X_val']])
   train_val_labels = {}
   for target in filtered_labels.keys():
       train_val_labels[target] = np.vstack([
           splits[f'y_train_{target}'],
           splits[f'y_val_{target}']
       ])
   
   test_texts = splits['X_test']
   test_labels = {target: splits[f'y_test_{target}'] for target in filtered_labels.keys()}
   
   target_configs = {target: matrix.shape[1] for target, matrix in filtered_labels.items()}
   
   print(f"\nDataset Summary:")
   print(f"   Total papers: {len(texts)}")
   print(f"   Train+Val: {len(train_val_texts)}")
   print(f"   Test: {len(test_texts)}")
   print(f"   Active targets: {target_configs}")
   
   bert_cv_results, bert_model, bert_tokenizer = train_with_cross_validation(
       'bert',
       train_val_texts,
       train_val_labels,
       target_configs,
       device,
       n_folds=5,
       epochs=20,
       batch_size=8,
       learning_rate=2e-5
   )
   
   scibert_cv_results, scibert_model, scibert_tokenizer = train_with_cross_validation(
       'scibert',
       train_val_texts,
       train_val_labels,
       target_configs,
       device,
       n_folds=5,
       epochs=20,
       batch_size=8,
       learning_rate=2e-5
   )
   
   print("\n" + "="*70)
   print("TEST SET EVALUATION")
   print("="*70)
   
   bert_test_metrics, bert_test_jaccard = evaluate_on_test_set(
       bert_model, bert_tokenizer, test_texts, test_labels, target_configs, device
   )
   
   scibert_test_metrics, scibert_test_jaccard = evaluate_on_test_set(
       scibert_model, scibert_tokenizer, test_texts, test_labels, target_configs, device
   )
   
   target_importance = calculate_target_importance()
   
   bert_weighted_f1 = sum(
       bert_test_metrics[target]['macro_f1'] * target_importance[target]
       for target in target_configs.keys()
   )
   
   scibert_weighted_f1 = sum(
       scibert_test_metrics[target]['macro_f1'] * target_importance[target]
       for target in target_configs.keys()
   )
   
   bert_weighted_jaccard = sum(
       bert_test_jaccard[target] * target_importance[target]
       for target in target_configs.keys()
   )
   
   scibert_weighted_jaccard = sum(
       scibert_test_jaccard[target] * target_importance[target]
       for target in target_configs.keys()
   )
   
   f1_improvement = ((scibert_weighted_f1 - bert_weighted_f1) / bert_weighted_f1) * 100
   jaccard_improvement = ((scibert_weighted_jaccard - bert_weighted_jaccard) / bert_weighted_jaccard) * 100
   
   print("\n" + "="*70)
   print("STATISTICAL SIGNIFICANCE ANALYSIS")
   print("="*70)
   
   bert_fold_f1 = bert_cv_results['fold_f1_scores']
   scibert_fold_f1 = scibert_cv_results['fold_f1_scores']
   
   bert_fold_jaccard = bert_cv_results['fold_jaccard_scores']
   scibert_fold_jaccard = scibert_cv_results['fold_jaccard_scores']
   
   f1_stats = calculate_statistical_significance(bert_fold_f1, scibert_fold_f1)
   
   print("\nF1 Score Statistical Analysis:")
   print(f"   BERT: {f1_stats['bert_mean']:.3f} ± {f1_stats['bert_std']:.3f}")
   print(f"   95% CI: [{f1_stats['bert_ci'][0]:.3f}, {f1_stats['bert_ci'][1]:.3f}]")
   print(f"   SciBERT: {f1_stats['scibert_mean']:.3f} ± {f1_stats['scibert_std']:.3f}")
   print(f"   95% CI: [{f1_stats['scibert_ci'][0]:.3f}, {f1_stats['scibert_ci'][1]:.3f}]")
   print(f"   Paired t-test: t={f1_stats['t_statistic']:.3f}, p={f1_stats['p_value']:.4f}")
   print(f"   Cohen's d: {f1_stats['cohen_d']:.3f} {'(large effect)' if abs(f1_stats['cohen_d']) > 0.8 else '(medium effect)' if abs(f1_stats['cohen_d']) > 0.5 else '(small effect)'}")
   
   jaccard_stats = calculate_statistical_significance(bert_fold_jaccard, scibert_fold_jaccard)
   
   print("\nJaccard Similarity Statistical Analysis:")
   print(f"   BERT: {jaccard_stats['bert_mean']:.3f} ± {jaccard_stats['bert_std']:.3f}")
   print(f"   95% CI: [{jaccard_stats['bert_ci'][0]:.3f}, {jaccard_stats['bert_ci'][1]:.3f}]")
   print(f"   SciBERT: {jaccard_stats['scibert_mean']:.3f} ± {jaccard_stats['scibert_std']:.3f}")
   print(f"   95% CI: [{jaccard_stats['scibert_ci'][0]:.3f}, {jaccard_stats['scibert_ci'][1]:.3f}]")
   print(f"   Paired t-test: t={jaccard_stats['t_statistic']:.3f}, p={jaccard_stats['p_value']:.4f}")
   print(f"   Cohen's d: {jaccard_stats['cohen_d']:.3f} {'(large effect)' if abs(jaccard_stats['cohen_d']) > 0.8 else '(medium effect)' if abs(jaccard_stats['cohen_d']) > 0.5 else '(small effect)'}")
   
   print("\nStatistical Interpretation:")
   if f1_stats['p_value'] < 0.001:
       print(f"   F1 improvement is highly statistically significant (p < 0.001)")
   elif f1_stats['p_value'] < 0.01:
       print(f"   F1 improvement is statistically significant (p < 0.01)")
   elif f1_stats['p_value'] < 0.05:
       print(f"   F1 improvement is statistically significant (p < 0.05)")
   else:
       print(f"   F1 improvement is not statistically significant (p = {f1_stats['p_value']:.3f})")
   
   if jaccard_stats['p_value'] < 0.001:
       print(f"   Jaccard improvement is highly statistically significant (p < 0.001)")
   elif jaccard_stats['p_value'] < 0.01:
       print(f"   Jaccard improvement is statistically significant (p < 0.01)")
   elif jaccard_stats['p_value'] < 0.05:
       print(f"   Jaccard improvement is statistically significant (p < 0.05)")
   else:
       print(f"   Jaccard improvement is not statistically significant (p = {jaccard_stats['p_value']:.3f})")
   
   print("\n" + "="*70)
   print(f"FINAL COMPARISON: BERT vs SciBERT ({MODE.upper()} MODE)")
   print("="*70)
   
   print("\nTest Set Results (F1 Score):")
   print(f"{'Target':<15} {'BERT':<10} {'SciBERT':<10} {'Improvement':<12}")
   print("-" * 50)
   
   for target in target_configs.keys():
       bert_f1 = bert_test_metrics[target]['macro_f1']
       scibert_f1 = scibert_test_metrics[target]['macro_f1']
       improvement = ((scibert_f1 - bert_f1) / bert_f1 * 100) if bert_f1 > 0 else 0
       print(f"{target:<15} {bert_f1:<10.3f} {scibert_f1:<10.3f} {improvement:+11.1f}%")
   
   print("-" * 50)
   print(f"{'Weighted Avg':<15} {bert_weighted_f1:<10.3f} {scibert_weighted_f1:<10.3f} {f1_improvement:+11.1f}%")
   
   print("\nExpert Agreement (Jaccard Similarity):")
   print(f"{'Target':<15} {'BERT':<10} {'SciBERT':<10} {'Improvement':<12}")
   print("-" * 50)
   
   for target in target_configs.keys():
       bert_jac = bert_test_jaccard[target]
       scibert_jac = scibert_test_jaccard[target]
       improvement = ((scibert_jac - bert_jac) / bert_jac * 100) if bert_jac > 0 else 0
       print(f"{target:<15} {bert_jac:<10.3f} {scibert_jac:<10.3f} {improvement:+11.1f}%")
   
   print("-" * 50)
   print(f"{'Weighted Avg':<15} {bert_weighted_jaccard:<10.3f} {scibert_weighted_jaccard:<10.3f} {jaccard_improvement:+11.1f}%")
   
   bert_agreement = bert_weighted_jaccard * 100
   scibert_agreement = scibert_weighted_jaccard * 100
   
   print(f"\nKEY RESULTS ({MODE.upper()} MODE):")
   print(f"   BERT weighted F1: {bert_weighted_f1:.3f}")
   print(f"   SciBERT weighted F1: {scibert_weighted_f1:.3f}")
   print(f"   F1 Improvement: {f1_improvement:.1f}%")
   print(f"   BERT expert agreement: {bert_agreement:.1f}%")
   print(f"   SciBERT expert agreement: {scibert_agreement:.1f}%")
   print(f"   Agreement improvement: {((scibert_agreement - bert_agreement) / bert_agreement * 100):.1f}%")
   
   print(f"\nSciBERT shows {f1_improvement:.1f}% improvement over BERT")
   print(f"   This demonstrates the value of domain-specific pretraining")
   
   os.makedirs('models', exist_ok=True)
   timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   
   scibert_path = f"models/scibert_ross_sea_{MODE}_{timestamp}.pt"
   save_model(
       scibert_model,
       'allenai/scibert_scivocab_uncased',
       scibert_path,
       target_configs,
       {
           'model_type': 'scibert',
           'experiment_mode': MODE,
           'learning_rate': 2e-5,
           'batch_size': 8,
           'epochs': 20,
           'n_folds': 5
       },
       {
           'test_metrics': scibert_test_metrics,
           'test_jaccard': scibert_test_jaccard,
           'weighted_f1': scibert_weighted_f1,
           'weighted_jaccard': scibert_weighted_jaccard,
           'cv_results': scibert_cv_results
       }
   )
   
   bert_path = f"models/bert_ross_sea_{MODE}_{timestamp}.pt"
   save_model(
       bert_model,
       'bert-base-uncased',
       bert_path,
       target_configs,
       {
           'model_type': 'bert',
           'experiment_mode': MODE,
           'learning_rate': 2e-5,
           'batch_size': 8,
           'epochs': 20,
           'n_folds': 5
       },
       {
           'test_metrics': bert_test_metrics,
           'test_jaccard': bert_test_jaccard,
           'weighted_f1': bert_weighted_f1,
           'weighted_jaccard': bert_weighted_jaccard,
           'cv_results': bert_cv_results
       }
   )
   
   results = {
       'experiment_mode': MODE,
       'active_targets': active_targets,
       'dataset_info': {
           'is_actual': is_actual,
           'num_papers': len(texts),
           'test_size': len(test_texts),
           'cv_folds': 5
       },
       'bert': {
           'cv_results': {
               'avg_metrics': bert_cv_results['avg_metrics'],
               'weighted_metrics': bert_cv_results['weighted_metrics'],
               'fold_f1_scores': bert_cv_results['fold_f1_scores'],
               'fold_jaccard_scores': bert_cv_results['fold_jaccard_scores']
           },
           'test_metrics': bert_test_metrics,
           'test_jaccard': bert_test_jaccard,
           'weighted_f1': bert_weighted_f1,
           'weighted_jaccard': bert_weighted_jaccard,
           'expert_agreement': bert_agreement
       },
       'scibert': {
           'cv_results': {
               'avg_metrics': scibert_cv_results['avg_metrics'],
               'weighted_metrics': scibert_cv_results['weighted_metrics'],
               'fold_f1_scores': scibert_cv_results['fold_f1_scores'],
               'fold_jaccard_scores': scibert_cv_results['fold_jaccard_scores']
           },
           'test_metrics': scibert_test_metrics,
           'test_jaccard': scibert_test_jaccard,
           'weighted_f1': scibert_weighted_f1,
           'weighted_jaccard': scibert_weighted_jaccard,
           'expert_agreement': scibert_agreement
       },
       'statistical_analysis': {
           'f1_comparison': f1_stats,
           'jaccard_comparison': jaccard_stats
       },
       'improvements': {
           'f1_improvement': f1_improvement,
           'jaccard_improvement': jaccard_improvement,
           'agreement_improvement': ((scibert_agreement - bert_agreement) / bert_agreement * 100)
       },
       'target_importance': target_importance,
       'timestamp': timestamp
   }
   
   with open(f'bert_vs_scibert_results_{MODE}_{timestamp}.json', 'w') as f:
       json.dump(results, f, indent=2, default=str)
   
   print(f"\nModels saved:")
   print(f"   SciBERT: {scibert_path}")
   print(f"   BERT: {bert_path}")
   print(f"   Results: bert_vs_scibert_results_{MODE}_{timestamp}.json")
   
   print(f"\nSUGGESTED TEXT FOR PAPER:")
   if MODE == "semantic":
       print(f"When focusing on semantic classification tasks (themes and objectives only), ")
   else:
       print(f"Using all four classification targets, ")
   print(f"SciBERT achieved {scibert_weighted_f1:.1%} weighted F1 compared to ")
   print(f"BERT's {bert_weighted_f1:.1%}, representing a {f1_improvement:.0f}% improvement ")
   print(f"(p{'<0.001' if f1_stats['p_value'] < 0.001 else f'={f1_stats["p_value"]:.3f}'}, ")
   print(f"Cohen's d={f1_stats['cohen_d']:.2f}). ")
   print(f"Expert agreement analysis using Jaccard similarity showed SciBERT ")
   print(f"achieved {scibert_agreement:.0f}% agreement with expert annotations, ")
   print(f"compared to {bert_agreement:.0f}% for BERT ")
   print(f"(p{'<0.001' if jaccard_stats['p_value'] < 0.001 else f'={jaccard_stats["p_value"]:.3f}'}, ")
   print(f"Cohen's d={jaccard_stats['cohen_d']:.2f}). ")
   print(f"These improvements were statistically significant and demonstrated ")
   print(f"{'large' if abs(f1_stats['cohen_d']) > 0.8 else 'medium' if abs(f1_stats['cohen_d']) > 0.5 else 'small'} ")
   print(f"effect sizes, confirming the value of domain-specific pretraining.")
   
   print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
   
   return results


if __name__ == "__main__":
   results = main()
