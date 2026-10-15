# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ! This is the file for the trainer object that would be used for
# !     training the given model using the given optimizer and loss
# !     function.
# !
# !     You need to implement the train_loop and eval_loop functions.
# !     You can also use additional packages in this file.
# !     If necessary, you can also add another functions as required.
# !
# ! Make sure that the final implementation is compatible with the
# !     other classes.
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

from torch.utils.data import DataLoader
import torch

from typing import Dict
from sklearn.metrics import f1_score
import wandb


class Trainer:
    def __init__(self, model, optimizer, loss_fn, project_name, device, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.project_name = project_name
        self.device = device

        self.__dict__.update(kwargs)
        self.model = self.model.to(self.device)

    def train(self, training_data: DataLoader, validation_data: DataLoader, epochs: int, best_f1=0.0):
        with wandb.init(project=self.project_name, entity=self.entity, config=self.config) as run:
            run_best = 0.0
            for epoch in range(epochs):
                train_result = self.train_loop(training_data)
                eval_result = self.eval_loop(validation_data)
                self.log_wandb(run, epoch, train_result, eval_result)

                score = eval_result['f1_evaluator']
                run_best = max(run_best, score)
                # we're saving only when this run beats the best checkpoint across experiments
                if score > best_f1:
                    best_f1 = score
                    self.model.save('./saved_objects/model.pt')
            run.summary['best_f1_evaluator'] = run_best
        return best_f1

    def train_loop(self, data: DataLoader) -> Dict:
        self.model.train()
        total_loss = correct = count = 0
        for X, y in data:
            X, targets = X.to(self.device), y.argmax(dim=1).to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.loss_fn(logits, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(targets)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            count += len(targets)
        return {'loss': total_loss / count, 'accuracy': correct / count}

    def eval_loop(self, data: DataLoader) -> Dict:
        self.model.eval()
        total_loss = batch_f1 = count = 0
        labels, predictions = [], []
        with torch.inference_mode():
            for X, y in data:
                X, targets = X.to(self.device), y.argmax(dim=1).to(self.device)
                logits = self.model(X)
                predicted = logits.argmax(dim=1)
                total_loss += self.loss_fn(logits, targets).item() * len(targets)
                count += len(targets)
                actual_cpu, predicted_cpu = targets.cpu(), predicted.cpu()
                labels.append(actual_cpu)
                predictions.append(predicted_cpu)
                # provided evaluator needs to average macro f1 separately over batches.
                batch_f1 += f1_score(actual_cpu.numpy(), predicted_cpu.numpy(), average='macro')
        return {
            'loss': total_loss / count,
            'f1_macro': f1_score(torch.cat(labels).numpy(), torch.cat(predictions).numpy(), average='macro'),
            'f1_evaluator': batch_f1 / len(data),
        }

    def log_wandb(self, run, epoch, training_result, eval_result):
        run.log({
            'epoch': epoch + 1,
            'train/loss': training_result['loss'],
            'train/accuracy': training_result['accuracy'],
            'validation/loss': eval_result['loss'],
            'validation/f1_macro': eval_result['f1_macro'],
            'validation/f1_evaluator': eval_result['f1_evaluator'],
        })
