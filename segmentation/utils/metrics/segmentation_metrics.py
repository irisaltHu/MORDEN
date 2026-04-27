import torch


class SegmentationMetrics:
    def __init__(self, num_class, ignore=-1):
        self.num_class = num_class
        self.ignore = ignore

    def __process(self, output, label):
        """
        unify the outputs and labels from model and dataloader
        :param output: requires torch.Tensor. outputs of a single data from model
        :param label: requires torch.Tensor. labels of a single data from dataloader
        :return: returns 2 torch.Tensor with shape torch.Size([x]). processed outputs and labels for computing metrics
        """
        output = output.detach()
        if self.num_class == 2:
            return torch.round(output).cpu().view(-1), label.cpu().view(-1)
        return torch.max(output, dim=3).indices.cpu().view(-1), label.cpu().view(-1)

    def iou(self, outputs, labels):
        """
        compute iou of each class and the mean iou
        :param outputs: requires torch.Tensor. outputs from model
        :param labels: requires torch.Tensor. labels from dataloader
        :return: returns dict. includes iou of each class and mean iou
        """
        iou_dict = {}
        batch_size = len(outputs)

        for cls in range(0, self.num_class):
            if self.ignore == cls:
                continue

            sum_iou = 0.0
            for i in range(batch_size):
                output, label = self.__process(outputs[i], labels[i])

                output_mask = output == cls
                label_mask = label == cls

                intersection = output_mask[label_mask].sum().item()
                union = output_mask.sum().item() + label_mask.sum().item() - intersection
                sum_iou += intersection / union

            iou_dict[cls] = sum_iou / batch_size

        iou_dict['mean'] = sum(iou_dict.values()) / len(iou_dict)

        return iou_dict

    def pixel_metrics(self, outputs, labels):
        """
        compute pixel metrics of each class and mean metrics
        :param outputs: requires torch.Tensor. outputs from model
        :param labels: requires torch.Tensor. labels from dataloader
        :return: returns dict. includes accuracy, precision, recall, f1-score and iou.
                The dict goes with struct like:
                {
                    precision:
                    {
                        0: xxx, 1:xxx, ···, mean:xxx
                    },
                    recall:
                    {
                        0: xxx, 1:xxx, ···, mean:xxx
                    },
                    f1-score:
                    {
                        0: xxx, 1:xxx, ···, mean:xxx
                    },
                    accuracy:
                    {
                        xxx
                    },
                    iou:
                    {
                        0: xxx, 1:xxx, ···, mean:xxx
                    }
                }
        """
        metrics_dict = {}
        precision_dict = {}
        recall_dict = {}
        f1_score_dict = {}
        iou_dict = {}
        batch_size = len(outputs)
        accuracies = [0.0 for _ in range(batch_size)]

        for cls in range(self.num_class):
            if self.ignore == cls:
                continue

            sum_precision = 0.0
            sum_recall = 0.0
            sum_f1_score = 0.0
            sum_iou = 0.0
            num_empty = 0
            for i in range(batch_size):
                try:
                    output, label = self.__process(outputs[i], labels[i])

                    output_mask = output == cls
                    label_mask = label == cls
                    output_sum = output_mask.sum().item()
                    label_sum = label_mask.sum().item()
                    if label_sum == 0:
                        if output_sum == 0:
                            sum_precision += 1.0
                            sum_recall += 1.0
                            sum_f1_score += 1.0
                            sum_iou += 1.0
                        else:
                            sum_precision += 0.0
                            sum_recall += 0.0
                            sum_f1_score += 0.0
                            sum_iou += 0.0
                        continue


                    total = output.shape[0] - (label == self.ignore).sum().item()

                    intersection = tp = output_mask[label_mask].sum().item()
                    union = output_sum + label_sum - intersection
                    fp = output_sum - tp
                    fn = label_sum - tp

                    accuracies[i] += tp / total
                    precision = tp / (tp + fp) if tp + fp != 0 else 0
                    recall = tp / (tp + fn) if tp + fn != 0 else 0
                    sum_precision += precision
                    sum_recall += recall
                    sum_f1_score += 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
                    sum_iou += intersection / union
                except Exception as e:
                    num_empty += 1

            precision_dict[cls] = sum_precision / (batch_size - num_empty)
            recall_dict[cls] = sum_recall / (batch_size - num_empty)
            f1_score_dict[cls] = sum_f1_score / (batch_size - num_empty)
            iou_dict[cls] = sum_iou / (batch_size - num_empty)

        cnt = self.num_class if self.ignore == -1 else self.num_class - 1
        precision_dict['mean'] = sum(precision_dict.values()) / cnt
        recall_dict['mean'] = sum(recall_dict.values()) / cnt
        f1_score_dict['mean'] = sum(f1_score_dict.values()) / cnt
        iou_dict['mean'] = sum(iou_dict.values()) / cnt
        metrics_dict['precision'] = precision_dict
        metrics_dict['recall'] = recall_dict
        metrics_dict['f1-score'] = f1_score_dict
        metrics_dict['accuracy'] = sum(accuracies) / batch_size
        metrics_dict['iou'] = iou_dict

        return metrics_dict
