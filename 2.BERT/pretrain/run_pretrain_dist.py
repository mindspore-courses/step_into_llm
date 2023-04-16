import os
import time
import mindspore
import mindspore.dataset as ds
from mindspore import ms_function, log, mutable
from mindspore.ops import cross_entropy
from mindspore import nn, ops
from mindspore.communication import init, get_rank, get_group_size
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean

from src.api import value_and_grad
from src.amp import auto_mixed_precision
from src.bert import BertForPretraining
from src.config import BertConfig

from tqdm import tqdm

device_id = int(os.getenv('DEVICE_ID'))
# mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target=args.device_target)
mindspore.set_context(device_id=device_id) # set device_id
init()

class Forward(nn.Cell):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def construct(self, input_ids, input_mask, segment_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights, next_sentence_label):
        (prediction_scores, seq_relationship_score, _) = self.model(input_ids, input_mask, segment_ids, None, None, masked_lm_positions)
        # ic(prediction_scores.shape) # (batch_size, 128, 30522)
        # ic(masked_lm_labels.shape) # (batch_size, 20)
        masked_lm_loss = cross_entropy(prediction_scores.view(-1, prediction_scores.shape[-1]),
                                       masked_lm_ids.view(-1), masked_lm_weights.view(-1))
        next_sentence_loss = cross_entropy(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        
        return total_loss, ops.stop_gradient(masked_lm_loss), ops.stop_gradient(next_sentence_loss)

class TrainStep(nn.TrainOneStepCell):
    def __init__(self, network, optimizer):
        super().__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)
    
    def construct(self, input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights, \
                   next_sentence_label, segment_ids):
        total_loss, masked_lm_loss, next_sentence_loss = self.network(input_ids, input_mask, segment_ids, \
                              masked_lm_ids, masked_lm_positions, masked_lm_weights, next_sentence_label)
        grad_fn = self.grad(self.network, self.weights)
        grads = grad_fn(input_ids, input_mask, segment_ids, \
                              masked_lm_ids, masked_lm_positions, masked_lm_weights, next_sentence_label)
        grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return total_loss, masked_lm_loss, next_sentence_loss

def train(model, optimizer, train_dataset, epochs, jit=True, amp=False):
    """
    Train function for Bert pre-training.
    """
    # 5. Define forward and grad function.

#     def train_step(input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights, \
#                    next_sentence_label, segment_ids):
#         (total_loss, masked_lm_loss, next_sentence_loss), grads = grad_fn(input_ids, input_mask, segment_ids, \
#                               masked_lm_ids, masked_lm_positions, masked_lm_weights, next_sentence_label)
#         grads = grad_reducer(grads)
#         optimizer(grads)
#         return total_loss, masked_lm_loss, next_sentence_loss

#     if jit:
#         train_step = ms_function(train_step)
    net_forward = Forward(model)
    train_step = TrainStep(net_forward, optimizer)
    model = mindspore.Model(train_step)
    loss_cb = mindspore.LossMonitor()

    model.train(epochs, train_dataset, callbacks=[loss_cb], dataset_sink_mode=True)
    # 6. train
    # total = train_dataset.get_dataset_size()
    # for epoch in range(0, epochs):
    #     # epoch begin
    #     print(f"Epoch {epoch+1}\n-------------------------------")
    #     with tqdm(total=total) as t:
    #         t.set_description('Epoch %i' % (epoch+1))
    #         loss_total = 0
    #         cur_step_nums = 0
    #         # step begin
    #         for input_ids, input_mask, masked_lm_ids, masked_lm_positions, masked_lm_weights, \
    #                 next_sentence_label, segment_ids in train_dataset.create_tuple_iterator():
    #             # print(input_ids.shape)
    #             # s = time.time()
    #             total_loss, masked_lm_loss, next_sentence_loss = train_step(input_ids, input_mask, segment_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights, next_sentence_label)
    #             # e = time.time()
    #             # print(e - s)
    #             loss_total = loss_total + total_loss.asnumpy()
    #             cur_step_nums += 1
    #             t.set_postfix(loss=loss_total/cur_step_nums)
    #             t.update(1)
    #             # step end
    #             # break
    #     # train epoch end
    #     t.close()
    
    print("Done!")

if __name__ == '__main__':
# def test_train():
    mindspore.set_auto_parallel_context(parallel_mode=mindspore.ParallelMode.DATA_PARALLEL,
                                        gradients_mean = True)


    mindspore.set_context(enable_graph_kernel=True)
    # mindspore.set_context(mode=mindspore.GRAPH_MODE)
    # profiler = Profiler()
    # 0. Define batch size and epochs.
    batch_size = 256
    epochs = 10
    # 1. Read pre-train dataset.
    rank_id = get_rank()
    rank_size = get_group_size()
    print(rank_id, rank_size)
        # mindspore.set_context(enable_graph_kernel=True)

    
    # 1. Read pre-train dataset.
    dataset_path = 'test_data/wiki_00.mindrecord'
    train_dataset = ds.MindDataset(dataset_files=dataset_path, num_shards=rank_size, shard_id=rank_id)
    # 2. Batchify the dataset.
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    config = BertConfig('./config/bert_config_small.json')
    # dataset_path = config.dataset_mindreocrd_dir
    # train_dataset = ds.MindDataset(dataset_files=dataset_path, num_samples=2560)
    # 2. Batchify the dataset.
    total = train_dataset.get_dataset_size()
    # train_dataset = train_dataset.take(2)
    # 3. Define model.
    # config = BertConfig()
    model = BertForPretraining(config)

    model = auto_mixed_precision(model, "O1")
    # 4. Define optimizer(trick: warm up).
    # optimizer = BertAdam(model.trainable_params(), lr=5e-5, warmup=0.16, t_total=total//batch_size)
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=0.001)
    # optimizer = AdamWeightDecayForBert(model.trainable_params(), learning_rate=0.1)
    # 6. Pretrain
    # mean = _get_gradients_mean()
    # degree = _get_device_num()
    # grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    train(model, optimizer, train_dataset, epochs, jit=True)

    # profiler.analyse()