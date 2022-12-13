from edited_roberta import *
from run_classifier import evaluate, load_and_cache_examples, accuracy, set_seed
import argparse
import glob
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
# from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup, AdamW,RobertaConfig,RobertaTokenizer)
# from utils import (compute_metrics, convert_examples_to_features,
#                         output_modes, processors)
import pandas as pd
from torch.nn.functional import cosine_similarity
ROOT_DIR = os.path.abspath('./train')
logger = logging.getLogger(__name__)


class CodeSearchBiencoderModel(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.query_encoder = RobertaModel(config, add_pooling_layer=False)
        self.code_encoder = RobertaModel(config, add_pooling_layer=False)
        self.loss_fn = BCEWithLogitsLoss()
        
        # This will initialize weights and apply final processing
        self.post_init()
 
    def forward(
        self,
        query_token_ids: Optional[torch.LongTensor] = None,
        code_token_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None, 
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
       
        outputs = self.query_encoder(
            query_token_ids,
            attention_mask=attention_mask,
        )

        query_emb = outputs[0][:, 0, :]
        
        outputs_code = self.code_encoder(
            code_token_ids,
            attention_mask=attention_mask,
        )

        code_emb = outputs_code[0][:, 0, :]
        cosine_sim = cosine_similarity(query_emb, code_emb)        
        loss = self.loss_fn(cosine_sim, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=cosine_sim,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class Args:
    def __init__(self):
        
        # Where to save things
        self.data_dir = os.path.join(ROOT_DIR, 'data')
        self.model_type = 'roberta'
        self.model_name_or_path = 'microsoft/codebert-base'
        self.task_name = 'codesearch'
        self.output_dir = os.path.join(ROOT_DIR, 'models')
        self.output_mode = 'codesearch'

        self.train_batch_size = 64
        self.eval_batch_size = 64
        self.gradient_accumulation_steps = 1
        self.learning_rate = 1e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 5 # NOTE: Change this to 1 if debugging so it runs faster # 5
        self.max_steps = -1
        self.warmup_steps = 0
        self.n_gpu = 1
        self.no_cuda = False

        # These are mostly configuration options for which pieces to run
        self.config_name = ""
        self.tokenizer_name = ""
        self.cache_dir = ""
        self.max_seq_length = 200
        self.do_train = True
        self.do_eval = True
        self.do_predict = False
        self.evaluate_during_training = False
        self.do_lower_case = False

        # How often we save things
        self.logging_steps = 1000
        self.save_steps = 1000
        self.eval_all_checkpoints = False
        self.overwrite_output_dir = True
        self.overwrite_cache = True
        self.seed = 42
        
        # Ignore all of these
        self.fp16 = False
        self.fp16_opt_level = 'O1'
        self.local_rank = -1
        self.server_ip = ""
        self.server_port = ""
        
        # Input and output files.
        self.train_file = "train_all_4k.csv" # CHANGE ME WHEN READY TO TRAIN!!!!!
        self.dev_file = "valid_all_700.csv"
        self.test_file = "test_all_2k.csv"
        self.pred_model_dir = os.path.join(ROOT_DIR, 'models', 'checkpoint-best')
        self.test_result_dir = os.path.join(ROOT_DIR, 'results')

args = Args()

def train(args, train_dataset, model, tokenizer, optimizer):
    """ Train the model """

    # The sampler specifies how we should access the training data, which
    # in this case is in a random order
    train_sampler = RandomSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False, sampler=train_sampler)
    
    # How many total steps we'll take
    t_total = len(train_dataloader) //  args.num_train_epochs

    # The scheduler helps decide how quickly to update the weights based on how much
    # training data we've seen. 
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = args.start_step
    tr_loss, logging_loss = 0.0, 0.0
    best_rmse = 0.0
    model.zero_grad()
    
    # Note that this "train_iterator" is just tdqm wrapper that prints out which
    # epoch we're currently in.     
    train_iterator = trange(args.start_epoch, int(args.num_train_epochs), desc="Epoch")
    
    set_seed(args) 
    
    # This tells pytorch that we're going to be changing the parameters so it needs
    # to start keeping track of stuff
    model.train()
    for idx, _ in enumerate(train_iterator):
        
        # Keep train of the training loss (how "bad" the performance is) for this epohch
        tr_loss = 0.0
        
        # For one epoch, loop over all the data, one batch at a time
        for step, batch in tqdm(enumerate(train_dataloader)):

            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'query_token_ids': batch[0],
                      'code_token_ids': batch[1],
                      'labels': batch[3]}
            
            ouputs = model(**inputs)
            loss = ouputs[0]        
            
            # Do the back propagration to figure out which parameters to change.
            # It's that easy!
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            
            # Update the parameters of our model based on the gradient and whatever
            # else the optimizer is keeping track of
            optimizer.step() 
            scheduler.step()  
            
            # This sets the gradient to zero before doing next update so we don't
            # accidentally update the model based on the last batch's performance
            model.zero_grad() 
            global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        # Once we finish an epoch, evaluate the model on the development data and see
        # how well it does. We'll use this information to decide which version of
        # the parameters to use.
        results = evaluate(args, model, tokenizer, checkpoint=str(args.start_epoch + idx))

        # 
        # Save the model and if we've already saved it, overwrite that saved model with 
        # the newly-trained parameters
        #
        last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
        if not os.path.exists(last_output_dir):
            os.makedirs(last_output_dir)
        model_to_save = model.module if hasattr(model,
                                                'module') else model 
        model_to_save.save_pretrained(last_output_dir)
        logger.info("Saving model checkpoint to %s", last_output_dir)
        idx_file = os.path.join(last_output_dir, 'idx_file.txt')
        with open(idx_file, 'w', encoding='utf-8') as idxf:
            idxf.write(str(args.start_epoch + idx) + '\n')

        torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

        step_file = os.path.join(last_output_dir, 'step_file.txt')
        with open(step_file, 'w', encoding='utf-8') as stepf:
            stepf.write(str(global_step) + '\n')

        # Optional part 1 goes here
        # save checkpoint for each epoch
        epoch_checkpoint_folder = 'checkpoint-epoch' + str(args.start_epoch + idx)
        epoch_output_dir = os.path.join(args.output_dir, epoch_checkpoint_folder)
        if not os.path.exists(epoch_output_dir):
            os.makedirs(epoch_output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model 
        model_to_save.save_pretrained(epoch_output_dir)
        logger.info("Saving model checkpoint to %s", epoch_output_dir)
        idx_file = os.path.join(epoch_output_dir, 'idx_file.txt')
        with open(idx_file, 'w', encoding='utf-8') as idxf:
            idxf.write(str(args.start_epoch + idx) + '\n')

        torch.save(optimizer.state_dict(), os.path.join(epoch_output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(epoch_output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", epoch_output_dir)

        step_file = os.path.join(epoch_output_dir, 'step_file.txt')
        with open(step_file, 'w', encoding='utf-8') as stepf:
            stepf.write(str(global_step) + '\n')

        #
        # If this model is better (on the training data) than the models from any of the 
        # past checkpoints, then keep a separate record of that too
        #
        if (results['rmse'] > best_rmse):
            best_rmse = results['rmse']
            output_dir = os.path.join(args.output_dir, 'checkpoint-best')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_{}.bin'.format(idx)))
            logger.info("Saving model checkpoint to %s", output_dir)

            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step

# Setup CUDA so we can run on the GPU
device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.device = device

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

# Set seed
set_seed(args)

# This code will help us if we restart training and want to pick back up where we left off
args.start_epoch = 0
args.start_step = 0
checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
    args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
    args.config_name = os.path.join(checkpoint_last, 'config.json')
    idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
    with open(idx_file, encoding='utf-8') as idxf:
        args.start_epoch = int(idxf.readlines()[0].strip()) + 1

    step_file = os.path.join(checkpoint_last, 'step_file.txt')
    if os.path.exists(step_file):
        with open(step_file, encoding='utf-8') as stepf:
            args.start_step = int(stepf.readlines()[0].strip())
    logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))


# We set num_labels = 1 because this is a regression class
# (compared to a classification task with many class labels)
num_labels = 1
config = RobertaConfig.from_pretrained('microsoft/codebert-base',
                                      num_labels=num_labels, finetuning_task=args.task_name)
# We'll treat relevance as a regression problem
config.problem_type = 'regression'
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Now comes the time to define our model. Let's specify the model class (which we'll need later)
model_class = CodeSearchBiencoderModel
# And we'll instantiate the model itself.
model = CodeSearchBiencoderModel(config)

# TODO: Initialize each of the coders using the "from_pretrained" method and
# specifying the pretrained model you want. Here, we'll use the CodeBERT model, 
# which is hosted on Huggingface https://huggingface.co/microsoft/codebert-base
# You should pass in the full name of the pretrained model (which includes the "/").
# Note that this code is going to look the same for both encoders and may 
# seem kind of easy to do but we want you to see how to do it yourself. :) 
model.query_encoder = AutoModel.from_pretrained("microsoft/codebert-base")
model.code_encoder = AutoModel.from_pretrained("microsoft/codebert-base")

# This will move the model's parameters onto the GPU so it runs fast
model.to(args.device)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': args.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, no_deprecation_warning=True)

# If we're restarting, load the optimizer's state at the last time step
optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
if os.path.exists(optimizer_last):
    optimizer.load_state_dict(torch.load(optimizer_last))


logger.info("Training/evaluation parameters %s", args)

# Training
# Load in the training dataset. Here, we've handled most of the data preprocessing for you
train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ttype='train')

# Call the training function that we defined above
global_step, tr_loss = train(args, train_dataset, model, tokenizer, optimizer)
logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

# Create output directory if needed
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Save the trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
logger.info("Saving model checkpoint to %s", args.output_dir)
model_to_save = model.module if hasattr(model, 'module') else model  
model_to_save.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

# Good practice: save your training arguments together with the trained model
torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

# Load a trained model and vocabulary that you have fine-tuned
model = AutoModel.from_pretrained(args.output_dir)
tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
model.to(args.device)

# Evaluate the best model on the test data
checkpoint = args.output_dir

logger.info("Evaluate the following checkpoint: %s", checkpoint)

print(checkpoint)
global_step = ""
model = model_class.from_pretrained(checkpoint)
model.to(args.device)
result = evaluate(args, model, tokenizer, checkpoint=checkpoint, prefix=global_step)
result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
print(result)

# Optional part 3 goes here
# evaluate for each epoch
models_list = [None]*args.num_train_epochs
results_list = [None]*args.num_train_epochs
for idx in range(0, args.num_train_epochs):
    epoch_checkpoint_folder = 'checkpoint-epoch' + str(idx)
    checkpoint = os.path.join(args.output_dir, epoch_checkpoint_folder)
    print(checkpoint)
    global_step = ""
    model = model_class.from_pretrained(checkpoint)
    model.to(args.device)
    models_list[idx] = model
    result = evaluate(args, model, tokenizer, checkpoint=checkpoint, prefix=global_step)
    result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    print(result)
    results_list[idx] = result

# plot rmse scores for each epoch
# import matplotlib.pyplot as plt
# x = list(range(args.num_train_epochs))
# y = [dict['f1_'] for dict in results_list]
# plt.plot(x, y)
# plt.ylabel('F1 scores')
# plt.xlabel('Epoch')
# plt.xticks(x)
# plt.show()


model.eval()
eval_dataset, instances = load_and_cache_examples(args, "codesearch", tokenizer, ttype='test')
eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

# This data structures will have our predictions and we'll fill them as we process each batch
relevance_predictions = np.array([])

for batch in tqdm(eval_dataloader, desc="Evaluating"):
    batch = tuple(t.to(args.device) for t in batch)

    with torch.no_grad():
        
        # Prepare the inputs
        inputs = {'query_token_ids': batch[0],
                  'code_token_ids': batch[1],
                  'labels': batch[3]}
       
        # Note that this is a list of outputs, which includes the cosine
        # similarity, among other stuff
        outputs = model(**inputs)

    _, cosine_sim = outputs[:2] 
    cosine_sims = cosine_sim.cpu().numpy()

    relevance_predictions = np.append(relevance_predictions, cosine_sims, axis=0)

if not os.path.exists(args.test_result_dir):
    os.makedirs(args.test_result_dir)

output_test_file = os.path.join(args.test_result_dir, 'relevance-scores.csv')

with open(output_test_file, "w") as outf:
    logger.info("***** Writing relevance predictions *****")
    all_logits = relevance_predictions.tolist()
    for score in all_logits:
        outf.write(score)

# Optional part 3 goes here
# you may find the list of files for each epoch's trained model's relevance predictions under folder results
# the files are name in the form of epochXrelavance_scores.csv, where X is the number of epoch from 0 to 4
for idx in range(0, args.num_train_epochs):
    model = models_list[idx]
    model.eval()
    relevance_predictions = np.array([])
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            # Prepare the inputs
            inputs = {'query_token_ids': batch[0],
                      'code_token_ids': batch[1],
                      'labels': batch[3]}
            outputs = model(**inputs)
        _, cosine_sim = outputs[:2] 
        cosine_sims = cosine_sim.cpu().numpy()

        # Add these similarities to our current similarities
        relevance_predictions = np.append(relevance_predictions, cosine_sims, axis=0)
    
    if not os.path.exists(args.test_result_dir):
        os.makedirs(args.test_result_dir)
    file_name = 'epoch' + str(idx) + 'relevance-scores.csv' # name each file here
    output_test_file = os.path.join(args.test_result_dir, file_name)

    with open(output_test_file, "w") as outf:
        logger.info("***** Writing relevance predictions *****")
        all_logits = relevance_predictions.tolist()
        for score in all_logits:
            outf.write(score)