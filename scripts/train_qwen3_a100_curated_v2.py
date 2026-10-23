"""Qwen3-14B curated-data QLoRA training; A100 40 GB; persistent resumable checkpoints.

Run in a fresh Python process after installing the pinned dependencies.
Full articles are retained. Loss is computed only on summary tokens.
This run starts a new adapter from the pinned base, never the pilot adapter.
"""
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('HF_XET_CHUNK_CACHE_SIZE_BYTES', '0')
import gc
import hashlib
import inspect
import json
import math
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path('/content/drive/MyDrive/Text_Summarization_Project/qwen3_14b_curated_v2_1200')
DATA = ROOT / 'training_data.json'
RUN = ROOT / 'training_run'
MODEL = 'Qwen/Qwen3-14B'
REVISION = '40c069824f4251a91eefaf281ebe4c544efd3e18'
INSTRUCTION = (
    'Summarize the article in English in one or two complete sentences, using at most 40 words. '
    'State the main event and its most important outcome or context. Use only information '
    'supported by the article. Preserve uncertainty, attribution, names, numbers and who did '
    'what to whom. Do not invent missing details. Write a concise, natural paraphrase without '
    'commentary, a heading, reasoning, or a list.'
)
SETTINGS = dict(epochs=3, learning_rate=5e-5, lora_rank=8, lora_alpha=16,
                lora_dropout=0.05, accumulation=8, save_steps=20,
                max_sequence_tokens=8192, seed=42)
PINNED = {'transformers': '5.18.0', 'peft': '0.21.1', 'accelerate': '1.15.0',
          'bitsandbytes': '0.50.2', 'datasets': '4.8.5'}


def write_json(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(row, tokenizer):
    prompt = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': INSTRUCTION},
         {'role': 'user', 'content': 'ARTICLE:\n' + row['article']}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion = tokenizer.encode(row['reviewed_summary'], add_special_tokens=False)
    completion.append(tokenizer.eos_token_id)
    ids = prompt_ids + completion
    if not prompt_ids or not completion:
        raise ValueError(f"Empty token sequence: {row['id']}")
    if len(ids) > SETTINGS['max_sequence_tokens']:
        raise ValueError(f"Article {row['id']} has {len(ids)} tokens; none was truncated. "
                         'No model training was started. The length policy needs adjustment.')
    return {'input_ids': ids, 'attention_mask': [1] * len(ids),
            'labels': [-100] * len(prompt_ids) + completion}


def main():
    for name, expected in PINNED.items():
        installed = version(name)
        if installed != expected:
            raise RuntimeError(f'{name}: expected {expected}, found {installed}. '
                               'Run the supplied pip cell, then run this file in a new Python process.')
    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainerCallback, TrainingArguments, set_seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A BF16-capable CUDA GPU is required.')
    if torch.cuda.mem_get_info()[0] < 32 * 2**30:
        raise RuntimeError('At least 32 GiB free GPU memory is required for this configuration.')
    if not DATA.exists():
        raise FileNotFoundError(f'Upload the prepared training_data.json first: {DATA}')
    data = json.loads(DATA.read_text(encoding='utf-8'))
    if (len(data['train']), len(data['validation'])) != (924, 231):
        raise ValueError('Expected the frozen 924/231 split; data was not modified.')
    combined = data['train'] + data['validation']
    ids = [str(r['id']) for r in combined]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate IDs in the prepared split.')
    for row in combined:
        if not isinstance(row.get('article'), str) or not row['article'].strip():
            raise ValueError(f"Missing source article: {row['id']}")
        summary = row.get('reviewed_summary')
        if not isinstance(summary, str) or not 0 < len(summary.split()) <= 40:
            raise ValueError(f"Invalid reviewed target: {row['id']}")
    RUN.mkdir(parents=True, exist_ok=True)
    manifest = dict(model=MODEL, base_revision=REVISION, settings=SETTINGS,
                    instruction=INSTRUCTION, packages=PINNED,
                    data_sha256=sha(DATA), train_ids=[r['id'] for r in data['train']],
                    validation_ids=[r['id'] for r in data['validation']],
                    method='New NF4 QLoRA adapter; BF16 computation; summary-token loss',
                    selection='Best full-validation teacher-forced loss across epochs; not a factuality certificate',
                    test_used=False)
    manifest_file = RUN / 'run_manifest.json'
    if manifest_file.exists():
        if json.loads(manifest_file.read_text()) != manifest:
            raise RuntimeError('Existing run has different data/settings. It was not overwritten.')
    else:
        write_json(manifest_file, manifest)
    if (RUN / 'training_complete.json').exists():
        print('This training run is already complete. No training repeated.')
        print('Best adapter:', RUN / 'best_adapter')
        return
    required = ['adapter_config.json', 'adapter_model.safetensors', 'trainer_state.json',
                'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'checkpoint_verified.json']
    checkpoints = sorted(RUN.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
    valid = [p for p in checkpoints if all((p / n).is_file() for n in required)]
    resume = valid[-1] if valid else None
    if checkpoints and resume is None:
        raise RuntimeError('Only incomplete checkpoints were found; no checkpoint was deleted or loaded.')
    signature = inspect.signature(TrainingArguments).parameters
    epoch_name = 'num_train_epochs' if 'num_train_epochs' in signature else 'num_epochs'
    options = dict(output_dir=str(RUN), per_device_train_batch_size=1, per_device_eval_batch_size=1,
                   gradient_accumulation_steps=SETTINGS['accumulation'],
                   learning_rate=SETTINGS['learning_rate'], weight_decay=0.01,
                   lr_scheduler_type='cosine', warmup_steps=math.ceil(116 * SETTINGS['epochs'] * 0.1),
                   bf16=True, fp16=False, optim='adamw_torch', max_grad_norm=1.0,
                   gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                   eval_strategy='epoch', prediction_loss_only=True,
                   save_strategy='steps', save_steps=SETTINGS['save_steps'], save_total_limit=2,
                   load_best_model_at_end=False, logging_steps=5, logging_first_step=True,
                   report_to='none', seed=42, data_seed=42, dataloader_num_workers=0,
                   remove_unused_columns=False, label_names=['labels'])
    options[epoch_name] = SETTINGS['epochs']
    missing = set(options) - set(signature)
    if missing:
        raise RuntimeError(f'TrainingArguments is incompatible: {sorted(missing)}')
    source = dict(revision=REVISION, trust_remote_code=False)
    set_seed(SETTINGS['seed'])
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Loading tokenizer and checking every full article before model loading.', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, **source)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenized = {s: [encode(r, tokenizer) for r in data[s]] for s in ('train', 'validation')}
    longest = max(tokenized['train'] + tokenized['validation'], key=lambda r: len(r['input_ids']))
    print('Train: 924 | Validation: 231 | Longest complete sequence:', len(longest['input_ids']), flush=True)
    print('No article truncation. Fresh base adapter. Resume checkpoint:', resume, flush=True)
    if shutil.disk_usage('/content').free < 35 * 2**30:
        # A completed local model cache may already occupy disk; let HF reuse it without new weights.
        source['local_files_only'] = True
        print('Low local disk: restricting model loading to existing cached files.', flush=True)
    print('Loading pinned model with 4-bit NF4 weights and BF16 computation.', flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, **source, dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                              bnb_4bit_use_double_quant=True,
                                              bnb_4bit_compute_dtype=torch.bfloat16),
        device_map={'': 0}, attn_implementation='sdpa', use_safetensors=True,
    )
    if 'logits_to_keep' not in inspect.signature(base.forward).parameters:
        raise RuntimeError('The model forward API lacks summary-only logits support.')
    base.config.use_cache = False
    base.config.pad_token_id = tokenizer.pad_token_id
    base.generation_config.pad_token_id = tokenizer.pad_token_id
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False})
    # The k-bit helper upcasts frozen non-quantized layers. Keep the two large frozen
    # vocabulary matrices in BF16 to avoid the memory issue observed in the earlier pilot.
    for layer in (base.get_input_embeddings(), base.get_output_embeddings()):
        if any(p.requires_grad for p in layer.parameters()):
            raise RuntimeError('Vocabulary matrices must remain frozen in this run.')
        layer.to(dtype=torch.bfloat16)
    model = get_peft_model(base, LoraConfig(
        task_type='CAUSAL_LM', r=SETTINGS['lora_rank'], lora_alpha=SETTINGS['lora_alpha'],
        lora_dropout=SETTINGS['lora_dropout'], target_modules='all-linear', bias='none'))
    model.print_trainable_parameters()

    def collate(items):
        if len(items) != 1:
            raise ValueError('This memory-bounded loss implementation requires microbatch size one.')
        return {k: torch.tensor([items[0][k]], dtype=torch.long)
                for k in ('input_ids', 'attention_mask', 'labels')}

    def summary_loss(current_model, batch):
        labels = batch['labels']
        if labels.shape[0] != 1:
            raise ValueError('Expected a single sequence.')
        first = int(torch.nonzero(labels[0] != -100, as_tuple=False)[0, 0].item())
        keep = labels.shape[1] - first + 1
        if first < 1:
            raise ValueError('Missing prompt before completion.')
        outputs = current_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                use_cache=False, logits_to_keep=keep)
        # The logit at the final prompt position predicts the first summary token.
        # Omit the final EOS-position logit; all article-token logits are never materialised.
        logits = outputs.logits[:, :-1, :].contiguous().float()
        targets = labels[:, first:].contiguous()
        if logits.shape[:2] != targets.shape:
            raise ValueError('Summary label/logit alignment failed.')
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return loss, outputs

    class SummaryTrainer(Trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, current_model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            loss, outputs = summary_loss(current_model, inputs)
            return (loss, outputs) if return_outputs else loss

    # A genuine full-length backward and optimizer step, reverted before training.
    print('Checking the longest full sequence, gradients and optimizer memory.', flush=True)
    trainable = [(p, p.detach().cpu().clone()) for p in model.parameters() if p.requires_grad]
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state())
    model.train()
    probe_optimizer = torch.optim.AdamW([p for p, _ in trainable], lr=SETTINGS['learning_rate'])
    probe_batch = {k: v.to('cuda') for k, v in collate([longest]).items()}
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        probe_loss, probe_outputs = summary_loss(model, probe_batch)
    if not torch.isfinite(probe_loss):
        raise RuntimeError('Nonfinite probe loss; no training was started.')
    probe_loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p, _ in trainable], 1.0)
    if not torch.isfinite(norm):
        raise RuntimeError('Nonfinite probe gradients; no training was started.')
    probe_optimizer.step()
    peak = torch.cuda.max_memory_reserved() / 2**30
    with torch.no_grad():
        for p, saved in trainable:
            p.copy_(saved.to(device=p.device, dtype=p.dtype))
    model.zero_grad(set_to_none=True)
    del trainable, probe_optimizer, probe_batch, probe_loss, probe_outputs
    gc.collect()
    torch.cuda.empty_cache()
    torch.set_rng_state(rng[0]); torch.cuda.set_rng_state_all(rng[1]); np.random.set_state(rng[2])
    print(f'Full-length backward/optimizer check passed; test update reverted. Peak reserved: {peak:.2f} GiB', flush=True)

    best_path = RUN / 'best_validation.json'
    prior_best = json.loads(best_path.read_text()) if best_path.exists() else None
    if prior_best:
        prior_weights = RUN / 'best_adapter' / 'adapter_model.safetensors'
        if not prior_weights.exists() or sha(prior_weights) != prior_best.get('adapter_sha256'):
            print('Interrupted best-adapter write detected; next validation will rebuild it.', flush=True)
            prior_best = None

    class Persistence(TrainerCallback):
        def __init__(self):
            self.best = prior_best['eval_loss'] if prior_best else float('inf')

        def on_save(self, args, state, control, **kwargs):
            checkpoint = RUN / f'checkpoint-{state.global_step}'
            missing = [n for n in required[:-1] if not (checkpoint / n).is_file()]
            if missing:
                raise RuntimeError(f'Checkpoint incomplete: {missing}')
            files = {n: (checkpoint / n).stat().st_size for n in required[:-1]}
            if any(size == 0 for size in files.values()):
                raise RuntimeError('Checkpoint contains an empty file.')
            write_json(checkpoint / 'checkpoint_verified.json',
                       {'global_step': state.global_step, 'files': files})
            print(f'Drive checkpoint verified: {checkpoint.name}', flush=True)

        def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
            value = float((metrics or {}).get('eval_loss', float('inf')))
            if not math.isfinite(value):
                raise RuntimeError('Validation loss is not finite.')
            if value < self.best:
                best_dir = RUN / 'best_adapter'
                best_dir.mkdir(exist_ok=True)
                model.save_pretrained(best_dir, safe_serialization=True)
                tokenizer.save_pretrained(best_dir)
                write_json(best_path, {'eval_loss': value, 'global_step': state.global_step,
                                      'epoch': state.epoch, 'adapter_sha256': sha(best_dir / 'adapter_model.safetensors')})
                self.best = value
                print(f'Best adapter saved on Drive: validation loss {value:.6f}', flush=True)
            write_json(RUN / 'training_history.json', state.log_history)

    trainer = SummaryTrainer(model=model, args=TrainingArguments(**options), processing_class=tokenizer,
                             data_collator=collate, callbacks=[Persistence()],
                             train_dataset=Dataset.from_list(tokenized['train']),
                             eval_dataset=Dataset.from_list(tokenized['validation']))
    print('STARTING TRAINING. Checkpoints every 20 optimizer steps; full validation each epoch.', flush=True)
    result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    # Periodic checkpoints carry optimizer/RNG state; save final logs and completion metadata.
    trainer.save_state()
    trainer.save_metrics('train', result.metrics)
    write_json(RUN / 'training_history.json', trainer.state.log_history)
    best = json.loads(best_path.read_text())
    if sha(RUN / 'best_adapter' / 'adapter_model.safetensors') != best['adapter_sha256']:
        raise RuntimeError('Best adapter verification failed.')
    write_json(RUN / 'training_complete.json',
               {'global_step': trainer.state.global_step, 'epoch': trainer.state.epoch,
                'best_validation': best, 'metrics': result.metrics,
                'test_used': False, 'next': 'Evaluate generated summaries; validation loss alone does not measure factuality.'})
    print('\nTRAINING COMPLETE', flush=True)
    print('Best adapter:', RUN / 'best_adapter', flush=True)
    print('Best full-validation loss:', best['eval_loss'], flush=True)
    print('Persistent run folder:', RUN, flush=True)


if __name__ == '__main__':
    main()
