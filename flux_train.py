# training with captions

# Swap blocks between CPU and GPU:
# This implementation is inspired by and based on the work of 2kpr.
# Many thanks to 2kpr for the original concept and implementation of memory-efficient offloading.
# The original idea has been adapted and extended to fit the current project's needs.

# Key features:
# - CPU offloading during forward and backward passes
# - Use of fused optimizer and grad_hook for efficient gradient processing
# - Per-block fused optimizer instances

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import math
import os
from multiprocessing import Value
import time
from typing import List, Optional, Tuple, Union
import toml

from tqdm import tqdm

import torch
import torch.nn as nn
from library import utils
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from library import deepspeed_utils, flux_train_utils, flux_utils, strategy_base, strategy_flux
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler

import library.train_util as train_util

from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

import library.config_util as config_util

# import library.sdxl_train_util as sdxl_train_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
from library.custom_train_functions import apply_masked_loss, add_custom_train_arguments


def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    # sdxl_train_util.verify_sdxl_training_args(args)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    # temporary: backward compatibility for deprecated options. remove in the future
    if not args.skip_cache_check:
        args.skip_cache_check = args.skip_latents_validity_check

    # assert (
    #     not args.weighted_captions
    # ), "weighted_captions is not supported currently / weighted_captionsは現在サポートされていません"
    if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
        logger.warning(
            "cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled / cache_text_encoder_outputs_to_diskが有効になっているため、cache_text_encoder_outputsも有効になります"
        )
        args.cache_text_encoder_outputs = True

    if args.cpu_offload_checkpointing and not args.gradient_checkpointing:
        logger.warning(
            "cpu_offload_checkpointing is enabled, so gradient_checkpointing is also enabled / cpu_offload_checkpointingが有効になっているため、gradient_checkpointingも有効になります"
        )
        args.gradient_checkpointing = True

    assert (
        args.blocks_to_swap is None or args.blocks_to_swap == 0
    ) or not args.cpu_offload_checkpointing, (
        "blocks_to_swap is not supported with cpu_offload_checkpointing / blocks_to_swapはcpu_offload_checkpointingと併用できません"
    )

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)  # 乱数系列を初期化する

    # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
    if args.cache_latents:
        latents_caching_strategy = strategy_flux.FluxLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # データセットを準備する
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args)
        val_dataset_group = None

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(16)  # TODO これでいいか確認

    model_type_str, _, _, _ = flux_utils.analyze_checkpoint_state(args.pretrained_model_name_or_path) # num_double, num_single, metadata ignored here for now
    if args.debug_dataset:
        if args.cache_text_encoder_outputs:
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(
                strategy_flux.FluxTextEncoderOutputsCachingStrategy(
                    args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, False
                )
            )
        if args.t5xxl_max_token_length is not None:
            t5xxl_max_token_length = args.t5xxl_max_token_length
        elif model_type_str == flux_utils.MODEL_TYPE_FLUX_SCHNELL or model_type_str == flux_utils.MODEL_TYPE_CHROMA: # Chroma uses Schnell's AE and likely similar token limits
            t5xxl_max_token_length = 256
        else: # Default for DEV or UNKNOWN (safer default)
            t5xxl_max_token_length = 512
        strategy_base.TokenizeStrategy.set_strategy(strategy_flux.FluxTokenizeStrategy(t5xxl_max_token_length))

        train_dataset_group.set_current_strategies()
        train_util.debug_dataset(train_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        logger.error(
            "No data found. Please verify the metadata file and train_data_dir option. / 画像がありません。メタデータおよびtrain_data_dirオプションを確認してください。"
        )
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    if args.cache_text_encoder_outputs:
        assert (
            train_dataset_group.is_text_encoder_output_cacheable()
        ), "when caching text encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / text encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

    # acceleratorを準備する
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    # モデルを読み込む

    # load VAE for caching latents
    ae = None
    if cache_latents:
        ae = flux_utils.load_ae(args.ae, weight_dtype, "cpu", args.disable_mmap_load_safetensors)
        ae.to(accelerator.device, dtype=weight_dtype)
        ae.requires_grad_(False)
        ae.eval()

        train_dataset_group.new_cache_latents(ae, accelerator)

        ae.to("cpu")  # if no sampling, vae can be deleted
        clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    # prepare tokenize strategy
    if args.t5xxl_max_token_length is not None:
        t5xxl_max_token_length = args.t5xxl_max_token_length
    elif model_type_str == flux_utils.MODEL_TYPE_FLUX_SCHNELL or model_type_str == flux_utils.MODEL_TYPE_CHROMA:
        t5xxl_max_token_length = 256
    else: # Default for DEV or UNKNOWN
        t5xxl_max_token_length = 512

    flux_tokenize_strategy = strategy_flux.FluxTokenizeStrategy(t5xxl_max_token_length)
    strategy_base.TokenizeStrategy.set_strategy(flux_tokenize_strategy)

    # load clip_l, t5xxl for caching text encoder outputs
    clip_l = flux_utils.load_clip_l(args.clip_l, weight_dtype, "cpu", args.disable_mmap_load_safetensors)
    t5xxl = flux_utils.load_t5xxl(args.t5xxl, weight_dtype, "cpu", args.disable_mmap_load_safetensors)
    clip_l.eval()
    t5xxl.eval()
    clip_l.requires_grad_(False)
    t5xxl.requires_grad_(False)

    text_encoding_strategy = strategy_flux.FluxTextEncodingStrategy(args.apply_t5_attn_mask)
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    # cache text encoder outputs
    sample_prompts_te_outputs = None
    if args.cache_text_encoder_outputs:
        # Text Encodes are eval and no grad here
        clip_l.to(accelerator.device)
        t5xxl.to(accelerator.device)

        text_encoder_caching_strategy = strategy_flux.FluxTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, False, False, args.apply_t5_attn_mask
        )
        strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_caching_strategy)

        with accelerator.autocast():
            train_dataset_group.new_cache_text_encoder_outputs([clip_l, t5xxl], accelerator)

        # cache sample prompt's embeddings to free text encoder's memory
        if args.sample_prompts is not None:
            logger.info(f"cache Text Encoder outputs for sample prompt: {args.sample_prompts}")

            text_encoding_strategy: strategy_flux.FluxTextEncodingStrategy = strategy_base.TextEncodingStrategy.get_strategy()

            prompts = train_util.load_prompts(args.sample_prompts)
            sample_prompts_te_outputs = {}  # key: prompt, value: text encoder outputs
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                        if p not in sample_prompts_te_outputs:
                            logger.info(f"cache Text Encoder outputs for prompt: {p}")
                            tokens_and_masks = flux_tokenize_strategy.tokenize(p)
                            sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                flux_tokenize_strategy, [clip_l, t5xxl], tokens_and_masks, args.apply_t5_attn_mask
                            )

        accelerator.wait_for_everyone()

        # now we can delete Text Encoders to free memory
        clip_l = None
        t5xxl = None
        clean_memory_on_device(accelerator.device)

    # load FLUX
    # model_type_str is already determined above from analyze_checkpoint_state
    _, flux = flux_utils.load_flow_model( # First return value (model_type_str from load_flow_model) is ignored as we have it
        args.pretrained_model_name_or_path, weight_dtype, "cpu", args.disable_mmap_load_safetensors
    )

    if args.gradient_checkpointing:
        flux.enable_gradient_checkpointing(cpu_offload=args.cpu_offload_checkpointing)

    flux.requires_grad_(True)

    # block swap

    # backward compatibility
    if args.blocks_to_swap is None:
        blocks_to_swap = args.double_blocks_to_swap or 0
        if args.single_blocks_to_swap is not None:
            blocks_to_swap += args.single_blocks_to_swap // 2
        if blocks_to_swap > 0:
            logger.warning(
                "double_blocks_to_swap and single_blocks_to_swap are deprecated. Use blocks_to_swap instead."
                " / double_blocks_to_swapとsingle_blocks_to_swapは非推奨です。blocks_to_swapを使ってください。"
            )
            logger.info(
                f"double_blocks_to_swap={args.double_blocks_to_swap} and single_blocks_to_swap={args.single_blocks_to_swap} are converted to blocks_to_swap={blocks_to_swap}."
            )
            args.blocks_to_swap = blocks_to_swap
        del blocks_to_swap

    is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
    if is_swapping_blocks:
        # Swap blocks between CPU and GPU to reduce memory usage, in forward and backward passes.
        # This idea is based on 2kpr's great work. Thank you!
        logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
        flux.enable_block_swap(args.blocks_to_swap, accelerator.device)

    if not cache_latents:
        # load VAE here if not cached
        ae = flux_utils.load_ae(args.ae, weight_dtype, "cpu", args.disable_mmap_load_safetensors) # Added disable_mmap
        ae.requires_grad_(False)
        ae.eval()
        ae.to(accelerator.device, dtype=weight_dtype)

    training_models = []
    params_to_optimize = []
    training_models.append(flux)
    name_and_params = list(flux.named_parameters())
    # single param group for now
    params_to_optimize.append({"params": [p for _, p in name_and_params], "lr": args.learning_rate})
    param_names = [[n for n, _ in name_and_params]]

    # calculate number of trainable parameters
    n_params = 0
    for group in params_to_optimize:
        for p in group["params"]:
            n_params += p.numel()

    accelerator.print(f"number of trainable parameters: {n_params}")

    # 学習に必要なクラスを準備する
    accelerator.print("prepare optimizer, data loader etc.")

    if args.blockwise_fused_optimizers:
        # fused backward pass: https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html
        # Instead of creating an optimizer for all parameters as in the tutorial, we create an optimizer for each block of parameters.
        # This balances memory usage and management complexity.

        # split params into groups. currently different learning rates are not supported
        grouped_params = []
        param_group = {}
        for group in params_to_optimize:
            named_parameters = list(flux.named_parameters())
            assert len(named_parameters) == len(group["params"]), "number of parameters does not match"
            for p, np_tuple in zip(group["params"], named_parameters): # Renamed np to np_tuple
                # determine target layer and block index for each parameter
                block_type = "other"  # double, single or other
                if np_tuple[0].startswith("double_blocks"):
                    block_index = int(np_tuple[0].split(".")[1])
                    block_type = "double"
                elif np_tuple[0].startswith("single_blocks"):
                    block_index = int(np_tuple[0].split(".")[1])
                    block_type = "single"
                else:
                    block_index = -1

                param_group_key = (block_type, block_index)
                if param_group_key not in param_group:
                    param_group[param_group_key] = []
                param_group[param_group_key].append(p)

        block_types_and_indices = []
        for param_group_key, param_group_val in param_group.items(): # Renamed param_group to param_group_val to avoid conflict
            block_types_and_indices.append(param_group_key)
            grouped_params.append({"params": param_group_val, "lr": args.learning_rate})

            num_params = 0
            for p_item in param_group_val: # Renamed p to p_item
                num_params += p_item.numel()
            accelerator.print(f"block {param_group_key}: {num_params} parameters")

        # prepare optimizers for each group
        optimizers = []
        for group in grouped_params:
            _, _, optimizer_instance = train_util.get_optimizer(args, trainable_params=[group]) # Renamed optimizer to optimizer_instance
            optimizers.append(optimizer_instance)
        optimizer = optimizers[0]  # avoid error in the following code

        logger.info(f"using {len(optimizers)} optimizers for blockwise fused optimizers")

        if train_util.is_schedulefree_optimizer(optimizers[0], args):
            raise ValueError("Schedule-free optimizer is not supported with blockwise fused optimizers")
        optimizer_train_fn = lambda: None  # dummy function
        optimizer_eval_fn = lambda: None  # dummy function
    else:
        _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)
        optimizer_train_fn, optimizer_eval_fn = train_util.get_optimizer_train_eval_fn(optimizer, args)

    # prepare dataloader
    # strategies are set here because they cannot be referenced in another process. Copy them with the dataset
    # some strategies can be None
    train_dataset_group.set_current_strategies()

    # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # 学習ステップ数を計算する
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(
            f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
        )

    # データセット側にも学習ステップを送信
    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # lr schedulerを用意する
    if args.blockwise_fused_optimizers:
        # prepare lr schedulers for each optimizer
        lr_schedulers = [train_util.get_scheduler_fix(args, opt, accelerator.num_processes) for opt in optimizers] # Changed optimizer to opt
        lr_scheduler = lr_schedulers[0]  # avoid error in the following code
    else:
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        flux.to(weight_dtype)
        if clip_l is not None:
            clip_l.to(weight_dtype)
            t5xxl.to(weight_dtype)  # TODO check works with fp16 or not
    elif args.full_bf16:
        assert (
            args.mixed_precision == "bf16"
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        accelerator.print("enable full bf16 training.")
        flux.to(weight_dtype)
        if clip_l is not None:
            clip_l.to(weight_dtype)
            t5xxl.to(weight_dtype)

    # if we don't cache text encoder outputs, move them to device
    if not args.cache_text_encoder_outputs:
        clip_l.to(accelerator.device)
        t5xxl.to(accelerator.device)

    clean_memory_on_device(accelerator.device)

    if args.deepspeed:
        ds_model = deepspeed_utils.prepare_deepspeed_model(args, mmdit=flux)
        # most of ZeRO stage uses optimizer partitioning, so we have to prepare optimizer and ds_model at the same time. # pull/1139#issuecomment-1986790007
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ds_model, optimizer, train_dataloader, lr_scheduler
        )
        training_models = [ds_model]

    else:
        # accelerator does some magic
        # if we doesn't swap blocks, we can move the model to device
        flux_prepared = accelerator.prepare(flux, device_placement=[not is_swapping_blocks]) # Renamed flux to flux_prepared
        if is_swapping_blocks:
            accelerator.unwrap_model(flux_prepared).move_to_device_except_swap_blocks(accelerator.device)  # reduce peak memory usage
        
        # After prepare, flux might be a DDP model or similar. We need to use the prepared model.
        flux = flux_prepared # Update flux to the prepared version

        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
        training_models = [flux] # Update training_models to use the prepared flux


    # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
    if args.full_fp16:
        # During deepseed training, accelerate not handles fp16/bf16|mixed precision directly via scaler. Let deepspeed engine do.
        # -> But we think it's ok to patch accelerator even if deepspeed is enabled.
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # resumeする
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    if args.fused_backward_pass:
        # use fused optimizer for backward pass: other optimizers will be supported in the future
        import library.adafactor_fused

        library.adafactor_fused.patch_adafactor_fused(optimizer)

        for param_group, param_name_group in zip(optimizer.param_groups, param_names):
            for parameter, param_name in zip(param_group["params"], param_name_group):
                if parameter.requires_grad:

                    def create_grad_hook(p_name, p_group):
                        def grad_hook(tensor: torch.Tensor):
                            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(tensor, args.max_grad_norm)
                            optimizer.step_param(tensor, p_group)
                            tensor.grad = None

                        return grad_hook

                    parameter.register_post_accumulate_grad_hook(create_grad_hook(param_name, param_group))

    elif args.blockwise_fused_optimizers:
        # prepare for additional optimizers and lr schedulers
        for i in range(1, len(optimizers)):
            optimizers[i] = accelerator.prepare(optimizers[i])
            lr_schedulers[i] = accelerator.prepare(lr_schedulers[i])

        # counters are used to determine when to step the optimizer
        global optimizer_hooked_count
        global num_parameters_per_group
        global parameter_optimizer_map

        optimizer_hooked_count = {}
        num_parameters_per_group = [0] * len(optimizers)
        parameter_optimizer_map = {}

        for opt_idx, opt_instance in enumerate(optimizers): # Changed opt to opt_instance
            for param_group in opt_instance.param_groups:
                for parameter in param_group["params"]:
                    if parameter.requires_grad:

                        def grad_hook(parameter_tensor: torch.Tensor): # Renamed parameter to parameter_tensor
                            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(parameter_tensor, args.max_grad_norm)

                            i = parameter_optimizer_map[parameter_tensor]
                            optimizer_hooked_count[i] += 1
                            if optimizer_hooked_count[i] == num_parameters_per_group[i]:
                                optimizers[i].step()
                                optimizers[i].zero_grad(set_to_none=True)

                        parameter.register_post_accumulate_grad_hook(grad_hook)
                        parameter_optimizer_map[parameter] = opt_idx
                        num_parameters_per_group[opt_idx] += 1

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    # 学習する
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num examples / サンプル数: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(
        f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
    )
    # accelerator.print(
    #     f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}"
    # )
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "finetuning" if args.log_tracker_name is None else args.log_tracker_name,
            config=train_util.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

    if is_swapping_blocks:
        accelerator.unwrap_model(flux).prepare_block_swap_before_forward()

    # For --sample_at_first
    optimizer_eval_fn()
    flux_train_utils.sample_images(accelerator, args, 0, global_step, flux, ae, [clip_l, t5xxl], sample_prompts_te_outputs)
    optimizer_train_fn()
    if len(accelerator.trackers) > 0:
        # log empty object to commit the sample images to wandb
        accelerator.log({}, step=0)

    loss_recorder = train_util.LossRecorder()
    epoch = 0  # avoid error when max_train_steps is 0
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step

            if args.blockwise_fused_optimizers:
                optimizer_hooked_count = {i: 0 for i in range(len(optimizers))}  # reset counter for each step

            with accelerator.accumulate(*training_models):
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
                else:
                    with torch.no_grad():
                        # encode images to latents. images are [-1, 1]
                        latents = ae.encode(batch["images"].to(ae.dtype)).to(accelerator.device, dtype=weight_dtype)

                    # NaNが含まれていれば警告を表示し0に置き換える
                    if torch.any(torch.isnan(latents)):
                        accelerator.print("NaN found in latents, replacing with zeros")
                        latents = torch.nan_to_num(latents, 0, out=latents)

                text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
                if text_encoder_outputs_list is not None:
                    text_encoder_conds = text_encoder_outputs_list
                else:
                    # not cached or training, so get from text encoders
                    # tokens_and_masks = batch["input_ids_list"] # This was the old way, now flux_tokenize_strategy.tokenize returns dict
                    with torch.no_grad():
                        # Re-tokenize if not cached, as batch["input_ids_list"] might not be what FluxTextEncodingStrategy expects
                        # Or ensure collator provides tokens_and_masks in the expected dict format
                        # For now, assuming batch["input_ids_list"] is compatible or collator handles it.
                        # If FluxTextEncodingStrategy.encode_tokens expects a dict like {'clip_l': ..., 't5xxl': ...}
                        # then the collator or this part needs to ensure that.
                        # Current flux_tokenize_strategy.tokenize returns such a dict.
                        # If batch["input_ids_list"] is a list of tensors, it needs adaptation.
                        # Assuming batch["input_ids_list"] is now the dict from tokenize strategy
                        tokens_and_masks_dict = batch["input_ids_list"] # This should be the dict
                        
                        text_encoder_conds = text_encoding_strategy.encode_tokens(
                            flux_tokenize_strategy, [clip_l, t5xxl], tokens_and_masks_dict, args.apply_t5_attn_mask
                        )
                        if args.full_fp16: # or full_bf16
                            text_encoder_conds = [c.to(weight_dtype) if c is not None else None for c in text_encoder_conds]


                # TODO support some features for noise implemented in get_noise_noisy_latents_and_timesteps

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # get noisy model input and timesteps
                noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
                    args, noise_scheduler_copy, latents, noise, accelerator.device, weight_dtype
                )

                # pack latents and get img_ids
                packed_noisy_model_input = flux_utils.pack_latents(noisy_model_input)  # b, c, h*2, w*2 -> b, h*w, c*4
                packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2
                img_ids = flux_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=accelerator.device)

                # get guidance: ensure args.guidance_scale is float
                guidance_vec = torch.full((bsz,), float(args.guidance_scale), device=accelerator.device)

                # call model
                l_pooled, t5_out, txt_ids, t5_attn_mask_from_conds = text_encoder_conds # Renamed t5_attn_mask to avoid conflict
                
                # Determine the t5_attn_mask to use for the model call
                # If args.apply_t5_attn_mask is True, we use the one from text_encoder_conds.
                # Otherwise, it should be None.
                actual_t5_attn_mask_for_model = t5_attn_mask_from_conds if args.apply_t5_attn_mask else None


                with accelerator.autocast():
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                    model_pred = flux(
                        img=packed_noisy_model_input,
                        img_ids=img_ids,
                        txt=t5_out,
                        txt_ids=txt_ids,
                        y=l_pooled,
                        timesteps=timesteps / 1000, # timesteps are 0-1000 from scheduler, model expects 0-1 range
                        guidance=guidance_vec,
                        txt_attention_mask=actual_t5_attn_mask_for_model,
                    )

                # unpack latents
                model_pred = flux_utils.unpack_latents(model_pred, packed_latent_height, packed_latent_width)

                # apply model prediction type
                model_pred, weighting = flux_train_utils.apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas)

                # flow matching loss: this is different from SD3
                target = noise - latents

                # calculate loss
                huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
                loss = train_util.conditional_loss(model_pred.float(), target.float(), args.loss_type, "none", huber_c)
                if weighting is not None:
                    loss = loss * weighting
                if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                    loss = apply_masked_loss(loss, batch)
                loss = loss.mean([1, 2, 3])

                loss_weights = batch["loss_weights"]  # 各sampleごとのweight
                loss = loss * loss_weights
                loss = loss.mean()

                # backward
                accelerator.backward(loss)

                if not (args.fused_backward_pass or args.blockwise_fused_optimizers):
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = []
                        for m_model in training_models: # Renamed m to m_model
                            params_to_clip.extend(m_model.parameters())
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    # optimizer.step() and optimizer.zero_grad() are called in the optimizer hook
                    lr_scheduler.step()
                    if args.blockwise_fused_optimizers:
                        for i in range(1, len(optimizers)):
                            lr_schedulers[i].step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                optimizer_eval_fn()
                flux_train_utils.sample_images(
                    accelerator, args, None, global_step, flux, ae, [clip_l, t5xxl], sample_prompts_te_outputs
                )

                # 指定ステップごとにモデルを保存
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        flux_train_utils.save_flux_model_on_epoch_end_or_stepwise(
                            args,
                            False,
                            accelerator,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            accelerator.unwrap_model(flux),
                        )
                optimizer_train_fn()

            current_loss = loss.detach().item()  # 平均なのでbatch sizeは関係ないはず
            if len(accelerator.trackers) > 0:
                logs = {"loss": current_loss}
                train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=True)

                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if len(accelerator.trackers) > 0:
            logs = {"loss/epoch": loss_recorder.moving_average}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        optimizer_eval_fn()
        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                flux_train_utils.save_flux_model_on_epoch_end_or_stepwise(
                    args,
                    True,
                    accelerator,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    accelerator.unwrap_model(flux),
                )

        flux_train_utils.sample_images(
            accelerator, args, epoch + 1, global_step, flux, ae, [clip_l, t5xxl], sample_prompts_te_outputs
        )
        optimizer_train_fn()

    is_main_process = accelerator.is_main_process
    # if is_main_process:
    flux = accelerator.unwrap_model(flux)

    accelerator.end_training()
    optimizer_eval_fn()

    if args.save_state or args.save_state_on_train_end:
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator  # この後メモリを使うのでこれは消す

    if is_main_process:
        flux_train_utils.save_flux_model_on_train_end(args, save_dtype, epoch, global_step, flux)
        logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)  # TODO split this
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_custom_train_arguments(parser)  # TODO remove this from here
    train_util.add_dit_training_arguments(parser)
    flux_train_utils.add_flux_train_arguments(parser)

    parser.add_argument(
        "--mem_eff_save",
        action="store_true",
        help="[EXPERIMENTAL] use memory efficient custom model saving method / メモリ効率の良い独自のモデル保存方法を使う",
    )

    parser.add_argument(
        "--fused_optimizer_groups",
        type=int,
        default=None,
        help="**this option is not working** will be removed in the future / このオプションは動作しません。将来削除されます",
    )
    parser.add_argument(
        "--blockwise_fused_optimizers",
        action="store_true",
        help="enable blockwise optimizers for fused backward pass and optimizer step / fused backward passとoptimizer step のためブロック単位のoptimizerを有効にする",
    )
    parser.add_argument(
        "--skip_latents_validity_check",
        action="store_true",
        help="[Deprecated] use 'skip_cache_check' instead / 代わりに 'skip_cache_check' を使用してください",
    )
    parser.add_argument(
        "--double_blocks_to_swap",
        type=int,
        default=None,
        help="[Deprecated] use 'blocks_to_swap' instead / 代わりに 'blocks_to_swap' を使用してください",
    )
    parser.add_argument(
        "--single_blocks_to_swap",
        type=int,
        default=None,
        help="[Deprecated] use 'blocks_to_swap' instead / 代わりに 'blocks_to_swap' を使用してください",
    )
    parser.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="[EXPERIMENTAL] enable offloading of tensors to CPU during checkpointing / チェックポイント時にテンソルをCPUにオフロードする",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    train(args)