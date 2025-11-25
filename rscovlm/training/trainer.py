import torch
from transformers.trainer import Trainer, is_sagemaker_mp_enabled, is_datasets_available, seed_worker
from .monkey_patch.varlen import monkey_patch_flash_attention_to_pass_position_ids


class CustomTrainer(Trainer):

    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        decay_params_names = self.get_decay_parameter_names(opt_model)

        def _group_params(params, lr=None):
            decay = [p for n, p in params.items() if n in decay_params_names and p.requires_grad]
            no_decay = [p for n, p in params.items() if n not in decay_params_names and p.requires_grad]
            kwargs = {"lr": lr} if lr is not None else {}
            return [
                {"params": decay, "weight_decay": self.args.weight_decay, **kwargs},
                {"params": no_decay, "weight_decay": 0.0, **kwargs},
            ]

        vision_params = {}
        merger_params = {}
        other_params = {}

        for name, p in opt_model.named_parameters():
            if "visual" in name and "merger" not in name:
                vision_params[name] = p
            elif "merger" in name:
                merger_params[name] = p
            else:
                other_params[name] = p

        optimizer_grouped_parameters = [
            *(_group_params(vision_params, self.args.vision_lr)),
            *(_group_params(merger_params, self.args.merger_lr)),
            *(_group_params(other_params)),
        ]

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

        assert 'params' not in optimizer_kwargs, "not support yet, please check"
        optimizer_kwargs["params"] = optimizer_grouped_parameters
        self.optimizer_cls_and_kwargs = (optimizer_cls, optimizer_kwargs)

        return super(CustomTrainer, self).create_optimizer()
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        if 'naive_position_ids' in inputs:
            naive_position_ids = inputs.pop('naive_position_ids')
            naive_position_ids = self._prepare_input(naive_position_ids)
            monkey_patch_flash_attention_to_pass_position_ids(naive_position_ids)
            
        return super(CustomTrainer, self).training_step(model, inputs, num_items_in_batch)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available():
            import datasets
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        naive_dataloader = torch.utils.data.DataLoader(train_dataset, **dataloader_params)

        if getattr(self.args, 'do_not_prepare_dataloader_with_accelerator', False):
            return naive_dataloader
        else:
            return self.accelerator.prepare(naive_dataloader)
