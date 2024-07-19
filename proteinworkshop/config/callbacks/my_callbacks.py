from lightning.pytorch.callbacks import Callback
import torch

class GradientCheckCallback(Callback):
    def on_after_backward(self, trainer, pl_module):
        # iterate over all the model parameters
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient found in {name}")
                # print(f'Gradient norm for {name}: {grad_norm}')
            else:
                print(f"No gradient for {name}")
                
# class ActivationNanCheckCallback(pl.Callback):
#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         for name, module in pl_module.named_modules():
#             module.register_forward_hook(self.check_activations)

#     def check_activations(self, module, input, output):
#         if torch.isnan(output).any():
#             print(f"NaN detected in the output of {module}")