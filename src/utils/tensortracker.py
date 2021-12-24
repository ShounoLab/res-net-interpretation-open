from collections import OrderedDict


class TensorTracker(object):
    """ """

    def __init__(
        self,
        model,
        candidate_layers=None,
        device="cpu",
        ignore_forward=False,
        ignore_backward=False,
        detach_tensors=True,
    ):
        """
        To track forward and backward tensors of pytorch model.
        If any candidates are not specified, the hook is registered to all the layers.
        Eg.
            tracker = TensorTracker(model)
            output = model(input)
            feature_map = tracker.find_fmap("layer")
            gradient_of_feature_map = tracker.find_grad("layer")
        """
        super(TensorTracker, self).__init__()
        self.model = model
        self.handlers = []
        self.fmap_pool = OrderedDict()
        self.grad_pool = OrderedDict()
        self.candidate_layers = candidate_layers  # list
        self._device = device
        self.detach_tensors = detach_tensors

        def forward_hook(key):
            def forward_hook_(module, input, output):
                # Save featuremaps
                if isinstance(output, dict):
                    ls = input[1]
                    if self.detach_tensors:
                        self.fmap_pool[key] = output[ls[0]].detach().to(self._device)
                    else:
                        self.fmap_pool[key] = output[ls[0]]
                else:
                    if self.detach_tensors:
                        self.fmap_pool[key] = output.detach().to(self._device)
                    else:
                        self.fmap_pool[key] = output

            return forward_hook_

        def backward_hook(key):
            def backward_hook_(module, grad_in, grad_out):
                # Save the gradients correspond to the featuremaps
                if self.detach_tensors:
                    self.grad_pool[key] = grad_out[0].detach().to(self._device)
                else:
                    self.grad_pool[key] = grad_out[0]

            return backward_hook_

        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                if not ignore_forward:
                    self.handlers.append(
                        module.register_forward_hook(forward_hook(name))
                    )
                if not ignore_backward:
                    self.handlers.append(
                        module.register_backward_hook(backward_hook(name))
                    )

    def __del__(self):
        self.remove()

    def remove(self):
        keys = list(self.fmap_pool.keys())
        for key in keys:
            del self.fmap_pool[key]
        keys = list(self.grad_pool.keys())
        for key in keys:
            del self.grad_pool[key]
        self.remove_hook()

    def find_fmap(self, target_layer):
        return self._find(self.fmap_pool, target_layer)

    def find_grad(self, target_layer):
        return self._find(self.grad_pool, target_layer)

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class TensorTrackerInputGrad(TensorTracker):
    def __init__(
        self,
        model,
        candidate_layers=None,
        device="cpu",
        ignore_BN=False,
        detach_tensors=True,
    ):
        if ignore_BN:
            ignore_forward = True
            ignore_backward = True
        else:
            ignore_forward = False
            ignore_backward = False

        super().__init__(
            model,
            candidate_layers,
            device,
            ignore_forward=ignore_forward,
            ignore_backward=ignore_backward,
            detach_tensors=detach_tensors,
        )

        self.grad_pool_in = OrderedDict()

        def backward_hook(key):
            def backward_hook_(module, grad_in, grad_out):
                # Save the gradients correspond to the featuremaps
                if grad_in is not None and grad_in[0] is not None:
                    self.grad_pool_in[key] = grad_in[0].detach().to(self._device)

            return backward_hook_

        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_backward_hook(backward_hook(name)))
