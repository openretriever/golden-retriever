from typing import Optional


class ModelActorBase(object):
    """Base class for model actors using Ray.

    This will be our default way of calling models within Ray.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        model: Optional[object] = None,
        model_name: Optional[str] = None,
    ):
        self.use_gpu = use_gpu
        self.model = model
        self.model_name = model_name

    def predict(self, *x, **kwargs):
        raise NotImplementedError


class BaseModelServer(object):
    """Base class for model servers using Ray Serve.

    We can serve models using Ray Serve when we need to handle heavier loads.
    """

    def __init__(self, use_gpu: bool = False, model: Optional[object] = None):
        self.use_gpu = use_gpu

    async def __call__(self, *x, **kwargs):
        raise NotImplementedError


class LangDetectBase(ModelActorBase):
    """Base class for language detection models.

    This class will be used for models that detect the language of text.
    """

    def __init__(self, use_gpu: bool = False, model: Optional[object] = None):
        super().__init__(use_gpu=use_gpu, model=model)

    def predict(self, *x, **kwargs):
        raise NotImplementedError


class LangSegBase(ModelActorBase):
    """Base class for language segmentation models.

    This class will be used for models that segment text into sentences.
    """

    def __init__(self, use_gpu: bool = False, model: Optional[object] = None):
        super().__init__(use_gpu=use_gpu, model=model)

    def predict(self, *x, **kwargs):
        raise NotImplementedError


class VisLangBase(ModelActorBase):
    """Base class for vision-language models.

    This class will be used for models that process both images and text.
    """

    def __init__(self, use_gpu: bool = False, model: Optional[object] = None):
        super().__init__(use_gpu=use_gpu, model=model)

    def predict(self, images, prompts, **kwargs):
        raise NotImplementedError
