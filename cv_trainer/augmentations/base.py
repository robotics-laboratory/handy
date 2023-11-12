from typing import List, Callable, Dict, Any

class AugmentationBase:
    def __call__(self, **data) -> Dict[str, Any]:
        raise NotImplementedError()